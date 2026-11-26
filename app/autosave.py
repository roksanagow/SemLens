"""The open work kept on disk as it changes, and offered back in a later session.

Analyse a word: each browser session writes into a folder of its own under
``STORAGE/autosave``. The large parts are written once, when they are made:

- ``usages-….json.gz``: the loaded usages;
- ``vectors-….npy``: their embeddings, in 16-bit;
- ``texts-….json.gz``: the source texts;
- ``definitions-….npy``: the definitions' embeddings.

The annotations, keyed by usage, have a file of their own,
``annotations-….jsonl``: written whole once, then each change appended as a
line (a usage and its senses, or none when cleared). It is written whole again
only when it has grown to twice what it holds. An annotation click therefore
writes a line, however many usages are annotated.

``state.json`` holds the rest of what the user changes (senses, definitions,
settings) and names the files it belongs with. It is written atomically after
each change, and files it no longer names are then removed, so an interrupted
write never leaves a mismatched set.

Only the two most recently changed folders are kept. Nothing is restored
automatically: a later session offers to continue, and continuing only reads
these files. No model runs and nothing is embedded again.

Before embeddings or texts are written, free disk space is checked against the
reserve SemLens always leaves free (:data:`semlens.resources.DISK_RESERVE`).
When space is short they are skipped and the user is told once. The
annotations, keyed by usage, are still kept.

Discover keeps each discovery in its own folder and saves its seeds, marks
and word decisions as they change, so continuing there reopens the most
recently changed discovery.
"""
from __future__ import annotations
import gzip
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from semlens.data_loading import TargetWordData, UsageInstance
from semlens.embeddings import embedded_usages_from_precomputed
from semlens.resources import DISK_RESERVE, format_bytes
from app.paths import STORAGE
from app.annotation_store import get_annotations, get_options, get_sense_classes

AUTOSAVE_ROOT = STORAGE/'autosave'
STATE = 'state.json'
KEEP = 2                      # this session's folder and the one before it
STORED_PRECISION = 'float16'


# --- Writing ---------------------------------------------------------------------------------------------

def _usage_ids(data):
    """Each usage's ID, computed once per loaded data set (each one is a hash)."""
    cached = st.session_state.get('_autosave_ids')
    if cached is None or cached[0] is not data:
        cached = (data, [u.usage_id for u in data.usages])
        st.session_state['_autosave_ids'] = cached
    return cached[1]


def _signature(value, *extra):
    """Identifies an object in this session: it is replaced, never changed in place, when it changes."""
    return None if value is None else [id(value), *extra]


def _signatures(data):
    state = st.session_state
    embedded, texts, definitions = state.get('embedded_usages'), state.get('source_texts') or None, state.get('definition_embeddings')
    return {'usages': _signature(data, len(data), data.word),
            'vectors': _signature(embedded, *(tuple(embedded.embeddings.shape) if embedded is not None else ())),
            'texts': _signature(texts, len(texts or ()), sum(len(t) for t in (texts or {}).values())),
            'definitions': _signature(definitions, *(tuple(definitions.shape) if definitions is not None else ()))}


def _atomic(path, write):
    fd, temp = tempfile.mkstemp(prefix='.saving-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            write(f)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _write_json_gz(path, value):
    text = json.dumps(value, ensure_ascii=False).encode()
    _atomic(path, lambda f: f.write(gzip.compress(text, compresslevel=5)))


def _read_json_gz(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def _room_for(folder, size):
    """Whether writing ``size`` bytes leaves the disk reserve free."""
    return shutil.disk_usage(folder).free-size >= DISK_RESERVE


def _skipped_once(kind, size):
    """Tell the user, once per session, that a large part was not kept."""
    told = st.session_state.setdefault('_autosave_told', set())
    if kind in told:
        return
    told.add(kind)
    what = {'vectors': 'The embeddings', 'texts': 'The source texts'}[kind]
    st.toast(f'{what} ({format_bytes(size)}) were not kept for next time: the disk would have less than '
             f'{format_bytes(DISK_RESERVE)} free. Annotations, senses and definitions are still kept as you work.', icon=':material/warning:')


def _slot():
    """This session's folder, made (and older folders removed) when it first has something to keep."""
    path = st.session_state.get('_autosave_slot')
    if path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = AUTOSAVE_ROOT/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    path.mkdir(parents=True)
    st.session_state['_autosave_slot'] = str(path)
    _prune(path)
    return path


def _changed(folder):
    try:
        return (folder/STATE).stat().st_mtime
    except OSError:
        return None


def _prune(current):
    """Keep this session's folder and the most recently changed other one."""
    others = []
    for folder in AUTOSAVE_ROOT.iterdir():
        if folder == current or not folder.is_dir() or folder.name.startswith('.'):
            continue
        changed = _changed(folder)
        if changed is None:
            # Unfinished: removed once it is clearly not another session starting up.
            if time.time()-folder.stat().st_mtime > 3600:
                shutil.rmtree(folder, ignore_errors=True)
            continue
        others.append((changed, folder))
    for _, folder in sorted(others, reverse=True)[KEEP-1:]:
        shutil.rmtree(folder, ignore_errors=True)


def _state(data, files, skipped):
    from app.ui_data import SAVED_SETTINGS
    state = st.session_state
    ids = _usage_ids(data)
    embedded = state.get('embedded_usages')
    annotated = sum(1 for i in get_annotations() if 0 <= i < len(ids))
    senses = get_sense_classes()
    return {
        'version': 1, 'files': files, 'skipped': sorted(skipped),
        'summary': {'word': data.word, 'usages': len(data), 'corpora': len(data.corpus_labels),
                    'model': embedded.metadata.get('model_id', '') if embedded is not None else None,
                    'definitions': len(state.get('definitions_formatted') or []),
                    'annotated': annotated, 'senses': len(senses)},
        'embedding_metadata': dict(embedded.metadata) if embedded is not None else None,
        'embeddings_source': state.get('embeddings_source', 'model'),
        'sense_classes': list(senses), 'annotation_options': dict(get_options()),
        'definitions_raw': list(state.get('definitions_raw') or []),
        'definitions_formatted': list(state.get('definitions_formatted') or []),
        'settings': {k: state[k] for k in SAVED_SETTINGS if k in state},
        'source': {'lines': state.get('source_lines', False), 'split': state.get('source_split', 'auto'),
                   'kind': state.get('_source_kind'), 'cache': state.get('saved_cache_path')},
        'analysis_name': state.get('_analysis_save_name'), 'analysis_path': state.get('_active_analysis_path'),
        'workspace': state.get('workspace'), 'data_source': state.get('_data_source'),
    }


def _line(usage_id, label):
    return json.dumps({'id': usage_id, 'label': label}, ensure_ascii=False) + '\n'


def _write_annotations(folder, data, written):
    """The annotations' file: written whole when there is none for these usages, else each change appended as a line."""
    ids = _usage_ids(data)
    keyed = {ids[i]: label for i, label in get_annotations().items() if 0 <= i < len(ids)}
    record = written.get('annotations')                      # (usages' signature, file name, lines)
    kept = st.session_state.get('_autosave_annotations')     # what the file holds, by usage
    signature = written.get('usages', (None,))[0]
    if (record is None or record[0] != signature or kept is None or not (folder/record[1]).is_file()
            or record[2] > 2*len(keyed) + 1000):
        name = f'annotations-{uuid.uuid4().hex[:12]}.jsonl'
        text = ''.join(_line(k, v) for k, v in keyed.items())
        _atomic(folder/name, lambda f: f.write(text.encode()))
        written['annotations'] = (signature, name, len(keyed))
        st.session_state['_autosave_annotations'] = {k: dict(v) for k, v in keyed.items()}
        return name
    changes = [(k, v) for k, v in keyed.items() if kept.get(k) != v]
    changes += [(k, None) for k in kept.keys() - keyed.keys()]
    if changes:
        with open(folder/record[1], 'a', encoding='utf-8') as f:
            f.write(''.join(_line(k, v) for k, v in changes))
        for k, v in changes:
            if v is None:
                kept.pop(k, None)
            else:
                kept[k] = dict(v)
        written['annotations'] = (signature, record[1], record[2] + len(changes))
    return record[1]


def _read_annotations(path):
    """The annotations in a file, later lines replacing earlier ones; an unfinished last line is skipped."""
    found = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get('label'):
            found[record['id']] = record['label']
        else:
            found.pop(record.get('id'), None)
    return found


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (set, tuple, np.ndarray)):
        return list(value)
    return str(value)


def _write(data):
    folder = _slot()
    written = st.session_state.setdefault('_autosave_written', {})
    if written and not (folder/STATE).is_file():
        # The folder was emptied (by another session's clean-up): write everything again.
        written.clear()
    skipped = st.session_state.setdefault('_autosave_skipped', set())
    signatures = _signatures(data)
    state = st.session_state
    for kind, signature in signatures.items():
        if signature is None:
            written[kind] = (None, None)
            skipped.discard(kind)
            continue
        if written.get(kind, (None,))[0] == signature:
            continue
        token = uuid.uuid4().hex[:12]
        if kind == 'usages':
            name = f'usages-{token}.json.gz'
            _write_json_gz(folder/name, {'word': data.word, 'forms': data.forms, 'corpus_order': data.corpus_labels,
                                         'time_label': data.time_label, 'usages': [asdict(u) for u in data.usages]})
        elif kind == 'vectors':
            vectors = state.embedded_usages.embeddings.detach().cpu().numpy()
            size = vectors.size*2
            if not _room_for(folder, size):
                skipped.add(kind)
                _skipped_once(kind, size)
                written[kind] = (signature, None)
                continue
            name = f'vectors-{token}.npy'
            _atomic(folder/name, lambda f: np.save(f, vectors.astype(np.float16), allow_pickle=False))
        elif kind == 'texts':
            texts = state.source_texts
            size = sum(len(t) for t in texts.values())//3
            if not _room_for(folder, size):
                skipped.add(kind)
                _skipped_once(kind, size)
                written[kind] = (signature, None)
                continue
            name = f'texts-{token}.json.gz'
            _write_json_gz(folder/name, texts)
        else:
            name = f'definitions-{token}.npy'
            vectors = state.definition_embeddings.detach().cpu().numpy().astype(np.float32)
            _atomic(folder/name, lambda f: np.save(f, vectors, allow_pickle=False))
        skipped.discard(kind)
        written[kind] = (signature, name)
    files = {kind: written.get(kind, (None, None))[1] for kind in signatures}
    files['annotations'] = _write_annotations(folder, data, written)
    text = json.dumps(_state(data, files, skipped), ensure_ascii=False, sort_keys=True, default=_jsonable)
    if text == st.session_state.get('_autosave_text') and (folder/STATE).is_file():
        return
    _atomic(folder/STATE, lambda f: f.write(text.encode()))
    st.session_state['_autosave_text'] = text
    # Large files that the new state no longer names.
    keep = {STATE, *[name for name in files.values() if name]}
    for path in folder.iterdir():
        if path.name not in keep and not path.name.startswith('.saving-'):
            path.unlink(missing_ok=True)


def autosave():
    """Keep the open analysis on disk. Called at the end of every run; writes only what changed."""
    if st.session_state.get('_autosave_off') or st.session_state.get('target_word_data') is None:
        return
    try:
        _write(st.session_state.target_word_data)
    except (OSError, TypeError, ValueError) as exc:
        # For example a read-only folder: the session goes on without autosave.
        st.session_state['_autosave_off'] = str(exc)


def autosave_status():
    """'on', 'partial' (embeddings or texts skipped for disk space), 'off' or None (nothing to keep yet)."""
    if st.session_state.get('_autosave_off'):
        return 'off'
    if not st.session_state.get('_autosave_text'):
        return None
    return 'partial' if st.session_state.get('_autosave_skipped') else 'on'


# --- Continuing --------------------------------------------------------------------------------------------

def recoverable():
    """Earlier sessions that can be continued, most recently changed first."""
    if not AUTOSAVE_ROOT.is_dir():
        return []
    own = st.session_state.get('_autosave_slot')
    found = []
    for folder in AUTOSAVE_ROOT.iterdir():
        if not folder.is_dir() or str(folder) == own:
            continue
        try:
            state = json.loads((folder/STATE).read_text())
            changed = (folder/STATE).stat().st_mtime
        except (OSError, ValueError):
            continue
        if state.get('version') != 1 or not (state.get('files') or {}).get('usages'):
            continue
        found.append({'folder': str(folder), 'changed': changed, 'summary': state.get('summary', {}),
                      'skipped': state.get('skipped', [])})
    return sorted(found, key=lambda s: -s['changed'])[:KEEP]


def restore(folder):
    """Open an earlier session's work, reading its files only; this session then keeps it in the same folder."""
    from app.ui_data import _load_snapshot
    folder = Path(folder)
    state = json.loads((folder/STATE).read_text())
    files = state['files']
    content = _read_json_gz(folder/files['usages'])
    data = TargetWordData(content['word'], [UsageInstance(**u) for u in content['usages']], content['corpus_order'],
                          content.get('forms', []), content.get('time_label', ''))
    embedded = None
    if files.get('vectors') and (folder/files['vectors']).is_file():
        vectors = np.load(folder/files['vectors'], allow_pickle=False)
        if vectors.ndim == 2 and len(vectors) == len(data):
            metadata = {**(state.get('embedding_metadata') or {}), 'stored_precision': STORED_PRECISION}
            embedded = embedded_usages_from_precomputed(data, torch.from_numpy(vectors.astype(np.float32)), metadata=metadata)
    formatted = list(state.get('definitions_formatted') or [])
    definitions = None
    if embedded is not None and files.get('definitions') and (folder/files['definitions']).is_file():
        vectors = np.load(folder/files['definitions'], allow_pickle=False)
        if vectors.ndim == 2 and len(vectors) == len(formatted) and vectors.shape[1] == embedded.dim:
            definitions = torch.from_numpy(vectors.astype(np.float32))
    texts = _read_json_gz(folder/files['texts']) if files.get('texts') and (folder/files['texts']).is_file() else {}
    # Annotations: their own file; a state written before it existed holds them itself.
    by_id = dict(state.get('annotations') or {})
    if files.get('annotations') and (folder/files['annotations']).is_file():
        by_id.update(_read_annotations(folder/files['annotations']))
    annotations = {i: by_id[u.usage_id] for i, u in enumerate(data.usages) if u.usage_id in by_id}
    _load_snapshot({'data': data, 'embedded': embedded, 'source_texts': texts, 'definitions': formatted,
                    'definition_embeddings': definitions, 'settings': state.get('settings') or {},
                    'annotations': annotations, 'sense_classes': state.get('sense_classes') or [],
                    'annotation_options': state.get('annotation_options')},
                   analysis_name=state.get('analysis_name') or data.word)
    session = st.session_state
    # Without their embeddings the definitions stay as text, ready to create the space again.
    session.definitions_raw = list(state.get('definitions_raw') or formatted)
    if definitions is None:
        session.definitions_formatted = []
    source = state.get('source') or {}
    session.source_lines = bool(source.get('lines'))
    session.source_split = source.get('split') or 'auto'
    session['_source_kind'] = source.get('kind') if texts else None
    session.saved_cache_path = source.get('cache')
    session.embeddings_source = state.get('embeddings_source') or 'model'
    if state.get('data_source'):
        session['_data_source'] = state['data_source']
    path = state.get('analysis_path')
    if path and Path(path).is_file():
        session['_active_analysis_path'] = path
    # Continue in the same folder: what was read is already on disk.
    session['_autosave_slot'] = str(folder)
    signatures = _signatures(data)
    session['_autosave_written'] = {
        'usages': (signatures['usages'], files['usages']),
        'vectors': (signatures['vectors'], files.get('vectors') if embedded is not None else None),
        'texts': (signatures['texts'], files.get('texts') if texts else None),
        'definitions': (signatures['definitions'], files.get('definitions') if definitions is not None else None)}
    # The annotations' file is written whole at the next change: a line left unfinished by an interruption is not appended to.
    session.pop('_autosave_annotations', None)
    session['_autosave_skipped'] = set(state.get('skipped') or [])
    session.pop('_autosave_text', None)
    return {'embedded': embedded is not None, 'definitions': definitions is not None,
            'workspace': state.get('workspace'), 'skipped': state.get('skipped') or []}


def latest_discovery(root):
    """The most recently changed saved discovery, for Discover's offer to continue."""
    from semlens.discover.project import DISCOVERY_FILE, list_discoveries
    found = []
    for item in list_discoveries(root):
        try:
            item['changed'] = (Path(item['folder'])/DISCOVERY_FILE).stat().st_mtime
        except OSError:
            continue
        found.append(item)
    return max(found, key=lambda d: d['changed']) if found else None


def when(timestamp):
    """'today at 14:02', 'yesterday at 09:15' or '24 Nov at 18:40'."""
    moment = time.localtime(timestamp)
    today = time.localtime()
    clock = time.strftime('%H:%M', moment)
    days = (time.mktime(today[:3]+(0, 0, 0, 0, 0, -1))-time.mktime(moment[:3]+(0, 0, 0, 0, 0, -1)))/86400
    if round(days) == 0:
        return f'today at {clock}'
    if round(days) == 1:
        return f'yesterday at {clock}'
    year = f' {moment.tm_year}' if moment.tm_year != today.tm_year else ''
    return f'{moment.tm_mday} {time.strftime("%b", moment)}{year} at {clock}'
