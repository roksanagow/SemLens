"""Portable analysis snapshots with JSON metadata and non-pickled NumPy arrays."""
from __future__ import annotations
from dataclasses import asdict
import io
import json
from pathlib import Path
import os
import tempfile
import zipfile
import numpy as np
import torch
from .data_loading import TargetWordData, UsageInstance, validate_embeddings
from .embeddings import embedded_usages_from_precomputed


def project_bytes(data, embedded, *, definitions=None, definition_embeddings=None, annotations=None,
                  sense_classes=None, settings=None, source_texts=None):
    ids={i:u.usage_id for i,u in enumerate(data.usages)}
    manifest={'version':1,'word':data.word,'forms':data.forms,'corpus_order':data.corpus_labels,
              'usages':[asdict(u) for u in data.usages], 'embedding_metadata':embedded.metadata,
              'definitions':list(definitions or []),'annotations':{ids[i]:v for i,v in (annotations or {}).items()},
              'sense_classes':list(sense_classes or []),'settings':settings or {},'source_texts':source_texts or {}}
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('project.json',json.dumps(manifest,ensure_ascii=False))
        arrays={'usages.npy':embedded.embeddings}
        if definition_embeddings is not None:
            arrays['definitions.npy']=definition_embeddings
        for name,tensor in arrays.items():
            content=io.BytesIO()
            np.save(content,tensor.detach().cpu().numpy(),allow_pickle=False)
            z.writestr(name,content.getvalue())
    return output.getvalue()


def load_project(source):
    if isinstance(source,(bytes,bytearray)):
        source=io.BytesIO(source)
    with zipfile.ZipFile(source) as z:
        manifest=json.loads(z.read('project.json'))
        if manifest.get('version')!=1:
            raise ValueError('Unsupported SemLens analysis version.')
        data=TargetWordData(manifest['word'],[UsageInstance(**u) for u in manifest['usages']],
                            manifest['corpus_order'],manifest.get('forms',[]))
        embs=validate_embeddings(torch.from_numpy(np.load(io.BytesIO(z.read('usages.npy')),allow_pickle=False)),len(data))
        embedded=embedded_usages_from_precomputed(data,embs,metadata=manifest.get('embedding_metadata',{}))
        defs=None
        if 'definitions.npy' in z.namelist():
            defs=validate_embeddings(torch.from_numpy(np.load(io.BytesIO(z.read('definitions.npy')),allow_pickle=False)),len(manifest['definitions']))
            if defs.shape[1]!=embedded.dim:
                raise ValueError('Saved definition vectors do not match usage dimensions.')
        annotation_map=manifest.get('annotations',{})
        annotations={i:annotation_map[u.usage_id] for i,u in enumerate(data.usages) if u.usage_id in annotation_map}
        return {'data':data,'embedded':embedded,'definitions':manifest.get('definitions',[]),
                'definition_embeddings':defs,'annotations':annotations,'sense_classes':manifest.get('sense_classes',[]),
                'settings':manifest.get('settings',{}),'source_texts':manifest.get('source_texts',{})}


def save_project(path: str | Path, content: bytes):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.saving-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(content)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def project_filename(name: str) -> str:
    """Keep readable names while restricting them to one portable filename."""
    name=name.strip()
    if name.lower().endswith('.semlens'):
        name=name[:-8].strip()
    if (not name or name.startswith('.') or name.endswith('.') or
            any(c in '<>:"/\\|?*' or ord(c)<32 for c in name)):
        raise ValueError('Enter a name without slashes or these characters: < > : " | ? *. Names cannot begin or end with a dot.')
    return f'{name}.semlens'


def _saved_file(path: str | Path) -> Path:
    path=Path(path)
    if path.suffix!='.semlens' or path.is_symlink() or not path.is_file():
        raise ValueError('That saved file is no longer available. Select another file.')
    return path


def rename_project(path: str | Path, name: str) -> Path:
    """Rename without recomputing vectors or replacing another saved file."""
    path=_saved_file(path)
    destination=path.with_name(project_filename(name))
    if destination==path:
        return path
    if destination.exists():
        # Allow a spelling/case correction on case-insensitive filesystems.
        if destination.name.casefold()==path.name.casefold() and destination.samefile(path):
            return path.rename(destination)
        raise FileExistsError(f'A saved file named “{destination.stem}” already exists. Choose another name.')
    # Both names are in one directory. Linking is fast even for large snapshots,
    # and refuses an existing destination if another session creates it first.
    os.link(path,destination)
    try:
        path.unlink()
    except OSError:
        destination.unlink()
        raise
    return destination


def delete_project(path: str | Path):
    """Delete only the selected snapshot, leaving corpus caches untouched."""
    _saved_file(path).unlink()
