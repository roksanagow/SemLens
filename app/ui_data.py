"""Data input, reusable corpus preparation, and portable analysis snapshots."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import time
import io
import json
from pathlib import Path
import re
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import torch
from semlens.timeline import time_from_corpus_name
from semlens.words import MAX_WORDS,default_word,entry_counts,word_names
from semlens.data_loading import (TargetWordData,normalise_forms,corpus_sentences,select_usages,load_from_csv,load_precomputed_embeddings,table_columns,times_from_corpus_names,
                                  validate_corpus_names,validate_embeddings,decode_text,count_words)
from semlens.utils import keyword_in_context,usage_search_pattern
from semlens.embeddings import embed_usages,embedded_usages_from_precomputed
from semlens.corpus_store import (CorpusCache,list_corpus_caches,estimate_storage,create_corpus_cache,format_bytes)
from semlens.projects import project_bytes,load_project,save_project,project_filename,rename_project,delete_project
from semlens.spaces import project_to_definition_space
from app.state import set_data,reset_downstream_of
from app.annotation_store import get_annotations,get_sense_classes,get_options,restore_annotations
from app.common import ROOT,STORAGE,get_model,show_usage,ordered_labels,advanced,natural_sort_key
from semlens.resources import count_tokens,embedding_cost
from app.costs import (MODEL_SIZES,_hub_model_size,_folder_weights,cost_check,known_speed,model_profile,
                       device_kind,measure_speed,remember_speed,model_is_downloaded)
from app.presentation import readable_table,concordance_html


# Shipped inside the package so that an installed copy has it too; a test keeps it equal to testing_texts/.
EXAMPLE_PATH=Path(__file__).resolve().parent/'data'/'bubble_usages.csv'
# The example's XL-LEXEME embeddings (16-bit) and four definitions, so that it opens without a model download.
# Made by scripts/build_example_embeddings.py; a test keeps them matched to the sentences.
EXAMPLE_VECTORS=Path(__file__).resolve().parent/'data'/'bubble_embeddings.npz'
EXAMPLE_SOURCE='Example: “bubble” in the Guardian, 1999–2026'
EXAMPLE_DESCRIPTION=('350 sentences containing *bubble* or *bubbles*, sampled from Guardian and Observer articles: 50 from each of seven periods '
                     'between 1999 and 2026, one per article. The sample was not chosen for particular senses, so it mixes soap bubbles, '
                     'financial bubbles, social and political bubbles, and the support bubbles of the Covid years.\n\n'
                     'It comes already embedded with XL-LEXEME, and with four definitions, so it opens at once, without downloading the model. '
                     'Publication date, title and link come with each sentence and are returned in the annotation export.')
SAVED_SETTINGS=('definition_settings','chronological','corpus_order','distance_metric','sampling_mode','sampling_repetitions','sampling_seed',
                'source_lines','source_split','palette_name','words_mode','word_mapping','group_by','words_shown','corpora_shown')
MODEL_PRESETS={
    'XL-LEXEME · trained to compare word meanings in context (recommended)':'pierluigic/xl-lexeme',
    'XLM-RoBERTa large · general purpose':'FacebookAI/xlm-roberta-large',
    'XLM-RoBERTa base · general purpose, smaller and faster':'FacebookAI/xlm-roberta-base',
    'BERT base · English only, small':'google-bert/bert-base-uncased',
    'Another model…':None,
}
MODEL_HELP=('The model that reads each sentence and turns each usage into numbers. All of these read many languages unless marked English.\n\n'
            'Choose **Another model…** to use any other encoder:\n\n'
            '- **From Hugging Face**: type its name as shown on its page (organisation/model, such as FacebookAI/xlm-roberta-base), or paste the page’s link.\n'
            '- **From GitHub or elsewhere**: download the model’s files (config.json, the weights and the tokenizer files) into a folder, '
            'and give the path of that folder.\n\n'
            'The first use downloads the model and keeps it for later runs. As a rough guide, a model needs about 4 GB of disk space per billion parameters '
            '(2 GB if it is stored in 16-bit), and about as much free memory while it embeds. Models stored only in the older .bin format can take twice the '
            'disk space, because a converted copy is saved alongside.')


def _parameters_text(count):
    return f'{count/1e9:.1f} billion' if count>=1e9 else f'{round(count/1e6):,} million'


def normalise_model_reference(text):
    """A Hugging Face name from a typed name or a pasted page link; other text is kept as it is."""
    text=text.strip()
    if text.startswith('~'):
        return str(Path(text).expanduser())
    match=re.match(r'^(?:https?://)?(?:www\.)?huggingface\.co/([^/\s]+/[^/\s?#]+)',text)
    return match.group(1) if match else text


def _model_size_note(model_id):
    """A rough statement of the disk space and memory a model needs."""
    if model_id in MODEL_SIZES:
        params,download,disk=MODEL_SIZES[model_id]
        note=f'About {_parameters_text(params)} parameters: a {download:g} GB download the first time, kept for later runs'
        if disk>download:
            note+=f'. Hugging Face also saves a converted copy of its weights, so allow about {disk:g} GB of disk space'
        return note+'. Embedding needs about as much free memory as the download.'
    if not model_id:
        return ''
    if Path(model_id).expanduser().is_dir():
        size=_folder_weights(model_id)
        return (f'This folder holds {size/1e9:.1f} GB of model weights; embedding needs about as much free memory.' if size
                else 'No weights (.safetensors or .bin files) were found in this folder.')
    found=_hub_model_size(model_id)
    if found is None:
        return ('Could not look this model up on Hugging Face (no connection, or no model of that name). '
                'As a rough guide: about 4 GB of disk space per billion parameters, and about as much free memory.')
    params,safe,binary=found
    weights=safe or binary
    if not weights:
        return 'Hugging Face lists no weight files for this model, so its size is unknown.'
    params=params or weights/4
    note=f'About {_parameters_text(params)} parameters: a {weights/1e9:.1f} GB download the first time, kept for later runs'
    if not safe:
        note+=f'. It is stored only in the older .bin format, so allow about {2*weights/1e9:.1f} GB of disk space for a converted copy'
    return note+'. Embedding needs about as much free memory as the download.'
GUIDE=(
    'SemLens shows how one word is used across collections of text, such as different periods, and helps you judge whether its meaning differs between them.\n\n'
    '1. **Data & Model** (this page). Give the word and one text per corpus. SemLens finds every sentence containing the word, '
    'and a language model turns each of those usages into numbers that capture how the word is used there.\n'
    '2. **Explore**. Every usage becomes a point; usages the model finds similar sit close together. Scores summarise how far apart two corpora are, '
    'and you can read the sentences behind any score.\n'
    '3. **Definitions**. Write the senses you expect as short definitions and see which usages lie nearest each one.\n'
    '4. **Annotation**. Name the senses you want to tell apart, label usages with them, then read off how often each sense occurs in each corpus.\n\n'
    'The model’s similarity is evidence to inspect, not a verdict: the tool is built so that every number leads back to sentences you can read. '
    'To try it without your own data, choose **Example** under Data source. The **Glossary** in the sidebar explains the terms, and each **(?)** explains its control.')


def _forms(value):
    return [s.strip() for s in re.split(r'[,\n]',value) if s.strip()]


def _set_words_mode(widget_key):
    st.session_state.words_mode=st.session_state[widget_key]


def render_words_question(entries,data,place):
    """Ask whether the forms are one word or different words; with different words, which word each form belongs to."""
    if len(entries)<2:
        return
    widget=f'_words_mode_{place}'
    st.session_state[widget]=st.session_state.get('words_mode','forms')
    mode=st.radio('Are these forms of one word, or different words to compare?',['forms','words'],key=widget,
                  on_change=_set_words_mode,args=(widget,),
                  format_func={'forms':'Forms of one word (run, runs, ran)','words':'Different words to compare (bubble, balloon)'}.get,
                  help='**Forms of one word**: every usage is analysed together, as now.\n\n'
                       '**Different words**: each word gets its own colour and can be compared with the others, across corpora or within one. '
                       'A switch at the top of Explore, Definitions and Annotation chooses between colouring by corpus and by word, '
                       'definitions are read as whole sentences by default, and Explore adds how often each word occurs in each corpus.')
    if mode!='words':
        return
    names=word_names(entries,st.session_state.word_mapping)
    counts=entry_counts(data,entries) if data is not None and data.word.casefold()==entries[0].casefold() else None
    table=pd.DataFrame({'Form':entries,'Word':[names[e] for e in entries]})
    if counts is not None:
        # How many usages each form found, once usages of this word are loaded.
        table['Found']=[counts[e] for e in entries]
    st.write('Which word does each form belong to?')
    key='word_table_'+hashlib.sha256(json.dumps(entries).encode()).hexdigest()[:12]
    edited=st.data_editor(table,hide_index=True,width='stretch',key=key,disabled=['Form','Found'],
                          column_config={'Form':st.column_config.TextColumn('Form',width='medium'),
                                         'Word':st.column_config.TextColumn('Word',width='medium',
                                             help='Give two forms the same word to count them as one. Capitals are ignored when comparing names.'),
                                         'Found':st.column_config.NumberColumn('Found',width='small',format='%d',
                                             help='Usages each form found among the loaded data.')})
    mapping=dict(st.session_state.word_mapping)
    for entry,word in zip(entries,edited['Word']):
        word=str(word or '').strip()
        if not word or word==default_word(entry):
            mapping.pop(entry,None)
        else:
            mapping[entry]=word
    if mapping!=st.session_state.word_mapping:
        st.session_state.word_mapping=mapping
        st.rerun()
    distinct=len(set(names.values()))
    st.caption('Give two forms the same word to count them as one. A wildcard such as balloon* is always one word. '
               'Changing this regroups the usages at once: nothing needs loading or embedding again, and annotations stay.')
    if distinct>MAX_WORDS:
        st.warning(f'SemLens compares up to {MAX_WORDS} words, and these forms make {distinct}. Give some forms the same word.')
    elif distinct<2:
        st.info('All the forms belong to one word, so there is nothing to compare. Give them different words, or choose “Forms of one word”.')


def _parse_cap(text):
    """'all' (or blank, or 0) keeps every occurrence; otherwise a positive whole number."""
    value=str(text).strip().casefold().replace(',','').replace('_','')
    if value in {'','all','0'}:
        return 0
    if not value.isdigit():
        raise ValueError('Maximum usages per corpus must be “all” or a positive whole number, such as 500.')
    return int(value)


def _cap(data,cap):
    if not cap:
        return data,list(range(len(data)))
    rng=np.random.default_rng(42)
    kept=[]
    for c in data.corpus_labels:
        ids=data.indices_for_corpus(c)
        kept.extend(sorted(rng.choice(ids,min(cap,len(ids)),replace=False).tolist()))
    kept=sorted(kept)
    return data.subset(kept),kept


def _set_loaded(data,embedded=None,source='model'):
    set_data(data,embedded,source=source)
    corpora=f'{len(data.corpus_labels)} corpus' if len(data.corpus_labels)==1 else f'{len(data.corpus_labels)} corpora'
    if embedded is None:
        _notify('success',f'Loaded {len(data):,} usages from {corpora}. Next: embed them, below.')
    else:
        _notify('success',f'Loaded {len(data):,} usages from {corpora}, with their embeddings. Next: explore them in Explore.')


def _example_embeddings(data):
    """The packaged embeddings and definitions of the example, or None when they do not match its sentences."""
    try:
        with np.load(EXAMPLE_VECTORS,allow_pickle=False) as saved:
            ids=saved['usage_ids'].tolist()
            vectors=saved['vectors'].astype(np.float32)
            definitions=saved['definition_vectors'].astype(np.float32)
            manifest=json.loads(saved['manifest'].item())
    except (OSError,ValueError,KeyError):
        return None
    if ids!=[u.usage_id for u in data.usages] or len(definitions)!=len(manifest.get('definitions_formatted',[])):
        return None
    return vectors,definitions,manifest


def load_example():
    """Load the example, with its packaged embeddings and definitions when they match; True when they were included."""
    data=load_from_csv(EXAMPLE_PATH,'bubble',forms=['bubbles'],time_col='date')
    packaged=_example_embeddings(data)
    if packaged is None:
        _set_loaded(data)
    else:
        vectors,definitions,manifest=packaged
        metadata={**manifest['embedding_metadata'],'stored_precision':'float16'}
        embedded=embedded_usages_from_precomputed(data,torch.from_numpy(vectors),metadata=metadata)
        set_data(data,embedded,source='model')
        definition_vectors=torch.from_numpy(definitions)
        st.session_state.definitions_raw=list(manifest['definitions_raw'])
        st.session_state.definitions_formatted=list(manifest['definitions_formatted'])
        st.session_state.definition_embeddings=definition_vectors
        st.session_state.def_space_all=project_to_definition_space(embedded.embeddings,definition_vectors)
        st.session_state.definition_settings=dict(manifest['definition_settings'])
        _notify('success',f'Loaded the example: {len(data):,} usages from {len(data.corpus_labels)} periods, already embedded with XL-LEXEME, '
                f'and {len(definitions)} definitions. Next: explore the usages in Explore.')
    st.session_state.source_texts={}
    st.session_state.saved_cache_path=None
    st.session_state.pop('_corpus_stats',None)
    st.session_state.chronological=True
    return packaged is not None


# --- Work that loading would replace ------------------------------------------------------------------

def _work_stamp():
    """The user's own work in the open analysis: what saving keeps and loading other usages can lose."""
    data=st.session_state.get('target_word_data')
    return json.dumps([id(data),{str(i):v for i,v in get_annotations().items()},get_sense_classes(),
                       st.session_state.get('definitions_formatted') or []],sort_keys=True,default=str)


def mark_saved():
    st.session_state['_saved_work']=_work_stamp()


def unsaved_work(word=None):
    """Definitions, annotations and senses that loading other usages would lose, unless they are saved.

    ``word`` is the word about to be loaded. Loading the open word again keeps the
    annotations of the usages that are loaded again, and the senses; ``None``
    replaces the whole analysis.
    """
    data=st.session_state.get('target_word_data')
    if data is None:
        return {}
    work={}
    if st.session_state.get('definitions_formatted'):
        work['definitions']=len(st.session_state.definitions_formatted)
    if word is None or word!=data.word:
        if get_annotations():
            work['annotated']=len(get_annotations())
        if get_sense_classes():
            work['senses']=len(get_sense_classes())
    if work and st.session_state.get('_saved_work')==_work_stamp():
        return {}
    return work


def describe_work(work):
    """'4 definitions and the annotations of 120 usages (3 senses)'."""
    parts=[]
    if work.get('definitions'):
        parts.append(f"{work['definitions']} definition{'s' if work['definitions']!=1 else ''}")
    senses=work.get('senses',0)
    senses_text=f"{senses} sense{'s' if senses!=1 else ''}"
    if work.get('annotated'):
        parts.append(f"the annotations of {work['annotated']:,} usage{'s' if work['annotated']!=1 else ''}"+(f' ({senses_text})' if senses else ''))
    elif senses:
        parts.append(senses_text)
    return ' and '.join(parts)


def _analysis_bytes():
    settings={k:st.session_state[k] for k in SAVED_SETTINGS}
    return project_bytes(st.session_state.target_word_data,st.session_state.embedded_usages,definitions=st.session_state.definitions_formatted,
                         definition_embeddings=st.session_state.definition_embeddings,annotations=get_annotations(),
                         sense_classes=get_sense_classes(),settings=settings,source_texts=st.session_state.source_texts,
                         annotation_options=get_options())


def save_open_analysis():
    """Save the open analysis under its name in Saved analysis, numbering the name rather than replacing another file."""
    data=st.session_state.target_word_data
    name=str(st.session_state.get('_analysis_save_name') or data.word).strip() or data.word
    folder=STORAGE/'projects'
    path=folder/project_filename(name)
    active=st.session_state.get('_active_analysis_path')
    number=2
    while path.exists() and str(path)!=active:
        path=folder/project_filename(f'{name} {number}')
        number+=1
    save_project(path,_analysis_bytes())
    st.session_state['_active_analysis_path']=str(path)
    mark_saved()
    return path


def _cancel_replacing():
    st.session_state.pop('_replace_pending',None)


def confirm_replacing(clicked,key,word=None,action='Loading'):
    """Whether a load may replace the open analysis: at once when no unsaved work would be lost, else once the user agrees.

    Called with the load button's value, just after it, so that the question appears
    under the button. ``word`` is the word about to be loaded (see :func:`unsaved_work`).
    """
    if clicked:
        if not unsaved_work(word):
            _cancel_replacing()
            return True
        st.session_state['_replace_pending']=key
    if st.session_state.get('_replace_pending')!=key:
        return False
    work=unsaved_work(word)
    if not work:
        _cancel_replacing()
        return False
    data=st.session_state.target_word_data
    kept=(f' The annotations of the usages of “{data.word}” that are loaded again are kept, with the senses.'
          if word is not None and word==data.word and get_annotations() else '')
    can_save=st.session_state.get('embedded_usages') is not None
    with st.container(border=True,key=f'replace_{key}'):
        st.warning(f'{action} replaces the open analysis of “{data.word}”. Not saved yet: {describe_work(work)}.{kept}',icon=':material/warning:')
        with st.container(horizontal=True):
            if st.button('Save it, then continue',type='primary',key=f'{key}_save_first',disabled=not can_save,
                         help='Saves the open analysis in Saved analysis under its name, then continues.' if can_save else
                              'An analysis can be saved once its usages are embedded.'):
                try:
                    path=save_open_analysis()
                    _notify('success',f'Saved the earlier analysis as “{path.stem}”: reopen it from Data source → Saved analysis.')
                    _cancel_replacing()
                    return True
                except (OSError,ValueError) as exc:
                    st.error(str(exc))
            if st.button('Continue without saving',key=f'{key}_discard'):
                _cancel_replacing()
                return True
            st.button('Cancel',key=f'{key}_cancel',on_click=_cancel_replacing)
    return False


def _notify(kind,message):
    st.session_state.setdefault('_load_notices',[]).append((kind,message))


def _finish_load():
    """Redraw from the top, so the sidebar summary and corpus order show the new data at once."""
    st.rerun()


def show_load_notices():
    """What the last load did, and what to do next: shown once, under the tabs of whichever workspace opens."""
    for kind,message in st.session_state.pop('_load_notices',[]):
        getattr(st,kind)(message)


def _load_snapshot(snapshot, *, analysis_path=None, analysis_name=None):
    set_data(snapshot['data'],snapshot['embedded'],source='saved',preserve_annotations=False)
    st.session_state.source_texts=snapshot['source_texts']
    st.session_state.definitions_raw=snapshot['definitions']
    st.session_state.definitions_formatted=snapshot['definitions']
    st.session_state.definition_embeddings=snapshot['definition_embeddings']
    if snapshot['definition_embeddings'] is not None:
        st.session_state.def_space_all=project_to_definition_space(snapshot['embedded'].embeddings,snapshot['definition_embeddings'])
    settings=snapshot['settings']
    for key in SAVED_SETTINGS:
        if key in settings:
            st.session_state[key]=settings[key]
    restore_annotations(snapshot['annotations'],snapshot['sense_classes'],snapshot.get('annotation_options'))
    st.session_state.pop('_corpus_stats',None)
    # Which input these source texts came from, so they are offered again only there.
    st.session_state['_source_kind']=(('Paste sentences' if st.session_state.source_lines else 'Upload raw corpora (.txt)')
                                      if snapshot['source_texts'] else None)
    if analysis_path is not None:
        st.session_state['_active_analysis_path']=str(analysis_path)
        st.session_state['_analysis_save_name']=analysis_path.stem
    elif analysis_name:
        st.session_state['_analysis_save_name']=analysis_name


def _cancel_file_deletion():
    st.session_state.pop('_delete_saved_file',None)


def _request_file_deletion(path):
    st.session_state['_delete_saved_file']=str(path)


def _rename_saved_file(path,name_key,choice_key,scope):
    try:
        renamed=rename_project(path,st.session_state[name_key])
        st.session_state[choice_key]=renamed
        if st.session_state.get('_active_analysis_path')==str(path):
            st.session_state['_active_analysis_path']=str(renamed)
            st.session_state['_analysis_save_name']=renamed.stem
        _cancel_file_deletion()
        st.session_state['_saved_file_notice']=(scope,'success',f'Renamed “{path.stem}” to “{renamed.stem}”.')
    except (OSError,ValueError) as exc:
        st.session_state['_saved_file_notice']=(scope,'error',str(exc))


def _delete_saved_file(path,choice_key,scope):
    try:
        delete_project(path)
        st.session_state.pop(choice_key,None)
        if st.session_state.get('_active_analysis_path')==str(path):
            st.session_state.pop('_active_analysis_path',None)
            st.session_state.pop('_prepared_analysis',None)
        _cancel_file_deletion()
        st.session_state['_saved_file_notice']=(scope,'success',f'Deleted “{path.stem}”. Any analysis already open remains available in this session.')
    except (OSError,ValueError) as exc:
        st.session_state['_saved_file_notice']=(scope,'error',str(exc))


def _saved_file_controls(word_only):
    scope='words' if word_only else 'projects'
    folder=STORAGE/scope
    paths=sorted((p for p in folder.glob('*.semlens') if p.is_file() and not p.is_symlink()),key=lambda p:p.name.casefold())
    notice=st.session_state.pop('_saved_file_notice',None)
    if notice and notice[0]==scope:
        getattr(st,notice[1])(notice[2])
    if not paths:
        st.caption('No word embeddings saved on this computer yet.' if word_only else
                   'No analyses saved on this computer yet. Once usages are embedded, **Save or export this analysis**, at the bottom of every page, saves one here.')
        return
    choice_key=f'saved_file_{scope}'
    if st.session_state.get(choice_key) not in paths:
        st.session_state[choice_key]=paths[0]
    choice=st.selectbox('Saved locally',paths,format_func=lambda p:p.stem,key=choice_key,on_change=_cancel_file_deletion)
    if confirm_replacing(st.button('Open saved word embeddings' if word_only else 'Open saved analysis',type='primary'),f'open_{scope}',action='Opening it'):
        try:
            _load_snapshot(load_project(choice),analysis_path=None if word_only else choice)
            mark_saved()
            _notify('success',f'Opened the word embeddings “{choice.stem}”, with their sentences. Next: explore them in Explore.' if word_only else
                    f'Opened “{choice.stem}”. Changes are kept automatically as you work; save them into this file with **Save or export this analysis**, below.')
            _finish_load()
        except Exception as exc:
            st.error(str(exc))
    with st.expander('Manage saved word embeddings' if word_only else 'Manage saved analysis',key=f'manage_saved_{scope}',on_change='rerun'):
        st.caption(f'Selected file: {choice.name}')
        name_key=f'rename_{scope}_{choice.name}'
        with st.form(f'rename_form_{scope}_{choice.name}'):
            st.text_input('New name',value=choice.stem,key=name_key)
            st.form_submit_button('Rename',on_click=_rename_saved_file,args=(choice,name_key,choice_key,scope))
        if st.session_state.get('_delete_saved_file')==str(choice):
            st.warning(f'Permanently delete “{choice.stem}”? This removes the saved file. The currently open analysis and separately saved corpus embeddings remain available.')
            cols=st.columns(2)
            cols[0].button('Confirm deletion',on_click=_delete_saved_file,args=(choice,choice_key,scope))
            cols[1].button('Cancel',on_click=_cancel_file_deletion)
        else:
            st.button('Delete saved file…',on_click=_request_file_deletion,args=(choice,))


def render_data():
    st.header('Data & Model')
    st.caption('Choose a word, add corpora, then explore their usages together.')
    with st.expander('New here? How SemLens works',expanded=st.session_state.target_word_data is None):
        st.markdown(GUIDE)
    sources=['Upload raw corpora (.txt)','Paste sentences','Upload CSV / TSV']
    if EXAMPLE_PATH.is_file():
        sources.append(EXAMPLE_SOURCE)
    # Vectors saved for one word, or made with other code, are for those who know what they hold.
    sources+=['Saved corpora']+(['Saved word embeddings','Pre-computed embeddings'] if advanced() else [])+['Saved analysis']
    # Remembered, so returning from another workspace does not reset the choice.
    remembered=st.session_state.get('_data_source')
    mode=st.selectbox('Data source',sources,index=sources.index(remembered) if remembered in sources else 0,
                      help='**Upload raw corpora**: one plain-text file per corpus; SemLens finds the word.\n\n'
                           '**Paste sentences**: type or paste text for each corpus.\n\n'
                           '**Upload CSV / TSV**: a table of sentences you have already selected, one row per usage.\n\n'
                           '**Example**: a small ready-made dataset to try the tool.\n\n'
                           '**Saved corpora / word embeddings / analysis**: work you saved earlier.\n\n'
                           '**Pre-computed embeddings**: vectors you produced with your own code.')
    st.session_state['_data_source']=mode
    corpus_input=None
    if mode==EXAMPLE_SOURCE:
        st.markdown(EXAMPLE_DESCRIPTION)
        if confirm_replacing(st.button('Load the example',type='primary'),'load_example',word='bubble'):
            try:
                load_example()
                _finish_load()
            except Exception as exc:
                st.error(str(exc))
        _loaded_summary()
        st.divider()
        _embedding_controls(None)
    elif mode in {'Saved analysis','Saved word embeddings'}:
        word_only=mode=='Saved word embeddings'
        if word_only:
            st.caption('Reopen vectors for a saved word and its forms, with the matching sentences and model/layer settings. To analyse other words, use Saved corpora.')
        _saved_file_controls(word_only)
        upload=st.file_uploader('Or open an exported embedding file' if word_only else 'Or open an exported analysis',type=['semlens'])
        if upload is not None and confirm_replacing(st.button('Open uploaded file'),'open_upload',action='Opening it'):
            try:
                _load_snapshot(load_project(upload),analysis_name=None if word_only else Path(upload.name).stem)
                mark_saved()
                _notify('success',f'Opened “{Path(upload.name).stem}” from the file. Next: explore it in Explore.')
                _finish_load()
            except Exception as exc:
                st.error(str(exc))
        _loaded_summary()
    else:
        word=st.text_input('Target word',value=st.session_state.target_word,key=f'target_{mode}',
                           help='The word to study, in its dictionary form: run, bubble, Blase. Add its other forms below.').strip()
        forms=_forms(st.text_input('Other forms (optional)',key=f'forms_{mode}',placeholder='For example: runs, running, ran',
                                   help='Other spellings or inflected forms to count as the same word, separated by commas: runs, running, ran. '
                                        'End a form with * to match any ending: runn* finds running and runner. '
                                        'Only whole words are matched, so run never finds runway.'))
        try:
            typed_entries=normalise_forms(word,forms) if word else []
        except ValueError:
            typed_entries=[]
        render_words_question(typed_entries,st.session_state.target_word_data,'input')
        with st.expander('Usage selection options'):
            cap_text=st.text_input('Maximum usages per corpus',value='all',key=f'cap_{mode}',
                                   help='“all” uses every usage. A number such as 500 selects a reproducible random subset of that size from each corpus before embedding; plots show the retained usages.')
            try:
                _parse_cap(cap_text)
            except ValueError as exc:
                st.error(str(exc))
            match_case=st.checkbox('Match upper and lower case exactly',value=False,key=f'case_{mode}',
                                   help='Off: Polish and polish are the same word. On: only the capitalisation you typed is found, '
                                        'which separates names from common nouns (May / may, Bush / bush).')
            split='auto'
            if mode=='Upload raw corpora (.txt)':
                divisions={'auto':'Automatically, at sentence punctuation','lines':'One sentence per line','paragraphs':'One paragraph per context'}
                split=st.selectbox('How the text is divided into contexts',list(divisions),format_func=divisions.get,key='raw_split',
                                   help='Each usage is read, and represented, inside one context.\n\n'
                                        '**Automatically**: sentences end at . ! ? and at blank lines. Titles such as Mr. and Dr. do not end a sentence.\n\n'
                                        '**One sentence per line**: for text already split into sentences, including text without punctuation.\n\n'
                                        '**One paragraph per context**: paragraphs separated by a blank line are kept whole, for when a sentence is too little context.')
            corpus_times=False
            if mode in {'Paste sentences','Upload raw corpora (.txt)'}:
                corpus_times=st.checkbox('Use corpus names as times',value=False,key=f'corpus_times_{mode}',
                                         help='When every corpus is named by a year (1999) or a range of years (1999–2003), give each usage that time: '
                                              'the year, or the middle of the range. This adds the Over time view and colouring by time. '
                                              'For a date per sentence, load a CSV with a date or year column instead.')
        if mode in {'Paste sentences','Upload raw corpora (.txt)'}:
            texts={}
            lines=mode=='Paste sentences'
            # Texts loaded earlier in this session stay available, so another word can be
            # looked up after visiting the other workspaces without entering them again.
            kept=st.session_state.source_texts if st.session_state.get('_source_kind')==mode else {}
            if lines:
                split='lines'
                count=int(st.number_input('Number of corpora',min_value=1,max_value=10,value=max(2,len(kept)) if kept else 2,step=1))
                names=[]
                values=[]
                for i in range(count):
                    with st.expander(f'Corpus {i+1}',expanded=count<=2):
                        earlier=list(kept.items())[i] if i<len(kept) else (f'corpus_{i+1}','')
                        names.append(st.text_input('Corpus name',value=earlier[0],key=f'paste_name_{i}',
                                                   help='A short label shown in plots and tables, such as a period or a genre.').strip())
                        values.append(st.text_area('Sentences (one per line)',value=earlier[1],key=f'paste_text_{i}',height=140))
                texts=dict(zip(names,values))
            else:
                uploads=st.file_uploader('Corpus text files (up to 10)',type=['txt'],accept_multiple_files=True,
                                         help='One plain-text (.txt) file per corpus, for example one per period. Export from Word or a PDF as plain text first.')
                names=[]
                values=[]
                for i,f in enumerate(uploads):
                    names.append(st.text_input(f'Corpus name for {f.name}',value=Path(f.name).stem,key=f'raw_name_{i}').strip())
                    text,encoding=decode_text(f.getvalue())
                    values.append(text)
                    if encoding!='UTF-8':
                        st.caption(f'{f.name} was read as {encoding}. If accented letters look wrong in the review below, save the file as UTF-8 and upload it again.')
                texts=dict(zip(names,values))
                if not uploads and kept:
                    names,texts=list(kept),dict(kept)
                    st.caption(f'Using the {len(kept)} texts loaded earlier in this session: {", ".join(kept)}. '
                               'Enter another word and load again, or upload files to replace them.')
            if confirm_replacing(st.button('Load usages',type='primary'),'load_texts',word=word):
                try:
                    validate_corpus_names(names)
                    records=corpus_sentences(texts,split=split)
                    found=select_usages(records,word,forms,match_case=match_case)
                    data,_=_cap(found,_parse_cap(cap_text))
                    if corpus_times:
                        data,unreadable=times_from_corpus_names(data)
                        if unreadable:
                            _notify('warning','No times were set: these corpus names are not a year or a range of years: '+', '.join(unreadable)+'.')
                        else:
                            # Corpora named by years are in time order: sort them by it.
                            data=TargetWordData(data.word,data.usages,sorted(data.corpus_labels,key=lambda c:time_from_corpus_name(c)),
                                                data.forms,data.time_label)
                            st.session_state.chronological=True
                    _set_loaded(data)
                    st.session_state.source_texts=texts
                    st.session_state.source_lines=lines
                    st.session_state.source_split=split
                    st.session_state['_source_kind']=mode
                    st.session_state.saved_cache_path=None
                    st.session_state['_corpus_stats']={name:{'words':count_words(text),'found':len(found.indices_for_corpus(name))}
                                                       for name,text in texts.items()}
                    missing=[c for c in names if c not in data.corpus_labels]
                    if missing:
                        _notify('warning','No selected forms found in: '+', '.join(missing))
                    _finish_load()
                except Exception as exc:
                    st.error(str(exc))
            # Store all source texts for precomputation even if no target is selected yet.
            corpus_input=(names,texts,split)
        elif mode=='Upload CSV / TSV':
            upload=st.file_uploader('Usage table',type=['csv','tsv'])
            st.caption('One row per sentence, with the columns **sentence** and **corpus**. Optional **start** and **end** give the character position of the '
                       'target in the sentence (counting from 0, end not included) and select exactly that usage; without them every usage of the word is found. '
                       'Any other columns, such as a date or a source, are kept and returned with your annotations. '
                       'With a time column (years, numbers or dates), the corpus column is optional: usages can be grouped into periods instead.')
            delimiter='\t' if upload is not None and upload.name.endswith('.tsv') else ','
            text=decode_text(upload.getvalue())[0] if upload is not None else ''
            columns=table_columns(text,delimiter) if upload is not None else []
            time_col=None
            periods=None
            period_method='width'
            if upload is not None:
                candidates=[c for c in columns if c not in {'sentence','start','end','sentence_id','matched_form'}]
                guess=next((c for c in candidates if c.casefold() in {'date','year','time','published','publication_date'}),None)
                chosen=st.selectbox('Time column (optional)',['None']+candidates,index=(['None']+candidates).index(guess) if guess else 0,
                                    key=f'time_col_{upload.name}',format_func=lambda c:'No time column' if c=='None' else c,
                                    help='A column of years (1999), other numbers, or dates (2020-05-10, 2020-05-10T02:14:27Z). '
                                         'Each usage keeps its time, which adds the Over time view and colouring by time. '
                                         'Dates written as 10/05/2020 are not read, because day and month order is ambiguous.')
                time_col=None if chosen=='None' else chosen
                if time_col:
                    by_period='corpus' not in columns
                    if not by_period:
                        by_period=st.radio('Corpora',['From the corpus column','Periods formed from the time column'],horizontal=True,
                                           key=f'time_corpora_{upload.name}')=='Periods formed from the time column'
                    else:
                        st.caption('There is no corpus column, so usages are grouped into periods by their time.')
                    if by_period:
                        c1,c2=st.columns(2)
                        periods=int(c1.number_input('Number of periods',min_value=2,max_value=10,value=5,step=1,key='time_periods',
                                                    help='Periods are the corpora that scores, heatmaps and frequencies compare. Up to ten.'))
                        period_method=c2.selectbox('Periods of',['width','count'],key='time_period_method',
                                                   format_func={'width':'Equal length','count':'About equal numbers of usages'}.get,
                                                   help='Equal length: each period spans the same number of years. Equal numbers: each period holds about the same '
                                                        'number of usages, which keeps sample sizes even but makes periods of different lengths. '
                                                        'Dates are grouped by calendar year.')
            if confirm_replacing(st.button('Load usages',type='primary',disabled=upload is None),'load_table',word=word):
                try:
                    data=load_from_csv(io.StringIO(text),word,delimiter=delimiter,forms=forms,match_case=match_case,
                                       time_col=time_col,periods=periods,period_method=period_method)
                    data,_=_cap(data,_parse_cap(cap_text))
                    if time_col:
                        undated=sum(t is None for t in data.times)
                        if undated:
                            _notify('warning',f'{undated:,} usages have no readable time in “{time_col}”. They are left out of the Over time view'
                                    +(' and grouped as “No time”.' if periods else '.'))
                        if periods:
                            st.session_state.chronological=True
                    _set_loaded(data)
                    st.session_state.source_texts={}
                    st.session_state.saved_cache_path=None
                    st.session_state.pop('_corpus_stats',None)
                    _finish_load()
                except Exception as exc:
                    st.error(str(exc))
        elif mode=='Pre-computed embeddings':
            vectors=st.file_uploader('Embeddings',type=['pt','npy'])
            metadata=st.file_uploader('Matching sentence metadata',type=['csv','tsv'])
            st.caption('One vector per metadata row. Include start/end when a sentence contains multiple matching usages.')
            model_id=st.text_input('Original model ID (needed for definitions)',key='import_model')
            col1,col2=st.columns(2)
            layer=int(col1.number_input('Saved layer (-1 = final output)',value=-1,step=1,key='import_layer',
                                        help='-1 is the final output, -2 the penultimate layer, and so on. 0 is the input embedding layer.'))
            representation=col2.selectbox('Saved representation',['word','sentence'],key='import_repr')
            revision=st.text_input('Model revision (optional)',key='import_revision')
            if confirm_replacing(st.button('Load embeddings',type='primary',disabled=vectors is None or metadata is None),'load_precomputed',word=word):
                try:
                    if vectors.name.endswith('.npy'):
                        embs=torch.from_numpy(np.load(io.BytesIO(vectors.getvalue()),allow_pickle=False))
                    else:
                        embs=torch.load(io.BytesIO(vectors.getvalue()),weights_only=True,map_location='cpu')
                    data=load_from_csv(io.StringIO(decode_text(metadata.getvalue())[0]),word,
                                       delimiter='\t' if metadata.name.endswith('.tsv') else ',',forms=forms,all_occurrences=False,match_case=match_case)
                    embs=validate_embeddings(embs,len(data))
                    data,indices=_cap(data,_parse_cap(cap_text))
                    config={'model_id':model_id.strip(),'revision':revision.strip() or None,'layer':layer,
                            'representation':representation,'pooling':'mean','max_length':256}
                    _set_loaded(data,embedded_usages_from_precomputed(data,embs[indices],metadata=config),'precomputed')
                    st.session_state.source_texts={}
                    st.session_state.saved_cache_path=None
                    st.session_state.pop('_corpus_stats',None)
                    _finish_load()
                except Exception as exc:
                    st.error(str(exc))
        elif mode=='Saved corpora':
            caches=list_corpus_caches(STORAGE/'corpora')
            with st.expander('Open a corpus cache from another location'):
                external=st.text_input('Cache directory',help='The directory containing index.sqlite and vectors.npy.')
            selected=None
            if caches:
                selected=st.selectbox('Available corpora',caches,format_func=lambda c:f"{c['name']} · {c['sentences']:,} sentences · {c['model']['model_id']} · layer {c['model']['layer']}")
            elif not external:
                st.info('Prepare reusable embeddings from pasted or uploaded corpora first.')
            path=external.strip() or (selected['path'] if selected else None)
            representation=st.radio('Representation',['word','sentence'],horizontal=True,key='cache_repr',
                                    help='Word representations require a token cache. Sentence representations average the stored token vectors. '
                                         'Saved corpora are encoded once for every word, so XL-LEXEME target markers cannot be applied to them; '
                                         'embed loaded usages directly when you want marked targets.')
            if confirm_replacing(st.button('Load word from saved corpora',type='primary',disabled=path is None),'load_cache',word=word):
                try:
                    cache=CorpusCache(path)
                    data,embedded=cache.select(word,forms,representation=representation,max_per_corpus=_parse_cap(cap_text),match_case=match_case)
                    _set_loaded(data,embedded,'saved')
                    st.session_state.source_texts=cache.source_texts()
                    st.session_state.source_lines=cache.info['splitter']=='lines_v1'
                    st.session_state.pop('_corpus_stats',None)
                    st.session_state['_source_kind']=mode
                    st.session_state.saved_cache_path=str(path)
                    _finish_load()
                except Exception as exc:
                    st.error(str(exc))

        _loaded_summary()
        if mode not in {'Saved corpora','Pre-computed embeddings'}:
            st.divider()
            _embedding_controls(corpus_input if mode in {'Paste sentences','Upload raw corpora (.txt)'} else None)

    data=st.session_state.target_word_data
    if data is not None and mode in {EXAMPLE_SOURCE,'Saved analysis','Saved word embeddings'}:
        # These sources have no form box: ask about the loaded forms instead.
        render_words_question(data.forms,data,'loaded')
    if data is not None:
        with st.expander('Word forms and concordance of the loaded usages',key='review_occurrences',on_change='rerun') as panel:
            if panel.open:
                _review_occurrences(data)
        if advanced():
            with st.expander('Corpus order'):
                _corpus_order_form(data)
    if (advanced() and mode in {'Saved corpora','Saved word embeddings','Saved analysis','Pre-computed embeddings'}
            and st.session_state.embedded_usages is not None):
        with st.expander('Save word embeddings'):
            _save_word_embeddings()


def _corpus_order_form(data):
    """The order of the corpora in plots, tables and the time views."""
    with st.form(f'order_form_{st.session_state.data_revision}',border=False):
        chronology=st.checkbox('Corpora have chronological order',value=st.session_state.chronological,
                               help='Adds the Change over time scores and reads each pair of corpora as earlier and later.')
        order=st.multiselect('Corpus order',data.corpus_labels,default=st.session_state.corpus_order,
                             help='Remove and re-add names to reorder them. Keep every corpus selected.')
        apply=st.form_submit_button('Apply corpus order')
        by_name=st.form_submit_button('Order by name',help='Sorts the corpora by name, with numbers in numeric order, so that names beginning with a year fall into time order.')
        if by_name:
            st.session_state.corpus_order=sorted(data.corpus_labels,key=natural_sort_key)
            st.session_state.chronological=chronology
            st.rerun()
        if apply:
            if len(order)==len(data.corpus_labels):
                st.session_state.corpus_order=order
                st.session_state.chronological=chronology
                st.rerun()
            else:
                st.error('Include all corpora in the order.')


def _loaded_summary():
    """Just under the load button: what the load did, and how many usages each corpus has."""
    show_load_notices()
    data=st.session_state.target_word_data
    if data is None:
        return
    order=ordered_labels(data)
    stats=st.session_state.get('_corpus_stats') or {}
    rows=[]
    for corpus in order:
        row={'Corpus':corpus,'Usages':len(data.indices_for_corpus(corpus))}
        if corpus in stats and stats[corpus]['words']:
            found,words=stats[corpus]['found'],stats[corpus]['words']
            row.update({'Found in the corpus':found,'Corpus size (words)':words,'Per million words':round(found/words*1e6,1)})
        rows.append(row)
    forms=f' and its forms ({", ".join(data.forms[1:])})' if len(data.forms)>1 else ''
    st.markdown(f'**Loaded: {len(data):,} usages of “{data.word}”{forms}**')
    readable_table(pd.DataFrame(rows))
    if stats:
        st.caption('Per million words is the number found divided by the size of the corpus, so corpora of different sizes can be compared. '
                   'Usages is smaller than the number found when a maximum per corpus is set (Usage selection options).')


def _review_occurrences(data):
    """Word forms and a concordance of what was loaded."""
    order=ordered_labels(data)
    forms=pd.crosstab(pd.Series([u.matched_form.casefold() for u in data.usages],name='Form'),pd.Series(data.corpus_list,name='Corpus'))
    forms=forms.reindex(columns=order,fill_value=0)
    if len(forms)>1:
        forms['Total']=forms.sum(axis=1)
        st.write('Word forms')
        readable_table(forms.sort_values('Total',ascending=False).reset_index())
    st.write('Concordance')
    cols=st.columns([2,2,2])
    corpora=cols[0].multiselect('Corpora',order,default=order,key=f'kwic_corpora_{st.session_state.data_revision}')
    query=cols[1].text_input('Lines containing',key='kwic_search',placeholder='For example: burst, hous*',
                             help='Whole words or phrases, ignoring case. Separate alternatives with commas; end a term with * to match any ending.').strip()
    sorting=cols[2].selectbox('Sort lines by',['Order in the corpus','Word after the target','Word before the target'],key='kwic_sort',
                              help='Sorting by a neighbouring word groups recurring phrases, such as “bubble burst” or “housing bubble”.')
    pattern=usage_search_pattern(query) if query else None
    chosen=[i for i,u in enumerate(data.usages) if u.corpus in corpora and (pattern is None or pattern.search(u.sentence))]
    lines=[(i,)+keyword_in_context(data.usages[i].sentence,data.usages[i].char_span) for i in chosen]
    if sorting=='Word after the target':
        lines.sort(key=lambda line:[w.casefold() for w in re.findall(r'\w+',line[3])[:3]])
    elif sorting=='Word before the target':
        lines.sort(key=lambda line:[w.casefold() for w in reversed(re.findall(r'\w+',line[1])[-3:])])
    if not lines:
        st.caption('No loaded usage matches.')
    else:
        per_page=100
        page=1
        if len(lines)>per_page:
            page=int(st.number_input('Page',min_value=1,max_value=-(-len(lines)//per_page),value=1,key=f'kwic_page_{len(lines)}_{sorting}'))
        shown=lines[(page-1)*per_page:page*per_page]
        st.html(concordance_html([(data.usages[i].corpus,left,target,right) for i,left,target,right in shown]))
        st.caption(f'{len(lines):,} of {len(data):,} usages'+(f', lines {(page-1)*per_page+1:,}–{(page-1)*per_page+len(shown):,}' if len(lines)>per_page else '')
                   +'. The highlighted word is the exact usage that is represented. Hover a point in any plot to read its whole sentence.')


def _save_word_embeddings():
    data=st.session_state.target_word_data
    embedded=st.session_state.embedded_usages
    if embedded is None:
        st.info('Embed the loaded usages first, choosing Target word or Whole sentence above.')
        return
    meta=embedded.metadata
    representation=meta.get('representation','unspecified')
    st.caption(f'Save exactly the {len(data):,} loaded usages of “{data.word}” and its selected forms. Includes their sentences and character positions. Selection caps apply; other corpus sentences are excluded.')
    st.write(f"Each usage read as the **{'whole sentence' if representation=='sentence' else 'target word'}** · Model: **{meta.get('model_id') or 'unspecified'}** · Layer: **{meta.get('layer','unspecified')}**"
             +(' · Target markers: **on**' if meta.get('target_markers') else ''))
    st.caption('To save another representation or layer, embed the usages with those settings first. Reopen these files through Saved word embeddings.')
    name=st.text_input('Word embedding file name',value=f'{data.word}_{representation}')
    if st.button('Prepare word embedding file'):
        content=project_bytes(data,embedded,settings={'corpus_order':st.session_state.corpus_order,'scope':'word'})
        st.session_state['_prepared_word_embeddings']=(content,st.session_state.data_revision,id(embedded))
    prepared=st.session_state.get('_prepared_word_embeddings')
    if prepared and prepared[1:]==(st.session_state.data_revision,id(embedded)):
        content=prepared[0]
        st.info(f'File size before saving: {format_bytes(len(content))}. Includes the vectors, sentences and metadata.')
        safe=re.sub(r'[^\w.-]+','_',name).strip('.') or 'word_embeddings'
        st.caption('Saving with an existing file name replaces that saved word file.')
        if st.button('Save word embeddings locally',type='primary'):
            save_project(STORAGE/'words'/f'{safe}.semlens',content)
            st.success('Saved. Reopen through Data source → Saved word embeddings, including after restarting SemLens.')
        st.download_button('Download word embeddings',data=content,file_name=f'{safe}.semlens',mime='application/zip')


def _model_cost(action,model_id,texts,vectors,layer,max_length,batch_size,disk_bytes=0.0,vector_bytes=4):
    """The cost of running ``model_id`` over ``texts``, or None when the model cannot be looked up."""
    profile=model_profile(model_id)
    if profile is None:
        return None
    loaded=st.session_state.get('_loaded_model')
    tokenizer=loaded.tokenizer if loaded is not None and loaded.name==model_id else None
    cost=embedding_cost(action,texts=texts,tokenizer=tokenizer,parameters=profile['parameters'],hidden=profile['hidden'],
                        layers=profile['layers'],device=device_kind(),vectors=vectors,batch_size=batch_size,max_length=max_length,
                        tokens_per_second=known_speed(model_id,batch_size,max_length),download_bytes=profile['download_bytes'],
                        disk_bytes=profile['extra_disk_bytes']+disk_bytes,vector_bytes=vector_bytes)
    if profile['download_bytes']:
        cost.details.append(f'the model is downloaded first ({format_bytes(profile["download_bytes"])}), once')
    return cost


def _offer_speed_trial(cost,model_id,texts,layer,max_length,batch_size,key):
    """For a long job with only a rough estimate, offer to time the model for five seconds."""
    if cost is None or not cost.rough_time or cost.seconds is None or cost.seconds<60:
        return
    if not model_is_downloaded(model_id):
        st.caption('The time is a rough guess from the model’s size. It is measured on this computer once the model is downloaded.')
        return
    if st.button('Measure the speed on this computer (about 5 seconds)',key=f'measure_speed_{key}'):
        with st.spinner('Timing the model on a few of your sentences…'):
            measure_speed(get_model(model_id),texts,batch_size,max_length,layer)
        st.rerun()


def _embedding_cost_check(data,model_id,layer,max_length,batch_size):
    """Show what embedding the loaded usages costs; True when it may start."""
    texts=list(dict.fromkeys(data.sentences))
    cost=_model_cost('Embedding the loaded usages',model_id,texts,len(data),layer,max_length,batch_size)
    if cost is None:
        st.caption('The cost could not be estimated: the model could not be looked up. Check its name, or the internet connection for a first download.')
        return True
    allowed=cost_check(cost,'embed_usages',remedy='Set a maximum number of usages per corpus under Usage selection options, '
                       'lower the inference batch size, or choose a smaller model.')
    _offer_speed_trial(cost,model_id,texts,layer,max_length,batch_size,'embed')
    return allowed


def _encoder_settings():
    """The model settings, under Advanced embedding settings: model, layer, context window, batch size and target markers."""
    with st.expander('Advanced embedding settings'):
        preset=st.selectbox('Language model',list(MODEL_PRESETS),key='encoder_preset',help=MODEL_HELP)
        model_id=MODEL_PRESETS[preset]
        if model_id is None:
            typed=st.text_input('Encoder model',key='encoder_model',
                                placeholder='organisation/model, a Hugging Face link, or a folder on this computer')
            st.caption('A model from Hugging Face: its name (such as FacebookAI/xlm-roberta-base) or its page’s link. '
                       'A model from GitHub or elsewhere: download its files into a folder (config.json, the weights and the tokenizer files) and give the folder’s path.')
            model_id=normalise_model_reference(typed)
            if re.search(r'github\.com',model_id,re.I):
                st.warning('A GitHub link cannot be loaded directly: GitHub pages hold code, and the model files usually need downloading separately. '
                           'Download the model’s files into a folder (from the repository or its releases), then give the path of that folder here.')
        note=_model_size_note(model_id)
        if note:
            st.caption(note)
        layer=int(st.number_input('Layer (-1 = final output)',step=1,key='encoder_layer',
                                  help='Negative values count back from the top: -1 is the final output, -2 the penultimate layer, -3 the one before it. '
                                       '0 is the input embedding layer; 1 and higher count transformer layers from the bottom.'))
        max_length=int(st.number_input('Context window (tokens)',min_value=32,max_value=4096,step=32,key='encoder_max_length',
                                       help='How much text the model reads at once, in subword tokens (roughly 0.75 words each). Longer contexts are read in overlapping windows, so nothing is cut off.'))
        batch_size=int(st.number_input('Contexts read together (batch size)',min_value=1,max_value=128,key='encoder_batch_size',
                                       help='How many contexts are processed together. It affects speed and memory only, never the results. Lower it if the computer runs out of memory.'))
        use_markers=st.checkbox('Mark the target word when the model supports it',key='encoder_target_markers',
                                help='XL-LEXEME was trained with the target wrapped in two special tokens, `<t>` and `</t>`. With this on, SemLens adds them '
                                     'around each target before encoding Target word representations, and around the word in each definition. '
                                     'Models without these tokens, and Whole sentence representations, are unaffected. '
                                     'Turn it off to encode the plain sentence, as earlier versions of SemLens did.')
    return model_id,layer,max_length,batch_size,use_markers


def _embedding_controls(corpus_input):
    st.subheader('Embeddings')
    st.caption('The language model reads every loaded sentence and represents each usage as a list of numbers (an embedding). '
               'Usages it treats as similar in meaning get similar numbers; everything in the other workspaces is built on them.')
    representation=st.radio('Represent each usage by',['Target word','Whole sentence'],horizontal=True,key='encoder_representation',
                            help='**Target word** (recommended): the model’s representation of the word itself, as coloured by its sentence. Best for word meaning.\n\n'
                                 '**Whole sentence**: the average over every word of the sentence. It also reflects what the sentence is about, '
                                 'so usages can differ because the topic differs.')
    rep='word' if representation=='Target word' else 'sentence'
    if advanced():
        model_id,layer,max_length,batch_size,use_markers=_encoder_settings()
    else:
        # The remembered settings (the defaults unless changed with Advanced options on).
        state=st.session_state
        model_id=MODEL_PRESETS.get(state.get('encoder_preset'))
        if model_id is None:
            model_id=normalise_model_reference(state.get('encoder_model') or '') or 'pierluigic/xl-lexeme'
        layer,max_length,batch_size=int(state.encoder_layer),int(state.encoder_max_length),int(state.encoder_batch_size)
        use_markers=bool(state.encoder_target_markers)
        name=next((label.split(' · ')[0] for label,preset in MODEL_PRESETS.items() if preset==model_id),model_id)
        st.caption(f'Model: {name}, {"final layer" if layer==-1 else f"layer {layer}"}. To choose another model or layer, '
                   'switch on **Advanced options** in the sidebar.')
    data=st.session_state.target_word_data
    allowed=data is not None and _embedding_cost_check(data,model_id,layer,max_length,batch_size)
    if st.session_state.get('embedded_usages') is not None and st.session_state.get('def_space_all') is not None:
        count=len(st.session_state.definitions_formatted)
        st.caption(f'Embedding again keeps the annotations, senses and the text of the {count} definition{"s" if count!=1 else ""}, '
                   'but the definitions map is cleared: create it again in Definitions afterwards.')
    if data is not None and not model_is_downloaded(model_id):
        size=f' {_model_size_note(model_id)}' if model_id in MODEL_SIZES else ''
        st.caption(f'Embedding downloads the model first, once; later runs use the downloaded copy.{size}')
    if st.button('Embed loaded usages',type='primary',disabled=not allowed):
        try:
            progress=st.progress(0.0)
            with st.spinner('Loading the encoder and embedding usages…'):
                model=get_model(model_id)
                started=time.perf_counter()
                embedded=embed_usages(model,st.session_state.target_word_data,representation=rep,layer=layer,max_length=max_length,
                                     batch_size=batch_size,progress_callback=lambda b,n:progress.progress(b/n),
                                     target_markers=None if use_markers else False)
                # The run itself measures the speed, for the next estimate.
                elapsed=time.perf_counter()-started
                if elapsed>2:
                    remember_speed(model.name,count_tokens(list(dict.fromkeys(data.sentences)),model.tokenizer)/elapsed,batch_size,max_length)
            reset_downstream_of('embeddings')
            st.session_state.embedded_usages=embedded
            st.session_state.model_name=model.name
            st.session_state.embeddings_source='model'
            progress.empty()
            marked=' with the target marked by <t>…</t>' if embedded.metadata.get('target_markers') else ''
            st.success(f'Created {embedded.dim}-dimensional {rep} embeddings{marked}. Next: explore the usages in Explore.')
        except Exception as exc:
            st.error(str(exc))
    with st.expander('Prepare reusable corpus embeddings'):
        scope=st.radio('Embedding save scope',['Whole corpus','Current word only'],horizontal=True,
                       help='Whole corpus allows changing the target later. Current word saves only the loaded usages and the representation already computed.')
        if scope=='Current word only':
            _save_word_embeddings()
        elif corpus_input is None:
            st.info('Paste or upload complete text corpora to prepare a whole-corpus collection. Current word only can save the loaded usage embeddings.')
        else:
            names,texts,split=corpus_input
            lines=split=='lines'
            st.write('Save the complete corpora once, then reopen them and switch words in a later session. Source text and sentences are included.')
            name=st.text_input('Saved corpus collection name',value='My corpora')
            storage_choice=st.selectbox('What to save',['Word and sentence representations','Sentence representations only'])
            storage='tokens' if storage_choice.startswith('Word') else 'sentences'
            dtype=(st.selectbox('Storage precision',['float32','float16'],help='float16 halves vector storage but rounds values. float32 preserves the computed precision.')
                   if advanced() else 'float32')
            signature=hashlib.sha256(json.dumps([texts,split,model_id,layer,max_length,storage,dtype],sort_keys=True).encode()).hexdigest()
            if st.button('Estimate storage'):
                try:
                    validate_corpus_names(names)
                    records=corpus_sentences(texts,split=split)
                    from transformers import AutoConfig,AutoTokenizer
                    config=AutoConfig.from_pretrained(model_id)
                    tokenizer=AutoTokenizer.from_pretrained(model_id,use_fast=True)
                    dimension=getattr(config,'hidden_size',getattr(config,'d_model',0))
                    estimate=estimate_storage(records,tokenizer,dimension,dtype=dtype,storage=storage,raw_text_bytes=sum(len(t.encode()) for t in texts.values()))
                    st.session_state['_storage_estimate']=(signature,estimate)
                except Exception as exc:
                    st.error(str(exc))
            current=st.session_state.get('_storage_estimate')
            if current and current[0]==signature:
                estimate=current[1]
                st.info(f'Estimated disk space: {format_bytes(estimate.estimated_bytes)} for {estimate.sentences:,} sentences and {estimate.tokens:,} tokens. Includes source text and an allowance for the index; final size may differ.')
                records=corpus_sentences(texts,split=split)
                cost=_model_cost('Saving corpus embeddings',model_id,[r.sentence for r in records],0,layer,max_length,batch_size,
                                 disk_bytes=estimate.estimated_bytes)
                allowed=True
                if cost is not None:
                    # Vectors are written to disk as they are made, so memory holds the model and one batch.
                    allowed=cost_check(cost,'save_corpora',remedy='Save sentence representations only, use float16, or save fewer or smaller corpora.')
                    _offer_speed_trial(cost,model_id,[r.sentence for r in records][:200],layer,max_length,batch_size,'save_corpora')
                if st.button('Compute & save corpus embeddings',type='primary',disabled=not allowed):
                    try:
                        validate_corpus_names(names)
                        progress=st.progress(0.0)
                        with st.spinner('Preparing reusable corpus embeddings…'):
                            model=get_model(model_id)
                            path=create_corpus_cache(STORAGE/'corpora',name,texts,model,storage=storage,dtype=dtype,layer=layer,
                                max_length=max_length,batch_size=batch_size,split=split,estimate=estimate,
                                progress_callback=lambda b,n:progress.progress(b/n))
                        progress.empty()
                        st.success(f'Saved {name}. Choose Saved corpora above to load any word, including after restarting SemLens.')
                        st.caption(str(path))
                    except Exception as exc:
                        st.error(str(exc))
            else:
                st.caption('Estimate storage with the current settings before computing and saving.')


def render_save_analysis():
    data=st.session_state.target_word_data
    embedded=st.session_state.embedded_usages
    if data is None or embedded is None:
        return
    with st.expander('Save or export this analysis',key='save_analysis_panel',on_change='rerun'):
        st.caption('Includes the current usages and vectors, definitions, annotations and available source text. Whole-corpus token caches stay in Saved corpora.')
        if '_analysis_save_name' not in st.session_state:
            st.session_state['_analysis_save_name']=data.word
        name=st.text_input('Analysis name',key='_analysis_save_name')
        settings={k:st.session_state[k] for k in SAVED_SETTINGS}
        if st.button('Prepare analysis file'):
            content=_analysis_bytes()
            st.session_state['_prepared_analysis']=(content,st.session_state.data_revision,
                json.dumps([get_annotations(),get_sense_classes(),get_options(),settings,st.session_state.definitions_formatted],sort_keys=True))
        prepared=st.session_state.get('_prepared_analysis')
        stamp=json.dumps([get_annotations(),get_sense_classes(),get_options(),settings,st.session_state.definitions_formatted],sort_keys=True)
        if prepared and prepared[1:]==(st.session_state.data_revision,stamp):
            content=prepared[0]
            try:
                filename=project_filename(name)
            except ValueError as exc:
                st.error(str(exc))
                return
            path=STORAGE/'projects'/filename
            st.caption(f'File size: {format_bytes(len(content))}')
            replace=True
            if path.exists():
                if st.session_state.get('_active_analysis_path')==str(path):
                    st.caption(f'Saving updates “{path.stem}”. Change the name to save a separate copy.')
                else:
                    replace=st.checkbox(f'Replace existing saved analysis “{path.stem}”',key=f'replace_analysis_{filename}')
            if st.button('Save analysis locally',disabled=not replace):
                try:
                    save_project(path,content)
                    st.session_state['_active_analysis_path']=str(path)
                    mark_saved()
                    st.success('Saved. Reopen through Data source → Saved analysis.')
                except OSError as exc:
                    st.error(str(exc))
            st.download_button('Download analysis',data=content,file_name=filename,mime='application/zip',on_click=mark_saved)
