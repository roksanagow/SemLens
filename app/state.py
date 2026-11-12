"""Session state and dependency invalidation for the interactive application."""
from __future__ import annotations
from copy import deepcopy
import streamlit as st
from app.annotation_store import reset_annotation_state, get_annotations, restore_annotations

_DEFAULTS = {
    'target_word':'', 'target_word_data':None, 'embedded_usages':None,
    'model_name':'', 'corpus_labels':[], 'corpus_order':[], 'source_texts':{}, 'source_lines':False,
    'embeddings_source':'model', 'definitions_raw':[], 'definitions_formatted':[],
    'definition_embeddings':None, 'def_space_all':None, 'definition_settings':{},
    'palette_name':'Tab10 (matplotlib)', 'distance_metric':'cosine',
    'sampling_mode':'balanced', 'sampling_repetitions':20, 'sampling_seed':42,
    'data_revision':0, 'chronological':False, 'saved_cache_path':None,
    'theme_mode':'Dark', 'hidden_corpora':[], 'source_split':'auto',
    'point_size':9, 'point_opacity':0.8, 'figure_format':'png',
}


def init_state():
    for key,value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key]=deepcopy(value)


def reset_downstream_of(stage: str):
    keys = {
        'data':['embedded_usages','definitions_raw','definitions_formatted','definition_embeddings','def_space_all','definition_settings'],
        'embeddings':['definition_embeddings','def_space_all','definition_settings'],
        'definitions':['definition_embeddings','def_space_all'],
    }
    for key in keys.get(stage,[]):
        st.session_state[key]=deepcopy(_DEFAULTS[key])
    if stage in {'data','embeddings','definitions'}:
        st.session_state.data_revision += 1
        st.session_state.pop('_annot_coords_cache',None)
        # Remove old selection payloads so a new dataset cannot inherit a click.
        for key in list(st.session_state):
            if key.startswith(('annot_chart_', '_last_selection_')):
                del st.session_state[key]
    if stage=='annotation':
        reset_annotation_state()


def set_data(data, embedded=None, *, source='model', preserve_annotations=True):
    old=st.session_state.target_word_data
    old_annotations = {old.usages[i].usage_id:label for i,label in get_annotations().items()
                       if old is not None and 0<=i<len(old)} if preserve_annotations else {}
    previous_order=st.session_state.corpus_order
    same_word=old is not None and old.word==data.word
    reset_downstream_of('data')
    st.session_state.pop('_active_analysis_path',None)
    st.session_state.pop('_prepared_analysis',None)
    st.session_state['_analysis_save_name']=data.word
    if not same_word:
        reset_annotation_state()
    st.session_state.target_word_data=data
    st.session_state.target_word=data.word
    st.session_state.corpus_labels=data.corpus_labels
    st.session_state.corpus_order=list(dict.fromkeys([c for c in previous_order if c in data.corpus_labels]+data.corpus_labels))
    st.session_state.embedded_usages=embedded
    st.session_state.embeddings_source=source
    if embedded is not None:
        st.session_state.model_name=embedded.metadata.get('model_id','')
    restore_annotations({i:old_annotations[u.usage_id] for i,u in enumerate(data.usages) if u.usage_id in old_annotations})
