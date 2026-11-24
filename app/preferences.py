"""Settings remembered between sessions, in ``settings.json`` in SemLens's storage folder.

Only preferences are kept: display, comparison, model and Discover settings.
Anything tied to particular data (words, corpora, seeds, annotations) belongs
to a saved analysis or discovery instead.

Every run re-saves these keys in the session, so their values also survive
moving between workspaces (Streamlit forgets the value of a widget that is not
drawn); their widgets therefore take their starting values from here, never
from ``value=`` or ``index=``. The file is written only when a value changes.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import streamlit as st
from app.paths import STORAGE
from app.ui_data import MODEL_PRESETS
from app.discover import MODELS as DISCOVER_MODELS

PREFERENCES_FILE=STORAGE/'settings.json'


def _one_of(*options):
    return lambda value:value in options


def _number(low,high,kind=int):
    def check(value):
        if isinstance(value,bool) or not isinstance(value,(int,float)):
            return False
        if kind is int and int(value)!=value:
            return False
        return (low is None or value>=low) and (high is None or value<=high)
    return check


def _text(value):
    return isinstance(value,str) and len(value)<2000


def _flag(value):
    return isinstance(value,bool)


def _palette(value):
    from semlens.utils import palette_names
    return value in palette_names()


def _first(options):
    return next(iter(options))


# key: (default, check)
SAVED={
    # Display and comparison (sidebar)
    'theme_mode':('Light',_one_of('Light','Dark')),
    'palette_name':('Tab10 (matplotlib)',_palette),
    'distance_metric':('cosine',_one_of('cosine','euclidean')),
    'sampling_mode':('balanced',_one_of('balanced','unequal','repeated')),
    'sampling_repetitions':(20,_number(2,100)),
    'sampling_seed':(42,_number(0,None)),
    'point_size':(9,_number(4,18)),
    'point_opacity':(0.8,_number(0.2,1.0,float)),
    'figure_format':('png',_one_of('png','svg')),
    # Embedding (Data & Model)
    'encoder_representation':('Target word',_one_of('Target word','Whole sentence')),
    'encoder_preset':(_first(MODEL_PRESETS),_one_of(*MODEL_PRESETS)),
    'encoder_model':('pierluigic/xl-lexeme',_text),
    'encoder_layer':(-1,_number(-200,200)),
    'encoder_max_length':(256,_number(32,4096)),
    'encoder_batch_size':(16,_number(1,128)),
    'encoder_target_markers':(True,_flag),
    # Discover
    'dv_cap':('100',_text),
    'dv_min_count':(5,_number(1,None)),
    'dv_min_length':(2,_number(1,10)),
    'dv_stop':(True,_flag),
    'dv_extra_stop':('',_text),
    'dv_model_preset':(_first(DISCOVER_MODELS),_one_of(*DISCOVER_MODELS)),
    'dv_model_other':('FacebookAI/xlm-roberta-base',_text),
    'dv_representation':('word',_one_of('word','slot')),
    'dv_max_length':(256,_number(32,1024)),
    'dv_batch':(16,_number(1,128)),
    'dv_k':(10,_number(1,100)),
    'dv_background':(False,_flag),
    'dv_rank_by':('knn',_one_of('knn','near')),
    'dv_min_sampled':(5,_number(1,None)),
    'dv_hide_seeds':(True,_flag),
    'dv_hide_rejected':(True,_flag),
    'dv_hide_spelling':(False,_flag),
    'dv_top':(50,_number(10,2000)),
    'dv_static_method':('word2vec',_one_of('word2vec','ppmi')),
    'dv_static_dims':(100,_number(10,500)),
    'dv_static_window':(5,_number(1,20)),
    'dv_static_min':(5,_number(1,None)),
    'dv_static_epochs':(10,_number(1,50)),
    'dv_static_workers':(1,_number(1,16)),
    'dv_static_k':(10,_number(1,50)),
    'dv_static_top':(50,_number(10,1000)),
    'dv_static_keep':(200,_number(10,5000)),
}
# Discover's layer is kept per model: dv_layer_<model>.
SAVED_PREFIXES={'dv_layer_':_number(0,200)}


def _check(key):
    if key in SAVED:
        return SAVED[key][1]
    for prefix,check in SAVED_PREFIXES.items():
        if key.startswith(prefix):
            return check
    return None


def _saved_keys():
    return [k for k in st.session_state if _check(k) is not None]


def read_preferences(path=None):
    """The valid settings in the file; unknown keys and unusable values are ignored."""
    path=Path(path or PREFERENCES_FILE)
    try:
        stored=json.loads(path.read_text())
    except (OSError,ValueError):
        return {}
    if not isinstance(stored,dict):
        return {}
    found={}
    for key,value in stored.items():
        check=_check(key)
        try:
            if check is not None and check(value):
                found[key]=value
        except Exception:
            continue
    return found


def apply_preferences():
    """At the start of every run: load the file once per session, keep the values alive, fill in defaults."""
    if not st.session_state.get('_preferences_loaded'):
        st.session_state['_preferences_loaded']=True
        loaded=read_preferences()
        for key,value in loaded.items():
            st.session_state[key]=value
        st.session_state['_preferences_written']=json.dumps(loaded,sort_keys=True)
    for key in _saved_keys():
        st.session_state[key]=st.session_state[key]
    for key,(default,_) in SAVED.items():
        if default is not None and key not in st.session_state:
            st.session_state[key]=default


def save_preferences():
    """Write the settings when they differ from the file. Called at the start and the end of each run."""
    values={}
    for key in _saved_keys():
        value=st.session_state[key]
        # Only changed settings: a default improved in a later version then reaches everyone who kept it.
        if key in SAVED and value==SAVED[key][0]:
            continue
        try:
            if _check(key)(value):
                values[key]=value
        except Exception:
            continue
    text=json.dumps(values,sort_keys=True)
    if text==st.session_state.get('_preferences_written'):
        return
    path=Path(PREFERENCES_FILE)
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(prefix='.settings-',dir=path.parent)
        with os.fdopen(fd,'w') as f:
            f.write(json.dumps(values,sort_keys=True,indent=1))
        os.replace(temp,path)
        st.session_state['_preferences_written']=text
    except OSError:
        # A read-only folder: the settings still last for this session.
        pass


def _restore_defaults():
    for key in _saved_keys():
        if key in SAVED and SAVED[key][0] is not None:
            st.session_state[key]=SAVED[key][0]
        else:
            del st.session_state[key]
    try:
        Path(PREFERENCES_FILE).unlink(missing_ok=True)
    except OSError:
        pass
    st.session_state['_preferences_written']=json.dumps({},sort_keys=True)


def render_preferences_note():
    """A line in the sidebar saying settings are remembered, with a way back to the defaults."""
    with st.expander('Remembered settings'):
        st.caption('Display, comparison, model and Discover settings are remembered on this computer for next time. '
                   'Words, corpora, seeds and annotations are not: save an analysis, or a discovery, to keep those.')
        st.button('Restore the default settings',on_click=_restore_defaults,key='restore_default_settings',
                  help='Every remembered setting goes back to its default, now and next time.')
