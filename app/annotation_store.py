"""Annotation state scoped to the browser session, with explicit snapshot support."""
from __future__ import annotations
from copy import deepcopy
import streamlit as st


def _store():
    if '_annotation_store' not in st.session_state:
        st.session_state['_annotation_store']={'annotations':{},'sense_classes':[],'active_sense_class':'','history':[]}
    return st.session_state['_annotation_store']


def get_annotations():
    return _store()['annotations']


def get_sense_classes():
    return _store()['sense_classes']


def get_active_sense_class():
    return _store()['active_sense_class']


def set_active_sense_class(name):
    _store()['active_sense_class']=name


def remember_annotations():
    history=_store()['history']
    history.append(deepcopy(get_annotations()))
    del history[:-30]


def undo_annotations():
    if _store()['history']:
        _store()['annotations']=_store()['history'].pop()


def set_annotation(idx,label):
    get_annotations()[int(idx)]=label


def remove_annotation(idx):
    get_annotations().pop(int(idx),None)


def clear_annotations():
    remember_annotations()
    get_annotations().clear()


def add_sense_class(name):
    if name not in get_sense_classes():
        get_sense_classes().append(name)


def remove_sense_class(name):
    if name in get_sense_classes():
        get_sense_classes().remove(name)
    for idx in [i for i,label in get_annotations().items() if label==name]:
        remove_annotation(idx)
    if get_active_sense_class()==name:
        set_active_sense_class(next(iter(get_sense_classes()),''))
    _store()['history']=[]


def restore_annotations(annotations, sense_classes=None):
    _store()['annotations']={int(k):v for k,v in annotations.items()}
    if sense_classes is not None:
        _store()['sense_classes']=list(sense_classes)
    _store()['history']=[]
    set_active_sense_class(next(iter(get_sense_classes()),''))


def reset_annotation_state():
    st.session_state['_annotation_store']={'annotations':{},'sense_classes':[],'active_sense_class':'','history':[]}
