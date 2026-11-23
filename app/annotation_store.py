"""Annotation state scoped to the browser session, with explicit snapshot support.

Each usage maps to ``{sense: rating}`` (see :mod:`semlens.annotations`). The
options say whether a usage may have several senses and whether senses are
rated; with both off, a usage has exactly one unrated sense, as before.
"""
from __future__ import annotations
from copy import deepcopy
import streamlit as st
from semlens.annotations import DEFAULT_SCALE, merge_ratings, normalise_annotations

# rating_entry: 'ask' opens a box to type the rating after each click or lasso;
# 'default' gives every new annotation the default rating chosen above the plot.
DEFAULT_OPTIONS = {'multiple': False, 'graded': False, 'scale': DEFAULT_SCALE, 'rating_entry': 'ask'}


def _empty():
    return {'annotations': {}, 'sense_classes': [], 'active_sense_class': '', 'history': [],
            'options': dict(DEFAULT_OPTIONS), 'active_rating': DEFAULT_SCALE}


def _store():
    if '_annotation_store' not in st.session_state:
        st.session_state['_annotation_store'] = _empty()
    store = st.session_state['_annotation_store']
    store.setdefault('options', dict(DEFAULT_OPTIONS))
    for name, value in DEFAULT_OPTIONS.items():
        store['options'].setdefault(name, value)
    store.setdefault('active_rating', store['options']['scale'])
    if not store.get('_normalised'):
        # A store written by an older version holds one sense name per usage.
        store['annotations'] = normalise_annotations(store['annotations'])
        store['_normalised'] = True
    return store


def get_annotations():
    return _store()['annotations']


def get_sense_classes():
    return _store()['sense_classes']


def get_active_sense_class():
    return _store()['active_sense_class']


def set_active_sense_class(name):
    _store()['active_sense_class'] = name


def get_options():
    return _store()['options']


def set_option(name, value):
    _store()['options'][name] = value
    if name == 'scale':
        _store()['active_rating'] = min(get_active_rating(), int(value))


def get_active_rating():
    """The rating given by a click or lasso, within the current scale."""
    return max(1, min(int(_store()['active_rating']), int(get_options()['scale'])))


def set_active_rating(value):
    if value is not None:
        _store()['active_rating'] = int(value)


def remember_annotations():
    history = _store()['history']
    history.append(deepcopy(get_annotations()))
    del history[:-30]


def undo_annotations():
    if _store()['history']:
        _store()['annotations'] = normalise_annotations(_store()['history'].pop())


def set_annotation(idx, sense, rating=None):
    """Give a usage a sense: added to its senses when several are allowed, replacing them otherwise."""
    idx = int(idx)
    if not get_options()['graded']:
        rating = None
    current = get_annotations().get(idx, {})
    if get_options()['multiple']:
        label = dict(current)
        label[sense] = rating
    else:
        label = {sense: rating}
    if label == current:
        return False
    get_annotations()[idx] = label
    return True


def set_label(idx, label):
    """Replace a usage's senses outright; an empty label removes the annotation."""
    idx = int(idx)
    label = {sense: rating for sense, rating in label.items() if sense}
    if label:
        get_annotations()[idx] = label
    else:
        get_annotations().pop(idx, None)


def remove_sense(idx, sense):
    label = get_annotations().get(int(idx))
    if not label or sense not in label:
        return False
    label = {s: r for s, r in label.items() if s != sense}
    set_label(idx, label)
    return True


def remove_annotation(idx):
    get_annotations().pop(int(idx), None)


def clear_annotations():
    remember_annotations()
    get_annotations().clear()


def keep_first_senses():
    """Reduce every usage to its first sense, so several senses can be switched off."""
    remember_annotations()
    for idx, label in list(get_annotations().items()):
        first = next(iter(label))
        get_annotations()[idx] = {first: label[first]}


def add_sense_class(name):
    if name not in get_sense_classes():
        get_sense_classes().append(name)


def remove_sense_class(name):
    if name in get_sense_classes():
        get_sense_classes().remove(name)
    for idx in [i for i, label in get_annotations().items() if name in label]:
        remove_sense(idx, name)
    if get_active_sense_class() == name:
        set_active_sense_class(next(iter(get_sense_classes()), ''))
    _store()['history'] = []


def rename_sense_class(old, new):
    """Rename a class; renaming onto an existing class merges the two."""
    new = new.strip()
    classes = get_sense_classes()
    if not new or old not in classes or new == old:
        return
    remember_annotations()
    if new in classes:
        classes.remove(old)
    else:
        classes[classes.index(old)] = new
    for idx, label in list(get_annotations().items()):
        if old not in label:
            continue
        renamed = {}
        for sense, rating in label.items():
            target = new if sense == old else sense
            renamed[target] = merge_ratings(renamed[target], rating) if target in renamed else rating
        get_annotations()[idx] = renamed
    if get_active_sense_class() == old:
        set_active_sense_class(new)


def restore_annotations(annotations, sense_classes=None, options=None):
    _store()['annotations'] = normalise_annotations(annotations)
    if sense_classes is not None:
        _store()['sense_classes'] = list(sense_classes)
    if options is not None:
        merged = dict(DEFAULT_OPTIONS)
        merged.update({k: options[k] for k in DEFAULT_OPTIONS if k in options})
        merged['scale'] = int(merged['scale'])
        _store()['options'] = merged
        _store()['active_rating'] = merged['scale']
    _store()['history'] = []
    set_active_sense_class(next(iter(get_sense_classes()), ''))


def reset_annotation_state():
    st.session_state['_annotation_store'] = _empty()
