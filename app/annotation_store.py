"""Per-session annotation store decoupled from Streamlit session_state keys.

The data is keyed by a generated session identifier so each browser session
keeps isolated annotation data, while the app code avoids using
``st.session_state.annotations`` directly.
"""

from __future__ import annotations

from uuid import uuid4

import streamlit as st


_SESSION_ID_KEY = "_annotation_store_session_id"
_STORE_BY_SESSION: dict[str, dict[str, object]] = {}


def _default_store() -> dict[str, object]:
    return {
        "annotations": {},
        "sense_classes": [],
        "active_sense_class": "",
    }


def _session_id() -> str:
    sid = st.session_state.get(_SESSION_ID_KEY)
    if not sid:
        sid = str(uuid4())
        st.session_state[_SESSION_ID_KEY] = sid
    return sid


def _store() -> dict[str, object]:
    sid = _session_id()
    if sid not in _STORE_BY_SESSION:
        _STORE_BY_SESSION[sid] = _default_store()
    return _STORE_BY_SESSION[sid]


def get_annotations() -> dict[int, str]:
    return _store()["annotations"]  # type: ignore[return-value]


def set_annotation(idx: int, label: str):
    get_annotations()[idx] = label


def remove_annotation(idx: int):
    get_annotations().pop(idx, None)


def clear_annotations():
    get_annotations().clear()


def get_sense_classes() -> list[str]:
    return _store()["sense_classes"]  # type: ignore[return-value]


def add_sense_class(name: str):
    sense_classes = get_sense_classes()
    if name not in sense_classes:
        sense_classes.append(name)


def remove_sense_class(name: str):
    sense_classes = get_sense_classes()
    if name in sense_classes:
        sense_classes.remove(name)

    annotations = get_annotations()
    for idx in [k for k, v in annotations.items() if v == name]:
        annotations.pop(idx, None)

    if get_active_sense_class() == name:
        set_active_sense_class(sense_classes[0] if sense_classes else "")


def get_active_sense_class() -> str:
    return _store()["active_sense_class"]  # type: ignore[return-value]


def set_active_sense_class(name: str):
    _store()["active_sense_class"] = name


def reset_annotation_state():
    store = _store()
    store["annotations"] = {}
    store["sense_classes"] = []
    store["active_sense_class"] = ""
