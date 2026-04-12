"""Session state management for the SemLens Streamlit app."""

from __future__ import annotations

import streamlit as st

from app.annotation_store import reset_annotation_state


# Keys and their default values
_DEFAULTS = {
    # Data
    "target_word": "",
    "target_word_data": None,       # TargetWordData
    "corpus_labels": [],

    # Embeddings
    "model_name": "",
    "embedded_usages": None,        # EmbeddedUsages
    "embeddings_source": "model",   # "model" or "precomputed"

    # Definitions
    "definitions_raw": [],          # list[str] — raw user-provided definitions
    "definitions_formatted": [],    # list[str] — formatted as "word: def"
    "definition_embeddings": None,  # Tensor [K, D]
    "def_space_all": None,          # Tensor [N, K] — all usages in def space

    # Display settings
    "palette_name": "Tab10 (matplotlib)",
    "reduction_method": "pca",

}


def init_state():
    """Initialise all session state keys with defaults (only if not set)."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_downstream_of(stage: str):
    """Clear state that depends on a given stage, preserving earlier stages.

    Stages (in dependency order):
        data → embeddings → definitions → def_space → annotation
    """
    stages = {
        "data": [
            "target_word_data", "corpus_labels",
            "embedded_usages", "embeddings_source",
            "definitions_raw", "definitions_formatted",
            "definition_embeddings", "def_space_all",
        ],
        "embeddings": [
            "embedded_usages",
            "definition_embeddings", "def_space_all",
        ],
        "definitions": [
            "definition_embeddings", "def_space_all",
        ],
        "annotation": [],
    }
    for key in stages.get(stage, []):
        st.session_state[key] = _DEFAULTS.get(key)

    if stage in {"data", "embeddings", "annotation"}:
        reset_annotation_state()
