"""Remember which corpora the user switched off in a plot legend.

Streamlit redraws a Plotly chart from scratch whenever its figure changes, which
would otherwise bring every corpus back. A small script reports legend toggles,
and the plots are then built with those corpora already switched off.
"""
from __future__ import annotations
from pathlib import Path
import streamlit as st

def _register():
    return st.components.v2.component(
        'semlens_corpus_visibility',
        js=(Path(__file__).parent / 'corpus_visibility.js').read_text(),
    )


_bridge = _register()


def _mount(**kwargs):
    """Components are registered per Streamlit runtime; register again if this one is new."""
    global _bridge
    try:
        return _bridge(**kwargs)
    except ValueError:
        _bridge = _register()
        return _bridge(**kwargs)


def hidden_corpora(labels) -> list[str]:
    """Corpora currently switched off, restricted to those that are loaded."""
    present = set(labels)
    return [c for c in st.session_state.get('hidden_corpora', []) if c in present]


def sync_corpus_visibility():
    """Mount once per run, before any plot, and apply a toggle made in a legend."""
    data = st.session_state.get('target_word_data')
    hidden = hidden_corpora(data.corpus_labels) if data is not None else []
    result = _mount(key='semlens_corpus_visibility', data={'hidden': hidden}, on_toggled_change=lambda: None)
    toggled = getattr(result, 'toggled', None)
    if toggled is not None:
        st.session_state['hidden_corpora'] = [str(c) for c in toggled]
