"""Where SemLens keeps files. No Streamlit import, so the launcher can use it."""
from __future__ import annotations
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def storage_folder(root=ROOT,environ=None,home=None):
    """Where saved corpora, embeddings and analyses are kept.

    ``SEMLENS_HOME`` wins when set. A checkout of the repository keeps its own
    ``.semlens`` folder, as before. An installed copy (for example through
    ``uvx``) lives inside a package directory that is not the user's to write
    to and may be discarded, so it uses ``.semlens`` in the home folder.
    """
    environ=os.environ if environ is None else environ
    if environ.get('SEMLENS_HOME'):
        return Path(environ['SEMLENS_HOME']).expanduser()
    if (Path(root)/'pyproject.toml').is_file():
        return Path(root)/'.semlens'
    return (Path.home() if home is None else Path(home))/'.semlens'


STORAGE=storage_folder()
