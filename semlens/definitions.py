"""Definition sourcing, formatting, and embedding.

Supports three definition sources:

1. **Manual input** — user provides a list of definition strings.
2. **Wiktionary** — fetched via the MediaWiki API (multilingual).
3. **JSON file** — load from a pre-existing definitions file.

Definitions are formatted as ``"word: definition text"`` before embedding,
matching the convention from Goworek & Dubossarsky (2026).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .utils import find_word_position


# ---------------------------------------------------------------------------
# Definition formatting
# ---------------------------------------------------------------------------

def format_definitions(word: str, raw_definitions: list[str]) -> list[str]:
    """Prepend ``'word: '`` to each definition for contextualised embedding.

    Strips whitespace and deduplicates.  Skips if already formatted.
    """
    formatted = []
    seen = set()
    prefix = f"{word.lower()}:"

    for d in raw_definitions:
        d = d.strip()
        if not d:
            continue
        if d.lower().startswith(prefix):
            key = d.lower()
            text = d
        else:
            key = f"{word}: {d}".lower()
            text = f"{word}: {d}"
        if key not in seen:
            seen.add(key)
            formatted.append(text)

    return formatted


# ---------------------------------------------------------------------------
# Wiktionary fetching (MediaWiki API)
# ---------------------------------------------------------------------------

WIKTIONARY_API_URL = "https://{lang}.wiktionary.org/w/api.php"


def fetch_wiktionary_definitions(
    word: str,
    language: str = "en",
    target_language_name: str | None = None,
) -> list[str]:
    """Fetch dictionary definitions from Wiktionary via the MediaWiki API.

    Parameters
    ----------
    word : str
        The word to look up.
    language : str
        Wiktionary edition language code (``"en"``, ``"de"``, ``"sv"``, etc.).
    target_language_name : str or None
        For non-English Wiktionary, the section heading of the target language
        (e.g. ``"English"`` on en.wiktionary).  If None, defaults to
        ``"English"`` for the English edition, or attempts to find the first
        language section otherwise.

    Returns
    -------
    list[str]
        Raw definition strings (without the ``word:`` prefix — call
        ``format_definitions`` afterwards).
    """
    import requests

    url = WIKTIONARY_API_URL.format(lang=language)
    params = {
        "action": "parse",
        "page": word,
        "prop": "wikitext",
        "format": "json",
    }
    headers = {
        "User-Agent": "SemLens/0.1 (https://github.com/semlens; academic research tool)",
    }

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise ValueError(
            f"Wiktionary lookup failed for '{word}': {data['error'].get('info', 'unknown error')}"
        )

    wikitext = data["parse"]["wikitext"]["*"]
    return _parse_wikitext_definitions(wikitext, target_language_name or "English")


def _parse_wikitext_definitions(wikitext: str, target_language: str) -> list[str]:
    """Extract definitions from Wiktionary wikitext.

    Parses numbered list items (lines starting with ``#``) under
    part-of-speech headings within the target language section.
    """
    lines = wikitext.split("\n")
    definitions: list[str] = []

    # State machine: find the target language section
    in_target_lang = False
    in_pos_section = False

    pos_headings = {
        "Noun", "Verb", "Adjective", "Adverb", "Preposition",
        "Conjunction", "Pronoun", "Determiner", "Interjection",
        "Numeral", "Particle", "Proper noun", "Proper Noun",
        # German/Romance
        "Substantiv", "Adjektiv", "Nombre", "Verbe", "Adjectif",
    }

    for line in lines:
        stripped = line.strip()

        # Level-2 heading: language section (e.g. ==English==)
        if re.match(r"^==[^=].*[^=]==$", stripped):
            lang_name = stripped.strip("= ").strip()
            in_target_lang = lang_name == target_language
            in_pos_section = False
            continue

        # Level-3 or level-4 heading: POS section
        if in_target_lang and re.match(r"^===+[^=].*[^=]===+$", stripped):
            heading = stripped.strip("= ").strip()
            in_pos_section = heading in pos_headings
            continue

        # Another language section starts — stop
        if in_target_lang and re.match(r"^==[^=]", stripped) and not stripped.startswith("==="):
            break

        # Definition line: starts with # (but not ## which is a sub-definition)
        if in_target_lang and in_pos_section and re.match(r"^#[^#*:]", stripped):
            defn = _clean_wikitext(stripped.lstrip("# ").strip())
            if defn and len(defn) > 3:
                definitions.append(defn)

    return definitions


def _clean_wikitext(text: str) -> str:
    """Remove wikitext markup from a definition string."""
    # Remove {{template|...|display}} — keep last parameter or display text
    text = re.sub(r"\{\{[^}]*\|([^}|]+)\}\}", r"\1", text)
    # Remove remaining {{templates}}
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove [[link|display]] → display
    text = re.sub(r"\[\[[^]]*\|([^\]]+)\]\]", r"\1", text)
    # Remove [[simple links]] → text
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Remove '' and '''
    text = text.replace("'''", "").replace("''", "")
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# JSON file loading
# ---------------------------------------------------------------------------

def load_definitions_from_json(
    path: str | Path,
    word: str | None = None,
) -> dict[str, list[str]]:
    """Load definitions from a JSON file.

    The file should map ``{word: [def1, def2, ...]}`` (matching the format
    used in the Rethinking_LSCD_Metrics repository).

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.
    word : str or None
        If given, return only definitions for that word.  Otherwise return all.

    Returns
    -------
    dict[str, list[str]]
        Mapping of word → list of raw definition strings.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if word is not None:
        # Try exact match, then without trailing info (e.g. "word_nn")
        key = word if word in data else word.split("_")[0]
        if key not in data:
            raise KeyError(f"Word '{word}' not found in {path}")
        return {word: data[key]}

    return data


# ---------------------------------------------------------------------------
# Definition embedding
# ---------------------------------------------------------------------------

def embed_definitions(
    loaded_model,  # embeddings.LoadedModel
    word: str,
    formatted_definitions: list[str],
    *,
    batch_size: int = 16,
    max_length: int = 128,
) -> Tensor:
    """Embed the target word contextualised within each definition.

    Each definition should already be formatted as ``"word: definition text"``.
    The target word's position in each definition is auto-detected, and the
    contextualised representation of that word is extracted.

    Parameters
    ----------
    loaded_model : LoadedModel
        From ``embeddings.load_model()``.
    word : str
        The target lemma.
    formatted_definitions : list[str]
        Already formatted (e.g. via ``format_definitions``).
    batch_size : int
        Inference batch size.
    max_length : int
        Max token length.

    Returns
    -------
    Tensor
        Shape ``[K, D]`` where K is the number of definitions.
    """
    from .embeddings import extract_word_embeddings

    # Find word position in each formatted definition
    char_spans = []
    for defn in formatted_definitions:
        start, end = find_word_position(defn, word)
        char_spans.append((start, end))

    return extract_word_embeddings(
        loaded_model,
        formatted_definitions,
        char_spans,
        batch_size=batch_size,
        max_length=max_length,
    )
