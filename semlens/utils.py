"""Shared utilities: colour palettes, edit distance, text formatting."""

from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Colour palettes (all colourblind-friendly)
# ---------------------------------------------------------------------------

PALETTES: dict[str, list[str]] = {
    # --- Colourblind-friendly ---
    "Wong (colourblind)": [
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#000000",  # black
    ],
    "Tol Bright (colourblind)": [
        "#4477AA",  # blue
        "#EE6677",  # red
        "#228833",  # green
        "#CCBB44",  # yellow
        "#66CCEE",  # cyan
        "#AA3377",  # purple
        "#BBBBBB",  # grey
    ],
    # --- General purpose ---
    "Tab10 (matplotlib)": [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # grey
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ],
    "Set2 (pastel)": [
        "#66c2a5",  # teal
        "#fc8d62",  # salmon
        "#8da0cb",  # lavender
        "#e78ac3",  # pink
        "#a6d854",  # lime
        "#ffd92f",  # yellow
        "#e5c494",  # tan
        "#b3b3b3",  # grey
    ],
    "Dark2 (bold)": [
        "#1b9e77",  # dark teal
        "#d95f02",  # dark orange
        "#7570b3",  # slate purple
        "#e7298a",  # hot pink
        "#66a61e",  # dark green
        "#e6ab02",  # dark yellow
        "#a6761d",  # brown
        "#666666",  # dark grey
    ],
    "Paired": [
        "#a6cee3",  # light blue
        "#1f78b4",  # dark blue
        "#b2df8a",  # light green
        "#33a02c",  # dark green
        "#fb9a99",  # light red
        "#e31a1c",  # dark red
        "#fdbf6f",  # light orange
        "#ff7f00",  # dark orange
        "#cab2d6",  # light purple
        "#6a3d9a",  # dark purple
    ],
    "Nord (cool)": [
        "#5E81AC",  # steel blue
        "#BF616A",  # muted red
        "#A3BE8C",  # sage green
        "#EBCB8B",  # warm yellow
        "#B48EAD",  # muted purple
        "#88C0D0",  # ice blue
        "#D08770",  # peach
        "#4C566A",  # dark slate
    ],
    "Retro (warm)": [
        "#E07A5F",  # terra cotta
        "#3D405B",  # dark blue grey
        "#81B29A",  # sage
        "#F2CC8F",  # sand
        "#5F0F40",  # burgundy
        "#9A8C98",  # mauve grey
        "#F4A261",  # sandy orange
        "#264653",  # dark teal
    ],
    "IBM Design": [
        "#648FFF",  # ultramarine
        "#DC267F",  # magenta
        "#FE6100",  # orange
        "#FFB000",  # gold
        "#785EF0",  # indigo
    ],
    "Petroff 10": [
        "#3f90da",  # blue
        "#ffa90e",  # orange
        "#bd1f01",  # red
        "#94a4a2",  # grey
        "#832db6",  # purple
        "#a96b59",  # brown
        "#e76300",  # dark orange
        "#b9ac70",  # olive
        "#717581",  # dark grey
        "#92dadd",  # light cyan
    ],
}

DEFAULT_PALETTE = "Tab10 (matplotlib)"


def get_palette(name: str | None = None) -> list[str]:
    """Return a named colour palette.  Falls back to Wong if unknown."""
    if name is None:
        name = DEFAULT_PALETTE
    return PALETTES.get(name, PALETTES[DEFAULT_PALETTE])


def palette_names() -> list[str]:
    """Return available palette names."""
    return list(PALETTES.keys())


# ---------------------------------------------------------------------------
# Edit-distance-based word position finding
# ---------------------------------------------------------------------------

def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def find_word_position(sentence: str, word: str) -> tuple[int, int]:
    """Find character span of *word* in *sentence*.

    First tries an exact whole-word regex match.  If that fails, falls back to
    the token with the smallest edit distance (handles morphological variants).

    Returns (start, end) character offsets.
    """
    # Exact match first
    match = re.search(rf"\b{re.escape(word)}\b", sentence, re.IGNORECASE)
    if match:
        return match.start(), match.end()

    # Edit-distance fallback
    tokens = list(re.finditer(r"\b\w+'?\w*\b", sentence))
    best_pos: tuple[int, int] | None = None
    best_dist = float("inf")
    for tok in tokens:
        d = edit_distance(word.lower(), tok.group(0).lower())
        if d < best_dist:
            best_dist = d
            best_pos = (tok.start(), tok.end())

    if best_pos is None:
        raise ValueError(f"Cannot locate '{word}' in sentence: {sentence!r}")
    return best_pos


# ---------------------------------------------------------------------------
# Hover-text helpers
# ---------------------------------------------------------------------------

def split_text(text: str, max_line_length: int = 95) -> str:
    """Insert HTML line breaks for hover display."""
    return "<br>".join(
        text[i : i + max_line_length]
        for i in range(0, len(text), max_line_length)
    )


def bold_word_in_sentence(sentence: str, word: str) -> str:
    """Bold all occurrences of *word* in *sentence* (case-insensitive, HTML)."""
    pattern = re.compile(rf"(\b{re.escape(word)}\b)", re.IGNORECASE)
    return pattern.sub(r"<b>\1</b>", sentence)


def format_hover_text(
    sentence: str,
    word: str,
    corpus: str,
    max_line_length: int = 95,
    nearest_definition: str | None = None,
) -> str:
    """Build a rich hover string for a single usage point."""
    text = bold_word_in_sentence(split_text(sentence, max_line_length), word)
    parts = [f"<b>[{corpus}]</b><br>{text}"]
    if nearest_definition:
        parts.append(f"<br><i>Nearest def: {nearest_definition}</i>")
    return "".join(parts)
