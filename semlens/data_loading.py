"""Corpus loading and input normalisation.

Accepts multiple input formats and normalises them into a unified
``TargetWordData`` structure ready for embedding.

Supported inputs
----------------
1. **Sentences + positions + corpus labels** (most specific)
2. **Sentences + target word + corpus labels** (positions auto-detected)
3. **Plain text sentences** (one per line, corpus label from filename or user)
4. **CSV / TSV** with configurable column names
5. **Pre-computed embeddings** (``.pt`` / ``.npy``) — bypasses embedding step
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .utils import find_word_position


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class UsageInstance:
    """A single occurrence of the target word in a sentence."""

    sentence: str
    word: str
    char_start: int
    char_end: int
    corpus: str  # user-provided label, e.g. "1800s", "modern", "corpus_1"

    @property
    def char_span(self) -> tuple[int, int]:
        return (self.char_start, self.char_end)


@dataclass
class TargetWordData:
    """All usages of a target word across corpora."""

    word: str
    usages: list[UsageInstance] = field(default_factory=list)

    # --- derived helpers ---------------------------------------------------

    @property
    def corpus_labels(self) -> list[str]:
        """Sorted unique corpus labels."""
        return sorted(set(u.corpus for u in self.usages))

    @property
    def sentences(self) -> list[str]:
        return [u.sentence for u in self.usages]

    @property
    def char_spans(self) -> list[tuple[int, int]]:
        return [u.char_span for u in self.usages]

    @property
    def corpus_list(self) -> list[str]:
        """Per-usage corpus label (parallel to sentences)."""
        return [u.corpus for u in self.usages]

    def usages_for_corpus(self, corpus: str) -> list[UsageInstance]:
        return [u for u in self.usages if u.corpus == corpus]

    def indices_for_corpus(self, corpus: str) -> list[int]:
        return [i for i, u in enumerate(self.usages) if u.corpus == corpus]

    def __len__(self) -> int:
        return len(self.usages)


# ---------------------------------------------------------------------------
# Public loading functions
# ---------------------------------------------------------------------------

def load_from_sentences(
    sentences: Sequence[str],
    word: str,
    corpus_labels: Sequence[str],
) -> TargetWordData:
    """Build ``TargetWordData`` from parallel lists of sentences and corpus labels.

    Word positions are detected automatically via exact match / edit-distance
    fallback.

    Parameters
    ----------
    sentences : list[str]
        One sentence per usage.
    word : str
        The target lemma.
    corpus_labels : list[str]
        Same length as *sentences*; the corpus each sentence belongs to.
    """
    if len(sentences) != len(corpus_labels):
        raise ValueError(
            f"sentences ({len(sentences)}) and corpus_labels "
            f"({len(corpus_labels)}) must have the same length."
        )
    usages: list[UsageInstance] = []
    for sent, corpus in zip(sentences, corpus_labels):
        start, end = find_word_position(sent, word)
        usages.append(UsageInstance(sent, word, start, end, corpus))
    return TargetWordData(word=word, usages=usages)


def load_from_sentences_with_positions(
    sentences: Sequence[str],
    word: str,
    starts: Sequence[int],
    ends: Sequence[int],
    corpus_labels: Sequence[str],
) -> TargetWordData:
    """Build ``TargetWordData`` when positions are already known."""
    n = len(sentences)
    if not (len(starts) == len(ends) == len(corpus_labels) == n):
        raise ValueError("All input sequences must have the same length.")
    usages = [
        UsageInstance(s, word, int(st), int(en), c)
        for s, st, en, c in zip(sentences, starts, ends, corpus_labels)
    ]
    return TargetWordData(word=word, usages=usages)


def load_from_csv(
    path: str | Path,
    word: str,
    *,
    sentence_col: str = "sentence",
    corpus_col: str = "corpus",
    start_col: str | None = None,
    end_col: str | None = None,
    delimiter: str | None = None,
) -> TargetWordData:
    """Load from a CSV or TSV file.

    If *start_col* and *end_col* are provided, uses them for positions;
    otherwise auto-detects positions.

    The delimiter is auto-detected from the file extension (.tsv → tab) unless
    explicitly provided.
    """
    path = Path(path)
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","

    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    sentences = [r[sentence_col] for r in rows]
    corpus_labels = [r[corpus_col] for r in rows]

    if start_col and end_col and start_col in rows[0] and end_col in rows[0]:
        starts = [int(r[start_col]) for r in rows]
        ends = [int(r[end_col]) for r in rows]
        return load_from_sentences_with_positions(
            sentences, word, starts, ends, corpus_labels
        )
    else:
        return load_from_sentences(sentences, word, corpus_labels)


def load_from_text_files(
    paths: dict[str, str | Path],
    word: str,
) -> TargetWordData:
    """Load from plain-text files (one sentence per line).

    Parameters
    ----------
    paths : dict[str, Path]
        Mapping of ``corpus_label → file_path``.  Each file contains one
        sentence per line.  Only sentences containing the target word are kept.
    word : str
        The target lemma.
    """
    usages: list[UsageInstance] = []
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)

    for corpus_label, fpath in paths.items():
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if pattern.search(line):
                    start, end = find_word_position(line, word)
                    usages.append(UsageInstance(line, word, start, end, corpus_label))

    if not usages:
        raise ValueError(
            f"No sentences containing '{word}' found in the provided files."
        )
    return TargetWordData(word=word, usages=usages)


def _split_into_sentences(text: str) -> list[str]:
    """Split raw text into sentences on full stops (and other sentence-final punctuation).

    Splits on ``.``, ``!``, ``?`` followed by whitespace or end-of-string,
    keeping the punctuation attached to the sentence.  Filters out fragments
    shorter than 5 characters.
    """
    # Split on sentence-ending punctuation followed by whitespace or EOF
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) >= 5:
            sentences.append(s)
    return sentences


def load_from_raw_corpora(
    paths: dict[str, str | Path],
    word: str,
) -> TargetWordData:
    """Load from raw corpus text files and extract sentences containing the target word.

    Each file is treated as continuous text.  Sentences are obtained by
    splitting on full stops (and ``!``, ``?``).  Only sentences where the
    target word appears as an exact token — surrounded by whitespace or
    punctuation — are kept.

    Parameters
    ----------
    paths : dict[str, Path]
        Mapping of ``corpus_label → file_path``.  e.g.
        ``{"historical": "old_corpus.txt", "modern": "new_corpus.txt"}``.
    word : str
        The target lemma.

    Returns
    -------
    TargetWordData
    """
    usages: list[UsageInstance] = []

    # Match the word as a whole token: preceded and followed by a word
    # boundary (handles whitespace, punctuation, start/end of string)
    pattern = re.compile(rf"\b{re.escape(word)}\b")

    for corpus_label, fpath in paths.items():
        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        sentences = _split_into_sentences(text)

        for sent in sentences:
            if pattern.search(sent):
                start, end = find_word_position(sent, word)
                usages.append(UsageInstance(sent, word, start, end, corpus_label))

    if not usages:
        raise ValueError(
            f"No sentences containing '{word}' found in the provided corpus files."
        )
    return TargetWordData(word=word, usages=usages)


def load_from_raw_text(
    texts: dict[str, str],
    word: str,
) -> TargetWordData:
    """Same as ``load_from_raw_corpora`` but accepts in-memory strings
    instead of file paths.  Useful for the Streamlit interface where file
    contents are already read.

    Parameters
    ----------
    texts : dict[str, str]
        Mapping of ``corpus_label → raw text string``.
    word : str
        The target lemma.
    """
    usages: list[UsageInstance] = []
    pattern = re.compile(rf"\b{re.escape(word)}\b")

    for corpus_label, text in texts.items():
        sentences = _split_into_sentences(text)
        for sent in sentences:
            if pattern.search(sent):
                start, end = find_word_position(sent, word)
                usages.append(UsageInstance(sent, word, start, end, corpus_label))

    if not usages:
        raise ValueError(
            f"No sentences containing '{word}' found in the provided text."
        )
    return TargetWordData(word=word, usages=usages)


# ---------------------------------------------------------------------------
# Pre-computed embeddings
# ---------------------------------------------------------------------------

def load_precomputed_embeddings(
    path: str | Path,
) -> torch.Tensor:
    """Load pre-computed embeddings from ``.pt`` (PyTorch) or ``.npy`` (NumPy).

    Returns a ``[N, D]`` tensor.  The caller is responsible for ensuring
    that the ordering matches the corresponding ``TargetWordData.usages``.
    """
    path = Path(path)
    if path.suffix == ".pt":
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a Tensor in {path}, got {type(tensor)}")
        return tensor
    elif path.suffix == ".npy":
        arr = np.load(path)
        return torch.from_numpy(arr).float()
    else:
        raise ValueError(
            f"Unsupported embedding file format: {path.suffix}. "
            "Use .pt (PyTorch) or .npy (NumPy)."
        )
