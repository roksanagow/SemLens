"""Corpus input, explicit occurrences, and reusable sentence records."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np
import torch

MAX_CORPORA = 10


def normalise_forms(word: str, forms: Sequence[str] | None = None) -> list[str]:
    if not word.strip():
        raise ValueError("Enter a target word.")
    result = []
    seen = set()
    for form in [word, *(forms or [])]:
        form = form.strip()
        if form and form.casefold() not in seen:
            result.append(form)
            seen.add(form.casefold())
    return result


def find_occurrences(sentence: str, word: str, forms: Sequence[str] | None = None):
    """Find every explicit form, case insensitively, including repeated occurrences."""
    variants = sorted(normalise_forms(word, forms), key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(v) for v in variants) + r")(?!\w)", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(sentence)]


@dataclass
class UsageInstance:
    sentence: str
    word: str
    char_start: int
    char_end: int
    corpus: str
    sentence_id: str = ""

    def __post_init__(self):
        if not isinstance(self.sentence, str) or not self.sentence.strip():
            raise ValueError("Each usage must have a non-empty sentence.")
        if not isinstance(self.corpus, str) or not self.corpus.strip():
            raise ValueError("Each usage must have a non-empty corpus name.")
        if not (0 <= self.char_start < self.char_end <= len(self.sentence)):
            raise ValueError(f"Invalid target span ({self.char_start}, {self.char_end}) in {self.sentence[:80]!r}.")
        if not self.matched_form.strip():
            raise ValueError("A target span cannot contain only whitespace.")

    @property
    def char_span(self) -> tuple[int, int]:
        return self.char_start, self.char_end

    @property
    def matched_form(self) -> str:
        return self.sentence[self.char_start:self.char_end]

    @property
    def usage_id(self) -> str:
        content = [self.corpus, self.sentence_id, self.sentence, self.char_start, self.char_end, self.word]
        return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


@dataclass
class TargetWordData:
    word: str
    usages: list[UsageInstance] = field(default_factory=list)
    corpus_order: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.forms = normalise_forms(self.word, self.forms)
        present = list(dict.fromkeys(u.corpus for u in self.usages))
        if len(present) > MAX_CORPORA:
            raise ValueError(f"SemLens supports up to {MAX_CORPORA} corpora per analysis.")
        self.corpus_order = list(dict.fromkeys([c for c in self.corpus_order if c in present] + present))

    @property
    def corpus_labels(self) -> list[str]:
        return self.corpus_order.copy()

    @property
    def sentences(self) -> list[str]:
        return [u.sentence for u in self.usages]

    @property
    def char_spans(self) -> list[tuple[int, int]]:
        return [u.char_span for u in self.usages]

    @property
    def corpus_list(self) -> list[str]:
        return [u.corpus for u in self.usages]

    def usages_for_corpus(self, corpus: str) -> list[UsageInstance]:
        return [u for u in self.usages if u.corpus == corpus]

    def indices_for_corpus(self, corpus: str) -> list[int]:
        return [i for i, u in enumerate(self.usages) if u.corpus == corpus]

    def subset(self, indices: Sequence[int]) -> TargetWordData:
        return TargetWordData(self.word, [self.usages[i] for i in indices], self.corpus_order, self.forms)

    def __len__(self):
        return len(self.usages)


@dataclass
class CorpusSentence:
    corpus: str
    sentence: str
    sentence_id: str


def validate_corpus_names(names: Sequence[str]):
    if not 1 <= len(names) <= MAX_CORPORA:
        raise ValueError(f"Provide between 1 and {MAX_CORPORA} corpora.")
    if any(not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("Corpus names must be non-empty and unique.")


def _split_into_sentences(text: str) -> list[str]:
    """Lightweight punctuation splitter; no text is discarded for being short."""
    text = re.sub(r"\s+", " ", text.lstrip("\ufeff")).strip()
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|(?<=[。！？])', text) if s.strip()]


def corpus_sentences(texts: dict[str, str], *, lines: bool = False) -> list[CorpusSentence]:
    validate_corpus_names(list(texts))
    records = []
    for corpus, text in texts.items():
        sentences = [s.strip() for s in text.splitlines() if s.strip()] if lines else _split_into_sentences(text)
        if not sentences:
            raise ValueError(f"Corpus {corpus!r} has no sentences.")
        for i, sentence in enumerate(sentences):
            sid = hashlib.sha256(f"{corpus}\0{i}\0{sentence}".encode()).hexdigest()
            records.append(CorpusSentence(corpus, sentence, sid))
    return records


def select_usages(records: Sequence[CorpusSentence], word: str, forms: Sequence[str] | None = None) -> TargetWordData:
    usages = []
    for r in records:
        for start, end in find_occurrences(r.sentence, word, forms):
            usages.append(UsageInstance(r.sentence, word, start, end, r.corpus, r.sentence_id))
    if not usages:
        raise ValueError(f"No sentences containing {', '.join(normalise_forms(word, forms))!r} found. Add other forms if needed.")
    return TargetWordData(word, usages, list(dict.fromkeys(r.corpus for r in records)), list(forms or []))


def load_from_sentences(sentences: Sequence[str], word: str, corpus_labels: Sequence[str], *,
                        forms: Sequence[str] | None = None, all_occurrences: bool = True) -> TargetWordData:
    if len(sentences) != len(corpus_labels):
        raise ValueError("sentences and corpus_labels must have the same length.")
    usages = []
    for i, (sent, corpus) in enumerate(zip(sentences, corpus_labels)):
        matches = find_occurrences(sent, word, forms)
        if not matches:
            raise ValueError(f"Row {i + 1}: no target form found in {sent[:100]!r}. Supply its position or add the form.")
        if not all_occurrences and len(matches) != 1:
            raise ValueError(f"Row {i + 1}: multiple target occurrences. Supply start/end columns to identify the embedding's occurrence.")
        for start, end in matches:
            usages.append(UsageInstance(sent, word, start, end, corpus, str(i)))
    return TargetWordData(word, usages, forms=list(forms or []))


def load_from_sentences_with_positions(sentences: Sequence[str], word: str, starts: Sequence[int],
                                      ends: Sequence[int], corpus_labels: Sequence[str]) -> TargetWordData:
    if not (len(starts) == len(ends) == len(corpus_labels) == len(sentences)):
        raise ValueError("All input sequences must have the same length.")
    usages = [UsageInstance(s, word, int(st), int(en), c, str(i))
              for i, (s, st, en, c) in enumerate(zip(sentences, starts, ends, corpus_labels))]
    return TargetWordData(word, usages)


def load_from_csv(path: str | Path | TextIO, word: str, *, sentence_col: str = "sentence",
                  corpus_col: str = "corpus", start_col: str | None = "start", end_col: str | None = "end",
                  delimiter: str | None = None, forms: Sequence[str] | None = None,
                  all_occurrences: bool = True) -> TargetWordData:
    if delimiter is None:
        name = str(path) if isinstance(path, (str, Path)) else getattr(path, "name", "")
        delimiter = "\t" if Path(name).suffix.lower() in (".tsv", ".tab") else ","
    if isinstance(path, (str, Path)):
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter=delimiter))
    else:
        rows = list(csv.DictReader(path, delimiter=delimiter))
    if not rows:
        raise ValueError("No data rows found.")
    if sentence_col not in rows[0] or corpus_col not in rows[0]:
        raise ValueError(f"Required columns: {sentence_col}, {corpus_col}.")
    usages = []
    for i, row in enumerate(rows):
        sent, corpus = row[sentence_col], row[corpus_col]
        start, end = row.get(start_col, "") if start_col else "", row.get(end_col, "") if end_col else ""
        has_start, has_end = start not in ("", None), end not in ("", None)
        if has_start != has_end:
            raise ValueError(f"Row {i + 1}: supply both start and end, or leave both empty.")
        if has_start:
            try:
                spans = [(int(start), int(end))]
            except (TypeError, ValueError):
                raise ValueError(f"Row {i + 1}: start/end must be integer character offsets.") from None
        else:
            spans = find_occurrences(sent or "", word, forms)
            if not spans:
                raise ValueError(f"Row {i + 1}: target missing; provide offsets or another form.")
            if not all_occurrences and len(spans) != 1:
                raise ValueError(f"Row {i + 1}: multiple occurrences; supply start/end for pre-computed embeddings.")
        sid = row.get("sentence_id") or str(i)
        for start, end in spans:
            usages.append(UsageInstance(sent, word, start, end, corpus, sid))
    return TargetWordData(word, usages, forms=list(forms or []))


def load_from_text_files(paths: dict[str, str | Path], word: str, *, forms=None) -> TargetWordData:
    texts = {c: Path(p).read_text(encoding="utf-8-sig") for c, p in paths.items()}
    return select_usages(corpus_sentences(texts, lines=True), word, forms)


def load_from_raw_corpora(paths: dict[str, str | Path], word: str, *, forms=None) -> TargetWordData:
    return load_from_raw_text({c: Path(p).read_text(encoding="utf-8-sig") for c, p in paths.items()}, word, forms=forms)


def load_from_raw_text(texts: dict[str, str], word: str, *, forms=None) -> TargetWordData:
    return select_usages(corpus_sentences(texts), word, forms)


def validate_embeddings(embeddings: torch.Tensor, count: int | None = None) -> torch.Tensor:
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or min(embeddings.shape) == 0:
        raise ValueError("Embeddings must be a non-empty [number of usages, dimensions] tensor.")
    if count is not None and embeddings.shape[0] != count:
        raise ValueError(f"Embedding count ({embeddings.shape[0]}) does not match usage count ({count}).")
    embeddings = embeddings.detach().cpu().float()
    if not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings contain missing or infinite values.")
    return embeddings


def load_precomputed_embeddings(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix == ".npy":
        tensor = torch.from_numpy(np.load(path, allow_pickle=False))
    else:
        raise ValueError("Use .pt (PyTorch) or .npy (NumPy) embeddings.")
    return validate_embeddings(tensor)
