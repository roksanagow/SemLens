"""Seeds: the words or usages that define the concept being looked for.

Seeds come in three kinds, combined into one list of seed vectors:

- **Seed words**: every sampled occurrence of the words (optionally only in
  some corpora) is a seed.
- **Seed usages**: sentences with a marked occurrence, from a table or from
  annotations made in SemLens. They are embedded with the same model and
  layer as the sample.
- **Additions**: sampled occurrences marked as fitting, and words accepted,
  during exploration.

Each seed has a key: ``("row", r)`` for a sampled occurrence and
``("usage", id)`` for an outside usage. Keys keep their order, so seeds added
later extend the list and only they need comparing.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass

import numpy as np

from ..utils import usage_search_pattern
from ..words import entry_of


@dataclass(frozen=True)
class SeedUsage:
    sentence: str
    start: int
    end: int
    corpus: str = ""
    source: str = ""

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps([self.sentence, self.start, self.end]).encode()).hexdigest()[:16]

    @property
    def word(self) -> str:
        return self.sentence[self.start:self.end]

    def as_dict(self) -> dict:
        return asdict(self)


def parse_entries(text: str) -> list[str]:
    """Seed words typed with commas or new lines between them; a trailing * matches any ending."""
    seen, entries = set(), []
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if part and part.casefold() not in seen:
            if part.strip("*") == "":
                raise ValueError("A seed word cannot be only an asterisk.")
            entries.append(part)
            seen.add(part.casefold())
    return entries


def seed_usages_from_table(text: str, *, forms: list[str] | None = None, delimiter: str = ",", source: str = "table") -> list[SeedUsage]:
    """Seed usages from a table with a ``sentence`` column.

    ``start`` and ``end`` columns (characters, from 0, end not included) mark
    the occurrence; without them, every occurrence of ``forms`` is a seed.
    A ``corpus`` column is kept when present.
    """
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        raise ValueError("The table has no rows.")
    columns = {c.strip().casefold(): c for c in rows[0].keys() if c}
    if "sentence" not in columns:
        raise ValueError("The table needs a column named sentence.")
    has_span = "start" in columns and "end" in columns
    pattern = usage_search_pattern(",".join(forms)) if forms else None
    if not has_span and pattern is None:
        raise ValueError("The table has no start and end columns: give the seed word, so its occurrences can be found.")
    found = []
    for number, row in enumerate(rows, start=2):
        sentence = (row.get(columns["sentence"]) or "").strip("\n")
        if not sentence.strip():
            continue
        corpus = (row.get(columns["corpus"]) or "").strip() if "corpus" in columns else ""
        if has_span and str(row.get(columns["start"]) or "").strip():
            try:
                start, end = int(row[columns["start"]]), int(row[columns["end"]])
            except ValueError:
                raise ValueError(f"Row {number}: start and end must be whole numbers.") from None
            if not 0 <= start < end <= len(sentence):
                raise ValueError(f"Row {number}: start and end fall outside the sentence.")
            found.append(SeedUsage(sentence, start, end, corpus, source))
        elif pattern is not None:
            for match in pattern.finditer(sentence):
                found.append(SeedUsage(sentence, match.start(), match.end(), corpus, source))
    if not found:
        raise ValueError("No seed occurrences were found in the table.")
    return list(dict.fromkeys(found))


def seed_usages_from_annotations(data, annotations: dict, senses, *, min_rating: int | None = None, source: str = "annotations") -> list[SeedUsage]:
    """The usages of a SemLens analysis annotated with any of ``senses``.

    With graded annotation, ``min_rating`` keeps only usages rated at least
    that high for one of the senses (unrated senses count as fitting).
    """
    chosen = set(senses)
    found = []
    for index, labels in annotations.items():
        if not 0 <= index < len(data.usages):
            continue
        ratings = [r for sense, r in (labels or {}).items() if sense in chosen]
        if not ratings:
            continue
        if min_rating is not None and not any(r is None or r >= min_rating for r in ratings):
            continue
        usage = data.usages[index]
        found.append(SeedUsage(usage.sentence, usage.char_start, usage.char_end, usage.corpus, source))
    return list(dict.fromkeys(found))


def seed_rows(sample, entries, corpora=None, valid=None) -> np.ndarray:
    """Sampled occurrences of the seed words, optionally only in ``corpora``."""
    entries = [e.strip().casefold() for e in entries if e.strip()]
    if not entries:
        return np.zeros(0, dtype=np.int64)
    matching = np.array([entry_of(w.casefold(), entries) is not None for w in sample.words], dtype=bool)
    keep = matching[sample.word]
    if corpora is not None:
        allowed = np.array([c in set(corpora) for c in sample.corpora], dtype=bool)
        keep &= allowed[sample.corpus]
    if valid is not None:
        keep &= valid
    return np.flatnonzero(keep)


def seed_word_table(sample, entries, valid=None):
    """How many sampled occurrences each seed entry has in each corpus."""
    import pandas as pd

    rows = []
    for entry in entries:
        found = seed_rows(sample, [entry], valid=valid)
        counts = np.bincount(sample.corpus[found], minlength=len(sample.corpora)) if len(found) else np.zeros(len(sample.corpora), int)
        rows.append({"Seed word": entry, **{c: int(n) for c, n in zip(sample.corpora, counts)}, "Total": int(counts.sum())})
    return pd.DataFrame(rows)


def distinct_seed_words(sample, keys, usages: dict[str, SeedUsage]) -> set[str]:
    """The different words among the seeds (lower case)."""
    words = set()
    for kind, value in keys:
        if kind == "row":
            words.add(sample.word_of(value).casefold())
        elif value in usages:
            words.add(usages[value].word.casefold())
    return words


def _pieces(tokenizer, word: str) -> list[str]:
    pieces = tokenizer.tokenize(" " + word)
    return [p.replace("▁", "").replace("Ġ", "").replace("##", "").casefold() for p in pieces if p not in {"▁", "Ġ"}]


def spelling_overlaps(tokenizer, candidates, seed_words) -> dict[str, str]:
    """Candidates that share a word piece with a seed word, with a note saying which.

    Contextual models build a word split into pieces partly from those pieces,
    so *bust* (bu + st) can look like *bubble* (bu + bble) for its spelling
    alone. A shared piece is noted when it has at least two letters and at
    least one of the two words is split.
    """
    seeds = {}
    for word in seed_words:
        # Lower case, as the candidates are: cased tokenizers split Bubble and bubble differently.
        word = word.rstrip("*").casefold()
        if word and word not in seeds:
            seeds[word] = _pieces(tokenizer, word)
    notes = {}
    for candidate in candidates:
        pieces = _pieces(tokenizer, candidate)
        best = None
        for seed, seed_pieces in seeds.items():
            if candidate.casefold() == seed or (len(pieces) < 2 and len(seed_pieces) < 2):
                continue
            shared = [p for p in dict.fromkeys(pieces) if len(p) >= 2 and p in seed_pieces]
            # The seed sharing the most letters; the first given wins a tie.
            if shared and (best is None or sum(map(len, shared)) > sum(map(len, best[1]))):
                best = (seed, shared)
        if best:
            notes[candidate] = f"shares “{'+'.join(best[1])}” with {best[0]}"
    return notes


def seed_keys(sample, *, words=(), corpora=None, usages=(), marked_rows=(), accepted_words=(), valid=None) -> list[tuple]:
    """Every seed's key, in a fixed order: seed words, seed usages, then additions in the order they were made."""
    keys: list[tuple] = [("row", int(r)) for r in seed_rows(sample, words, corpora, valid)]
    keys += [("usage", u.key) for u in usages]
    keys += [("row", int(r)) for r in marked_rows if valid is None or valid[int(r)]]
    for word in accepted_words:
        keys += [("row", int(r)) for r in seed_rows(sample, [word], valid=valid)]
    return list(dict.fromkeys(keys))


def seed_matrix(keys, store, usage_vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Unit-length vectors for the seed keys, in order. Seed usages without a vector are an error."""
    rows = sorted({value for kind, value in keys if kind == "row"})
    by_row = dict(zip(rows, store.normalised(np.asarray(rows, dtype=np.int64)))) if rows else {}
    out = np.zeros((len(keys), store.dim), dtype=np.float32)
    for i, (kind, value) in enumerate(keys):
        if kind == "row":
            out[i] = by_row[value]
        elif value in usage_vectors:
            out[i] = usage_vectors[value]
        else:
            raise KeyError(f"Seed usage {value} has not been embedded.")
    return out
