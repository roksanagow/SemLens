"""Which words occur in the corpora, and an even sample of their occurrences.

Discover compares every word of the corpora with the seeds. Before anything
is embedded, the corpora are divided into contexts, optionally restricted to
contexts containing given words or phrases, and every word is counted. Each
word keeps at most ``cap`` occurrences per corpus, drawn at random from all of
its occurrences (reservoir sampling), so the sample is spread evenly through
the corpus rather than taken from its beginning.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..data_loading import corpus_sentences
from ..utils import usage_search_pattern

# A word: letters, with internal hyphens or apostrophes (dot-com, don't). Numbers and punctuation are not words.
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")


def default_stop_words() -> set[str]:
    """English function words (scikit-learn's list), which rarely carry the concept being looked for."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return set(ENGLISH_STOP_WORDS)


@dataclass(frozen=True)
class Context:
    corpus: str
    text: str


def contexts_from_texts(texts: dict[str, str], *, split: str = "auto", restrict: str | None = None) -> list[Context]:
    """The contexts of each corpus, keeping only those that contain ``restrict`` when it is given.

    ``restrict`` uses SemLens's search syntax: words or phrases separated by
    commas, any of which may appear, and a trailing ``*`` for any ending.
    """
    pattern = usage_search_pattern(restrict) if restrict and restrict.strip() else None
    return [Context(r.corpus, r.sentence) for r in corpus_sentences(texts, split=split)
            if pattern is None or pattern.search(r.sentence)]


def words_in(text: str, lowercase: bool = True) -> Iterable[tuple[str, int, int]]:
    for match in WORD_PATTERN.finditer(text):
        word = match.group(0)
        yield (word.casefold() if lowercase else word), match.start(), match.end()


def count_occurrences(contexts: Sequence[Context], lowercase: bool = True) -> Counter:
    """Occurrences of every word in every corpus, keyed by (corpus, word). Nothing is filtered yet."""
    counts: Counter = Counter()
    for context in contexts:
        for word, _, _ in words_in(context.text, lowercase):
            counts[(context.corpus, word)] += 1
    return counts


def _kept(word: str, n: int, *, min_count: int, min_length: int, stop: set, keep: set, only: set | None) -> bool:
    folded = word.casefold()
    if folded in keep:
        return True
    if only is not None and folded not in only:
        return False
    return n >= min_count and len(word) >= min_length and folded not in stop


def cap_estimates(counts: Counter, caps: Sequence[int], *, min_count: int = 5, min_length: int = 2, stop_words: Iterable[str] = (),
                  keep_words: Iterable[str] = (), only_words: Iterable[str] | None = None) -> pd.DataFrame:
    """For each cap (0 for no cap): words kept, occurrences to embed, and the share of all occurrences that is."""
    stop = {w.casefold() for w in stop_words}
    keep = {w.casefold() for w in keep_words}
    only = None if only_words is None else {w.casefold() for w in only_words}
    kept = [n for (corpus, word), n in counts.items()
            if _kept(word, n, min_count=min_count, min_length=min_length, stop=stop, keep=keep, only=only)]
    total = sum(kept)
    words = len({word for (corpus, word), n in counts.items()
                 if _kept(word, n, min_count=min_count, min_length=min_length, stop=stop, keep=keep, only=only)})
    rows = []
    for cap in caps:
        sampled = total if not cap else sum(min(n, cap) for n in kept)
        rows.append({"cap": cap, "words": words, "occurrences": sampled, "share": sampled / total if total else 0.0,
                     "capped": sum(1 for n in kept if cap and n > cap)})
    return pd.DataFrame(rows)


@dataclass
class Sample:
    """Sampled occurrences: one row per occurrence, with its corpus, context, position and word."""
    contexts: list[Context]
    corpora: list[str]
    words: list[str]
    corpus: np.ndarray      # index into ``corpora``
    context: np.ndarray     # index into ``contexts``
    start: np.ndarray
    end: np.ndarray
    word: np.ndarray        # index into ``words``
    counts: dict[str, dict[str, int]]   # occurrences of each kept word in each corpus, before the cap
    settings: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.word)

    def word_of(self, row: int) -> str:
        return self.words[int(self.word[row])]

    def text_of(self, row: int) -> str:
        return self.contexts[int(self.context[row])].text

    def span_of(self, row: int) -> tuple[int, int]:
        return int(self.start[row]), int(self.end[row])

    def contexts_to_encode(self) -> list[int]:
        """Contexts holding at least one sampled occurrence, in corpus order."""
        return sorted(set(self.context.tolist()))

    def summary(self) -> pd.DataFrame:
        """Per corpus: contexts, words counted, words kept, occurrences sampled, and what the cap left out."""
        cap = self.settings.get("cap")
        rows = []
        for c, corpus in enumerate(self.corpora):
            counts = self.counts.get(corpus, {})
            sampled = int((self.corpus == c).sum())
            capped = sum(1 for n in counts.values() if cap and n > cap)
            rows.append({"Corpus": corpus,
                         "Contexts": sum(1 for x in self.contexts if x.corpus == corpus),
                         "Word occurrences counted": int(self.settings.get("tokens", {}).get(corpus, 0)),
                         "Words kept": len(counts),
                         "Occurrences sampled": sampled,
                         "Words over the cap": capped,
                         "Occurrences left out by the cap": int(sum(counts.values()) - sampled)})
        return pd.DataFrame(rows)

    # --- Saving -------------------------------------------------------------------------------------

    def save(self, folder: str | Path):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "sample.npz", corpus=self.corpus, context=self.context, start=self.start, end=self.end, word=self.word)
        (folder / "sample.json").write_text(json.dumps({
            "contexts": [[c.corpus, c.text] for c in self.contexts], "corpora": self.corpora, "words": self.words,
            "counts": self.counts, "settings": self.settings}, ensure_ascii=False))

    @classmethod
    def load(cls, folder: str | Path) -> "Sample":
        folder = Path(folder)
        meta = json.loads((folder / "sample.json").read_text())
        arrays = np.load(folder / "sample.npz")
        return cls([Context(c, t) for c, t in meta["contexts"]], meta["corpora"], meta["words"],
                   arrays["corpus"], arrays["context"], arrays["start"], arrays["end"], arrays["word"],
                   meta["counts"], meta["settings"])


def sample_occurrences(contexts: Sequence[Context], *, cap: int = 100, min_count: int = 5,
                       stop_words: Iterable[str] = (), keep_words: Iterable[str] = (), min_length: int = 2,
                       lowercase: bool = True, seed: int = 42, corpora: Sequence[str] | None = None,
                       only_words: Iterable[str] | None = None) -> Sample:
    """Count every word in ``contexts`` and keep at most ``cap`` occurrences of each, per corpus.

    A word is kept in a corpus when it occurs there at least ``min_count``
    times, is at least ``min_length`` letters long and is not a stop word.
    ``keep_words`` (seed words, say) are always kept; with ``only_words``, no
    other words are. Each word's occurrences are sampled uniformly at random
    from all of them, with a fixed ``seed``. A cap of 0 keeps every occurrence.
    """
    if cap < 0:
        raise ValueError("The cap must be a positive number of occurrences per word, or 0 for all.")
    cap = cap or 10**12
    stop = {w.casefold() for w in stop_words}
    keep = {w.casefold() for w in keep_words}
    only = None if only_words is None else {w.casefold() for w in only_words}
    rng = random.Random(seed)
    reservoirs: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    seen: Counter = Counter()
    tokens: Counter = Counter()
    for index, context in enumerate(contexts):
        for word, start, end in words_in(context.text, lowercase):
            key_word = word.casefold()
            if key_word not in keep and (len(word) < min_length or key_word in stop or (only is not None and key_word not in only)):
                continue
            key = (context.corpus, word)
            tokens[context.corpus] += 1
            seen[key] += 1
            n = seen[key]
            reservoir = reservoirs.setdefault(key, [])
            if n <= cap:
                reservoir.append((index, start, end))
            else:
                # Reservoir sampling: every occurrence ends up kept with the same chance, cap / n.
                j = rng.randrange(n)
                if j < cap:
                    reservoir[j] = (index, start, end)
    order = list(corpora) if corpora else list(dict.fromkeys(c.corpus for c in contexts))
    kept = sorted((key for key, n in seen.items() if n >= min_count or key[1].casefold() in keep),
                  key=lambda k: (order.index(k[0]) if k[0] in order else len(order), k[1]))
    words = sorted({w for _, w in kept})
    word_index = {w: i for i, w in enumerate(words)}
    rows = []
    counts: dict[str, dict[str, int]] = {c: {} for c in order}
    for corpus, word in kept:
        counts.setdefault(corpus, {})[word] = int(seen[(corpus, word)])
        c = order.index(corpus)
        for ctx, start, end in sorted(reservoirs[(corpus, word)]):
            rows.append((c, ctx, start, end, word_index[word]))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    array = np.array(rows, dtype=np.int64).reshape(-1, 5)
    settings = {"cap": 0 if cap >= 10**12 else cap, "min_count": min_count, "only_words": None if only is None else len(only), "min_length": min_length, "lowercase": lowercase, "seed": seed,
                "stop_words": len(stop), "keep_words": sorted(keep), "tokens": {c: int(tokens[c]) for c in order}}
    return Sample(list(contexts), order, words, array[:, 0].astype(np.int16), array[:, 1].astype(np.int32),
                  array[:, 2].astype(np.int32), array[:, 3].astype(np.int32), array[:, 4].astype(np.int32), counts, settings)
