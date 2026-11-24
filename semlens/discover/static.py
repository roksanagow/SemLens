"""A cheap first pass: one vector per word and corpus, aligned across corpora.

Each corpus gets its own static word vectors, from word2vec (gensim) or, for
small corpora, from PPMI co-occurrence counts reduced with SVD. The spaces are
rotated onto the first corpus's (orthogonal Procrustes on the words they
share), so that seed words from every corpus can serve as one set of seeds.
A word's score in a corpus is its mean cosine distance to its ``k`` nearest
seed vectors (kNN-AMD with one vector per word).

Static vectors blend all of a word's senses into one, and need a fair amount
of text: below about a million words per corpus, word2vec is noisy and PPMI
with SVD is the steadier choice.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..words import entry_of
from .vocabulary import Context, words_in

SMALL_CORPUS_WORDS = 1_000_000


@dataclass
class Space:
    words: list[str]
    vectors: np.ndarray        # unit length, centred
    counts: dict[str, int]

    def index(self) -> dict[str, int]:
        return {w: i for i, w in enumerate(self.words)}


def tokenised(contexts: Sequence[Context], corpus: str, lowercase: bool = True) -> list[list[str]]:
    """The contexts of one corpus as lists of words (stop words kept: they are context for the others)."""
    return [[w for w, _, _ in words_in(c.text, lowercase)] for c in contexts if c.corpus == corpus]


def recommended_method(word_counts: dict[str, int]) -> str:
    """word2vec when every corpus has at least a million words, otherwise PPMI with SVD."""
    return "word2vec" if word_counts and min(word_counts.values()) >= SMALL_CORPUS_WORDS else "ppmi"


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors - vectors.mean(axis=0, keepdims=True)
    return (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def word2vec_space(sentences: list[list[str]], *, dimensions: int = 100, window: int = 5, min_count: int = 5,
                   epochs: int = 10, seed: int = 42, workers: int = 1) -> Space:
    """word2vec (skip-gram) vectors. One worker keeps the result reproducible for the same seed."""
    from gensim.models import Word2Vec

    model = Word2Vec(sentences=sentences, vector_size=dimensions, window=window, min_count=min_count, sg=1,
                     epochs=epochs, seed=seed, workers=workers)
    words = list(model.wv.index_to_key)
    counts = Counter(w for s in sentences for w in s)
    return Space(words, _normalise(model.wv[words]), {w: counts[w] for w in words})


def cooccurrence_counts(sentences: list[list[str]], words: list[str], window: int = 5):
    """How often each pair of ``words`` occurs within ``window`` words of each other, in either order (a sparse matrix)."""
    from scipy.sparse import coo_matrix

    index = {w: i for i, w in enumerate(words)}
    # Every token as a word number (-1 for words left out) and its sentence number, so
    # pairs within the window are found with array shifts rather than loops.
    ids = np.fromiter((index.get(w, -1) for s in sentences for w in s), dtype=np.int64)
    sentence = np.repeat(np.arange(len(sentences)), [len(s) for s in sentences])
    n = len(words)
    keys = []
    for d in range(1, window + 1):
        a, b = ids[:-d], ids[d:]
        keep = (a >= 0) & (b >= 0) & (sentence[:-d] == sentence[d:])
        keys.append(a[keep] * n + b[keep])
        keys.append(b[keep] * n + a[keep])
    keys = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64)
    pairs, values = np.unique(keys, return_counts=True)
    return coo_matrix((values.astype(np.float64), (pairs // n, pairs % n)), shape=(n, n)).tocsr()


def ppmi_space(sentences: list[list[str]], *, dimensions: int = 100, window: int = 5, min_count: int = 5, seed: int = 42) -> Space:
    """Positive pointwise mutual information of words within ``window``, reduced to ``dimensions`` with SVD."""
    from scipy.sparse import coo_matrix
    from sklearn.decomposition import TruncatedSVD

    counts = Counter(w for s in sentences for w in s)
    words = sorted(w for w, n in counts.items() if n >= min_count)
    if len(words) < 3:
        raise ValueError("Too few words occur often enough to build vectors. Lower the minimum frequency or add text.")
    matrix = cooccurrence_counts(sentences, words, window)
    if not matrix.nnz:
        raise ValueError("No words occur near each other often enough to build vectors.")
    n = len(words)
    total = matrix.sum()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    # Context counts raised to 0.75, as in word2vec, so rare contexts do not dominate.
    col_sums = np.asarray(matrix.sum(axis=0)).ravel() ** 0.75
    col_sums = col_sums / col_sums.sum() * total
    matrix = matrix.tocoo()
    pmi = np.log(matrix.data * total / (row_sums[matrix.row] * col_sums[matrix.col]))
    keep = pmi > 0
    ppmi = coo_matrix((pmi[keep], (matrix.row[keep], matrix.col[keep])), shape=matrix.shape).tocsr()
    dimensions = max(2, min(dimensions, n - 1))
    vectors = TruncatedSVD(n_components=dimensions, random_state=seed).fit_transform(ppmi)
    return Space(words, _normalise(vectors), {w: counts[w] for w in words})


def static_cost(tokens: int, vocabulary: int, method: str, *, dimensions: int = 100, window: int = 5, epochs: int = 10, workers: int = 1):
    """Rough time and memory for building one corpus's static vectors."""
    from ..resources import Cost

    if method == "word2vec":
        # Roughly 300,000 words a second per worker for skip-gram with a window of 5.
        seconds = tokens * epochs * (window / 5) / (300_000 * max(1, workers))
        memory = vocabulary * dimensions * 4 * 3 + tokens * 60
    else:
        pairs = 2 * window * tokens
        seconds = pairs / 5e6 + vocabulary * dimensions * 50 / 1e9
        memory = pairs * 8 * 3 + tokens * 16 + vocabulary * dimensions * 8 * 4
    return Cost(f"Building {'word2vec' if method == 'word2vec' else 'PPMI'} vectors", memory_bytes=memory, seconds=seconds, rough_time=True,
                details=[f"{tokens:,} words of text, {vocabulary:,} words in the vocabulary"])


def build_spaces(contexts: Sequence[Context], corpora: Sequence[str], method: str = "ppmi", *, dimensions: int = 100,
                 window: int = 5, min_count: int = 5, epochs: int = 10, workers: int = 1, progress_callback=None) -> dict[str, Space]:
    """Static vectors for every corpus, with the same number of dimensions, aligned onto the first corpus."""
    sentences = {c: tokenised(contexts, c) for c in corpora}
    if method == "ppmi":
        # SVD gives at most one dimension fewer than the words: every corpus gets as many as the smallest allows.
        smallest = min(sum(1 for n in Counter(w for s in sents for w in s).values() if n >= min_count) for sents in sentences.values())
        if smallest < 3:
            raise ValueError("A corpus has too few words that occur often enough. Lower the minimum frequency, or add text.")
        dimensions = max(2, min(dimensions, smallest - 1))
    spaces = {}
    for i, corpus in enumerate(corpora):
        if method == "word2vec":
            spaces[corpus] = word2vec_space(sentences[corpus], dimensions=dimensions, window=window, min_count=min_count, epochs=epochs,
                                            workers=workers)
        else:
            spaces[corpus] = ppmi_space(sentences[corpus], dimensions=dimensions, window=window, min_count=min_count)
        if progress_callback:
            progress_callback(i + 1, len(corpora))
    return align(spaces) if len(spaces) > 1 else spaces


def align(spaces: dict[str, Space]) -> dict[str, Space]:
    """Rotate every space onto the first one, using the words they share (orthogonal Procrustes)."""
    from scipy.linalg import orthogonal_procrustes

    names = list(spaces)
    if not names:
        return {}
    base = spaces[names[0]]
    base_index = base.index()
    aligned = {names[0]: base}
    for name in names[1:]:
        space = spaces[name]
        shared = [w for w in space.words if w in base_index]
        if len(shared) < 10:
            raise ValueError(f"“{name}” shares too few words with “{names[0]}” to align them.")
        if space.vectors.shape[1] != base.vectors.shape[1]:
            raise ValueError("The corpora's vectors have different numbers of dimensions, so they cannot be aligned.")
        own = space.index()
        rotation, _ = orthogonal_procrustes(space.vectors[[own[w] for w in shared]], base.vectors[[base_index[w] for w in shared]])
        rotated = space.vectors @ rotation
        rotated = rotated / np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
        aligned[name] = Space(space.words, rotated.astype(np.float32), space.counts)
    return aligned


def static_scores(spaces: dict[str, Space], seed_entries: Sequence[str], k: int = 10, *, hide_seeds: bool = True) -> pd.DataFrame:
    """Each word's mean cosine distance to its ``k`` nearest seed vectors, in every corpus.

    Rows for "All corpora" come first: each word's mean over the corpora it
    occurs in, with ``corpora`` saying how many. Seed vectors are the seed words' vectors in every (aligned) corpus; a seed
    word is never its own neighbour.
    """
    entries = [e.strip().casefold() for e in seed_entries if e.strip()]
    seed_vectors, seed_ids = [], []
    for name, space in spaces.items():
        for i, w in enumerate(space.words):
            if entry_of(w.casefold(), entries) is not None:
                seed_vectors.append(space.vectors[i])
                seed_ids.append((name, w))
    if not seed_vectors:
        raise ValueError("None of the seed words occurs often enough in these corpora.")
    seeds = np.stack(seed_vectors)
    k = max(1, min(k, len(seeds)))
    rows = []
    for name, space in spaces.items():
        distance = 1.0 - space.vectors @ seeds.T
        for i, w in enumerate(space.words):
            own = [j for j, (n, sw) in enumerate(seed_ids) if sw == w]
            if own:
                distance[i, own] = np.inf
        nearest = np.sort(distance, axis=1)[:, :k]
        nearest = np.where(np.isfinite(nearest), nearest, np.nan)
        for i, w in enumerate(space.words):
            is_seed = entry_of(w.casefold(), entries) is not None
            if hide_seeds and is_seed:
                continue
            rows.append({"corpus": name, "word": w, "occurrences": space.counts.get(w, 0), "knn_amd": float(np.nanmean(nearest[i]))})
    frame = pd.DataFrame(rows).sort_values(["corpus", "knn_amd"]).reset_index(drop=True)
    pooled = frame.groupby("word").agg(occurrences=("occurrences", "sum"), knn_amd=("knn_amd", "mean"), corpora=("corpus", "size")).reset_index()
    pooled["corpus"] = "All corpora"
    frame["corpora"] = 1
    return pd.concat([pooled.sort_values("knn_amd"), frame], ignore_index=True)
