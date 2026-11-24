"""Scoring words by how close their usages come to the seed usages.

For each sampled occurrence of a candidate word, the distances to its ``k``
nearest seed usages are found; their mean is the occurrence's kNN distance
(kNN-AMD for one usage). A word then gets two scores in each corpus:

- **kNN-AMD**: the mean kNN distance over its occurrences. Lower is closer.
- **Share near the concept**: the share of its occurrences whose kNN distance
  is within a threshold. The threshold is calibrated on the seeds themselves:
  by default, a usage counts as near when it is at least as close to the seeds
  as a typical seed (the median) is to the seeds of other words. A word whose
  minority sense fits the concept shows up here even when its average does not.

Only the ``k`` nearest seeds of each occurrence are kept, so adding seeds later
is incremental: only the new seeds are compared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..words import entry_of

ALL_CORPORA = "All corpora"


def _device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Neighbours:
    """For each candidate occurrence, its ``k`` nearest seeds: distances (ascending) and seed positions."""
    distances: np.ndarray   # [occurrences, k], float32
    seeds: np.ndarray       # [occurrences, k], int32

    @property
    def knn_distance(self) -> np.ndarray:
        """Each occurrence's mean distance to its nearest seeds."""
        return np.nanmean(np.where(np.isfinite(self.distances), self.distances, np.nan), axis=1)


def nearest_seeds(candidates, seeds: np.ndarray, k: int = 10, *, own_seed: np.ndarray | None = None,
                  seed_offset: int = 0, chunk: int = 8192, device: str | None = None, progress_callback=None) -> Neighbours:
    """The ``k`` nearest seeds of every candidate, by cosine distance.

    ``candidates`` is a callable returning unit-length vectors for a slice of
    rows, plus its length (``candidates.rows``), so that millions of vectors
    are read from disk a chunk at a time. ``own_seed[i]`` is the seed position
    of candidate ``i`` when it is itself a seed (or -1), so it is not its own
    neighbour.
    """
    dev = _device(device)
    m = len(seeds)
    if m == 0:
        raise ValueError("Add at least one seed.")
    # Always k columns: with fewer seeds than k, the rest stay empty (infinite
    # distance), so seeds added later fill them as a full computation would.
    k = max(1, int(k))
    found = min(k, m)
    seed_tensor = torch.from_numpy(np.ascontiguousarray(seeds, dtype=np.float32)).to(dev)
    n = candidates.rows
    out_d = np.full((n, k), np.inf, dtype=np.float32)
    out_i = np.full((n, k), -1, dtype=np.int32)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        block = torch.from_numpy(np.ascontiguousarray(candidates(slice(start, stop)), dtype=np.float32)).to(dev)
        distance = 1.0 - block @ seed_tensor.T
        if own_seed is not None:
            own = torch.from_numpy(own_seed[start:stop].astype(np.int64) - seed_offset)
            hit = (own >= 0) & (own < m)
            rows = torch.nonzero(hit).flatten()
            if len(rows):
                distance[rows.to(dev), own[rows].to(dev)] = float("inf")
        values, index = torch.topk(distance, found, dim=1, largest=False)
        out_d[start:stop, :found] = values.float().cpu().numpy()
        out_i[start:stop, :found] = index.int().cpu().numpy() + seed_offset
        if progress_callback:
            progress_callback(stop, n)
    return Neighbours(out_d, out_i)


def add_seeds(neighbours: Neighbours, candidates, new_seeds: np.ndarray, first_new: int, *,
              own_seed: np.ndarray | None = None, chunk: int = 8192, device: str | None = None) -> Neighbours:
    """Neighbours after adding seeds at positions ``first_new`` onwards: only the new seeds are compared."""
    k = neighbours.distances.shape[1]
    fresh = nearest_seeds(candidates, new_seeds, k, own_seed=own_seed, seed_offset=first_new, chunk=chunk, device=device)
    distances = np.concatenate([neighbours.distances, fresh.distances], axis=1)
    seeds = np.concatenate([neighbours.seeds, fresh.seeds], axis=1)
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return Neighbours(np.take_along_axis(distances, order, axis=1), np.take_along_axis(seeds, order, axis=1))


def calibrate_threshold(seeds: np.ndarray, k: int = 10, percentile: float = 50.0, groups=None) -> float | None:
    """The kNN distance that ``percentile`` % of the seed usages reach among the other seeds.

    With ``groups`` (the word of each seed), a seed is compared only with the
    seeds of other words: uses of one word resemble each other more than any
    other word's uses can, so a threshold set among them is out of reach for
    every other word. With the default (the median), a usage counts as near
    the concept when it is at least as close to the seeds as a typical seed
    is to the seeds of other words.
    Without two different words there is no such threshold, and None is returned.
    """
    m = len(seeds)
    if m < 2:
        return None
    distance = 1.0 - seeds.astype(np.float32) @ seeds.astype(np.float32).T
    if groups is not None:
        groups = np.asarray([str(g).casefold() for g in groups])
        if len(set(groups.tolist())) < 2:
            return None
        distance[groups[:, None] == groups[None, :]] = np.inf
    else:
        np.fill_diagonal(distance, np.inf)
    available = np.isfinite(distance).sum(axis=1)
    k = max(1, min(k, int(available.min())))
    nearest = np.sort(distance, axis=1)[:, :k].mean(axis=1)
    return float(np.percentile(nearest, percentile))


def rows_for_words(sample, entries) -> np.ndarray:
    """Sampled occurrences of the given words; entries may end in ``*`` for any ending."""
    entries = [e.strip().casefold() for e in entries if e.strip()]
    if not entries:
        return np.zeros(0, dtype=np.int64)
    matching = np.array([entry_of(w.casefold(), entries) is not None for w in sample.words], dtype=bool)
    return np.flatnonzero(matching[sample.word])


def background_rows(sample, valid: np.ndarray, size: int = 3000, seed: int = 0, exclude_rows=()) -> np.ndarray:
    """A random sample of usages, for :func:`score_words`'s background correction."""
    pool = np.setdiff1d(np.flatnonzero(valid), np.asarray(exclude_rows, dtype=np.int64))
    if not len(pool):
        return pool
    return np.sort(np.random.default_rng(seed).choice(pool, size=min(size, len(pool)), replace=False))


def score_words(sample, neighbours: Neighbours, threshold: float | None, *, valid: np.ndarray | None = None,
                hide_words=(), background: Neighbours | None = None, pooled: bool = False) -> pd.DataFrame:
    """One row per word and corpus: sampled occurrences, kNN-AMD, share near the concept, and all occurrences.

    With ``background`` (each occurrence's nearest usages from a random sample
    of all usages), each occurrence's distance to the seeds is reduced by its
    distance to usages in general. Words close to everything (*lot*, *kind*)
    then lose their advantage; a usage counts as near the concept when it is
    closer to the seeds than to usages in general.

    With ``pooled``, each word gets one row over all corpora (corpus "All corpora").
    """
    distance = neighbours.knn_distance
    if background is not None:
        distance = distance - background.knn_distance
        threshold = 0.0
    keep = np.isfinite(distance) if valid is None else (valid & np.isfinite(distance))
    hidden = {w.casefold() for w in hide_words}
    frame = pd.DataFrame({"corpus": sample.corpus[keep], "word": sample.word[keep], "knn": distance[keep]})
    if threshold is not None:
        frame["near"] = frame["knn"] < threshold if background is not None else frame["knn"] <= threshold
    else:
        frame["near"] = np.nan
    if pooled:
        frame["corpus"] = -1
    grouped = frame.groupby(["corpus", "word"]).agg(sampled=("knn", "size"), knn_amd=("knn", "mean"), share_near=("near", "mean"))
    grouped = grouped.reset_index()
    grouped["corpus"] = [ALL_CORPORA if int(c) < 0 else sample.corpora[int(c)] for c in grouped["corpus"]]
    grouped["word"] = [sample.words[int(w)] for w in grouped["word"]]
    grouped["occurrences"] = [sum(sample.counts.get(c, {}).get(w, 0) for c in sample.corpora) if c == ALL_CORPORA
                              else sample.counts.get(c, {}).get(w, 0) for c, w in zip(grouped["corpus"], grouped["word"])]
    if hidden:
        grouped = grouped[~grouped["word"].str.casefold().isin(hidden)]
    return grouped.sort_values(["corpus", "knn_amd"]).reset_index(drop=True)


class StoreCandidates:
    """Unit-length vectors of a store's occurrences, read a chunk at a time."""

    def __init__(self, store):
        self.store = store
        self.rows = len(store.sample)

    def __call__(self, rows):
        return self.store.normalised(rows)


@dataclass
class Scoring:
    """Neighbours of every occurrence among a list of seeds, kept so that later seeds can be added."""
    keys: tuple
    k: int
    neighbours: Neighbours
    threshold: float | None


def own_seed_positions(rows: int, keys) -> np.ndarray:
    """For each occurrence, its position among the seeds when it is one, else -1."""
    own = np.full(rows, -1, dtype=np.int64)
    for position, (kind, value) in enumerate(keys):
        if kind == "row":
            own[int(value)] = position
    return own


def update_scoring(candidates, keys, vectors: np.ndarray, k: int, previous: Scoring | None = None, *, groups=None,
                   device: str | None = None, progress_callback=None, calibration_size: int = 4000, seed: int = 0) -> Scoring:
    """Neighbours for these seeds; only the seeds added since ``previous`` are compared when it is a prefix of them.

    ``groups`` gives each seed's word, for the threshold (see :func:`calibrate_threshold`).
    """
    keys = tuple(keys)
    own = own_seed_positions(candidates.rows, keys)
    if previous is not None and previous.k == k and len(previous.keys) < len(keys) and keys[:len(previous.keys)] == previous.keys:
        first = len(previous.keys)
        neighbours = add_seeds(previous.neighbours, candidates, vectors[first:], first, own_seed=own, device=device)
    else:
        neighbours = nearest_seeds(candidates, vectors, k, own_seed=own, device=device, progress_callback=progress_callback)
    chosen = np.arange(len(vectors))
    if len(vectors) > calibration_size:
        chosen = np.sort(np.random.default_rng(seed).choice(len(vectors), calibration_size, replace=False))
    return Scoring(keys, k, neighbours, calibrate_threshold(vectors[chosen], k, groups=None if groups is None else np.asarray(groups)[chosen]))


def knn_cost(rows: int, seeds: int, dim: int, k: int, device: str):
    """Time and memory for comparing every occurrence with every seed."""
    from ..resources import ROUGH_FLOPS, Cost

    flops = 2.0 * rows * seeds * dim
    read = rows * dim * 2 / 1e9          # 16-bit vectors read from disk, at about 1 GB/s
    memory = rows * k * 8 + 8192 * (seeds + dim) * 4 * 3 + seeds * dim * 4
    return Cost("Comparing every occurrence with the seeds", memory_bytes=memory,
                seconds=flops / ROUGH_FLOPS.get(device, ROUGH_FLOPS["cpu"]) + read, rough_time=True,
                details=[f"{rows:,} occurrences × {seeds:,} seeds"])
