"""LSCD scores, reproducible sampling, and the usage matches behind AMD."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import Tensor


def _validate_pair(a: Tensor, b: Tensor):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] == 0:
        raise ValueError("Expected two [usages, dimensions] tensors with matching dimensions.")
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Each compared corpus needs at least one usage.")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Embeddings must contain only finite values.")


def _pairwise_distance(embs_1: Tensor, embs_2: Tensor, distance: str = "cosine") -> Tensor:
    _validate_pair(embs_1, embs_2)
    if distance == "cosine":
        e1 = torch.nn.functional.normalize(embs_1, dim=1)
        e2 = torch.nn.functional.normalize(embs_2, dim=1)
        return (1.0 - e1 @ e2.T).clamp(0.0, 2.0)
    if distance in {"euclidean", "absolute"}:
        if distance == "absolute" and embs_1.shape[1] != 1:
            raise ValueError("Absolute distance is for single definition dimensions.")
        return torch.cdist(embs_1, embs_2)
    raise ValueError(f"Unsupported distance metric: {distance}")


def calculate_apd(embs_1: Tensor, embs_2: Tensor, distance: str = "cosine") -> float:
    return _pairwise_distance(embs_1, embs_2, distance).mean().item()


def calculate_prt(embs_1: Tensor, embs_2: Tensor, eps: float = 1e-8, distance: str = "cosine") -> float:
    _validate_pair(embs_1, embs_2)
    c1, c2 = embs_1.mean(0), embs_2.mean(0)
    if distance == "cosine":
        if c1.norm() < eps or c2.norm() < eps:
            return float("nan")  # a zero centroid has no direction
        return (1.0 - torch.dot(c1 / c1.norm(), c2 / c2.norm()).clamp(-1, 1)).item()
    if distance in {"euclidean", "absolute"}:
        return torch.linalg.vector_norm(c1 - c2).item()
    raise ValueError(f"Unsupported distance metric: {distance}")


def calculate_directional_amd(embs_from: Tensor, embs_to: Tensor, distance: str = "cosine") -> float:
    """Mean distance to the closest counterpart in the other corpus."""
    return _pairwise_distance(embs_from, embs_to, distance).min(1).values.mean().item()


def calculate_amd(embs_1: Tensor, embs_2: Tensor, distance: str = "cosine") -> float:
    d = _pairwise_distance(embs_1, embs_2, distance)
    return ((d.min(1).values.mean() + d.min(0).values.mean()) / 2).item()


def _samd_from_distances(dist: Tensor, chunk: int = 4096) -> float:
    """Greedy global-minimum matching, with deterministic row-major tie breaks.

    Pairs are taken in order of distance. They are read a chunk at a time, and
    pairs whose usages were already matched before the chunk are dropped all
    at once, so only the remaining few are checked one by one; the result is
    the same as checking every pair in turn.
    """
    values = dist.detach().cpu().numpy()
    n1, n2 = values.shape
    used1, used2 = np.zeros(n1, dtype=bool), np.zeros(n2, dtype=bool)
    target = min(n1, n2)
    matched = []
    # Sorting once avoids cloning/scanning the full matrix for every match.
    rows, cols = np.divmod(np.argsort(values, axis=None, kind="stable"), n2)
    for start in range(0, rows.size, chunk):
        r, c = rows[start:start + chunk], cols[start:start + chunk]
        open_ = ~used1[r] & ~used2[c]
        for i, j in zip(r[open_].tolist(), c[open_].tolist()):
            if used1[i] or used2[j]:
                continue
            matched.append(float(values[i, j]))
            used1[i] = used2[j] = True
            if len(matched) == target:
                return float(np.mean(matched))
    return float(np.mean(matched))


def calculate_samd(embs_1: Tensor, embs_2: Tensor, distance: str = "cosine") -> float:
    """Greedy one-to-one matching; unequal inputs leave surplus usages unmatched."""
    return _samd_from_distances(_pairwise_distance(embs_1, embs_2, distance))


def compute_all_metrics(embs_c1: Tensor, embs_c2: Tensor,
                        corpus_labels: tuple[str, str] = ("corpus_1", "corpus_2"),
                        distance: str = "cosine") -> dict[str, float]:
    c1, c2 = corpus_labels
    d = _pairwise_distance(embs_c1, embs_c2, distance)
    d1, d2 = d.min(1).values.mean().item(), d.min(0).values.mean().item()
    return {"apd": d.mean().item(), "prt": calculate_prt(embs_c1, embs_c2, distance=distance),
            "amd": (d1 + d2) / 2, "samd": _samd_from_distances(d),
            f"amd_{c1}_to_{c2}": d1, f"amd_{c2}_to_{c1}": d2}


def compute_per_definition_metrics(def_space_c1: Tensor, def_space_c2: Tensor, definition_labels: list[str],
                                   corpus_labels: tuple[str, str] = ("corpus_1", "corpus_2"),
                                   sort_by: str = "amd", distance: str = "absolute") -> pd.DataFrame:
    """Compare scalar definition distances using |x-y|, independently of vector distance.

    ``distance`` remains as a compatibility argument; cosine is never applied to
    these scalar features. The projection itself still uses cosine distance.
    """
    _validate_pair(def_space_c1, def_space_c2)
    if def_space_c1.shape[1] != len(definition_labels):
        raise ValueError("One label is required per definition dimension.")
    c1, c2 = corpus_labels
    records = []
    for k, label in enumerate(definition_labels):
        a, b = def_space_c1[:, k:k+1], def_space_c2[:, k:k+1]
        metrics = compute_all_metrics(a, b, corpus_labels, distance="absolute")
        # On one scalar dimension PRT is the absolute difference between the corpus means.
        records.append({"definition": label, "dim": k, **metrics,
                        f"avg_cos_dist_{c1}": a.mean().item(), f"avg_cos_dist_{c2}": b.mean().item(),
                        "mean_distance_difference": b.mean().item() - a.mean().item()})
    df = pd.DataFrame(records)
    if sort_by in df:
        df = df.iloc[np.argsort(-np.abs(df[sort_by].to_numpy()), kind="stable")].reset_index(drop=True)
    return df


def sample_indices(n1: int, n2: int, *, mode: str = "balanced", repetitions: int = 20, seed: int = 42):
    """Indices local to each corpus. Each draw is without replacement."""
    if min(n1, n2) < 1:
        raise ValueError("Each compared corpus needs at least one usage.")
    if mode not in {"balanced", "unequal", "repeated"}:
        raise ValueError("Unknown sampling mode.")
    if repetitions < 1:
        raise ValueError("Use at least one repetition.")
    if mode == "unequal":
        return [(np.arange(n1), np.arange(n2))]
    rng = np.random.default_rng(seed)
    n = min(n1, n2)
    draws = []
    for _ in range(repetitions if mode == "repeated" else 1):
        draws.append(tuple(np.sort(rng.choice(size, n, replace=False)) if size > n else np.arange(size)
                           for size in (n1, n2)))
    return draws


@dataclass
class SampledComparison:
    scores: pd.DataFrame
    draws: list[tuple[np.ndarray, np.ndarray]]

    @property
    def summary(self) -> pd.DataFrame:
        return self.scores.agg(["mean", "min", "max", "std"]).T.fillna({"std": 0.0})


def compute_sampled_metrics(a: Tensor, b: Tensor, corpus_labels=("corpus_1", "corpus_2"), *,
                            distance="cosine", mode="balanced", repetitions=20, seed=42) -> SampledComparison:
    draws = sample_indices(len(a), len(b), mode=mode, repetitions=repetitions, seed=seed)
    scores = [compute_all_metrics(a[i], b[j], corpus_labels, distance) for i, j in draws]
    return SampledComparison(pd.DataFrame(scores), draws)


def within_amd(x: Tensor, distance: str = "cosine") -> float:
    """AMD within one group, leaving each usage out: its distance to its closest *other* usage, averaged.

    NaN with fewer than two usages.
    """
    if len(x) < 2:
        return float("nan")
    d = _pairwise_distance(x, x, distance).clone()
    d.fill_diagonal_(float("inf"))
    return d.min(1).values.mean().item()


def within_amd_baseline(a: Tensor, b: Tensor, draws, distance: str = "cosine") -> tuple[float, float, float]:
    """AMD within each of two groups, on the usages sampled in each draw and averaged over the draws, and the mean of the two.

    The baseline for an AMD between the groups computed on the same draws: an AMD
    near this is no larger than the variation inside each group.
    """
    in_a = float(np.mean([within_amd(a[i], distance) for i, _ in draws]))
    in_b = float(np.mean([within_amd(b[j], distance) for _, j in draws]))
    return in_a, in_b, (in_a + in_b) / 2


def nearest_counterparts(a: Tensor, b: Tensor, distance="cosine") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every usage's nearest counterpart in both directions, weakest first."""
    d = _pairwise_distance(a, b, distance)
    def records(matrix):
        values, indices = matrix.min(1)
        return pd.DataFrame({"source_index": np.arange(len(matrix)),
                             "counterpart_index": indices.cpu().numpy(), "distance": values.cpu().numpy()}) \
            .sort_values("distance", ascending=False, kind="stable").reset_index(drop=True)
    return records(d), records(d.T)


def closest_usages(query: Tensor, candidates: Tensor, k: int = 3, distance="cosine") -> tuple[np.ndarray, np.ndarray]:
    """Positions and distances of the ``k`` candidates closest to one usage, nearest first."""
    d = _pairwise_distance(query.reshape(1, -1), candidates, distance)[0]
    order = torch.argsort(d, stable=True)[:max(1, int(k))]
    return order.cpu().numpy(), d[order].cpu().numpy()
