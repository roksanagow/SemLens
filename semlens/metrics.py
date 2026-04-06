"""Lexical Semantic Change Detection metrics.

Implements the four metrics from Goworek & Dubossarsky (2026):

- **APD** — Average Pairwise Distance
- **PRT** — Prototype (centroid) cosine distance
- **AMD** — Average Minimum Distance (with directional decomposition)
- **SAMD** — Symmetric Average Minimum Distance (greedy one-to-one matching)

All metrics use cosine distance: δ(x, y) = 1 − cos(x, y).
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from torch import Tensor


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def calculate_apd(embs_1: Tensor, embs_2: Tensor) -> float:
    """Average Pairwise Distance between all cross-corpus usage pairs.

    APD(A, B) = (1 / |A||B|) Σ_{a∈A} Σ_{b∈B} δ(a, b)
    """
    e1 = torch.nn.functional.normalize(embs_1, dim=1)
    e2 = torch.nn.functional.normalize(embs_2, dim=1)
    pairwise_dist = 1.0 - torch.mm(e1, e2.T)  # [N1, N2]
    return pairwise_dist.mean().item()


def calculate_prt(embs_1: Tensor, embs_2: Tensor, eps: float = 1e-8) -> float:
    """Prototype Distance — cosine distance between corpus centroids.

    PRT(A, B) = δ(μ(A), μ(B))
    """
    c1 = embs_1.mean(dim=0)
    c2 = embs_2.mean(dim=0)
    c1 = c1 / (c1.norm() + eps)
    c2 = c2 / (c2.norm() + eps)
    cos_sim = torch.dot(c1, c2).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).item()


def calculate_directional_amd(
    embs_from: Tensor,
    embs_to: Tensor,
) -> float:
    """Directional AMD: average nearest-neighbour distance from one corpus to another.

    AMD(A → B) = (1/|A|) Σ_{a∈A} min_{b∈B} ||a − b||

    High AMD(A→B) = usages in A that cannot be matched in B
    (sense disappearance / narrowing if A is the earlier corpus).
    """
    dists = torch.cdist(embs_from, embs_to)  # [N_from, N_to]
    min_dists = dists.min(dim=1).values  # [N_from]
    return min_dists.mean().item()


def calculate_amd(embs_1: Tensor, embs_2: Tensor) -> float:
    """Symmetric AMD — average of both directional AMDs.

    AMD(A, B) = (AMD(A→B) + AMD(B→A)) / 2
    """
    d1 = calculate_directional_amd(embs_1, embs_2)
    d2 = calculate_directional_amd(embs_2, embs_1)
    return (d1 + d2) / 2


def calculate_samd(
    embs_1: Tensor,
    embs_2: Tensor,
) -> float:
    """Symmetric Average Minimum Distance via greedy one-to-one matching.

    Greedily selects the smallest remaining pairwise distance, removes the
    matched pair, and repeats.  Matches min(|A|, |B|) pairs.
    """
    assert embs_1.dim() == 2 and embs_2.dim() == 2
    n1, d1 = embs_1.shape
    n2, d2 = embs_2.shape
    assert d1 == d2

    if n1 == 0 or n2 == 0:
        return float("nan")

    dist = torch.cdist(embs_1, embs_2)  # [n1, n2]

    used1 = torch.zeros(n1, dtype=torch.bool)
    used2 = torch.zeros(n2, dtype=torch.bool)
    matched: list[float] = []
    num_pairs = min(n1, n2)

    for _ in range(num_pairs):
        masked = dist.clone()
        if used1.any():
            masked[used1, :] = float("inf")
        if used2.any():
            masked[:, used2] = float("inf")

        flat_idx = torch.argmin(masked)
        i = flat_idx // n2
        j = flat_idx % n2
        val = masked[i, j]

        if torch.isinf(val):
            break

        matched.append(val.item())
        used1[i] = True
        used2[j] = True

    if not matched:
        return float("nan")
    return float(np.mean(matched))


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    embs_c1: Tensor,
    embs_c2: Tensor,
    corpus_labels: tuple[str, str] = ("corpus_1", "corpus_2"),
) -> dict[str, float]:
    """Compute all LSCD metrics between two corpora.

    Parameters
    ----------
    embs_c1, embs_c2 : Tensor
        Usage embeddings for the two corpora, shape ``[N, D]``.
    corpus_labels : tuple of str
        Names of the two corpora, used as keys in directional AMD.

    Returns
    -------
    dict[str, float]
        Keys: ``'apd'``, ``'prt'``, ``'amd'``, ``'samd'``,
        ``'amd_{c1}_to_{c2}'``, ``'amd_{c2}_to_{c1}'``.
    """
    c1_name, c2_name = corpus_labels
    return {
        "apd": calculate_apd(embs_c1, embs_c2),
        "prt": calculate_prt(embs_c1, embs_c2),
        "amd": calculate_amd(embs_c1, embs_c2),
        "samd": calculate_samd(embs_c1, embs_c2),
        f"amd_{c1_name}_to_{c2_name}": calculate_directional_amd(embs_c1, embs_c2),
        f"amd_{c2_name}_to_{c1_name}": calculate_directional_amd(embs_c2, embs_c1),
    }


# ---------------------------------------------------------------------------
# Per-definition metric breakdown
# ---------------------------------------------------------------------------

def compute_per_definition_metrics(
    def_space_c1: Tensor,
    def_space_c2: Tensor,
    definition_labels: list[str],
    corpus_labels: tuple[str, str] = ("corpus_1", "corpus_2"),
    sort_by: str = "amd",
) -> pd.DataFrame:
    """Compute LSCD metrics along each definition dimension independently.

    For each definition k, the k-th coordinate (a scalar per usage) is
    extracted from the definition-space projection.  All metrics are then
    computed on these 1D distributions.

    Parameters
    ----------
    def_space_c1, def_space_c2 : Tensor
        Definition-space projections, shape ``[N1, K]`` and ``[N2, K]``.
    definition_labels : list[str]
        Human-readable labels for each definition (length K).
    corpus_labels : tuple of str
        Names of the two corpora.
    sort_by : str
        Column to sort by (descending absolute value).

    Returns
    -------
    pd.DataFrame
        Rows = definitions, columns = metric values.
    """
    K = def_space_c1.shape[1]
    assert K == def_space_c2.shape[1] == len(definition_labels)

    c1_name, c2_name = corpus_labels
    records: list[dict] = []

    for k in range(K):
        # Extract k-th dimension as [N, 1] tensors (1D embeddings)
        col_c1 = def_space_c1[:, k : k + 1]
        col_c2 = def_space_c2[:, k : k + 1]

        record = {
            "definition": definition_labels[k],
            "dim": k,
            "amd": calculate_amd(col_c1, col_c2),
            "samd": calculate_samd(col_c1, col_c2),
            f"amd_{c1_name}_to_{c2_name}": calculate_directional_amd(col_c1, col_c2),
            f"amd_{c2_name}_to_{c1_name}": calculate_directional_amd(col_c2, col_c1),
            # Average cosine distance from each corpus's usages to this definition.
            # Small value = usages in that corpus are semantically close to this sense.
            f"avg_cos_dist_{c1_name}": col_c1.mean().item(),
            f"avg_cos_dist_{c2_name}": col_c2.mean().item(),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Sort by absolute value of the chosen metric
    if sort_by in df.columns:
        df = df.reindex(
            df[sort_by].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

    return df
