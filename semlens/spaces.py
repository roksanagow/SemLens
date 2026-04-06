"""Representation space projections.

Two spaces are supported:

- **Full space** — the original high-dimensional encoder output (identity).
- **Definition space** — usage embeddings projected onto dictionary definitions
  via cosine distance, producing an interpretable K-dimensional representation
  where each dimension corresponds to a specific word sense.

These spaces are used for **metric computation** (in the full K or D
dimensions), *not* for 2D visualisation — see ``reduction.py`` for that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def project_to_definition_space(
    usage_embeddings: Tensor,
    definition_embeddings: Tensor,
) -> Tensor:
    """Project usage embeddings into definition-aligned space.

    For each usage embedding v and each definition embedding z_k, computes:

        φ_def(v) = (δ(v, z_1), ..., δ(v, z_K))

    where δ is cosine distance: δ(x, y) = 1 − cos(x, y).

    This follows the definition-space construction from Goworek &
    Dubossarsky (2026), Section 3.4.

    Parameters
    ----------
    usage_embeddings : Tensor
        Shape ``[N, D]`` — contextualised word embeddings from the encoder.
    definition_embeddings : Tensor
        Shape ``[K, D]`` — embeddings of the target word in each definition.

    Returns
    -------
    Tensor
        Shape ``[N, K]`` where entry ``(i, k)`` is the cosine distance
        from usage *i* to definition *k*.
    """
    normed_usage = F.normalize(usage_embeddings, dim=1)
    normed_defs = F.normalize(definition_embeddings, dim=1)

    cosine_sim = torch.mm(normed_usage, normed_defs.T)  # [N, K]
    cosine_dist = 1.0 - cosine_sim

    return cosine_dist


def split_by_corpus(
    embeddings: Tensor,
    corpus_indices: dict[str, list[int]],
    corpus_a: str,
    corpus_b: str,
) -> tuple[Tensor, Tensor]:
    """Extract embeddings for two specific corpora.

    Parameters
    ----------
    embeddings : Tensor
        Shape ``[N, D]`` (full or definition space).
    corpus_indices : dict
        ``{corpus_label: [indices]}`` into the embeddings tensor.
    corpus_a, corpus_b : str
        The two corpus labels to compare.

    Returns
    -------
    (embs_a, embs_b) : tuple of Tensors
        Embeddings for each corpus.
    """
    idx_a = corpus_indices[corpus_a]
    idx_b = corpus_indices[corpus_b]
    return embeddings[idx_a], embeddings[idx_b]
