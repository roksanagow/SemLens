"""Linear Discriminant Analysis for corpus separation and interpretability.

With 2 corpora, LDA yields exactly 1 discriminant dimension (LD1).
The visualisation uses:

- **x-axis**: LD1 — the direction that maximally separates the two corpora
- **y-axis**: PC1 of the residual — the first principal component after
  projecting out LD1, capturing the most remaining variance

When applied in definition space, the LD1 weights are directly interpretable:
positive weights indicate definitions more strongly associated with corpus 2
(later), and negative weights with corpus 1 (earlier).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import Tensor


@dataclass
class LDAResult:
    """Results of LDA projection for visualisation and interpretation."""

    coords_2d: np.ndarray          # [N, 2] — (LD1, PC1-of-residual)
    lda_model: LinearDiscriminantAnalysis
    pca_residual: PCA
    lda_directions_normed: np.ndarray  # [D, n_lda] — normalised LDA directions
    corpus_labels_used: list[str]  # the labels used for fitting
    feature_names: list[str] | None  # e.g. definition labels, if in def space
    n_lda_components: int          # 1 for 2-corpus comparison


def lda_projection(
    embeddings: Tensor,
    corpus_labels: list[str],
    feature_names: list[str] | None = None,
) -> LDAResult:
    """Project embeddings to 2D via LDA + residual PCA.

    Parameters
    ----------
    embeddings : Tensor
        Shape ``[N, D]`` — can be full-space or definition-space embeddings.
    corpus_labels : list[str]
        Length N, one corpus label per usage (e.g. ``["old", "old", "new", ...]``).
    feature_names : list[str] or None
        Names for each embedding dimension.  Required for interpretable LDA
        weights in definition space (one name per definition).

    Returns
    -------
    LDAResult
        Contains 2D coordinates, the fitted LDA model, and metadata.
    """
    X = embeddings.detach().cpu().numpy() if isinstance(embeddings, Tensor) else embeddings
    y = np.array(corpus_labels)

    unique_labels = sorted(set(corpus_labels))
    n_classes = len(unique_labels)

    # LDA gives min(n_classes - 1, n_features) components
    n_lda = min(n_classes - 1, X.shape[1])

    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    X_lda = lda.fit_transform(X, y)  # [N, n_lda]

    # Residual: project out the LDA direction(s) and take PC1.
    # lda.scalings_ are NOT orthonormal, so we must normalise each
    # column before computing the projection matrix.
    lda_directions = lda.scalings_[:, :n_lda].copy()  # [D, n_lda]
    # Normalise each direction to unit length
    for i in range(n_lda):
        norm = np.linalg.norm(lda_directions[:, i])
        if norm > 1e-12:
            lda_directions[:, i] /= norm

    # Orthogonal projection matrix: P = V @ V.T (since columns are now unit)
    # For n_lda=1 this is just the outer product of the normalised direction
    X_proj = X @ lda_directions @ lda_directions.T  # projection onto LDA subspace
    X_residual = X - X_proj

    pca_res = PCA(n_components=1)
    X_pc1 = pca_res.fit_transform(X_residual)  # [N, 1]

    # Combine: LD1 on x-axis, PC1(residual) on y-axis
    coords_2d = np.column_stack([X_lda[:, 0], X_pc1[:, 0]])

    return LDAResult(
        coords_2d=coords_2d,
        lda_model=lda,
        pca_residual=pca_res,
        lda_directions_normed=lda_directions,
        corpus_labels_used=unique_labels,
        feature_names=feature_names,
        n_lda_components=n_lda,
    )


def lda_transform_new_points(
    lda_result: LDAResult,
    embeddings: Tensor | np.ndarray,
) -> np.ndarray:
    """Project new points through an existing LDA transform.

    Uses the same LDA model and residual PCA fitted during
    ``lda_projection`` to map new embeddings (e.g. definition anchors)
    into the same 2D space.

    Returns
    -------
    np.ndarray
        Shape ``[N_new, 2]``.
    """
    X = embeddings.detach().cpu().numpy() if isinstance(embeddings, Tensor) else embeddings

    # LD1 coordinate
    X_lda = lda_result.lda_model.transform(X)  # [N_new, n_lda]

    # Residual PC1 coordinate
    directions = lda_result.lda_directions_normed
    X_proj = X @ directions @ directions.T
    X_residual = X - X_proj
    X_pc1 = lda_result.pca_residual.transform(X_residual)  # [N_new, 1]

    return np.column_stack([X_lda[:, 0], X_pc1[:, 0]])


def lda_definition_weights(
    lda_result: LDAResult,
    definition_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Extract LD1 weights for each dimension (interpretable in definition space).

    The weights come from the LDA ``scalings_`` matrix (the transformation
    that projects from the original D-dimensional space to the discriminant
    axis).  In definition space, each dimension corresponds to a specific
    dictionary definition, so the weights show which senses most strongly
    differentiate the two corpora.

    Parameters
    ----------
    lda_result : LDAResult
        From ``lda_projection()``.
    definition_labels : list[str] or None
        Human-readable labels for each dimension.  If None, uses the
        ``feature_names`` stored in the LDAResult, or generic indices.

    Returns
    -------
    pd.DataFrame
        Columns: ``'definition'``, ``'weight'``, ``'abs_weight'``.
        Sorted by ``abs_weight`` descending.

        Convention:
        - Positive weight → stronger association with corpus 2 (later/second)
        - Negative weight → stronger association with corpus 1 (earlier/first)
    """
    lda = lda_result.lda_model
    weights = lda.scalings_[:, 0]  # LD1 weights, shape [D]

    if definition_labels is None:
        definition_labels = lda_result.feature_names

    if definition_labels is None:
        definition_labels = [f"dim_{i}" for i in range(len(weights))]

    if len(definition_labels) != len(weights):
        raise ValueError(
            f"Number of labels ({len(definition_labels)}) does not match "
            f"number of LDA weights ({len(weights)})."
        )

    df = pd.DataFrame({
        "definition": definition_labels,
        "weight": weights,
        "abs_weight": np.abs(weights),
    })

    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
