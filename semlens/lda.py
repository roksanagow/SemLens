"""Supervised views of all selected corpora, fitted in a shared coordinate system."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import Tensor


@dataclass
class LDAResult:
    coords_2d: np.ndarray
    lda_model: LinearDiscriminantAnalysis
    pca_residual: PCA | None
    lda_directions_normed: np.ndarray
    corpus_labels_used: list[str]
    feature_names: list[str] | None
    n_lda_components: int
    axis_labels: tuple[str, str] = ("LD1", "PC1 (residual)")


def lda_projection(embeddings: Tensor, corpus_labels: list[str], feature_names=None, *, corpus_order=None) -> LDAResult:
    X = embeddings.detach().cpu().numpy() if isinstance(embeddings, Tensor) else np.asarray(embeddings)
    labels = list(dict.fromkeys(corpus_order or corpus_labels))
    if len(labels) < 2:
        raise ValueError("LDA needs at least two corpora. Use PCA for a single corpus.")
    if len(corpus_labels) != len(X) or any(c not in labels for c in corpus_labels):
        raise ValueError("Corpus labels must match the usages.")
    if any(corpus_labels.count(c) < 2 for c in labels):
        raise ValueError("LDA needs at least two usages in each corpus. Use PCA for smaller samples.")
    y = np.array([labels.index(c) for c in corpus_labels])
    lda = LinearDiscriminantAnalysis(n_components=min(len(labels)-1, X.shape[1], 2))
    try:
        X_lda = lda.fit_transform(X, y)
    except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
        raise ValueError("LDA could not find a stable separating direction. Try PCA or more varied usages.") from exc
    if X_lda.shape[1] == 0:
        raise ValueError("LDA found no varying discriminant dimension. Use PCA for these data.")
    # Give a binary plot a consistent left-to-right orientation in user-selected order.
    if len(labels) == 2 and X_lda[y == 1, 0].mean() < X_lda[y == 0, 0].mean():
        lda.scalings_[:, 0] *= -1
        X_lda[:, 0] *= -1
    n_lda = X_lda.shape[1]
    directions, _ = np.linalg.qr(lda.scalings_[:, :n_lda])
    pca_res = None
    if n_lda >= 2:
        coords = X_lda[:, :2]
        axes = ("LD1", "LD2")
    else:
        residual = X - X @ directions @ directions.T
        pca_res = PCA(n_components=1)
        second = pca_res.fit_transform(residual)[:, 0] if X.shape[1] > 1 else np.zeros(len(X))
        if X.shape[1] == 1:
            pca_res.fit(residual)
        coords = np.column_stack([X_lda[:, 0], second])
        axes = ("LD1", "PC1 (residual)")
    return LDAResult(coords, lda, pca_res, directions, labels, feature_names, n_lda, axes)


def lda_transform_new_points(lda_result: LDAResult, embeddings) -> np.ndarray:
    X = embeddings.detach().cpu().numpy() if isinstance(embeddings, Tensor) else np.asarray(embeddings)
    transformed = lda_result.lda_model.transform(X)
    if lda_result.n_lda_components >= 2:
        return transformed[:, :2]
    directions = lda_result.lda_directions_normed
    residual = X - X @ directions @ directions.T
    second = lda_result.pca_residual.transform(residual)[:, 0] if X.shape[1] > 1 else np.zeros(len(X))
    return np.column_stack([transformed[:, 0], second])


def lda_definition_weights(lda_result: LDAResult, definition_labels=None) -> pd.DataFrame:
    """LD1 coefficients on distance features, not sense-frequency estimates.

    A positive coefficient means increasing that distance increases LD1.
    Corpus means and the plotted corpus positions are needed for interpretation.
    """
    weights = lda_result.lda_model.scalings_[:, 0]
    labels = definition_labels if definition_labels is not None else lda_result.feature_names
    if labels is None:
        labels = [f"dim_{i}" for i in range(len(weights))]
    if len(labels) != len(weights):
        raise ValueError("One label is required per LDA feature.")
    return pd.DataFrame({"definition": labels, "weight": weights, "abs_weight": np.abs(weights)}) \
        .sort_values("abs_weight", ascending=False).reset_index(drop=True)
