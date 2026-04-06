"""Dimensionality reduction for 2D visualisation.

These methods are used **only for visualisation** — metric computation
always operates on the full-dimensional embeddings (full or definition space).

All methods fit on the joint set of embeddings from all corpora so that
points share a common coordinate system.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def reduce_to_2d(
    embeddings: Tensor,
    method: str = "pca",
    **kwargs,
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D for scatter plots.

    Parameters
    ----------
    embeddings : Tensor
        Shape ``[N, D]``.  Should be the concatenation of embeddings from
        all corpora (so they share the same 2D coordinate space).
    method : str
        One of ``"pca"``, ``"umap"``, ``"tsne"``.
    **kwargs
        Passed to the underlying reducer (e.g. ``perplexity`` for t-SNE,
        ``n_neighbors`` for UMAP).

    Returns
    -------
    np.ndarray
        Shape ``[N, 2]``.
    """
    X = embeddings.detach().cpu().numpy() if isinstance(embeddings, Tensor) else embeddings

    if method == "pca":
        return _pca_2d(X, **kwargs)
    elif method == "umap":
        return _umap_2d(X, **kwargs)
    elif method == "tsne":
        return _tsne_2d(X, **kwargs)
    else:
        raise ValueError(f"Unknown reduction method: {method!r}. Use 'pca', 'umap', or 'tsne'.")


def _pca_2d(X: np.ndarray, **kwargs) -> np.ndarray:
    from sklearn.decomposition import PCA

    n_components = min(2, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, **kwargs)
    result = pca.fit_transform(X)
    # If only 1 component possible, pad with zeros
    if result.shape[1] == 1:
        result = np.hstack([result, np.zeros((result.shape[0], 1))])
    return result


def _umap_2d(X: np.ndarray, **kwargs) -> np.ndarray:
    from umap import UMAP

    defaults = {"n_components": 2, "random_state": 42}
    defaults.update(kwargs)
    reducer = UMAP(**defaults)
    return reducer.fit_transform(X)


def _tsne_2d(X: np.ndarray, **kwargs) -> np.ndarray:
    from sklearn.manifold import TSNE

    defaults = {"n_components": 2, "random_state": 42}
    # Adapt perplexity if N is small
    n = X.shape[0]
    if n <= 30:
        defaults["perplexity"] = max(2, n // 3)
    defaults.update(kwargs)
    reducer = TSNE(**defaults)
    return reducer.fit_transform(X)


def available_methods() -> list[str]:
    """Return names of available 2D reduction methods."""
    methods = ["pca"]  # always available
    try:
        import umap  # noqa: F401
        methods.append("umap")
    except ImportError:
        pass
    try:
        from sklearn.manifold import TSNE  # noqa: F401
        methods.append("tsne")
    except ImportError:
        pass
    return methods
