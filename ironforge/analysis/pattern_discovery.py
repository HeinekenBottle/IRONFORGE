"""Pattern discovery using UMAP for dimensionality reduction and HDBSCAN for clustering.

This module exposes a ``discover_patterns`` helper that standardizes numeric
features, computes UMAP embeddings, clusters the embeddings with HDBSCAN and
derives dependency structures via mutual information and Graphical Lasso.

The implementation is intentionally lightweight so it can operate on synthetic
or real session-level features within the constraints of IRONFORGE's discovery
workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.covariance import GraphicalLasso
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


@dataclass
class DiscoveryResult:
    """Container for outputs of :func:`discover_patterns`.

    Attributes:
        embeddings: Low-dimensional representation produced by UMAP.
        labels: Cluster labels from HDBSCAN (``-1`` denotes noise).
        mutual_information: Symmetric matrix of pairwise mutual information
            between features used for discovery.
        precision: Estimated precision (inverse covariance) matrix from the
            Graphical Lasso model.
    """

    embeddings: np.ndarray
    labels: np.ndarray
    mutual_information: np.ndarray
    precision: np.ndarray


def _compute_mutual_information(features: np.ndarray) -> np.ndarray:
    """Compute a symmetric mutual-information matrix for continuous features."""
    n_features = features.shape[1]
    mi_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi = mutual_info_regression(features[:, [i]], features[:, j])
            mi_matrix[i, j] = mi[0]
            mi_matrix[j, i] = mi[0]
    return mi_matrix


def discover_patterns(
    data: pd.DataFrame | np.ndarray,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    n_components: int = 10,
    cluster_min_size: int = 25,
    random_state: int = 42,
    alpha: float = 0.01,
) -> DiscoveryResult:
    """Run the full discovery pipeline on ``data``.

    Args:
        data: 2D array-like structure containing numeric features.
        n_neighbors: UMAP ``n_neighbors`` parameter controlling local/global
            balance.
        min_dist: UMAP ``min_dist`` parameter controlling cluster tightness.
        n_components: Target dimensionality for UMAP embeddings.
        cluster_min_size: Minimum cluster size for HDBSCAN.
        random_state: Random seed for reproducibility.
        alpha: Regularisation strength for the Graphical Lasso model.

    Returns:
        :class:`DiscoveryResult` with embeddings, cluster labels, mutual
        information matrix and precision matrix.
    """
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must be 2-dimensional")

    scaled = StandardScaler().fit_transform(array)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
    )
    embeddings = reducer.fit_transform(scaled)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_min_size)
    labels = clusterer.fit_predict(embeddings)

    mi_matrix = _compute_mutual_information(scaled)

    model = GraphicalLasso(alpha=alpha)
    model.fit(scaled)

    return DiscoveryResult(
        embeddings=embeddings,
        labels=labels,
        mutual_information=mi_matrix,
        precision=model.precision_,
    )
