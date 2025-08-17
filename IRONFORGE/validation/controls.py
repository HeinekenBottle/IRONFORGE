"""
Negative Controls for IRONFORGE Validation (Wave 4)
===================================================
Implements negative controls and synthetic shuffles to establish baselines
and detect spurious patterns in temporal graph data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def time_shuffle_edges(
    edge_index: np.ndarray, edge_times: np.ndarray, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle edge times across edges to break temporal signal.

    This negative control destroys temporal relationships while preserving
    the graph structure and edge feature distributions.

    Parameters
    ----------
    edge_index : np.ndarray
        Edge connectivity matrix of shape (2, num_edges).
    edge_times : np.ndarray
        Temporal timestamps for each edge.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (edge_index, shuffled_edge_times)

    Examples
    --------
    >>> edge_index = np.array([[0, 1, 2], [1, 2, 0]])
    >>> edge_times = np.array([100, 200, 300])
    >>> new_edge_index, shuffled_times = time_shuffle_edges(edge_index, edge_times, seed=42)
    >>> # Times are shuffled but edge connectivity preserved
    """
    rng = np.random.RandomState(seed)

    # Create a copy to avoid modifying original
    shuffled_times = edge_times.copy()

    # Shuffle the temporal component
    rng.shuffle(shuffled_times)

    return edge_index.copy(), shuffled_times


def label_permutation(labels: np.ndarray, seed: int = 7) -> np.ndarray:
    """Permute labels as a negative control baseline.

    This control destroys the relationship between features and labels
    while preserving label distribution.

    Parameters
    ----------
    labels : np.ndarray
        Original labels to permute.
    seed : int
        Random seed for reproducible permutation.

    Returns
    -------
    np.ndarray
        Permuted labels with same distribution but random assignment.

    Examples
    --------
    >>> labels = np.array([0, 0, 1, 1, 1])
    >>> permuted = label_permutation(labels, seed=42)
    >>> # Same number of 0s and 1s, but randomly assigned
    """
    rng = np.random.RandomState(seed)

    # Create a copy and shuffle
    permuted_labels = labels.copy()
    rng.shuffle(permuted_labels)

    return permuted_labels


def node_feature_shuffle(
    node_features: np.ndarray,
    feature_groups: dict[str, Sequence[int]] | None = None,
    seed: int = 7,
) -> np.ndarray:
    """Shuffle node features within or across feature groups.

    Parameters
    ----------
    node_features : np.ndarray
        Node feature matrix of shape (num_nodes, num_features).
    feature_groups : Dict[str, Sequence[int]], optional
        Groups of feature indices to shuffle independently.
        If None, shuffle all features together.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    np.ndarray
        Feature matrix with shuffled values.
    """
    rng = np.random.RandomState(seed)
    shuffled_features = node_features.copy()

    if feature_groups is None:
        # Shuffle all features independently
        for col in range(shuffled_features.shape[1]):
            rng.shuffle(shuffled_features[:, col])
    else:
        # Shuffle within each feature group
        for group_name, feature_indices in feature_groups.items():
            for feat_idx in feature_indices:
                if feat_idx < shuffled_features.shape[1]:
                    rng.shuffle(shuffled_features[:, feat_idx])

    return shuffled_features


def edge_direction_shuffle(edge_index: np.ndarray, seed: int = 7) -> np.ndarray:
    """Randomly flip edge directions to test directional sensitivity.

    Parameters
    ----------
    edge_index : np.ndarray
        Edge connectivity matrix of shape (2, num_edges).
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    np.ndarray
        Edge index with randomly flipped directions.
    """
    rng = np.random.RandomState(seed)
    shuffled_edges = edge_index.copy()

    # Randomly select edges to flip
    num_edges = shuffled_edges.shape[1]
    flip_mask = rng.random(num_edges) < 0.5

    # Flip selected edges by swapping source and target
    shuffled_edges[:, flip_mask] = shuffled_edges[[1, 0], :][:, flip_mask]

    return shuffled_edges


def temporal_block_shuffle(
    edge_times: np.ndarray, block_size_mins: int = 60, seed: int = 7
) -> np.ndarray:
    """Shuffle temporal blocks to preserve local structure but break global patterns.

    Parameters
    ----------
    edge_times : np.ndarray
        Temporal timestamps for edges.
    block_size_mins : int
        Size of temporal blocks in minutes.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    np.ndarray
        Edge times with shuffled temporal blocks.
    """
    rng = np.random.RandomState(seed)

    if len(edge_times) == 0:
        return edge_times.copy()

    # Sort times and create blocks
    sorted_indices = np.argsort(edge_times)
    sorted_times = edge_times[sorted_indices]

    # Find block boundaries
    time_range = sorted_times[-1] - sorted_times[0]
    if time_range <= 0:
        return edge_times.copy()

    num_blocks = max(1, int(time_range / block_size_mins))
    block_boundaries = np.linspace(sorted_times[0], sorted_times[-1], num_blocks + 1)

    # Assign each time to a block
    block_assignments = np.digitize(sorted_times, block_boundaries) - 1
    block_assignments = np.clip(block_assignments, 0, num_blocks - 1)

    # Create shuffled mapping
    unique_blocks = np.unique(block_assignments)
    shuffled_blocks = unique_blocks.copy()
    rng.shuffle(shuffled_blocks)
    block_mapping = dict(zip(unique_blocks, shuffled_blocks, strict=False))

    # Apply shuffling
    shuffled_assignments = np.array([block_mapping[b] for b in block_assignments])

    # Map back to original indices
    shuffled_times = edge_times.copy()
    for i, orig_idx in enumerate(sorted_indices):
        new_block = shuffled_assignments[i]
        orig_block = block_assignments[i]

        # Shift time to new block position
        block_offset = (new_block - orig_block) * block_size_mins
        shuffled_times[orig_idx] = edge_times[orig_idx] + block_offset

    return shuffled_times


def create_control_variants(
    edge_index: np.ndarray,
    edge_times: np.ndarray,
    node_features: np.ndarray,
    labels: np.ndarray,
    controls: Sequence[str],
    seed: int = 7,
) -> dict[str, dict[str, Any]]:
    """Create multiple negative control variants for comprehensive testing.

    Parameters
    ----------
    edge_index : np.ndarray
        Edge connectivity matrix.
    edge_times : np.ndarray
        Edge timestamps.
    node_features : np.ndarray
        Node feature matrix.
    labels : np.ndarray
        Target labels.
    controls : Sequence[str]
        List of control types to generate.
    seed : int
        Base random seed.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Control variants with modified data components.
    """
    variants = {}

    for i, control_type in enumerate(controls):
        control_seed = seed + i  # Ensure different seeds for each control

        if control_type == "time_shuffle":
            new_edge_index, new_edge_times = time_shuffle_edges(
                edge_index, edge_times, seed=control_seed
            )
            variants[control_type] = {
                "edge_index": new_edge_index,
                "edge_times": new_edge_times,
                "node_features": node_features.copy(),
                "labels": labels.copy(),
                "description": "Temporal signal destroyed via edge time shuffling",
            }

        elif control_type == "label_perm":
            variants[control_type] = {
                "edge_index": edge_index.copy(),
                "edge_times": edge_times.copy(),
                "node_features": node_features.copy(),
                "labels": label_permutation(labels, seed=control_seed),
                "description": "Feature-label relationship destroyed via label permutation",
            }

        elif control_type == "node_shuffle":
            variants[control_type] = {
                "edge_index": edge_index.copy(),
                "edge_times": edge_times.copy(),
                "node_features": node_feature_shuffle(node_features, seed=control_seed),
                "labels": labels.copy(),
                "description": "Node feature relationships destroyed via shuffling",
            }

        elif control_type == "edge_direction":
            variants[control_type] = {
                "edge_index": edge_direction_shuffle(edge_index, seed=control_seed),
                "edge_times": edge_times.copy(),
                "node_features": node_features.copy(),
                "labels": labels.copy(),
                "description": "Edge directionality randomized",
            }

        elif control_type == "temporal_blocks":
            variants[control_type] = {
                "edge_index": edge_index.copy(),
                "edge_times": temporal_block_shuffle(edge_times, seed=control_seed),
                "node_features": node_features.copy(),
                "labels": labels.copy(),
                "description": "Temporal structure disrupted via block shuffling",
            }

    return variants
