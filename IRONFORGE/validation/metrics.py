"""
Validation Metrics for IRONFORGE (Wave 4)
==========================================
Core metrics for evaluating temporal pattern discovery with emphasis on
stability, temporal consistency, and archaeological significance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def precision_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int = 20) -> float:
    """Calculate precision at top-k predictions.

    Parameters
    ----------
    y_true : Sequence[int]
        True binary labels (0 or 1).
    y_score : Sequence[float]
        Prediction scores (higher = more likely positive).
    k : int
        Number of top predictions to consider.

    Returns
    -------
    float
        Precision at k (0.0 to 1.0).

    Examples
    --------
    >>> y_true = [1, 0, 1, 0, 1, 0]
    >>> y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    >>> precision_at_k(y_true, y_score, k=3)
    0.6666666666666666
    """
    if k <= 0:
        return 0.0

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have same length")

    if len(y_true) == 0:
        return 0.0

    # Get top-k indices by score
    top_k_indices = np.argsort(y_score)[-k:][::-1]  # Descending order

    # Count true positives in top-k
    top_k_true = y_true[top_k_indices]
    true_positives = np.sum(top_k_true)

    return float(true_positives) / k


def temporal_auc(
    y_true: Sequence[int], y_score: Sequence[float], timestamps: Sequence[int]
) -> float:
    """AUC computed with chronological tie-breaking to reduce look-ahead bias.

    Parameters
    ----------
    y_true : Sequence[int]
        True binary labels.
    y_score : Sequence[float]
        Prediction scores.
    timestamps : Sequence[int]
        Timestamps for chronological ordering.

    Returns
    -------
    float
        Temporal AUC score (0.0 to 1.0).
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    timestamps = np.asarray(timestamps)

    if len(set([len(y_true), len(y_score), len(timestamps)])) > 1:
        raise ValueError("All input arrays must have same length")

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5  # Random performance for edge cases

    # Sort by timestamp to ensure chronological order
    time_order = np.argsort(timestamps)
    y_true_sorted = y_true[time_order]
    y_score_sorted = y_score[time_order]

    # For ties in scores, break by temporal order (earlier = lower rank)
    # This reduces look-ahead bias by prioritizing earlier predictions
    adjusted_scores = y_score_sorted + np.arange(len(y_score_sorted)) * 1e-10

    try:
        return roc_auc_score(y_true_sorted, adjusted_scores)
    except ValueError:
        # Handle edge case where all labels are the same
        return 0.5


def motif_half_life(hits_timestamps: Sequence[int]) -> float:
    """Estimate stability/decay using exponential fit on inter-hit intervals.

    Parameters
    ----------
    hits_timestamps : Sequence[int]
        Timestamps when patterns/motifs were detected.

    Returns
    -------
    float
        Estimated half-life in the same units as timestamps.
        Higher values indicate more stable/persistent patterns.

    Examples
    --------
    >>> hits = [100, 120, 140, 180, 250]  # Increasing intervals
    >>> half_life = motif_half_life(hits)
    >>> # Returns estimated half-life of pattern occurrence
    """
    hits_timestamps = np.asarray(hits_timestamps)

    if len(hits_timestamps) < 2:
        return float("inf")  # Can't estimate with < 2 points

    # Sort timestamps
    sorted_hits = np.sort(hits_timestamps)

    # Calculate inter-hit intervals
    intervals = np.diff(sorted_hits)

    if len(intervals) == 0:
        return float("inf")

    # Fit exponential decay to intervals
    # Model: interval_i = exp(decay_rate * i)
    # Taking log: log(interval_i) = decay_rate * i

    x = np.arange(len(intervals))
    y = np.log(intervals + 1e-6)  # Add small constant to avoid log(0)

    try:
        # Linear regression on log-transformed data
        slope, _, r_value, _, _ = stats.linregress(x, y)

        # Convert decay rate to half-life
        # Half-life = ln(2) / |decay_rate|
        if abs(slope) < 1e-10:
            return float("inf")  # No decay

        half_life = np.log(2) / abs(slope)
        return float(half_life)

    except (ValueError, ZeroDivisionError):
        # Fallback: use median interval as proxy
        return float(np.median(intervals))


def pattern_stability_score(
    y_score: Sequence[float], timestamps: Sequence[int], window_size: int = 60
) -> float:
    """Measure temporal stability of pattern scores across time windows.

    Parameters
    ----------
    y_score : Sequence[float]
        Pattern confidence scores.
    timestamps : Sequence[int]
        Corresponding timestamps.
    window_size : int
        Size of time windows for stability analysis.

    Returns
    -------
    float
        Stability score (0.0 = unstable, 1.0 = perfectly stable).
    """
    y_score = np.asarray(y_score)
    timestamps = np.asarray(timestamps)

    if len(y_score) != len(timestamps) or len(y_score) < 2:
        return 0.0

    # Sort by timestamp
    order = np.argsort(timestamps)
    sorted_scores = y_score[order]
    sorted_times = timestamps[order]

    # Create time windows
    time_range = sorted_times[-1] - sorted_times[0]
    if time_range <= 0:
        return 1.0  # Perfect stability for single time point

    num_windows = max(2, int(time_range / window_size))
    window_boundaries = np.linspace(sorted_times[0], sorted_times[-1], num_windows + 1)

    # Calculate scores for each window
    window_scores = []
    for i in range(num_windows):
        start_time = window_boundaries[i]
        end_time = window_boundaries[i + 1]

        mask = (sorted_times >= start_time) & (sorted_times < end_time)
        if i == num_windows - 1:  # Include endpoint for last window
            mask = (sorted_times >= start_time) & (sorted_times <= end_time)

        if np.any(mask):
            window_mean = np.mean(sorted_scores[mask])
            window_scores.append(window_mean)

    if len(window_scores) < 2:
        return 1.0

    # Calculate stability as 1 - coefficient of variation
    window_scores = np.array(window_scores)
    mean_score = np.mean(window_scores)

    if mean_score == 0:
        return 1.0 if np.std(window_scores) == 0 else 0.0

    cv = np.std(window_scores) / abs(mean_score)
    stability = 1.0 / (1.0 + cv)  # Transforms CV to [0, 1] scale

    return float(stability)


def archaeological_significance(
    pattern_scores: Sequence[float], pattern_types: Sequence[str], temporal_spans: Sequence[float]
) -> dict[str, float]:
    """Calculate archaeological significance metrics for discovered patterns.

    Parameters
    ----------
    pattern_scores : Sequence[float]
        Confidence scores for discovered patterns.
    pattern_types : Sequence[str]
        Types/categories of discovered patterns.
    temporal_spans : Sequence[float]
        Duration spans of each pattern.

    Returns
    -------
    Dict[str, float]
        Archaeological significance metrics.
    """
    pattern_scores = np.asarray(pattern_scores)
    pattern_types = np.asarray(pattern_types)
    temporal_spans = np.asarray(temporal_spans)

    if len(set([len(pattern_scores), len(pattern_types), len(temporal_spans)])) > 1:
        raise ValueError("All input arrays must have same length")

    if len(pattern_scores) == 0:
        return {
            "diversity_index": 0.0,
            "temporal_coverage": 0.0,
            "pattern_density": 0.0,
            "significance_weighted_score": 0.0,
        }

    # Diversity index (Shannon entropy of pattern types)
    unique_types, type_counts = np.unique(pattern_types, return_counts=True)
    type_probs = type_counts / len(pattern_types)
    diversity_index = -np.sum(type_probs * np.log(type_probs + 1e-10))

    # Temporal coverage (proportion of time with patterns)
    total_span = np.sum(temporal_spans)
    max_possible_span = np.max(temporal_spans) * len(temporal_spans)
    temporal_coverage = total_span / max(max_possible_span, 1e-10)

    # Pattern density (patterns per unit time)
    mean_span = np.mean(temporal_spans) if len(temporal_spans) > 0 else 1.0
    pattern_density = len(pattern_scores) / max(mean_span, 1e-10)

    # Significance-weighted score
    weights = temporal_spans / max(np.sum(temporal_spans), 1e-10)
    significance_weighted_score = np.sum(pattern_scores * weights)

    return {
        "diversity_index": float(diversity_index),
        "temporal_coverage": float(np.clip(temporal_coverage, 0.0, 1.0)),
        "pattern_density": float(pattern_density),
        "significance_weighted_score": float(significance_weighted_score),
    }


def compute_validation_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    timestamps: Sequence[int],
    pattern_metadata: dict | None = None,
    k_values: Sequence[int] = (5, 10, 20),
) -> dict[str, float]:
    """Compute comprehensive validation metrics for pattern discovery.

    Parameters
    ----------
    y_true : Sequence[int]
        True binary labels.
    y_score : Sequence[float]
        Prediction scores.
    timestamps : Sequence[int]
        Temporal timestamps.
    pattern_metadata : Dict, optional
        Additional pattern information for archaeological metrics.
    k_values : Sequence[int]
        K values for precision@k calculation.

    Returns
    -------
    Dict[str, float]
        Comprehensive metrics dictionary.
    """
    metrics = {}

    # Core metrics
    metrics["temporal_auc"] = temporal_auc(y_true, y_score, timestamps)

    # Precision at various k values
    for k in k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k=k)

    # Stability metrics
    metrics["pattern_stability"] = pattern_stability_score(y_score, timestamps)

    # Half-life analysis for positive predictions
    positive_mask = np.asarray(y_true) == 1
    if np.any(positive_mask):
        positive_timestamps = np.asarray(timestamps)[positive_mask]
        metrics["motif_half_life"] = motif_half_life(positive_timestamps)
    else:
        metrics["motif_half_life"] = float("inf")

    # Archaeological significance if metadata available
    if pattern_metadata:
        arch_metrics = archaeological_significance(
            pattern_metadata.get("scores", y_score),
            pattern_metadata.get("types", ["unknown"] * len(y_score)),
            pattern_metadata.get("spans", [1.0] * len(y_score)),
        )
        metrics.update(arch_metrics)

    return metrics
