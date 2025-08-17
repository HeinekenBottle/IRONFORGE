"""Unit tests for validation metrics (Wave 4)."""

import numpy as np
import pytest

from ironforge.validation.metrics import (
    archaeological_significance,
    compute_validation_metrics,
    motif_half_life,
    pattern_stability_score,
    precision_at_k,
    temporal_auc,
)


class TestPrecisionAtK:
    """Test cases for precision@k metric."""

    def test_perfect_precision(self):
        """Test precision@k with perfect predictions."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2])

        precision = precision_at_k(y_true, y_score, k=3)

        assert precision == 1.0  # Top 3 are all positive

    def test_worst_precision(self):
        """Test precision@k with worst predictions."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

        precision = precision_at_k(y_true, y_score, k=3)

        assert precision == 0.0  # Top 3 are all negative

    def test_partial_precision(self):
        """Test precision@k with partial precision."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        precision = precision_at_k(y_true, y_score, k=3)

        assert precision == 2.0 / 3.0  # 2 out of top 3 are positive

    def test_k_larger_than_data(self):
        """Test when k is larger than data size."""
        y_true = np.array([1, 0, 1])
        y_score = np.array([0.9, 0.5, 0.8])

        precision = precision_at_k(y_true, y_score, k=5)

        # Should use all available data
        assert precision == 2.0 / 5.0  # 2 positives out of k=5

    def test_k_zero(self):
        """Test with k=0."""
        y_true = np.array([1, 0, 1])
        y_score = np.array([0.9, 0.5, 0.8])

        precision = precision_at_k(y_true, y_score, k=0)

        assert precision == 0.0

    def test_empty_data(self):
        """Test with empty data."""
        y_true = np.array([])
        y_score = np.array([])

        precision = precision_at_k(y_true, y_score, k=3)

        assert precision == 0.0

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        y_true = np.array([1, 0, 1])
        y_score = np.array([0.9, 0.5])  # Different length

        with pytest.raises(ValueError, match="must have same length"):
            precision_at_k(y_true, y_score, k=2)


class TestTemporalAUC:
    """Test cases for temporal AUC metric."""

    def test_perfect_auc(self):
        """Test temporal AUC with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        timestamps = np.array([100, 200, 300, 400])

        auc = temporal_auc(y_true, y_score, timestamps)

        assert auc == 1.0

    def test_worst_auc(self):
        """Test temporal AUC with worst predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        timestamps = np.array([100, 200, 300, 400])

        auc = temporal_auc(y_true, y_score, timestamps)

        assert auc == 0.0

    def test_random_auc(self):
        """Test temporal AUC with random predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        timestamps = np.array([100, 200, 300, 400])

        auc = temporal_auc(y_true, y_score, timestamps)

        assert abs(auc - 0.5) < 0.1  # Should be close to random

    def test_chronological_ordering(self):
        """Test that temporal ordering affects tie-breaking."""
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])  # All tied
        timestamps = np.array([400, 100, 300, 200])  # Unsorted

        auc = temporal_auc(y_true, y_score, timestamps)

        # Should handle chronological ordering
        assert 0.0 <= auc <= 1.0

    def test_single_class(self):
        """Test with single class."""
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.8, 0.9, 0.7])
        timestamps = np.array([100, 200, 300])

        auc = temporal_auc(y_true, y_score, timestamps)

        assert auc == 0.5  # Default for single class

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])
        timestamps = np.array([100])  # Different length

        with pytest.raises(ValueError, match="must have same length"):
            temporal_auc(y_true, y_score, timestamps)


class TestMotifHalfLife:
    """Test cases for motif half-life calculation."""

    def test_increasing_intervals(self):
        """Test half-life with increasing intervals."""
        hits = np.array([100, 120, 150, 200, 280])  # Increasing gaps

        half_life = motif_half_life(hits)

        # Should return finite positive value
        assert half_life > 0
        assert np.isfinite(half_life)

    def test_decreasing_intervals(self):
        """Test half-life with decreasing intervals."""
        hits = np.array([100, 150, 180, 200, 210])  # Decreasing gaps

        half_life = motif_half_life(hits)

        # Should return finite positive value
        assert half_life > 0
        assert np.isfinite(half_life)

    def test_constant_intervals(self):
        """Test half-life with constant intervals."""
        hits = np.array([100, 200, 300, 400])  # Constant 100-unit gaps

        half_life = motif_half_life(hits)

        # Should return infinite (no decay)
        assert half_life == float("inf")

    def test_single_hit(self):
        """Test half-life with single hit."""
        hits = np.array([100])

        half_life = motif_half_life(hits)

        assert half_life == float("inf")

    def test_no_hits(self):
        """Test half-life with no hits."""
        hits = np.array([])

        half_life = motif_half_life(hits)

        assert half_life == float("inf")

    def test_two_hits(self):
        """Test half-life with two hits."""
        hits = np.array([100, 200])

        half_life = motif_half_life(hits)

        # Should return finite value based on single interval
        assert half_life > 0
        assert np.isfinite(half_life)

    def test_unsorted_hits(self):
        """Test half-life with unsorted hits."""
        hits = np.array([300, 100, 200, 400])

        half_life = motif_half_life(hits)

        # Should handle unsorted input
        assert half_life > 0
        assert np.isfinite(half_life)


class TestPatternStabilityScore:
    """Test cases for pattern stability score."""

    def test_perfect_stability(self):
        """Test stability with constant scores."""
        y_score = np.array([0.8, 0.8, 0.8, 0.8])
        timestamps = np.array([100, 200, 300, 400])

        stability = pattern_stability_score(y_score, timestamps, window_size=50)

        assert stability == 1.0  # Perfect stability

    def test_unstable_pattern(self):
        """Test stability with highly variable scores."""
        y_score = np.array([0.1, 0.9, 0.1, 0.9])
        timestamps = np.array([100, 200, 300, 400])

        stability = pattern_stability_score(y_score, timestamps, window_size=50)

        assert 0.0 <= stability < 0.5  # Should be low stability

    def test_single_window(self):
        """Test stability with single time window."""
        y_score = np.array([0.5, 0.6, 0.7])
        timestamps = np.array([100, 105, 110])  # Small time range

        stability = pattern_stability_score(y_score, timestamps, window_size=50)

        assert stability == 1.0  # Single window = perfect stability

    def test_empty_data(self):
        """Test stability with empty data."""
        y_score = np.array([])
        timestamps = np.array([])

        stability = pattern_stability_score(y_score, timestamps)

        assert stability == 0.0

    def test_single_point(self):
        """Test stability with single data point."""
        y_score = np.array([0.7])
        timestamps = np.array([100])

        stability = pattern_stability_score(y_score, timestamps)

        assert stability == 1.0

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        y_score = np.array([0.5, 0.6])
        timestamps = np.array([100])

        stability = pattern_stability_score(y_score, timestamps)

        assert stability == 0.0  # Should handle gracefully


class TestArchaeologicalSignificance:
    """Test cases for archaeological significance metrics."""

    def test_basic_significance(self):
        """Test basic archaeological significance calculation."""
        pattern_scores = np.array([0.8, 0.9, 0.7])
        pattern_types = np.array(["fvg", "poi", "fvg"])
        temporal_spans = np.array([10.0, 15.0, 8.0])

        metrics = archaeological_significance(pattern_scores, pattern_types, temporal_spans)

        # Should return all expected metrics
        assert "diversity_index" in metrics
        assert "temporal_coverage" in metrics
        assert "pattern_density" in metrics
        assert "significance_weighted_score" in metrics

        # All metrics should be finite
        for value in metrics.values():
            assert np.isfinite(value)

    def test_single_pattern_type(self):
        """Test with single pattern type."""
        pattern_scores = np.array([0.8, 0.7, 0.9])
        pattern_types = np.array(["fvg", "fvg", "fvg"])
        temporal_spans = np.array([10.0, 10.0, 10.0])

        metrics = archaeological_significance(pattern_scores, pattern_types, temporal_spans)

        # Diversity should be zero (single type)
        assert metrics["diversity_index"] == 0.0
        assert metrics["pattern_density"] > 0

    def test_diverse_patterns(self):
        """Test with diverse pattern types."""
        pattern_scores = np.array([0.8, 0.7, 0.9, 0.6])
        pattern_types = np.array(["fvg", "poi", "bos", "liq"])
        temporal_spans = np.array([10.0, 15.0, 20.0, 5.0])

        metrics = archaeological_significance(pattern_scores, pattern_types, temporal_spans)

        # Diversity should be high (all different types)
        assert metrics["diversity_index"] > 1.0

    def test_empty_patterns(self):
        """Test with empty pattern data."""
        pattern_scores = np.array([])
        pattern_types = np.array([])
        temporal_spans = np.array([])

        metrics = archaeological_significance(pattern_scores, pattern_types, temporal_spans)

        # Should return zero metrics
        for value in metrics.values():
            assert value == 0.0

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        pattern_scores = np.array([0.8, 0.7])
        pattern_types = np.array(["fvg"])  # Different length
        temporal_spans = np.array([10.0, 15.0])

        with pytest.raises(ValueError, match="must have same length"):
            archaeological_significance(pattern_scores, pattern_types, temporal_spans)


class TestComputeValidationMetrics:
    """Test cases for comprehensive validation metrics."""

    def test_basic_metrics_computation(self):
        """Test basic validation metrics computation."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        timestamps = np.array([100, 200, 300, 400, 500])

        metrics = compute_validation_metrics(y_true, y_score, timestamps)

        # Should contain core metrics
        assert "temporal_auc" in metrics
        assert "precision_at_5" in metrics
        assert "precision_at_10" in metrics
        assert "precision_at_20" in metrics
        assert "pattern_stability" in metrics
        assert "motif_half_life" in metrics

        # All metrics should be finite
        for key, value in metrics.items():
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"

    def test_custom_k_values(self):
        """Test with custom k values for precision."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        timestamps = np.array([100, 200, 300, 400, 500])

        metrics = compute_validation_metrics(y_true, y_score, timestamps, k_values=[1, 3, 7])

        # Should contain custom k values
        assert "precision_at_1" in metrics
        assert "precision_at_3" in metrics
        assert "precision_at_7" in metrics
        assert "precision_at_5" not in metrics  # Default not included

    def test_with_pattern_metadata(self):
        """Test with additional pattern metadata."""
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.9])
        timestamps = np.array([100, 200, 300, 400])

        pattern_metadata = {
            "scores": [0.85, 0.75, 0.90],
            "types": ["fvg", "poi", "fvg"],
            "spans": [10.0, 15.0, 8.0],
        }

        metrics = compute_validation_metrics(
            y_true, y_score, timestamps, pattern_metadata=pattern_metadata
        )

        # Should include archaeological metrics
        assert "diversity_index" in metrics
        assert "temporal_coverage" in metrics
        assert "pattern_density" in metrics
        assert "significance_weighted_score" in metrics

    def test_no_positive_samples(self):
        """Test with no positive samples."""
        y_true = np.array([0, 0, 0, 0])
        y_score = np.array([0.2, 0.3, 0.1, 0.4])
        timestamps = np.array([100, 200, 300, 400])

        metrics = compute_validation_metrics(y_true, y_score, timestamps)

        # Should handle gracefully
        assert "motif_half_life" in metrics
        assert metrics["motif_half_life"] == float("inf")

    def test_all_positive_samples(self):
        """Test with all positive samples."""
        y_true = np.array([1, 1, 1, 1])
        y_score = np.array([0.7, 0.8, 0.6, 0.9])
        timestamps = np.array([100, 200, 300, 400])

        metrics = compute_validation_metrics(y_true, y_score, timestamps)

        # Should handle edge case
        assert "temporal_auc" in metrics
        assert metrics["temporal_auc"] == 0.5  # Single class default


@pytest.mark.parametrize(
    "metric_func,args",
    [
        (precision_at_k, ([1, 0, 1, 0], [0.8, 0.6, 0.7, 0.3], 2)),
        (temporal_auc, ([1, 0, 1, 0], [0.8, 0.6, 0.7, 0.3], [100, 200, 300, 400])),
        (motif_half_life, ([100, 200, 350, 600],)),
        (pattern_stability_score, ([0.5, 0.6, 0.7, 0.8], [100, 200, 300, 400])),
    ],
)
def test_metric_bounds(metric_func, args):
    """Test that all metrics return reasonable bounds."""
    result = metric_func(*args)

    # All metrics should return finite values
    assert np.isfinite(result)

    # Most metrics should be non-negative
    if metric_func != motif_half_life:  # Half-life can be inf
        assert result >= 0.0

    # Bounded metrics should respect bounds
    if metric_func in [precision_at_k, temporal_auc, pattern_stability_score]:
        assert result <= 1.0


def test_metrics_with_edge_cases():
    """Test all metrics with various edge cases."""
    # Empty data
    empty_metrics = compute_validation_metrics(np.array([]), np.array([]), np.array([]))

    for value in empty_metrics.values():
        assert np.isfinite(value) or value == float("inf")

    # Single sample
    single_metrics = compute_validation_metrics(np.array([1]), np.array([0.8]), np.array([100]))

    for value in single_metrics.values():
        assert np.isfinite(value) or value == float("inf")

    # Extreme values
    extreme_metrics = compute_validation_metrics(
        np.array([0, 1]), np.array([0.0, 1.0]), np.array([0, 1000000])
    )

    for value in extreme_metrics.values():
        assert np.isfinite(value) or value == float("inf")
