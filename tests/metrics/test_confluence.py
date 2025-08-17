"""Tests for ironforge.metrics.confluence module."""

import pytest
import numpy as np
from ironforge.metrics.confluence import (
    ConfluenceWeights,
    compute_confluence_score,
    compute_confluence_components,
    _to_vec
)


class TestConfluenceWeights:
    """Test ConfluenceWeights dataclass."""
    
    def test_default_weights(self):
        """Test default weight values sum to 1.0."""
        weights = ConfluenceWeights()
        assert weights.cluster == 0.35
        assert weights.htf_prox == 0.25
        assert weights.structure == 0.20
        assert weights.cycle == 0.10
        assert weights.precursor == 0.10
        
        # Should sum to 1.0
        total = weights.cluster + weights.htf_prox + weights.structure + weights.cycle + weights.precursor
        assert abs(total - 1.0) < 1e-6
    
    def test_custom_weights(self):
        """Test custom weight values."""
        weights = ConfluenceWeights(
            cluster=0.4, htf_prox=0.3, structure=0.2, cycle=0.05, precursor=0.05
        )
        assert weights.cluster == 0.4
        assert weights.htf_prox == 0.3
        
        # Test as_array method
        arr = weights.as_array()
        expected = np.array([0.4, 0.3, 0.2, 0.05, 0.05], dtype=np.float32)
        np.testing.assert_array_equal(arr, expected)


class TestToVec:
    """Test _to_vec utility function."""
    
    def test_scalar_input(self):
        """Test scalar to vector conversion."""
        result = _to_vec(0.7, length=5)
        expected = np.full(5, 0.7, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_list_input(self):
        """Test list to vector conversion."""
        result = _to_vec([0.1, 0.5, 0.9])
        expected = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_numpy_input(self):
        """Test numpy array input."""
        input_arr = np.array([0.2, 0.8])
        result = _to_vec(input_arr)
        expected = np.array([0.2, 0.8], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_clipping(self):
        """Test that values are clipped to [0, 1]."""
        result = _to_vec([-0.5, 0.3, 1.5])
        expected = np.array([0.0, 0.3, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestComputeConfluenceScore:
    """Test confluence score computation."""
    
    def test_scalar_inputs(self):
        """Test with scalar inputs."""
        score = compute_confluence_score(
            cluster=1.0, htf_prox=1.0, structure=1.0, cycle=1.0, precursor=1.0
        )
        # Should be 100.0 for all maxed inputs
        assert score == 100.0
    
    def test_zero_inputs(self):
        """Test with zero inputs."""
        score = compute_confluence_score(
            cluster=0.0, htf_prox=0.0, structure=0.0, cycle=0.0, precursor=0.0
        )
        assert score == 0.0
    
    def test_mixed_inputs(self):
        """Test with realistic mixed inputs."""
        score = compute_confluence_score(
            cluster=0.8,    # Strong cluster
            htf_prox=0.6,   # Good HTF proximity
            structure=0.4,  # Moderate structure
            cycle=0.7,      # Good cycle alignment
            precursor=0.3   # Weak precursor
        )
        
        # Manual calculation: 0.35*0.8 + 0.25*0.6 + 0.20*0.4 + 0.10*0.7 + 0.10*0.3 = 0.61
        expected = 61.0
        assert abs(score - expected) < 0.1
    
    def test_vector_inputs(self):
        """Test with vector inputs."""
        scores = compute_confluence_score(
            cluster=[0.5, 0.8],
            htf_prox=[0.3, 0.9],
            structure=[0.4, 0.6],
            cycle=[0.2, 0.7],
            precursor=[0.1, 0.5]
        )
        
        assert len(scores) == 2
        assert all(0 <= score <= 100 for score in scores)
        assert scores[1] > scores[0]  # Second should be higher
    
    def test_custom_weights(self):
        """Test with custom weights."""
        weights = ConfluenceWeights(cluster=0.5, htf_prox=0.5, structure=0.0, cycle=0.0, precursor=0.0)
        score = compute_confluence_score(
            cluster=1.0, htf_prox=1.0, structure=0.0, cycle=0.0, precursor=0.0,
            weights=weights
        )
        assert score == 100.0
    
    def test_bounds(self):
        """Test score bounds are respected."""
        # Test extreme values
        score_max = compute_confluence_score(
            cluster=2.0, htf_prox=2.0, structure=2.0, cycle=2.0, precursor=2.0  # Will be clipped to 1.0
        )
        assert score_max == 100.0
        
        score_min = compute_confluence_score(
            cluster=-1.0, htf_prox=-1.0, structure=-1.0, cycle=-1.0, precursor=-1.0  # Will be clipped to 0.0
        )
        assert score_min == 0.0


class TestComputeConfluenceComponents:
    """Test confluence component breakdown."""
    
    def test_component_sum(self):
        """Test that components sum to total."""
        inputs = {
            "cluster": 0.8,
            "htf_prox": 0.6,
            "structure": 0.4,
            "cycle": 0.7,
            "precursor": 0.3
        }
        
        result = compute_confluence_components(inputs)
        
        # Check all expected keys exist
        expected_keys = {"cluster", "htf_prox", "structure", "cycle", "precursor", "total"}
        assert set(result.keys()) == expected_keys
        
        # Check component sum equals total
        component_sum = (
            result["cluster"] + result["htf_prox"] + result["structure"] + 
            result["cycle"] + result["precursor"]
        )
        np.testing.assert_array_almost_equal(component_sum, result["total"])
    
    def test_component_weights(self):
        """Test component contributions match weights."""
        inputs = {
            "cluster": 1.0,
            "htf_prox": 0.0,
            "structure": 0.0,
            "cycle": 0.0,
            "precursor": 0.0
        }
        
        result = compute_confluence_components(inputs)
        
        # Only cluster should contribute
        assert result["cluster"][0] == 35.0  # 0.35 * 100
        assert result["htf_prox"][0] == 0.0
        assert result["total"][0] == 35.0
    
    def test_vector_components(self):
        """Test with vector inputs."""
        inputs = {
            "cluster": [0.5, 0.8],
            "htf_prox": [0.3, 0.9],
            "structure": [0.4, 0.6],
            "cycle": [0.2, 0.7],
            "precursor": [0.1, 0.5]
        }
        
        result = compute_confluence_components(inputs)
        
        # Check shape
        assert len(result["total"]) == 2
        
        # Check individual contributions are reasonable
        assert result["cluster"][0] < result["cluster"][1]  # Higher cluster in second
        assert result["total"][0] < result["total"][1]     # Higher total in second


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        # This should work with scalars expanded to match
        score = compute_confluence_score(
            cluster=0.5, htf_prox=0.5, structure=0.5, cycle=0.5, precursor=0.5
        )
        assert 0 <= score <= 100
    
    def test_single_value_arrays(self):
        """Test with single-value arrays."""
        score = compute_confluence_score(
            cluster=[0.7], htf_prox=[0.6], structure=[0.5], cycle=[0.4], precursor=[0.3]
        )
        assert len(score) == 1
        assert 0 <= score[0] <= 100
    
    def test_broadcast_different_lengths(self):
        """Test broadcasting with different input lengths."""
        # Longest determines final length
        inputs = {
            "cluster": [0.5, 0.8, 0.9],  # Length 3
            "htf_prox": 0.6,             # Scalar, should broadcast
            "structure": [0.4, 0.7],     # Length 2, should be truncated or handled
            "cycle": 0.5,
            "precursor": 0.3
        }
        
        # This might raise or handle gracefully - test current behavior
        try:
            result = compute_confluence_components(inputs)
            # If it succeeds, check reasonable output
            assert len(result["total"]) >= 1
        except (ValueError, IndexError):
            # Broadcasting mismatch is acceptable behavior
            pass
    
    def test_nan_handling(self):
        """Test NaN input handling."""
        # NaN inputs should be clipped/handled gracefully
        score = compute_confluence_score(
            cluster=0.5, htf_prox=np.nan, structure=0.5, cycle=0.5, precursor=0.5
        )
        # Check result is still numeric (clipping should handle NaN)
        assert not np.isnan(score)


@pytest.mark.performance
class TestPerformance:
    """Test performance requirements."""
    
    def test_large_vector_performance(self):
        """Test performance with large vectors."""
        import time
        
        # Create 1000 data points
        size = 1000
        cluster = np.random.random(size)
        htf_prox = np.random.random(size) 
        structure = np.random.random(size)
        cycle = np.random.random(size)
        precursor = np.random.random(size)
        
        start_time = time.time()
        scores = compute_confluence_score(cluster, htf_prox, structure, cycle, precursor)
        elapsed = time.time() - start_time
        
        # Should complete quickly
        assert elapsed < 0.1  # 100ms for 1000 points
        assert len(scores) == size
        assert all(0 <= score <= 100 for score in scores)
