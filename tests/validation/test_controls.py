"""Unit tests for validation controls (Wave 4)."""

import numpy as np
import pytest

from ironforge.validation.controls import (
    create_control_variants,
    edge_direction_shuffle,
    label_permutation,
    node_feature_shuffle,
    temporal_block_shuffle,
    time_shuffle_edges,
)


class TestTimeShuffleEdges:
    """Test cases for edge time shuffling."""

    def test_basic_shuffle(self):
        """Test basic edge time shuffling."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        edge_times = np.array([100, 200, 300])

        new_edge_index, shuffled_times = time_shuffle_edges(edge_index, edge_times, seed=42)

        # Edge connectivity should be preserved
        np.testing.assert_array_equal(new_edge_index, edge_index)

        # Times should be shuffled (different order with same values)
        assert len(shuffled_times) == len(edge_times)
        assert set(shuffled_times) == set(edge_times)
        # Should be different order (with high probability)
        assert not np.array_equal(shuffled_times, edge_times)

    def test_single_edge(self):
        """Test shuffling with single edge."""
        edge_index = np.array([[0], [1]])
        edge_times = np.array([100])

        new_edge_index, shuffled_times = time_shuffle_edges(edge_index, edge_times, seed=42)

        # Single edge should remain unchanged
        np.testing.assert_array_equal(new_edge_index, edge_index)
        np.testing.assert_array_equal(shuffled_times, edge_times)

    def test_empty_edges(self):
        """Test shuffling with no edges."""
        edge_index = np.array([[], []]).reshape(2, 0)
        edge_times = np.array([])

        new_edge_index, shuffled_times = time_shuffle_edges(edge_index, edge_times, seed=42)

        assert new_edge_index.shape == (2, 0)
        assert len(shuffled_times) == 0

    def test_reproducibility(self):
        """Test that shuffling is reproducible with same seed."""
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        edge_times = np.array([100, 200, 300, 400])

        _, shuffled1 = time_shuffle_edges(edge_index, edge_times, seed=123)
        _, shuffled2 = time_shuffle_edges(edge_index, edge_times, seed=123)

        np.testing.assert_array_equal(shuffled1, shuffled2)


class TestLabelPermutation:
    """Test cases for label permutation."""

    def test_basic_permutation(self):
        """Test basic label permutation."""
        labels = np.array([0, 0, 1, 1, 1])

        permuted = label_permutation(labels, seed=42)

        # Should preserve label distribution
        assert len(permuted) == len(labels)
        assert np.sum(permuted == 0) == np.sum(labels == 0)
        assert np.sum(permuted == 1) == np.sum(labels == 1)

        # Should be different order (with high probability)
        assert not np.array_equal(permuted, labels)

    def test_single_label(self):
        """Test permutation with single label."""
        labels = np.array([1])

        permuted = label_permutation(labels, seed=42)

        np.testing.assert_array_equal(permuted, labels)

    def test_multiclass_labels(self):
        """Test permutation with multiclass labels."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        permuted = label_permutation(labels, seed=42)

        # Should preserve class distribution
        for class_label in [0, 1, 2]:
            assert np.sum(permuted == class_label) == np.sum(labels == class_label)

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        labels = np.array([0, 1, 0, 1, 1, 0])

        permuted1 = label_permutation(labels, seed=456)
        permuted2 = label_permutation(labels, seed=456)

        np.testing.assert_array_equal(permuted1, permuted2)


class TestNodeFeatureShuffle:
    """Test cases for node feature shuffling."""

    def test_basic_shuffle(self):
        """Test basic node feature shuffling."""
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        shuffled = node_feature_shuffle(features, seed=42)

        # Shape should be preserved
        assert shuffled.shape == features.shape

        # Column means should be preserved (approximately)
        np.testing.assert_array_almost_equal(np.mean(shuffled, axis=0), np.mean(features, axis=0))

        # Should be different from original (with high probability)
        assert not np.array_equal(shuffled, features)

    def test_feature_groups(self):
        """Test shuffling within feature groups."""
        features = np.random.random((10, 6))
        feature_groups = {"group1": [0, 1, 2], "group2": [3, 4, 5]}

        shuffled = node_feature_shuffle(features, feature_groups, seed=42)

        # Shape preserved
        assert shuffled.shape == features.shape

        # Each group should be shuffled independently
        for _group_name, indices in feature_groups.items():
            for idx in indices:
                # Column means should be preserved
                assert abs(np.mean(shuffled[:, idx]) - np.mean(features[:, idx])) < 1e-10

    def test_single_feature(self):
        """Test shuffling with single feature."""
        features = np.array([[1], [2], [3]])

        shuffled = node_feature_shuffle(features, seed=42)

        assert shuffled.shape == features.shape
        assert np.mean(shuffled) == np.mean(features)


class TestEdgeDirectionShuffle:
    """Test cases for edge direction shuffling."""

    def test_basic_direction_shuffle(self):
        """Test basic edge direction shuffling."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])

        shuffled = edge_direction_shuffle(edge_index, seed=42)

        # Shape should be preserved
        assert shuffled.shape == edge_index.shape

        # Some edges should be flipped (with high probability)
        assert not np.array_equal(shuffled, edge_index)

    def test_self_loops(self):
        """Test direction shuffling with self-loops."""
        edge_index = np.array([[0, 1, 1], [0, 1, 2]])  # Self-loop on node 0

        shuffled = edge_direction_shuffle(edge_index, seed=42)

        # Self-loops should remain self-loops when flipped
        self_loop_mask = edge_index[0] == edge_index[1]
        shuffled_self_loop_mask = shuffled[0] == shuffled[1]
        np.testing.assert_array_equal(self_loop_mask, shuffled_self_loop_mask)

    def test_reproducibility(self):
        """Test reproducibility of direction shuffling."""
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])

        shuffled1 = edge_direction_shuffle(edge_index, seed=789)
        shuffled2 = edge_direction_shuffle(edge_index, seed=789)

        np.testing.assert_array_equal(shuffled1, shuffled2)


class TestTemporalBlockShuffle:
    """Test cases for temporal block shuffling."""

    def test_basic_block_shuffle(self):
        """Test basic temporal block shuffling."""
        edge_times = np.array([100, 110, 200, 210, 300, 310])

        shuffled = temporal_block_shuffle(edge_times, block_size_mins=50, seed=42)

        # Should have same length
        assert len(shuffled) == len(edge_times)

        # Values should be shifted but structure preserved
        assert not np.array_equal(shuffled, edge_times)

    def test_single_block(self):
        """Test shuffling with single block."""
        edge_times = np.array([100, 110, 120])  # Small range

        shuffled = temporal_block_shuffle(edge_times, block_size_mins=100, seed=42)

        # With large block size, should be minimal change
        assert len(shuffled) == len(edge_times)

    def test_empty_times(self):
        """Test shuffling with empty times."""
        edge_times = np.array([])

        shuffled = temporal_block_shuffle(edge_times, block_size_mins=60, seed=42)

        assert len(shuffled) == 0

    def test_single_time(self):
        """Test shuffling with single timestamp."""
        edge_times = np.array([100])

        shuffled = temporal_block_shuffle(edge_times, block_size_mins=60, seed=42)

        np.testing.assert_array_equal(shuffled, edge_times)


class TestCreateControlVariants:
    """Test cases for creating control variants."""

    def setup_method(self):
        """Set up test data."""
        self.edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        self.edge_times = np.array([100, 200, 300])
        self.node_features = np.random.random((3, 5))
        self.labels = np.array([0, 1, 0])

    def test_single_control(self):
        """Test creating single control variant."""
        controls = ["time_shuffle"]

        variants = create_control_variants(
            self.edge_index, self.edge_times, self.node_features, self.labels, controls, seed=42
        )

        assert len(variants) == 1
        assert "time_shuffle" in variants

        variant = variants["time_shuffle"]
        assert "edge_index" in variant
        assert "edge_times" in variant
        assert "node_features" in variant
        assert "labels" in variant
        assert "description" in variant

    def test_multiple_controls(self):
        """Test creating multiple control variants."""
        controls = ["time_shuffle", "label_perm", "node_shuffle"]

        variants = create_control_variants(
            self.edge_index, self.edge_times, self.node_features, self.labels, controls, seed=42
        )

        assert len(variants) == 3
        for control in controls:
            assert control in variants
            assert "description" in variants[control]

    def test_all_control_types(self):
        """Test all available control types."""
        controls = [
            "time_shuffle",
            "label_perm",
            "node_shuffle",
            "edge_direction",
            "temporal_blocks",
        ]

        variants = create_control_variants(
            self.edge_index, self.edge_times, self.node_features, self.labels, controls, seed=42
        )

        assert len(variants) == len(controls)

        # Each variant should modify the appropriate component
        assert not np.array_equal(variants["time_shuffle"]["edge_times"], self.edge_times)
        assert not np.array_equal(variants["label_perm"]["labels"], self.labels)
        # Other components should be unchanged
        np.testing.assert_array_equal(variants["time_shuffle"]["edge_index"], self.edge_index)

    def test_empty_controls(self):
        """Test with empty controls list."""
        variants = create_control_variants(
            self.edge_index, self.edge_times, self.node_features, self.labels, [], seed=42
        )

        assert len(variants) == 0

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        controls = ["time_shuffle"]

        variants1 = create_control_variants(
            self.edge_index, self.edge_times, self.node_features, self.labels, controls, seed=123
        )

        variants2 = create_control_variants(
            self.edge_index, self.edge_times, self.node_features, self.labels, controls, seed=456
        )

        # Should produce different shuffled times
        assert not np.array_equal(
            variants1["time_shuffle"]["edge_times"], variants2["time_shuffle"]["edge_times"]
        )


@pytest.mark.parametrize(
    "control_type",
    ["time_shuffle", "label_perm", "node_shuffle", "edge_direction", "temporal_blocks"],
)
def test_control_preserves_data_structure(control_type):
    """Test that each control preserves data structure."""
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_times = np.array([100, 200, 300, 400])
    node_features = np.random.random((4, 10))
    labels = np.array([0, 1, 0, 1])

    variants = create_control_variants(
        edge_index, edge_times, node_features, labels, [control_type], seed=42
    )

    variant = variants[control_type]

    # All components should have correct shapes
    assert variant["edge_index"].shape == edge_index.shape
    assert variant["edge_times"].shape == edge_times.shape
    assert variant["node_features"].shape == node_features.shape
    assert variant["labels"].shape == labels.shape


def test_control_destroys_signal():
    """Test that controls actually destroy the signal they're meant to."""
    # Create data with clear temporal signal
    edge_index = np.array([[0, 1, 2], [1, 2, 0]])
    edge_times = np.array([100, 200, 300])  # Ordered times
    node_features = np.array([[1, 0], [2, 0], [3, 0]])  # Ordered features
    labels = np.array([0, 1, 0])  # Pattern in labels

    variants = create_control_variants(
        edge_index,
        edge_times,
        node_features,
        labels,
        ["time_shuffle", "label_perm", "node_shuffle"],
        seed=42,
    )

    # Time shuffle should break temporal order
    assert not np.array_equal(variants["time_shuffle"]["edge_times"], edge_times)

    # Label permutation should break feature-label relationship
    assert not np.array_equal(variants["label_perm"]["labels"], labels)

    # Node shuffle should break feature structure
    assert not np.array_equal(variants["node_shuffle"]["node_features"], node_features)
