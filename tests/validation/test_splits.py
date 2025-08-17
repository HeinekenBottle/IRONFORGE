"""Unit tests for validation splits (Wave 4)."""

import numpy as np
import pytest

from ironforge.validation.splits import (
    PurgedKFold,
    oos_split,
    temporal_train_test_split,
)


class TestPurgedKFold:
    """Test cases for PurgedKFold splitter."""

    def test_basic_split(self):
        """Test basic purged k-fold splitting."""
        timestamps = np.array([100, 200, 300, 400, 500, 600])
        splitter = PurgedKFold(n_splits=3, embargo_mins=10)

        splits = list(splitter.split(timestamps))

        # Should return 3 splits
        assert len(splits) == 3

        # Each split should return train and test indices
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert len(test_idx) > 0  # Should have test samples

    def test_embargo_enforcement(self):
        """Test that embargo period is properly enforced."""
        timestamps = np.array([100, 110, 120, 130, 140, 150])
        splitter = PurgedKFold(n_splits=2, embargo_mins=15)

        splits = list(splitter.split(timestamps))

        for train_idx, test_idx in splits:
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Check that there's sufficient gap between train and test
                max_train_time = np.max(timestamps[train_idx])
                min_test_time = np.min(timestamps[test_idx])

                # Should have at least embargo_mins gap
                gap = min_test_time - max_train_time
                assert gap >= splitter.embargo_mins or gap <= -splitter.embargo_mins

    def test_temporal_ordering(self):
        """Test that splits respect temporal ordering."""
        timestamps = np.array([500, 100, 300, 200, 400])  # Unsorted
        splitter = PurgedKFold(n_splits=2, embargo_mins=5)

        splits = list(splitter.split(timestamps))

        # Verify that the splits work despite unsorted input
        assert len(splits) >= 1

        for train_idx, test_idx in splits:
            # All indices should be valid
            assert np.all(train_idx >= 0) and np.all(train_idx < len(timestamps))
            assert np.all(test_idx >= 0) and np.all(test_idx < len(timestamps))

    def test_insufficient_samples(self):
        """Test behavior with insufficient samples."""
        timestamps = np.array([100, 200])  # Only 2 samples
        splitter = PurgedKFold(n_splits=5, embargo_mins=10)

        with pytest.raises(ValueError, match="Cannot split"):
            list(splitter.split(timestamps))

    def test_single_split(self):
        """Test with single split (should work like train-test split)."""
        timestamps = np.array([100, 200, 300, 400, 500])
        splitter = PurgedKFold(n_splits=1, embargo_mins=5)

        splits = list(splitter.split(timestamps))
        assert len(splits) == 1

        train_idx, test_idx = splits[0]
        assert len(test_idx) > 0


class TestOOSSplit:
    """Test cases for out-of-sample split."""

    def test_basic_oos_split(self):
        """Test basic OOS splitting functionality."""
        timestamps = np.array([100, 200, 300, 400, 500])
        cutoff_ts = 300

        train_idx, test_idx = oos_split(timestamps, cutoff_ts)

        # Check split correctness
        assert np.all(timestamps[train_idx] < cutoff_ts)
        assert np.all(timestamps[test_idx] >= cutoff_ts)

        # Check that all samples are assigned
        assert len(train_idx) + len(test_idx) == len(timestamps)
        assert len(np.intersect1d(train_idx, test_idx)) == 0  # No overlap

    def test_edge_cases(self):
        """Test edge cases for OOS split."""
        timestamps = np.array([100, 200, 300, 400, 500])

        # Cutoff before all data
        train_idx, test_idx = oos_split(timestamps, cutoff_ts=50)
        assert len(train_idx) == 0
        assert len(test_idx) == len(timestamps)

        # Cutoff after all data
        train_idx, test_idx = oos_split(timestamps, cutoff_ts=600)
        assert len(train_idx) == len(timestamps)
        assert len(test_idx) == 0

        # Cutoff exactly on boundary
        train_idx, test_idx = oos_split(timestamps, cutoff_ts=300)
        assert np.all(timestamps[train_idx] < 300)
        assert np.all(timestamps[test_idx] >= 300)

    def test_empty_input(self):
        """Test OOS split with empty input."""
        timestamps = np.array([])

        train_idx, test_idx = oos_split(timestamps, cutoff_ts=100)

        assert len(train_idx) == 0
        assert len(test_idx) == 0

    def test_single_timestamp(self):
        """Test OOS split with single timestamp."""
        timestamps = np.array([300])

        # Cutoff before
        train_idx, test_idx = oos_split(timestamps, cutoff_ts=200)
        assert len(train_idx) == 0
        assert len(test_idx) == 1

        # Cutoff after
        train_idx, test_idx = oos_split(timestamps, cutoff_ts=400)
        assert len(train_idx) == 1
        assert len(test_idx) == 0


class TestTemporalTrainTestSplit:
    """Test cases for temporal train-test split."""

    def test_basic_split(self):
        """Test basic temporal train-test split."""
        timestamps = np.array([100, 200, 300, 400, 500])

        train_idx, test_idx = temporal_train_test_split(timestamps, test_size=0.4, embargo_mins=10)

        # Check approximate split ratio
        expected_test_size = int(len(timestamps) * 0.4)
        assert len(test_idx) == expected_test_size

        # Check temporal ordering (test should be later)
        if len(train_idx) > 0 and len(test_idx) > 0:
            max_train_time = np.max(timestamps[train_idx])
            min_test_time = np.min(timestamps[test_idx])
            # With embargo, train should be sufficiently before test
            assert min_test_time > max_train_time

    def test_embargo_effect(self):
        """Test that embargo reduces training set size appropriately."""
        timestamps = np.array([100, 110, 120, 130, 140, 150])

        # Without embargo
        train_idx_no_embargo, test_idx_no_embargo = temporal_train_test_split(
            timestamps, test_size=0.3, embargo_mins=0
        )

        # With embargo
        train_idx_embargo, test_idx_embargo = temporal_train_test_split(
            timestamps, test_size=0.3, embargo_mins=20
        )

        # Embargo should reduce training set
        assert len(train_idx_embargo) <= len(train_idx_no_embargo)
        # Test set size should be the same
        assert len(test_idx_embargo) == len(test_idx_no_embargo)

    def test_edge_cases(self):
        """Test edge cases for temporal train-test split."""
        timestamps = np.array([100, 200, 300])

        # Very large test size
        train_idx, test_idx = temporal_train_test_split(timestamps, test_size=0.9, embargo_mins=5)
        assert len(test_idx) == int(len(timestamps) * 0.9)

        # Very small test size
        train_idx, test_idx = temporal_train_test_split(timestamps, test_size=0.1, embargo_mins=5)
        assert len(test_idx) == int(len(timestamps) * 0.1)

    def test_unsorted_timestamps(self):
        """Test with unsorted timestamps."""
        timestamps = np.array([300, 100, 500, 200, 400])

        train_idx, test_idx = temporal_train_test_split(timestamps, test_size=0.4, embargo_mins=10)

        # Should handle unsorted input correctly
        assert len(train_idx) + len(test_idx) <= len(timestamps)
        # All indices should be valid
        assert np.all(train_idx >= 0) and np.all(train_idx < len(timestamps))
        assert np.all(test_idx >= 0) and np.all(test_idx < len(timestamps))


@pytest.mark.parametrize(
    "n_splits,embargo_mins",
    [
        (3, 10),
        (5, 30),
        (2, 60),
    ],
)
def test_purged_kfold_consistency(n_splits, embargo_mins):
    """Test that PurgedKFold produces consistent results with same seed."""
    timestamps = np.arange(100, 200, 2)  # 50 timestamps

    splitter = PurgedKFold(n_splits=n_splits, embargo_mins=embargo_mins)

    splits1 = list(splitter.split(timestamps))
    splits2 = list(splitter.split(timestamps))

    # Should produce identical splits
    assert len(splits1) == len(splits2)

    for (train1, test1), (train2, test2) in zip(splits1, splits2, strict=False):
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)


def test_split_coverage():
    """Test that all validation splits provide reasonable coverage."""
    timestamps = np.arange(100, 200, 1)  # 100 timestamps

    # Test PurgedKFold coverage
    splitter = PurgedKFold(n_splits=5, embargo_mins=5)
    splits = list(splitter.split(timestamps))

    all_test_indices = set()
    for _, test_idx in splits:
        all_test_indices.update(test_idx)

    # Should cover a reasonable fraction of the data
    coverage = len(all_test_indices) / len(timestamps)
    assert coverage > 0.5  # At least 50% coverage

    # Test OOS split coverage
    train_idx, test_idx = oos_split(timestamps, cutoff_ts=150)
    assert len(train_idx) + len(test_idx) == len(timestamps)

    # Test temporal split coverage
    train_idx, test_idx = temporal_train_test_split(timestamps, test_size=0.3)
    # With embargo, might not cover all data, but should be reasonable
    coverage = (len(train_idx) + len(test_idx)) / len(timestamps)
    assert coverage > 0.7  # At least 70% coverage
