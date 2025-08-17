"""
Time-Series Safe Data Splitting for IRONFORGE (Wave 4)
======================================================
Implements purged K-fold with embargo and out-of-sample splits to prevent
look-ahead bias in temporal pattern validation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedKFold:
    """Time-ordered, leakage-safe K-fold splits with embargo period.

    Parameters
    ----------
    n_splits : int
        Number of folds for cross-validation.
    embargo_mins : int
        Embargo period in minutes to prevent look-ahead leakage.

    Notes
    -----
    The embargo creates a buffer between train and test periods to ensure
    that training data cannot leak into test evaluation through temporal
    dependencies.
    """

    n_splits: int = 5
    embargo_mins: int = 30

    def split(self, timestamps: Sequence[int]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate time-ordered, leakage-safe splits.

        Parameters
        ----------
        timestamps : Sequence[int]
            Epoch minutes (or any sortable temporal values).

        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
            (train_indices, test_indices) for each fold.

        Examples
        --------
        >>> timestamps = [100, 200, 300, 400, 500, 600]
        >>> splitter = PurgedKFold(n_splits=3, embargo_mins=30)
        >>> for train_idx, test_idx in splitter.split(timestamps):
        ...     print(f"Train: {train_idx}, Test: {test_idx}")
        """
        timestamps = np.asarray(timestamps)
        n_samples = len(timestamps)

        if n_samples < self.n_splits:
            raise ValueError(f"Cannot split {n_samples} samples into {self.n_splits} folds")

        # Sort indices by timestamp to ensure temporal order
        sorted_indices = np.argsort(timestamps)
        sorted_timestamps = timestamps[sorted_indices]

        # Calculate fold boundaries
        fold_size = n_samples // self.n_splits

        for fold in range(self.n_splits):
            # Define test period for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples

            test_indices = sorted_indices[test_start:test_end]

            # Define embargo boundaries
            test_start_time = sorted_timestamps[test_start]
            test_end_time = (
                sorted_timestamps[test_end - 1] if test_end > test_start else test_start_time
            )

            # Training data: exclude test period and embargo windows
            embargo_start = test_start_time - self.embargo_mins
            embargo_end = test_end_time + self.embargo_mins

            # Find valid training indices (outside test + embargo)
            valid_train_mask = (sorted_timestamps < embargo_start) | (
                sorted_timestamps > embargo_end
            )

            train_indices = sorted_indices[valid_train_mask]

            # Ensure we have some training data
            if len(train_indices) == 0:
                continue

            yield train_indices, test_indices


def oos_split(timestamps: Sequence[int], cutoff_ts: int) -> tuple[np.ndarray, np.ndarray]:
    """Simple out-of-sample split based on timestamp cutoff.

    Parameters
    ----------
    timestamps : Sequence[int]
        Epoch minutes (or any sortable temporal values).
    cutoff_ts : int
        Cutoff timestamp. Train = ts < cutoff, Test = ts >= cutoff.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)

    Examples
    --------
    >>> timestamps = [100, 200, 300, 400, 500]
    >>> train_idx, test_idx = oos_split(timestamps, cutoff_ts=300)
    >>> print(f"Train: {train_idx}, Test: {test_idx}")
    Train: [0 1], Test: [2 3 4]
    """
    timestamps = np.asarray(timestamps)

    train_mask = timestamps < cutoff_ts
    test_mask = timestamps >= cutoff_ts

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    return train_indices, test_indices


def temporal_train_test_split(
    timestamps: Sequence[int], test_size: float = 0.2, embargo_mins: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Train-test split with temporal ordering and embargo period.

    Parameters
    ----------
    timestamps : Sequence[int]
        Epoch minutes (or any sortable temporal values).
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0).
    embargo_mins : int
        Embargo period in minutes between train and test.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)
    """
    timestamps = np.asarray(timestamps)
    n_samples = len(timestamps)

    # Sort by timestamp
    sorted_indices = np.argsort(timestamps)
    sorted_timestamps = timestamps[sorted_indices]

    # Calculate split point
    n_test = int(n_samples * test_size)
    test_start_idx = n_samples - n_test

    # Apply embargo
    test_start_time = sorted_timestamps[test_start_idx]
    embargo_cutoff = test_start_time - embargo_mins

    # Find valid training indices (before embargo)
    train_mask = sorted_timestamps < embargo_cutoff
    valid_train_indices = sorted_indices[train_mask]

    # Test indices are the last n_test samples
    test_indices = sorted_indices[test_start_idx:]

    return valid_train_indices, test_indices
