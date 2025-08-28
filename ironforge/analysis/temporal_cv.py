"""
Temporal cross-validation utilities (walk-forward) for metamorphosis validation.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Tuple


class TemporalCVRunner:
    def __init__(self, sessions: Sequence[str], n_splits: int = 5, oos_fraction: float = 0.2):
        self.sessions = list(sessions)
        self.n_splits = max(2, int(n_splits))
        self.oos_fraction = min(0.9, max(0.05, float(oos_fraction)))

    def split(self) -> List[Tuple[List[str], List[str]]]:
        n = len(self.sessions)
        if n < self.n_splits:
            return [(self.sessions[: max(1, n - 1)], self.sessions[max(1, n - 1) :])]
        fold_size = max(1, n // self.n_splits)
        splits = []
        for i in range(self.n_splits - 1):
            train_end = (i + 1) * fold_size
            train = self.sessions[:train_end]
            valid = self.sessions[train_end : train_end + fold_size]
            if valid:
                splits.append((train, valid))
        # OOS final split
        oos_start = int(n * (1 - self.oos_fraction))
        if oos_start > 0:
            splits.append((self.sessions[:oos_start], self.sessions[oos_start:]))
        return splits

    def evaluate(self, detector_fn: Callable[[List[str]], Dict], metrics_fn: Callable[[Dict], Dict]) -> Dict:
        results = []
        for train, valid in self.split():
            fold_result = detector_fn(valid)
            metrics = metrics_fn(fold_result)
            results.append({"train": train, "valid": valid, "metrics": metrics})
        return {"folds": results}
