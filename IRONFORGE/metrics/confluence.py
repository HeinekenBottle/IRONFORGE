from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, int, Sequence[float], Sequence[int], np.ndarray]


@dataclass(frozen=True)
class ConfluenceWeights:
    cluster: float = 0.35
    htf_prox: float = 0.25
    structure: float = 0.20
    cycle: float = 0.10
    precursor: float = 0.10

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.cluster, self.htf_prox, self.structure, self.cycle, self.precursor],
            dtype=np.float32,
        )


def _to_vec(x: ArrayLike, length: int | None = None) -> np.ndarray:
    if isinstance(x, (int, float)):
        arr = np.array([x], dtype=np.float32)
    elif isinstance(x, np.ndarray):
        arr = x.astype(np.float32)
    else:
        arr = np.array(list(x), dtype=np.float32)

    # Handle scalar expansion to length
    if length is not None and arr.size == 1 and length > 1:
        arr = np.full((length,), float(arr[0]), dtype=np.float32)

    # Handle NaN values by replacing with 0
    arr = np.nan_to_num(arr, nan=0.0)

    return np.clip(arr, 0.0, 1.0)


def compute_confluence_score(
    cluster: ArrayLike,
    htf_prox: ArrayLike,
    structure: ArrayLike,
    cycle: ArrayLike,
    precursor: ArrayLike,
    weights: ConfluenceWeights | None = None,
) -> np.ndarray:
    """
    Compute Confluence Score in [0, 100].
    Inputs are expected in [0, 1]. Returns vectorized scores.
    """
    w = (weights or ConfluenceWeights()).as_array()
    # broadcast all to same length
    # infer length from the longest array-like
    lengths = [
        np.size(cluster),
        np.size(htf_prox),
        np.size(structure),
        np.size(cycle),
        np.size(precursor),
    ]
    L = int(max(lengths))
    c = _to_vec(cluster, L)
    h = _to_vec(htf_prox, L)
    s = _to_vec(structure, L)
    cy = _to_vec(cycle, L)
    p = _to_vec(precursor, L)
    M = np.stack([c, h, s, cy, p], axis=0)  # shape (5, L)
    score01 = (w[:, None] * M).sum(axis=0)
    return (score01 * 100.0).astype(np.float32)


def compute_confluence_components(
    inputs: Mapping[str, ArrayLike],
    weights: ConfluenceWeights | None = None,
) -> dict[str, np.ndarray]:
    """
    Convenience: returns per-component contribution (0..100) and total.
    Expected keys: cluster, htf_prox, structure, cycle, precursor.
    """
    w = (weights or ConfluenceWeights()).as_array()
    L = int(max(np.size(v) for v in inputs.values()))
    ordered = ["cluster", "htf_prox", "structure", "cycle", "precursor"]
    mats = [_to_vec(inputs[k], L) for k in ordered]
    M = np.stack(mats, axis=0)
    contrib = (M * w[:, None]) * 100.0
    out = {k: contrib[i].astype(np.float32) for i, k in enumerate(ordered)}
    out["total"] = contrib.sum(axis=0).astype(np.float32)
    return out
