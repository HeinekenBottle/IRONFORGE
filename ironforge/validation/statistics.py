"""
Statistical utilities for BMAD Temporal Pattern Metamorphosis research.
Lightweight implementations with NumPy-only dependencies to preserve portability.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import math
import numpy as np


def _to_numpy(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def empirical_p_value(observed: float, background: Sequence[float], alternative: str = "greater") -> float:
    """
    Empirical p-value against a background distribution.

    - greater: P(X >= observed)
    - less:    P(X <= observed)
    - two-sided: 2 * min(tails)
    """
    bg = _to_numpy(background)
    if bg.size == 0 or not np.isfinite(observed):
        return 1.0
    if alternative == "greater":
        count = np.sum(bg >= observed)
        p = (count + 1) / (bg.size + 1)
    elif alternative == "less":
        count = np.sum(bg <= observed)
        p = (count + 1) / (bg.size + 1)
    elif alternative == "two-sided":
        p_upper = (np.sum(bg >= observed) + 1) / (bg.size + 1)
        p_lower = (np.sum(bg <= observed) + 1) / (bg.size + 1)
        p = 2 * min(p_upper, p_lower)
        p = min(1.0, p)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'
")
    return float(p)


def bootstrap_ci(values: Sequence[float], alpha: float = 0.05, n_boot: int = 5000, seed: int = 42,
                 stat: str = "mean") -> Tuple[float, float]:
    """
    Nonparametric bootstrap confidence interval for mean or median.
    Returns (lower, upper) at (1 - alpha) confidence.
    """
    rng = np.random.default_rng(seed)
    data = _to_numpy(values)
    if data.size == 0:
        return (0.0, 0.0)
    if stat == "mean":
        stat_fn = np.mean
    elif stat == "median":
        stat_fn = np.median
    else:
        raise ValueError("stat must be 'mean' or 'median'")
    n = data.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boots[i] = stat_fn(sample)
    lower = float(np.quantile(boots, alpha / 2))
    upper = float(np.quantile(boots, 1 - alpha / 2))
    return (lower, upper)


def fdr_bh(p_values: Sequence[float], q: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg FDR control. Returns a list of booleans indicating rejection.
    """
    p = _to_numpy(p_values)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    rejected = ranked <= thresh
    if not np.any(rejected):
        return [False] * m
    k = np.max(np.where(rejected)[0])
    cutoff = ranked[k]
    return [bool(val <= cutoff) for val in p]


def cohen_d(sample: Sequence[float], baseline: Sequence[float]) -> float:
    """
    Cohen's d (with pooled SD). If SD ~ 0, returns 0.
    """
    x = _to_numpy(sample)
    y = _to_numpy(baseline)
    if x.size == 0 or y.size == 0:
        return 0.0
    mean_diff = float(np.mean(x) - np.mean(y))
    sx = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    sy = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    nx, ny = max(1, x.size), max(1, y.size)
    pooled = math.sqrt(((nx - 1) * sx + (ny - 1) * sy) / max(1, (nx + ny - 2)))
    if pooled == 0.0:
        return 0.0
    return mean_diff / pooled


def hedges_g(sample: Sequence[float], baseline: Sequence[float]) -> float:
    """
    Hedges' g: small-sample corrected Cohen's d.
    """
    x = _to_numpy(sample)
    y = _to_numpy(baseline)
    nx, ny = max(1, x.size), max(1, y.size)
    d = cohen_d(x, y)
    # Small sample correction
    J = 1.0 - 3.0 / (4.0 * (nx + ny) - 9.0) if (nx + ny) > 2 else 1.0
    return d * J
