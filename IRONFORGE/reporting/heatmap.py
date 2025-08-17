from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import numpy as np
    from PIL import Image
except Exception:  # pragma: no cover
    np = None  # type: ignore
    Image = None  # type: ignore


@dataclass(frozen=True)
class TimelineHeatmapSpec:
    width: int = 1024
    height: int = 160
    pad: int = 8
    colormap: str = "viridis"  # placeholder; Wave 5 keeps defaults simple


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    m, M = x.min(), x.max()
    if m >= M:
        return np.zeros_like(x)
    return (x - m) / (M - m)


def build_session_heatmap(
    minute_bins: np.ndarray,
    densities: np.ndarray,
    spec: TimelineHeatmapSpec = TimelineHeatmapSpec(),
) -> Image.Image:
    """
    Render a single-session timeline heatmap.
    minute_bins: shape (T,), minute offsets from session open.
    densities:   shape (T,), nonnegative event density per minute.
    Returns: PIL.Image (RGBA).
    """
    if np is None or Image is None:
        raise ImportError("Wave 5 reporting requires numpy and Pillow")
    assert minute_bins.ndim == 1 and densities.ndim == 1
    assert minute_bins.shape[0] == densities.shape[0]

    W, H, P = spec.width, spec.height, spec.pad
    T = int(minute_bins.max()) + 1 if minute_bins.size else 1
    # Raster width maps to T; resample to desired W
    x = np.zeros((1, T), dtype="float32")
    x[0, minute_bins.astype(int)] = densities
    x = _normalize(x)
    # Nearest-neighbour stretch to (H-2P, W-2P)
    img = np.kron(x, np.ones((H - 2 * P, math.ceil((W - 2 * P) / T)), dtype="float32"))
    img = img[:, : (W - 2 * P)]
    # Simple grayscale → RGBA (we keep it minimal in Wave 5)
    rgba = np.stack([img, img, img, np.ones_like(img)], axis=-1)
    rgba = (rgba * 255).clip(0, 255).astype("uint8")
    canvas = np.zeros((H, W, 4), dtype="uint8")
    canvas[P : P + img.shape[0], P : P + img.shape[1], :] = rgba
    return Image.fromarray(canvas, mode="RGBA")
