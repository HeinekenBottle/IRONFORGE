from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    np = None  # type: ignore
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore


@dataclass(frozen=True)
class ConfluenceStripSpec:
    width: int = 1024
    height: int = 54
    pad: int = 6
    marker_radius: int = 3


def build_confluence_strip(
    minute_bins: np.ndarray,
    scores_0_100: np.ndarray,
    marker_minutes: np.ndarray | None = None,
    spec: ConfluenceStripSpec = ConfluenceStripSpec(),
) -> Image.Image:
    """
    Render a 0–100 confluence strip with optional event markers.
    """
    if np is None or Image is None or ImageDraw is None:
        raise ImportError("Wave 5 reporting requires numpy and Pillow")
    assert minute_bins.ndim == 1 and scores_0_100.ndim == 1
    assert minute_bins.shape[0] == scores_0_100.shape[0]

    W, H, P = spec.width, spec.height, spec.pad
    T = int(minute_bins.max()) + 1 if minute_bins.size else 1
    strip = np.zeros((1, T), dtype="float32")
    strip[0, minute_bins.astype(int)] = (scores_0_100 / 100.0).clip(0, 1)
    # stretch to canvas width
    row = np.kron(strip, np.ones((H - 2 * P, max(1, (W - 2 * P) // T)), dtype="float32"))[0]
    row = row[: (W - 2 * P)]
    # vertical gradient by score (greyscale)
    img = np.tile(row[None, :], (H - 2 * P, 1))
    rgba = np.stack([img, img, img, np.ones_like(img)], axis=-1)
    rgba = (rgba * 255).clip(0, 255).astype("uint8")
    canvas = np.zeros((H, W, 4), dtype="uint8")
    canvas[P : P + img.shape[0], P : P + img.shape[1], :] = rgba
    im = Image.fromarray(canvas, "RGBA")
    if marker_minutes is not None and marker_minutes.size:
        draw = ImageDraw.Draw(im)
        for m in marker_minutes.astype(int):
            x = P + int((m / max(1, T - 1)) * (W - 2 * P - 1))
            y = H // 2
            r = spec.marker_radius
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0, 255))
    return im
