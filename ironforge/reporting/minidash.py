from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .writers import save_html, save_png


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_minidash(run_dir: str, out_html: str, out_png: str) -> None:
    """Build a minimal dashboard for a run.

    Parameters
    ----------
    run_dir: str
        Directory containing run artifacts.
    out_html: str
        Path where HTML report will be saved.
    out_png: str
        Path where PNG summary will be saved.
    """
    run_path = Path(run_dir)

    patterns_path = run_path / "patterns.parquet"
    confluence_path = run_path / "confluence_scores.parquet"
    motifs_path = run_path / "motifs.json"

    patterns = pd.read_parquet(patterns_path) if patterns_path.exists() else pd.DataFrame()
    confluence = pd.read_parquet(confluence_path) if confluence_path.exists() else pd.DataFrame()
    motifs = _load_json(motifs_path)

    # Build a tiny figure summarizing counts
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.imshow([[0, 1], [1, 0]], cmap="viridis", aspect="auto")
    ax.set_axis_off()
    ax.set_title("Timeline")
    save_png(fig, out_png)
    plt.close(fig)

    html = (
        "<html><body>"
        "<h1>Minidash</h1>"
        f"<p>patterns: {len(patterns)}</p>"
        f"<p>confluence: {len(confluence)}</p>"
        f"<p>motifs: {len(motifs)}</p>"
        "</body></html>"
    )
    save_html(html, out_html)
