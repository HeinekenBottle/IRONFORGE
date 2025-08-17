from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def scan_motifs(confluence_path: str, out_dir: str) -> str:
    """Generate a trivial motifs JSON from confluence scores."""
    df = pd.read_parquet(confluence_path) if Path(confluence_path).exists() else pd.DataFrame()
    motifs = (
        [{"pattern": row.pattern, "score": row.score} for row in df.itertuples()]
        if not df.empty
        else []
    )
    out_path = Path(out_dir) / "motifs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(motifs, f)
    return str(out_path)
