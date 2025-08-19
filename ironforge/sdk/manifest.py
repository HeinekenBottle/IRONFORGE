from __future__ import annotations

from pathlib import Path
import json
import re
import glob
import platform
import sys
from datetime import datetime, timezone


def _feat_dim(pq_path: str) -> int | None:
    try:
        import pyarrow.parquet as pq

        cols = pq.read_table(pq_path).column_names
        return sum(1 for c in cols if re.fullmatch(r"f\d+", c))
    except Exception:
        return None


def write_for_run(run_dir: str, window_bars: int = 512, version: str = "unknown") -> Path:
    """Write a lightweight manifest.json into the given run directory.

    Backward-compatible helper used by report-minimal when IRONFORGE_WRITE_MANIFEST=1.
    """
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)

    nodes = sorted(glob.glob(str(d / "patterns" / "*.parquet"))) or sorted(
        glob.glob(str(d / "embeddings" / "*.parquet"))
    )
    edges = sorted(glob.glob(str(d / "edges" / "*.parquet")))

    node_dim = _feat_dim(nodes[0]) if nodes else None
    edge_dim = _feat_dim(edges[0]) if edges else None

    name = d.name
    htf = "htf" in name.lower()
    sym, tf = (None, None)
    m = re.search(r"([A-Z]+)_(\d+)m", name)
    if m:
        try:
            sym, tf = (m.group(1), int(m.group(2)))
        except Exception:
            sym, tf = (m.group(1), None)

    manifest = {
        "version": version,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "run_dir": str(d),
        "params": {
            "symbol": sym,
            "tf": tf,
            "window_bars": window_bars,
            "htf_context": bool(htf),
        },
        "invariants": {
            "node_feature_dim": node_dim,
            "edge_feature_dim": edge_dim,
            "taxonomy_events": 6,
            "edge_intents": 4,
        },
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }

    out = d / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out

