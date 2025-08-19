#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def one(patterns: list[str]) -> str | None:
    for p in patterns:
        m = sorted(glob.glob(p))
        if m:
            return m[0]
    return None


def rel_paths(base: Path, patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        for f in glob.glob(str(base / p)):
            try:
                out.append(str(Path(f).relative_to(base)))
            except Exception:
                out.append(str(f))
    return sorted(set(out))


def read_cols(parquet_path: str) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path).column_names


def count_rows(paths: list[str]) -> int:
    if not paths:
        return 0
    import pyarrow.parquet as pq

    rows = 0
    for p in paths:
        try:
            rows += pq.read_table(p, columns=[]).num_rows  # columns=[] still loads metadata
        except Exception:
            try:
                rows += pq.read_table(p).num_rows
            except Exception:
                pass
    return rows


def count_feat_cols(cols: list[str]) -> int:
    return sum(1 for c in cols if re.fullmatch(r"f\d+", c))


def guess_params_from_name(run_dir: Path) -> tuple[str | None, int | None]:
    name = run_dir.name
    m = re.search(r"([A-Z]+)_(\d+)m", name)
    if m:
        try:
            return m.group(1), int(m.group(2))
        except Exception:
            return m.group(1), None
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Create manifest.json for a run directory")
    ap.add_argument("run_dir", help="Path to run directory (runs/YYYY-MM-DD/<SYMBOL_TFm>[_htf]/)")
    ap.add_argument("--window-bars", type=int, default=512)
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    # Identify representative parquet files
    node_pq = one([str(run / "patterns" / "*.parquet"), str(run / "embeddings" / "*.parquet"), str(run / "*.parquet")])
    edge_pq = one([str(run / "edges" / "*.parquet"), str(run / "patterns" / "*edges*.parquet")])

    # Dimensions
    node_dim = None
    if node_pq:
        try:
            node_dim = count_feat_cols(read_cols(node_pq))
        except Exception:
            node_dim = None
    edge_dim = None
    if edge_pq:
        try:
            edge_dim = count_feat_cols(read_cols(edge_pq))
        except Exception:
            edge_dim = None

    # Counts
    candidates = sorted(glob.glob(str(run / "patterns" / "*.parquet")))
    confluence = sorted(glob.glob(str(run / "confluence" / "*.parquet")))
    motifs = sorted(glob.glob(str(run / "motifs" / "*.parquet")))
    confluence_rows = count_rows(confluence)

    # Guess params and versions
    symbol, tf = guess_params_from_name(run)
    try:
        import ironforge as pkg

        version = getattr(pkg, "__version__", "unknown")
    except Exception:
        version = "unknown"

    # Determine HTF context preference
    htf_context = True if node_dim == 51 else False

    manifest = {
        "version": version,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "run_dir": str(run),
        "params": {
            "symbol": symbol,
            "tf": tf,
            "window_bars": args.window_bars,
            "htf_context": htf_context,
        },
        "invariants": {
            "node_feature_dim": node_dim,
            "edge_feature_dim": edge_dim,
            "taxonomy_events": 6,
            "edge_intents": 4,
        },
        "counts": {
            "candidates": len(candidates),
            "confluence_rows": confluence_rows,
            "motifs": len(motifs),
            "reports_present": (run.parent / "report_45d").exists() or (run.parent / "report_51d").exists(),
        },
        "files": {
            "patterns": rel_paths(run, ["patterns/*.parquet"]),
            "confluence": rel_paths(run, ["confluence/*.parquet"]),
            "motifs": rel_paths(run, ["motifs/*.parquet"]),
            "reports": rel_paths(run.parent, ["report_*/*"]),
        },
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }

    out_path = run / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()

