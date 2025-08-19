#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


def find_confluence_paths(run_glob: str) -> list[Path]:
    paths: list[Path] = []
    for run_dir in glob.glob(run_glob):
        d = Path(run_dir)
        paths.extend(Path(p) for p in glob.glob(str(d / "confluence" / "*.parquet")))
    return paths


def build_watchlist(confluence_paths: list[Path], top: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in confluence_paths:
        try:
            df = pd.read_parquet(p)
            # Normalize expected columns
            cols = set(df.columns)
            if "score" not in cols:
                continue
            if "pattern" not in cols and "pattern_path" in cols:
                df = df.rename(columns={"pattern_path": "pattern"})
            df["run_dir"] = str(Path(p).parent.parent)
            frames.append(df[[c for c in ["pattern", "score", "run_dir"] if c in df.columns]])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["pattern", "score", "run_dir"])
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("score", ascending=False).head(top)
    return all_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit next-session watchlist from confluence outputs")
    ap.add_argument("run_glob", help="Run directory glob, e.g. runs/2025-08-19/NQ_5m_*")
    ap.add_argument("--top", type=int, default=25, help="Top-N candidates by score")
    ap.add_argument("--out", default="watchlist.csv", help="Output CSV path")
    args = ap.parse_args()

    confluence_paths = find_confluence_paths(args.run_glob)
    df = build_watchlist(confluence_paths, args.top)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[watchlist] wrote {args.out} with {len(df)} rows")


if __name__ == "__main__":
    main()

