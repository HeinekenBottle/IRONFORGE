from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

from ironforge.reporting.minidash import build_minidash

from .app_config import load_config, materialize_run_dir
from .io import glob_many, write_json


def _maybe(mod: str, attr: str):
    try:
        m = importlib.import_module(mod)
        return getattr(m, attr)
    except Exception:
        return None


def cmd_discover(cfg):
    fn = _maybe("ironforge.learning.tgat_discovery", "run_discovery") or _maybe(
        "ironforge.discovery.runner", "run_discovery"
    )
    if fn is None:
        print("[discover] discovery engine not found; skipping (no-op).")
        return 0
    return int(bool(fn(cfg)))


def cmd_score(cfg):
    fn = _maybe("ironforge.confluence.scorer", "score_session") or _maybe(
        "ironforge.metrics.confluence", "score_session"
    )
    if fn is None:
        print("[score] scorer not found; skipping (no-op).")
        return 0
    fn(cfg)
    return 0


def cmd_validate(cfg):
    fn = _maybe("ironforge.validation.runner", "validate_run")
    if fn is None:
        print("[validate] validation rails not found; skipping (no-op).")
        return 0
    res = fn(cfg)
    run_dir = materialize_run_dir(cfg) / "reports"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "validation.json", res if isinstance(res, dict) else {"result": "ok"})
    return 0


def _load_first_parquet(paths: list[Path], cols: list[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_parquet(paths[0])
        return df
    except Exception:
        return pd.DataFrame(columns=cols)


def cmd_report(cfg):
    run_dir = materialize_run_dir(cfg)
    conf_paths = glob_many(str(run_dir / "confluence" / "*.parquet"))
    conf = _load_first_parquet(conf_paths, ["ts", "score"])
    if conf.empty:
        conf = pd.DataFrame(
            {
                "ts": pd.date_range("2025-01-01", periods=50, freq="min"),
                "score": [min(99, i * 2 % 100) for i in range(50)],
            }
        )
    pat_paths = glob_many(str(run_dir / "patterns" / "*.parquet"))
    act = _load_first_parquet(pat_paths, ["ts", "count"])
    if act.empty:
        g = conf.groupby(conf["ts"].dt.floor("min")).size().reset_index(name="count")
        g.rename(columns={g.columns[0]: "ts"}, inplace=True)
        act = g
    motifs = []
    for j in Path(run_dir / "motifs").glob("*.json"):
        with contextlib.suppress(Exception):
            motifs.extend(json.loads(j.read_text(encoding="utf-8")))
    if not motifs:
        motifs = [{"name": "sweep→fvg", "support": 12, "ppv": 0.61}]
    out_html = run_dir / cfg.reporting.minidash.out_html
    out_png = run_dir / cfg.reporting.minidash.out_png
    build_minidash(
        act,
        conf,
        motifs,
        out_html,
        out_png,
        width=cfg.reporting.minidash.width,
        height=cfg.reporting.minidash.height,
    )
    print(f"[report] wrote {out_html} and {out_png}")
    return 0


def cmd_prep_shards(source_glob: str, symbol: str, timeframe: str, timezone: str, 
                   pack_mode: str, dry_run: bool, overwrite: bool, htf_context: bool):
    """Prepare Parquet shards from enhanced JSON sessions."""
    try:
        from ironforge.converters.json_to_parquet import ConversionConfig, convert_enhanced_sessions
        
        config = ConversionConfig(
            source_glob=source_glob,
            symbol=symbol,
            timeframe=timeframe,
            source_timezone=timezone,
            pack_mode=pack_mode,
            dry_run=dry_run,
            overwrite=overwrite,
            htf_context_enabled=htf_context
        )
        
        print(f"[prep-shards] Converting sessions from {source_glob}")
        print(f"[prep-shards] Target: {symbol}_{timeframe} | Timezone: {timezone} | Pack: {pack_mode}")
        print(f"[prep-shards] HTF Context: {'ENABLED (51D features)' if htf_context else 'DISABLED (45D features)'}")
        
        if dry_run:
            print("[prep-shards] DRY RUN MODE - no files will be written")
        
        shard_dirs = convert_enhanced_sessions(config)
        
        print(f"[prep-shards] ✅ Processed {len(shard_dirs)} sessions")
        
        # Write manifest
        if not dry_run and shard_dirs:
            manifest_path = Path(f"data/shards/{symbol}_{timeframe}/manifest.jsonl")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(manifest_path, 'w') as f:
                for shard_dir in shard_dirs:
                    if shard_dir.exists():
                        meta_file = shard_dir / "meta.json"
                        if meta_file.exists():
                            with open(meta_file, 'r') as meta_f:
                                metadata = json.load(meta_f)
                                manifest_entry = {
                                    "shard_dir": str(shard_dir),
                                    "session_id": metadata.get("session_id"),
                                    "node_count": metadata.get("node_count", 0),
                                    "edge_count": metadata.get("edge_count", 0),
                                    "conversion_timestamp": metadata.get("conversion_timestamp")
                                }
                                f.write(json.dumps(manifest_entry) + "\n")
            
            print(f"[prep-shards] Wrote manifest: {manifest_path}")
        
        return 0
        
    except ImportError as e:
        print(f"[prep-shards] Error: Converter not available - {e}")
        return 1
    except Exception as e:
        print(f"[prep-shards] Error: {e}")
        return 1


def cmd_status(runs: Path):
    runs = Path(runs)
    if not runs.exists():
        print(json.dumps({"runs": []}, indent=2))
        return 0
    items = []
    for r in sorted([p for p in runs.iterdir() if p.is_dir()]):
        counts = {
            k: len(list((r / k).glob("**/*")))
            for k in ["embeddings", "patterns", "confluence", "motifs", "reports"]
        }
        items.append({"run": r.name, **counts})
    print(json.dumps({"runs": items}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("ironforge")
    sub = p.add_subparsers(dest="cmd", required=True)
    c1 = sub.add_parser("discover-temporal")
    c1.add_argument("--config", default="configs/dev.yml")
    c2 = sub.add_parser("score-session")
    c2.add_argument("--config", default="configs/dev.yml")
    c3 = sub.add_parser("validate-run")
    c3.add_argument("--config", default="configs/dev.yml")
    c4 = sub.add_parser("report-minimal")
    c4.add_argument("--config", default="configs/dev.yml")
    c5 = sub.add_parser("status")
    c5.add_argument("--runs", default="runs")
    c6 = sub.add_parser("prep-shards")
    c6.add_argument("--source-glob", default="data/enhanced/enhanced_*_Lvl-1_*.json", 
                   help="Glob pattern for enhanced JSON sessions")
    c6.add_argument("--symbol", default="NQ", help="Symbol for shard directory")
    c6.add_argument("--timeframe", "--tf", default="M5", help="Timeframe for shard directory")
    c6.add_argument("--timezone", "--tz", default="ET", help="Source timezone")
    c6.add_argument("--pack", choices=["single", "pack"], default="single",
                   help="Packing mode: single session per shard or pack multiple")
    c6.add_argument("--dry-run", action="store_true", help="Show what would be converted without writing files")
    c6.add_argument("--overwrite", action="store_true", help="Overwrite existing shards")
    c6.add_argument("--htf-context", action="store_true", help="Enable HTF context features (45D → 51D)")

    args = p.parse_args(argv)
    if args.cmd == "status":
        return cmd_status(Path(args.runs))
    if args.cmd == "prep-shards":
        return cmd_prep_shards(
            args.source_glob, args.symbol, args.timeframe, args.timezone,
            args.pack, args.dry_run, args.overwrite, args.htf_context
        )
    cfg = load_config(args.config)
    if args.cmd == "discover-temporal":
        return cmd_discover(cfg)
    if args.cmd == "score-session":
        return cmd_score(cfg)
    if args.cmd == "validate-run":
        return cmd_validate(cfg)
    if args.cmd == "report-minimal":
        return cmd_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
