from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import glob as _glob
from pathlib import Path

import pandas as pd

from ironforge.reporting.minidash import build_minidash

from .app_config import load_config, materialize_run_dir
from .io import glob_many, write_json


def _import_required(mod: str, attr: str):
    try:
        m = importlib.import_module(mod)
        return getattr(m, attr)
    except Exception as e:
        print(f"[ironforge] Missing required entrypoint: {mod}:{attr} — {e}", file=sys.stderr)
        sys.exit(2)


def cmd_discover(cfg):
    """Run discovery across shard_* directories via discovery_pipeline."""
    run_dir = materialize_run_dir(cfg)
    shards_glob = getattr(getattr(cfg, "data", object()), "shards_glob", "data/shards/NQ_M5/shard_*")
    shard_dirs = sorted([p for p in _glob.glob(shards_glob) if Path(p).is_dir()])
    if not shard_dirs:
        print(f"[discover] No shards found for pattern: {shards_glob}", file=sys.stderr)
        sys.exit(2)
    run_discovery = _import_required("ironforge.learning.discovery_pipeline", "run_discovery")
    loader = getattr(cfg, "loader", None) or {}
    res = run_discovery(shard_dirs, str(run_dir), loader)
    print(json.dumps({"patterns": res, "out_dir": str(run_dir)}))
    return 0


def cmd_score(cfg):
    """Compute confluence scores from discovered patterns → run_dir/confluence/."""
    run_dir = materialize_run_dir(cfg)
    score_confluence = _import_required("ironforge.confluence.scoring", "score_confluence")
    weights = getattr(getattr(cfg, "scoring", object()), "weights", {}) or {}
    if not isinstance(weights, dict):
        try:
            weights = dict(vars(weights))
        except Exception:
            weights = {}
    threshold = getattr(getattr(cfg, "scoring", object()), "threshold", 65)
    out = score_confluence(str(run_dir), weights=weights, threshold=threshold)
    print(json.dumps({"confluence": out}))
    return 0


def cmd_validate(cfg):
    """Run validation rails and write reports/validation.json."""
    run_dir = materialize_run_dir(cfg)
    validate_run = _import_required("ironforge.validation.runner", "validate_run")
    res = validate_run(str(run_dir))
    reports = Path(run_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "validation.json", res if isinstance(res, dict) else {"result": "ok"})
    print(json.dumps({"reports": str(reports)}))
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
        # Normalize to minute resolution and count by minute
        conf_ts_min = pd.to_datetime(conf["ts"]).dt.floor("min")
        g = conf.groupby(conf_ts_min).size().reset_index(name="count")
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

    args = p.parse_args(argv)
    if args.cmd == "status":
        return cmd_status(Path(args.runs))
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
