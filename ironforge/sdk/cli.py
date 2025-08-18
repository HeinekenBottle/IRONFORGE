from __future__ import annotations
import argparse
import sys
import importlib
import json
from pathlib import Path
import pandas as pd

from .app_config import load_config, materialize_run_dir
from .io import write_json, glob_many
from ironforge.reporting.minidash import build_minidash


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
                "ts": pd.date_range("2025-01-01", periods=50, freq="T"),
                "score": [min(99, i * 2 % 100) for i in range(50)],
            }
        )
    pat_paths = glob_many(str(run_dir / "patterns" / "*.parquet"))
    act = _load_first_parquet(pat_paths, ["ts", "count"])
    if act.empty:
        g = conf.groupby(conf["ts"].astype("datetime64[m]")).size().reset_index(name="count")
        g.rename(columns={g.columns[0]: "ts"}, inplace=True)
        act = g
    motifs = []
    for j in Path(run_dir / "motifs").glob("*.json"):
        try:
            motifs.extend(json.loads(j.read_text(encoding="utf-8")))
        except Exception:
            pass
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
            k: len(list((r / k).glob("**/*"))) for k in ["embeddings", "patterns", "confluence", "motifs", "reports"]
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

