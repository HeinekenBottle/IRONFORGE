#!/usr/bin/env python3
"""
Generate a small machine-readable fact sheet for agents.

Writes: docs/context.json
- Reads configs/dev.yml if available (PyYAML optional).
- Falls back to NQ/M5 defaults if config not present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULTS = {
    "symbol": "NQ",
    "timeframe": "M5",
    "shards_glob": "data/shards/NQ_M5/shard_*",
    "threshold": 65,
}


def load_cfg_defaults():
    cfg_path = Path("configs/dev.yml")
    symbol = DEFAULTS["symbol"]
    tf = DEFAULTS["timeframe"]
    shards = DEFAULTS["shards_glob"]
    threshold = DEFAULTS["threshold"]

    if cfg_path.exists():
        try:
            import yaml  # type: ignore
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            # Best-effort extraction
            symbol = (
                (cfg.get("paths", {}) or {}).get("symbol")
                or (cfg.get("data", {}) or {}).get("symbol")
                or symbol
            )
            tf = (
                (cfg.get("paths", {}) or {}).get("timeframe")
                or (cfg.get("data", {}) or {}).get("timeframe")
                or tf
            )
            shards = (
                (cfg.get("paths", {}) or {}).get("shards_dir")
                or (cfg.get("data", {}) or {}).get("shards_glob")
                or shards
            )
            threshold = (
                (cfg.get("scoring", {}) or {}).get("threshold")
                or (cfg.get("confluence", {}) or {}).get("threshold")
                or threshold
            )
        except Exception:
            pass
    return symbol, tf, shards, threshold


def main():
    symbol, tf, shards, threshold = load_cfg_defaults()

    fact = {
        "version": "0.7.1",
        "cli_commands": [
            "discover-temporal",
            "score-session",
            "validate-run",
            "report-minimal",
            "status",
        ],
        "defaults": {
            "symbol": symbol,
            "timeframe": tf,
            "shards_glob": shards,
            "threshold": threshold,
        },
        "entrypoints": {
            "discovery": "ironforge.learning.discovery_pipeline:run_discovery",
            "confluence": "ironforge.confluence.scoring:score_confluence",
            "validation": "ironforge.validation.runner:validate_run",
        },
        "nodes": {
            "dims": 51,
            "htf_v1_1": [
                "f45_sv_m15_z",
                "f46_sv_h1_z",
                "f47_barpos_m15",
                "f48_barpos_h1",
                "f49_dist_daily_mid",
                "f50_htf_regime",
            ],
        },
        "edges": {"dims": 20},
        "contracts": {
            "shard_dir": "data/shards/<SYMBOL_TF>/shard_*/",
            "shard_files": ["nodes.parquet", "edges.parquet"],
            "run_dir": "runs/YYYY-MM-DD/",
        },
    }

    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "context.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(fact, f, indent=2, sort_keys=True)
    print(f"[context] wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

