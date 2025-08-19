#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import glob
from pathlib import Path

FAIL = 1
OK = 0


def find_one(patterns: list[str]) -> str | None:
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    return None


def count_feat_cols(cols: list[str]) -> int:
    return sum(1 for c in cols if re.fullmatch(r"f\d+", c))


def read_schema_cols(parquet_path: str) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(parquet_path).column_names
    except Exception as e:  # pragma: no cover
        print(
            json.dumps(
                {"status": "error", "step": "read_schema", "path": parquet_path, "error": str(e)}
            )
        )
        sys.exit(FAIL)


def main() -> None:
    ap = argparse.ArgumentParser(description="IRONFORGE contracts validation")
    ap.add_argument("run_dir", help="Path to a single run directory (e.g., runs/2025-08-19/NQ_5m/)")
    ap.add_argument("--expect-node-dims", dest="expect_node_dims", type=int, choices=[45, 51])
    ap.add_argument("--name", default=None, help="Label for this check in output")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out: dict[str, object] = {"name": args.name or run_dir.name, "run_dir": str(run_dir), "checks": {}}

    # Layout checks
    manifest = run_dir / "manifest.json"
    out["checks"]["manifest_exists"] = manifest.exists()

    # Try common artifact locations
    node_candidate = find_one(
        [
            str(run_dir / "patterns" / "*.parquet"),
            str(run_dir / "embeddings" / "*.parquet"),
            str(run_dir / "*.parquet"),
        ]
    )
    edge_candidate = find_one(
        [
            str(run_dir / "edges" / "*.parquet"),
            str(run_dir / "patterns" / "*edges*.parquet"),
        ]
    )

    # Node dims
    if node_candidate:
        cols = read_schema_cols(node_candidate)
        node_dim = count_feat_cols(cols)
        out["checks"]["node_feature_dim"] = node_dim
        if args.expect_node_dims is not None:
            out["checks"]["node_dim_ok"] = node_dim == args.expect_node_dims
        else:
            out["checks"]["node_dim_ok"] = node_dim in (45, 51)
    else:
        out["checks"]["node_feature_dim"] = None
        out["checks"]["node_dim_ok"] = False

    # Edge dims (expect 20)
    if edge_candidate:
        cols_e = read_schema_cols(edge_candidate)
        edge_dim = count_feat_cols(cols_e)
        out["checks"]["edge_feature_dim"] = edge_dim
        out["checks"]["edge_dim_ok"] = edge_dim == 20
    else:
        out["checks"]["edge_feature_dim"] = None
        out["checks"]["edge_dim_ok"] = False

    # Taxonomy/intents presence (best-effort)
    TAX_KEYS = {"event", "taxonomy_event", "label_event"}
    INT_KEYS = {"intent", "edge_intent", "label_intent"}

    def has_any(keys: set[str], cols: set[str]) -> bool:
        return any(k in cols for k in keys)

    tax_ok = False
    intent_ok = False
    if node_candidate:
        cols = set(read_schema_cols(node_candidate))
        tax_ok = has_any(TAX_KEYS, cols)
        intent_ok = has_any(INT_KEYS, cols)
    out["checks"]["taxonomy_cols_present"] = tax_ok
    out["checks"]["intent_cols_present"] = intent_ok

    # Reports exist (either flavor)
    report45 = run_dir.parent / "report_45d"
    report51 = run_dir.parent / "report_51d"
    out["checks"]["report_dirs_present"] = any(p.exists() for p in (report45, report51))

    # Final verdict
    required = [
        bool(out["checks"].get("manifest_exists")),
        bool(out["checks"].get("node_dim_ok")),
        bool(out["checks"].get("edge_dim_ok")),
    ]
    out["status"] = "ok" if all(required) else "fail"
    print(json.dumps(out, indent=2))
    sys.exit(OK if out["status"] == "ok" else FAIL)


if __name__ == "__main__":
    main()

