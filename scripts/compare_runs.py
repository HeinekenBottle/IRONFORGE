#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load_manifest(run_glob: str) -> tuple[Path, dict]:
    matches = sorted(glob.glob(run_glob))
    if not matches:
        raise SystemExit(f"no runs matched: {run_glob}")
    run = Path(matches[0])
    mf = run / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"manifest not found: {mf}")
    data = json.loads(mf.read_text(encoding="utf-8"))
    return run, data


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two IRONFORGE run manifests")
    ap.add_argument("run_a", help="Run A (glob), e.g. runs/2025-08-19/NQ_5m_*")
    ap.add_argument("run_b", help="Run B (glob)")
    ap.add_argument("--out", default="compare.html", help="Output HTML path")
    args = ap.parse_args()

    ra, a = load_manifest(args.run_a)
    rb, b = load_manifest(args.run_b)

    def g(d: dict, path: list[str], default=None):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    rows = []
    keys = [
        ("node_feature_dim", ["invariants", "node_feature_dim"]),
        ("edge_feature_dim", ["invariants", "edge_feature_dim"]),
        ("candidates", ["counts", "candidates"]),
        ("confluence_rows", ["counts", "confluence_rows"]),
        ("motifs", ["counts", "motifs"]),
        ("reports_present", ["counts", "reports_present"]),
    ]
    for label, path in keys:
        va = g(a, path)
        vb = g(b, path)
        rows.append((label, va, vb, (None if va is None or vb is None else (vb - va if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None))))

    html_rows = "\n".join(
        f"<tr><td>{k}</td><td>{va}</td><td>{vb}</td><td>{'%.2f'%delta if isinstance(delta,(int,float)) else ''}</td></tr>"
        for k, va, vb, delta in rows
    )
    html = f"""
    <html><head><meta charset='utf-8'><style>
    body{{font:14px system-ui}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:6px}}
    </style></head><body>
    <h1>IRONFORGE — Run Comparator</h1>
    <p>A: {ra}</p>
    <p>B: {rb}</p>
    <table><tr><th>Metric</th><th>A</th><th>B</th><th>Δ (B−A)</th></tr>
    {html_rows}
    </table>
    </body></html>
    """
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()

