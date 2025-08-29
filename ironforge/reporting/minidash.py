from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_minidash(
    activity: pd.DataFrame,
    confluence: pd.DataFrame,
    motifs: list[dict[str, Any]],
    out_html: str | Path,
    out_png: str | Path,
    width: int = 1200,
    height: int = 700,
    htf_regime_data: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    if activity.empty:
        activity = pd.DataFrame(
            {
                "ts": pd.date_range("2025-01-01", periods=10, freq="min"),
                "count": list(range(10)),
            }
        )
    if confluence.empty:
        confluence = pd.DataFrame(
            {
                "ts": activity["ts"],
                "score": np.minimum(99, np.arange(len(activity)) * 10),
            }
        )

    activity = activity.sort_values("ts")
    confluence = confluence.sort_values("ts")

    fig = plt.figure(figsize=(width / 100, height / 100))
    ax1 = fig.add_axes([0.08, 0.58, 0.9, 0.35])
    ax2 = fig.add_axes([0.08, 0.12, 0.9, 0.35])

    ax1.bar(range(len(activity)), activity["count"], align="center")
    ax1.set_title("Session Activity")
    ax1.set_xticks([])
    ax1.set_ylabel("Count")

    ax2.plot(range(len(confluence)), confluence["score"])
    ax2.set_ylim(0, 100)
    ax2.set_title("Confluence (0–100)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Score")

    out_png = Path(out_png)
    out_html = Path(out_html)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    rows = "".join(
        f"<tr><td>{m.get('name','')}</td><td>{m.get('support','')}</td><td>{m.get('ppv','')}</td></tr>"
        for m in motifs
    )

    # HTF Regime Ribbon (minimal text badges)
    htf_ribbon = ""
    if htf_regime_data:
        regime_dist = htf_regime_data.get("regime_distribution", {})
        total_zones = htf_regime_data.get("total_zones", 0)
        theory_b_zones = htf_regime_data.get("theory_b_zones", 0)
        quality_score = htf_regime_data.get("quality_score", 0.0)

        # Create regime badges
        regime_badges = []
        regime_colors = {
            "consolidation": "#ffc107",
            "transition": "#17a2b8",
            "expansion": "#dc3545",
        }

        for regime, count in regime_dist.items():
            color = regime_colors.get(regime, "#6c757d")
            badge = f'<span style="background:{color};color:white;padding:2px 6px;border-radius:3px;margin:2px">{regime.title()}: {count}</span>'
            regime_badges.append(badge)

        htf_ribbon = f"""
        <div style="background:#f8f9fa;padding:10px;margin:10px 0;border-radius:5px;">
            <strong>HTF Context (v0.7.1):</strong> 
            {' '.join(regime_badges)}
            <br><small>
                Zones: {total_zones} | Theory B: {theory_b_zones} | Quality: {quality_score:.2f}
            </small>
        </div>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>body{{font:14px system-ui}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:6px}}</style>
    <h1>IRONFORGE — Minimal Report</h1>
    {htf_ribbon}
    <img src="{out_png.name}" alt="Confluence & Activity" />
    <h2>Motifs</h2><table><tr><th>Name</th><th>Support</th><th>PPV</th></tr>{rows}</table>"""
    out_html.write_text(html, encoding="utf-8")
    return out_html, out_png
