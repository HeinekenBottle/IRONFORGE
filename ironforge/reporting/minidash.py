from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt


def build_minidash(
    activity: pd.DataFrame,
    confluence: pd.DataFrame,
    motifs: List[Dict[str, Any]],
    out_html: str | Path,
    out_png: str | Path,
    width: int = 1200,
    height: int = 700,
) -> tuple[Path, Path]:
    if activity.empty:
        activity = pd.DataFrame(
            {
                "ts": pd.date_range("2025-01-01", periods=10, freq="T"),
                "count": list(range(10)),
            }
        )
    if confluence.empty:
        confluence = pd.DataFrame(
            {
                "ts": activity["ts"],
                "score": [min(99, i * 10) for i in range(len(activity))],
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
    html = f"""<!doctype html><meta charset="utf-8">
    <style>body{{font:14px system-ui}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:6px}}</style>
    <h1>IRONFORGE — Minimal Report</h1>
    <img src="{out_png.name}" alt="Confluence & Activity" />
    <h2>Motifs</h2><table><tr><th>Name</th><th>Support</th><th>PPV</th></tr>{rows}</table>"""
    out_html.write_text(html, encoding="utf-8")
    return out_html, out_png

