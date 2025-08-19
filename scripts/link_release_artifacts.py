#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    runs = Path("runs")
    if not runs.exists():
        raise SystemExit("runs/ not found")

    # Determine date folder
    date: str
    if len(sys.argv) > 1:
        date = sys.argv[1]
        if not (runs / date).exists():
            raise SystemExit(f"runs/{date} not found")
    else:
        dates = sorted([p.name for p in runs.iterdir() if p.is_dir()])
        if not dates:
            raise SystemExit("no run dates found under runs/")
        date = dates[-1]

    p = {
        "run45": f"runs/{date}/NQ_5m/",
        "run51": f"runs/{date}/NQ_5m_htf/",
        "rep45": f"runs/{date}/report_45d/",
        "rep51": f"runs/{date}/report_51d/",
        "man45": f"runs/{date}/NQ_5m/manifest.json",
        "man51": f"runs/{date}/NQ_5m_htf/manifest.json",
    }

    notes = Path("docs/releases/1.0.0.md")
    if not notes.exists():
        raise SystemExit("docs/releases/1.0.0.md not found")

    block = (
        f"\n## Run Artifacts ({date})\n"
        f"- [45D run]({p['run45']}) — [manifest]({p['man45']})\n"
        f"- [51D run]({p['run51']}) — [manifest]({p['man51']})\n"
        f"- Reports: [45D]({p['rep45']}), [51D]({p['rep51']})\n"
    )

    notes.write_text(notes.read_text(encoding="utf-8") + block, encoding="utf-8")
    print(f"Appended artifact links for {date} to {notes}")


if __name__ == "__main__":
    main()

