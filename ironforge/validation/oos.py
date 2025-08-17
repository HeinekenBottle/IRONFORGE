from __future__ import annotations

import json
from pathlib import Path


def run_oos(run_dir: str) -> dict[str, str]:
    """Stub out-of-sample validation that writes a report."""
    report = {"status": "ok"}
    out_path = Path(run_dir) / "reports"
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return report
