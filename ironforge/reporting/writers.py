from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def save_html(html_str: str, path: str) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)


def save_png(fig, path: str) -> None:  # type: ignore[no-untyped-def]
    """Save a Matplotlib figure as PNG."""
    _ensure_parent(path)
    fig.savefig(path)


def write_report(data: Any, output_path: str) -> None:
    """Write report data to file (JSON format)."""
    save_json(data, output_path)
