from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_html(path: str | Path, html: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")


def glob_many(glob: str) -> list[Path]:
    from glob import glob as pyglob
    return [Path(p) for p in pyglob(glob)]
