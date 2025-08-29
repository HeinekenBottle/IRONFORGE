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
    """Glob files, handling both relative and absolute paths."""
    glob_path = Path(glob)
    if glob_path.is_absolute():
        # For absolute paths, use the parent directory and pattern
        parent = glob_path.parent
        pattern = glob_path.name
        return list(parent.glob(pattern))
    else:
        # For relative paths, use current directory
        return list(Path().glob(glob))
