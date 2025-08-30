from __future__ import annotations
from typing import Any, Dict, List


def summarize_patterns(patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(patterns),
        "avg_confidence": sum(p.get("confidence", 0.0) for p in patterns) / max(1, len(patterns)),
    }
