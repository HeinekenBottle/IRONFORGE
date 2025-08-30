from __future__ import annotations
from typing import Any, Dict, List


class PatternAnalyzer:
    def analyze(self, pattern_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "count": len(pattern_data),
            "avg_confidence": sum(p.get("confidence", 0.0) for p in pattern_data) / max(1, len(pattern_data)),
        }
