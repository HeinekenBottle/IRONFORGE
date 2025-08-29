from __future__ import annotations
from typing import Any, Dict, List


class StatisticalAnalysis:
    def calculate_pattern_frequency(self, motifs: List[Dict[str, Any]]) -> Dict[str, Any]:
        freq: Dict[str, int] = {}
        for m in motifs:
            key = m.get("type", "unknown")
            freq[key] = freq.get(key, 0) + 1
        return {"frequency": freq}
