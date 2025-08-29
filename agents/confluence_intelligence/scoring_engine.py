from __future__ import annotations
from typing import Any, Dict


class ConfluenceScoringEngine:
    def score_pattern(self, pattern: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
        authenticity = float(pattern.get("authenticity", {}).get("score", pattern.get("confidence", 0.0) * 100.0))
        temporal = float(pattern.get("temporal", 0.0)) * 100.0
        archaeological = float(pattern.get("archaeological", 0.0)) * 100.0
        other = float(pattern.get("other", 0.0)) * 100.0
        score = (
            authenticity * weights.get("authenticity", 0.4)
            + temporal * weights.get("temporal", 0.3)
            + archaeological * weights.get("archaeological", 0.15)
            + other * weights.get("other", 0.15)
        )
        return {**pattern, "confluence_score": score}
