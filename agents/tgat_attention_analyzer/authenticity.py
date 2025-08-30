from __future__ import annotations
from typing import Any, Dict, List


class AuthenticityScorer:
    def score_pattern_authenticity(self, pattern: Dict[str, Any], threshold: float = 87.0) -> Dict[str, Any]:
        score = float(pattern.get("confidence", 0.0)) * 100.0
        return {"score": score, "passed": score >= threshold}

    def validate_graduation_criteria(self, scores: List[Dict[str, Any]]) -> bool:
        return all(s.get("passed", False) for s in scores)
