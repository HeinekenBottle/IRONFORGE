from __future__ import annotations
from typing import Any, Dict, List


class PatternGraduator:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def apply_graduation_criteria(self, pattern: Dict[str, Any] | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patterns = pattern if isinstance(pattern, list) else [pattern]
        graduated: List[Dict[str, Any]] = []
        for p in patterns:
            auth = p.get("authenticity")
            if isinstance(auth, dict):
                if float(auth.get("score", 0.0)) >= self.threshold and bool(auth.get("passed", False)):
                    graduated.append({**p, "graduated": True})
            else:
                # If not validated yet, compute via naive confidence
                score = float(p.get("confidence", 0.0)) * 100.0
                if score >= self.threshold:
                    graduated.append({**p, "graduated": True, "authenticity": {"score": score, "passed": True}})
        return graduated
