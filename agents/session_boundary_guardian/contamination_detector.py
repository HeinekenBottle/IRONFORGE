from __future__ import annotations
from typing import Any, Dict


class ContaminationDetector:
    def scan_for_cross_session_edges(self, graph: Dict[str, Any]) -> int:
        return sum(1 for e in graph.get("edges", []) if e.get("from_session") != e.get("to_session"))

    def identify_learning_contamination(self, models: Dict[str, Any]) -> Dict[str, Any]:
        issues = models.get("issues", [])
        return {"issues_found": len(issues), "details": issues}
