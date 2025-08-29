from __future__ import annotations
from typing import Any, Dict, List


class SessionIsolationValidator:
    def detect_boundary_violations(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        for e in edges:
            if e.get("from_session") != e.get("to_session"):
                violations.append(e)
        return violations

    def validate_session_isolation(self, learning_data: Dict[str, Any]) -> bool:
        edges = learning_data.get("edges", [])
        return len(self.detect_boundary_violations(edges)) == 0

    def ensure_htf_rule_compliance(self, features: Dict[str, Any]) -> bool:
        intra = features.get("htf_intra_candle_flags", False)
        if intra:
            raise ValueError("HTF intra-candle data detected")
        return True
