"""
Session Boundary Guardian Agent for IRONFORGE
============================================

Purpose: Enforce session isolation and prevent cross-session contamination.
"""
from __future__ import annotations

from typing import Any, Dict

from .boundary_validator import SessionIsolationValidator
from .contamination_detector import ContaminationDetector


class SessionBoundaryGuardian:
    def __init__(self) -> None:
        self.validator = SessionIsolationValidator()
        self.detector = ContaminationDetector()

    def validate_session_isolation(self, graph_data: Dict[str, Any]) -> bool:
        return self.validator.validate_session_isolation(graph_data)

    def enforce_boundary_constraints(self, pipeline_stage: str) -> bool:
        return True

    def audit_cross_session_contamination(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        return self.detector.identify_learning_contamination(analysis)
