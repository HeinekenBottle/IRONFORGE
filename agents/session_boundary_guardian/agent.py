"""
Session Boundary Guardian Agent for IRONFORGE
============================================

Purpose: Enforce session isolation and prevent cross-session contamination.
"""
from __future__ import annotations

from typing import Any, Dict

from ..base import PlanningBackedAgent
from .boundary_validator import SessionIsolationValidator
from .contamination_detector import ContaminationDetector


class SessionBoundaryGuardian(PlanningBackedAgent):
    def __init__(self, agent_name: str = "session_boundary_guardian") -> None:
        super().__init__(agent_name=agent_name)
        self.validator = SessionIsolationValidator()
        self.detector = ContaminationDetector()

    def validate_session_isolation(self, graph_data: Dict[str, Any]) -> bool:
        return self.validator.validate_session_isolation(graph_data)

    def enforce_boundary_constraints(self, pipeline_stage: str) -> bool:
        return True

    def audit_cross_session_contamination(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        return self.detector.identify_learning_contamination(analysis)

    async def execute_primary_function(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the primary session boundary validation function using planning context.
        
        Args:
            validation_data: Data to validate for session boundary compliance
            
        Returns:
            Dict containing validation results and recommendations
        """
        results = {
            "session_isolation_valid": False,
            "boundary_constraints_enforced": False,
            "contamination_detected": False,
            "violations": [],
            "recommendations": []
        }
        
        try:
            # Get behavior from planning documents
            behavior = await self.get_behavior_from_planning()
            dependencies = await self.get_dependencies_from_planning()
            
            # Extract validation parameters from dependencies
            strict_mode = dependencies.get("BOUNDARY_VALIDATION_STRICT", "true").lower() == "true"
            contamination_enabled = dependencies.get("CONTAMINATION_DETECTION_ENABLED", "true").lower() == "true"
            
            # Perform session isolation validation
            if "graph_data" in validation_data:
                isolation_valid = self.validate_session_isolation(validation_data["graph_data"])
                results["session_isolation_valid"] = isolation_valid
                
                if not isolation_valid:
                    results["violations"].append("Session isolation violation detected in graph structure")
                    results["recommendations"].append("Review graph construction for cross-session edges")
            
            # Enforce boundary constraints
            if "pipeline_stage" in validation_data:
                constraints_enforced = self.enforce_boundary_constraints(validation_data["pipeline_stage"])
                results["boundary_constraints_enforced"] = constraints_enforced
                
                if not constraints_enforced:
                    results["violations"].append("Boundary constraints could not be enforced")
                    results["recommendations"].append("Investigate boundary constraint configuration")
            
            # Check for contamination if enabled
            if contamination_enabled and "analysis_data" in validation_data:
                contamination_result = self.audit_cross_session_contamination(validation_data["analysis_data"])
                contamination_detected = len(contamination_result.get("violations", [])) > 0
                results["contamination_detected"] = contamination_detected
                
                if contamination_detected:
                    results["violations"].extend(contamination_result.get("violations", []))
                    results["recommendations"].append("Implement contamination remediation measures")
            
            # Apply strict mode logic
            if strict_mode and results["violations"]:
                results["status"] = "FAILED"
                results["message"] = "Strict mode: Any violations result in failure"
            else:
                results["status"] = "PASSED"
                results["message"] = "Session boundary validation completed"
                
        except Exception as e:
            results["status"] = "ERROR"
            results["message"] = f"Validation error: {str(e)}"
            results["violations"].append(f"System error during validation: {str(e)}")
        
        return results
