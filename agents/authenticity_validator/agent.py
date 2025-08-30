"""
Authenticity Validator Agent for IRONFORGE
=========================================

Purpose: Validate >87% authenticity threshold for pattern graduation.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import PlanningBackedAgent
from .scoring import calculate_authenticity
from .graduation import PatternGraduator
from .quality_gates import QualityGateManager


class AuthenticityValidator(PlanningBackedAgent):
    def __init__(self, threshold: float = 87.0, agent_name: str = "authenticity_validator") -> None:
        super().__init__(agent_name=agent_name)
        self.threshold = threshold
        self.graduator = PatternGraduator(threshold)
        self.quality = QualityGateManager()

    def validate_authenticity_threshold(self, patterns: List[Dict[str, Any]], threshold: float | None = None) -> List[Dict[str, Any]]:
        thr = float(threshold if threshold is not None else self.threshold)
        results: List[Dict[str, Any]] = []
        for p in patterns:
            score = calculate_authenticity(p)
            results.append({**p, "authenticity": {"score": score, "passed": score >= thr}})
        return results

    def graduate_patterns(self, validated_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.graduator.apply_graduation_criteria(validated_patterns)

    def generate_validation_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.quality.track_quality_metrics(results)

    async def execute_primary_function(self, patterns_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the primary authenticity validation function using planning context.
        
        Args:
            patterns_data: Data containing patterns to validate for authenticity
            
        Returns:
            Dict containing validation results, graduated patterns, and reports
        """
        results = {
            "validation_completed": False,
            "patterns_validated": [],
            "patterns_graduated": [],
            "authenticity_report": {},
            "threshold_status": "unknown",
            "recommendations": []
        }
        
        try:
            # Get behavior and dependencies from planning documents
            behavior = await self.get_behavior_from_planning()
            dependencies = await self.get_dependencies_from_planning()
            
            # Extract configuration from dependencies
            threshold = float(dependencies.get("AUTHENTICITY_THRESHOLD", str(self.threshold)))
            production_threshold = float(dependencies.get("PRODUCTION_THRESHOLD", "90.0"))
            strict_mode = dependencies.get("QUALITY_GATE_STRICT_MODE", "true").lower() == "true"
            graduation_enabled = dependencies.get("GRADUATION_WORKFLOW_ENABLED", "true").lower() == "true"
            
            # Extract patterns from input data
            patterns = patterns_data.get("patterns", [])
            if not patterns:
                results["status"] = "ERROR"
                results["message"] = "No patterns provided for validation"
                return results
            
            # Validate authenticity threshold
            validated_patterns = self.validate_authenticity_threshold(patterns, threshold)
            results["patterns_validated"] = validated_patterns
            
            # Calculate pass/fail statistics
            passed_patterns = [p for p in validated_patterns if p.get("authenticity", {}).get("passed", False)]
            failed_patterns = [p for p in validated_patterns if not p.get("authenticity", {}).get("passed", False)]
            
            results["threshold_status"] = f"{len(passed_patterns)}/{len(patterns)} patterns passed {threshold}% threshold"
            
            # Graduate patterns if enabled
            if graduation_enabled and passed_patterns:
                graduated_patterns = self.graduate_patterns(passed_patterns)
                results["patterns_graduated"] = graduated_patterns
                
                # Check production threshold
                production_ready = [p for p in graduated_patterns 
                                  if p.get("authenticity", {}).get("score", 0) >= production_threshold]
                
                if production_ready:
                    results["recommendations"].append(f"{len(production_ready)} patterns ready for production deployment")
                else:
                    results["recommendations"].append("No patterns meet production threshold requirements")
            
            # Generate validation report
            validation_report = self.generate_validation_report(validated_patterns)
            results["authenticity_report"] = validation_report
            
            # Apply strict mode logic
            if strict_mode and failed_patterns:
                results["status"] = "FAILED"
                results["message"] = f"Strict mode: {len(failed_patterns)} patterns failed authenticity threshold"
                results["recommendations"].append("Review and improve failed patterns before resubmission")
            else:
                results["status"] = "PASSED" 
                results["message"] = f"Authenticity validation completed successfully"
            
            results["validation_completed"] = True
            
            # Add improvement recommendations
            if failed_patterns:
                results["recommendations"].append("Consider pattern discovery refinement for failed patterns")
            
            if len(passed_patterns) < len(patterns) * 0.5:  # Less than 50% pass rate
                results["recommendations"].append("Low pass rate suggests need for discovery pipeline optimization")
                
        except Exception as e:
            results["status"] = "ERROR"
            results["message"] = f"Validation error: {str(e)}"
            results["recommendations"].append(f"System error during validation: {str(e)}")
        
        return results
