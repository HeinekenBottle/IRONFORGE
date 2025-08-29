"""
TGAT Attention Analyzer Agent for IRONFORGE
==========================================

Purpose: Analyze TGAT attention weights and validate pattern authenticity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from ..base import PlanningBackedAgent

logger = logging.getLogger(__name__)


class TGATAttentionAnalyzer(PlanningBackedAgent):
    def __init__(self, agent_name: str = "tgat_attention_analyzer") -> None:
        super().__init__(agent_name=agent_name)
        from .attention_tools import AttentionWeightProcessor
        from .authenticity import AuthenticityScorer

        self.weight_processor = AttentionWeightProcessor()
        self.authenticity = AuthenticityScorer()

    def analyze_attention_weights(self, tgat_output: Dict[str, Any]) -> Dict[str, Any]:
        weights = self.weight_processor.extract_attention_matrices(tgat_output)
        scores = self.weight_processor.calculate_attention_scores(weights)
        relationships = self.weight_processor.identify_key_relationships(scores)
        return {
            "weights": weights,
            "scores": scores,
            "relationships": relationships,
        }

    def validate_pattern_authenticity(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                **p,
                "authenticity": self.authenticity.score_pattern_authenticity(p),
            }
            for p in patterns
        ]

    def interpret_attention_patterns(self, weights: Any) -> Dict[str, Any]:
        # Simple summarization for now
        return {"num_layers": getattr(weights, "num_layers", None)}

    def generate_explainability_report(self, analysis: Dict[str, Any]) -> str:
        return f"Attention Analysis: relationships={len(analysis.get('relationships', []))}"

    async def execute_primary_function(self, tgat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the primary TGAT attention analysis function using planning context.
        
        Args:
            tgat_data: Data containing TGAT outputs and patterns for analysis
            
        Returns:
            Dict containing attention analysis, authenticity validation, and reports
        """
        results = {
            "analysis_completed": False,
            "attention_analysis": {},
            "pattern_authenticity": [],
            "interpretability": {},
            "explainability_report": "",
            "authenticity_summary": {},
            "recommendations": []
        }
        
        try:
            # Get behavior and dependencies from planning documents
            behavior = await self.get_behavior_from_planning()
            dependencies = await self.get_dependencies_from_planning()
            
            # Extract configuration from dependencies
            authenticity_threshold = float(dependencies.get("AUTHENTICITY_THRESHOLD", "87.0"))
            deep_analysis_enabled = dependencies.get("DEEP_ANALYSIS_ENABLED", "true").lower() == "true"
            interpretability_enabled = dependencies.get("INTERPRETABILITY_ANALYSIS_ENABLED", "true").lower() == "true"
            explainability_enabled = dependencies.get("EXPLAINABILITY_REPORTING_ENABLED", "true").lower() == "true"
            
            # Analyze TGAT attention weights
            if "tgat_output" in tgat_data:
                attention_analysis = self.analyze_attention_weights(tgat_data["tgat_output"])
                results["attention_analysis"] = attention_analysis
                logger.info(f"Attention analysis completed: {len(attention_analysis.get('relationships', []))} relationships identified")
            
            # Validate pattern authenticity if patterns provided
            if "patterns" in tgat_data:
                patterns = tgat_data["patterns"]
                authenticated_patterns = self.validate_pattern_authenticity(patterns)
                results["pattern_authenticity"] = authenticated_patterns
                
                # Calculate authenticity summary
                total_patterns = len(authenticated_patterns)
                passed_patterns = [p for p in authenticated_patterns 
                                 if p.get("authenticity", {}).get("score", 0) >= authenticity_threshold]
                
                results["authenticity_summary"] = {
                    "total_patterns": total_patterns,
                    "passed_threshold": len(passed_patterns),
                    "pass_rate": len(passed_patterns) / total_patterns if total_patterns > 0 else 0,
                    "threshold": authenticity_threshold
                }
                
                if len(passed_patterns) < total_patterns * 0.87:  # Less than 87% pass rate
                    results["recommendations"].append("Low authenticity pass rate suggests need for TGAT model optimization")
            
            # Generate interpretability analysis if enabled
            if interpretability_enabled and results["attention_analysis"]:
                attention_weights = results["attention_analysis"].get("weights")
                if attention_weights:
                    interpretability = self.interpret_attention_patterns(attention_weights)
                    results["interpretability"] = interpretability
            
            # Generate explainability report if enabled
            if explainability_enabled and results["attention_analysis"]:
                explainability_report = self.generate_explainability_report(results["attention_analysis"])
                results["explainability_report"] = explainability_report
            
            # Add performance recommendations
            if results["attention_analysis"]:
                relationships_count = len(results["attention_analysis"].get("relationships", []))
                if relationships_count < 5:
                    results["recommendations"].append("Low attention relationship count may indicate insufficient model complexity")
                elif relationships_count > 100:
                    results["recommendations"].append("High attention relationship count may indicate model overfitting")
            
            results["analysis_completed"] = True
            results["status"] = "PASSED"
            results["message"] = "TGAT attention analysis completed successfully"
            
        except Exception as e:
            logger.error(f"TGAT attention analysis error: {str(e)}")
            results["status"] = "ERROR"
            results["message"] = f"Analysis error: {str(e)}"
            results["recommendations"].append(f"System error during analysis: {str(e)}")
        
        return results
