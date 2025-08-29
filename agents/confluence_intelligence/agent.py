"""
Confluence Intelligence Agent for IRONFORGE
==========================================

Purpose: Optimize confluence scoring with adaptive weight management.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import PlanningBackedAgent
from .weight_optimizer import AdaptiveWeightOptimizer
from .scoring_engine import ConfluenceScoringEngine
from .temporal_scoring import TemporalScoringEngine


class ConfluenceIntelligence(PlanningBackedAgent):
    def __init__(self, agent_name: str = "confluence_intelligence") -> None:
        super().__init__(agent_name=agent_name)
        self.optimizer = AdaptiveWeightOptimizer()
        self.scoring = ConfluenceScoringEngine()
        self.temporal = TemporalScoringEngine()

    def optimize_scoring_weights(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        weights = self.optimizer.calculate_optimal_weights(historical_data)
        self.optimizer.update_weight_configuration(weights)
        return weights

    def adapt_weights_dynamically(self, performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        return self.optimizer.adapt(weights=self.optimizer.current_weights, metrics=performance_metrics)

    def enhance_pattern_evaluation(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        temporal = self.temporal.apply_temporal_intelligence(patterns)
        return [self.scoring.score_pattern(p, self.optimizer.current_weights) for p in temporal]

    async def execute_primary_function(self, confluence_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute primary confluence intelligence optimization using planning context.
        
        Args:
            confluence_data: Dict containing patterns, historical data, and performance metrics
            
        Returns:
            Dict containing optimized weights and enhanced pattern evaluations
        """
        results = {
            "optimized_weights": {},
            "adapted_weights": {},
            "enhanced_patterns": [],
            "performance_improvement": 0.0,
            "recommendations": []
        }
        
        try:
            # Get behavior and dependencies from planning context
            behavior = await self.get_behavior_from_planning()
            dependencies = await self.get_dependencies_from_planning()
            
            # Extract configuration from planning context
            enable_dynamic_adaptation = dependencies.get("ENABLE_DYNAMIC_WEIGHT_ADAPTATION", "true").lower() == "true"
            weight_optimization_enabled = dependencies.get("WEIGHT_OPTIMIZATION_ENABLED", "true").lower() == "true"
            temporal_intelligence_enabled = dependencies.get("TEMPORAL_INTELLIGENCE_ENABLED", "true").lower() == "true"
            
            # Extract confluence data
            patterns = confluence_data.get("patterns", [])
            historical_data = confluence_data.get("historical_data", [])
            performance_metrics = confluence_data.get("performance_metrics", {})
            
            if not patterns:
                results["status"] = "WARNING"
                results["message"] = "No patterns provided for confluence optimization"
                results["recommendations"].append("Ensure patterns are discovered before confluence optimization")
                return results
            
            # Optimize scoring weights if enabled and historical data available
            if weight_optimization_enabled and historical_data:
                optimized_weights = self.optimize_scoring_weights(historical_data)
                results["optimized_weights"] = optimized_weights
                
                # Calculate performance improvement
                if "baseline_performance" in performance_metrics:
                    baseline = performance_metrics["baseline_performance"]
                    current_performance = performance_metrics.get("current_performance", baseline)
                    results["performance_improvement"] = (current_performance - baseline) / baseline if baseline > 0 else 0.0
            
            # Adapt weights dynamically if enabled and performance metrics available
            if enable_dynamic_adaptation and performance_metrics:
                adapted_weights = self.adapt_weights_dynamically(performance_metrics)
                results["adapted_weights"] = adapted_weights
            
            # Enhance pattern evaluation
            if temporal_intelligence_enabled:
                enhanced_patterns = self.enhance_pattern_evaluation(patterns)
                results["enhanced_patterns"] = [
                    {
                        "pattern_id": i,
                        "original_score": patterns[i].get("score", 0.0),
                        "enhanced_score": enhanced_patterns[i].get("score", 0.0),
                        "temporal_features": enhanced_patterns[i].get("temporal_features", {}),
                        "confidence_boost": enhanced_patterns[i].get("confidence_boost", 0.0)
                    }
                    for i in range(min(len(patterns), len(enhanced_patterns)))
                ]
            else:
                # Simple scoring without temporal intelligence
                results["enhanced_patterns"] = [
                    {
                        "pattern_id": i,
                        "original_score": pattern.get("score", 0.0),
                        "enhanced_score": self.scoring.score_pattern(pattern, self.optimizer.current_weights).get("score", 0.0)
                    }
                    for i, pattern in enumerate(patterns)
                ]
            
            # Generate recommendations based on behavior
            if behavior.get("PROVIDE_OPTIMIZATION_RECOMMENDATIONS", True):
                recommendations = []
                
                # Weight optimization recommendations
                if weight_optimization_enabled and "optimized_weights" in results:
                    weight_changes = len([k for k, v in results["optimized_weights"].items() if abs(v - self.optimizer.current_weights.get(k, 0)) > 0.1])
                    if weight_changes > 0:
                        recommendations.append(f"Applied {weight_changes} significant weight optimizations")
                    else:
                        recommendations.append("Current weights appear well-optimized for historical data")
                
                # Performance improvement recommendations
                if results["performance_improvement"] < 0.05:  # Less than 5% improvement
                    recommendations.append("Consider reviewing confluence scoring parameters for better performance")
                elif results["performance_improvement"] > 0.2:  # More than 20% improvement
                    recommendations.append("Excellent performance improvement achieved through weight optimization")
                
                # Pattern enhancement recommendations
                if results["enhanced_patterns"]:
                    avg_enhancement = sum(p["enhanced_score"] - p["original_score"] for p in results["enhanced_patterns"]) / len(results["enhanced_patterns"])
                    if avg_enhancement < 0.1:
                        recommendations.append("Low average pattern enhancement. Consider temporal intelligence parameter tuning")
                    elif avg_enhancement > 0.5:
                        recommendations.append("High pattern enhancement achieved. Monitor for over-optimization")
                
                # Dynamic adaptation recommendations
                if enable_dynamic_adaptation and "adapted_weights" in results:
                    adaptation_magnitude = sum(abs(v) for v in results["adapted_weights"].values()) / len(results["adapted_weights"])
                    if adaptation_magnitude > 0.3:
                        recommendations.append("High weight adaptation detected. Monitor performance stability")
                
                results["recommendations"] = recommendations
            
            pattern_count = len(patterns)
            enhanced_count = len(results["enhanced_patterns"])
            
            results["status"] = "SUCCESS"
            results["message"] = f"Optimized confluence scoring for {pattern_count} patterns, enhanced {enhanced_count} evaluations with {results['performance_improvement']:.1%} improvement"
            
        except Exception as e:
            results["status"] = "ERROR"
            results["message"] = f"Confluence intelligence optimization failed: {str(e)}"
            results["recommendations"].append("Check pattern data format and confluence optimizer configuration")
        
        return results
