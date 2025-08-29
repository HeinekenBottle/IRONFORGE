"""
HTF Cascade Predictor Agent for IRONFORGE
========================================

Purpose: HTF temporal cascade analysis with sub-second precision.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import PlanningBackedAgent
from .cascade_analyzer import TemporalCascadeAnalyzer
from .echo_detection import EchoDetector
from .temporal_tools import TemporalTools


class HTFCascadePredictor(PlanningBackedAgent):
    def __init__(self, agent_name: str = "htf_cascade_predictor") -> None:
        super().__init__(agent_name=agent_name)
        self.analyzer = TemporalCascadeAnalyzer()
        self.echo = EchoDetector()
        self.tools = TemporalTools()

    def analyze_htf_cascades(self, multi_timeframe_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.analyzer.process_f45_f50_features(multi_timeframe_data)

    def predict_cascade_timing(self, htf_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.tools.predict_timing(htf_patterns)

    def detect_temporal_echoes(self, cascade_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.echo.detect_forward_propagating_patterns(cascade_data)

    async def execute_primary_function(self, timeframe_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute primary HTF cascade prediction using planning context.
        
        Args:
            timeframe_data: Dict containing multi-timeframe data and HTF patterns
            
        Returns:
            Dict containing cascade analysis results and timing predictions
        """
        results = {
            "htf_cascades": [],
            "cascade_timing": {},
            "temporal_echoes": [],
            "precision_score": 0.0,
            "recommendations": []
        }
        
        try:
            # Get behavior and dependencies from planning context
            behavior = await self.get_behavior_from_planning()
            dependencies = await self.get_dependencies_from_planning()
            
            # Extract configuration from planning context
            enable_echo_detection = dependencies.get("ENABLE_TEMPORAL_ECHO_DETECTION", "true").lower() == "true"
            precision_threshold = float(dependencies.get("CASCADE_PRECISION_THRESHOLD", "0.8"))
            sub_second_precision = dependencies.get("SUB_SECOND_PRECISION", "true").lower() == "true"
            
            # Extract timeframe data
            multi_tf_data = timeframe_data.get("multi_timeframe_data", {})
            htf_patterns = timeframe_data.get("htf_patterns", [])
            
            if not multi_tf_data:
                results["status"] = "WARNING"
                results["message"] = "No multi-timeframe data provided for cascade analysis"
                results["recommendations"].append("Ensure HTF data (f45-f50 features) is available for cascade prediction")
                return results
            
            # Analyze HTF cascades
            cascade_analysis = self.analyze_htf_cascades(multi_tf_data)
            results["htf_cascades"] = cascade_analysis.get("cascades", [])
            
            # Predict cascade timing if patterns are available
            if htf_patterns:
                timing_prediction = self.predict_cascade_timing(htf_patterns)
                results["cascade_timing"] = timing_prediction
                
                # Extract precision score
                if "precision_score" in timing_prediction:
                    results["precision_score"] = timing_prediction["precision_score"]
            
            # Detect temporal echoes if enabled
            if enable_echo_detection and results["htf_cascades"]:
                cascade_data = {
                    "cascades": results["htf_cascades"],
                    "timing": results["cascade_timing"]
                }
                echo_analysis = self.detect_temporal_echoes(cascade_data)
                results["temporal_echoes"] = echo_analysis.get("echoes", [])
            
            # Generate recommendations based on behavior
            if behavior.get("PROVIDE_TIMING_RECOMMENDATIONS", True):
                recommendations = []
                
                # Check precision
                if results["precision_score"] < precision_threshold:
                    recommendations.append(f"Cascade precision below threshold ({results['precision_score']:.2f} < {precision_threshold})")
                
                # Check sub-second precision requirement
                if sub_second_precision and "timing_error" in results["cascade_timing"]:
                    timing_error = results["cascade_timing"]["timing_error"]
                    if timing_error > 1.0:  # More than 1 second error
                        recommendations.append(f"Timing error exceeds sub-second requirement: {timing_error:.2f}s")
                
                # Echo detection recommendations
                if enable_echo_detection and not results["temporal_echoes"]:
                    recommendations.append("No temporal echoes detected. Consider expanding echo detection parameters")
                
                # HTF feature utilization
                if "f45_f50_utilization" in cascade_analysis:
                    utilization = cascade_analysis["f45_f50_utilization"]
                    if utilization < 0.8:
                        recommendations.append(f"Low HTF feature utilization ({utilization:.1%}). Review f45-f50 feature extraction")
                
                results["recommendations"] = recommendations
            
            cascade_count = len(results["htf_cascades"])
            echo_count = len(results["temporal_echoes"])
            
            results["status"] = "SUCCESS"
            results["message"] = f"Analyzed {cascade_count} HTF cascades with {echo_count} temporal echoes, {results['precision_score']:.2f} precision"
            
        except Exception as e:
            results["status"] = "ERROR"
            results["message"] = f"HTF cascade prediction failed: {str(e)}"
            results["recommendations"].append("Check multi-timeframe data format and cascade analyzer configuration")
        
        return results
