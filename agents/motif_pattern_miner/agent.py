"""
Motif Pattern Miner Agent for IRONFORGE
======================================

Purpose: Discover recurring motifs across sessions and timeframes.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import PlanningBackedAgent
from .motif_mining import RecurringPatternDetector
from .pattern_detection import HierarchicalMotifAnalyzer
from .statistical_analysis import StatisticalAnalysis


class MotifPatternMiner(PlanningBackedAgent):
    def __init__(self, agent_name: str = "motif_pattern_miner") -> None:
        super().__init__(agent_name=agent_name)
        self.detector = RecurringPatternDetector()
        self.hier = HierarchicalMotifAnalyzer()
        self.stats = StatisticalAnalysis()

    def discover_recurring_motifs(self, pattern_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        motifs = self.detector.identify_structural_patterns(pattern_data)
        stability = self.analyze_motif_stability(motifs)
        return {"motifs": motifs, "stability": stability}

    def analyze_motif_stability(self, motifs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.stats.calculate_pattern_frequency(motifs)

    def mine_cross_timeframe_patterns(self, multi_tf_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.hier.detect_dimensional_anchoring_patterns(multi_tf_data)

    async def execute_primary_function(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute primary motif pattern mining using planning context.
        
        Args:
            pattern_data: Dict containing pattern data and multi-timeframe information
            
        Returns:
            Dict containing discovered motifs and stability analysis
        """
        results = {
            "recurring_motifs": [],
            "motif_stability": {},
            "cross_timeframe_patterns": {},
            "hierarchical_analysis": {},
            "recommendations": []
        }
        
        try:
            # Get behavior and dependencies from planning context
            behavior = await self.get_behavior_from_planning()
            dependencies = await self.get_dependencies_from_planning()
            
            # Extract configuration from planning context
            enable_stability_analysis = dependencies.get("ENABLE_MOTIF_STABILITY_ANALYSIS", "true").lower() == "true"
            enable_cross_timeframe = dependencies.get("ENABLE_CROSS_TIMEFRAME_MINING", "true").lower() == "true"
            min_motif_frequency = float(dependencies.get("MIN_MOTIF_FREQUENCY", "0.3"))
            stability_threshold = float(dependencies.get("STABILITY_THRESHOLD", "0.7"))
            
            # Extract pattern data
            patterns = pattern_data.get("patterns", [])
            multi_tf_data = pattern_data.get("multi_timeframe_data", {})
            temporal_data = pattern_data.get("temporal_data", {})
            
            if not patterns:
                results["status"] = "WARNING"
                results["message"] = "No pattern data provided for motif mining"
                results["recommendations"].append("Ensure patterns are discovered before motif mining")
                return results
            
            # Discover recurring motifs
            motif_discovery = self.discover_recurring_motifs(patterns)
            results["recurring_motifs"] = motif_discovery["motifs"]
            
            # Analyze motif stability if enabled
            if enable_stability_analysis:
                stability_analysis = motif_discovery["stability"]
                results["motif_stability"] = stability_analysis
                
                # Filter stable motifs
                stable_motifs = [
                    motif for motif in results["recurring_motifs"]
                    if stability_analysis.get(motif.get("id", ""), {}).get("stability_score", 0.0) >= stability_threshold
                ]
                results["stable_motifs_count"] = len(stable_motifs)
            
            # Mine cross-timeframe patterns if enabled and data available
            if enable_cross_timeframe and multi_tf_data:
                cross_tf_analysis = self.mine_cross_timeframe_patterns(multi_tf_data)
                results["cross_timeframe_patterns"] = cross_tf_analysis
                
                # Extract hierarchical analysis if available
                if "hierarchical_motifs" in cross_tf_analysis:
                    results["hierarchical_analysis"] = cross_tf_analysis["hierarchical_motifs"]
            
            # Generate recommendations based on behavior
            if behavior.get("PROVIDE_MOTIF_RECOMMENDATIONS", True):
                recommendations = []
                
                # Motif frequency recommendations
                motif_count = len(results["recurring_motifs"])
                if motif_count < 3:
                    recommendations.append(f"Low motif count ({motif_count}). Consider lowering frequency threshold or expanding pattern dataset")
                elif motif_count > 20:
                    recommendations.append(f"High motif count ({motif_count}). Consider raising frequency threshold for focused analysis")
                
                # Stability recommendations
                if enable_stability_analysis and "motif_stability" in results:
                    avg_stability = sum(
                        analysis.get("stability_score", 0.0) 
                        for analysis in results["motif_stability"].values()
                    ) / len(results["motif_stability"]) if results["motif_stability"] else 0.0
                    
                    if avg_stability < stability_threshold:
                        recommendations.append(f"Low average motif stability ({avg_stability:.2f}). Consider refining pattern detection parameters")
                    
                    stable_count = results.get("stable_motifs_count", 0)
                    if stable_count < motif_count * 0.5:
                        recommendations.append(f"Only {stable_count}/{motif_count} motifs meet stability threshold. Review stability criteria")
                
                # Cross-timeframe recommendations
                if enable_cross_timeframe:
                    if not multi_tf_data:
                        recommendations.append("Cross-timeframe mining enabled but no multi-timeframe data provided")
                    elif "dimensional_anchors" in results["cross_timeframe_patterns"]:
                        anchor_count = len(results["cross_timeframe_patterns"]["dimensional_anchors"])
                        if anchor_count == 0:
                            recommendations.append("No dimensional anchoring patterns found. Review timeframe data quality")
                        elif anchor_count > 10:
                            recommendations.append(f"Many dimensional anchors found ({anchor_count}). Consider consolidation analysis")
                
                # Hierarchical analysis recommendations
                if "hierarchical_analysis" in results and results["hierarchical_analysis"]:
                    hierarchy_depth = results["hierarchical_analysis"].get("max_depth", 0)
                    if hierarchy_depth < 2:
                        recommendations.append("Shallow motif hierarchy detected. Consider multi-scale pattern analysis")
                    elif hierarchy_depth > 5:
                        recommendations.append("Deep motif hierarchy detected. Consider hierarchy pruning for clarity")
                
                results["recommendations"] = recommendations
            
            # Success metrics
            total_motifs = len(results["recurring_motifs"])
            stable_motifs = results.get("stable_motifs_count", 0)
            cross_tf_patterns = len(results["cross_timeframe_patterns"]) if results["cross_timeframe_patterns"] else 0
            
            results["status"] = "SUCCESS"
            results["message"] = f"Mined {total_motifs} recurring motifs ({stable_motifs} stable) with {cross_tf_patterns} cross-timeframe patterns"
            
        except Exception as e:
            results["status"] = "ERROR"
            results["message"] = f"Motif pattern mining failed: {str(e)}"
            results["recommendations"].append("Check pattern data format and motif mining configuration")
        
        return results
