"""
Pattern Intelligence Analyst Agent for IRONFORGE
================================================

Purpose: Deep pattern analysis with archaeological insights and multi-agent synthesis.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import PlanningBackedAgent
from .intelligence_engine import ArchaeologicalIntelligenceEngine
from .synthesis_tools import MultiAgentSynthesizer
from .pattern_analyzer import PatternAnalyzer


class PatternIntelligenceAnalyst(PlanningBackedAgent):
    def __init__(self, agent_name: str = "pattern_intelligence_analyst") -> None:
        super().__init__(agent_name=agent_name)
        self.engine = ArchaeologicalIntelligenceEngine()
        self.synth = MultiAgentSynthesizer()
        self.analyzer = PatternAnalyzer()

    def analyze_discovered_patterns(self, pattern_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        analysis = self.analyzer.analyze(pattern_data)
        arch = self.engine.generate_intelligence_reports(analysis)
        return {"analysis": analysis, "archaeological": arch}

    def synthesize_multi_agent_insights(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.synth.synthesize_cross_agent_insights(agent_outputs)

    def generate_archaeological_intelligence(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.generate_intelligence_reports(analysis)

    async def execute_primary_function(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute primary pattern intelligence analysis using planning context.
        
        Args:
            pattern_data: Dict containing pattern data and multi-agent outputs
            
        Returns:
            Dict containing intelligence analysis results and recommendations
        """
        results = {
            "pattern_analysis": {},
            "archaeological_intelligence": {},
            "multi_agent_synthesis": {},
            "insights": [],
            "recommendations": []
        }
        
        try:
            # Get behavior and dependencies from planning context
            behavior = await self.get_behavior_from_planning()
            dependencies = await self.get_dependencies_from_planning()
            
            # Extract configuration from planning context
            enable_multi_agent = dependencies.get("ENABLE_MULTI_AGENT_SYNTHESIS", "true").lower() == "true"
            intelligence_depth = dependencies.get("INTELLIGENCE_ANALYSIS_DEPTH", "standard")
            
            # Extract pattern data
            discovered_patterns = pattern_data.get("discovered_patterns", [])
            agent_outputs = pattern_data.get("agent_outputs", {})
            
            if not discovered_patterns:
                results["status"] = "WARNING"
                results["message"] = "No patterns provided for analysis"
                results["recommendations"].append("Ensure pattern discovery is completed before intelligence analysis")
                return results
            
            # Perform pattern analysis
            analysis_result = self.analyze_discovered_patterns(discovered_patterns)
            results["pattern_analysis"] = analysis_result["analysis"]
            results["archaeological_intelligence"] = analysis_result["archaeological"]
            
            # Generate multi-agent synthesis if enabled
            if enable_multi_agent and agent_outputs:
                synthesis_result = self.synthesize_multi_agent_insights(agent_outputs)
                results["multi_agent_synthesis"] = synthesis_result
                
                # Extract cross-agent insights
                if "insights" in synthesis_result:
                    results["insights"].extend(synthesis_result["insights"])
            
            # Generate recommendations based on behavior
            if behavior.get("PROVIDE_OPTIMIZATION_RECOMMENDATIONS", True):
                recommendations = []
                
                # Check pattern quality
                pattern_count = len(discovered_patterns)
                if pattern_count < 5:
                    recommendations.append(f"Low pattern count ({pattern_count}). Consider adjusting discovery parameters")
                
                # Check archaeological alignment
                arch_intel = results["archaeological_intelligence"]
                if arch_intel and "authenticity_score" in arch_intel:
                    auth_score = arch_intel["authenticity_score"]
                    if auth_score < 0.87:
                        recommendations.append(f"Archaeological authenticity below threshold ({auth_score:.1%}). Review pattern graduation")
                
                # Multi-agent coordination recommendations
                if enable_multi_agent and "coordination_gaps" in results["multi_agent_synthesis"]:
                    gaps = results["multi_agent_synthesis"]["coordination_gaps"]
                    if gaps:
                        recommendations.append(f"Found {len(gaps)} multi-agent coordination gaps. Consider workflow optimization")
                
                results["recommendations"] = recommendations
            
            # Extract insights from analysis
            if "key_insights" in results["pattern_analysis"]:
                results["insights"].extend(results["pattern_analysis"]["key_insights"])
            
            results["status"] = "SUCCESS"
            results["message"] = f"Analyzed {pattern_count} patterns with {len(results['insights'])} key insights generated"
            
        except Exception as e:
            results["status"] = "ERROR"
            results["message"] = f"Pattern intelligence analysis failed: {str(e)}"
            results["recommendations"].append("Check pattern data format and intelligence engine configuration")
        
        return results
