"""
Pattern Intelligence Analyst Agent for IRONFORGE
================================================

Purpose: Deep pattern analysis with archaeological insights and multi-agent synthesis.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .intelligence_engine import ArchaeologicalIntelligenceEngine
from .synthesis_tools import MultiAgentSynthesizer
from .pattern_analyzer import PatternAnalyzer


class PatternIntelligenceAnalyst:
    def __init__(self) -> None:
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
