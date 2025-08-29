"""
Motif Pattern Miner Agent for IRONFORGE
======================================

Purpose: Discover recurring motifs across sessions and timeframes.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .motif_mining import RecurringPatternDetector
from .pattern_detection import HierarchicalMotifAnalyzer
from .statistical_analysis import StatisticalAnalysis


class MotifPatternMiner:
    def __init__(self) -> None:
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
