"""
Confluence Intelligence Agent for IRONFORGE
==========================================

Purpose: Optimize confluence scoring with adaptive weight management.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .weight_optimizer import AdaptiveWeightOptimizer
from .scoring_engine import ConfluenceScoringEngine
from .temporal_scoring import TemporalScoringEngine


class ConfluenceIntelligence:
    def __init__(self) -> None:
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
