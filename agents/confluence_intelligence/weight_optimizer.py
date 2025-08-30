from __future__ import annotations
from typing import Any, Dict, List


class AdaptiveWeightOptimizer:
    def __init__(self) -> None:
        self.current_weights: Dict[str, float] = {
            "authenticity": 0.4,
            "temporal": 0.3,
            "archaeological": 0.15,
            "other": 0.15,
        }

    def calculate_optimal_weights(self, pattern_outcomes: List[Dict[str, Any]]) -> Dict[str, float]:
        # Simple heuristic: favor accuracy proxy
        success_rate = (
            sum(1 for p in pattern_outcomes if p.get("success", False)) / max(1, len(pattern_outcomes))
        )
        return {
            **self.current_weights,
            "authenticity": min(0.6, 0.4 + success_rate * 0.2),
            "other": max(0.05, 0.15 - success_rate * 0.1),
        }

    def update_weight_configuration(self, new_weights: Dict[str, float]) -> None:
        self.current_weights = new_weights

    def validate_weight_effectiveness(self, results: List[Dict[str, Any]]) -> bool:
        return True

    def adapt(self, weights: Dict[str, float], metrics: Dict[str, Any]) -> Dict[str, float]:
        adjusted = dict(weights)
        if metrics.get("latency_ms", 0) > 2000:
            adjusted["other"] = min(0.25, adjusted.get("other", 0.15) + 0.05)
        return adjusted
