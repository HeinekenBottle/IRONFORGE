"""
Authenticity Validator Agent for IRONFORGE
=========================================

Purpose: Validate >87% authenticity threshold for pattern graduation.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .scoring import calculate_authenticity
from .graduation import PatternGraduator
from .quality_gates import QualityGateManager


class AuthenticityValidator:
    def __init__(self, threshold: float = 87.0) -> None:
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
