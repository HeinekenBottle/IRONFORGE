"""
Pattern Graduation System
87% threshold validation pipeline for discovered patterns
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PatternGraduation:
    """
    Validation system ensuring discovered patterns exceed 87% baseline accuracy
    """

    def __init__(self, baseline_accuracy: float = 0.87):
        self.baseline_accuracy = baseline_accuracy
        # REMOVED: self.validation_history = [] to ensure session independence
        logger.info(f"Pattern Graduation initialized with {baseline_accuracy*100}% threshold")

    def validate_patterns(self, discovered_patterns: dict[str, Any]) -> dict[str, Any]:
        """
        Validate discovered patterns against 87% accuracy threshold

        Args:
            discovered_patterns: Results from TGAT discovery

        Returns:
            Validation results with graduation status
        """
        try:
            session_name = discovered_patterns.get("session_name", "unknown")
            significant_patterns = discovered_patterns.get("significant_patterns", [])
            discovered_patterns.get("session_metrics", {})

            logger.info(f"Validating {len(significant_patterns)} patterns from {session_name}")

            # Calculate pattern validation metrics
            validation_results = self._calculate_validation_metrics(discovered_patterns)

            # Apply graduation criteria
            graduation_status = self._apply_graduation_criteria(validation_results)

            # Prepare graduation results
            results = {
                "session_name": session_name,
                "validation_timestamp": datetime.now().isoformat(),
                "baseline_threshold": self.baseline_accuracy,
                "validation_metrics": validation_results,
                "graduation_status": graduation_status["status"],
                "graduation_score": graduation_status["score"],
                "graduated_patterns": graduation_status["graduated_patterns"],
                "rejected_patterns": graduation_status["rejected_patterns"],
                "production_ready": graduation_status["status"] == "GRADUATED",
            }

            # REMOVED: validation_history.append() to ensure session independence
            # Each session validation is completely isolated

            logger.info(
                f"Graduation complete: {graduation_status['status']} "
                f"(score: {graduation_status['score']:.3f})"
            )

            return results

        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            return {
                "session_name": discovered_patterns.get("session_name", "unknown"),
                "status": "ERROR",
                "error": str(e),
                "production_ready": False,
            }

    def _calculate_validation_metrics(self, patterns: dict[str, Any]) -> dict[str, float]:
        """Calculate comprehensive validation metrics"""

        significant_patterns = patterns.get("significant_patterns", [])
        session_metrics = patterns.get("session_metrics", {})
        patterns.get("raw_results", {})

        if not significant_patterns:
            return {
                "pattern_confidence": 0.0,
                "significance_score": 0.0,
                "attention_coherence": 0.0,
                "pattern_consistency": 0.0,
                "archaeological_value": 0.0,
            }

        # Pattern confidence - average of highest pattern scores
        pattern_scores = []
        for pattern in significant_patterns:
            max_score = max(pattern.get("pattern_scores", [0.0]))
            pattern_scores.append(max_score)

        pattern_confidence = np.mean(pattern_scores) if pattern_scores else 0.0

        # Significance score - archaeological significance
        significance_scores = [
            pattern.get("archaeological_significance", 0.0) for pattern in significant_patterns
        ]
        significance_score = np.mean(significance_scores) if significance_scores else 0.0

        # Attention coherence - measure of attention consistency
        attention_scores = [
            pattern.get("attention_received", 0.0) for pattern in significant_patterns
        ]
        attention_coherence = 1.0 - np.std(attention_scores) if len(attention_scores) > 1 else 1.0
        attention_coherence = max(0.0, min(1.0, attention_coherence))

        # Pattern consistency - similar patterns should have similar scores
        pattern_consistency = self._calculate_pattern_consistency(significant_patterns)

        # Archaeological value - session-level quality metrics
        archaeological_value = session_metrics.get("average_significance", 0.0)

        return {
            "pattern_confidence": pattern_confidence,
            "significance_score": significance_score,
            "attention_coherence": attention_coherence,
            "pattern_consistency": pattern_consistency,
            "archaeological_value": archaeological_value,
        }

    def _calculate_pattern_consistency(self, patterns: list[dict[str, Any]]) -> float:
        """Calculate consistency metric across similar pattern types"""

        if len(patterns) < 2:
            return 1.0

        # Group patterns by type
        pattern_groups = {}
        for pattern in patterns:
            pattern_types = tuple(pattern.get("pattern_types", []))
            if pattern_types not in pattern_groups:
                pattern_groups[pattern_types] = []
            pattern_groups[pattern_types].append(pattern)

        # Calculate consistency within each group
        consistencies = []
        for group_patterns in pattern_groups.values():
            if len(group_patterns) < 2:
                consistencies.append(1.0)
                continue

            # Calculate score variance within group
            scores = [p.get("archaeological_significance", 0.0) for p in group_patterns]
            consistency = 1.0 - np.std(scores) if len(scores) > 1 else 1.0
            consistencies.append(max(0.0, min(1.0, consistency)))

        return np.mean(consistencies) if consistencies else 1.0

    def _apply_graduation_criteria(self, validation_metrics: dict[str, float]) -> dict[str, Any]:
        """Apply graduation criteria and determine final status"""

        # Weight validation metrics for overall score
        weights = {
            "pattern_confidence": 0.3,
            "significance_score": 0.25,
            "attention_coherence": 0.2,
            "pattern_consistency": 0.15,
            "archaeological_value": 0.1,
        }

        # Calculate weighted graduation score
        graduation_score = sum(
            validation_metrics.get(metric, 0.0) * weight for metric, weight in weights.items()
        )

        # STRICT 87% THRESHOLD ENFORCEMENT - NO COMPROMISES
        # Architecture requires strict 87% threshold with no conditional graduation
        if graduation_score >= self.baseline_accuracy:
            status = "GRADUATED"
            graduated_patterns = validation_metrics  # All patterns graduate
            rejected_patterns = {}
            logger.info(
                f"PATTERNS GRADUATED: Score {graduation_score:.3f} >= {self.baseline_accuracy:.3f}"
            )
        else:
            status = "REJECTED"
            graduated_patterns = {}
            rejected_patterns = validation_metrics
            logger.warning(
                f"PATTERNS REJECTED: Score {graduation_score:.3f} < {self.baseline_accuracy:.3f} (strict threshold)"
            )

        return {
            "status": status,
            "score": graduation_score,
            "graduated_patterns": graduated_patterns,
            "rejected_patterns": rejected_patterns,
        }

    def get_graduation_summary(self) -> dict[str, Any]:
        """Get summary of graduation system configuration

        NOTE: No historical data is maintained to ensure session independence.
        Use external logging/storage if cross-session analytics are needed.
        """

        # REMOVED validation_history dependency for session independence
        return {
            "baseline_threshold": self.baseline_accuracy,
            "session_independence": True,
            "note": "No cross-session state maintained - each validation is isolated",
        }
