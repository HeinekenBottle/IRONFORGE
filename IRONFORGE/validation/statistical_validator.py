"""
IRONFORGE Statistical Validation Framework
==========================================

Comprehensive statistical validation for pattern quality, authenticity, and archaeological significance.
Ensures patterns meet rigorous statistical criteria before graduation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.stats as stats

try:
    from config import get_config
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import get_config

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """
    Statistical validation framework for IRONFORGE pattern quality assessment.

    Validates patterns against multiple statistical criteria:
    - Distribution normality and outlier detection
    - Temporal consistency and stationarity
    - Archaeological significance testing
    - Pattern authenticity metrics
    - 87% baseline compliance validation
    """

    def __init__(self, confidence_level: float = 0.95, baseline_threshold: float = 0.87):
        self.confidence_level = confidence_level
        self.baseline_threshold = baseline_threshold
        self.alpha = 1 - confidence_level

        config = get_config()
        self.validation_output_path = Path(config.get_reports_path()) / "validation"
        self.validation_output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Statistical Validator initialized: {confidence_level*100}% confidence, {baseline_threshold*100}% baseline"
        )

    def validate_pattern_quality(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive statistical validation of discovered patterns

        Args:
            patterns: Pattern discovery results from TGAT

        Returns:
            Detailed statistical validation report
        """
        try:
            session_name = patterns.get("session_name", "unknown")
            significant_patterns = patterns.get("significant_patterns", [])
            session_metrics = patterns.get("session_metrics", {})

            logger.info(
                f"Running statistical validation on {len(significant_patterns)} patterns from {session_name}"
            )

            validation_results = {
                "session_name": session_name,
                "validation_timestamp": datetime.now().isoformat(),
                "confidence_level": self.confidence_level,
                "baseline_threshold": self.baseline_threshold,
                "patterns_validated": len(significant_patterns),
                "statistical_tests": {},
                "quality_metrics": {},
                "validation_summary": {},
                "recommendations": [],
            }

            if not significant_patterns:
                validation_results["validation_summary"] = {
                    "overall_quality": "INSUFFICIENT_DATA",
                    "statistical_significance": False,
                    "baseline_compliance": False,
                    "recommendation": "No patterns available for validation",
                }
                return validation_results

            # Extract pattern metrics for statistical analysis
            pattern_metrics = self._extract_pattern_metrics(significant_patterns)

            # Run comprehensive statistical tests
            validation_results["statistical_tests"] = self._run_statistical_tests(pattern_metrics)

            # Calculate quality metrics
            validation_results["quality_metrics"] = self._calculate_quality_metrics(
                pattern_metrics, session_metrics
            )

            # Validate baseline compliance
            baseline_results = self._validate_baseline_compliance(
                validation_results["quality_metrics"]
            )
            validation_results["baseline_compliance"] = baseline_results

            # Generate validation summary
            validation_results["validation_summary"] = self._generate_validation_summary(
                validation_results
            )

            # Generate recommendations
            validation_results["recommendations"] = self._generate_recommendations(
                validation_results
            )

            # Save validation report
            self._save_validation_report(validation_results)

            logger.info(
                f"Statistical validation complete: {validation_results['validation_summary']['overall_quality']}"
            )
            return validation_results

        except Exception as e:
            logger.error(f"Statistical validation failed: {e}")
            return {
                "session_name": patterns.get("session_name", "unknown"),
                "validation_timestamp": datetime.now().isoformat(),
                "status": "ERROR",
                "error": str(e),
                "validation_summary": {
                    "overall_quality": "VALIDATION_FAILED",
                    "statistical_significance": False,
                    "baseline_compliance": False,
                },
            }

    def _extract_pattern_metrics(self, patterns: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract numerical metrics from patterns for statistical analysis"""

        metrics = {
            "pattern_scores": [],
            "attention_weights": [],
            "archaeological_significance": [],
            "temporal_consistency": [],
            "confidence_values": [],
        }

        for pattern in patterns:
            # Pattern scores
            scores = pattern.get("pattern_scores", [])
            if scores:
                metrics["pattern_scores"].extend(scores)

            # Attention received
            attention = pattern.get("attention_received", 0.0)
            if attention > 0:
                metrics["attention_weights"].append(attention)

            # Archaeological significance
            arch_sig = pattern.get("archaeological_significance", 0.0)
            if arch_sig > 0:
                metrics["archaeological_significance"].append(arch_sig)

            # Temporal consistency (if available)
            temp_consistency = pattern.get("temporal_consistency", 0.0)
            if temp_consistency > 0:
                metrics["temporal_consistency"].append(temp_consistency)

            # Confidence values
            confidence = pattern.get("confidence", 0.0)
            if confidence > 0:
                metrics["confidence_values"].append(confidence)

        # Convert to numpy arrays for statistical analysis
        for key in metrics:
            metrics[key] = np.array(metrics[key]) if metrics[key] else np.array([])

        return metrics

    def _run_statistical_tests(self, metrics: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run comprehensive statistical tests on pattern metrics"""

        tests = {}

        for metric_name, values in metrics.items():
            if len(values) < 3:  # Need minimum data for statistical tests
                tests[metric_name] = {
                    "status": "INSUFFICIENT_DATA",
                    "sample_size": len(values),
                    "note": "Minimum 3 data points required for statistical tests",
                }
                continue

            metric_tests = {}

            # Normality test (Shapiro-Wilk)
            if len(values) >= 3:
                try:
                    stat, p_value = stats.shapiro(values)
                    metric_tests["normality"] = {
                        "test": "shapiro_wilk",
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "is_normal": p_value > self.alpha,
                        "interpretation": (
                            "Normal distribution"
                            if p_value > self.alpha
                            else "Non-normal distribution"
                        ),
                    }
                except Exception as e:
                    metric_tests["normality"] = {"error": str(e)}

            # Outlier detection (Modified Z-score)
            try:
                outliers = self._detect_outliers_modified_zscore(values)
                metric_tests["outliers"] = {
                    "method": "modified_zscore",
                    "outlier_count": int(np.sum(outliers)),
                    "outlier_percentage": float(np.sum(outliers) / len(values) * 100),
                    "outlier_indices": outliers.nonzero()[0].tolist(),
                    "clean_data_percentage": float(
                        (len(values) - np.sum(outliers)) / len(values) * 100
                    ),
                }
            except Exception as e:
                metric_tests["outliers"] = {"error": str(e)}

            # Descriptive statistics
            try:
                metric_tests["descriptive"] = {
                    "count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                    "coefficient_of_variation": (
                        float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0.0
                    ),
                }
            except Exception as e:
                metric_tests["descriptive"] = {"error": str(e)}

            # Statistical significance test (One-sample t-test against baseline)
            if metric_name in [
                "pattern_scores",
                "confidence_values",
                "archaeological_significance",
            ]:
                try:
                    stat, p_value = stats.ttest_1samp(values, self.baseline_threshold)
                    metric_tests["baseline_significance"] = {
                        "test": "one_sample_t_test",
                        "null_hypothesis": f"Mean equals {self.baseline_threshold}",
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "significantly_above_baseline": (stat > 0) and (p_value < self.alpha),
                        "mean_vs_baseline": float(np.mean(values)) - self.baseline_threshold,
                        "interpretation": self._interpret_baseline_test(stat, p_value),
                    }
                except Exception as e:
                    metric_tests["baseline_significance"] = {"error": str(e)}

            tests[metric_name] = metric_tests

        return tests

    def _detect_outliers_modified_zscore(
        self, data: np.ndarray, threshold: float = 3.5
    ) -> np.ndarray:
        """Detect outliers using modified Z-score method"""

        if len(data) < 2:
            return np.zeros(len(data), dtype=bool)

        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad == 0:
            # If MAD is 0, use standard deviation
            mad = np.std(data)
            if mad == 0:
                return np.zeros(len(data), dtype=bool)

        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold

    def _interpret_baseline_test(self, stat: float, p_value: float) -> str:
        """Interpret the results of baseline significance testing"""

        if p_value < self.alpha:
            if stat > 0:
                return f"Significantly above baseline (p={p_value:.4f})"
            else:
                return f"Significantly below baseline (p={p_value:.4f})"
        else:
            return f"No significant difference from baseline (p={p_value:.4f})"

    def _calculate_quality_metrics(
        self, metrics: Dict[str, np.ndarray], session_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""

        quality = {}

        # Overall pattern quality score
        pattern_scores = metrics.get("pattern_scores", np.array([]))
        if len(pattern_scores) > 0:
            quality["mean_pattern_score"] = float(np.mean(pattern_scores))
            quality["pattern_score_consistency"] = (
                float(1.0 - np.std(pattern_scores) / np.mean(pattern_scores))
                if np.mean(pattern_scores) > 0
                else 0.0
            )
            quality["patterns_above_baseline"] = float(
                np.sum(pattern_scores >= self.baseline_threshold) / len(pattern_scores)
            )
        else:
            quality["mean_pattern_score"] = 0.0
            quality["pattern_score_consistency"] = 0.0
            quality["patterns_above_baseline"] = 0.0

        # Archaeological significance quality
        arch_sig = metrics.get("archaeological_significance", np.array([]))
        if len(arch_sig) > 0:
            quality["mean_archaeological_significance"] = float(np.mean(arch_sig))
            quality["archaeological_consistency"] = (
                float(1.0 - np.std(arch_sig) / np.mean(arch_sig)) if np.mean(arch_sig) > 0 else 0.0
            )
        else:
            quality["mean_archaeological_significance"] = 0.0
            quality["archaeological_consistency"] = 0.0

        # Attention distribution quality
        attention = metrics.get("attention_weights", np.array([]))
        if len(attention) > 0:
            quality["mean_attention"] = float(np.mean(attention))
            quality["attention_distribution_quality"] = (
                float(1.0 - stats.entropy(attention / np.sum(attention)) / np.log(len(attention)))
                if len(attention) > 1
                else 1.0
            )
        else:
            quality["mean_attention"] = 0.0
            quality["attention_distribution_quality"] = 0.0

        # Temporal consistency
        temporal = metrics.get("temporal_consistency", np.array([]))
        if len(temporal) > 0:
            quality["mean_temporal_consistency"] = float(np.mean(temporal))
        else:
            quality["mean_temporal_consistency"] = 0.0

        # Overall quality composite score
        quality_components = [
            quality["mean_pattern_score"],
            quality["pattern_score_consistency"],
            quality["patterns_above_baseline"],
            quality["mean_archaeological_significance"],
            quality["archaeological_consistency"],
            quality["attention_distribution_quality"],
            quality["mean_temporal_consistency"],
        ]

        # Weight components (archaeological significance and baseline compliance are most important)
        weights = [0.25, 0.15, 0.30, 0.20, 0.05, 0.03, 0.02]
        quality["composite_quality_score"] = float(np.average(quality_components, weights=weights))

        return quality

    def _validate_baseline_compliance(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance with 87% baseline threshold"""

        compliance = {}

        # Primary compliance check
        mean_score = quality_metrics.get("mean_pattern_score", 0.0)
        compliance["mean_score_compliant"] = mean_score >= self.baseline_threshold
        compliance["mean_score_margin"] = mean_score - self.baseline_threshold

        # Pattern percentage compliance
        patterns_above = quality_metrics.get("patterns_above_baseline", 0.0)
        compliance["pattern_percentage_compliant"] = (
            patterns_above >= 0.80
        )  # At least 80% of patterns above baseline
        compliance["patterns_above_baseline_percentage"] = patterns_above * 100

        # Archaeological significance compliance
        arch_sig = quality_metrics.get("mean_archaeological_significance", 0.0)
        compliance["archaeological_significance_compliant"] = arch_sig >= self.baseline_threshold
        compliance["archaeological_significance_margin"] = arch_sig - self.baseline_threshold

        # Composite compliance
        compliance_checks = [
            compliance["mean_score_compliant"],
            compliance["pattern_percentage_compliant"],
            compliance["archaeological_significance_compliant"],
        ]
        compliance["overall_compliant"] = all(compliance_checks)
        compliance["compliance_score"] = sum(compliance_checks) / len(compliance_checks)

        return compliance

    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""

        quality_metrics = validation_results.get("quality_metrics", {})
        baseline_compliance = validation_results.get("baseline_compliance", {})
        statistical_tests = validation_results.get("statistical_tests", {})

        # Determine overall quality rating
        composite_score = quality_metrics.get("composite_quality_score", 0.0)
        baseline_compliant = baseline_compliance.get("overall_compliant", False)

        if composite_score >= 0.90 and baseline_compliant:
            overall_quality = "EXCELLENT"
        elif composite_score >= 0.87 and baseline_compliant:
            overall_quality = "GOOD"
        elif composite_score >= 0.70:
            overall_quality = "ACCEPTABLE"
        elif composite_score >= 0.50:
            overall_quality = "POOR"
        else:
            overall_quality = "INSUFFICIENT"

        # Check statistical significance
        statistical_significance = False
        for _metric_name, tests in statistical_tests.items():
            baseline_test = tests.get("baseline_significance", {})
            if baseline_test.get("significantly_above_baseline", False):
                statistical_significance = True
                break

        summary = {
            "overall_quality": overall_quality,
            "composite_quality_score": composite_score,
            "statistical_significance": statistical_significance,
            "baseline_compliance": baseline_compliant,
            "compliance_score": baseline_compliance.get("compliance_score", 0.0),
            "patterns_validated": validation_results.get("patterns_validated", 0),
            "confidence_level": self.confidence_level,
            "recommendation": self._get_quality_recommendation(
                overall_quality, baseline_compliant, statistical_significance
            ),
        }

        return summary

    def _get_quality_recommendation(
        self, quality: str, baseline_compliant: bool, statistically_significant: bool
    ) -> str:
        """Get recommendation based on validation results"""

        if quality == "EXCELLENT" and baseline_compliant and statistically_significant:
            return "GRADUATE - Patterns exceed all quality thresholds"
        elif quality == "GOOD" and baseline_compliant:
            return "GRADUATE - Patterns meet baseline requirements"
        elif quality == "ACCEPTABLE" and baseline_compliant:
            return "CONDITIONAL_GRADUATE - Patterns meet minimum requirements"
        elif not baseline_compliant:
            return "REJECT - Patterns do not meet 87% baseline threshold"
        elif quality in ["POOR", "INSUFFICIENT"]:
            return "REJECT - Pattern quality insufficient for graduation"
        else:
            return "REVIEW - Manual review recommended"

    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for improvement"""

        recommendations = []
        quality_metrics = validation_results.get("quality_metrics", {})
        baseline_compliance = validation_results.get("baseline_compliance", {})
        statistical_tests = validation_results.get("statistical_tests", {})

        # Pattern quality recommendations
        if quality_metrics.get("mean_pattern_score", 0.0) < self.baseline_threshold:
            recommendations.append(
                f"Improve pattern scores - current mean {quality_metrics.get('mean_pattern_score', 0.0):.3f} below {self.baseline_threshold} threshold"
            )

        if quality_metrics.get("pattern_score_consistency", 0.0) < 0.7:
            recommendations.append(
                "Improve pattern consistency - high variance in pattern scores detected"
            )

        # Archaeological significance recommendations
        if quality_metrics.get("mean_archaeological_significance", 0.0) < self.baseline_threshold:
            recommendations.append(
                "Enhance archaeological significance - patterns lack sufficient archaeological value"
            )

        # Outlier recommendations
        for metric_name, tests in statistical_tests.items():
            outliers = tests.get("outliers", {})
            if outliers.get("outlier_percentage", 0.0) > 20:
                recommendations.append(
                    f"Address outliers in {metric_name} - {outliers.get('outlier_percentage', 0.0):.1f}% outliers detected"
                )

        # Baseline compliance recommendations
        if not baseline_compliance.get("overall_compliant", False):
            recommendations.append(
                "Critical: Address baseline compliance failures to meet 87% graduation threshold"
            )

        if not recommendations:
            recommendations.append(
                "No specific improvements recommended - patterns meet all quality criteria"
            )

        return recommendations

    def _save_validation_report(self, validation_results: Dict[str, Any]):
        """Save validation report to file"""

        try:
            session_name = validation_results.get("session_name", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistical_validation_{session_name}_{timestamp}.json"
            filepath = self.validation_output_path / filename

            with open(filepath, "w") as f:
                json.dump(validation_results, f, indent=2, default=str)

            logger.info(f"Statistical validation report saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation system configuration"""

        return {
            "confidence_level": self.confidence_level,
            "baseline_threshold": self.baseline_threshold,
            "alpha": self.alpha,
            "output_path": str(self.validation_output_path),
            "statistical_tests_available": [
                "normality_test",
                "outlier_detection",
                "baseline_significance",
                "descriptive_statistics",
            ],
            "quality_metrics_computed": [
                "composite_quality_score",
                "pattern_score_consistency",
                "archaeological_significance",
                "attention_distribution_quality",
                "temporal_consistency",
            ],
        }
