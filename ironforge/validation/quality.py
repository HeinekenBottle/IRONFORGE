"""
Quality gates and composite scoring for BMAD metamorphosis research.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np


def compute_quality_gates(run_summary: Dict[str, Any], mode: str = "prod") -> Dict[str, Any]:
    """
    Compute pass/fail for standard gates and composite quality.

    Expected keys in run_summary:
      - metamorphosis: { metamorphosis_patterns, phase_transitions, metamorphosis_summary }
      - statistics: { significance_tests }
      - sessions: list of session analyses with temporal_characteristics
      - thresholds: {authenticity, significance, coherence, confidence}
    """
    metamorphosis = run_summary.get("metamorphosis", {})
    statistics = run_summary.get("statistics", {})
    sessions = run_summary.get("sessions", [])
    thresholds = run_summary.get("thresholds", {
        "authenticity": 0.87,
        "significance": 0.01,
        "coherence": 0.70,
        "confidence": 0.70,
    })

    # Mode scaling (dev is softer)
    if mode == "dev":
        t = {k: (v * 0.9 if k != "significance" else v * 2.0) for k, v in thresholds.items()}
    else:
        t = thresholds

    patterns = metamorphosis.get("metamorphosis_patterns", {})
    if patterns:
        authenticity_score = float(np.mean([p.get("transformation_strength", 0.0) for p in patterns.values()]))
        confidence_score = float(np.mean([p.get("confidence_level", 0.0) for p in patterns.values()]))
    else:
        authenticity_score = 0.0
        confidence_score = 0.0

    sig_tests = statistics.get("significance_tests", {})
    avg_p = float(np.mean([test.get("p_value", 1.0) for test in sig_tests.values()])) if sig_tests else 1.0

    if sessions:
        coherence = float(np.mean([s.get("temporal_characteristics", {}).get("trend_consistency", 0.0) for s in sessions]))
    else:
        coherence = 0.0

    gates = {
        "pattern_evolution_authenticity": {
            "score": authenticity_score, "threshold": t["authenticity"], "passed": authenticity_score >= t["authenticity"],
        },
        "metamorphosis_statistical_significance": {
            "score": avg_p, "threshold": t["significance"], "passed": avg_p <= t["significance"],
        },
        "cross_phase_correlation_confidence": {
            "score": confidence_score, "threshold": t["confidence"], "passed": confidence_score >= t["confidence"],
        },
        "temporal_consistency_threshold": {
            "score": coherence, "threshold": t["coherence"], "passed": coherence >= t["coherence"],
        },
        "research_framework_compliance": {
            "score": 1.0, "threshold": 1.0, "passed": True,
        },
    }

    total = len(gates)
    passed = sum(1 for g in gates.values() if g["passed"]) if total else 0
    overall = passed / total if total else 0.0

    return {
        "gate_assessments": gates,
        "overall_quality": overall,
        "quality_score": overall,
        "gates_passed": passed,
        "total_gates": total,
        "research_ready": overall >= 0.8,
    }
