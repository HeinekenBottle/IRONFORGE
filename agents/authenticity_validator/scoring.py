from __future__ import annotations
from typing import Any, Dict


def calculate_authenticity(pattern: Dict[str, Any]) -> float:
    # Confidence can be given on 0-1 or 0-100 scale. Normalize to 0-1.
    raw_confidence = float(pattern.get("confidence", 0.0))
    confidence = raw_confidence if raw_confidence <= 1.0 else raw_confidence / 100.0
    temporal_coherence = float(pattern.get("temporal_coherence", 0.0))
    theory_b = 1.0 if pattern.get("theory_b_compliant", False) else 0.0
    precision = float(pattern.get("precision", 0.0))

    score = (
        confidence * 40.0
        + temporal_coherence * 25.0
        + theory_b * 15.0
        + min(10.0, (precision / 7.55) * 10.0)
    )
    # Graduation boost when high-quality criteria are met
    if theory_b >= 1.0 and precision >= 7.0 and confidence >= 0.85 and temporal_coherence >= 0.7:
        score += 7.0
    return max(0.0, min(100.0, score))
