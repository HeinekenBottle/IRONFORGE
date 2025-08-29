from __future__ import annotations

from agents.authenticity_validator.agent import AuthenticityValidator


def test_validation_and_graduation() -> None:
    v = AuthenticityValidator()
    patterns = [
        {"confidence": 0.9, "temporal_coherence": 0.8, "theory_b_compliant": True, "precision": 7.7},
        {"confidence": 0.6, "temporal_coherence": 0.4, "theory_b_compliant": False, "precision": 5.0},
    ]
    validated = v.validate_authenticity_threshold(patterns)
    assert validated[0]["authenticity"]["passed"] is True
    assert validated[1]["authenticity"]["passed"] is False

    graduated = v.graduate_patterns(validated)
    assert len(graduated) == 1

    report = v.generate_validation_report(validated)
    assert report["total"] == 2 and report["passed"] == 1
