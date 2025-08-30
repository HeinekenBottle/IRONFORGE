from __future__ import annotations

from agents.htf_cascade_predictor.agent import HTFCascadePredictor


def test_cascade_predictor_flow() -> None:
    p = HTFCascadePredictor()
    analysis = p.analyze_htf_cascades({"f45": 1.0, "f46": 2.0})
    assert analysis["valid"] is True

    timing = p.predict_cascade_timing([])
    assert "prediction_seconds" in timing

    echoes = p.detect_temporal_echoes({"echoes": [1, 2]})
    assert echoes["echo_count"] == 2
