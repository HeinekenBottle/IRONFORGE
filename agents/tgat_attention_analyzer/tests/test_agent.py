from __future__ import annotations

from agents.tgat_attention_analyzer.agent import TGATAttentionAnalyzer


def test_analyze_attention_weights_handles_missing() -> None:
    analyzer = TGATAttentionAnalyzer()
    out = analyzer.analyze_attention_weights({})
    assert "weights" in out and out["scores"] == []


def test_validate_pattern_authenticity() -> None:
    analyzer = TGATAttentionAnalyzer()
    patterns = [{"confidence": 0.91}, {"confidence": 0.86}]
    results = analyzer.validate_pattern_authenticity(patterns)
    assert results[0]["authenticity"]["passed"] is True
    assert results[1]["authenticity"]["passed"] is False
