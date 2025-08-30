from __future__ import annotations

from agents.pattern_intelligence_analyst.agent import PatternIntelligenceAnalyst


def test_pattern_intelligence_flow() -> None:
    a = PatternIntelligenceAnalyst()
    out = a.analyze_discovered_patterns([{"confidence": 0.9}])
    assert out["analysis"]["count"] == 1

    synth = a.synthesize_multi_agent_insights({"agent": True})
    assert synth["synthesized"] is True
