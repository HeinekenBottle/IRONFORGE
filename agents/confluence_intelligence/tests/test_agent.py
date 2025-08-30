from __future__ import annotations

from agents.confluence_intelligence.agent import ConfluenceIntelligence


def test_weight_optimization_and_scoring() -> None:
    agent = ConfluenceIntelligence()
    weights = agent.optimize_scoring_weights([
        {"success": True}, {"success": False}, {"success": True}
    ])
    assert 0.4 <= weights["authenticity"] <= 0.6

    patterns = [
        {"confidence": 0.9, "temporal_coherence": 0.7, "other": 0.1},
    ]
    scored = agent.enhance_pattern_evaluation(patterns)
    assert "confluence_score" in scored[0]
