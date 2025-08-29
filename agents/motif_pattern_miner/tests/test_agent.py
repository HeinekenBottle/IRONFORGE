from __future__ import annotations

from agents.motif_pattern_miner.agent import MotifPatternMiner


def test_motif_miner_flow() -> None:
    m = MotifPatternMiner()
    motifs = m.discover_recurring_motifs([{"type": "A"}, {"type": "A"}, {"type": "B"}])
    assert "motifs" in motifs and "stability" in motifs

    cross = m.mine_cross_timeframe_patterns({})
    assert cross["anchored"] is True
