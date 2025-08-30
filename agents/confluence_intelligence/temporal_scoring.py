from __future__ import annotations
from typing import Any, Dict, List


class TemporalScoringEngine:
    def integrate_archaeological_awareness(self, scoring: Dict[str, Any]) -> Dict[str, Any]:
        return scoring

    def apply_temporal_intelligence(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in patterns:
            forward_coherence = float(p.get("temporal_coherence", 0.0))
            out.append({**p, "temporal": forward_coherence})
        return out
