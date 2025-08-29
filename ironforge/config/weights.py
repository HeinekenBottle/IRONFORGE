"""Unified weights configuration for IRONFORGE.

This module centralizes weight definitions used across confluence scoring,
temporal coherence, and optional DAG topology features. It does not replace the
public SDK `WeightsCfg` yet, but provides an internal unified representation
for progressive adoption.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class UnifiedWeights:
    # Confluence-style weights
    cluster_z: float = 0.30
    htf_prox: float = 0.25

    # Temporal coherence / pattern weights
    temporal_coherence: float = 0.25
    pattern_strength: float = 0.30

    # Experimental DAG topology weight (feature-flagged)
    dag_topology_weight: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    def validate_bounds(self) -> None:
        for name, value in self.to_dict().items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"UnifiedWeights.{name} must be within [0, 1]")

