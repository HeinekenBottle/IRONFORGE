from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnifiedWeights:
    """
    Unified weights for confluence, scoring, and optional DAG topology.

    This consolidates previously fragmented weight definitions and provides
    sensible defaults. All values are expected to be in [0, 1].
    """

    # Confluence weights
    temporal_coherence: float = 0.25
    pattern_strength: float = 0.30

    # Scoring weights
    cluster_z: float = 0.30
    htf_prox: float = 0.25
    structure: float = 0.20
    cycle: float = 0.15
    precursor: float = 0.10

    # Optional DAG topology weight (feature-flagged in downstream)
    dag_topology_weight: float = 0.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            v = float(value)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"UnifiedWeights.{name} must be in [0,1], got {value}")

