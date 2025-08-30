from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GoldenInvariantSet:
    events: List[str]
    edge_intents: List[str]
    node_features: int
    edge_features: int
