from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from .agent import GOLDEN_INVARIANTS, PERFORMANCE_REQUIREMENTS


@dataclass
class IronforgeConfig:
    config_path: str = "configs/dev.yml"

    def __post_init__(self) -> None:
        self.performance: Dict[str, Any] = PERFORMANCE_REQUIREMENTS
        self.golden_invariants: Dict[str, Any] = GOLDEN_INVARIANTS
