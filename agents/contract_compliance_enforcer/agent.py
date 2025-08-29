"""
Contract Compliance Enforcer Agent for IRONFORGE
================================================

Purpose: Validate data integrity and enforce IRONFORGE golden invariants.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

try:
    from ironforge.api import load_config
except Exception:  # Fallback if ironforge not available in runtime context
    def load_config(path: str) -> Dict[str, Any]:  # type: ignore
        return {"path": path}

logger = logging.getLogger(__name__)

# Golden invariants and performance requirements
GOLDEN_INVARIANTS: Dict[str, Any] = {
    "events": [
        "Expansion",
        "Consolidation",
        "Retracement",
        "Reversal",
        "Liquidity Taken",
        "Redelivery",
    ],
    "edge_intents": [
        "TEMPORAL_NEXT",
        "MOVEMENT_TRANSITION",
        "LIQ_LINK",
        "CONTEXT",
    ],
    "node_features": 51,  # f0-f50
    "edge_features": 20,   # e0-e19
}

PERFORMANCE_REQUIREMENTS: Dict[str, Any] = {
    "session_processing": 3.0,
    "full_discovery": 180.0,
    "memory_usage": 100,
    "authenticity_threshold": 87.0,
    "initialization": 2.0,
}


@dataclass
class IronforgeConfig:
    """Standard configuration wrapper for agents."""
    config_path: str = "configs/dev.yml"

    def __post_init__(self) -> None:
        # Be resilient if the local dev config is missing in test environments
        try:
            self.config: Dict[str, Any] = load_config(self.config_path)
        except FileNotFoundError:
            self.config = {"path": self.config_path}
        self.performance: Dict[str, Any] = PERFORMANCE_REQUIREMENTS
        self.golden_invariants: Dict[str, Any] = GOLDEN_INVARIANTS


class ContractComplianceEnforcer:
    """
    Validate data integrity and enforce golden invariants across IRONFORGE pipeline.
    """

    def __init__(self, config: Optional[IronforgeConfig] = None) -> None:
        self.config = config or IronforgeConfig()
        # Lazy imports to avoid hard dependency at import time
        from .validators import GoldenInvariantValidator, SessionBoundaryGuard
        from .performance import PerformanceContractChecker

        self.validator = GoldenInvariantValidator(GOLDEN_INVARIANTS)
        self.session_guard = SessionBoundaryGuard()
        self.performance_checker = PerformanceContractChecker(PERFORMANCE_REQUIREMENTS)
        logger.debug("ContractComplianceEnforcer initialized")

    # --- Public API ---
    def validate_golden_invariants(self, session_data: Dict[str, Any]) -> Dict[str, bool]:
        events: List[str] = session_data.get("events", [])
        edge_intents: List[str] = session_data.get("edge_intents", [])
        node_feature_dim: int = session_data.get("node_feature_dim", 0)
        edge_feature_dim: int = session_data.get("edge_feature_dim", 0)

        results = {
            "event_types": self.validator.validate_event_types(events),
            "edge_intents": self.validator.validate_edge_intents(edge_intents),
            "feature_dimensions": self.validator.validate_feature_dimensions(
                node_feature_dim, edge_feature_dim
            ),
        }
        return results

    def enforce_session_boundaries(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        cross_edges = self.session_guard.detect_cross_session_edges(graph_data)
        isolation_ok = self.session_guard.validate_session_isolation(graph_data)
        return {"cross_session_edges": cross_edges, "isolation_ok": isolation_ok}

    def validate_htf_compliance(self, features: Dict[str, Any]) -> bool:
        return self.session_guard.ensure_htf_rule_compliance(features)

    def check_performance_contracts(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        return self.performance_checker.check(metrics)


# Factory

def create_contract_compliance_enforcer(config_path: str = "configs/dev.yml") -> ContractComplianceEnforcer:
    return ContractComplianceEnforcer(IronforgeConfig(config_path))
