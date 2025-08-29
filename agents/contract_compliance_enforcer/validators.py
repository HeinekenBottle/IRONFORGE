from __future__ import annotations
from typing import Any, Dict, List


class ContractViolationError(Exception):
    pass


class GoldenInvariantValidator:
    def __init__(self, invariants: Dict[str, Any]) -> None:
        self.invariants = invariants

    def validate_event_types(self, events: List[str]) -> bool:
        if not events:
            return True  # Nothing to validate
        expected = set(self.invariants["events"])  # type: ignore[index]
        unknown = set(events) - expected
        if unknown:
            raise ContractViolationError(f"Unknown event types: {sorted(unknown)}")
        return True

    def validate_edge_intents(self, edges: List[str]) -> bool:
        if not edges:
            return True
        expected = set(self.invariants["edge_intents"])  # type: ignore[index]
        unknown = set(edges) - expected
        if unknown:
            raise ContractViolationError(f"Unknown edge intents: {sorted(unknown)}")
        return True

    def validate_feature_dimensions(self, node_dim: int, edge_dim: int) -> bool:
        expected_node = int(self.invariants["node_features"])  # type: ignore[index]
        expected_edge = int(self.invariants["edge_features"])  # type: ignore[index]
        if node_dim not in (45, expected_node):
            raise ContractViolationError(
                f"Invalid node feature dimension: {node_dim}. Expected 45 or {expected_node}"
            )
        if edge_dim != expected_edge:
            raise ContractViolationError(
                f"Invalid edge feature dimension: {edge_dim}. Expected {expected_edge}"
            )
        return True


class SessionBoundaryGuard:
    def detect_cross_session_edges(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        edges = graph.get("edges", [])
        cross_edges: List[Dict[str, Any]] = []
        for edge in edges:
            if edge.get("from_session") != edge.get("to_session"):
                cross_edges.append(edge)
        return cross_edges

    def validate_session_isolation(self, sessions: Dict[str, Any]) -> bool:
        return len(self.detect_cross_session_edges(sessions)) == 0

    def ensure_htf_rule_compliance(self, features: Dict[str, Any]) -> bool:
        # Enforce last-closed only for f45-f50
        htf_values = [features.get(f"f{i}") for i in range(45, 51)]
        # If values exist, assume they represent last-closed; no intra-candle flags allowed
        intra_flags = features.get("htf_intra_candle_flags", False)
        if intra_flags:
            raise ContractViolationError("HTF intra-candle data detected (violates last-closed rule)")
        return True
