from __future__ import annotations

import pytest

from agents.contract_compliance_enforcer.agent import (
    ContractComplianceEnforcer,
    create_contract_compliance_enforcer,
)


def test_factory_creates_agent() -> None:
    agent = create_contract_compliance_enforcer()
    assert isinstance(agent, ContractComplianceEnforcer)


def test_validate_golden_invariants_pass() -> None:
    agent = create_contract_compliance_enforcer()
    session = {
        "events": [
            "Expansion",
            "Consolidation",
            "Retracement",
            "Reversal",
            "Liquidity Taken",
            "Redelivery",
        ],
        "edge_intents": ["TEMPORAL_NEXT", "MOVEMENT_TRANSITION", "LIQ_LINK", "CONTEXT"],
        "node_feature_dim": 51,
        "edge_feature_dim": 20,
    }
    results = agent.validate_golden_invariants(session)
    assert all(results.values())


def test_edge_intents_fail_raises() -> None:
    agent = create_contract_compliance_enforcer()
    session = {
        "events": ["Expansion"],
        "edge_intents": ["INVALID"],
        "node_feature_dim": 45,
        "edge_feature_dim": 20,
    }
    with pytest.raises(Exception):
        agent.validate_golden_invariants(session)


def test_session_boundary_enforcement() -> None:
    agent = create_contract_compliance_enforcer()
    graph = {
        "edges": [
            {"from_session": "A", "to_session": "A"},
            {"from_session": "A", "to_session": "B"},
        ]
    }
    result = agent.enforce_session_boundaries(graph)
    assert result["isolation_ok"] is False
    assert len(result["cross_session_edges"]) == 1


def test_htf_rule_compliance() -> None:
    agent = create_contract_compliance_enforcer()
    features = {"f45": 1.0, "f50": 2.0}
    assert agent.validate_htf_compliance(features) is True


def test_performance_contracts() -> None:
    agent = create_contract_compliance_enforcer()
    results = agent.check_performance_contracts(
        {"session_processing": 2.9, "full_discovery": 170.0, "memory_usage": 90}
    )
    assert results == {"session_processing": True, "full_discovery": True, "memory_usage": True}
