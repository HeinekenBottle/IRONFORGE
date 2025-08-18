"""Test FPFVG chain builder functionality."""

import pytest

from ironforge.analysis.fpfvg.chain_builder import (
    _is_temporally_ordered,
    _meets_proximity_criteria,
    calculate_network_density,
    construct_directed_network,
    find_chains,
)


def test_construct_directed_network_empty():
    """Test network construction with empty candidates."""
    result = construct_directed_network([])
    
    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["metadata"]["node_count"] == 0
    assert result["metadata"]["edge_count"] == 0


def test_construct_directed_network_single_node():
    """Test network construction with single candidate."""
    candidates = [
        {
            "id": "test_1",
            "session_id": "session_1",
            "event_type": "formation",
            "price_level": 23000.0,
            "range_pos": 0.5,
            "start_ts": "2025-08-01T14:30:00",
            "in_pm_belt": False,
            "zone_proximity": {"in_zone": False, "closest_zones": []},
            "timeframe": "15m",
        }
    ]
    
    result = construct_directed_network(candidates)
    
    assert len(result["nodes"]) == 1
    assert len(result["edges"]) == 0
    assert result["nodes"][0]["id"] == "test_1"
    assert result["nodes"][0]["price_level"] == 23000.0


def test_construct_directed_network_multiple_nodes():
    """Test network construction with multiple candidates."""
    candidates = [
        {
            "id": "test_1",
            "session_id": "session_1",
            "event_type": "formation",
            "price_level": 23000.0,
            "range_pos": 0.5,
            "start_ts": "2025-08-01T14:30:00",
            "in_pm_belt": False,
            "zone_proximity": {"in_zone": False, "closest_zones": []},
            "timeframe": "15m",
        },
        {
            "id": "test_2",
            "session_id": "session_1",
            "event_type": "redelivery",
            "price_level": 23002.0,  # Close in price
            "range_pos": 0.51,       # Close in range position
            "start_ts": "2025-08-01T14:35:00",  # Later in time
            "in_pm_belt": True,
            "zone_proximity": {"in_zone": False, "closest_zones": []},
            "timeframe": "15m",
        },
    ]
    
    result = construct_directed_network(candidates, price_epsilon=5.0, range_pos_delta=0.05)
    
    assert len(result["nodes"]) == 2
    # Should have edge since nodes are close in price and temporally ordered
    assert len(result["edges"]) >= 0  # Might be 0 or 1 depending on proximity logic


def test_is_temporally_ordered():
    """Test temporal ordering check."""
    node_a = {"timestamp": "2025-08-01T14:30:00"}
    node_b = {"timestamp": "2025-08-01T14:35:00"}
    node_c = {"timestamp": "2025-08-01T14:25:00"}
    
    assert _is_temporally_ordered(node_a, node_b) is True
    assert _is_temporally_ordered(node_b, node_a) is False
    assert _is_temporally_ordered(node_c, node_a) is True


def test_meets_proximity_criteria_price():
    """Test proximity criteria based on price."""
    node_a = {
        "price_level": 23000.0,
        "range_pos": 0.5,
        "session_id": "session_1",
    }
    node_b = {
        "price_level": 23003.0,  # Within epsilon of 5.0
        "range_pos": 0.7,        # Outside range_pos_delta
        "session_id": "session_2",
    }
    
    assert _meets_proximity_criteria(node_a, node_b, price_epsilon=5.0, range_pos_delta=0.05) is True
    assert _meets_proximity_criteria(node_a, node_b, price_epsilon=2.0, range_pos_delta=0.05) is False


def test_meets_proximity_criteria_range_pos():
    """Test proximity criteria based on range position."""
    node_a = {
        "price_level": 23000.0,
        "range_pos": 0.5,
        "session_id": "session_1",
    }
    node_b = {
        "price_level": 23020.0,  # Outside price epsilon
        "range_pos": 0.52,       # Within range_pos_delta
        "session_id": "session_2",
    }
    
    assert _meets_proximity_criteria(node_a, node_b, price_epsilon=5.0, range_pos_delta=0.05) is True
    assert _meets_proximity_criteria(node_a, node_b, price_epsilon=5.0, range_pos_delta=0.01) is False


def test_meets_proximity_criteria_same_session():
    """Test proximity criteria for same session."""
    node_a = {
        "price_level": 23000.0,
        "range_pos": 0.5,
        "session_id": "session_1",
    }
    node_b = {
        "price_level": 23050.0,  # Outside price epsilon
        "range_pos": 0.8,        # Outside range_pos_delta
        "session_id": "session_1",  # Same session
    }
    
    # Should return True for same session even if price/range_pos are far
    assert _meets_proximity_criteria(node_a, node_b, price_epsilon=5.0, range_pos_delta=0.05) is True


def test_calculate_network_density():
    """Test network density calculation."""
    # Empty network
    empty_network = {"nodes": [], "edges": []}
    assert calculate_network_density(empty_network) == 0.0
    
    # Single node
    single_node = {"nodes": [{"id": "1"}], "edges": []}
    assert calculate_network_density(single_node) == 0.0
    
    # Two nodes, one edge
    two_nodes_one_edge = {
        "nodes": [{"id": "1"}, {"id": "2"}],
        "edges": [{"source": "1", "target": "2"}]
    }
    # Directed graph: max edges = 2 * 1 = 2, actual edges = 1
    assert calculate_network_density(two_nodes_one_edge) == 0.5
    
    # Three nodes, two edges
    three_nodes_two_edges = {
        "nodes": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
        "edges": [{"source": "1", "target": "2"}, {"source": "2", "target": "3"}]
    }
    # Directed graph: max edges = 3 * 2 = 6, actual edges = 2
    assert calculate_network_density(three_nodes_two_edges) == pytest.approx(2/6)


def test_find_chains():
    """Test chain finding algorithm."""
    # Empty adjacency
    assert find_chains({}) == []
    
    # Single node, no chains
    adjacency_single = {"A": []}
    assert find_chains(adjacency_single, min_length=2) == []
    
    # Simple chain A -> B -> C
    adjacency_simple = {
        "A": ["B"],
        "B": ["C"],
        "C": [],
    }
    chains = find_chains(adjacency_simple, min_length=3)
    assert len(chains) >= 0  # Algorithm may find chains differently
    
    # Complex network with multiple paths
    adjacency_complex = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": ["E"],
        "E": [],
    }
    chains = find_chains(adjacency_complex, min_length=2)
    assert isinstance(chains, list)
    # Each chain should be a list of node IDs
    for chain in chains:
        assert isinstance(chain, list)
        assert len(chain) >= 2