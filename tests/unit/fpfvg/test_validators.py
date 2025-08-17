"""Test FPFVG validators functionality."""

import pytest

from ironforge.analysis.fpfvg.validators import (
    validate_candidates,
    validate_network_graph,
    is_in_pm_belt,
    safe_float,
)


def test_validate_candidates_empty():
    """Test validation with empty candidate list."""
    result = validate_candidates([])
    
    assert result["valid"] is False
    assert result["candidate_count"] == 0
    assert "No candidates provided" in result["errors"]


def test_validate_candidates_valid():
    """Test validation with valid candidates."""
    candidates = [
        {
            "id": "test_1",
            "session_id": "session_1",
            "event_type": "formation",
            "price_level": 23000.0,
            "range_pos": 0.5,
            "start_ts": "2025-08-01T14:30:00",
            "in_pm_belt": False,
            "zone_proximity": {"in_zone": False},
            "timeframe": "15m",
        },
        {
            "id": "test_2",
            "session_id": "session_1",
            "event_type": "redelivery",
            "price_level": 23010.0,
            "range_pos": 0.6,
            "start_ts": "2025-08-01T14:35:00",
            "in_pm_belt": True,
            "zone_proximity": {"in_zone": True},
            "timeframe": "15m",
        },
    ]
    
    result = validate_candidates(candidates)
    
    assert result["valid"] is True
    assert result["candidate_count"] == 2
    assert len(result["errors"]) == 0


def test_validate_candidates_missing_fields():
    """Test validation with missing required fields."""
    candidates = [
        {
            "id": "test_1",
            # Missing session_id, event_type, etc.
            "price_level": 23000.0,
        }
    ]
    
    result = validate_candidates(candidates)
    
    assert result["valid"] is False
    assert len(result["errors"]) > 0
    assert any("Missing 'session_id'" in error for error in result["errors"])
    assert any("Missing 'event_type'" in error for error in result["errors"])


def test_validate_candidates_invalid_types():
    """Test validation with invalid data types."""
    candidates = [
        {
            "id": "test_1",
            "session_id": "session_1",
            "event_type": "formation",
            "price_level": "not_a_number",  # Invalid type
            "range_pos": "also_not_a_number",  # Invalid type
            "start_ts": "",  # Invalid timestamp
            "in_pm_belt": "not_a_boolean",  # Invalid type
            "zone_proximity": {"in_zone": False},
            "timeframe": "15m",
        }
    ]
    
    result = validate_candidates(candidates)
    
    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_validate_candidates_invalid_ranges():
    """Test validation with values outside valid ranges."""
    candidates = [
        {
            "id": "test_1",
            "session_id": "session_1",
            "event_type": "invalid_type",  # Invalid event type
            "price_level": -100.0,  # Negative price
            "range_pos": 1.5,  # Range position > 1.0
            "start_ts": "2025-08-01T14:30:00",
            "in_pm_belt": False,
            "zone_proximity": {"in_zone": False},
            "timeframe": "15m",
        }
    ]
    
    result = validate_candidates(candidates)
    
    assert result["valid"] is False
    assert len(result["errors"]) > 0
    assert any("Invalid event type" in error for error in result["errors"])


def test_validate_candidates_duplicate_ids():
    """Test validation with duplicate candidate IDs."""
    candidates = [
        {
            "id": "test_1",
            "session_id": "session_1",
            "event_type": "formation",
            "price_level": 23000.0,
            "range_pos": 0.5,
            "start_ts": "2025-08-01T14:30:00",
            "in_pm_belt": False,
            "zone_proximity": {"in_zone": False},
            "timeframe": "15m",
        },
        {
            "id": "test_1",  # Duplicate ID
            "session_id": "session_1",
            "event_type": "redelivery",
            "price_level": 23010.0,
            "range_pos": 0.6,
            "start_ts": "2025-08-01T14:35:00",
            "in_pm_belt": True,
            "zone_proximity": {"in_zone": True},
            "timeframe": "15m",
        },
    ]
    
    result = validate_candidates(candidates)
    
    assert result["valid"] is False
    assert any("Duplicate candidate ID" in error for error in result["errors"])


def test_validate_network_graph_valid():
    """Test validation with valid network graph."""
    network_graph = {
        "nodes": [
            {"id": "node_1", "session_id": "session_1", "event_type": "formation", 
             "price_level": 23000.0, "range_pos": 0.5, "timestamp": "2025-08-01T14:30:00"},
            {"id": "node_2", "session_id": "session_1", "event_type": "redelivery",
             "price_level": 23010.0, "range_pos": 0.6, "timestamp": "2025-08-01T14:35:00"},
        ],
        "edges": [
            {"source": "node_1", "target": "node_2"},
        ],
        "metadata": {"node_count": 2, "edge_count": 1},
    }
    
    result = validate_network_graph(network_graph)
    
    assert result["valid"] is True
    assert len(result["errors"]) == 0


def test_validate_network_graph_missing_keys():
    """Test validation with missing required keys."""
    network_graph = {
        "nodes": [],
        # Missing edges and metadata
    }
    
    result = validate_network_graph(network_graph)
    
    assert result["valid"] is False
    assert any("Missing required key: edges" in error for error in result["errors"])
    assert any("Missing required key: metadata" in error for error in result["errors"])


def test_validate_network_graph_invalid_edge_references():
    """Test validation with invalid edge references."""
    network_graph = {
        "nodes": [
            {"id": "node_1", "session_id": "session_1", "event_type": "formation", 
             "price_level": 23000.0, "range_pos": 0.5, "timestamp": "2025-08-01T14:30:00"},
        ],
        "edges": [
            {"source": "node_1", "target": "nonexistent_node"},  # Invalid reference
        ],
        "metadata": {"node_count": 1, "edge_count": 1},
    }
    
    result = validate_network_graph(network_graph)
    
    assert result["valid"] is False
    assert any("references invalid target node" in error for error in result["errors"])


def test_is_in_pm_belt():
    """Test PM belt detection."""
    # Test timestamps in PM belt
    assert is_in_pm_belt("2025-08-01T14:35:00") is True
    assert is_in_pm_belt("2025-08-01T14:36:30") is True
    assert is_in_pm_belt("2025-08-01T14:37:15") is True
    assert is_in_pm_belt("2025-08-01T14:38:00") is True
    
    # Test timestamps outside PM belt
    assert is_in_pm_belt("2025-08-01T14:30:00") is False
    assert is_in_pm_belt("2025-08-01T14:40:00") is False
    assert is_in_pm_belt("2025-08-01T15:00:00") is False
    
    # Test invalid timestamps
    assert is_in_pm_belt("invalid_timestamp") is False
    assert is_in_pm_belt("") is False


def test_safe_float():
    """Test safe float conversion."""
    # Valid conversions
    assert safe_float(42) == 42.0
    assert safe_float(3.14) == 3.14
    assert safe_float("123.45") == 123.45
    assert safe_float("0") == 0.0
    
    # Invalid conversions with default
    assert safe_float("not_a_number") == 0.0
    assert safe_float(None) == 0.0
    assert safe_float("") == 0.0
    
    # Invalid conversions with custom default
    assert safe_float("not_a_number", default=99.0) == 99.0
    assert safe_float(None, default=-1.0) == -1.0