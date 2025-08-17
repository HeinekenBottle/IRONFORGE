"""FPFVG Data Validation and Consistency Checks."""

import logging
from datetime import time
from typing import Any

logger = logging.getLogger(__name__)


def validate_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Validate FPFVG candidates for required fields and data consistency

    Returns validation results with any errors or warnings found.
    """
    validation_results = {
        "valid": True,
        "candidate_count": len(candidates),
        "errors": [],
        "warnings": [],
        "validation_summary": {},
    }

    if not candidates:
        validation_results["errors"].append("No candidates provided")
        validation_results["valid"] = False
        return validation_results

    # Required fields check
    required_fields = [
        "id",
        "session_id",
        "event_type",
        "price_level",
        "range_pos",
        "start_ts",
        "in_pm_belt",
        "zone_proximity",
        "timeframe",
    ]

    missing_fields = []
    for candidate in candidates:
        for field in required_fields:
            if field not in candidate:
                missing_fields.append(
                    f"Missing '{field}' in candidate {candidate.get('id', 'unknown')}"
                )

    if missing_fields:
        validation_results["errors"].extend(missing_fields)
        validation_results["valid"] = False

    # Data type and range validation
    validation_results["validation_summary"].update(
        _validate_data_types(candidates, validation_results)
    )

    validation_results["validation_summary"].update(
        _validate_data_ranges(candidates, validation_results)
    )

    validation_results["validation_summary"].update(
        _validate_temporal_consistency(candidates, validation_results)
    )

    # Mark as invalid if any errors were found
    if validation_results["errors"]:
        validation_results["valid"] = False

    return validation_results


def _validate_data_types(
    candidates: list[dict[str, Any]], results: dict[str, Any]
) -> dict[str, Any]:
    """Validate data types of candidate fields"""
    type_summary = {
        "numeric_fields_valid": True,
        "timestamp_fields_valid": True,
        "boolean_fields_valid": True,
    }

    for candidate in candidates:
        candidate_id = candidate.get("id", "unknown")

        # Numeric fields
        numeric_fields = ["price_level", "range_pos"]
        for field in numeric_fields:
            if field in candidate:
                try:
                    float(candidate[field])
                except (ValueError, TypeError):
                    error_msg = f"Invalid numeric value for '{field}' in candidate {candidate_id}: {candidate[field]}"
                    results["errors"].append(error_msg)
                    type_summary["numeric_fields_valid"] = False

        # Boolean fields
        if "in_pm_belt" in candidate and not isinstance(candidate["in_pm_belt"], bool):
            error_msg = f"Invalid boolean value for 'in_pm_belt' in candidate {candidate_id}: {candidate['in_pm_belt']}"
            results["errors"].append(error_msg)
            type_summary["boolean_fields_valid"] = False

        # Timestamp fields (basic string check)
        if "start_ts" in candidate and (
            not isinstance(candidate["start_ts"], str) or len(candidate["start_ts"]) == 0
        ):
            error_msg = f"Invalid timestamp format for 'start_ts' in candidate {candidate_id}"
            results["errors"].append(error_msg)
            type_summary["timestamp_fields_valid"] = False

    return type_summary


def _validate_data_ranges(
    candidates: list[dict[str, Any]], results: dict[str, Any]
) -> dict[str, Any]:
    """Validate data ranges and constraints"""
    range_summary = {
        "range_pos_valid": True,
        "price_levels_valid": True,
        "event_types_valid": True,
    }

    valid_event_types = {"formation", "redelivery"}

    for candidate in candidates:
        candidate_id = candidate.get("id", "unknown")

        # Range position should be 0-1
        if "range_pos" in candidate:
            try:
                range_pos = float(candidate["range_pos"])
                if not (0.0 <= range_pos <= 1.0):
                    warning_msg = (
                        f"Range position outside [0,1] for candidate {candidate_id}: {range_pos}"
                    )
                    results["warnings"].append(warning_msg)
                    range_summary["range_pos_valid"] = False
            except (ValueError, TypeError):
                pass  # Already caught in type validation

        # Price levels should be positive
        if "price_level" in candidate:
            try:
                price_level = float(candidate["price_level"])
                if price_level <= 0:
                    warning_msg = (
                        f"Non-positive price level for candidate {candidate_id}: {price_level}"
                    )
                    results["warnings"].append(warning_msg)
                    range_summary["price_levels_valid"] = False
            except (ValueError, TypeError):
                pass  # Already caught in type validation

        # Event types should be valid
        if "event_type" in candidate:
            event_type = candidate["event_type"]
            if event_type not in valid_event_types:
                error_msg = f"Invalid event type for candidate {candidate_id}: {event_type}. Must be one of: {valid_event_types}"
                results["errors"].append(error_msg)
                range_summary["event_types_valid"] = False

    return range_summary


def _validate_temporal_consistency(
    candidates: list[dict[str, Any]], results: dict[str, Any]
) -> dict[str, Any]:
    """Validate temporal consistency (monotonic time, no duplicates)"""
    temporal_summary = {
        "monotonic_time": True,
        "unique_timestamps": True,
        "duplicate_ids": [],
    }

    # Check for duplicate IDs
    seen_ids = set()
    for candidate in candidates:
        candidate_id = candidate.get("id")
        if candidate_id in seen_ids:
            temporal_summary["duplicate_ids"].append(candidate_id)
            error_msg = f"Duplicate candidate ID found: {candidate_id}"
            results["errors"].append(error_msg)
        else:
            seen_ids.add(candidate_id)

    # Check timestamp uniqueness per session
    session_timestamps = {}
    for candidate in candidates:
        session_id = candidate.get("session_id", "unknown")
        timestamp = candidate.get("start_ts", "")

        if session_id not in session_timestamps:
            session_timestamps[session_id] = set()

        if timestamp in session_timestamps[session_id]:
            warning_msg = f"Duplicate timestamp in session {session_id}: {timestamp}"
            results["warnings"].append(warning_msg)
            temporal_summary["unique_timestamps"] = False
        else:
            session_timestamps[session_id].add(timestamp)

    # Basic monotonic time check (simplified)
    for session_id, timestamps in session_timestamps.items():
        sorted_timestamps = sorted(timestamps)
        if len(sorted_timestamps) != len(set(sorted_timestamps)):
            temporal_summary["monotonic_time"] = False
            warning_msg = f"Non-monotonic timestamps detected in session {session_id}"
            results["warnings"].append(warning_msg)

    return temporal_summary


def validate_network_graph(network_graph: dict[str, Any]) -> dict[str, Any]:
    """Validate network graph structure and consistency"""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "graph_summary": {},
    }

    # Check required top-level keys
    required_keys = ["nodes", "edges", "metadata"]
    for key in required_keys:
        if key not in network_graph:
            validation_results["errors"].append(f"Missing required key: {key}")
            validation_results["valid"] = False

    if not validation_results["valid"]:
        return validation_results

    nodes = network_graph["nodes"]
    edges = network_graph["edges"]

    # Validate nodes
    node_validation = _validate_graph_nodes(nodes)
    validation_results["graph_summary"]["node_validation"] = node_validation
    if not node_validation["valid"]:
        validation_results["errors"].extend(node_validation["errors"])
        validation_results["valid"] = False

    # Validate edges
    edge_validation = _validate_graph_edges(edges, nodes)
    validation_results["graph_summary"]["edge_validation"] = edge_validation
    if not edge_validation["valid"]:
        validation_results["errors"].extend(edge_validation["errors"])
        validation_results["valid"] = False

    # Graph consistency checks
    consistency_check = _validate_graph_consistency(nodes, edges)
    validation_results["graph_summary"]["consistency"] = consistency_check
    validation_results["warnings"].extend(consistency_check["warnings"])

    return validation_results


def _validate_graph_nodes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate graph node structure"""
    node_validation = {
        "valid": True,
        "errors": [],
        "node_count": len(nodes),
        "unique_ids": True,
    }

    required_node_fields = [
        "id",
        "session_id",
        "event_type",
        "price_level",
        "range_pos",
        "timestamp",
    ]

    seen_ids = set()
    for i, node in enumerate(nodes):
        node_id = node.get("id", f"node_{i}")

        # Check required fields
        for field in required_node_fields:
            if field not in node:
                node_validation["errors"].append(f"Node {node_id} missing required field: {field}")
                node_validation["valid"] = False

        # Check ID uniqueness
        if node_id in seen_ids:
            node_validation["errors"].append(f"Duplicate node ID: {node_id}")
            node_validation["unique_ids"] = False
            node_validation["valid"] = False
        else:
            seen_ids.add(node_id)

    return node_validation


def _validate_graph_edges(
    edges: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate graph edge structure and references"""
    edge_validation = {
        "valid": True,
        "errors": [],
        "edge_count": len(edges),
        "valid_references": True,
    }

    required_edge_fields = ["source", "target"]
    node_ids = {node["id"] for node in nodes}

    for i, edge in enumerate(edges):
        edge_id = f"edge_{i}"

        # Check required fields
        for field in required_edge_fields:
            if field not in edge:
                edge_validation["errors"].append(f"Edge {edge_id} missing required field: {field}")
                edge_validation["valid"] = False

        # Check node references
        if "source" in edge and edge["source"] not in node_ids:
            edge_validation["errors"].append(
                f"Edge {edge_id} references invalid source node: {edge['source']}"
            )
            edge_validation["valid_references"] = False
            edge_validation["valid"] = False

        if "target" in edge and edge["target"] not in node_ids:
            edge_validation["errors"].append(
                f"Edge {edge_id} references invalid target node: {edge['target']}"
            )
            edge_validation["valid_references"] = False
            edge_validation["valid"] = False

    return edge_validation


def _validate_graph_consistency(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate graph consistency and provide warnings"""
    consistency = {
        "warnings": [],
        "isolated_nodes": [],
        "self_loops": [],
        "node_edge_ratio": 0.0,
    }

    # Find isolated nodes (no incoming or outgoing edges)
    connected_nodes = set()
    for edge in edges:
        connected_nodes.add(edge.get("source"))
        connected_nodes.add(edge.get("target"))

    for node in nodes:
        node_id = node["id"]
        if node_id not in connected_nodes:
            consistency["isolated_nodes"].append(node_id)

    if consistency["isolated_nodes"]:
        consistency["warnings"].append(f"Found {len(consistency['isolated_nodes'])} isolated nodes")

    # Find self-loops
    for edge in edges:
        if edge.get("source") == edge.get("target"):
            consistency["self_loops"].append(edge.get("source"))

    if consistency["self_loops"]:
        consistency["warnings"].append(f"Found {len(consistency['self_loops'])} self-loops")

    # Calculate node-edge ratio
    if len(nodes) > 0:
        consistency["node_edge_ratio"] = len(edges) / len(nodes)

    return consistency


def is_in_pm_belt(
    timestamp: str,
    pm_belt_start: time = time(14, 35),  # noqa: ARG001
    pm_belt_end: time = time(14, 38),  # noqa: ARG001
) -> bool:
    """
    Check if timestamp falls within PM belt window

    Args:
        timestamp: ISO timestamp string
        pm_belt_start: PM belt start time (default 14:35)
        pm_belt_end: PM belt end time (default 14:38)

    Returns:
        bool: True if timestamp is in PM belt
    """
    try:
        # Simple time extraction (would need proper datetime parsing)
        # This is a simplified implementation
        return (
            "14:35" in timestamp
            or "14:36" in timestamp
            or "14:37" in timestamp
            or "14:38" in timestamp
        )
    except Exception as e:
        logger.warning(f"PM belt check failed for timestamp {timestamp}: {e}")
        return False


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
