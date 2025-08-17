"""FPFVG Network Chain Builder - Pure DFS/Graph Construction Logic."""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def construct_directed_network(
    candidates: list[dict[str, Any]], 
    price_epsilon: float = 5.0,
    range_pos_delta: float = 0.05,
    max_temporal_gap_hours: float = 12.0  # noqa: ARG001
) -> dict[str, Any]:
    """
    Construct directed network of FPFVG events
    
    Network Rules:
    - Node = FPFVG instance (formation or redelivery)
    - Edge A→B if:
      1. B.price_level within ±ε of A.price_level OR B.range_pos within ±δ of A.range_pos
      2. A.end_ts < B.start_ts (temporal ordering)
      3. Optional: same structural strand (within same HTF range)
    """
    nodes = []
    edges = []

    # Create nodes
    for candidate in candidates:
        node = {
            "id": candidate["id"],
            "session_id": candidate["session_id"],
            "event_type": candidate["event_type"],
            "price_level": candidate["price_level"],
            "range_pos": candidate["range_pos"],
            "timestamp": candidate["start_ts"],
            "in_pm_belt": candidate["in_pm_belt"],
            "zone_proximity": candidate["zone_proximity"],
            "timeframe": candidate["timeframe"],
        }
        nodes.append(node)

    # Create edges based on proximity and temporal ordering
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes):
            if i >= j:  # Skip self and already processed pairs
                continue

            # Check temporal ordering
            if not _is_temporally_ordered(node_a, node_b):
                continue

            # Check proximity criteria
            if _meets_proximity_criteria(node_a, node_b, price_epsilon, range_pos_delta):
                edge = _create_network_edge(node_a, node_b)
                edges.append(edge)

    network_graph = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "construction_timestamp": datetime.now().isoformat(),
        },
    }

    logger.info(f"Constructed network: {len(nodes)} nodes, {len(edges)} edges")
    return network_graph


def _is_temporally_ordered(node_a: dict[str, Any], node_b: dict[str, Any]) -> bool:
    """Check if node_a occurs before node_b temporally"""
    try:
        # Simplified temporal comparison (would need proper datetime parsing)
        timestamp_a = node_a["timestamp"]
        timestamp_b = node_b["timestamp"]

        # Basic string comparison (assumes timestamps are in sortable format)
        return timestamp_a < timestamp_b
    except Exception:
        return False


def _meets_proximity_criteria(
    node_a: dict[str, Any], 
    node_b: dict[str, Any],
    price_epsilon: float = 5.0,
    range_pos_delta: float = 0.05
) -> bool:
    """Check if nodes meet proximity criteria for edge creation"""
    # Price proximity check
    price_a = node_a["price_level"]
    price_b = node_b["price_level"]

    if price_a > 0 and price_b > 0:
        price_distance = abs(price_a - price_b)
        if price_distance <= price_epsilon:
            return True

    # Range position proximity check
    range_pos_a = node_a["range_pos"]
    range_pos_b = node_b["range_pos"]
    range_pos_distance = abs(range_pos_a - range_pos_b)

    if range_pos_distance <= range_pos_delta:
        return True

    # Optional: same structural strand check (simplified)
    return node_a["session_id"] == node_b["session_id"]


def _create_network_edge(node_a: dict[str, Any], node_b: dict[str, Any]) -> dict[str, Any]:
    """Create network edge between two nodes"""
    # Calculate edge features
    price_distance = abs(node_a["price_level"] - node_b["price_level"])
    delta_range_pos = abs(node_a["range_pos"] - node_b["range_pos"])
    
    # Time delta calculation (simplified)
    delta_t_minutes = _calculate_time_delta_minutes(
        node_a["timestamp"], node_b["timestamp"]
    )

    # Zone alignment flags
    zone_prox_a = node_a["zone_proximity"]
    zone_prox_b = node_b["zone_proximity"]
    
    same_zone_flags = {}
    if "closest_zones" in zone_prox_a and "closest_zones" in zone_prox_b:
        zones_a = zone_prox_a["closest_zones"]
        zones_b = zone_prox_b["closest_zones"]
        
        # Check if both nodes are in the same zones
        for zone in ["20%", "40%", "50%", "61.8%", "80%"]:
            same_zone_flags[zone] = zone in zones_a and zone in zones_b

    edge = {
        "source": node_a["id"],
        "target": node_b["id"],
        "source_session": node_a["session_id"],
        "target_session": node_b["session_id"],
        "price_distance": price_distance,
        "delta_range_pos": delta_range_pos,
        "delta_t_minutes": delta_t_minutes,
        "same_zone_flags": same_zone_flags,
        "cross_session": node_a["session_id"] != node_b["session_id"],
        "edge_type": f"{node_a['event_type']}_to_{node_b['event_type']}",
    }

    return edge


def _calculate_time_delta_minutes(timestamp_a: str, timestamp_b: str) -> float:
    """Calculate time difference in minutes between two timestamps"""
    try:
        # This would need proper datetime parsing in practice
        # For now, return a placeholder
        if timestamp_a == timestamp_b:
            return 0.0
        
        # Simplified: use string comparison as proxy
        if timestamp_a < timestamp_b:
            return 30.0  # Placeholder
        else:
            return -30.0  # Should not happen if properly ordered
            
    except Exception as e:
        logger.warning(f"Time delta calculation failed: {e}")
        return 0.0


def calculate_network_density(network_graph: dict[str, Any]) -> float:
    """Calculate network density (edges / max_possible_edges)"""
    nodes = len(network_graph["nodes"])
    edges = len(network_graph["edges"])
    
    if nodes <= 1:
        return 0.0
    
    max_possible_edges = nodes * (nodes - 1)  # Directed graph
    return edges / max_possible_edges


def identify_network_motifs(network_graph: dict[str, Any]) -> dict[str, Any]:
    """Identify common network motifs (chains, convergences, divergences)"""
    adjacency = {}
    
    # Build adjacency list
    for node in network_graph["nodes"]:
        adjacency[node["id"]] = []
    
    for edge in network_graph["edges"]:
        source = edge["source"]
        target = edge["target"]
        if source in adjacency:
            adjacency[source].append(target)
    
    # Find motifs
    chains = find_chains(adjacency, min_length=3)
    convergences = _find_convergences(adjacency)
    divergences = _find_divergences(adjacency)
    
    return {
        "chains": chains,
        "convergences": convergences,
        "divergences": divergences,
        "chain_count": len(chains),
        "convergence_count": len(convergences),
        "divergence_count": len(divergences),
    }


def find_chains(adjacency: dict[str, list[str]], min_length: int = 3) -> list[list[str]]:
    """Find chains in the network using DFS"""
    chains = []
    visited_in_chain = set()

    def dfs_chain(node, current_chain, visited):
        visited.add(node)
        current_chain.append(node)
        
        # Explore neighbors
        extended = False
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                dfs_chain(neighbor, current_chain.copy(), visited.copy())
                extended = True
        
        # If no extensions and chain is long enough, save it
        if not extended and len(current_chain) >= min_length:
            chain_key = tuple(current_chain)
            if chain_key not in visited_in_chain:
                chains.append(current_chain.copy())
                visited_in_chain.add(chain_key)

    # Start DFS from each node
    for start_node in adjacency:
        if start_node not in visited_in_chain:
            dfs_chain(start_node, [], set())

    return chains


def _find_convergences(adjacency: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Find convergence motifs (multiple nodes pointing to one)"""
    convergences = []
    in_degree = {}
    
    # Calculate in-degrees
    for node in adjacency:
        in_degree[node] = 0
    
    for _node, neighbors in adjacency.items():
        for neighbor in neighbors:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    
    # Find nodes with in-degree >= 2
    for node, degree in in_degree.items():
        if degree >= 2:
            # Find all predecessors
            predecessors = []
            for pred_node, neighbors in adjacency.items():
                if node in neighbors:
                    predecessors.append(pred_node)
            
            convergences.append({
                "target": node,
                "sources": predecessors,
                "in_degree": degree,
            })
    
    return convergences


def _find_divergences(adjacency: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Find divergence motifs (one node pointing to multiple)"""
    divergences = []
    
    for node, neighbors in adjacency.items():
        if len(neighbors) >= 2:
            divergences.append({
                "source": node,
                "targets": neighbors,
                "out_degree": len(neighbors),
            })
    
    return divergences