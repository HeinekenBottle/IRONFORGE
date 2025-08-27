"""
Unit tests for DAG acyclicity validation
Focused tests without complex dependencies
"""

import pytest
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict

# Test the standalone DAG edge building function
from ironforge.learning.dag_graph_builder import build_dag_edges


def test_dag_edges_temporal_ordering():
    """Test that DAG edges respect temporal ordering"""
    
    # Create sample nodes with timestamps
    nodes = [
        {"timestamp": 1642140000, "seq_idx": 0, "event_type": "event_0"},
        {"timestamp": 1642140300, "seq_idx": 1, "event_type": "event_1"}, 
        {"timestamp": 1642140600, "seq_idx": 2, "event_type": "event_2"},
        {"timestamp": 1642140900, "seq_idx": 3, "event_type": "event_3"}
    ]
    
    # Build DAG edges
    edges = build_dag_edges(nodes, dt_min=1, dt_max=120, k=3)
    
    # Verify temporal causality
    for src, dst, edge_data in edges:
        src_timestamp = nodes[src]["timestamp"]
        dst_timestamp = nodes[dst]["timestamp"]
        
        # Destination must be at same time or later
        assert dst_timestamp >= src_timestamp, f"Temporal violation: {src_timestamp} -> {dst_timestamp}"
        
        # Time delta should be within bounds
        dt_minutes = edge_data["dt_minutes"]
        assert 1 <= dt_minutes <= 120, f"Time delta {dt_minutes} outside bounds [1, 120]"


def test_dag_acyclicity_with_networkx():
    """Test that constructed DAG is acyclic using NetworkX"""
    
    # Create larger set of nodes
    nodes = []
    base_time = 1642140000
    
    for i in range(10):
        nodes.append({
            "timestamp": base_time + i * 300,  # 5 minutes apart
            "seq_idx": i,
            "event_type": f"event_{i}"
        })
    
    # Build edges
    edges = build_dag_edges(nodes, dt_min=1, dt_max=60, k=3)
    
    # Create NetworkX DiGraph
    dag = nx.DiGraph()
    for i in range(len(nodes)):
        dag.add_node(i)
    
    for src, dst, edge_data in edges:
        dag.add_edge(src, dst, **edge_data)
    
    # Test acyclicity
    assert nx.is_directed_acyclic_graph(dag), "DAG must be acyclic"
    
    # Test topological sort exists
    topo_order = list(nx.topological_sort(dag))
    assert len(topo_order) == len(nodes)


def test_same_timestamp_seq_idx_ordering():
    """Test seq_idx ordering for nodes with same timestamp"""
    
    # Create nodes with same timestamp but different seq_idx
    nodes = [
        {"timestamp": 1642140000, "seq_idx": 0, "event_type": "event_0"},
        {"timestamp": 1642140000, "seq_idx": 1, "event_type": "event_1"},
        {"timestamp": 1642140000, "seq_idx": 2, "event_type": "event_2"},
        {"timestamp": 1642140300, "seq_idx": 3, "event_type": "event_3"}
    ]
    
    edges = build_dag_edges(nodes, dt_min=0, dt_max=30, k=2)
    
    # Create DAG
    dag = nx.DiGraph()
    for i in range(len(nodes)):
        dag.add_node(i, **nodes[i])
    
    for src, dst, edge_data in edges:
        dag.add_edge(src, dst, **edge_data)
    
    # Check that nodes with same timestamp are ordered by seq_idx
    for src, dst in dag.edges():
        src_timestamp = dag.nodes[src]["timestamp"]
        dst_timestamp = dag.nodes[dst]["timestamp"]
        
        if src_timestamp == dst_timestamp:
            src_seq = dag.nodes[src]["seq_idx"]
            dst_seq = dag.nodes[dst]["seq_idx"] 
            assert dst_seq >= src_seq, f"seq_idx violation: {src_seq} -> {dst_seq}"


def test_edge_connectivity_parameters():
    """Test that k parameter controls connectivity"""
    
    nodes = []
    for i in range(8):
        nodes.append({
            "timestamp": 1642140000 + i * 600,  # 10 minutes apart
            "seq_idx": i,
            "event_type": f"event_{i}"
        })
    
    # Test different k values
    for k in [1, 2, 4]:
        edges = build_dag_edges(nodes, dt_min=1, dt_max=120, k=k)
        
        # Count outgoing edges per node
        outgoing_counts = {}
        for src, dst, edge_data in edges:
            outgoing_counts[src] = outgoing_counts.get(src, 0) + 1
        
        # Most nodes should have <= k outgoing edges
        for node_idx, count in outgoing_counts.items():
            assert count <= k, f"Node {node_idx} has {count} > {k} outgoing edges"


def test_temporal_distance_bounds():
    """Test temporal distance constraints"""
    
    nodes = []
    base_time = 1642140000
    
    # Create nodes with varying time gaps
    times = [0, 60, 120, 300, 600, 1200, 2400]  # Seconds from base
    for i, time_offset in enumerate(times):
        nodes.append({
            "timestamp": base_time + time_offset,
            "seq_idx": i,
            "event_type": f"event_{i}"
        })
    
    # Test with tight time bounds
    edges = build_dag_edges(nodes, dt_min=1, dt_max=5, k=3)  # Max 5 minutes
    
    for src, dst, edge_data in edges:
        dt_minutes = edge_data["dt_minutes"]
        assert 1 <= dt_minutes <= 5, f"Time delta {dt_minutes} violates bounds [1, 5]"


def test_empty_and_single_node():
    """Test edge cases: empty and single node"""
    
    # Empty nodes
    edges = build_dag_edges([], dt_min=1, dt_max=120, k=3)
    assert len(edges) == 0
    
    # Single node
    single_node = [{"timestamp": 1642140000, "seq_idx": 0, "event_type": "single"}]
    edges = build_dag_edges(single_node, dt_min=1, dt_max=120, k=3)
    assert len(edges) == 0  # No edges possible with single node


def test_large_dag_performance():
    """Test performance with larger DAGs"""
    
    # Create 100 nodes
    nodes = []
    base_time = 1642140000
    
    for i in range(100):
        nodes.append({
            "timestamp": base_time + i * 60 + np.random.randint(-30, 30),  # Add jitter
            "seq_idx": i,
            "event_type": f"event_{i % 5}"
        })
    
    # Sort by timestamp, seq_idx to ensure proper ordering
    nodes.sort(key=lambda x: (x["timestamp"], x["seq_idx"]))
    
    # Build edges
    edges = build_dag_edges(nodes, dt_min=1, dt_max=30, k=4)
    
    # Create and validate DAG
    dag = nx.DiGraph()
    for i in range(len(nodes)):
        dag.add_node(i)
    
    for src, dst, edge_data in edges:
        dag.add_edge(src, dst)
    
    # Must be acyclic
    assert nx.is_directed_acyclic_graph(dag)
    
    # Should have reasonable connectivity
    assert 0 < len(edges) < len(nodes) * 10  # Not too sparse or too dense


if __name__ == "__main__":
    # Run tests
    test_functions = [
        test_dag_edges_temporal_ordering,
        test_dag_acyclicity_with_networkx, 
        test_same_timestamp_seq_idx_ordering,
        test_edge_connectivity_parameters,
        test_temporal_distance_bounds,
        test_empty_and_single_node,
        test_large_dag_performance
    ]
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            print(f"✅ {test_func.__name__} passed")
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            
    print("\nDAG acyclicity tests completed!")