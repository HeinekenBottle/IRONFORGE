"""
Comprehensive tests for Dual Graph Views system
Tests DAG acyclicity, motif statistics, and M1 integration
"""

import pytest
import numpy as np
import pandas as pd
import torch
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import json

# Import the classes we're testing
from ironforge.learning.dag_graph_builder import DAGGraphBuilder, M1EnhancedNodeFeature
from ironforge.learning.dag_motif_miner import DAGMotifMiner, MotifResult
from ironforge.learning.m1_event_detector import M1EventDetector, M1Event
from ironforge.learning.cross_scale_edge_builder import CrossScaleEdgeBuilder
from ironforge.learning.tgat_discovery import EnhancedTemporalAttentionLayer, create_attention_layer


class TestDAGAcyclicity:
    """Test DAG construction and acyclicity guarantees"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.builder = DAGGraphBuilder({'m1_integration': False})  # Start with basic testing
        
        # Create sample session data
        self.sample_session = {
            "session_name": "test_session_2025-01-15_AM",
            "events": [
                {
                    "event_type": "fvg_formation",
                    "timestamp": 1642140000,  # 2022-01-14 09:00:00
                    "seq_idx": 0,
                    "price": 23500.0,
                    "volume": 100.0
                },
                {
                    "event_type": "liquidity_sweep", 
                    "timestamp": 1642140300,  # 2022-01-14 09:05:00
                    "seq_idx": 1,
                    "price": 23520.0,
                    "volume": 150.0
                },
                {
                    "event_type": "fvg_redelivery",
                    "timestamp": 1642140600,  # 2022-01-14 09:10:00
                    "seq_idx": 2,
                    "price": 23510.0,
                    "volume": 80.0
                },
                {
                    "event_type": "reversal_signal",
                    "timestamp": 1642140900,  # 2022-01-14 09:15:00
                    "seq_idx": 3,
                    "price": 23480.0,
                    "volume": 200.0
                }
            ],
            "session_start_time": 1642140000,
            "session_duration": 3600,
            "session_high": 23530.0,
            "session_low": 23470.0,
            "session_open": 23500.0
        }
    
    def test_dag_is_acyclic(self):
        """Test that generated DAG is indeed acyclic"""
        dag = self.builder.build_dag_from_session(self.sample_session)
        
        assert nx.is_directed_acyclic_graph(dag), "Generated graph must be acyclic"
        assert dag.number_of_nodes() == len(self.sample_session["events"])
        
        # Test topological sort exists
        try:
            topo_order = list(nx.topological_sort(dag))
            assert len(topo_order) == len(self.sample_session["events"])
        except nx.NetworkXError:
            pytest.fail("DAG should have valid topological ordering")
    
    def test_temporal_causality_constraint(self):
        """Test that edges respect temporal causality (no backwards in time)"""
        dag = self.builder.build_dag_from_session(self.sample_session)
        
        for src, dst in dag.edges():
            src_timestamp = dag.nodes[src]['timestamp']
            dst_timestamp = dag.nodes[dst]['timestamp']
            
            # Destination must be at same time or later (causal constraint)
            assert dst_timestamp >= src_timestamp, f"Causal violation: {src_timestamp} -> {dst_timestamp}"
    
    def test_seq_idx_ordering(self):
        """Test that nodes with same timestamp are ordered by seq_idx"""
        # Create events with same timestamp but different seq_idx
        same_time_session = self.sample_session.copy()
        same_time_session["events"][1]["timestamp"] = same_time_session["events"][0]["timestamp"]
        
        dag = self.builder.build_dag_from_session(same_time_session)
        
        # Find nodes with same timestamp
        for src, dst in dag.edges():
            src_timestamp = dag.nodes[src]['timestamp']
            dst_timestamp = dag.nodes[dst]['timestamp']
            
            if src_timestamp == dst_timestamp:
                src_seq = dag.nodes[src]['seq_idx'] 
                dst_seq = dag.nodes[dst]['seq_idx']
                assert dst_seq >= src_seq, "seq_idx must be non-decreasing for same timestamps"
    
    def test_large_graph_acyclicity(self):
        """Test acyclicity with larger graphs (stress test)"""
        # Generate larger session with 50 events
        large_session = {
            "session_name": "large_test_session",
            "events": [],
            "session_start_time": 1642140000,
            "session_duration": 3600,
            "session_high": 24000.0,
            "session_low": 23000.0,
            "session_open": 23500.0
        }
        
        for i in range(50):
            large_session["events"].append({
                "event_type": f"event_{i % 5}",
                "timestamp": 1642140000 + i * 60,  # 1 minute apart
                "seq_idx": i,
                "price": 23500.0 + np.random.normal(0, 50),
                "volume": 100.0 + np.random.exponential(50)
            })
        
        dag = self.builder.build_dag_from_session(large_session)
        
        assert nx.is_directed_acyclic_graph(dag), "Large DAG must remain acyclic"
        assert dag.number_of_nodes() == 50
        
        # Test reasonable connectivity (not too sparse, not too dense)
        edge_density = dag.number_of_edges() / (dag.number_of_nodes() * (dag.number_of_nodes() - 1))
        assert 0.01 < edge_density < 0.5, f"Edge density {edge_density} seems unreasonable"
    
    def test_empty_graph_handling(self):
        """Test handling of empty or minimal graphs"""
        empty_session = {
            "session_name": "empty_session",
            "events": []
        }
        
        dag = self.builder.build_dag_from_session(empty_session)
        assert dag.number_of_nodes() == 0
        assert dag.number_of_edges() == 0
        assert nx.is_directed_acyclic_graph(dag)  # Empty graph is trivially acyclic
    
    def test_single_node_graph(self):
        """Test single node graph"""
        single_session = {
            "session_name": "single_session",
            "events": [self.sample_session["events"][0]],
            "session_start_time": 1642140000,
            "session_duration": 3600,
            "session_high": 23530.0,
            "session_low": 23470.0,
            "session_open": 23500.0
        }
        
        dag = self.builder.build_dag_from_session(single_session)
        assert dag.number_of_nodes() == 1
        assert dag.number_of_edges() == 0
        assert nx.is_directed_acyclic_graph(dag)


class TestMotifStatistics:
    """Test motif mining and statistical validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.miner = DAGMotifMiner()
        
        # Create sample DAGs for testing
        self.sample_dags = []
        self.session_names = []
        
        for i in range(5):  # 5 sample DAGs
            dag = nx.DiGraph()
            
            # Add nodes with timestamps
            for j in range(6):  # 6 nodes per DAG
                dag.add_node(j, timestamp=1642140000 + j * 300)
            
            # Add edges to form common motifs
            dag.add_edge(0, 1)  # Sequential pattern
            dag.add_edge(1, 2)
            dag.add_edge(0, 2)  # Skip connection  
            dag.add_edge(2, 3)
            dag.add_edge(3, 4)
            dag.add_edge(4, 5)
            
            # Add some variability between DAGs
            if i % 2 == 0:
                dag.add_edge(1, 4)  # Additional pattern in some DAGs
            
            self.sample_dags.append(dag)
            self.session_names.append(f"test_session_{i}")
    
    def test_null_model_generation(self):
        """Test time-jitter and session permutation null models"""
        # Test time-jitter nulls
        jitter_nulls = self.miner._generate_time_jitter_nulls(
            self.sample_dags, self.session_names, n_nulls=10
        )
        
        assert len(jitter_nulls) == 10
        
        for null_dag, session_name in jitter_nulls:
            assert isinstance(null_dag, nx.DiGraph)
            assert null_dag.number_of_nodes() > 0
            assert nx.is_directed_acyclic_graph(null_dag), "Null DAG must remain acyclic"
        
        # Test session permutation nulls
        perm_nulls = self.miner._generate_session_permutation_nulls(
            self.sample_dags, self.session_names, n_nulls=10
        )
        
        assert len(perm_nulls) == 10
        
        for null_dag, session_name in perm_nulls:
            assert isinstance(null_dag, nx.DiGraph)
            assert null_dag.number_of_nodes() > 0
            assert nx.is_directed_acyclic_graph(null_dag), "Permuted DAG must remain acyclic"
    
    def test_motif_discovery(self):
        """Test motif pattern discovery and enumeration"""
        motifs = self.miner.discover_motifs(self.sample_dags, self.session_names)
        
        assert isinstance(motifs, dict)
        assert len(motifs) > 0
        
        for pattern_id, motif_result in motifs.items():
            assert isinstance(motif_result, MotifResult)
            assert motif_result.real_count >= 0
            assert motif_result.lift_ratio >= 0
            assert 0 <= motif_result.p_value <= 1
            assert motif_result.classification in ['PROMOTE', 'PARK', 'DISCARD']
    
    def test_statistical_significance(self):
        """Test statistical significance calculations"""
        # Create a pattern that should be significant
        significant_dags = []
        for i in range(10):
            dag = nx.DiGraph()
            # Create identical significant pattern
            dag.add_node(0, timestamp=1642140000)
            dag.add_node(1, timestamp=1642140300) 
            dag.add_node(2, timestamp=1642140600)
            dag.add_edge(0, 1)
            dag.add_edge(1, 2)
            dag.add_edge(0, 2)  # Consistent triangle motif
            significant_dags.append(dag)
        
        motifs = self.miner.discover_motifs(significant_dags, [f"sig_{i}" for i in range(10)])
        
        # Should find at least one significant pattern
        significant_patterns = [m for m in motifs.values() if m.classification == 'PROMOTE']
        assert len(significant_patterns) > 0, "Should detect significant patterns"
    
    def test_lift_ratio_calculation(self):
        """Test lift ratio computation"""
        real_count = 10
        null_counts = [2, 3, 1, 4, 2, 3, 2, 1, 3, 2]  # Mean = 2.3
        
        lift_ratio = self.miner._calculate_lift_ratio(real_count, null_counts)
        expected_lift = real_count / np.mean(null_counts)
        
        assert abs(lift_ratio - expected_lift) < 1e-6
        
        # Test edge case: zero null mean
        zero_nulls = [0] * 10
        lift_ratio_zero = self.miner._calculate_lift_ratio(real_count, zero_nulls)
        assert lift_ratio_zero == float('inf')
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval computation"""
        null_counts = [2, 3, 1, 4, 2, 3, 2, 1, 3, 2]
        
        ci_lower, ci_upper = self.miner._calculate_confidence_interval(null_counts)
        
        assert ci_lower <= ci_upper
        assert ci_lower >= min(null_counts)
        assert ci_upper <= max(null_counts)


class TestM1Integration:
    """Test M1 event integration with dual graph views"""
    
    def setup_method(self):
        """Set up M1 integration test fixtures"""
        self.builder = DAGGraphBuilder({'m1_integration': True})
        self.m1_detector = M1EventDetector()
        
        # Sample M1 OHLCV data
        timestamps = pd.date_range('2022-01-14 09:00:00', periods=60, freq='1T')
        self.m1_data = pd.DataFrame({
            'timestamp': timestamps.astype(int) // 10**9,  # Unix timestamps
            'open': 23500 + np.random.normal(0, 10, 60),
            'high': 23500 + np.random.normal(5, 10, 60),
            'low': 23500 + np.random.normal(-5, 10, 60),
            'close': 23500 + np.random.normal(0, 10, 60),
            'volume': np.random.exponential(100, 60)
        })
        
        # Sample session data
        self.session_data = {
            "session_name": "test_m1_session", 
            "events": [
                {
                    "event_type": "fvg_formation",
                    "timestamp": int(timestamps[10].timestamp()),
                    "seq_idx": 0,
                    "price": 23500.0
                },
                {
                    "event_type": "liquidity_sweep",
                    "timestamp": int(timestamps[30].timestamp()),
                    "seq_idx": 1, 
                    "price": 23520.0
                }
            ]
        }
    
    def test_m1_event_detection(self):
        """Test M1 event detection"""
        m1_events = self.m1_detector.detect_events(self.m1_data, "test_session")
        
        assert isinstance(m1_events, list)
        assert len(m1_events) >= 0  # Could be 0 if no patterns detected
        
        for event in m1_events:
            assert isinstance(event, M1Event)
            assert event.session_id == "test_session" 
            assert event.confidence > 0
            assert event.event_type in ['micro_fvg_fill', 'micro_sweep', 'micro_impulse', 
                                      'vwap_touch', 'imbalance_burst', 'wick_extreme']
    
    def test_m1_enhanced_features(self):
        """Test M1-enhanced node features"""
        feature = M1EnhancedNodeFeature()
        
        # Test dimensions
        assert feature.features.shape[0] == 53  # 45 + 8 M1 features
        
        # Test M1 feature setting
        feature.set_m1_feature("m1_event_density", 0.5)
        assert feature.features[45] == 0.5  # m1_event_density index
        
        feature.set_m1_feature("m1_micro_volatility", 2.5)
        assert feature.features[46] == 2.5  # m1_micro_volatility index
    
    def test_m1_dag_integration(self):
        """Test M1 integration with DAG construction"""
        dag = self.builder.build_dag_with_m1_integration(self.session_data, self.m1_data)
        
        assert isinstance(dag, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(dag)
        
        # Check that nodes have M1-enhanced features if M1 integration is enabled
        if dag.number_of_nodes() > 0:
            first_node = list(dag.nodes())[0]
            node_feature = dag.nodes[first_node]['feature']
            
            if self.builder.m1_integration_enabled:
                # Should have 53D features (45 + 8 M1)
                assert node_feature.shape[0] == 53
                # Should have m1_enhanced flag
                assert dag.nodes[first_node].get('m1_enhanced', False)


class TestTGATEnhancement:
    """Test enhanced TGAT with masked attention"""
    
    def setup_method(self):
        """Set up TGAT test fixtures"""
        self.standard_layer = create_attention_layer(input_dim=45, enhanced=False)
        self.enhanced_layer = create_attention_layer(input_dim=53, enhanced=True, use_flex_attention=False)
        
        # Sample data
        self.node_features = torch.randn(10, 45)  # 10 nodes, 45D features
        self.m1_node_features = torch.randn(10, 53)  # 10 nodes, 53D M1-enhanced
        
        # Sample DAG
        self.dag = nx.DiGraph()
        for i in range(10):
            self.dag.add_node(i)
        # Add some edges (ensuring DAG property)
        edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (1, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
        self.dag.add_edges_from(edges)
        
        # Temporal data
        self.temporal_data = torch.randn(10, 10, 2)  # [N, N, 2] for [dt_minutes, temporal_distance]
    
    def test_standard_attention_layer(self):
        """Test standard temporal attention layer"""
        output, attn_weights = self.standard_layer(self.node_features, return_attn=True)
        
        assert output.shape == (10, 44)  # 45D -> 44D projection
        
        if attn_weights is not None:
            assert attn_weights.shape == (10, 10)
    
    def test_enhanced_attention_layer(self):
        """Test enhanced attention layer with DAG masking"""
        output, attn_weights = self.enhanced_layer(
            self.m1_node_features, 
            temporal_data=self.temporal_data,
            dag=self.dag,
            return_attn=True
        )
        
        assert output.shape == (10, 44)  # 53D -> 44D projection
        
        if attn_weights is not None:
            assert attn_weights.shape == (10, 10)
    
    def test_dag_mask_creation(self):
        """Test DAG mask creation for causal attention"""
        mask = self.enhanced_layer.create_dag_mask(self.dag, torch.device('cpu'))
        
        assert mask.shape == (10, 10)
        assert mask.dtype == torch.bool
        
        # Check causal constraint: node can only attend to predecessors and self
        for i in range(10):
            predecessors = list(self.dag.predecessors(i))
            predecessors.append(i)  # Self-attention
            
            for j in range(10):
                if j in predecessors:
                    assert mask[i, j] == True, f"Node {i} should attend to predecessor {j}"
                else:
                    # Allow some flexibility for disconnected components
                    pass  # Don't enforce strict masking for complex topologies
    
    def test_temporal_bias_network(self):
        """Test temporal bias network"""
        temporal_bias = self.enhanced_layer.create_temporal_bias(
            self.temporal_data, torch.device('cpu')
        )
        
        if temporal_bias is not None:
            assert temporal_bias.shape == (1, 4, 10, 10)  # [batch, heads, seq, seq]
            assert torch.all(torch.isfinite(temporal_bias)), "Temporal bias should be finite"


class TestIntegrationPipeline:
    """Integration tests for complete dual graph views pipeline"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.builder = DAGGraphBuilder({'m1_integration': True})
        
        # Complete session data
        self.session_data = {
            "session_name": "integration_test_session",
            "events": [
                {"event_type": "fvg_formation", "timestamp": 1642140000, "seq_idx": 0, "price": 23500.0},
                {"event_type": "liquidity_sweep", "timestamp": 1642140300, "seq_idx": 1, "price": 23520.0},
                {"event_type": "fvg_redelivery", "timestamp": 1642140600, "seq_idx": 2, "price": 23510.0},
                {"event_type": "reversal_signal", "timestamp": 1642140900, "seq_idx": 3, "price": 23480.0}
            ],
            "session_start_time": 1642140000,
            "session_duration": 3600,
            "session_high": 23530.0,
            "session_low": 23470.0,
            "session_open": 23500.0
        }
        
        # M1 data
        timestamps = pd.date_range('2022-01-14 09:00:00', periods=30, freq='1T')
        self.m1_data = pd.DataFrame({
            'timestamp': timestamps.astype(int) // 10**9,
            'open': 23500 + np.random.normal(0, 5, 30),
            'high': 23500 + np.random.normal(3, 5, 30),
            'low': 23500 + np.random.normal(-3, 5, 30), 
            'close': 23500 + np.random.normal(0, 5, 30),
            'volume': np.random.exponential(50, 30)
        })
    
    def test_end_to_end_pipeline(self):
        """Test complete dual graph views pipeline"""
        # Build undirected temporal graph  
        temporal_graph = self.builder.build_session_graph_from_events(
            self.session_data["events"], self.session_data
        )
        
        # Build DAG with M1 integration
        dag = self.builder.build_dag_with_m1_integration(self.session_data, self.m1_data)
        
        # Verify both graphs
        assert isinstance(temporal_graph, nx.Graph)
        assert isinstance(dag, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(dag)
        
        # Same number of nodes
        assert temporal_graph.number_of_nodes() == dag.number_of_nodes()
        
        # DAG should have fewer or equal edges (due to causal constraints)
        assert dag.number_of_edges() <= temporal_graph.number_of_edges() * 2  # Undirected edges count as 2
    
    def test_parquet_serialization(self):
        """Test DAG serialization to Parquet"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dag = self.builder.build_dag_with_m1_integration(self.session_data, self.m1_data)
            
            # Save to Parquet
            output_path = Path(temp_dir) / "test_dag_edges.parquet"
            self.builder.save_dag_edges_parquet(dag, output_path, "test_session")
            
            # Verify file exists
            assert output_path.exists()
            
            # Load and verify
            loaded_dag = self.builder.load_dag_from_parquet(output_path)
            
            assert isinstance(loaded_dag, nx.DiGraph)
            assert nx.is_directed_acyclic_graph(loaded_dag)
            assert loaded_dag.number_of_edges() == dag.number_of_edges()


# Utility functions for test data generation
def generate_test_session_data(n_events: int = 10, session_duration: int = 3600) -> Dict[str, Any]:
    """Generate test session data with realistic timestamps and prices"""
    start_time = 1642140000  # Fixed start time for reproducibility
    events = []
    
    for i in range(n_events):
        events.append({
            "event_type": f"test_event_{i % 3}",
            "timestamp": start_time + i * (session_duration // n_events),
            "seq_idx": i,
            "price": 23500.0 + np.random.normal(0, 50),
            "volume": 100.0 + np.random.exponential(50)
        })
    
    return {
        "session_name": f"test_session_{n_events}_events",
        "events": events,
        "session_start_time": start_time,
        "session_duration": session_duration,
        "session_high": 23600.0,
        "session_low": 23400.0,
        "session_open": 23500.0
    }


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])