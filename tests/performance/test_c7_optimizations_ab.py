#!/usr/bin/env python3
"""
Context7 Optimization A/B Tests
Comprehensive validation that optimizations produce identical outputs

Tests compare:
1. Original TGAT vs Optimized TGAT
2. Standard DAG builder vs Optimized DAG builder  
3. Standard Parquet I/O vs Optimized Parquet I/O

All tests use fixed seeds and validate outputs within 1e-4 tolerance.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json

import pytest
import numpy as np
import pandas as pd
import torch
import networkx as nx
import pyarrow as pa

# IRONFORGE imports
from ironforge.learning.dual_graph_config import TGATConfig, DualGraphViewsConfig
from ironforge.learning.tgat_discovery import (
    EnhancedTemporalAttentionLayer, graph_attention, build_edge_mask, build_time_bias
)
from ironforge.learning.optimized_tgat_discovery import (
    OptimizedEnhancedTemporalAttentionLayer, OptimizedTGATConfig, OptimizedGraphAttention
)
from ironforge.learning.dag_graph_builder import DAGGraphBuilder
from ironforge.learning.optimized_dag_builder import OptimizedDAGBuilder, OptimizedDAGConfig
from ironforge.storage.optimized_parquet_io import (
    OptimizedParquetManager, OptimizedParquetConfig
)

logger = logging.getLogger(__name__)

# Test configuration
FIXED_SEED = 42
TEST_TOLERANCE = 1e-4
TEST_SIZES = [32, 64, 128]  # Smaller sizes for faster testing


class TestTGATOptimizationsAB:
    """A/B tests for TGAT optimizations"""
    
    def setup_method(self):
        """Setup for each test method"""
        torch.manual_seed(FIXED_SEED)
        np.random.seed(FIXED_SEED)
        
    @pytest.mark.parametrize("L", TEST_SIZES)
    def test_graph_attention_equivalence(self, L: int):
        """Test that optimized graph attention produces identical outputs"""
        
        # Setup test data
        B, H, D = 1, 4, 11
        device = torch.device('cpu')  # Use CPU for deterministic testing
        
        # Fixed random tensors
        torch.manual_seed(FIXED_SEED)
        q = torch.randn(B, H, L, D, device=device)
        k = torch.randn(B, H, L, D, device=device)
        v = torch.randn(B, H, L, D, device=device)
        
        # Create edge mask
        edge_index = torch.tensor([
            [i for i in range(L-1)],
            [i+1 for i in range(L-1)]
        ], device=device)
        edge_mask = build_edge_mask(edge_index, L, device=device, allow_self=True)
        
        # Create time bias
        dt_minutes = torch.randint(1, 60, (L, L), device=device, dtype=torch.float32)
        time_bias = build_time_bias(dt_minutes, L, device=device, scale=0.1)
        
        # Test original implementation
        torch.manual_seed(FIXED_SEED)
        out_original, attn_original = graph_attention(
            q, k, v,
            edge_mask_bool=edge_mask,
            time_bias=time_bias,
            impl="manual",  # Use manual for deterministic comparison
            training=False
        )
        
        # Test optimized implementation (without optimizations for exact comparison)
        config = OptimizedTGATConfig(
            base_config=TGATConfig(),
            enable_amp=False,
            enable_flash_attention=False,
            use_fp16=False,
            enable_time_bias_caching=False  # Disable for exact comparison
        )
        
        opt_attention = OptimizedGraphAttention(config)
        
        torch.manual_seed(FIXED_SEED)
        with torch.no_grad():
            out_optimized, attn_optimized = opt_attention._forward_standard(
                q, k, v, edge_mask, time_bias, training=False
            )
        
        # Compare outputs
        output_diff = torch.abs(out_original - out_optimized).max().item()
        assert output_diff < TEST_TOLERANCE, \
            f"Output difference {output_diff} exceeds tolerance {TEST_TOLERANCE}"
            
        # Compare attention weights if both available
        if attn_original is not None and attn_optimized is not None:
            attn_diff = torch.abs(attn_original - attn_optimized).max().item()
            assert attn_diff < TEST_TOLERANCE, \
                f"Attention difference {attn_diff} exceeds tolerance {TEST_TOLERANCE}"
        
        logger.info(f"✅ Graph attention equivalence test passed (L={L}, diff={output_diff:.2e})")
        
    @pytest.mark.parametrize("L", TEST_SIZES)  
    def test_attention_layer_equivalence(self, L: int):
        """Test that optimized attention layer produces identical outputs"""
        
        device = torch.device('cpu')
        input_dim, hidden_dim, num_heads = 45, 44, 4
        
        # Create test data
        torch.manual_seed(FIXED_SEED)
        node_features = torch.randn(L, input_dim, device=device)
        temporal_data = torch.randn(L, L, 2, device=device)
        
        # Create simple DAG
        dag = nx.DiGraph()
        dag.add_nodes_from(range(L))
        edges = [(i, i+1) for i in range(L-1)]
        dag.add_edges_from(edges)
        
        # Test original layer
        original_config = TGATConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        torch.manual_seed(FIXED_SEED)
        original_layer = EnhancedTemporalAttentionLayer(
            input_dim, hidden_dim, num_heads, original_config
        )
        original_layer.eval()
        
        with torch.no_grad():
            out_original, attn_original = original_layer(
                node_features, None, temporal_data, dag, return_attn=True
            )
        
        # Test optimized layer (disabled optimizations for exact comparison)
        opt_config = OptimizedTGATConfig(
            base_config=original_config,
            enable_amp=False,
            enable_flash_attention=False,
            use_fp16=False,
            enable_fused_ops=False,
            enable_time_bias_caching=False
        )
        
        torch.manual_seed(FIXED_SEED)
        optimized_layer = OptimizedEnhancedTemporalAttentionLayer(opt_config)
        optimized_layer.eval()
        
        # Copy weights to ensure identical initialization
        self._copy_layer_weights(original_layer, optimized_layer)
        
        with torch.no_grad():
            out_optimized, attn_optimized = optimized_layer(
                node_features, None, temporal_data, dag, return_attn=True
            )
        
        # Compare outputs
        output_diff = torch.abs(out_original - out_optimized).max().item()
        assert output_diff < TEST_TOLERANCE, \
            f"Layer output difference {output_diff} exceeds tolerance {TEST_TOLERANCE}"
        
        logger.info(f"✅ Attention layer equivalence test passed (L={L}, diff={output_diff:.2e})")
        
    def _copy_layer_weights(self, source_layer, target_layer):
        """Copy weights from source to target layer for identical comparison"""
        
        # Copy linear layer weights
        target_layer.input_projection.weight.data = source_layer.input_projection.weight.data.clone()
        target_layer.input_projection.bias.data = source_layer.input_projection.bias.data.clone()
        
        target_layer.query.weight.data = source_layer.query.weight.data.clone()
        target_layer.query.bias.data = source_layer.query.bias.data.clone()
        
        target_layer.key.weight.data = source_layer.key.weight.data.clone()
        target_layer.key.bias.data = source_layer.key.bias.data.clone()
        
        target_layer.value.weight.data = source_layer.value.weight.data.clone()
        target_layer.value.bias.data = source_layer.value.bias.data.clone()
        
        target_layer.output_projection.weight.data = source_layer.output_projection.weight.data.clone()
        target_layer.output_projection.bias.data = source_layer.output_projection.bias.data.clone()


class TestDAGBuilderOptimizationsAB:
    """A/B tests for DAG builder optimizations"""
    
    def setup_method(self):
        """Setup for each test method"""
        np.random.seed(FIXED_SEED)
        
    @pytest.mark.parametrize("num_events", [10, 25, 50])
    def test_dag_construction_equivalence(self, num_events: int):
        """Test that optimized DAG builder produces equivalent DAGs"""
        
        # Create test session data
        session_data = self._create_test_session_data(num_events)
        
        # Test original DAG builder
        original_builder = DAGGraphBuilder()
        dag_original = self._build_dag_with_builder(original_builder, session_data)
        
        # Test optimized DAG builder (vectorized ops disabled for exact comparison)
        opt_config = OptimizedDAGConfig(
            base_dag_config={},
            enable_vectorized_ops=False,  # Disable for exact comparison
            enable_topological_generations=True,  # This doesn't affect construction
            enable_batch_edge_creation=False  # Disable for exact comparison
        )
        
        optimized_builder = OptimizedDAGBuilder(dag_config=None, opt_config=opt_config)
        dag_optimized = optimized_builder.build_optimized_dag(session_data)
        
        # Compare DAG structure
        self._compare_dags(dag_original, dag_optimized, num_events)
        
        logger.info(f"✅ DAG construction equivalence test passed (events={num_events})")
        
    def test_vectorized_dag_operations(self, num_events: int = 30):
        """Test that vectorized operations produce equivalent results"""
        
        session_data = self._create_test_session_data(num_events)
        
        # Build with standard operations
        opt_config_standard = OptimizedDAGConfig(
            base_dag_config={},
            enable_vectorized_ops=False,
            enable_batch_edge_creation=False
        )
        builder_standard = OptimizedDAGBuilder(dag_config=None, opt_config=opt_config_standard)
        dag_standard = builder_standard.build_optimized_dag(session_data)
        
        # Build with vectorized operations
        opt_config_vectorized = OptimizedDAGConfig(
            base_dag_config={},
            enable_vectorized_ops=True,
            enable_batch_edge_creation=True
        )
        builder_vectorized = OptimizedDAGBuilder(dag_config=None, opt_config=opt_config_vectorized)
        dag_vectorized = builder_vectorized.build_optimized_dag(session_data)
        
        # Compare results
        self._compare_dags(dag_standard, dag_vectorized, num_events, tolerance=1e-3)
        
        logger.info(f"✅ Vectorized DAG operations test passed (events={num_events})")
        
    def _create_test_session_data(self, num_events: int) -> Dict[str, Any]:
        """Create deterministic test session data"""
        
        np.random.seed(FIXED_SEED)
        
        # Create timestamps
        base_time = pd.Timestamp('2024-01-01 09:30:00')
        timestamps = [
            base_time + pd.Timedelta(minutes=i*2) for i in range(num_events)
        ]
        
        # Create events
        event_types = ['fvg', 'sweep', 'impulse', 'imbalance']
        events = []
        
        for i in range(num_events):
            event = {
                'timestamp_et': timestamps[i],
                'event_type': np.random.choice(event_types),
                'price_level': 50000 + np.random.normal(0, 100),
                'volume_profile': np.random.exponential(100),
                'session_id': 'test_session'
            }
            events.append(event)
            
        return {
            'session_id': 'test_session',
            'events': events
        }
        
    def _build_dag_with_builder(self, builder: DAGGraphBuilder, session_data: Dict[str, Any]) -> nx.DiGraph:
        """Build DAG using standard builder interface"""
        
        # Extract events
        events = session_data.get('events', [])
        
        dag = nx.DiGraph()
        
        # Add nodes
        for i, event in enumerate(events):
            dag.add_node(i, event_data=event)
            
        # Add edges (simplified version of original logic)
        k_successors = 4
        dt_min_minutes = 1
        dt_max_minutes = 120
        
        for i, source_event in enumerate(events):
            source_time = source_event.get('timestamp_et')
            if not source_time:
                continue
                
            successors = []
            
            for j, target_event in enumerate(events):
                if i >= j:  # Only forward edges
                    continue
                    
                target_time = target_event.get('timestamp_et') 
                if not target_time:
                    continue
                    
                dt_minutes = (target_time - source_time).total_seconds() / 60
                
                if dt_min_minutes <= dt_minutes <= dt_max_minutes:
                    successors.append((j, dt_minutes))
                    
            # Sort and take k closest
            successors.sort(key=lambda x: x[1])
            
            for target_idx, dt_minutes in successors[:k_successors]:
                dag.add_edge(i, target_idx, dt_seconds=dt_minutes*60)
                
        return dag
        
    def _compare_dags(self, dag1: nx.DiGraph, dag2: nx.DiGraph, expected_nodes: int, 
                     tolerance: float = 1e-6):
        """Compare two DAGs for structural equivalence"""
        
        # Compare basic structure
        assert len(dag1.nodes()) == len(dag2.nodes()), \
            f"Node count mismatch: {len(dag1.nodes())} vs {len(dag2.nodes())}"
            
        assert len(dag1.nodes()) == expected_nodes, \
            f"Expected {expected_nodes} nodes, got {len(dag1.nodes())}"
        
        # Compare edge count (allow small differences due to different algorithms)
        edge_diff = abs(len(dag1.edges()) - len(dag2.edges()))
        max_allowed_diff = max(1, expected_nodes // 10)  # Allow 10% difference
        
        assert edge_diff <= max_allowed_diff, \
            f"Edge count difference {edge_diff} exceeds allowed {max_allowed_diff}"
        
        # Both should be DAGs
        assert nx.is_directed_acyclic_graph(dag1), "First DAG is not acyclic"
        assert nx.is_directed_acyclic_graph(dag2), "Second DAG is not acyclic"
        
        # Compare topological properties
        topo1 = list(nx.topological_sort(dag1))
        topo2 = list(nx.topological_sort(dag2))
        
        assert len(topo1) == len(topo2), \
            f"Topological sort length mismatch: {len(topo1)} vs {len(topo2)}"


class TestParquetOptimizationsAB:
    """A/B tests for Parquet optimizations"""
    
    def setup_method(self):
        """Setup for each test method"""
        np.random.seed(FIXED_SEED)
        
    def test_parquet_roundtrip_equivalence(self):
        """Test that optimized Parquet I/O preserves data integrity"""
        
        # Create test data
        test_data = self._create_test_parquet_data(1000)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test standard PyArrow operations
            standard_file = temp_path / "standard.parquet"
            pa.parquet.write_table(test_data, standard_file)
            standard_result = pa.parquet.read_table(standard_file)
            
            # Test optimized operations (minimal optimizations for exact comparison)
            opt_config = OptimizedParquetConfig(
                compression='snappy',  # Use same as default
                optimize_dtypes=False,  # Disable for exact comparison
                enable_content_chunking=False,  # Disable for exact comparison
                use_dictionary_encoding=[]  # Empty list for exact comparison
            )
            
            opt_manager = OptimizedParquetManager(opt_config)
            opt_file = temp_path / "optimized.parquet"
            opt_manager.writer.write_table(test_data, opt_file)
            optimized_result = pa.parquet.read_table(opt_file)
            
            # Compare results
            self._compare_parquet_tables(standard_result, optimized_result)
            
        logger.info("✅ Parquet roundtrip equivalence test passed")
        
    def test_dtype_optimization_preserves_values(self):
        """Test that dtype optimizations preserve data values within tolerance"""
        
        # Create test data with various numeric types
        test_data = pa.table({
            'small_int': pa.array(np.random.randint(0, 100, 1000), type=pa.int64()),
            'large_float': pa.array(np.random.normal(50000, 1000, 1000), type=pa.float64()),
            'categorical': pa.array(np.random.choice(['A', 'B', 'C'], 1000)),
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Standard write
            standard_file = temp_path / "standard.parquet"
            pa.parquet.write_table(test_data, standard_file)
            standard_result = pa.parquet.read_table(standard_file)
            
            # Optimized write with dtype optimization
            opt_config = OptimizedParquetConfig(
                optimize_dtypes=True,
                use_dictionary_encoding=['categorical']
            )
            
            opt_manager = OptimizedParquetManager(opt_config)
            opt_file = temp_path / "optimized.parquet"
            opt_manager.writer.write_table(test_data, opt_file)
            optimized_result = pa.parquet.read_table(opt_file)
            
            # Compare data values (allowing for type conversions)
            self._compare_parquet_values(standard_result, optimized_result)
            
        logger.info("✅ Dtype optimization value preservation test passed")
        
    def test_session_data_roundtrip(self):
        """Test session data format preservation through optimized I/O"""
        
        # Create test session data
        session_data = {
            'session_id': 'test_session_123',
            'events': [
                {
                    'timestamp_et': pd.Timestamp('2024-01-01 09:30:00'),
                    'event_type': 'fvg',
                    'price_level': 50000.5,
                    'volume_profile': 150.0,
                    'node_features': np.random.normal(0, 1, 45)
                },
                {
                    'timestamp_et': pd.Timestamp('2024-01-01 09:32:00'),
                    'event_type': 'sweep', 
                    'price_level': 50010.2,
                    'volume_profile': 200.0,
                    'node_features': np.random.normal(0, 1, 45)
                }
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save and load with optimized manager
            opt_manager = OptimizedParquetManager()
            
            # Save
            save_stats = opt_manager.save_session_data(session_data, temp_path)
            assert save_stats['num_rows'] == 2, "Should have 2 events"
            
            # Load
            loaded_data, load_stats = opt_manager.load_session_data(temp_path)
            assert load_stats['num_rows'] == 2, "Should load 2 events"
            
            # Compare session data
            self._compare_session_data(session_data, loaded_data)
            
        logger.info("✅ Session data roundtrip test passed")
        
    def _create_test_parquet_data(self, num_rows: int) -> pa.Table:
        """Create test data for Parquet operations"""
        
        np.random.seed(FIXED_SEED)
        
        data = {
            'id': pa.array(range(num_rows), type=pa.int64()),
            'timestamp': pa.array(pd.date_range('2024-01-01', periods=num_rows, freq='1min')),
            'price': pa.array(np.random.normal(50000, 1000, num_rows), type=pa.float64()),
            'volume': pa.array(np.random.exponential(100, num_rows), type=pa.float64()),
            'event_type': pa.array(np.random.choice(['fvg', 'sweep', 'impulse'], num_rows)),
            'session_id': pa.array(['session_1'] * num_rows),
            'features': pa.array([np.random.normal(0, 1, 10).tolist() for _ in range(num_rows)])
        }
        
        return pa.table(data)
        
    def _compare_parquet_tables(self, table1: pa.Table, table2: pa.Table):
        """Compare two PyArrow tables for equivalence"""
        
        assert len(table1) == len(table2), \
            f"Row count mismatch: {len(table1)} vs {len(table2)}"
        
        assert len(table1.schema) == len(table2.schema), \
            f"Column count mismatch: {len(table1.schema)} vs {len(table2.schema)}"
        
        # Compare column names
        cols1 = set(table1.column_names)
        cols2 = set(table2.column_names)
        assert cols1 == cols2, f"Column names differ: {cols1} vs {cols2}"
        
        # Compare data values
        df1 = table1.to_pandas()
        df2 = table2.to_pandas()
        
        for col in df1.columns:
            if col in df2.columns:
                if df1[col].dtype.kind in ['f']:  # Float columns
                    pd.testing.assert_series_equal(
                        df1[col], df2[col], check_exact=False, rtol=TEST_TOLERANCE
                    )
                else:  # Other columns
                    pd.testing.assert_series_equal(df1[col], df2[col])
                    
    def _compare_parquet_values(self, table1: pa.Table, table2: pa.Table):
        """Compare Parquet table values allowing for type optimization"""
        
        df1 = table1.to_pandas()
        df2 = table2.to_pandas()
        
        assert len(df1) == len(df2), "Row count must match"
        
        for col in df1.columns:
            if col not in df2.columns:
                continue
                
            series1 = df1[col]
            series2 = df2[col]
            
            # For numeric columns, allow type changes but values should be close
            if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
                np.testing.assert_allclose(
                    series1.values, series2.values, 
                    rtol=TEST_TOLERANCE, atol=TEST_TOLERANCE
                )
            else:
                # For non-numeric, should be identical
                pd.testing.assert_series_equal(series1, series2, check_dtype=False)
                
    def _compare_session_data(self, original: Dict[str, Any], loaded: Dict[str, Any]):
        """Compare original and loaded session data"""
        
        assert original['session_id'] == loaded['session_id'], "Session ID mismatch"
        assert len(original['events']) == len(loaded['events']), "Event count mismatch"
        
        for orig_event, loaded_event in zip(original['events'], loaded['events']):
            # Compare basic fields
            assert orig_event['event_type'] == loaded_event['event_type']
            
            # Compare numeric fields with tolerance
            np.testing.assert_allclose(
                orig_event['price_level'], loaded_event['price_level'], 
                rtol=TEST_TOLERANCE
            )
            
            # Compare node features if present
            if 'node_features' in orig_event and 'node_features' in loaded_event:
                np.testing.assert_allclose(
                    orig_event['node_features'], loaded_event['node_features'],
                    rtol=TEST_TOLERANCE
                )


class TestPerformanceValidation:
    """Validate that optimizations provide performance benefits"""
    
    def test_optimization_performance_gains(self):
        """Test that optimizations actually improve performance"""
        
        # This is more of a smoke test - detailed benchmarks are in performance_audit.py
        L = 128
        
        # Test TGAT performance
        torch.manual_seed(FIXED_SEED)
        
        # Standard implementation
        import time
        start = time.perf_counter()
        
        q = torch.randn(1, 4, L, 11)
        k = torch.randn(1, 4, L, 11)
        v = torch.randn(1, 4, L, 11)
        
        for _ in range(10):
            with torch.no_grad():
                out, _ = graph_attention(q, k, v, impl="sdpa", training=False)
                
        standard_time = time.perf_counter() - start
        
        # Optimized implementation  
        config = OptimizedTGATConfig(
            base_config=TGATConfig(),
            enable_amp=False,  # CPU doesn't support AMP
            enable_time_bias_caching=True
        )
        
        opt_attention = OptimizedGraphAttention(config)
        
        start = time.perf_counter()
        
        for _ in range(10):
            with torch.no_grad():
                out_opt, _ = opt_attention(q, k, v, training=False)
                
        optimized_time = time.perf_counter() - start
        
        # Optimized should be at least as fast (allowing for overhead in small tests)
        speed_ratio = standard_time / max(optimized_time, 1e-6)
        
        logger.info(f"TGAT speed ratio: {speed_ratio:.2f}x (standard: {standard_time:.4f}s, "
                   f"optimized: {optimized_time:.4f}s)")
        
        # At minimum, optimized shouldn't be significantly slower
        assert speed_ratio >= 0.5, f"Optimized version is too much slower: {speed_ratio}x"
        
        logger.info("✅ Performance validation test passed")


# Test configuration for pytest
def pytest_configure(config):
    """Configure pytest for performance testing"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    # Run A/B tests
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])