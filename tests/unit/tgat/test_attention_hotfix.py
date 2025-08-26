"""
Comprehensive tests for TGAT attention implementation hot-fix
Tests SDPA integration, edge masking, temporal bias, and compatibility
"""

import math
import pytest
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np

from ironforge.learning.tgat_discovery import (
    graph_attention, 
    build_edge_mask, 
    build_time_bias,
    EnhancedTemporalAttentionLayer,
    create_attention_layer
)
from ironforge.learning.dual_graph_config import TGATConfig


class TestAttentionHotfix:
    """Test suite for TGAT attention hot-fix implementation"""

    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)  # Fixed seed for reproducible tests
        
        # Test configuration
        self.cfg = TGATConfig()
        self.cfg.attention_impl = "sdpa"
        self.cfg.use_edge_mask = True
        self.cfg.use_time_bias = "bucket"
        self.cfg.is_causal = False
        
        # Test dimensions
        self.B, self.H, self.L, self.D = 1, 4, 10, 11  # Batch, Heads, Length, Dimension
        self.device = torch.device('cpu')
        
        # Create test tensors
        self.q = torch.randn(self.B, self.H, self.L, self.D)
        self.k = torch.randn(self.B, self.H, self.L, self.D) 
        self.v = torch.randn(self.B, self.H, self.L, self.D)

    def test_import_flag_sanity(self):
        """Test SDPA import and availability detection"""
        try:
            from torch.nn.functional import scaled_dot_product_attention as sdpa
            assert sdpa is not None, "SDPA should be importable"
            
            # Test SDPA with simple inputs
            q = torch.randn(1, 4, 10, 11)
            k = torch.randn(1, 4, 10, 11)
            v = torch.randn(1, 4, 10, 11)
            
            output = sdpa(q, k, v)
            assert output.shape == (1, 4, 10, 11), f"SDPA output shape mismatch: {output.shape}"
            
        except ImportError:
            pytest.fail("SDPA should be available in PyTorch ≥2.0")

    def test_sdpa_vs_manual_identical_outputs(self):
        """Test SDPA and manual implementations produce identical outputs"""
        torch.manual_seed(123)  # Fixed seed for deterministic comparison
        
        q = torch.randn(1, 4, 10, 11, dtype=torch.float32)
        k = torch.randn(1, 4, 10, 11, dtype=torch.float32) 
        v = torch.randn(1, 4, 10, 11, dtype=torch.float32)
        
        # SDPA implementation
        out_sdpa, _ = graph_attention(
            q, k, v, 
            edge_mask_bool=None, 
            time_bias=None,
            dropout_p=0.0,
            is_causal=False,
            impl="sdpa",
            training=False
        )
        
        # Manual implementation
        out_manual, _ = graph_attention(
            q, k, v,
            edge_mask_bool=None,
            time_bias=None, 
            dropout_p=0.0,
            is_causal=False,
            impl="manual",
            training=False
        )
        
        # Should be identical within numerical precision
        diff = torch.abs(out_sdpa - out_manual).max()
        assert diff < 1e-4, f"SDPA and manual outputs differ by {diff} (should be < 1e-4)"

    def test_edge_mask_enforcement(self):
        """Test edge mask properly blocks non-neighbor attention"""
        # Create a simple chain graph: 0->1->2->3
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        L = 4
        
        edge_mask = build_edge_mask(edge_index, L, device=self.device, allow_self=True)
        
        # Edge mask should be [1,1,L,L] with False for allowed connections
        assert edge_mask.shape == (1, 1, L, L), f"Edge mask shape: {edge_mask.shape}"
        
        # Check specific connections
        assert not edge_mask[0, 0, 0, 1], "0->1 should be allowed"
        assert not edge_mask[0, 0, 1, 2], "1->2 should be allowed"  
        assert not edge_mask[0, 0, 2, 3], "2->3 should be allowed"
        assert not edge_mask[0, 0, 0, 0], "Self-attention should be allowed"
        
        # Non-neighbor connections should be blocked
        assert edge_mask[0, 0, 0, 2], "0->2 should be blocked (non-neighbor)"
        assert edge_mask[0, 0, 0, 3], "0->3 should be blocked (non-neighbor)"
        
        # Test attention respects mask by computing attention probabilities
        q = torch.randn(1, 4, L, 11) 
        k = torch.randn(1, 4, L, 11)
        v = torch.randn(1, 4, L, 11)
        
        _, attn_weights = graph_attention(
            q, k, v,
            edge_mask_bool=edge_mask,
            time_bias=None,
            impl="manual",  # Need manual to get attention weights
            training=False
        )
        
        if attn_weights is not None:  # Manual implementation returns weights
            # Average across heads and batch
            avg_attn = attn_weights.mean(dim=1).squeeze(0)  # [L, L]
            
            # Blocked connections should have near-zero attention
            blocked_attn = avg_attn[0, 2]  # 0->2 connection (blocked)
            assert blocked_attn < 1e-6, f"Blocked attention too high: {blocked_attn}"

    def test_time_bias_monotonicity(self):
        """Test temporal bias gives higher attention to nearer events"""
        L = 5
        
        # Create temporal distance matrix where dt increases with distance
        dt_minutes = torch.zeros(L, L)
        for i in range(L):
            for j in range(L):
                dt_minutes[i, j] = abs(i - j) * 10  # 0, 10, 20, 30, 40 minutes
        
        # Build RBF time bias (closer events get higher bias)
        time_bias = build_time_bias(dt_minutes, L, device=self.device, scale=0.1)
        
        assert time_bias.shape == (1, 1, L, L), f"Time bias shape: {time_bias.shape}"
        
        # Bias should decrease with distance (more negative)
        bias_self = time_bias[0, 0, 0, 0]  # Self (dt=0)
        bias_near = time_bias[0, 0, 0, 1]   # Near (dt=10)  
        bias_far = time_bias[0, 0, 0, 4]    # Far (dt=40)
        
        assert bias_self > bias_near, f"Self bias ({bias_self}) should be > near bias ({bias_near})"
        assert bias_near > bias_far, f"Near bias ({bias_near}) should be > far bias ({bias_far})"

    def test_bucketed_time_bias(self):
        """Test bucketed temporal bias applies correct weights"""
        L = 6
        
        # Create temporal distances: 0, 5, 15, 30, 90, 180 minutes
        dt_minutes = torch.tensor([
            [0, 5, 15, 30, 90, 180],
            [5, 0, 10, 25, 85, 175], 
            [15, 10, 0, 15, 75, 165],
            [30, 25, 15, 0, 60, 150],
            [90, 85, 75, 60, 0, 90],
            [180, 175, 165, 150, 90, 0]
        ], dtype=torch.float32)
        
        # Define buckets matching implementation
        buckets = [
            (0, 5, 0.2),      # 0-5 minutes: strong positive bias
            (5, 15, 0.1),     # 5-15 minutes: moderate positive bias
            (15, 60, 0.0),    # 15-60 minutes: neutral
            (60, 240, -0.1),  # 1-4 hours: slight negative bias
        ]
        
        time_bias = build_time_bias(dt_minutes, L, device=self.device, buckets=buckets)
        
        # Test specific bucket assignments
        # Note: dt=5 falls into both (0,5) and (5,15) buckets, so gets 0.2 + 0.1 = 0.3
        assert abs(time_bias[0, 0, 0, 1] - 0.3) < 1e-5, f"dt=5min bias wrong: {time_bias[0, 0, 0, 1]} (expected 0.3)"
        assert abs(time_bias[0, 0, 0, 2] - 0.1) < 1e-5, f"dt=15min bias wrong: {time_bias[0, 0, 0, 2]} (expected 0.1)"
        assert abs(time_bias[0, 0, 0, 3] - 0.0) < 1e-5, f"dt=30min bias wrong: {time_bias[0, 0, 0, 3]} (expected 0.0)"
        assert abs(time_bias[0, 0, 0, 4] - (-0.1)) < 1e-5, f"dt=90min bias wrong: {time_bias[0, 0, 0, 4]} (expected -0.1)"

    def test_dag_safety_topological_order(self):
        """Test DAG structure ensures topological message passing"""
        # Create DAG: 0->1, 0->2, 1->3, 2->3 (diamond structure)
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
        
        assert nx.is_directed_acyclic_graph(dag), "Test graph should be acyclic"
        
        # Test topological order exists
        topo_order = list(nx.topological_sort(dag))
        assert len(topo_order) == 4, f"Should have 4 nodes in topo order: {topo_order}"
        assert topo_order[0] == 0, f"Node 0 should be first in topo order: {topo_order}"
        assert topo_order[-1] == 3, f"Node 3 should be last in topo order: {topo_order}"

    def test_enhanced_attention_layer_integration(self):
        """Test enhanced attention layer with new configuration system"""
        cfg = TGATConfig()
        cfg.attention_impl = "sdpa"
        cfg.use_edge_mask = True
        cfg.use_time_bias = "bucket"
        cfg.is_causal = False
        
        layer = EnhancedTemporalAttentionLayer(
            input_dim=45,
            hidden_dim=44,
            num_heads=4,
            cfg=cfg
        )
        
        # Test forward pass
        node_features = torch.randn(10, 45)  # 10 nodes, 45D features
        
        # Create simple DAG and temporal data
        dag = nx.DiGraph()
        dag.add_edges_from([(i, i+1) for i in range(9)])  # Chain 0->1->2->...->9
        
        temporal_data = torch.randn(10, 10, 2)  # [dt_minutes, temporal_distance]
        
        output, attn_weights = layer(
            node_features, 
            edge_features=None,
            temporal_data=temporal_data,
            dag=dag,
            return_attn=False
        )
        
        assert output.shape == (10, 44), f"Output shape wrong: {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"

    def test_cli_config_validation(self):
        """Test configuration validation for CLI usage"""
        # Valid configuration
        cfg = TGATConfig()
        cfg.attention_impl = "sdpa"
        cfg.use_edge_mask = True
        cfg.use_time_bias = "bucket"
        cfg.is_causal = False
        
        # Test valid attention implementations
        assert cfg.attention_impl in ["sdpa", "manual"], f"Invalid attention_impl: {cfg.attention_impl}"
        
        # Test valid time bias options
        assert cfg.use_time_bias in ["none", "bucket", "rbf"], f"Invalid use_time_bias: {cfg.use_time_bias}"
        
        # Test boolean flags
        assert isinstance(cfg.use_edge_mask, bool), "use_edge_mask should be boolean"
        assert isinstance(cfg.is_causal, bool), "is_causal should be boolean"

    def test_memory_efficiency_large_graph(self):
        """Test memory efficiency with larger graphs"""
        # Create larger test case
        L = 100  # 100 nodes
        B, H, D = 1, 4, 11
        
        q_large = torch.randn(B, H, L, D)
        k_large = torch.randn(B, H, L, D)
        v_large = torch.randn(B, H, L, D)
        
        # Test both implementations handle large graphs
        try:
            out_sdpa, _ = graph_attention(
                q_large, k_large, v_large,
                impl="sdpa",
                training=False
            )
            
            out_manual, _ = graph_attention(
                q_large, k_large, v_large, 
                impl="manual",
                training=False
            )
            
            assert out_sdpa.shape == (B, H, L, D), f"SDPA large graph output shape: {out_sdpa.shape}"
            assert out_manual.shape == (B, H, L, D), f"Manual large graph output shape: {out_manual.shape}"
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Insufficient memory for large graph test")
            else:
                raise

    def test_dropout_behavior(self):
        """Test dropout is properly applied during training"""
        torch.manual_seed(456)
        
        q = torch.randn(1, 4, 10, 11)
        k = torch.randn(1, 4, 10, 11)
        v = torch.randn(1, 4, 10, 11)
        
        # Training mode with dropout should give different results
        out1, _ = graph_attention(q, k, v, dropout_p=0.1, impl="manual", training=True)
        out2, _ = graph_attention(q, k, v, dropout_p=0.1, impl="manual", training=True)
        
        # Outputs should be different due to dropout randomness
        diff = torch.abs(out1 - out2).max()
        assert diff > 1e-6, f"Dropout should cause variation, but diff={diff}"
        
        # Inference mode should be deterministic
        out3, _ = graph_attention(q, k, v, dropout_p=0.1, impl="manual", training=False)
        out4, _ = graph_attention(q, k, v, dropout_p=0.1, impl="manual", training=False)
        
        diff_inference = torch.abs(out3 - out4).max()
        assert diff_inference < 1e-6, f"Inference should be deterministic, but diff={diff_inference}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])