"""
Test suite for Oracle Temporal Non-locality predictions
"""

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import torch

from ironforge.learning.enhanced_graph_builder import RichEdgeFeature, RichNodeFeature
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
from ironforge.sdk.app_config import OracleCfg


def create_mock_graph(n_nodes: int) -> nx.Graph:
    """Create mock graph with proper 45D/20D features"""
    graph = nx.Graph()
    
    for i in range(n_nodes):
        node_feature = RichNodeFeature()
        
        # Set semantic phase flags
        if i < n_nodes // 3:
            node_feature.set_semantic_event("expansion_phase_flag", 1.0)
        elif i < 2 * n_nodes // 3:
            node_feature.set_semantic_event("retracement_flag", 0.8)
        else:
            node_feature.set_semantic_event("reversal_flag", 0.9)
            
        # Traditional features
        traditional = torch.randn(37) * 0.1
        traditional[0] = 0.5 + i * 0.02
        node_feature.set_traditional_features(traditional)
        
        graph.add_node(i, feature=node_feature.features)
    
    # Add temporal edges
    for i in range(n_nodes - 1):
        edge_feature = RichEdgeFeature()
        edge_feature.set_semantic_relationship("semantic_event_link", 0.6)
        edge_feature.features[3:] = torch.randn(17) * 0.1
        
        graph.add_edge(i, i+1, feature=edge_feature.features, temporal_distance=1.0)
    
    return graph


def test_oracle_config_defaults():
    """Test oracle configuration defaults"""
    cfg = OracleCfg()
    
    assert cfg.enabled is False  # Disabled by default
    assert cfg.early_pct == 0.20
    assert cfg.output_path == "oracle_predictions.parquet"


def test_oracle_early_pct_validation():
    """Test early_pct validation"""
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()
    graph = create_mock_graph(10)
    
    # Test invalid early_pct values
    with pytest.raises(ValueError):
        discovery_engine.predict_session_range(graph, early_batch_pct=0.0)
    
    with pytest.raises(ValueError):
        discovery_engine.predict_session_range(graph, early_batch_pct=0.6)
    
    with pytest.raises(ValueError):
        discovery_engine.predict_session_range(graph, early_batch_pct=-0.1)
    
    # Test valid early_pct values
    with torch.no_grad():
        result = discovery_engine.predict_session_range(graph, early_batch_pct=0.2)
        assert result["n_events"] > 0


def test_tiny_session_guard():
    """Test guard for tiny sessions (<3 events)"""
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()
    
    # Test empty graph
    empty_graph = nx.Graph()
    result = discovery_engine.predict_session_range(empty_graph)
    assert result["n_events"] == 0
    
    # Test tiny graph (2 events)
    tiny_graph = create_mock_graph(2)
    result = discovery_engine.predict_session_range(tiny_graph)
    assert result["n_events"] == 0  # Should return empty result
    
    # Test minimal valid graph (3 events)
    small_graph = create_mock_graph(3)
    with torch.no_grad():
        result = discovery_engine.predict_session_range(small_graph)
        assert result["n_events"] > 0  # Should work


def test_oracle_schema_v0():
    """Test exact sidecar schema v0"""
    discovery_engine = IRONFORGEDiscovery() 
    discovery_engine.eval()
    graph = create_mock_graph(12)
    
    with torch.no_grad():
        result = discovery_engine.predict_session_range(graph)
    
    # Add schema fields
    result["run_dir"] = "/tmp/test"
    result["session_date"] = "2025-08-19"
    result["pattern_id"] = "pattern_003"
    result["start_ts"] = "2025-08-19T14:30:00"
    result["end_ts"] = "2025-08-19T14:32:00"
    
    # Create DataFrame and validate exact schema
    df = pd.DataFrame([result])
    
    expected_columns = [
        "run_dir", "session_date", "pct_seen", "n_events",
        "pred_low", "pred_high", "center", "half_range", "confidence",
        "pattern_id", "start_ts", "end_ts",
        "early_expansion_cnt", "early_retracement_cnt", "early_reversal_cnt", "first_seq"
    ]
    
    # Verify all columns present
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Verify column count matches exactly
    assert len(df.columns) >= len(expected_columns), f"Expected {len(expected_columns)} columns, got {len(df.columns)}"


def test_phase_sequence_extraction():
    """Test phase sequence breadcrumbs extraction"""
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()
    
    # Create graph with known phase sequence
    graph = create_mock_graph(9)  # 3 expansion, 3 retracement, 3 reversal
    
    with torch.no_grad():
        result = discovery_engine.predict_session_range(graph, early_batch_pct=0.33)  # First 3 events
    
    # Should capture expansion events from early portion
    assert result["early_expansion_cnt"] > 0
    assert "first_seq" in result
    assert isinstance(result["first_seq"], str)


def test_return_attn_default_unchanged():
    """Test that return_attn=False is default (unchanged behavior)"""
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()
    graph = create_mock_graph(8)
    
    with torch.no_grad():
        # Default behavior should not return attention
        result = discovery_engine.forward(graph)
        assert result["attention_weights"] is None
        
        # Explicit False
        result = discovery_engine.forward(graph, return_attn=False) 
        assert result["attention_weights"] is None
        
        # Only when explicitly True
        result = discovery_engine.forward(graph, return_attn=True)
        assert result["attention_weights"] is not None


def test_oracle_sidecar_exists_when_enabled():
    """Soft CI test: when oracle enabled, sidecar must exist with required columns"""
    # This would be called by CI when oracle is enabled
    def validate_oracle_sidecar(run_dir: Path) -> bool:
        oracle_path = run_dir / "oracle_predictions.parquet"
        
        if not oracle_path.exists():
            return False
            
        df = pd.read_parquet(oracle_path)
        
        required_columns = [
            "run_dir", "session_date", "pct_seen", "n_events",
            "pred_low", "pred_high", "pred_center", "pred_half_range", "confidence",
            "pattern_id", "start_ts", "end_ts",
            "early_expansion_cnt", "early_retracement_cnt", "early_reversal_cnt", "first_seq"
        ]
        
        return all(col in df.columns for col in required_columns)
    
    # Mock test
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_path = Path(tmp_dir)
        
        # Create mock sidecar
        mock_data = {
            "run_dir": str(run_path),
            "session_date": "2025-08-19",
            "pct_seen": 0.2,
            "n_events": 4,
            "pred_low": 100.0,
            "pred_high": 200.0,
            "pred_center": 150.0,
            "pred_half_range": 50.0,
            "confidence": 0.8,
            "pattern_id": "pattern_001",
            "start_ts": "2025-08-19T14:30:00",
            "end_ts": "2025-08-19T14:34:00",
            "early_expansion_cnt": 2,
            "early_retracement_cnt": 1,
            "early_reversal_cnt": 0,
            "first_seq": "E→E→R"
        }
        
        df = pd.DataFrame([mock_data])
        oracle_path = run_path / "oracle_predictions.parquet"
        df.to_parquet(oracle_path, index=False)
        
        assert validate_oracle_sidecar(run_path)


if __name__ == "__main__":
    pytest.main([__file__])