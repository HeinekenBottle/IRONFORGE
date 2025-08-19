"""
Test suite for Oracle Temporal Non-locality predictions
"""

import json
import tempfile
from pathlib import Path
import pytest
import torch
import pandas as pd
import networkx as nx

from ironforge.learning.tgat_discovery import IRONFORGEDiscovery, infer_shard_embeddings
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder, RichNodeFeature, RichEdgeFeature
from ironforge.sdk.app_config import OracleCfg


class MockLoaderCfg:
    """Mock loader configuration for testing"""
    def __init__(self, oracle_enabled=True, oracle_early_pct=0.20):
        self.oracle = oracle_enabled
        self.oracle_early_pct = oracle_early_pct


def create_mock_session_data(n_events=20, session_name="test_session"):
    """Create mock session data for testing"""
    events = []
    base_price = 23000.0
    
    for i in range(n_events):
        price = base_price + (i * 10) + torch.randn(1).item() * 5
        events.append({
            "timestamp": f"2025-08-19T14:{30+i:02d}:00",
            "price": price,
            "volume": 100 + torch.randint(0, 50, (1,)).item(),
            "event_type": "price_move",
            "confidence": 0.8 + torch.rand(1).item() * 0.2
        })
    
    return {
        "session_name": session_name,
        "events": events,
        "session_start": "2025-08-19T14:30:00",
        "session_end": f"2025-08-19T14:{49+n_events}:00"
    }


def create_mock_graph(n_nodes=10):
    """Create a mock NetworkX graph with proper 45D/20D features"""
    graph = nx.Graph()
    
    # Add nodes with 45D features
    for i in range(n_nodes):
        node_feature = RichNodeFeature()
        # Set some semantic flags
        if i % 3 == 0:
            node_feature.set_semantic_event("fvg_redelivery_flag", 1.0)
        if i % 5 == 0:
            node_feature.set_semantic_event("expansion_phase_flag", 1.0)
            
        # Set traditional features (37D)
        traditional_features = torch.randn(37) * 0.1
        node_feature.set_traditional_features(traditional_features)
        
        graph.add_node(i, feature=node_feature.features)
    
    # Add edges with 20D features
    for i in range(n_nodes - 1):
        edge_feature = RichEdgeFeature()
        edge_feature.set_semantic_relationship("semantic_event_link", 0.5)
        # Add traditional edge features
        edge_feature.features[3:] = torch.randn(17) * 0.1
        
        graph.add_edge(i, i+1, 
                      feature=edge_feature.features,
                      temporal_distance=1.0)
    
    return graph


def test_oracle_configuration():
    """Test oracle configuration classes"""
    oracle_cfg = OracleCfg()
    
    assert oracle_cfg.enabled == True
    assert oracle_cfg.early_pct == 0.20
    assert oracle_cfg.output_path == "oracle_predictions.parquet"
    
    # Test custom configuration
    custom_cfg = OracleCfg(enabled=False, early_pct=0.15, output_path="custom_oracle.parquet")
    assert custom_cfg.enabled == False
    assert custom_cfg.early_pct == 0.15
    assert custom_cfg.output_path == "custom_oracle.parquet"


def test_predict_session_range():
    """Test the core oracle prediction functionality"""
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()
    
    # Create mock graph
    graph = create_mock_graph(n_nodes=15)
    
    # Test oracle predictions
    with torch.no_grad():
        oracle_result = discovery_engine.predict_session_range(graph, early_batch_pct=0.20)
    
    # Validate result structure
    required_keys = ["pct_seen", "n_events", "pred_high", "pred_low", 
                     "center", "half_range", "confidence", "notes"]
    
    for key in required_keys:
        assert key in oracle_result, f"Missing key: {key}"
    
    # Validate result values
    assert 0.0 <= oracle_result["pct_seen"] <= 1.0
    assert oracle_result["n_events"] > 0
    assert oracle_result["pred_high"] > oracle_result["pred_low"]
    assert oracle_result["half_range"] >= 0
    assert 0.0 <= oracle_result["confidence"] <= 1.0
    assert isinstance(oracle_result["notes"], str)
    
    print(f"✅ Oracle prediction test passed:")
    print(f"   Predicted range: {oracle_result['pred_low']:.2f} - {oracle_result['pred_high']:.2f}")
    print(f"   Confidence: {oracle_result['confidence']:.3f}")
    print(f"   Events analyzed: {oracle_result['n_events']} ({oracle_result['pct_seen']:.1%})")


def test_attention_weight_exposure():
    """Test that attention weights can be optionally returned"""
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()
    
    graph = create_mock_graph(n_nodes=8)
    
    with torch.no_grad():
        # Test with attention weights enabled
        results_with_attn = discovery_engine.forward(graph, return_attn=True)
        assert results_with_attn["attention_weights"] is not None
        assert results_with_attn["attention_weights"].shape[0] == 8
        
        # Test with attention weights disabled
        results_no_attn = discovery_engine.forward(graph, return_attn=False)
        assert results_no_attn["attention_weights"] is None
    
    print("✅ Attention weight exposure test passed")


def test_empty_graph_handling():
    """Test oracle predictions with empty graphs"""
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()
    
    empty_graph = nx.Graph()
    
    with torch.no_grad():
        oracle_result = discovery_engine.predict_session_range(empty_graph)
    
    # Should return empty result structure
    assert oracle_result["pct_seen"] == 0.0
    assert oracle_result["n_events"] == 0
    assert oracle_result["confidence"] == 0.0
    assert "empty session" in oracle_result["notes"]
    
    print("✅ Empty graph handling test passed")


def test_parquet_schema_validation():
    """Test that oracle predictions produce valid parquet with correct schema"""
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()
    
    graph = create_mock_graph(n_nodes=12)
    
    with torch.no_grad():
        oracle_result = discovery_engine.predict_session_range(graph)
    
    # Add metadata fields
    oracle_result["run_dir"] = "/tmp/test_run"
    oracle_result["ts_generated"] = pd.Timestamp.utcnow()
    
    # Create DataFrame and validate schema
    df = pd.DataFrame([oracle_result])
    
    expected_columns = ["run_dir", "pct_seen", "n_events", "pred_high", "pred_low", 
                       "center", "half_range", "confidence", "ts_generated"]
    
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Test parquet serialization
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
        df.to_parquet(tmp_file.name, index=False)
        
        # Read back and validate
        df_read = pd.read_parquet(tmp_file.name)
        assert len(df_read) == 1
        assert df_read["n_events"].iloc[0] > 0
    
    print("✅ Parquet schema validation test passed")


def run_quick_test():
    """Quick validation test for oracle predictions"""
    print("🔮 Running Oracle Temporal Non-locality Quick Test")
    print("=" * 60)
    
    try:
        test_oracle_configuration()
        test_predict_session_range()
        test_attention_weight_exposure()
        test_empty_graph_handling()
        test_parquet_schema_validation()
        
        print("=" * 60)
        print("✅ All oracle prediction tests PASSED!")
        print("🔮 Oracle Temporal Non-locality system is ready for production")
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_quick_test()