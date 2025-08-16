#!/usr/bin/env python3
"""
Unit Test for HTF Scale Edge Validation
"""

import json
import unittest
import torch
from learning.enhanced_graph_builder import EnhancedGraphBuilder

class TestHTFScaleEdges(unittest.TestCase):
    """Test suite for HTF scale edge creation and validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.builder = EnhancedGraphBuilder()
        
        # Create synthetic HTF session data
        self.synthetic_htf_data = {
            "session_metadata": {
                "session_type": "test_session",
                "session_date": "2025-08-12",
                "session_start": "13:30:00",
                "session_end": "15:30:00",
                "session_duration": 120
            },
            # Include price_movements for validation
            "price_movements": [
                {
                    "timestamp": "13:30:00",
                    "price_level": 23000.0,
                    "event_type": "session_level",
                    "contamination_risk": False
                },
                {
                    "timestamp": "13:31:00",
                    "price_level": 23010.0,
                    "event_type": "session_level",
                    "contamination_risk": False
                }
            ],
            "pythonnodes": {
                "1m": [
                    {
                        "id": 0,
                        "timestamp": "13:30:00",
                        "price_level": 23000.0,
                        "open": 23000.0,
                        "high": 23005.0,
                        "low": 22995.0,
                        "close": 23002.0,
                        "event_type": "session_level",
                        "meta": {"coverage": 1, "source": "1m_movement"}
                    },
                    {
                        "id": 1,
                        "timestamp": "13:31:00",
                        "price_level": 23010.0,
                        "open": 23002.0,
                        "high": 23015.0,
                        "low": 23000.0,
                        "close": 23010.0,
                        "event_type": "session_level",
                        "meta": {"coverage": 1, "source": "1m_movement"}
                    }
                ],
                "5m": [
                    {
                        "id": 0,
                        "timestamp": "13:30:00",
                        "open": 23000.0,
                        "high": 23015.0,
                        "low": 22995.0,
                        "close": 23010.0,
                        "timeframe": "5m",
                        "pd_array": {"type": "swing_high", "level": 23015.0},
                        "liquidity_sweep": True,
                        "fpfvg": {"gap": True, "level": 23005.0},
                        "meta": {"coverage": 5, "period_minutes": 5}
                    }
                ]
            },
            "htf_cross_map": {
                "1m_to_5m": {
                    "0": 0,  # 1m node 0 maps to 5m node 0
                    "1": 0   # 1m node 1 maps to 5m node 0
                }
            }
        }
        
    def test_htf_scale_edge_creation(self):
        """Test that HTF scale edges are created correctly"""
        
        # Build graph with synthetic HTF data
        graph = self.builder.build_rich_graph(self.synthetic_htf_data)
        
        # Verify nodes were created
        self.assertEqual(graph['metadata']['total_nodes'], 3, "Should have 3 total nodes (2 1m + 1 5m)")
        self.assertEqual(len(graph['nodes']['1m']), 2, "Should have 2 1m nodes")
        self.assertEqual(len(graph['nodes']['5m']), 1, "Should have 1 5m node")
        
        # Verify scale edges exist
        scale_edges = graph['edges']['scale']
        htf_scale_edges = [e for e in scale_edges if 'tf_source' in e and 'tf_target' in e]
        
        self.assertGreater(len(htf_scale_edges), 0, "Should have HTF scale edges")
        
        # Verify scale edge properties
        for edge in htf_scale_edges[:2]:  # Check first 2 scale edges
            self.assertIn('tf_source', edge, "Scale edge should have tf_source")
            self.assertIn('tf_target', edge, "Scale edge should have tf_target") 
            self.assertIn('coverage', edge, "Scale edge should have coverage")
            self.assertIn('parent_metadata', edge, "Scale edge should have parent metadata")
            
            # Check metadata structure
            parent_meta = edge['parent_metadata']
            self.assertIn('pd_array', parent_meta, "Parent metadata should include pd_array")
            self.assertIn('fpfvg', parent_meta, "Parent metadata should include fpfvg")
            self.assertIn('liquidity_sweep', parent_meta, "Parent metadata should include liquidity_sweep")
            
        print(f"✅ Created {len(htf_scale_edges)} HTF scale edges with proper metadata")
        
    def test_scale_edge_mappings(self):
        """Test that scale edges follow the htf_cross_map correctly"""
        
        graph = self.builder.build_rich_graph(self.synthetic_htf_data)
        scale_edges = [e for e in graph['edges']['scale'] if 'tf_source' in e]
        
        # Should have 2 scale edges (1m_node_0 -> 5m_node_0, 1m_node_1 -> 5m_node_0)
        mappings_1m_to_5m = [e for e in scale_edges if e['tf_source'] == '1m' and e['tf_target'] == '5m']
        
        self.assertEqual(len(mappings_1m_to_5m), 2, "Should have 2 1m->5m scale edges per cross_map")
        
        # Verify edge metadata
        for edge in mappings_1m_to_5m:
            self.assertEqual(edge['coverage'], 5, "5m node should have coverage=5")
            
        print(f"✅ Scale edge mappings follow htf_cross_map correctly")
        
    def test_scale_edge_features(self):
        """Test that scale edges have proper 17D features"""
        
        graph = self.builder.build_rich_graph(self.synthetic_htf_data)
        
        # Get scale edges and their features
        scale_edges = [e for e in graph['edges']['scale'] if 'tf_source' in e]
        self.assertGreater(len(scale_edges), 0, "Should have scale edges")
        
        # Check edge features
        rich_edge_features = graph['rich_edge_features']
        
        for edge in scale_edges:
            feature_idx = edge['feature_idx']
            self.assertLess(feature_idx, len(rich_edge_features), "Feature index should be valid")
            
            edge_feature = rich_edge_features[feature_idx]
            feature_tensor = edge_feature.to_tensor()
            
            self.assertEqual(feature_tensor.shape[0], 17, "Edge features should be 17-dimensional")
            self.assertTrue(torch.is_tensor(feature_tensor), "Should be a valid tensor")
            
        print(f"✅ Scale edges have proper 17D features")
        
    def test_tgat_compatibility(self):
        """Test that HTF graphs maintain TGAT compatibility"""
        
        graph = self.builder.build_rich_graph(self.synthetic_htf_data)
        
        # Convert to TGAT format
        X, edge_index, edge_times, metadata, edge_attr = self.builder.to_tgat_format(graph)
        
        # Verify tensor shapes
        self.assertEqual(X.shape[1], 27, "Node features should be 27D")
        self.assertEqual(edge_attr.shape[1], 17, "Edge features should be 17D")
        self.assertEqual(edge_index.shape[0], 2, "Edge index should have 2 rows")
        self.assertEqual(edge_index.shape[1], edge_times.shape[0], "Edge index and times should match")
        self.assertEqual(edge_index.shape[1], edge_attr.shape[0], "Edge index and features should match")
        
        # Verify metadata
        self.assertIn('edge_feature_dimensions', metadata, "Metadata should include edge feature dims")
        self.assertEqual(metadata['edge_feature_dimensions'], 17, "Should report 17D edge features")
        self.assertIn('total_edges', metadata, "Metadata should include total edges")
        
        print(f"✅ TGAT compatibility maintained: {X.shape} nodes, {edge_index.shape[1]} edges")
        
    def test_fallback_behavior(self):
        """Test graceful fallback when HTF data is missing"""
        
        # Create session data without HTF components
        regular_session = {
            "session_metadata": {
                "session_type": "test_session",
                "session_date": "2025-08-12", 
                "session_start": "13:30:00",
                "session_duration": 60
            },
            "price_movements": [
                {
                    "timestamp": "13:30:00",
                    "price_level": 23000.0,
                    "event_type": "session_level",
                    "contamination_risk": False
                },
                {
                    "timestamp": "13:31:00",
                    "price_level": 23005.0,
                    "event_type": "session_level",
                    "contamination_risk": False
                }
            ]
        }
        
        # Should fall back to regular processing
        graph = self.builder.build_rich_graph(regular_session)
        
        # Should have nodes only in 1m and possibly 15m/1h timeframes (no HTF specific ones)
        timeframe_counts = graph['metadata']['timeframe_counts']
        total_htf_nodes = timeframe_counts.get('5m', 0) + timeframe_counts.get('D', 0) + timeframe_counts.get('W', 0)
        self.assertEqual(total_htf_nodes, 0, "Should have no HTF nodes in fallback mode")
        
        # Should not have HTF scale edges
        scale_edges = graph['edges']['scale']
        htf_scale_edges = [e for e in scale_edges if 'tf_source' in e]
        self.assertEqual(len(htf_scale_edges), 0, "Should have no HTF scale edges in fallback mode")
        
        print(f"✅ Graceful fallback to 1m-only mode works correctly")

def run_scale_edge_tests():
    """Run all scale edge tests"""
    
    print("🧪 Running HTF Scale Edge Unit Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHTFScaleEdges)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 50)
    if result.wasSuccessful():
        print("🎉 All HTF Scale Edge Tests PASSED")
    else:
        print("❌ Some HTF Scale Edge Tests FAILED")
        
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_scale_edge_tests()
    exit(0 if success else 1)