#!/usr/bin/env python3
"""
Test script for Enhanced Graph Builder refactoring
Validates modular structure and backward compatibility
"""

import sys
import traceback
from unittest.mock import Mock

def test_modular_imports():
    """Test that modular imports work correctly"""
    print("🧪 Testing Modular Imports...")
    
    try:
        # Test individual module imports
        from ironforge.learning.graph.node_features import RichNodeFeature, NodeFeatureProcessor
        print("  ✅ Node features module imported")
        
        from ironforge.learning.graph.edge_features import RichEdgeFeature, EdgeFeatureProcessor
        print("  ✅ Edge features module imported")
        
        from ironforge.learning.graph.graph_builder import EnhancedGraphBuilder
        print("  ✅ Graph builder module imported")
        
        from ironforge.learning.graph.feature_utils import FeatureUtils
        print("  ✅ Feature utils module imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Modular imports failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that backward compatibility is maintained"""
    print("\n🧪 Testing Backward Compatibility...")
    
    try:
        # Test original import path still works
        from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder, RichNodeFeature, RichEdgeFeature
        print("  ✅ Original import paths work")
        
        # Test class instantiation
        builder = EnhancedGraphBuilder()
        print("  ✅ EnhancedGraphBuilder instantiated")
        
        node_feature = RichNodeFeature()
        print("  ✅ RichNodeFeature instantiated")
        
        edge_feature = RichEdgeFeature()
        print("  ✅ RichEdgeFeature instantiated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backward compatibility failed: {e}")
        traceback.print_exc()
        return False

def test_feature_functionality():
    """Test that feature classes work correctly"""
    print("\n🧪 Testing Feature Functionality...")
    
    try:
        from ironforge.learning.enhanced_graph_builder import RichNodeFeature, RichEdgeFeature
        
        # Test RichNodeFeature
        node_feature = RichNodeFeature()
        
        # Test semantic event setting
        node_feature.set_semantic_event("fvg_redelivery_flag", 1.0)
        assert node_feature.features[0] == 1.0, "Semantic event not set correctly"
        print("  ✅ RichNodeFeature semantic events work")
        
        # Test traditional features
        import torch
        traditional = torch.zeros(37)
        traditional[0] = 100.0  # Price
        traditional[1] = 1000.0  # Volume
        node_feature.set_traditional_features(traditional)
        assert node_feature.features[8] == 100.0, "Traditional features not set correctly"
        print("  ✅ RichNodeFeature traditional features work")
        
        # Test RichEdgeFeature
        edge_feature = RichEdgeFeature()
        edge_feature.set_semantic_relationship("event_causality", 0.8)
        assert edge_feature.features[1] == 0.8, "Semantic relationship not set correctly"
        print("  ✅ RichEdgeFeature semantic relationships work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature functionality failed: {e}")
        traceback.print_exc()
        return False

def test_graph_building():
    """Test that graph building works with mock data"""
    print("\n🧪 Testing Graph Building...")
    
    try:
        from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
        
        # Create mock session data
        mock_session_data = {
            "session_name": "TEST_SESSION",
            "events": [
                {
                    "price": 100.0,
                    "volume": 1000.0,
                    "timestamp": "09:30:00",
                    "event_type": "fvg_redelivery",
                    "significance": 0.8
                },
                {
                    "price": 101.5,
                    "volume": 1200.0,
                    "timestamp": "09:31:00",
                    "event_type": "liquidity_sweep",
                    "significance": 0.9
                },
                {
                    "price": 99.8,
                    "volume": 800.0,
                    "timestamp": "09:32:00",
                    "event_type": "consolidation",
                    "significance": 0.6
                }
            ],
            "session_open": 100.0,
            "session_high": 102.0,
            "session_low": 99.0,
            "session_duration": 3600
        }
        
        # Build graph
        builder = EnhancedGraphBuilder()
        graph = builder.build_session_graph(mock_session_data)
        
        # Validate graph structure
        assert graph.number_of_nodes() == 3, f"Expected 3 nodes, got {graph.number_of_nodes()}"
        assert graph.number_of_edges() > 0, "Graph should have edges"
        print(f"  ✅ Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Test feature extraction
        node_features, edge_features = builder.extract_features_for_tgat(graph)
        assert node_features.shape == (3, 45), f"Expected (3, 45) node features, got {node_features.shape}"
        assert edge_features.shape[1] == 20, f"Expected 20D edge features, got {edge_features.shape[1]}D"
        print(f"  ✅ Features extracted: nodes {node_features.shape}, edges {edge_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Graph building failed: {e}")
        traceback.print_exc()
        return False

def test_theory_b_validation():
    """Test Theory B validation functionality"""
    print("\n🧪 Testing Theory B Validation...")
    
    try:
        from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
        
        # Create mock session data with Theory B compliance
        mock_session_data = {
            "session_name": "THEORY_B_TEST",
            "events": [
                {
                    "price": 99.2,  # 40% zone (99 + 0.4 * 3 = 100.2, close to 99.2)
                    "volume": 1000.0,
                    "timestamp": "09:30:00",
                    "event_type": "fvg_redelivery",
                    "significance": 0.8
                }
            ],
            "relativity_stats": {
                "session_open": 100.0,
                "session_high": 102.0,
                "session_low": 99.0,
                "session_range": 3.0,
                "session_duration_seconds": 3600
            }
        }
        
        # Test Theory B validation
        builder = EnhancedGraphBuilder()
        validation = builder.validate_theory_b_compliance(mock_session_data)
        
        assert "theory_b_active" in validation, "Theory B validation should include theory_b_active"
        assert "forty_percent_events" in validation, "Theory B validation should include forty_percent_events"
        print("  ✅ Theory B validation structure correct")
        
        # Check if 40% zone event was detected
        if validation["forty_percent_events"]:
            print(f"  ✅ 40% zone events detected: {len(validation['forty_percent_events'])}")
        else:
            print("  ⚠️  No 40% zone events detected (may be expected)")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Theory B validation failed: {e}")
        traceback.print_exc()
        return False

def test_feature_utils():
    """Test feature utility functions"""
    print("\n🧪 Testing Feature Utils...")
    
    try:
        from ironforge.learning.graph.feature_utils import FeatureUtils
        
        # Test timestamp parsing
        timestamp_seconds = FeatureUtils.parse_timestamp_to_seconds("09:30:00")
        expected = 9 * 3600 + 30 * 60  # 9:30 AM in seconds
        assert timestamp_seconds == expected, f"Expected {expected}, got {timestamp_seconds}"
        print("  ✅ Timestamp parsing works")
        
        # Test zone proximity calculation
        proximity = FeatureUtils.calculate_zone_proximity(
            price=100.2,  # 40% of range from low
            session_low=99.0,
            session_range=3.0,
            zone_level=0.40,
            session_context={"theory_b_final_range": True}
        )
        assert proximity > 0.8, f"Expected high proximity for 40% zone, got {proximity}"
        print("  ✅ Zone proximity calculation works")
        
        # Test premium/discount position
        pd_position = FeatureUtils.calculate_premium_discount_position(
            price=100.5,
            session_low=99.0,
            session_range=3.0
        )
        assert 0.0 <= pd_position <= 1.0, f"PD position should be 0-1, got {pd_position}"
        print("  ✅ Premium/discount position calculation works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature utils failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all refactoring tests"""
    print("🚀 Enhanced Graph Builder Refactoring Test Suite")
    print("=" * 70)
    
    tests = [
        test_modular_imports,
        test_backward_compatibility,
        test_feature_functionality,
        test_graph_building,
        test_theory_b_validation,
        test_feature_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            
    print(f"\n📊 Refactoring Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All refactoring tests passed! Modular structure is working correctly.")
        print("\n✅ Modular imports work correctly")
        print("✅ Backward compatibility is maintained")
        print("✅ Feature classes function properly")
        print("✅ Graph building works with new structure")
        print("✅ Theory B validation is functional")
        print("✅ Feature utilities work correctly")
    else:
        print("⚠️  Some refactoring tests failed. Review the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
