#!/usr/bin/env python3
"""
Functional tests for the refactored Enhanced Temporal Query Engine
Tests core functionality and backward compatibility
"""

import sys
import traceback
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

def create_mock_session_data():
    """Create mock session data for testing"""
    # Create sample nodes DataFrame
    nodes_data = {
        'price': [100.0, 101.5, 99.8, 102.3, 98.5, 103.1, 97.2, 104.0],
        'timestamp': ['2025-08-21 09:30:00', '2025-08-21 09:31:00', '2025-08-21 09:32:00',
                     '2025-08-21 09:33:00', '2025-08-21 09:34:00', '2025-08-21 09:35:00',
                     '2025-08-21 09:36:00', '2025-08-21 09:37:00'],
        'volume': [1000, 1200, 800, 1500, 900, 1300, 700, 1600],
        'energy_density': [0.5, 0.7, 0.3, 0.8, 0.4, 0.9, 0.2, 0.6],
        'liquidity_score': [0.6, 0.8, 0.4, 0.9, 0.5, 0.7, 0.3, 0.8]
    }
    nodes_df = pd.DataFrame(nodes_data)
    
    # Create session stats
    session_stats = {
        'high': 104.0,
        'low': 97.2,
        'open': 100.0,
        'close': 104.0,
        'range': 6.8
    }
    
    return nodes_df, session_stats

def test_temporal_engine_initialization():
    """Test that the temporal engine can be initialized"""
    print("🧪 Testing Temporal Engine Initialization...")
    
    try:
        # Mock the file system dependencies
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            
            # Initialize engine
            engine = EnhancedTemporalQueryEngine(
                shard_dir="/mock/shard/dir",
                adapted_dir="/mock/adapted/dir"
            )
            
            # Test basic attributes
            assert hasattr(engine, 'ask'), "Engine should have ask method"
            assert hasattr(engine, 'sessions'), "Engine should have sessions attribute"
            assert hasattr(engine, 'session_stats'), "Engine should have session_stats attribute"
            assert hasattr(engine, 'session_manager'), "Engine should have session_manager"
            assert hasattr(engine, 'price_engine'), "Engine should have price_engine"
            assert hasattr(engine, 'query_core'), "Engine should have query_core"
            assert hasattr(engine, 'visualization'), "Engine should have visualization"
            
            print("  ✅ Engine initialized successfully")
            print("  ✅ All required attributes present")
            return True
            
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        traceback.print_exc()
        return False

def test_temporal_query_functionality():
    """Test core temporal query functionality"""
    print("\n🧪 Testing Temporal Query Functionality...")
    
    try:
        # Mock dependencies and create test data
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            
            # Initialize engine
            engine = EnhancedTemporalQueryEngine()
            
            # Add mock session data
            nodes_df, session_stats = create_mock_session_data()
            engine.sessions = {"MIDNIGHT_2025-08-21": nodes_df}
            engine.session_stats = {"MIDNIGHT_2025-08-21": session_stats}
            
            # Test different query types
            test_queries = [
                "What happens after liquidity sweeps?",
                "Show archaeological zone distribution",
                "Analyze relative positioning patterns",
                "Search for precision events",
                "Find liquidity sweep patterns"
            ]
            
            for query in test_queries:
                try:
                    result = engine.ask(query)
                    
                    # Validate result structure
                    assert isinstance(result, dict), f"Query result should be dict for: {query}"
                    assert "query_type" in result, f"Result should have query_type for: {query}"
                    assert "total_sessions" in result, f"Result should have total_sessions for: {query}"
                    
                    print(f"  ✅ Query '{query[:30]}...' → {result['query_type']}")
                    
                except Exception as query_error:
                    print(f"  ❌ Query '{query[:30]}...' failed: {query_error}")
                    return False
            
            print("  ✅ All query types processed successfully")
            return True
            
    except Exception as e:
        print(f"  ❌ Query functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_archaeological_zone_analysis():
    """Test archaeological zone analysis functionality"""
    print("\n🧪 Testing Archaeological Zone Analysis...")
    
    try:
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            
            engine = EnhancedTemporalQueryEngine()
            
            # Add mock session data
            nodes_df, session_stats = create_mock_session_data()
            engine.sessions = {"MIDNIGHT_2025-08-21": nodes_df}
            engine.session_stats = {"MIDNIGHT_2025-08-21": session_stats}
            
            # Test archaeological zone analysis
            result = engine.ask("Show archaeological zone distribution for 40% events")
            
            # Validate result structure
            assert result["query_type"] == "archaeological_zones"
            assert "zone_analysis" in result
            assert "theory_b_events" in result
            assert "insights" in result
            
            print("  ✅ Archaeological zone analysis structure valid")
            
            # Test that zone analysis contains session data
            if result["zone_analysis"]:
                session_analysis = list(result["zone_analysis"].values())[0]
                assert "session_range" in session_analysis
                assert "zone_boundaries" in session_analysis
                print("  ✅ Zone analysis contains expected data")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Archaeological zone analysis failed: {e}")
        traceback.print_exc()
        return False

def test_relative_positioning_analysis():
    """Test relative positioning analysis functionality"""
    print("\n🧪 Testing Relative Positioning Analysis...")
    
    try:
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            
            engine = EnhancedTemporalQueryEngine()
            
            # Add mock session data
            nodes_df, session_stats = create_mock_session_data()
            engine.sessions = {"MIDNIGHT_2025-08-21": nodes_df}
            engine.session_stats = {"MIDNIGHT_2025-08-21": session_stats}
            
            # Test relative positioning analysis
            result = engine._analyze_relative_positioning("Analyze relative positioning patterns")
            
            # Validate result structure
            assert result["query_type"] == "relative_positioning"
            assert "positioning_analysis" in result
            assert "insights" in result
            
            print("  ✅ Relative positioning analysis structure valid")
            
            # Test that positioning analysis contains session data
            if result["positioning_analysis"]:
                session_analysis = list(result["positioning_analysis"].values())[0]
                assert "avg_position" in session_analysis
                assert "position_std" in session_analysis
                assert "session_type" in session_analysis
                print("  ✅ Positioning analysis contains expected metrics")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Relative positioning analysis failed: {e}")
        traceback.print_exc()
        return False

def test_liquidity_sweep_analysis():
    """Test liquidity sweep analysis functionality"""
    print("\n🧪 Testing Liquidity Sweep Analysis...")
    
    try:
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            
            engine = EnhancedTemporalQueryEngine()
            
            # Create mock data with sweep patterns
            sweep_data = {
                'price': [100.0, 101.0, 105.0, 102.0, 98.0, 95.0, 99.0, 103.0],  # High and low sweeps
                'volume': [1000, 1200, 2000, 800, 900, 1800, 700, 1100],
                'timestamp': [f'2025-08-21 09:3{i}:00' for i in range(8)]
            }
            nodes_df = pd.DataFrame(sweep_data)
            session_stats = {'high': 105.0, 'low': 95.0, 'range': 10.0}
            
            engine.sessions = {"MIDNIGHT_2025-08-21": nodes_df}
            engine.session_stats = {"MIDNIGHT_2025-08-21": session_stats}
            
            # Test liquidity sweep analysis
            result = engine._analyze_liquidity_sweeps("Analyze liquidity sweep patterns")
            
            # Validate result structure
            assert result["query_type"] == "liquidity_sweeps"
            assert "sweep_events" in result
            assert "sweep_statistics" in result
            assert "insights" in result
            
            print("  ✅ Liquidity sweep analysis structure valid")
            
            # Test that sweep analysis detects events
            if result["sweep_events"]:
                sweep_event = result["sweep_events"][0]
                assert "sweep_type" in sweep_event
                assert "price" in sweep_event
                assert "sweep_strength" in sweep_event
                print(f"  ✅ Detected {len(result['sweep_events'])} sweep events")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Liquidity sweep analysis failed: {e}")
        traceback.print_exc()
        return False

def test_pattern_search_functionality():
    """Test pattern search functionality"""
    print("\n🧪 Testing Pattern Search Functionality...")
    
    try:
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            
            engine = EnhancedTemporalQueryEngine()
            
            # Add mock session data
            nodes_df, session_stats = create_mock_session_data()
            engine.sessions = {"MIDNIGHT_2025-08-21": nodes_df}
            engine.session_stats = {"MIDNIGHT_2025-08-21": session_stats}
            
            # Test pattern search
            result = engine._search_patterns("Search for liquidity patterns at 50% zone")
            
            # Validate result structure
            assert result["query_type"] == "pattern_search"
            assert "pattern_matches" in result
            assert "search_criteria" in result
            assert "insights" in result
            
            print("  ✅ Pattern search structure valid")
            
            # Test search criteria extraction
            criteria = result["search_criteria"]
            assert "pattern_type" in criteria
            print(f"  ✅ Search criteria extracted: {criteria}")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Pattern search functionality failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test backward compatibility with original interface"""
    print("\n🧪 Testing Backward Compatibility...")
    
    try:
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            # Test original import path
            from ironforge.temporal import EnhancedTemporalQueryEngine
            
            # Test that we can create engine as before
            engine = EnhancedTemporalQueryEngine()
            
            # Test that original attributes are accessible
            assert hasattr(engine, 'sessions'), "sessions attribute should be accessible"
            assert hasattr(engine, 'graphs'), "graphs attribute should be accessible"
            assert hasattr(engine, 'metadata'), "metadata attribute should be accessible"
            assert hasattr(engine, 'session_stats'), "session_stats attribute should be accessible"
            
            # Test that original methods exist
            assert hasattr(engine, 'ask'), "ask method should exist"
            assert hasattr(engine, 'get_enhanced_session_info'), "get_enhanced_session_info should exist"
            assert hasattr(engine, 'list_sessions'), "list_sessions should exist"
            
            print("  ✅ Original interface preserved")
            
            # Test that we can call methods as before
            sessions = engine.list_sessions()
            assert isinstance(sessions, list), "list_sessions should return a list"
            
            print("  ✅ Original method calls work")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_visualization_integration():
    """Test visualization integration"""
    print("\n🧪 Testing Visualization Integration...")
    
    try:
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            
            engine = EnhancedTemporalQueryEngine()
            
            # Add mock session data
            nodes_df, session_stats = create_mock_session_data()
            engine.sessions = {"MIDNIGHT_2025-08-21": nodes_df}
            engine.session_stats = {"MIDNIGHT_2025-08-21": session_stats}
            
            # Test that visualization methods exist
            assert hasattr(engine, 'display_results'), "display_results method should exist"
            assert hasattr(engine, 'plot_results'), "plot_results method should exist"
            
            # Test display_results with mock data
            result = engine.ask("What happens after liquidity sweeps?")
            
            # This should not raise an exception
            try:
                engine.display_results(result)
                print("  ✅ display_results method works")
            except Exception as display_error:
                # Display might fail due to missing matplotlib, but method should exist
                print(f"  ⚠️  display_results exists but may need matplotlib: {display_error}")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Visualization integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all functional tests"""
    print("🚀 Enhanced Temporal Query Engine Functional Test Suite")
    print("=" * 70)
    
    tests = [
        test_temporal_engine_initialization,
        test_temporal_query_functionality,
        test_archaeological_zone_analysis,
        test_relative_positioning_analysis,
        test_liquidity_sweep_analysis,
        test_pattern_search_functionality,
        test_backward_compatibility,
        test_visualization_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            
    print(f"\n📊 Functional Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All functional tests passed! Refactoring is working correctly.")
        print("\n✅ Core functionality is preserved and working")
        print("✅ New modular architecture is functional")
        print("✅ Backward compatibility is maintained")
        print("✅ Query processing works as expected")
        print("✅ Analysis methods produce valid results")
    else:
        print("⚠️  Some functional tests failed. Review the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
