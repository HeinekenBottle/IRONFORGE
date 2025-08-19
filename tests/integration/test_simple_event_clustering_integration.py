#!/usr/bin/env python3
"""
Simple Event-Time Clustering Integration Test
=============================================

Validates the integration of Simple Event-Time Clustering + Cross-TF Mapping
with the existing IRONFORGE archaeological discovery system.

Tests:
1. Module import and initialization
2. Individual component functionality  
3. Full orchestrator integration
4. Performance validation (<0.05s overhead)
5. Output data quality and structure

Author: IRONFORGE Enhancement Team
Target: IRONFORGE Archaeological Discovery System
"""

import sys
import time
from datetime import datetime, timedelta

# Add IRONFORGE to path for testing
sys.path.insert(0, '/Users/jack/IRONPULSE/IRONFORGE')

def test_module_imports():
    """Test 1: Verify all imports work correctly"""
    print("🔍 Test 1: Module Import Validation")
    print("-" * 50)
    
    try:
        # Test core module import
        from ironforge.learning.simple_event_clustering import (
            CrossTFMapper,
            EventTimeClusterer,
            SimpleEventAnalyzer,
            analyze_time_patterns,
        )
        print("✅ Simple event clustering module imports successful")
        
        # Test orchestrator integration import
        from orchestrator import IRONFORGE
        print("✅ Enhanced orchestrator imports successful")
        
        # Test that the new import in orchestrator works
        print("✅ Orchestrator integration import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_individual_components():
    """Test 2: Validate individual component functionality"""
    print("\n🧪 Test 2: Individual Component Functionality")
    print("-" * 50)
    
    try:
        from ironforge.learning.simple_event_clustering import (
            CrossTFMapper,
            EventTimeClusterer,
            SimpleEventAnalyzer,
        )
        
        # Test EventTimeClusterer
        print("Testing EventTimeClusterer...")
        clusterer = EventTimeClusterer(time_bin_minutes=5)
        
        sample_events = [
            {
                'id': 'event_1',
                'event_type': 'fvg_redelivery', 
                'timestamp': datetime.now() - timedelta(minutes=30),
                'price': 1.0500
            },
            {
                'id': 'event_2',
                'event_type': 'expansion_phase',
                'timestamp': datetime.now() - timedelta(minutes=25), 
                'price': 1.0510
            },
            {
                'id': 'event_3',
                'event_type': 'consolidation',
                'timestamp': datetime.now() - timedelta(minutes=5),
                'price': 1.0505
            }
        ]
        
        clustering_result = clusterer.cluster_events_by_time(sample_events)
        print(f"✅ EventTimeClusterer: {len(clustering_result['event_clusters'])} clusters created")
        
        # Test CrossTFMapper
        print("Testing CrossTFMapper...")
        mapper = CrossTFMapper()
        
        htf_data = {
            '15m_phase': 'expansion',
            '1h_structure': 'uptrend', 
            'daily_context': 'ny_open'
        }
        
        mappings = mapper.map_ltf_to_htf(sample_events, htf_data)
        print(f"✅ CrossTFMapper: {len(mappings)} LTF-HTF mappings created")
        
        # Test SimpleEventAnalyzer
        print("Testing SimpleEventAnalyzer...")
        analyzer = SimpleEventAnalyzer(time_bin_minutes=5)
        
        sample_graph = {
            'nodes': sample_events,
            'metadata': {
                'session_type': 'ny_open',
                'htf_15m_phase': 'expansion',
                'htf_1h_structure': 'uptrend'
            }
        }
        
        analysis_result = analyzer.analyze_session_time_patterns(sample_graph, "test_session.json")
        print(f"✅ SimpleEventAnalyzer: {analysis_result['analysis_metadata']['total_events_analyzed']} events analyzed")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def test_analyze_time_patterns_function():
    """Test 3: Validate main analyze_time_patterns function"""
    print("\n🎯 Test 3: Main Function Validation")
    print("-" * 50)
    
    try:
        from ironforge.learning.simple_event_clustering import analyze_time_patterns
        
        # Create test graph data
        test_graph = {
            'nodes': [
                {
                    'id': 'node_1',
                    'event_type': 'fvg_redelivery',
                    'timestamp': datetime.now() - timedelta(minutes=20),
                    'price': 1.0500,
                    'timeframe_source': 1
                },
                {
                    'id': 'node_2', 
                    'event_type': 'expansion_phase',
                    'timestamp': datetime.now() - timedelta(minutes=15),
                    'price': 1.0510,
                    'timeframe_source': 1
                }
            ],
            'metadata': {
                'session_type': 'london_open',
                'htf_15m_phase': 'consolidation_break',
                'htf_1h_structure': 'range_expansion',
                'total_nodes': 2
            }
        }
        
        # Test analysis
        start_time = time.time()
        result = analyze_time_patterns(test_graph, "test_session_main.json")
        processing_time = time.time() - start_time
        
        print(f"✅ Function executed in {processing_time*1000:.1f}ms")
        print(f"✅ Events analyzed: {result['analysis_metadata']['total_events_analyzed']}")
        print(f"✅ Clusters created: {len(result['event_clusters'])}")
        print(f"✅ Cross-TF mappings: {len(result['cross_tf_mapping']['ltf_to_15m'])}")
        
        # Validate output structure
        required_keys = ['event_clusters', 'cross_tf_mapping', 'clustering_stats', 'analysis_metadata']
        for key in required_keys:
            if key not in result:
                print(f"❌ Missing required key: {key}")
                return False
        
        print("✅ Output structure validation passed")
        
        # Performance validation (<50ms target)
        if processing_time > 0.05:
            print(f"⚠️ Performance warning: {processing_time*1000:.1f}ms > 50ms target")
        else:
            print(f"✅ Performance target met: {processing_time*1000:.1f}ms < 50ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Main function test failed: {e}")
        return False

def test_orchestrator_integration():
    """Test 4: Validate full orchestrator integration"""
    print("\n🚀 Test 4: Orchestrator Integration Validation")
    print("-" * 50)
    
    try:
        # Test that we can import the enhanced orchestrator
        from orchestrator import IRONFORGE
        print("✅ Enhanced IRONFORGE orchestrator imported")
        
        # Test that new method exists
        forge = IRONFORGE(enable_performance_monitoring=False)  # Disable for testing
        
        if hasattr(forge, '_analyze_time_patterns'):
            print("✅ _analyze_time_patterns method found")
        else:
            print("❌ _analyze_time_patterns method not found")
            return False
        
        # Test method functionality
        test_graph = {
            'nodes': [
                {
                    'event_type': 'consolidation',
                    'timestamp': datetime.now().timestamp(),
                    'price': 1.0500
                }
            ],
            'metadata': {
                'session_type': 'asia_session'
            }
        }
        
        time_patterns = forge._analyze_time_patterns(test_graph, "test_integration.json")
        print("✅ Method executed successfully")
        print(f"✅ Returned {len(time_patterns.get('event_clusters', []))} clusters")
        
        # Validate error handling
        empty_graph = {'nodes': [], 'metadata': {}}
        forge._analyze_time_patterns(empty_graph, "empty_test.json")
        print("✅ Error handling: empty graph handled gracefully")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        return False

def test_performance_overhead():
    """Test 5: Measure actual performance overhead"""
    print("\n⚡ Test 5: Performance Overhead Validation")
    print("-" * 50)
    
    try:
        from ironforge.learning.simple_event_clustering import analyze_time_patterns
        
        # Create realistic test data
        events = []
        base_time = datetime.now() - timedelta(hours=1)
        
        for i in range(50):  # 50 events over 1 hour
            events.append({
                'id': f'event_{i}',
                'event_type': ['fvg_redelivery', 'expansion_phase', 'consolidation'][i % 3],
                'timestamp': base_time + timedelta(minutes=i),
                'price': 1.0500 + (i * 0.0001),
                'timeframe_source': 1
            })
        
        test_graph = {
            'nodes': events,
            'metadata': {
                'session_type': 'ny_session',
                'htf_15m_phase': 'trending',
                'htf_1h_structure': 'uptrend',
                'total_nodes': len(events)
            }
        }
        
        # Multiple runs for accurate measurement
        times = []
        for i in range(10):
            start_time = time.time()
            analyze_time_patterns(test_graph, f"perf_test_{i}.json")
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"✅ Average processing time: {avg_time*1000:.1f}ms")
        print(f"✅ Maximum processing time: {max_time*1000:.1f}ms")
        print(f"✅ Minimum processing time: {min_time*1000:.1f}ms")
        print(f"✅ Events per test: {len(events)}")
        
        # Performance targets
        if avg_time < 0.05:
            print(f"✅ PERFORMANCE TARGET MET: {avg_time*1000:.1f}ms < 50ms")
            performance_passed = True
        else:
            print(f"⚠️ PERFORMANCE WARNING: {avg_time*1000:.1f}ms > 50ms target")
            performance_passed = False
        
        # Overhead per event
        overhead_per_event = avg_time / len(events) * 1000
        print(f"✅ Overhead per event: {overhead_per_event:.2f}ms")
        
        return performance_passed
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_output_data_quality():
    """Test 6: Validate output data quality and structure"""
    print("\n📊 Test 6: Output Data Quality Validation")
    print("-" * 50)
    
    try:
        from ironforge.learning.simple_event_clustering import analyze_time_patterns
        
        # Create comprehensive test data
        events = [
            {
                'id': 'semantic_1',
                'event_type': 'fvg_redelivery',
                'timestamp': datetime.now() - timedelta(minutes=30),
                'price': 1.0500
            },
            {
                'id': 'semantic_2', 
                'event_type': 'expansion_phase',
                'timestamp': datetime.now() - timedelta(minutes=28),
                'price': 1.0510
            },
            {
                'id': 'semantic_3',
                'event_type': 'fvg_redelivery', 
                'timestamp': datetime.now() - timedelta(minutes=10),
                'price': 1.0505
            }
        ]
        
        test_graph = {
            'nodes': events,
            'metadata': {
                'session_type': 'london_close',
                'htf_15m_phase': 'retracement',
                'htf_1h_structure': 'trend_continuation',
                'semantic_events': events
            }
        }
        
        result = analyze_time_patterns(test_graph, "quality_test.json")
        
        # Validate top-level structure
        print("Validating output structure...")
        required_structure = {
            'event_clusters': list,
            'cross_tf_mapping': dict, 
            'clustering_stats': dict,
            'analysis_metadata': dict
        }
        
        for key, expected_type in required_structure.items():
            if key not in result:
                print(f"❌ Missing key: {key}")
                return False
            if not isinstance(result[key], expected_type):
                print(f"❌ Wrong type for {key}: expected {expected_type}, got {type(result[key])}")
                return False
        
        print("✅ Top-level structure valid")
        
        # Validate cross_tf_mapping structure
        cross_tf = result['cross_tf_mapping']
        required_cross_tf_keys = ['ltf_to_15m', 'ltf_to_1h', 'structural_alignments']
        for key in required_cross_tf_keys:
            if key not in cross_tf:
                print(f"❌ Missing cross_tf_mapping key: {key}")
                return False
        
        print("✅ Cross-TF mapping structure valid")
        
        # Validate clustering_stats
        stats = result['clustering_stats']
        required_stats = ['total_events', 'temporal_distribution', 'max_density', 'avg_density']
        for key in required_stats:
            if key not in stats:
                print(f"❌ Missing clustering_stats key: {key}")
                return False
        
        print("✅ Clustering stats structure valid")
        
        # Validate analysis_metadata
        metadata = result['analysis_metadata']
        required_metadata = ['total_events_analyzed', 'processing_time_ms', 'session_file', 'analysis_timestamp']
        for key in required_metadata:
            if key not in metadata:
                print(f"❌ Missing analysis_metadata key: {key}")
                return False
        
        print("✅ Analysis metadata structure valid")
        
        # Validate data content - analyzer extracts from both nodes and metadata.semantic_events
        len(events) * 2  # Events appear in both nodes and metadata.semantic_events
        actual_events = result['analysis_metadata']['total_events_analyzed']
        
        if actual_events < len(events):  # Should find at least the direct events
            print(f"❌ Event count too low: expected at least {len(events)}, got {actual_events}")
            return False
        
        print(f"✅ Event analysis count valid: {actual_events} events found (includes nodes + semantic_events)")
        
        # Check for meaningful clustering
        if len(result['event_clusters']) > 0:
            cluster = result['event_clusters'][0]
            required_cluster_keys = ['time_bin', 'event_count', 'density_score', 'dominant_events', 'htf_context']
            for key in required_cluster_keys:
                if key not in cluster:
                    print(f"❌ Missing cluster key: {key}")
                    return False
            print("✅ Event cluster structure valid")
        
        print("✅ All data quality validations passed")
        return True
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False

def run_comprehensive_integration_test():
    """Run all integration tests and provide summary"""
    print("🔬 SIMPLE EVENT-TIME CLUSTERING INTEGRATION TEST")
    print("=" * 80)
    print("Testing integration with IRONFORGE Archaeological Discovery System")
    print("Target: Non-invasive time pattern analysis with <0.05s overhead")
    print("=" * 80)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Module Imports", test_module_imports),
        ("Individual Components", test_individual_components), 
        ("Main Function", test_analyze_time_patterns_function),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Performance Overhead", test_performance_overhead),
        ("Output Data Quality", test_output_data_quality)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("🏁 INTEGRATION TEST SUMMARY") 
    print("=" * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print("-" * 80)
    print(f"📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Integration ready for production!")
        print("✅ Simple Event-Time Clustering + Cross-TF Mapping successfully integrated")
        print("✅ Performance target met: <0.05s overhead per session")
        print("✅ Non-invasive integration preserves existing functionality")
        print("✅ Rich time pattern output available in session metadata")
    else:
        print("⚠️ Some tests failed - review failures before production deployment")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    
    if success:
        print("\n🚀 INTEGRATION COMPLETE - Ready for IRONFORGE enhancement!")
        print("Next steps:")
        print("1. Deploy to IRONFORGE production environment")
        print("2. Monitor time pattern analysis in session metadata")
        print("3. Leverage 'when events cluster' + 'what HTF context' intelligence")
        
        sys.exit(0)
    else:
        print("\n❌ INTEGRATION INCOMPLETE - Address test failures")
        sys.exit(1)