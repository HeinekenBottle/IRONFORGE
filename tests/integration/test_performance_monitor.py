#!/usr/bin/env python3
"""
IRONFORGE Performance Monitor Test Suite
========================================

Test suite for Sprint 2 performance monitoring system.
Validates monitoring functionality, regression detection, and quality gates.

This script demonstrates and tests:
- Performance metric collection
- Regression analysis with fail-fast behavior  
- Quality gate validation (37D features, 4 edge types, 100% validation accuracy)
- Sprint 2 enhancement tracking (structural edges, regime clustering, precursors)
- Integration with IRONFORGE orchestrator

NO FALLBACKS POLICY: Tests fail fast on performance regression or quality gate failures.
"""

import sys
import time
from pathlib import Path

# Add IRONFORGE to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from orchestrator import IRONFORGE
    from scripts.utilities.performance_monitor import (
        BaselineMetrics,
        PerformanceMetrics,
        PerformanceMonitor,
        create_graph_analysis,
    )
except ImportError as e:
    print(f"❌ Failed to import IRONFORGE components: {e}")
    sys.exit(1)

def create_mock_orchestrator_results(good_performance: bool = True) -> dict:
    """Create mock orchestrator results for testing"""
    if good_performance:
        return {
            'sessions_processed': 10,
            'patterns_discovered': [{'type': f'pattern_{i}', 'confidence': 0.9} for i in range(42)],
            'graphs_preserved': ['session_1.json', 'session_2.json'],
            'skipped_sessions': [],
            'processing_errors': []
        }
    else:
        # Simulate poor performance
        return {
            'sessions_processed': 5,
            'patterns_discovered': [{'type': f'pattern_{i}', 'confidence': 0.6} for i in range(8)],
            'graphs_preserved': ['session_1.json'],
            'skipped_sessions': [{'file': 'bad_session.json', 'reason': 'data_error'}],
            'processing_errors': [{'file': 'error_session.json', 'error': 'processing_failed'}]
        }

def create_mock_validation_results(good_performance: bool = True) -> dict:
    """Create mock validation results for testing"""
    if good_performance:
        return {
            'total_patterns': 42,
            'validated': 42,  # 100% validation success
            'results': [{'status': 'VALIDATED', 'improvement': 0.15} for _ in range(42)]
        }
    else:
        return {
            'total_patterns': 8,
            'validated': 6,  # 75% validation success (below 100% requirement)
            'results': [{'status': 'VALIDATED' if i < 6 else 'FAILED', 'improvement': 0.05} for i in range(8)]
        }

def create_mock_graph_analysis(sprint_2_active: bool = True) -> dict:
    """Create mock graph analysis for testing"""
    if sprint_2_active:
        return {
            'total_edges': 1000,
            'structural_edges_created': 120,  # 12% structural edges
            'feature_dimensions': 37,  # Correct Sprint 2 dimension count
            'edge_type_count': 4,  # temporal, scale, discovered, structural_context
            'regime_clusters_identified': 5,
            'precursor_patterns_detected': 8
        }
    else:
        return {
            'total_edges': 800,
            'structural_edges_created': 0,  # No structural edges (baseline)
            'feature_dimensions': 37,
            'edge_type_count': 3,  # Missing structural_context edges
            'regime_clusters_identified': 0,
            'precursor_patterns_detected': 0
        }

def test_performance_metrics_collection():
    """Test basic performance metrics collection"""
    print("🧪 Testing performance metrics collection...")
    
    monitor = PerformanceMonitor(regression_threshold=0.15)
    monitor.start_monitoring()
    
    # Simulate some processing stages
    time.sleep(0.1)
    monitor.stage_checkpoint('graph_building')
    time.sleep(0.05)
    monitor.stage_checkpoint('tgat_training')
    time.sleep(0.02)
    monitor.stage_checkpoint('validation')
    
    # Collect metrics
    orchestrator_results = create_mock_orchestrator_results(good_performance=True)
    validation_results = create_mock_validation_results(good_performance=True)
    graph_analysis = create_mock_graph_analysis(sprint_2_active=True)
    
    metrics = monitor.collect_metrics(orchestrator_results, validation_results, graph_analysis)
    
    # Validate collected metrics
    assert metrics.processing_time_sec > 0.15, "Processing time should be recorded"
    assert metrics.sessions_processed == 10, "Should track sessions processed"
    assert metrics.patterns_discovered == 42, "Should track patterns discovered"
    assert metrics.structural_edges_created == 120, "Should track structural edges"
    assert metrics.feature_dimension_count == 37, "Should track 37D features"
    assert metrics.edge_type_count == 4, "Should track 4 edge types"
    assert metrics.validation_accuracy == 1.0, "Should achieve 100% validation accuracy"
    assert metrics.tgat_compatibility is True, "Should be TGAT compatible"
    
    print("   ✅ Performance metrics collection working correctly")
    return metrics

def test_regression_detection_pass():
    """Test regression detection with acceptable performance"""
    print("🧪 Testing regression detection (should pass)...")
    
    # Create baseline
    baseline = BaselineMetrics(
        processing_time_sec=1.0,
        memory_usage_mb=100.0,
        discovery_rate_per_session=4.0,
        validation_success_rate=1.0,
        structural_edges_created=0,  # Baseline has no structural edges
        edge_type_count=3
    )
    
    # Create current metrics with acceptable performance
    current_metrics = PerformanceMetrics(
        processing_time_sec=1.1,  # 10% slower (within 15% threshold)
        memory_usage_mb=110.0,    # 10% more memory (within threshold)
        peak_memory_mb=120.0,
        gpu_utilization_pct=None,
        sessions_processed=10,
        patterns_discovered=42,
        validation_success_rate=1.0,  # Maintain 100%
        discovery_rate_per_session=4.2,  # Improved discovery rate
        structural_edges_created=120,  # New Sprint 2 capability
        structural_edge_ratio=0.12,
        regime_clusters_identified=5,
        precursor_patterns_detected=8,
        failed_sessions=0,
        failure_rate=0.0,
        error_count=0,
        skipped_sessions=0,
        feature_dimension_count=37,
        edge_type_count=4,  # Upgraded from 3 to 4
        tgat_compatibility=True,
        validation_accuracy=1.0,
        graph_building_time_sec=0.5,
        tgat_training_time_sec=0.4,
        validation_time_sec=0.1,
        preservation_time_sec=0.1
    )
    
    monitor = PerformanceMonitor(regression_threshold=0.15)
    analysis = monitor.analyze_performance_regression(current_metrics, baseline)
    
    assert analysis['status'] == 'pass', "Should pass regression analysis"
    assert len(analysis['regressions']) == 0, "Should have no regressions"
    assert len(analysis['quality_failures']) == 0, "Should have no quality failures"
    
    print("   ✅ Regression detection correctly passes acceptable performance")
    return analysis

def test_regression_detection_fail():
    """Test regression detection with unacceptable performance (should raise error)"""
    print("🧪 Testing regression detection (should fail fast)...")
    
    # Create baseline
    baseline = BaselineMetrics(
        processing_time_sec=1.0,
        memory_usage_mb=100.0,
        discovery_rate_per_session=4.0,
        validation_success_rate=1.0,
        structural_edges_created=0,
        edge_type_count=3
    )
    
    # Create current metrics with unacceptable regression
    current_metrics = PerformanceMetrics(
        processing_time_sec=2.0,  # 100% slower (exceeds 15% threshold)
        memory_usage_mb=150.0,    # 50% more memory (exceeds threshold)
        peak_memory_mb=180.0,
        gpu_utilization_pct=None,
        sessions_processed=10,
        patterns_discovered=20,   # Fewer patterns discovered
        validation_success_rate=0.8,  # Failed validation accuracy requirement
        discovery_rate_per_session=2.0,  # 50% worse discovery rate
        structural_edges_created=50,
        structural_edge_ratio=0.05,
        regime_clusters_identified=2,
        precursor_patterns_detected=3,
        failed_sessions=2,
        failure_rate=0.2,
        error_count=3,
        skipped_sessions=1,
        feature_dimension_count=35,  # Wrong dimension count
        edge_type_count=3,  # Missing structural_context edges
        tgat_compatibility=False,
        validation_accuracy=0.8,  # Below 100% requirement
        graph_building_time_sec=1.0,
        tgat_training_time_sec=0.8,
        validation_time_sec=0.1,
        preservation_time_sec=0.1
    )
    
    monitor = PerformanceMonitor(regression_threshold=0.15)
    
    # Should raise RuntimeError due to NO FALLBACKS policy
    try:
        monitor.analyze_performance_regression(current_metrics, baseline)
        raise AssertionError("Should have raised RuntimeError for regression")
    except RuntimeError as e:
        assert "PERFORMANCE REGRESSION DETECTED" in str(e), "Should indicate regression"
        print("   ✅ Regression detection correctly fails fast on poor performance")
        return str(e)

def test_quality_gates():
    """Test quality gate validation"""
    print("🧪 Testing quality gates...")
    
    # Test passing quality gates
    good_metrics = PerformanceMetrics(
        processing_time_sec=1.0, memory_usage_mb=100.0, peak_memory_mb=120.0, gpu_utilization_pct=None,
        sessions_processed=10, patterns_discovered=42, validation_success_rate=1.0, discovery_rate_per_session=4.2,
        structural_edges_created=120, structural_edge_ratio=0.12, regime_clusters_identified=5, precursor_patterns_detected=8,
        failed_sessions=0, failure_rate=0.0, error_count=0, skipped_sessions=0,
        feature_dimension_count=37,  # Correct
        edge_type_count=4,          # Correct
        tgat_compatibility=True,    # Correct
        validation_accuracy=1.0,    # Correct
        graph_building_time_sec=0.5, tgat_training_time_sec=0.4, validation_time_sec=0.1, preservation_time_sec=0.1
    )
    
    monitor = PerformanceMonitor()
    analysis = monitor.analyze_performance_regression(good_metrics, None)  # No baseline
    
    assert analysis['status'] == 'pass', "Quality gates should pass"
    
    # Test failing quality gates
    bad_metrics = PerformanceMetrics(
        processing_time_sec=1.0, memory_usage_mb=100.0, peak_memory_mb=120.0, gpu_utilization_pct=None,
        sessions_processed=10, patterns_discovered=42, validation_success_rate=1.0, discovery_rate_per_session=4.2,
        structural_edges_created=120, structural_edge_ratio=0.12, regime_clusters_identified=5, precursor_patterns_detected=8,
        failed_sessions=0, failure_rate=0.0, error_count=0, skipped_sessions=0,
        feature_dimension_count=35,  # Wrong dimension count
        edge_type_count=3,          # Missing edge type
        tgat_compatibility=False,   # Failed compatibility
        validation_accuracy=0.8,    # Below requirement
        graph_building_time_sec=0.5, tgat_training_time_sec=0.4, validation_time_sec=0.1, preservation_time_sec=0.1
    )
    
    try:
        analysis = monitor.analyze_performance_regression(bad_metrics, None)
        raise AssertionError("Should have raised RuntimeError for quality gate failures")
    except RuntimeError as e:
        assert "Quality Gate Failures" in str(e), "Should indicate quality gate failures"
        print("   ✅ Quality gates correctly enforce requirements")

def test_sprint_2_enhancement_tracking():
    """Test Sprint 2 enhancement effectiveness assessment"""
    print("🧪 Testing Sprint 2 enhancement tracking...")
    
    monitor = PerformanceMonitor()
    
    # Test excellent enhancements
    excellent_metrics = PerformanceMetrics(
        processing_time_sec=1.0, memory_usage_mb=100.0, peak_memory_mb=120.0, gpu_utilization_pct=None,
        sessions_processed=10, patterns_discovered=42, validation_success_rate=1.0, discovery_rate_per_session=4.2,
        structural_edges_created=200, structural_edge_ratio=0.15,  # Excellent structural edges
        regime_clusters_identified=8, precursor_patterns_detected=12,  # Excellent counts
        failed_sessions=0, failure_rate=0.0, error_count=0, skipped_sessions=0,
        feature_dimension_count=37, edge_type_count=4, tgat_compatibility=True, validation_accuracy=1.0,
        graph_building_time_sec=0.5, tgat_training_time_sec=0.4, validation_time_sec=0.1, preservation_time_sec=0.1
    )
    
    effectiveness = monitor._assess_enhancement_effectiveness(excellent_metrics)
    
    assert effectiveness['structural_context'] == 'EXCELLENT', "Should assess structural context as excellent"
    assert effectiveness['regime_segmentation'] == 'EXCELLENT', "Should assess regime segmentation as excellent"  
    assert effectiveness['precursor_detection'] == 'EXCELLENT', "Should assess precursor detection as excellent"
    
    # Test inactive enhancements
    inactive_metrics = PerformanceMetrics(
        processing_time_sec=1.0, memory_usage_mb=100.0, peak_memory_mb=120.0, gpu_utilization_pct=None,
        sessions_processed=10, patterns_discovered=42, validation_success_rate=1.0, discovery_rate_per_session=4.2,
        structural_edges_created=0, structural_edge_ratio=0.0,  # No structural edges
        regime_clusters_identified=0, precursor_patterns_detected=0,  # No enhancements active
        failed_sessions=0, failure_rate=0.0, error_count=0, skipped_sessions=0,
        feature_dimension_count=37, edge_type_count=4, tgat_compatibility=True, validation_accuracy=1.0,
        graph_building_time_sec=0.5, tgat_training_time_sec=0.4, validation_time_sec=0.1, preservation_time_sec=0.1
    )
    
    effectiveness = monitor._assess_enhancement_effectiveness(inactive_metrics)
    
    assert effectiveness['structural_context'] == 'NOT_ACTIVE', "Should assess structural context as not active"
    assert effectiveness['regime_segmentation'] == 'NOT_ACTIVE', "Should assess regime segmentation as not active"
    assert effectiveness['precursor_detection'] == 'NOT_ACTIVE', "Should assess precursor detection as not active"
    
    print("   ✅ Sprint 2 enhancement tracking working correctly")

def test_orchestrator_integration():
    """Test integration with IRONFORGE orchestrator"""
    print("🧪 Testing orchestrator integration...")
    
    # Create temporary IRONFORGE instance with monitoring
    forge = IRONFORGE(use_enhanced=True, enable_performance_monitoring=True)
    
    assert forge.performance_monitor is not None, "Performance monitor should be initialized"
    
    # Test that the monitor has been hooked into the orchestrator
    # (We can't test full processing without valid session data, but we can test the setup)
    
    print("   ✅ Orchestrator integration working correctly")

def test_report_generation():
    """Test performance report generation"""
    print("🧪 Testing performance report generation...")
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    time.sleep(0.1)
    monitor.stage_checkpoint('test_stage')
    
    # Create comprehensive test data
    orchestrator_results = create_mock_orchestrator_results(good_performance=True)
    validation_results = create_mock_validation_results(good_performance=True)
    graph_analysis = create_mock_graph_analysis(sprint_2_active=True)
    
    metrics = monitor.collect_metrics(orchestrator_results, validation_results, graph_analysis)
    report = monitor.generate_performance_report(metrics, include_regression=False)
    
    # Validate report structure
    assert 'timestamp' in report, "Report should include timestamp"
    assert 'sprint_version' in report, "Report should identify Sprint 2"
    assert 'monitoring_session' in report, "Report should include monitoring session data"
    assert 'discovery_performance' in report, "Report should include discovery performance"
    assert 'sprint_2_enhancements' in report, "Report should include Sprint 2 enhancements"
    assert 'quality_gates' in report, "Report should include quality gates"
    assert 'system_stability' in report, "Report should include system stability"
    
    # Validate Sprint 2 specific content
    enhancements = report['sprint_2_enhancements']
    assert enhancements['structural_edges_created'] == 120, "Should track structural edges"
    assert enhancements['regime_clusters_identified'] == 5, "Should track regime clusters"
    assert enhancements['precursor_patterns_detected'] == 8, "Should track precursor patterns"
    
    quality_gates = report['quality_gates']
    assert quality_gates['feature_dimensions']['status'] == 'PASS', "Feature dimensions should pass"
    assert quality_gates['edge_types']['status'] == 'PASS', "Edge types should pass"
    assert quality_gates['validation_accuracy']['status'] == 'PASS', "Validation accuracy should pass"
    
    print("   ✅ Performance report generation working correctly")
    return report

def run_comprehensive_test():
    """Run comprehensive test suite for performance monitoring system"""
    print("🚀 IRONFORGE Performance Monitor - Comprehensive Test Suite")
    print("=" * 70)
    print("Testing Sprint 2 structural intelligence performance monitoring")
    print()
    
    tests_passed = 0
    total_tests = 7
    
    try:
        # Test 1: Basic metrics collection
        test_performance_metrics_collection()
        tests_passed += 1
        
        # Test 2: Regression detection (pass case)
        test_regression_detection_pass()
        tests_passed += 1
        
        # Test 3: Regression detection (fail case)
        test_regression_detection_fail()
        tests_passed += 1
        
        # Test 4: Quality gates
        test_quality_gates()
        tests_passed += 1
        
        # Test 5: Sprint 2 enhancement tracking
        test_sprint_2_enhancement_tracking()
        tests_passed += 1
        
        # Test 6: Orchestrator integration
        test_orchestrator_integration()
        tests_passed += 1
        
        # Test 7: Report generation
        test_report_generation()
        tests_passed += 1
        
        print(f"\n✅ ALL TESTS PASSED ({tests_passed}/{total_tests})")
        print("🎉 Performance monitoring system ready for Sprint 2 structural intelligence!")
        print()
        print("Key capabilities validated:")
        print("  ✅ 37D feature monitoring with 4 edge types")
        print("  ✅ Regression detection with 15% threshold")
        print("  ✅ 100% validation accuracy enforcement")
        print("  ✅ Structural context edge tracking")
        print("  ✅ Regime clustering effectiveness assessment")
        print("  ✅ Precursor detection pattern monitoring")
        print("  ✅ NO FALLBACKS policy compliance (fail fast on regression)")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print(f"Tests passed: {tests_passed}/{total_tests}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)