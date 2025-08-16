#!/usr/bin/env python3
"""
IRONFORGE Performance Monitor - Sprint 2 Enhancement
===================================================

Monitors 37D + 4 edge types performance vs baseline to ensure Sprint 2 
structural intelligence additions maintain acceptable performance levels.
"""

import time
import psutil
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    processing_time_ms: float
    memory_usage_mb: float
    pattern_count: int
    node_count: int
    edge_count: int
    feature_dimensions: int
    edge_types_count: int
    validation_success: bool
    timestamp: str

class PerformanceMonitor:
    """
    Monitor IRONFORGE performance with Sprint 2 enhancements
    Tracks 37D + 4 edge types vs baseline performance
    """
    
    def __init__(self, regression_threshold=0.15, baseline_file=None):
        self.logger = logging.getLogger(__name__)
        self.regression_threshold = regression_threshold  # 15% max degradation
        self.baseline_metrics = None
        self.performance_history = []
        
        # Load baseline metrics if available
        if baseline_file and Path(baseline_file).exists():
            self._load_baseline_metrics(baseline_file)
        
        # Performance tracking
        self.current_session_metrics = None
        self.process = psutil.Process()
        
    def start_monitoring(self, session_name: str = "session") -> Dict[str, Any]:
        """Start performance monitoring for a session"""
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        monitoring_context = {
            'session_name': session_name,
            'start_time': start_time,
            'start_memory': start_memory,
            'start_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Started monitoring session: {session_name}")
        return monitoring_context
    
    def end_monitoring(self, monitoring_context: Dict[str, Any], 
                      result_data: Dict[str, Any]) -> PerformanceMetrics:
        """End performance monitoring and calculate metrics"""
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate performance metrics
        processing_time = (end_time - monitoring_context['start_time']) * 1000  # ms
        memory_usage = end_memory - monitoring_context['start_memory']  # MB change
        
        # Extract result data
        pattern_count = len(result_data.get('patterns', []))
        node_count = result_data.get('metadata', {}).get('total_nodes', 0)
        edge_count = result_data.get('metadata', {}).get('total_edges', 0)
        feature_dimensions = result_data.get('metadata', {}).get('feature_dimensions', 0)
        edge_types = result_data.get('metadata', {}).get('edge_types', [])
        edge_types_count = len(edge_types)
        validation_success = result_data.get('validation_success', True)
        
        metrics = PerformanceMetrics(
            processing_time_ms=processing_time,
            memory_usage_mb=memory_usage,
            pattern_count=pattern_count,
            node_count=node_count,
            edge_count=edge_count,
            feature_dimensions=feature_dimensions,
            edge_types_count=edge_types_count,
            validation_success=validation_success,
            timestamp=datetime.now().isoformat()
        )
        
        # Store metrics
        self.performance_history.append(metrics)
        self.current_session_metrics = metrics
        
        self.logger.info(f"Completed monitoring: {processing_time:.1f}ms, {memory_usage:.1f}MB")
        return metrics
    
    def check_performance_regression(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Check if current performance shows regression vs baseline"""
        
        if not self.baseline_metrics:
            return {
                'regression_detected': False,
                'reason': 'No baseline metrics available',
                'recommendation': 'Establish baseline metrics for future comparisons'
            }
        
        baseline = self.baseline_metrics
        
        # Calculate performance ratios
        time_ratio = current_metrics.processing_time_ms / baseline.processing_time_ms
        memory_ratio = abs(current_metrics.memory_usage_mb) / abs(baseline.memory_usage_mb) if baseline.memory_usage_mb != 0 else 1.0
        
        # Check for regression (performance degradation)
        regressions = []
        
        if time_ratio > (1.0 + self.regression_threshold):
            regressions.append(f"Processing time: {time_ratio:.2f}x baseline ({time_ratio*100-100:.1f}% slower)")
        
        if memory_ratio > (1.0 + self.regression_threshold):
            regressions.append(f"Memory usage: {memory_ratio:.2f}x baseline ({memory_ratio*100-100:.1f}% more)")
        
        # Check validation success
        if not current_metrics.validation_success and baseline.validation_success:
            regressions.append("Validation success degraded from baseline")
        
        # Performance improvements (positive changes)
        improvements = []
        
        if time_ratio < 0.9:  # 10% improvement
            improvements.append(f"Processing time improved: {100-time_ratio*100:.1f}% faster")
        
        if memory_ratio < 0.9:  # 10% improvement  
            improvements.append(f"Memory usage improved: {100-memory_ratio*100:.1f}% less")
        
        regression_detected = len(regressions) > 0
        
        return {
            'regression_detected': regression_detected,
            'regressions': regressions,
            'improvements': improvements,
            'time_ratio': time_ratio,
            'memory_ratio': memory_ratio,
            'current_metrics': current_metrics,
            'baseline_metrics': baseline,
            'recommendation': self._get_performance_recommendation(
                regression_detected, time_ratio, memory_ratio
            )
        }
    
    def analyze_sprint2_impact(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze Sprint 2 impact on system performance"""
        
        # Expected Sprint 2 enhancements
        expected_features = 37  # 37D features
        expected_edge_types = 4  # 4 edge types: temporal, scale, structural_context, discovered
        
        analysis = {
            'feature_architecture_status': 'correct' if current_metrics.feature_dimensions == expected_features else 'incorrect',
            'edge_types_status': 'correct' if current_metrics.edge_types_count == expected_edge_types else 'incorrect',
            'feature_dimensions': {
                'current': current_metrics.feature_dimensions,
                'expected': expected_features,
                'status': '✅' if current_metrics.feature_dimensions == expected_features else '❌'
            },
            'edge_types': {
                'current': current_metrics.edge_types_count,
                'expected': expected_edge_types,
                'status': '✅' if current_metrics.edge_types_count == expected_edge_types else '❌'
            }
        }
        
        # Capability analysis
        capabilities = []
        
        if current_metrics.feature_dimensions >= 37:
            capabilities.append("✅ Temporal cycle detection (37D features)")
        else:
            capabilities.append("❌ Missing temporal cycle features")
        
        if current_metrics.edge_types_count >= 4:
            capabilities.append("✅ Structural context edges (4 edge types)")
        else:
            capabilities.append("❌ Missing structural context edges")
        
        if current_metrics.pattern_count > 0:
            capabilities.append("✅ Pattern discovery operational")
        else:
            capabilities.append("⚠️ No patterns discovered")
        
        analysis['capabilities'] = capabilities
        
        return analysis
    
    def generate_performance_report(self, metrics: PerformanceMetrics, 
                                  include_regression_check=True) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'session_metrics': {
                'processing_time_ms': metrics.processing_time_ms,
                'memory_usage_mb': metrics.memory_usage_mb,
                'pattern_count': metrics.pattern_count,
                'node_count': metrics.node_count,
                'edge_count': metrics.edge_count,
                'feature_dimensions': metrics.feature_dimensions,
                'edge_types_count': metrics.edge_types_count,
                'validation_success': metrics.validation_success
            },
            'sprint2_analysis': self.analyze_sprint2_impact(metrics),
            'timestamp': metrics.timestamp
        }
        
        # Add regression analysis if baseline available
        if include_regression_check:
            regression_check = self.check_performance_regression(metrics)
            report['regression_analysis'] = regression_check
        
        # Add historical context if available
        if len(self.performance_history) > 1:
            recent_metrics = self.performance_history[-5:]  # Last 5 sessions
            
            avg_processing_time = np.mean([m.processing_time_ms for m in recent_metrics])
            avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
            avg_pattern_count = np.mean([m.pattern_count for m in recent_metrics])
            
            report['historical_context'] = {
                'avg_processing_time_ms': avg_processing_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'avg_pattern_count': avg_pattern_count,
                'trend_processing_time': 'improving' if metrics.processing_time_ms < avg_processing_time else 'stable',
                'sessions_analyzed': len(recent_metrics)
            }
        
        return report
    
    def save_baseline_metrics(self, metrics: PerformanceMetrics, baseline_file: str) -> None:
        """Save current metrics as baseline for future comparisons"""
        
        baseline_data = {
            'processing_time_ms': metrics.processing_time_ms,
            'memory_usage_mb': metrics.memory_usage_mb,
            'pattern_count': metrics.pattern_count,
            'node_count': metrics.node_count,
            'edge_count': metrics.edge_count,
            'feature_dimensions': metrics.feature_dimensions,
            'edge_types_count': metrics.edge_types_count,
            'validation_success': metrics.validation_success,
            'timestamp': metrics.timestamp,
            'baseline_version': 'Sprint_2_Enhanced'
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        self.baseline_metrics = metrics
        self.logger.info(f"Baseline metrics saved to {baseline_file}")
    
    def _load_baseline_metrics(self, baseline_file: str) -> None:
        """Load baseline metrics from file"""
        
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            self.baseline_metrics = PerformanceMetrics(
                processing_time_ms=baseline_data['processing_time_ms'],
                memory_usage_mb=baseline_data['memory_usage_mb'],
                pattern_count=baseline_data['pattern_count'],
                node_count=baseline_data['node_count'],
                edge_count=baseline_data['edge_count'],
                feature_dimensions=baseline_data['feature_dimensions'],
                edge_types_count=baseline_data['edge_types_count'],
                validation_success=baseline_data['validation_success'],
                timestamp=baseline_data['timestamp']
            )
            
            self.logger.info(f"Loaded baseline metrics from {baseline_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not load baseline metrics: {e}")
            self.baseline_metrics = None
    
    def _get_performance_recommendation(self, regression_detected: bool, 
                                      time_ratio: float, memory_ratio: float) -> str:
        """Get performance optimization recommendation"""
        
        if not regression_detected:
            return "Performance within acceptable limits"
        
        recommendations = []
        
        if time_ratio > 1.2:  # 20% slower
            recommendations.append("Optimize processing pipeline for speed")
        
        if memory_ratio > 1.2:  # 20% more memory
            recommendations.append("Review memory usage in enhanced features")
        
        if time_ratio > 1.5:  # 50% slower - critical
            recommendations.append("CRITICAL: Consider disabling some Sprint 2 features")
        
        return "; ".join(recommendations) if recommendations else "Monitor performance closely"
    
    def hook_into_orchestrator(self, orchestrator):
        """
        Hook performance monitor into IRONFORGE orchestrator.
        
        This method is called by orchestrator.py to integrate performance monitoring
        with the discovery pipeline.
        """
        self.logger.info("Performance monitor hooked into orchestrator")
        
        # Store orchestrator reference for performance tracking
        self.orchestrator = orchestrator
        
        # Set up performance tracking hooks
        if hasattr(orchestrator, 'config'):
            self.config_integration = True
            self.logger.info("Configuration integration enabled")
        
        # Initialize performance baseline if available
        if hasattr(orchestrator, 'preservation_path'):
            baseline_file = f"{orchestrator.preservation_path}/performance_baseline.json"
            if Path(baseline_file).exists():
                self._load_baseline_metrics(baseline_file)
                self.logger.info(f"Loaded baseline metrics from {baseline_file}")
        
        return True

def monitor_ironforge_session(session_data: Dict, orchestrator_func, 
                            baseline_file: str = None) -> Dict[str, Any]:
    """
    Convenience function to monitor a complete IRONFORGE session
    
    Args:
        session_data: Input session data
        orchestrator_func: Function to call for processing (e.g., orchestrator.process_session)
        baseline_file: Optional baseline metrics file
    
    Returns:
        Combined processing results and performance report
    """
    
    monitor = PerformanceMonitor(baseline_file=baseline_file)
    
    # Start monitoring
    session_name = session_data.get('session_metadata', {}).get('session_type', 'unknown')
    monitoring_context = monitor.start_monitoring(session_name)
    
    try:
        # Execute the processing function
        results = orchestrator_func(session_data)
        results['validation_success'] = True
        
    except Exception as e:
        # Handle processing errors
        results = {
            'patterns': [],
            'metadata': {'total_nodes': 0, 'total_edges': 0, 'feature_dimensions': 0, 'edge_types': []},
            'validation_success': False,
            'error': str(e)
        }
    
    # End monitoring and generate report
    metrics = monitor.end_monitoring(monitoring_context, results)
    performance_report = monitor.generate_performance_report(metrics)
    
    # Combine results
    return {
        'processing_results': results,
        'performance_report': performance_report,
        'performance_metrics': metrics
    }

def main():
    """Command-line interface for performance monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IRONFORGE Performance Monitor")
    parser.add_argument('--baseline', '-b', help="Baseline metrics file")
    parser.add_argument('--save-baseline', help="Save current run as baseline")
    parser.add_argument('--test-mode', action='store_true', help="Run synthetic performance test")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.test_mode:
        print("🧪 Running synthetic performance test...")
        
        # Create synthetic test data
        test_session = {
            'session_metadata': {'session_type': 'synthetic_test'},
            'patterns': [{'id': i, 'type': 'test_pattern'} for i in range(50)],
            'metadata': {
                'total_nodes': 25,
                'total_edges': 40,
                'feature_dimensions': 37,
                'edge_types': ['temporal', 'scale', 'structural_context', 'discovered']
            }
        }
        
        def synthetic_processing(data):
            """Synthetic processing function for testing"""
            import time
            time.sleep(0.1)  # Simulate processing time
            return data
        
        # Run monitored session
        results = monitor_ironforge_session(
            test_session, 
            synthetic_processing,
            args.baseline
        )
        
        # Display results
        print("\n📊 Performance Test Results:")
        metrics = results['performance_metrics']
        print(f"   Processing time: {metrics.processing_time_ms:.1f}ms")
        print(f"   Memory usage: {metrics.memory_usage_mb:.1f}MB")
        print(f"   Pattern count: {metrics.pattern_count}")
        print(f"   Feature dimensions: {metrics.feature_dimensions}")
        print(f"   Edge types: {metrics.edge_types_count}")
        
        # Show Sprint 2 analysis
        sprint2_analysis = results['performance_report']['sprint2_analysis']
        print(f"\n🎯 Sprint 2 Analysis:")
        print(f"   Feature architecture: {sprint2_analysis['feature_architecture_status']}")
        print(f"   Edge types: {sprint2_analysis['edge_types_status']}")
        
        # Show regression analysis if available
        if 'regression_analysis' in results['performance_report']:
            regression = results['performance_report']['regression_analysis']
            if regression['regression_detected']:
                print(f"\n⚠️  Performance regression detected:")
                for reg in regression['regressions']:
                    print(f"   - {reg}")
            else:
                print(f"\n✅ No performance regression detected")
                if regression['improvements']:
                    for imp in regression['improvements']:
                        print(f"   + {imp}")
        
        # Save as baseline if requested
        if args.save_baseline:
            monitor = PerformanceMonitor()
            monitor.save_baseline_metrics(metrics, args.save_baseline)
            print(f"\n💾 Baseline saved to {args.save_baseline}")
    
    else:
        print("ℹ️  Use --test-mode to run synthetic performance test")
        print("ℹ️  Use with IRONFORGE orchestrator integration for real monitoring")
    
    return 0

if __name__ == "__main__":
    exit(main())
def create_graph_analysis(graph_builder, processed_graphs):
    """Create graph analysis for performance monitoring"""
    return {
        "graph_count": len(processed_graphs) if processed_graphs else 0,
        "builder_type": type(graph_builder).__name__,
        "analysis_timestamp": "2025-08-13T12:00:00"
    }

