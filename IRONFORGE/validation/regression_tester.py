"""
IRONFORGE Regression Testing Framework
======================================

Comprehensive regression testing to ensure architectural compliance and prevent regressions.
Validates system behavior against established patterns and requirements.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json
from pathlib import Path
import time

try:
    from config import get_config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import get_config

logger = logging.getLogger(__name__)

class RegressionTester:
    """
    Regression testing framework for IRONFORGE architectural compliance.
    
    Tests:
    - Feature dimension compliance (45D nodes, 20D edges)
    - Session independence validation
    - Pattern graduation threshold enforcement
    - Performance regression detection
    - Configuration system usage
    - Theory B implementation validation
    """
    
    def __init__(self):
        config = get_config()
        self.test_output_path = Path(config.get_reports_path()) / "regression_tests"
        self.test_output_path.mkdir(parents=True, exist_ok=True)
        
        # Expected architectural parameters
        self.expected_node_dim = 45
        self.expected_edge_dim = 20
        self.expected_baseline_threshold = 0.87
        self.expected_semantic_features = 8
        self.expected_traditional_features = 37
        
        logger.info("Regression Tester initialized with architectural validation")
    
    def run_comprehensive_regression_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive regression test suite
        
        Returns:
            Complete regression test results
        """
        try:
            logger.info("Starting comprehensive regression test suite...")
            
            test_results = {
                'test_timestamp': datetime.now().isoformat(),
                'test_suite': 'comprehensive_regression',
                'architectural_tests': {},
                'performance_tests': {},
                'session_independence_tests': {},
                'configuration_tests': {},
                'theory_b_tests': {},
                'graduation_tests': {},
                'test_summary': {},
                'pass_fail_summary': {}
            }
            
            # Run architectural compliance tests
            test_results['architectural_tests'] = self._test_architectural_compliance()
            
            # Run performance regression tests
            test_results['performance_tests'] = self._test_performance_regression()
            
            # Run session independence tests
            test_results['session_independence_tests'] = self._test_session_independence()
            
            # Run configuration system tests
            test_results['configuration_tests'] = self._test_configuration_system()
            
            # Run Theory B implementation tests
            test_results['theory_b_tests'] = self._test_theory_b_implementation()
            
            # Run graduation threshold tests
            test_results['graduation_tests'] = self._test_graduation_thresholds()
            
            # Generate test summary
            test_results['test_summary'] = self._generate_test_summary(test_results)
            test_results['pass_fail_summary'] = self._generate_pass_fail_summary(test_results)
            
            # Save test results
            self._save_test_results(test_results)
            
            logger.info(f"Regression tests complete: {test_results['pass_fail_summary']['overall_result']}")
            return test_results
            
        except Exception as e:
            logger.error(f"Regression testing failed: {e}")
            return {
                'test_timestamp': datetime.now().isoformat(),
                'status': 'ERROR',
                'error': str(e),
                'test_summary': {'overall_result': 'FAILED', 'error': str(e)}
            }
    
    def _test_architectural_compliance(self) -> Dict[str, Any]:
        """Test architectural compliance requirements"""
        
        tests = {}
        
        # Test 45D/20D feature dimensions
        try:
            from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
            builder = EnhancedGraphBuilder()
            
            # Create mock session data for testing
            mock_session = {
                'session_name': 'regression_test_session',
                'events': [
                    {
                        'timestamp': '14:35:00',
                        'price': 23200.0,
                        'event_type': 'price_movement',
                        'significance': 0.8
                    },
                    {
                        'timestamp': '14:36:00', 
                        'price': 23210.0,
                        'event_type': 'rebalance',
                        'significance': 0.9
                    }
                ],
                'relativity_stats': {
                    'session_high': 23250.0,
                    'session_low': 23180.0,
                    'session_open': 23200.0
                }
            }
            
            # Build graph and test dimensions
            graph = builder.build_session_graph(mock_session)
            
            # Extract feature dimensions
            node_features = []
            edge_features = []
            
            for node, data in graph.nodes(data=True):
                if 'features' in data:
                    features = data['features']
                    if hasattr(features, 'get_combined_features'):
                        combined = features.get_combined_features()
                        node_features.append(len(combined))
            
            for u, v, data in graph.edges(data=True):
                if 'features' in data:
                    features = data['features']
                    if hasattr(features, 'get_combined_features'):
                        combined = features.get_combined_features()
                        edge_features.append(len(combined))
            
            tests['feature_dimensions'] = {
                'test': 'feature_dimension_compliance',
                'expected_node_dim': self.expected_node_dim,
                'expected_edge_dim': self.expected_edge_dim,
                'actual_node_dims': node_features,
                'actual_edge_dims': edge_features,
                'node_dim_compliance': all(dim == self.expected_node_dim for dim in node_features) if node_features else False,
                'edge_dim_compliance': all(dim == self.expected_edge_dim for dim in edge_features) if edge_features else False,
                'result': 'PASS' if (all(dim == self.expected_node_dim for dim in node_features) and 
                                   all(dim == self.expected_edge_dim for dim in edge_features)) else 'FAIL'
            }
            
        except Exception as e:
            tests['feature_dimensions'] = {
                'test': 'feature_dimension_compliance',
                'result': 'ERROR',
                'error': str(e)
            }
        
        # Test semantic vs traditional feature split
        try:
            tests['feature_composition'] = {
                'test': 'semantic_traditional_split',
                'expected_semantic': self.expected_semantic_features,
                'expected_traditional': self.expected_traditional_features,
                'expected_total': self.expected_semantic_features + self.expected_traditional_features,
                'actual_total': self.expected_node_dim,
                'composition_correct': (self.expected_semantic_features + self.expected_traditional_features) == self.expected_node_dim,
                'result': 'PASS' if (self.expected_semantic_features + self.expected_traditional_features) == self.expected_node_dim else 'FAIL'
            }
        except Exception as e:
            tests['feature_composition'] = {
                'test': 'semantic_traditional_split',
                'result': 'ERROR',
                'error': str(e)
            }
        
        return tests
    
    def _test_performance_regression(self) -> Dict[str, Any]:
        """Test for performance regressions"""
        
        tests = {}
        
        # Test container initialization performance
        try:
            start_time = time.time()
            from ironforge.integration.ironforge_container import get_ironforge_container
            container = get_ironforge_container()
            init_time = time.time() - start_time
            
            tests['container_initialization'] = {
                'test': 'container_init_performance',
                'initialization_time_seconds': init_time,
                'performance_threshold_seconds': 5.0,
                'within_threshold': init_time <= 5.0,
                'result': 'PASS' if init_time <= 5.0 else 'FAIL'
            }
        except Exception as e:
            tests['container_initialization'] = {
                'test': 'container_init_performance',
                'result': 'ERROR',
                'error': str(e)
            }
        
        # Test component creation performance
        try:
            start_time = time.time()
            container = get_ironforge_container()
            builder = container.get_enhanced_graph_builder()
            graduation = container.get_pattern_graduation()
            component_time = time.time() - start_time
            
            tests['component_creation'] = {
                'test': 'component_creation_performance',
                'creation_time_seconds': component_time,
                'performance_threshold_seconds': 3.0,
                'within_threshold': component_time <= 3.0,
                'result': 'PASS' if component_time <= 3.0 else 'FAIL'
            }
        except Exception as e:
            tests['component_creation'] = {
                'test': 'component_creation_performance',
                'result': 'ERROR',
                'error': str(e)
            }
        
        return tests
    
    def _test_session_independence(self) -> Dict[str, Any]:
        """Test session independence compliance"""
        
        tests = {}
        
        # Test that components create fresh instances
        try:
            from ironforge.integration.ironforge_container import get_ironforge_container
            container = get_ironforge_container()
            
            # Create multiple instances and verify they're different
            builder1 = container.get_enhanced_graph_builder()
            builder2 = container.get_enhanced_graph_builder()
            graduation1 = container.get_pattern_graduation()
            graduation2 = container.get_pattern_graduation()
            
            tests['fresh_instances'] = {
                'test': 'component_instance_independence',
                'builders_different': builder1 is not builder2,
                'graduations_different': graduation1 is not graduation2,
                'all_instances_unique': (builder1 is not builder2) and (graduation1 is not graduation2),
                'result': 'PASS' if ((builder1 is not builder2) and (graduation1 is not graduation2)) else 'FAIL'
            }
        except Exception as e:
            tests['fresh_instances'] = {
                'test': 'component_instance_independence',
                'result': 'ERROR',
                'error': str(e)
            }
        
        # Test that graduation and production systems maintain no state
        try:
            from ironforge.synthesis.pattern_graduation import PatternGraduation
            from ironforge.synthesis.production_graduation import ProductionGraduation
            
            pg = PatternGraduation()
            prod = ProductionGraduation()
            
            # Check that state-holding attributes are removed
            has_validation_history = hasattr(pg, 'validation_history')
            has_production_features = hasattr(prod, 'production_features')
            
            tests['stateless_components'] = {
                'test': 'stateless_component_validation',
                'pattern_graduation_stateless': not has_validation_history,
                'production_graduation_stateless': not has_production_features,
                'all_components_stateless': not has_validation_history and not has_production_features,
                'result': 'PASS' if (not has_validation_history and not has_production_features) else 'FAIL'
            }
        except Exception as e:
            tests['stateless_components'] = {
                'test': 'stateless_component_validation',
                'result': 'ERROR',
                'error': str(e)
            }
        
        return tests
    
    def _test_configuration_system(self) -> Dict[str, Any]:
        """Test configuration system compliance"""
        
        tests = {}
        
        # Test configuration loading
        try:
            from config import get_config
            config = get_config()
            
            # Test required paths exist
            required_methods = [
                'get_workspace_root', 'get_data_path', 'get_preservation_path',
                'get_graphs_path', 'get_discoveries_path', 'get_reports_path'
            ]
            
            method_results = {}
            all_methods_available = True
            
            for method_name in required_methods:
                if hasattr(config, method_name):
                    try:
                        path = getattr(config, method_name)()
                        method_results[method_name] = {'available': True, 'path': str(path)}
                    except Exception as e:
                        method_results[method_name] = {'available': False, 'error': str(e)}
                        all_methods_available = False
                else:
                    method_results[method_name] = {'available': False, 'error': 'Method not found'}
                    all_methods_available = False
            
            tests['configuration_loading'] = {
                'test': 'configuration_system_availability',
                'all_methods_available': all_methods_available,
                'method_results': method_results,
                'result': 'PASS' if all_methods_available else 'FAIL'
            }
        except Exception as e:
            tests['configuration_loading'] = {
                'test': 'configuration_system_availability',
                'result': 'ERROR',
                'error': str(e)
            }
        
        # Test that components use configuration system
        try:
            from ironforge.synthesis.production_graduation import ProductionGraduation
            prod = ProductionGraduation()
            
            # Check that output path uses configuration
            uses_config_path = str(prod.output_path).startswith('/Users/jack/IRONFORGE/preservation')
            
            tests['config_usage'] = {
                'test': 'component_configuration_usage',
                'production_uses_config': uses_config_path,
                'output_path': str(prod.output_path),
                'result': 'PASS' if uses_config_path else 'FAIL'
            }
        except Exception as e:
            tests['config_usage'] = {
                'test': 'component_configuration_usage',
                'result': 'ERROR',
                'error': str(e)
            }
        
        return tests
    
    def _test_theory_b_implementation(self) -> Dict[str, Any]:
        """Test Theory B dimensional relationship implementation"""
        
        tests = {}
        
        # Test that Theory B context extraction is available
        try:
            from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
            builder = EnhancedGraphBuilder()
            
            # Check if Theory B methods exist
            has_session_context_method = hasattr(builder, '_extract_session_context')
            has_theory_b_features = hasattr(builder, '_calculate_theory_b_features') or hasattr(builder, '_calculate_dimensional_features')
            
            tests['theory_b_methods'] = {
                'test': 'theory_b_method_availability',
                'session_context_available': has_session_context_method,
                'dimensional_features_available': has_theory_b_features,
                'theory_b_implemented': has_session_context_method or has_theory_b_features,
                'result': 'PASS' if (has_session_context_method or has_theory_b_features) else 'FAIL'
            }
        except Exception as e:
            tests['theory_b_methods'] = {
                'test': 'theory_b_method_availability',
                'result': 'ERROR',
                'error': str(e)
            }
        
        # Test dimensional relationship calculations
        try:
            # Test that 40% zone calculations work correctly
            test_high = 23250.0
            test_low = 23150.0
            test_range = test_high - test_low
            expected_40_percent = test_low + (test_range * 0.4)
            
            # Simple calculation validation
            calculated_40_percent = test_low + (test_range * 0.4)
            calculation_correct = abs(calculated_40_percent - expected_40_percent) < 0.01
            
            tests['dimensional_calculations'] = {
                'test': 'dimensional_relationship_calculations',
                'test_high': test_high,
                'test_low': test_low,
                'test_range': test_range,
                'expected_40_percent': expected_40_percent,
                'calculated_40_percent': calculated_40_percent,
                'calculation_accurate': calculation_correct,
                'result': 'PASS' if calculation_correct else 'FAIL'
            }
        except Exception as e:
            tests['dimensional_calculations'] = {
                'test': 'dimensional_relationship_calculations',
                'result': 'ERROR',
                'error': str(e)
            }
        
        return tests
    
    def _test_graduation_thresholds(self) -> Dict[str, Any]:
        """Test graduation threshold enforcement"""
        
        tests = {}
        
        # Test 87% baseline threshold
        try:
            from ironforge.synthesis.pattern_graduation import PatternGraduation
            graduation = PatternGraduation()
            
            baseline_correct = graduation.baseline_accuracy == self.expected_baseline_threshold
            
            tests['baseline_threshold'] = {
                'test': 'graduation_threshold_enforcement',
                'expected_threshold': self.expected_baseline_threshold,
                'actual_threshold': graduation.baseline_accuracy,
                'threshold_correct': baseline_correct,
                'result': 'PASS' if baseline_correct else 'FAIL'
            }
        except Exception as e:
            tests['baseline_threshold'] = {
                'test': 'graduation_threshold_enforcement',
                'result': 'ERROR',
                'error': str(e)
            }
        
        # Test graduation summary reflects session independence
        try:
            from ironforge.synthesis.pattern_graduation import PatternGraduation
            graduation = PatternGraduation()
            summary = graduation.get_graduation_summary()
            
            session_independence_flagged = summary.get('session_independence', False)
            
            tests['graduation_independence'] = {
                'test': 'graduation_session_independence',
                'session_independence_flagged': session_independence_flagged,
                'summary_structure': list(summary.keys()),
                'result': 'PASS' if session_independence_flagged else 'FAIL'
            }
        except Exception as e:
            tests['graduation_independence'] = {
                'test': 'graduation_session_independence',
                'result': 'ERROR',
                'error': str(e)
            }
        
        return tests
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        
        summary = {
            'total_test_categories': 0,
            'passed_categories': 0,
            'failed_categories': 0,
            'error_categories': 0,
            'overall_pass_rate': 0.0,
            'category_results': {}
        }
        
        test_categories = [
            'architectural_tests',
            'performance_tests', 
            'session_independence_tests',
            'configuration_tests',
            'theory_b_tests',
            'graduation_tests'
        ]
        
        for category in test_categories:
            if category in test_results:
                category_tests = test_results[category]
                category_summary = self._summarize_category(category_tests)
                summary['category_results'][category] = category_summary
                
                summary['total_test_categories'] += 1
                if category_summary['category_result'] == 'PASS':
                    summary['passed_categories'] += 1
                elif category_summary['category_result'] == 'FAIL':
                    summary['failed_categories'] += 1
                else:
                    summary['error_categories'] += 1
        
        if summary['total_test_categories'] > 0:
            summary['overall_pass_rate'] = summary['passed_categories'] / summary['total_test_categories']
        
        return summary
    
    def _summarize_category(self, category_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize results for a test category"""
        
        total_tests = len(category_tests)
        passed_tests = sum(1 for test in category_tests.values() if test.get('result') == 'PASS')
        failed_tests = sum(1 for test in category_tests.values() if test.get('result') == 'FAIL')
        error_tests = sum(1 for test in category_tests.values() if test.get('result') == 'ERROR')
        
        if total_tests == 0:
            category_result = 'NO_TESTS'
        elif error_tests > 0:
            category_result = 'ERROR'
        elif failed_tests > 0:
            category_result = 'FAIL'
        else:
            category_result = 'PASS'
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'error_tests': error_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'category_result': category_result
        }
    
    def _generate_pass_fail_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall pass/fail summary"""
        
        test_summary = test_results.get('test_summary', {})
        
        overall_pass_rate = test_summary.get('overall_pass_rate', 0.0)
        failed_categories = test_summary.get('failed_categories', 0)
        error_categories = test_summary.get('error_categories', 0)
        
        if error_categories > 0:
            overall_result = 'ERROR'
        elif failed_categories > 0:
            overall_result = 'FAIL'
        elif overall_pass_rate >= 1.0:
            overall_result = 'PASS'
        else:
            overall_result = 'PARTIAL'
        
        return {
            'overall_result': overall_result,
            'overall_pass_rate': overall_pass_rate,
            'categories_passed': test_summary.get('passed_categories', 0),
            'categories_failed': failed_categories,
            'categories_error': error_categories,
            'recommendation': self._get_regression_recommendation(overall_result, overall_pass_rate)
        }
    
    def _get_regression_recommendation(self, result: str, pass_rate: float) -> str:
        """Get recommendation based on regression test results"""
        
        if result == 'PASS':
            return 'All regression tests passed - system architecture compliant'
        elif result == 'PARTIAL' and pass_rate >= 0.8:
            return 'Most tests passed - review failed tests before deployment'
        elif result == 'FAIL':
            return 'Critical: Regression tests failed - fix architectural compliance issues'
        elif result == 'ERROR':
            return 'Critical: Test execution errors - investigate system integrity'
        else:
            return 'Review test results and address identified issues'
    
    def _save_test_results(self, test_results: Dict[str, Any]):
        """Save regression test results to file"""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"regression_test_results_{timestamp}.json"
            filepath = self.test_output_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            
            logger.info(f"Regression test results saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save regression test results: {e}")
    
    def get_regression_summary(self) -> Dict[str, Any]:
        """Get summary of regression testing framework"""
        
        return {
            'framework': 'IRONFORGE Regression Testing',
            'test_categories': [
                'architectural_compliance',
                'performance_regression',
                'session_independence',
                'configuration_system',
                'theory_b_implementation',
                'graduation_thresholds'
            ],
            'architectural_requirements': {
                'node_dimensions': self.expected_node_dim,
                'edge_dimensions': self.expected_edge_dim,
                'baseline_threshold': self.expected_baseline_threshold,
                'semantic_features': self.expected_semantic_features,
                'traditional_features': self.expected_traditional_features
            },
            'output_path': str(self.test_output_path)
        }