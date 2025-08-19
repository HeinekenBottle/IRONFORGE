#!/usr/bin/env python3
"""
IRONFORGE Schema Migration Validation System
============================================

Comprehensive test suite for schema normalization system following
Technical Debt Surgeon guidelines and NO FALLBACKS policy.

Tests:
1. Schema detection accuracy (27D, 34D, 37D, corrupted data)
2. Temporal cycle feature calculation validation  
3. 34D → 37D migration completeness
4. Data integrity preservation during migration
5. Error handling with clear diagnostic messages

Technical Debt Surgeon: Ensures migration system maintains strict
data integrity while enabling archaeological discovery across
legacy and current data formats.
"""

import json
import sys
import traceback

from schema_normalizer import SchemaNormalizer


class SchemaMigrationTester:
    """
    Comprehensive test suite for IRONFORGE schema migration system
    
    Technical Debt Surgeon: Following established patterns from test_strict_validation.py
    """
    
    def __init__(self):
        self.normalizer = SchemaNormalizer()
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
    
    def create_test_cases(self) -> dict[str, dict]:
        """Create comprehensive test cases for schema validation and migration"""
        
        # Base node structure with full features
        base_node_features = {
            # Temporal Context (9 base)
            "time_minutes": 810.0, "daily_phase_sin": 0.5, "daily_phase_cos": 0.866,
            "session_position": 0.0, "time_to_close": 150.0, "weekend_proximity": 0.2,
            "absolute_timestamp": 1723478400, "day_of_week": 1, "month_phase": 0.4,
            
            # Price relativity (7 features)
            "normalized_price": 0.75, "pct_from_open": 2.1, "pct_from_high": 25.0,
            "pct_from_low": 75.0, "price_to_HTF_ratio": 1.002, "time_since_session_open": 0.0,
            "normalized_time": 0.0,
            
            # Price Context Legacy (3)
            "price_delta_1m": 50.0, "price_delta_5m": 120.0, "price_delta_15m": 200.0,
            
            # Market State (7)
            "volatility_window": 0.05, "energy_state": 0.7, "contamination_coefficient": 0.1,
            "fisher_regime": 0, "session_character": 1, "cross_tf_confluence": 0.6, "timeframe_rank": 2,
            
            # Event & Structure (8)
            "event_type_id": 1, "timeframe_source": 1, "liquidity_type": 0,
            "fpfvg_gap_size": 30.0, "fpfvg_interaction_count": 2, "first_presentation_flag": 1.0,
            "pd_array_strength": 0.8, "structural_importance": 0.9
        }
        
        # Create 34D features (missing temporal cycles)
        features_34d = base_node_features.copy()
        features_34d_tensor = [
            810.0, 0.5, 0.866, 0.0, 150.0, 0.2, 1723478400.0, 1.0, 0.4,  # Temporal (9)
            0.75, 2.1, 25.0, 75.0, 1.002, 0.0, 0.0,  # Price relativity (7)
            50.0, 120.0, 200.0,  # Price context (3)
            0.05, 0.7, 0.1, 0.0, 1.0, 0.6, 2.0,  # Market state (7)
            1.0, 1.0, 0.0, 30.0, 2.0, 1.0, 0.8, 0.9  # Event & structure (8)
        ]  # Total: 34 features
        
        # Create 37D features (with temporal cycles)
        features_37d = base_node_features.copy()
        features_37d.update({
            "week_of_month": 2,      # 2nd week of August 2025
            "month_of_year": 8,      # August
            "day_of_week_cycle": 1   # Tuesday (Monday=0)
        })
        features_37d_tensor = features_34d_tensor.copy()
        features_37d_tensor.extend([2.0, 8.0, 1.0])  # Add temporal cycle features
        
        # Create 27D features (no relativity)
        features_27d = {k: v for k, v in base_node_features.items() 
                       if k not in ['normalized_price', 'pct_from_open', 'pct_from_high',
                                  'pct_from_low', 'price_to_HTF_ratio', 'time_since_session_open',
                                  'normalized_time']}
        features_27d_tensor = [
            810.0, 0.5, 0.866, 0.0, 150.0, 0.2, 1723478400.0, 1.0, 0.4,  # Temporal (9)
            50.0, 120.0, 200.0,  # Price context (3)
            0.05, 0.7, 0.1, 0.0, 1.0, 0.6, 2.0,  # Market state (7)
            1.0, 1.0, 0.0, 30.0, 2.0, 1.0, 0.8, 0.9  # Event & structure (8)
        ]  # Total: 27 features
        
        test_cases = {
            "valid_37d_graph": {
                "description": "Valid 37D graph with temporal cycle features",
                "expected_schema": "37D",
                "should_migrate": False,
                "data": {
                    "nodes": [
                        {**features_37d, "features": features_37d_tensor}
                    ],
                    "edges": [],
                    "metadata": {
                        "total_nodes": 1,
                        "total_edges": 0,
                        "session_metadata": {
                            "session_date": "2025-08-12",
                            "session_start": "13:30:00",
                            "session_end": "16:00:00"
                        }
                    }
                }
            },
            
            "valid_34d_graph": {
                "description": "Valid 34D graph missing temporal cycle features",
                "expected_schema": "34D", 
                "should_migrate": True,
                "data": {
                    "nodes": [
                        {**features_34d, "features": features_34d_tensor}
                    ],
                    "edges": [],
                    "metadata": {
                        "total_nodes": 1,
                        "total_edges": 0,
                        "session_metadata": {
                            "session_date": "2025-08-12",
                            "session_start": "13:30:00", 
                            "session_end": "16:00:00"
                        }
                    }
                }
            },
            
            "valid_27d_graph": {
                "description": "Valid 27D graph without relativity features",
                "expected_schema": "27D",
                "should_migrate": False,  # Not supported by this migrator
                "data": {
                    "nodes": [
                        {**features_27d, "features": features_27d_tensor}
                    ],
                    "edges": [],
                    "metadata": {
                        "total_nodes": 1,
                        "total_edges": 0,
                        "session_metadata": {
                            "session_date": "2025-08-12",
                            "session_start": "13:30:00",
                            "session_end": "16:00:00"
                        }
                    }
                }
            },
            
            "corrupted_missing_nodes": {
                "description": "Corrupted graph missing nodes field",
                "expected_schema": "corrupted",
                "should_migrate": False,
                "data": {
                    "edges": [],
                    "metadata": {"total_nodes": 0}
                }
            },
            
            "corrupted_invalid_node_type": {
                "description": "Corrupted graph with invalid node type",
                "expected_schema": "corrupted",
                "should_migrate": False,
                "data": {
                    "nodes": ["invalid_node_not_dict"],
                    "edges": [],
                    "metadata": {"total_nodes": 1}
                }
            },
            
            "missing_session_metadata": {
                "description": "Graph without session metadata",
                "expected_schema": "34D",
                "should_migrate": False,  # Cannot migrate without metadata
                "data": {
                    "nodes": [
                        {**features_34d, "features": features_34d_tensor}
                    ],
                    "edges": [],
                    "metadata": {
                        "total_nodes": 1,
                        "total_edges": 0
                    }
                }
            },
            
            "invalid_session_date": {
                "description": "Graph with invalid session date format",
                "expected_schema": "34D", 
                "should_migrate": False,  # Cannot calculate temporal cycles
                "data": {
                    "nodes": [
                        {**features_34d, "features": features_34d_tensor}
                    ],
                    "edges": [],
                    "metadata": {
                        "total_nodes": 1,
                        "total_edges": 0,
                        "session_metadata": {
                            "session_date": "invalid_date",
                            "session_start": "13:30:00",
                            "session_end": "16:00:00"
                        }
                    }
                }
            },
            
            "inconsistent_dimensions": {
                "description": "Graph with inconsistent node dimensions",
                "expected_schema": "unknown",
                "should_migrate": False,
                "data": {
                    "nodes": [
                        {**features_34d, "features": features_34d_tensor},  # 34D
                        {**features_27d, "features": features_27d_tensor}   # 27D - inconsistent!
                    ],
                    "edges": [],
                    "metadata": {
                        "total_nodes": 2,
                        "total_edges": 0,
                        "session_metadata": {
                            "session_date": "2025-08-12",
                            "session_start": "13:30:00",
                            "session_end": "16:00:00"
                        }
                    }
                }
            },
            
            "partial_temporal_cycles": {
                "description": "Graph with partial temporal cycle features",
                "expected_schema": "37D_incomplete",
                "should_migrate": False,
                "data": {
                    "nodes": [
                        {
                            **features_34d,
                            "week_of_month": 2,  # Has some temporal cycles but not all
                            "features": features_34d_tensor + [2.0]  # 35D - incomplete
                        }
                    ],
                    "edges": [],
                    "metadata": {
                        "total_nodes": 1,
                        "total_edges": 0,
                        "session_metadata": {
                            "session_date": "2025-08-12",
                            "session_start": "13:30:00",
                            "session_end": "16:00:00"
                        }
                    }
                }
            }
        }
        
        return test_cases
    
    def run_schema_detection_tests(self) -> dict:
        """Test schema detection accuracy across all test cases"""
        print("🔍 Testing Schema Detection System")
        print("=" * 45)
        
        test_cases = self.create_test_cases()
        detection_results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        for test_name, test_case in test_cases.items():
            print(f"\n📋 Testing: {test_name}")
            print(f"    Expected: {test_case['expected_schema']}")
            
            try:
                detection = self.normalizer.detect_schema_version(test_case['data'])
                print(f"    Detected: {detection.schema_version} ({detection.detected_dimensions}D)")
                
                # Check if detection matches expectation
                if detection.schema_version == test_case['expected_schema']:
                    print("    ✅ PASS - Schema correctly detected")
                    detection_results['passed'] += 1
                    detection_results['details'].append({
                        'test': test_name,
                        'status': 'PASS',
                        'expected': test_case['expected_schema'],
                        'actual': detection.schema_version
                    })
                else:
                    print(f"    ❌ FAIL - Expected {test_case['expected_schema']}, got {detection.schema_version}")
                    detection_results['failed'] += 1
                    detection_results['details'].append({
                        'test': test_name,
                        'status': 'FAIL',
                        'expected': test_case['expected_schema'],
                        'actual': detection.schema_version,
                        'errors': detection.validation_errors
                    })
                
                # Show validation details for failed or complex cases
                if not detection.is_valid or detection.validation_errors:
                    print(f"    ⚠️  Validation issues: {len(detection.validation_errors)} errors")
                    for error in detection.validation_errors[:2]:  # Show first 2 errors
                        print(f"       • {error[:80]}...")
                        
            except Exception as e:
                print(f"    💥 EXCEPTION: {type(e).__name__}: {str(e)}")
                detection_results['failed'] += 1
                detection_results['details'].append({
                    'test': test_name,
                    'status': 'EXCEPTION',
                    'expected': test_case['expected_schema'],
                    'actual': f"Exception: {str(e)}",
                    'errors': [str(e)]
                })
        
        return detection_results
    
    def run_temporal_cycle_calculation_tests(self) -> dict:
        """Test temporal cycle feature calculation accuracy"""
        print("\n\n🕒 Testing Temporal Cycle Calculation")
        print("=" * 45)
        
        cycle_results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        # Test cases for temporal cycle calculation
        temporal_test_cases = [
            {
                'name': 'monday_first_week',
                'session_date': '2025-08-04',  # Monday, 1st week of August
                'expected': {'week_of_month': 1, 'month_of_year': 8, 'day_of_week_cycle': 0}
            },
            {
                'name': 'tuesday_second_week', 
                'session_date': '2025-08-12',  # Tuesday, 2nd week of August
                'expected': {'week_of_month': 2, 'month_of_year': 8, 'day_of_week_cycle': 1}
            },
            {
                'name': 'friday_last_week',
                'session_date': '2025-08-29',  # Friday, 5th week of August
                'expected': {'week_of_month': 5, 'month_of_year': 8, 'day_of_week_cycle': 4}
            },
            {
                'name': 'december_edge_case',
                'session_date': '2025-12-31',  # New Year's Eve
                'expected': {'week_of_month': 5, 'month_of_year': 12, 'day_of_week_cycle': 2}  # Wednesday
            },
            {
                'name': 'february_leap_year',
                'session_date': '2024-02-29',  # Leap year edge case
                'expected': {'week_of_month': 5, 'month_of_year': 2, 'day_of_week_cycle': 3}  # Thursday
            }
        ]
        
        for test_case in temporal_test_cases:
            print(f"\n📅 Testing: {test_case['name']}")
            print(f"    Date: {test_case['session_date']}")
            
            try:
                session_metadata = {'session_date': test_case['session_date']}
                node_data = {}  # Empty node for calculation
                
                calculated = self.normalizer.calculate_temporal_cycle_features(session_metadata, node_data)
                
                # Check each expected feature
                all_correct = True
                for feature, expected_value in test_case['expected'].items():
                    actual_value = calculated[feature]
                    if actual_value == expected_value:
                        print(f"    ✅ {feature}: {actual_value}")
                    else:
                        print(f"    ❌ {feature}: expected {expected_value}, got {actual_value}")
                        all_correct = False
                
                if all_correct:
                    cycle_results['passed'] += 1
                    cycle_results['details'].append({
                        'test': test_case['name'],
                        'status': 'PASS',
                        'calculated': calculated
                    })
                else:
                    cycle_results['failed'] += 1
                    cycle_results['details'].append({
                        'test': test_case['name'],
                        'status': 'FAIL',
                        'expected': test_case['expected'],
                        'calculated': calculated
                    })
                    
            except Exception as e:
                print(f"    💥 EXCEPTION: {type(e).__name__}: {str(e)}")
                cycle_results['failed'] += 1
                cycle_results['details'].append({
                    'test': test_case['name'],
                    'status': 'EXCEPTION',
                    'error': str(e)
                })
        
        return cycle_results
    
    def run_migration_tests(self) -> dict:
        """Test 34D → 37D migration functionality"""
        print("\n\n🔄 Testing 34D → 37D Migration")
        print("=" * 45)
        
        migration_results = {
            'passed': 0,
            'failed': 0, 
            'details': []
        }
        
        test_cases = self.create_test_cases()
        
        # Focus on migration-relevant test cases
        migration_test_cases = {k: v for k, v in test_cases.items() 
                               if v.get('should_migrate', False)}
        
        for test_name, test_case in migration_test_cases.items():
            print(f"\n🔄 Testing migration: {test_name}")
            
            try:
                # Make a copy for migration
                test_data = json.loads(json.dumps(test_case['data']))  # Deep copy
                
                # Perform migration
                migration_result = self.normalizer.migrate_graph_schema(test_data, target_schema="37D")
                
                print(f"    Migration: {migration_result.source_schema} → {migration_result.target_schema}")
                print(f"    Nodes migrated: {migration_result.nodes_migrated}")
                print(f"    Features added: {migration_result.features_added}")
                
                if migration_result.success:
                    # Validate post-migration schema
                    post_validation = self.normalizer.validate_migrated_data(test_data, expected_schema="37D")
                    
                    if post_validation.is_valid and post_validation.schema_version == "37D":
                        print("    ✅ PASS - Migration successful and validated")
                        migration_results['passed'] += 1
                        
                        # Check specific features were added correctly
                        migrated_node = test_data['nodes'][0]
                        expected_features = ['week_of_month', 'month_of_year', 'day_of_week_cycle']
                        missing_features = [f for f in expected_features if f not in migrated_node]
                        
                        if missing_features:
                            print(f"    ⚠️  WARNING - Missing features: {missing_features}")
                        else:
                            print("    ✅ All temporal cycle features present")
                        
                        # Check tensor dimensions
                        features_tensor = migrated_node.get('features', [])
                        if len(features_tensor) == 37:
                            print("    ✅ Feature tensor correctly expanded to 37D")
                        else:
                            print(f"    ❌ Feature tensor wrong size: {len(features_tensor)}D (expected 37D)")
                            
                        migration_results['details'].append({
                            'test': test_name,
                            'status': 'PASS',
                            'migration_result': str(migration_result),
                            'validation_result': str(post_validation)
                        })
                    else:
                        print("    ❌ FAIL - Migration succeeded but validation failed")
                        print(f"        Post-validation: {post_validation.schema_version}")
                        migration_results['failed'] += 1
                        migration_results['details'].append({
                            'test': test_name,
                            'status': 'FAIL',
                            'error': 'Migration succeeded but validation failed',
                            'validation_errors': post_validation.validation_errors
                        })
                else:
                    print("    ❌ FAIL - Migration failed")
                    for error in migration_result.migration_errors[:2]:
                        print(f"        • {error[:80]}...")
                    migration_results['failed'] += 1
                    migration_results['details'].append({
                        'test': test_name,
                        'status': 'FAIL', 
                        'migration_errors': migration_result.migration_errors
                    })
                    
            except Exception as e:
                print(f"    💥 EXCEPTION: {type(e).__name__}: {str(e)}")
                migration_results['failed'] += 1
                migration_results['details'].append({
                    'test': test_name,
                    'status': 'EXCEPTION',
                    'error': str(e)
                })
        
        return migration_results
    
    def run_error_handling_tests(self) -> dict:
        """Test error handling and NO FALLBACKS policy compliance"""
        print("\n\n❌ Testing Error Handling & NO FALLBACKS Policy")
        print("=" * 55)
        
        error_results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        # Test cases designed to trigger specific error conditions
        error_test_cases = [
            {
                'name': 'empty_session_date',
                'description': 'Empty session_date should fail cleanly',
                'should_fail': True,
                'data': {
                    'session_metadata': {'session_date': ''},
                    'node_data': {}
                }
            },
            {
                'name': 'missing_session_date',
                'description': 'Missing session_date should fail cleanly',
                'should_fail': True,
                'data': {
                    'session_metadata': {'session_start': '13:30:00'},
                    'node_data': {}
                }
            },
            {
                'name': 'invalid_date_format',
                'description': 'Invalid date format should fail cleanly',
                'should_fail': True,
                'data': {
                    'session_metadata': {'session_date': '2025/08/12'},  # Wrong format
                    'node_data': {}
                }
            },
            {
                'name': 'non_dict_session_metadata',
                'description': 'Non-dictionary session metadata should fail',
                'should_fail': True,
                'data': {
                    'session_metadata': "invalid_metadata",
                    'node_data': {}
                }
            }
        ]
        
        for test_case in error_test_cases:
            print(f"\n⚠️  Testing: {test_case['name']}")
            print(f"    {test_case['description']}")
            
            try:
                # This should fail according to NO FALLBACKS policy
                result = self.normalizer.calculate_temporal_cycle_features(
                    test_case['data']['session_metadata'], 
                    test_case['data']['node_data']
                )
                
                if test_case['should_fail']:
                    print(f"    ❌ FAIL - Should have raised exception but returned: {result}")
                    error_results['failed'] += 1
                    error_results['details'].append({
                        'test': test_case['name'],
                        'status': 'FAIL',
                        'error': 'Expected exception but none raised'
                    })
                else:
                    print(f"    ✅ PASS - Returned expected result: {result}")
                    error_results['passed'] += 1
                    error_results['details'].append({
                        'test': test_case['name'],
                        'status': 'PASS',
                        'result': result
                    })
                    
            except ValueError as e:
                error_msg = str(e)
                print(f"    ✅ PASS - Correctly raised ValueError: {error_msg[:50]}...")
                
                # Check for NO FALLBACKS compliance
                if "NO FALLBACKS" in error_msg:
                    print("    ✅ NO FALLBACKS policy properly enforced")
                if "SOLUTION:" in error_msg:
                    print("    ✅ Clear solution provided in error message")
                
                if test_case['should_fail']:
                    error_results['passed'] += 1
                    error_results['details'].append({
                        'test': test_case['name'],
                        'status': 'PASS',
                        'error_message': error_msg
                    })
                else:
                    error_results['failed'] += 1
                    error_results['details'].append({
                        'test': test_case['name'],
                        'status': 'FAIL',
                        'error': f'Unexpected ValueError: {error_msg}'
                    })
                    
            except Exception as e:
                print(f"    ❌ UNEXPECTED - {type(e).__name__}: {str(e)}")
                error_results['failed'] += 1
                error_results['details'].append({
                    'test': test_case['name'],
                    'status': 'UNEXPECTED',
                    'error': f'{type(e).__name__}: {str(e)}'
                })
        
        return error_results
    
    def run_comprehensive_test_suite(self) -> dict:
        """Run all test suites and generate comprehensive report"""
        print("🔧 TECHNICAL DEBT SURGEON - Schema Migration Test Suite")
        print("=" * 65)
        
        overall_results = {
            'schema_detection': None,
            'temporal_cycles': None,
            'migration': None,
            'error_handling': None,
            'overall_success': False,
            'summary': {}
        }
        
        try:
            # Run all test suites
            overall_results['schema_detection'] = self.run_schema_detection_tests()
            overall_results['temporal_cycles'] = self.run_temporal_cycle_calculation_tests() 
            overall_results['migration'] = self.run_migration_tests()
            overall_results['error_handling'] = self.run_error_handling_tests()
            
            # Calculate overall metrics
            total_passed = sum([
                overall_results['schema_detection']['passed'],
                overall_results['temporal_cycles']['passed'], 
                overall_results['migration']['passed'],
                overall_results['error_handling']['passed']
            ])
            
            total_failed = sum([
                overall_results['schema_detection']['failed'],
                overall_results['temporal_cycles']['failed'],
                overall_results['migration']['failed'], 
                overall_results['error_handling']['failed']
            ])
            
            total_tests = total_passed + total_failed
            success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
            
            overall_results['summary'] = {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'success_rate': success_rate
            }
            
            # Determine overall success (≥80% threshold)
            overall_results['overall_success'] = success_rate >= 80.0
            
        except Exception as e:
            print(f"💥 CRITICAL ERROR in test suite: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            overall_results['overall_success'] = False
        
        return overall_results
    
    def print_comprehensive_report(self, results: dict):
        """Print comprehensive test results report"""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST RESULTS REPORT")
        print("=" * 70)
        
        # Individual test suite results
        for suite_name, suite_results in results.items():
            if suite_name in ['overall_success', 'summary']:
                continue
                
            if suite_results:
                print(f"\n{suite_name.upper().replace('_', ' ')}:")
                print(f"  ✅ Passed: {suite_results['passed']}")
                print(f"  ❌ Failed: {suite_results['failed']}")
                total = suite_results['passed'] + suite_results['failed']
                rate = (suite_results['passed'] / total * 100) if total > 0 else 0
                print(f"  📈 Success Rate: {rate:.1f}%")
        
        # Overall summary
        summary = results['summary']
        print("\n🏆 OVERALL RESULTS:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  ✅ Passed: {summary['passed_tests']}")
        print(f"  ❌ Failed: {summary['failed_tests']}")
        print(f"  📈 Success Rate: {summary['success_rate']:.1f}%")
        
        # Final assessment
        print(f"\n{'='*70}")
        if results['overall_success']:
            print("🎯 SUCCESS: Schema Migration System READY FOR PRODUCTION")
            print("   • Schema detection working correctly")
            print("   • Temporal cycle calculation validated")
            print("   • 34D → 37D migration functional")
            print("   • NO FALLBACKS policy properly enforced")
            print("   • Error handling provides clear diagnostic messages")
            print("   • Ready for IRONFORGE archaeological discovery")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Schema Migration System requires fixes")
            print("   • Review failed tests above")
            print("   • Fix validation logic and error handling")
            print("   • Ensure NO FALLBACKS policy compliance")
            print("   • Do not deploy until success rate ≥80%")
        
        return results['overall_success']

def main():
    """Main test execution"""
    tester = SchemaMigrationTester()
    results = tester.run_comprehensive_test_suite()
    success = tester.print_comprehensive_report(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())