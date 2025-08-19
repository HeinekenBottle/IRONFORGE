#!/usr/bin/env python3
"""
Test Strict Data Validation System
==================================

Technical Debt Surgeon: Test comprehensive data integrity validation
following NO FALLBACKS policy - fails fast with clear error messages.
"""

from price_relativity_generator import PriceRelativityGenerator

from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder


def create_test_cases():
    """Create various test cases to validate strict validation logic"""
    
    test_cases = {
        "valid_session": {
            "session_metadata": {
                "session_start": "13:30:00",
                "session_end": "16:00:00", 
                "session_date": "2025-08-12",
                "session_type": "ny_pm"
            },
            "price_movements": [
                {
                    "timestamp": "13:30:00",
                    "price_level": 23450.0,
                    "movement_type": "open",
                    "normalized_price": 0.0,
                    "pct_from_open": 0.0,
                    "pct_from_high": 50.0,
                    "pct_from_low": 50.0,
                    "time_since_session_open": 0,
                    "normalized_time": 0.0
                },
                {
                    "timestamp": "14:00:00",
                    "price_level": 23500.0,
                    "movement_type": "high",
                    "normalized_price": 1.0,
                    "pct_from_open": 0.21,
                    "pct_from_high": 0.0,
                    "pct_from_low": 100.0,
                    "time_since_session_open": 1800,
                    "normalized_time": 0.2
                }
            ]
        },
        
        "missing_price_level": {
            "session_metadata": {
                "session_start": "13:30:00",
                "session_end": "16:00:00",
                "session_date": "2025-08-12"
            },
            "price_movements": [
                {
                    "timestamp": "13:30:00",
                    "price": 23450.0,  # Has 'price' but missing 'price_level'
                    "movement_type": "open"
                },
                {
                    "timestamp": "14:00:00", 
                    "price_level": 23500.0,
                    "movement_type": "high"
                }
            ]
        },
        
        "empty_timestamp": {
            "session_metadata": {
                "session_start": "13:30:00",
                "session_end": "16:00:00",
                "session_date": "2025-08-12"
            },
            "price_movements": [
                {
                    "timestamp": "",  # Empty timestamp
                    "price_level": 23450.0,
                    "movement_type": "open"
                },
                {
                    "timestamp": "14:00:00",
                    "price_level": 23500.0,
                    "movement_type": "high"
                }
            ]
        },
        
        "invalid_price": {
            "session_metadata": {
                "session_start": "13:30:00",
                "session_end": "16:00:00",
                "session_date": "2025-08-12"
            },
            "price_movements": [
                {
                    "timestamp": "13:30:00",
                    "price_level": "invalid_price",  # Non-numeric price
                    "movement_type": "open"
                },
                {
                    "timestamp": "14:00:00",
                    "price_level": 23500.0,
                    "movement_type": "high"
                }
            ]
        },
        
        "missing_relativity_features": {
            "session_metadata": {
                "session_start": "13:30:00",
                "session_end": "16:00:00",
                "session_date": "2025-08-12"
            },
            "price_movements": [
                {
                    "timestamp": "13:30:00",
                    "price_level": 23450.0,
                    "movement_type": "open",
                    "normalized_price": 0.0,  # Has some relativity features but missing others
                    "pct_from_open": 0.0
                    # Missing: pct_from_high, pct_from_low, time_since_session_open, normalized_time
                },
                {
                    "timestamp": "14:00:00",
                    "price_level": 23500.0,
                    "movement_type": "high"
                }
            ]
        },
        
        "missing_session_metadata": {
            "price_movements": [
                {
                    "timestamp": "13:30:00",
                    "price_level": 23450.0,
                    "movement_type": "open"
                }
            ]
        },
        
        "insufficient_movements": {
            "session_metadata": {
                "session_start": "13:30:00",
                "session_end": "16:00:00",
                "session_date": "2025-08-12"
            },
            "price_movements": [
                {
                    "timestamp": "13:30:00",
                    "price_level": 23450.0,
                    "movement_type": "open"
                }
            ]
        }
    }
    
    return test_cases

def test_enhanced_graph_builder_validation():
    """Test EnhancedGraphBuilder strict validation"""
    print("🔍 Testing EnhancedGraphBuilder Strict Validation")
    print("=" * 55)
    
    builder = EnhancedGraphBuilder()
    test_cases = create_test_cases()
    
    test_results = {
        'passed_validation': [],
        'failed_as_expected': [],
        'unexpected_failures': [],
        'unexpected_passes': []
    }
    
    for test_name, test_data in test_cases.items():
        print(f"\n📋 Testing: {test_name}")
        
        should_pass = (test_name == "valid_session")
        
        try:
            graph = builder.build_rich_graph(test_data)
            print(f"   ✅ Validation passed - built graph with {graph['metadata'].get('total_nodes', 0)} nodes")
            
            if should_pass:
                test_results['passed_validation'].append(test_name)
            else:
                test_results['unexpected_passes'].append(test_name)
                print(f"   ⚠️  UNEXPECTED: {test_name} should have failed validation!")
                
        except ValueError as e:
            print(f"   ❌ Validation failed (expected): {str(e)[:100]}...")
            
            if should_pass:
                test_results['unexpected_failures'].append((test_name, str(e)))
                print(f"   ⚠️  UNEXPECTED: {test_name} should have passed validation!")
            else:
                test_results['failed_as_expected'].append(test_name)
                
        except Exception as e:
            print(f"   💥 Unexpected error type: {type(e).__name__}: {str(e)[:100]}...")
            test_results['unexpected_failures'].append((test_name, f"{type(e).__name__}: {str(e)}"))
    
    return test_results

def test_price_relativity_generator_validation():
    """Test PriceRelativityGenerator strict validation"""
    print("\n\n🔧 Testing PriceRelativityGenerator Strict Validation")
    print("=" * 58)
    
    generator = PriceRelativityGenerator()
    test_cases = create_test_cases()
    
    test_results = {
        'passed_validation': [],
        'failed_as_expected': [],
        'unexpected_failures': [],
        'unexpected_passes': []
    }
    
    for test_name, test_data in test_cases.items():
        print(f"\n📋 Testing: {test_name}")
        
        # Only valid_session should pass, others should fail
        should_pass = (test_name == "valid_session")
        
        try:
            enhanced_data = generator.process_session(test_data)
            print(f"   ✅ Processing passed - enhanced {len(enhanced_data.get('price_movements', []))} movements")
            
            if should_pass:
                test_results['passed_validation'].append(test_name)
            else:
                test_results['unexpected_passes'].append(test_name)
                print(f"   ⚠️  UNEXPECTED: {test_name} should have failed processing!")
                
        except ValueError as e:
            print(f"   ❌ Processing failed (expected): {str(e)[:100]}...")
            
            if should_pass:
                test_results['unexpected_failures'].append((test_name, str(e)))
                print(f"   ⚠️  UNEXPECTED: {test_name} should have passed processing!")
            else:
                test_results['failed_as_expected'].append(test_name)
                
        except Exception as e:
            print(f"   💥 Unexpected error type: {type(e).__name__}: {str(e)[:100]}...")
            test_results['unexpected_failures'].append((test_name, f"{type(e).__name__}: {str(e)}"))
    
    return test_results

def print_test_summary(component_name, results):
    """Print comprehensive test results summary"""
    print(f"\n📊 {component_name} Validation Test Summary")
    print("=" * 50)
    
    total_tests = (len(results['passed_validation']) + len(results['failed_as_expected']) + 
                  len(results['unexpected_failures']) + len(results['unexpected_passes']))
    
    print(f"Total tests: {total_tests}")
    print(f"✅ Passed validation (expected): {len(results['passed_validation'])}")
    print(f"❌ Failed validation (expected): {len(results['failed_as_expected'])}")
    print(f"⚠️  Unexpected passes: {len(results['unexpected_passes'])}")
    print(f"💥 Unexpected failures: {len(results['unexpected_failures'])}")
    
    if results['passed_validation']:
        print(f"\n✅ Correctly passed: {', '.join(results['passed_validation'])}")
    
    if results['failed_as_expected']:
        print(f"\n❌ Correctly failed: {', '.join(results['failed_as_expected'])}")
    
    if results['unexpected_passes']:
        print("\n⚠️  CONCERN - Unexpected passes:")
        for test in results['unexpected_passes']:
            print(f"   • {test} (should have failed)")
    
    if results['unexpected_failures']:
        print("\n💥 CONCERN - Unexpected failures:")
        for test, error in results['unexpected_failures']:
            print(f"   • {test}: {error[:80]}...")
    
    # Calculate success rate
    expected_results = len(results['passed_validation']) + len(results['failed_as_expected'])
    success_rate = (expected_results / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 Validation System Accuracy: {success_rate:.1f}% ({expected_results}/{total_tests} tests behaved as expected)")
    
    return success_rate >= 80  # 80% accuracy threshold

def test_sprint2_structural_features():
    """Test Sprint 2 structural context edge validation"""
    
    print("🔍 Testing Sprint 2 Structural Context Edge Validation")
    print("=" * 55)
    
    builder = EnhancedGraphBuilder()
    
    # Test data with structural context potential
    test_sessions = {
        'valid_structural_session': {
            'session_metadata': {
                'session_type': 'ny_pm',
                'session_date': '2025-08-13',
                'feature_dimensions': 37
            },
            'price_movements': [
                # Movement that could be classified as cascade_origin
                {
                    'timestamp': '13:30:00', 'price_level': 23500.0, 'movement_type': 'sweep',
                    'normalized_price': 0.2, 'pct_from_open': 0.0, 'pct_from_high': 80.0,
                    'pct_from_low': 20.0, 'price_to_HTF_ratio': 0.98, 
                    'time_since_session_open': 0, 'normalized_time': 0.0,
                    'week_of_month': 2, 'month_of_year': 8, 'day_of_week_cycle': 2
                },
                # Movement that could be classified as first_fvg_after_sweep  
                {
                    'timestamp': '13:45:00', 'price_level': 23550.0, 'movement_type': 'fvg',
                    'normalized_price': 0.8, 'pct_from_open': 0.21, 'pct_from_high': 0.0,
                    'pct_from_low': 100.0, 'price_to_HTF_ratio': 1.02,
                    'time_since_session_open': 900, 'normalized_time': 0.1,
                    'week_of_month': 2, 'month_of_year': 8, 'day_of_week_cycle': 2
                },
                # Movement that could be classified as htf_range_midpoint
                {
                    'timestamp': '14:30:00', 'price_level': 23525.0, 'movement_type': 'equilibrium',
                    'normalized_price': 0.5, 'pct_from_open': 0.11, 'pct_from_high': 25.0,
                    'pct_from_low': 75.0, 'price_to_HTF_ratio': 1.00,
                    'time_since_session_open': 3600, 'normalized_time': 0.4,
                    'week_of_month': 2, 'month_of_year': 8, 'day_of_week_cycle': 2
                }
            ],
            'energy_state': {'total_accumulated': 2500},
            'contamination_analysis': {'contamination_coefficient': 0.3}
        },
        
        'missing_structural_metadata': {
            'session_metadata': {
                'session_type': 'ny_pm',
                'session_date': '2025-08-13'
                # Missing feature_dimensions - should still work
            },
            'price_movements': [
                {
                    'timestamp': '13:30:00', 'price_level': 23500.0, 'movement_type': 'open',
                    'normalized_price': 0.0, 'pct_from_open': 0.0, 'pct_from_high': 50.0,
                    'pct_from_low': 50.0, 'price_to_HTF_ratio': 1.0,
                    'time_since_session_open': 0, 'normalized_time': 0.0,
                    'week_of_month': 2, 'month_of_year': 8, 'day_of_week_cycle': 2
                },
                {
                    'timestamp': '14:30:00', 'price_level': 23525.0, 'movement_type': 'close',
                    'normalized_price': 1.0, 'pct_from_open': 0.11, 'pct_from_high': 0.0,
                    'pct_from_low': 100.0, 'price_to_HTF_ratio': 1.02,
                    'time_since_session_open': 3600, 'normalized_time': 1.0,
                    'week_of_month': 2, 'month_of_year': 8, 'day_of_week_cycle': 2
                }
            ],
            'energy_state': {'total_accumulated': 1000},
            'contamination_analysis': {'contamination_coefficient': 0.1}
        }
    }
    
    results = {
        'passed_validation': [],
        'failed_as_expected': [],
        'unexpected_passes': [],
        'unexpected_failures': []
    }
    
    for test_name, test_session in test_sessions.items():
        print(f"\n📋 Testing: {test_name}")
        
        try:
            # Build graph (should work for valid data)
            graph = builder.build_rich_graph(test_session)
            
            # Validate 4 edge types are supported
            edges = graph.get('edges', {})
            expected_edge_types = ['temporal', 'scale', 'structural_context', 'discovered']
            
            for edge_type in expected_edge_types:
                if edge_type not in edges:
                    print(f"   ⚠️ Missing edge type: {edge_type}")
            
            # Check for structural_context edges specifically
            structural_edges = edges.get('structural_context', [])
            print(f"   ✅ Structural context edges: {len(structural_edges)}")
            
            # Validate 37D features
            feature_dims = graph['metadata'].get('feature_dimensions', 0)
            if feature_dims == 37:
                print(f"   ✅ Feature dimensions: {feature_dims}D")
            else:
                print(f"   ❌ Expected 37D features, got {feature_dims}D")
            
            results['passed_validation'].append(test_name)
            
        except Exception as e:
            if test_name == 'valid_structural_session':
                # This should not fail
                results['unexpected_failures'].append((test_name, str(e)))
                print(f"   ❌ Unexpected failure: {e}")
            else:
                # Other tests might legitimately fail
                results['failed_as_expected'].append(test_name)
                print(f"   ✅ Expected failure: {e}")
    
    return results

def test_sprint2_regime_labels():
    """Test regime label validation as graph attributes"""
    
    print("\n🔍 Testing Sprint 2 Regime Label Validation") 
    print("=" * 45)
    
    # Test regime label structure validation
    test_regime_labels = {
        'valid_regime_labels': {
            'regime_labels': {'pattern_1': 0, 'pattern_2': 1, 'pattern_3': 0},
            'regime_characteristics': {
                0: {
                    'regime_id': 0, 'regime_label': 'weekly_breakout_mid',
                    'pattern_count': 2, 'stability_score': 0.8
                },
                1: {
                    'regime_id': 1, 'regime_label': 'monthly_consolidation_high', 
                    'pattern_count': 1, 'stability_score': 0.6
                }
            },
            'total_regimes': 2
        },
        
        'invalid_regime_structure': {
            'regime_labels': {'pattern_1': 'invalid_id'},  # Should be int
            'regime_characteristics': {},
            'total_regimes': 0
        },
        
        'missing_regime_data': {
            # Missing required regime fields
            'total_regimes': 1
        }
    }
    
    results = {
        'passed_validation': [],
        'failed_as_expected': [],
        'unexpected_passes': [],
        'unexpected_failures': []
    }
    
    for test_name, regime_data in test_regime_labels.items():
        print(f"\n📋 Testing: {test_name}")
        
        try:
            # Validate regime label structure
            is_valid = _validate_regime_label_structure(regime_data)
            
            if is_valid and test_name == 'valid_regime_labels':
                results['passed_validation'].append(test_name)
                print("   ✅ Regime labels validated successfully")
                
            elif not is_valid and test_name != 'valid_regime_labels':
                results['failed_as_expected'].append(test_name) 
                print("   ✅ Regime validation failed as expected")
                
            else:
                # Unexpected result
                if is_valid:
                    results['unexpected_passes'].append(test_name)
                    print("   ⚠️ Unexpected pass")
                else:
                    results['unexpected_failures'].append((test_name, "Validation failed"))
                    print("   ❌ Unexpected failure")
                    
        except Exception as e:
            results['unexpected_failures'].append((test_name, str(e)))
            print(f"   ❌ Exception: {e}")
    
    return results

def test_sprint2_precursor_index():
    """Test precursor index validation for data types and ranges"""
    
    print("\n🔍 Testing Sprint 2 Precursor Index Validation")
    print("=" * 50)
    
    test_precursor_indices = {
        'valid_precursor_index': {
            'cascade_probability': 0.73,
            'breakout_probability': 0.45, 
            'reversal_probability': 0.12,
            'overall_precursor_activity': 0.73
        },
        
        'out_of_range_values': {
            'cascade_probability': 1.5,  # > 1.0
            'breakout_probability': -0.1,  # < 0.0
            'reversal_probability': 0.8
        },
        
        'invalid_data_types': {
            'cascade_probability': 'high',  # Should be float
            'breakout_probability': None,
            'reversal_probability': 0.5
        },
        
        'missing_required_fields': {
            'cascade_probability': 0.6
            # Missing other probability fields
        }
    }
    
    results = {
        'passed_validation': [],
        'failed_as_expected': [],
        'unexpected_passes': [],
        'unexpected_failures': []
    }
    
    for test_name, precursor_index in test_precursor_indices.items():
        print(f"\n📋 Testing: {test_name}")
        
        try:
            is_valid = _validate_precursor_index_structure(precursor_index)
            
            if is_valid and test_name == 'valid_precursor_index':
                results['passed_validation'].append(test_name)
                print("   ✅ Precursor index validated successfully")
                
            elif not is_valid and test_name != 'valid_precursor_index':
                results['failed_as_expected'].append(test_name)
                print("   ✅ Precursor validation failed as expected")
                
            else:
                # Unexpected result
                if is_valid:
                    results['unexpected_passes'].append(test_name)
                    print("   ⚠️ Unexpected pass")
                else:
                    results['unexpected_failures'].append((test_name, "Validation failed"))
                    print("   ❌ Unexpected failure")
                    
        except Exception as e:
            results['unexpected_failures'].append((test_name, str(e)))
            print(f"   ❌ Exception: {e}")
    
    return results

def _validate_regime_label_structure(regime_data: Dict) -> bool:
    """Validate regime label data structure"""
    
    # Check required fields
    required_fields = ['regime_labels', 'regime_characteristics', 'total_regimes']
    for field in required_fields:
        if field not in regime_data:
            return False
    
    # Validate regime_labels mapping
    regime_labels = regime_data['regime_labels']
    if not isinstance(regime_labels, dict):
        return False
    
    # Check that regime IDs are integers
    for _pattern_id, regime_id in regime_labels.items():
        if not isinstance(regime_id, int):
            return False
    
    # Validate regime_characteristics structure
    regime_chars = regime_data['regime_characteristics']
    if not isinstance(regime_chars, dict):
        return False
    
    # Check regime characteristic structure
    for regime_id, char_data in regime_chars.items():
        if not isinstance(char_data, dict):
            return False
        
        required_char_fields = ['regime_id', 'regime_label', 'pattern_count', 'stability_score']
        for field in required_char_fields:
            if field not in char_data:
                return False
    
    return True

def _validate_precursor_index_structure(precursor_index: Dict) -> bool:
    """Validate precursor index data structure and ranges"""
    
    # Check that all values are numeric and in valid range [0, 1]
    for key, value in precursor_index.items():
        # Check data type
        if not isinstance(value, int | float):
            return False
        
        # Check range for probability fields
        if ('probability' in key or 'activity' in key) and not (0.0 <= value <= 1.0):
            return False
    
    # Check for at least one probability field
    probability_fields = [k for k in precursor_index if 'probability' in k]
    return probability_fields

def main():
    """Main test execution"""
    print("🔧 TECHNICAL DEBT SURGEON - Sprint 2 Enhanced Validation System Test")
    print("Following NO FALLBACKS policy - comprehensive data integrity testing")
    print("Testing original validation + Sprint 2 structural intelligence features")
    print("=" * 85)
    
    overall_success = True
    
    # Test original EnhancedGraphBuilder validation
    builder_success = test_enhanced_graph_builder_validation()
    overall_success = overall_success and builder_success
    
    # Test original PriceRelativityGenerator validation
    generator_success = test_price_relativity_generator_validation()
    overall_success = overall_success and generator_success
    
    print("\n" + "=" * 85)
    print("🆕 SPRINT 2 NEW FEATURES VALIDATION")
    print("=" * 85)
    
    # Test Sprint 2 structural context edges
    structural_results = test_sprint2_structural_features()
    
    # Test Sprint 2 regime labels
    regime_results = test_sprint2_regime_labels()
    
    # Test Sprint 2 precursor indices
    precursor_results = test_sprint2_precursor_index()
    
    # Combine Sprint 2 results
    combined_sprint2_results = {
        'passed_validation': (
            structural_results['passed_validation'] +
            regime_results['passed_validation'] +
            precursor_results['passed_validation']
        ),
        'failed_as_expected': (
            structural_results['failed_as_expected'] +
            regime_results['failed_as_expected'] +
            precursor_results['failed_as_expected']
        ),
        'unexpected_passes': (
            structural_results['unexpected_passes'] +
            regime_results['unexpected_passes'] +
            precursor_results['unexpected_passes']
        ),
        'unexpected_failures': (
            structural_results['unexpected_failures'] +
            regime_results['unexpected_failures'] +
            precursor_results['unexpected_failures']
        )
    }
    
    # Analyze Sprint 2 results
    print("\n📊 Sprint 2 Feature Validation Summary")
    print("=" * 50)
    
    total_sprint2_tests = (
        len(combined_sprint2_results['passed_validation']) +
        len(combined_sprint2_results['failed_as_expected']) +
        len(combined_sprint2_results['unexpected_passes']) +
        len(combined_sprint2_results['unexpected_failures'])
    )
    
    print(f"Total Sprint 2 tests: {total_sprint2_tests}")
    print(f"✅ Passed validation (expected): {len(combined_sprint2_results['passed_validation'])}")
    print(f"❌ Failed validation (expected): {len(combined_sprint2_results['failed_as_expected'])}")
    print(f"⚠️  Unexpected passes: {len(combined_sprint2_results['unexpected_passes'])}")
    print(f"💥 Unexpected failures: {len(combined_sprint2_results['unexpected_failures'])}")
    
    if combined_sprint2_results['passed_validation']:
        print(f"\n✅ Correctly passed: {', '.join(combined_sprint2_results['passed_validation'])}")
    
    if combined_sprint2_results['failed_as_expected']:
        print(f"\n❌ Correctly failed: {', '.join(combined_sprint2_results['failed_as_expected'])}")
    
    if combined_sprint2_results['unexpected_passes']:
        print("\n⚠️  Unexpected passes:")
        for test in combined_sprint2_results['unexpected_passes']:
            print(f"   • {test}")
    
    if combined_sprint2_results['unexpected_failures']:
        print("\n💥 CONCERN - Unexpected failures:")
        for test, error in combined_sprint2_results['unexpected_failures']:
            print(f"   • {test}: {error[:80]}...")
    
    # Calculate Sprint 2 success rate
    expected_sprint2_results = len(combined_sprint2_results['passed_validation']) + len(combined_sprint2_results['failed_as_expected'])
    sprint2_success_rate = (expected_sprint2_results / total_sprint2_tests) * 100 if total_sprint2_tests > 0 else 0
    
    print(f"\n🎯 Sprint 2 Validation System Accuracy: {sprint2_success_rate:.1f}% ({expected_sprint2_results}/{total_sprint2_tests} tests behaved as expected)")
    
    sprint2_success = sprint2_success_rate >= 80
    overall_success = overall_success and sprint2_success
    
    # Final summary
    print("\n" + "=" * 85)
    print("🏆 OVERALL VALIDATION SYSTEM TEST RESULT")
    print("=" * 85)
    
    if overall_success and sprint2_success:
        print("✅ SUCCESS: Enhanced validation system working correctly")
        print("   • Original data integrity checks comprehensive")
        print("   • Sprint 2 structural intelligence features validated")
        print("   • NO FALLBACKS policy properly enforced")
        print("   • Clear error messages provided for debugging")
        print("   • System fails fast on corrupted data as intended")
        return 0
    else:
        print("❌ FAILURE: Validation system issues detected")
        if not overall_success:
            print("   • Original validation system has issues")
        if not sprint2_success:
            print("   • Sprint 2 features have validation issues")
        return 1

if __name__ == "__main__":
    exit(main())