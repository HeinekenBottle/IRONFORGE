#!/usr/bin/env python3
"""
Test script for the refactored Oracle system
Validates that the modular architecture maintains all functionality
"""

import sys
import traceback
from pathlib import Path

def test_oracle_imports():
    """Test that all Oracle modules can be imported correctly"""
    print("🧪 Testing Oracle Module Imports...")
    
    try:
        # Test core module imports
        from oracle.core import (
            ORACLE_VERSION, ERROR_CODES, SESSION_TYPES,
            OracleError, AuditError, SessionMappingError,
            get_error_description, normalize_timeframe
        )
        print("  ✅ Core module imported successfully")
        
        # Test models module imports
        from oracle.models import (
            SessionMetadata, TrainingPair, AuditResult,
            TrainingManifest, OraclePrediction, TrainingDataset
        )
        print("  ✅ Models module imported successfully")
        
        # Test data module imports
        from oracle.data import OracleAuditor, SessionMapper
        print("  ✅ Data module imported successfully")
        
        # Test evaluation module imports
        from oracle.evaluation import OracleEvaluator
        print("  ✅ Evaluation module imported successfully")
        
        # Test main Oracle module import
        from oracle import (
            ORACLE_VERSION as main_version,
            OracleAuditor as main_auditor,
            SessionMapper as main_mapper,
            get_oracle_info, list_available_components
        )
        print("  ✅ Main Oracle module imported successfully")
        
        # Verify version consistency
        assert ORACLE_VERSION == main_version, f"Version mismatch: {ORACLE_VERSION} != {main_version}"
        print("  ✅ Version consistency verified")
        
        print("🎉 All Oracle imports successful!")
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_oracle_auditor():
    """Test the Oracle Auditor functionality"""
    print("\n🧪 Testing Oracle Auditor...")
    
    try:
        from oracle.data import OracleAuditor
        from oracle.models import SessionMetadata
        
        # Create auditor instance
        auditor = OracleAuditor()
        print("  ✅ OracleAuditor instance created")
        
        # Test basic auditor functionality
        sample_metadata = SessionMetadata(
            session_id="test_session_001",
            timeframe="NY_AM",
            date="2025-08-22",
            quality_score=0.85,
            event_count=150,
            duration_seconds=1800
        )
        
        # Test audit functionality
        result = auditor.audit_session_metadata(sample_metadata)
        print(f"  ✅ Session audit completed: {result.status}")
        
        print("🎉 Oracle Auditor tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Auditor test failed: {e}")
        traceback.print_exc()
        return False

def test_session_mapper():
    """Test the Session Mapper functionality"""
    print("\n🧪 Testing Session Mapper...")
    
    try:
        from oracle.data import SessionMapper
        
        # Create mapper instance
        mapper = SessionMapper()
        print("  ✅ SessionMapper instance created")
        
        # Test mapping functionality
        session_data = {
            "timeframe": "NY_AM",
            "date": "2025-08-22",
            "events": []
        }
        
        mapped_session = mapper.map_session_data(session_data)
        print(f"  ✅ Session mapping completed: {mapped_session.session_id}")
        
        print("🎉 Session Mapper tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Session Mapper test failed: {e}")
        traceback.print_exc()
        return False

def test_oracle_evaluator():
    """Test the Oracle Evaluator functionality"""
    print("\n🧪 Testing Oracle Evaluator...")
    
    try:
        from oracle.evaluation import OracleEvaluator
        from oracle.models import TrainingDataset, OraclePrediction
        
        # Create evaluator instance
        evaluator = OracleEvaluator()
        print("  ✅ OracleEvaluator instance created")
        
        # Test evaluation functionality with mock data
        mock_dataset = TrainingDataset(
            name="test_dataset",
            version="1.0.0",
            samples=[],
            metadata={}
        )
        
        mock_prediction = OraclePrediction(
            session_id="test_session",
            prediction_score=0.75,
            confidence=0.85,
            timeframe="NY_AM"
        )
        
        evaluation_result = evaluator.evaluate_prediction(mock_prediction, mock_dataset)
        print(f"  ✅ Prediction evaluation completed: {evaluation_result}")
        
        print("🎉 Oracle Evaluator tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Oracle Evaluator test failed: {e}")
        traceback.print_exc()
        return False

def test_oracle_integration():
    """Test full Oracle system integration"""
    print("\n🧪 Testing Oracle Integration...")
    
    try:
        from oracle import get_oracle_info, list_available_components
        
        # Test Oracle info
        oracle_info = get_oracle_info()
        print(f"  ✅ Oracle info retrieved: {oracle_info['version']}")
        
        # Test component listing
        components = list_available_components()
        print(f"  ✅ Available components: {len(components)} found")
        
        # Verify expected components are available
        expected_components = ['auditor', 'mapper', 'evaluator']
        for component in expected_components:
            assert component in components, f"Expected component '{component}' not found"
        print("  ✅ All expected components are available")
        
        print("🎉 Oracle Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Oracle Integration test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all Oracle refactor tests"""
    print("🚀 Starting Oracle Refactor Test Suite")
    print("=" * 50)
    
    tests = [
        test_oracle_imports,
        test_oracle_auditor,
        test_session_mapper,
        test_oracle_evaluator,
        test_oracle_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test.__name__}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Oracle refactor tests PASSED!")
        return True
    else:
        print("⚠️  Some Oracle refactor tests FAILED!")
        return False

if __name__ == "__main__":
    # Add the project root to Python path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    success = run_all_tests()
    sys.exit(0 if success else 1)