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
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_oracle_constants():
    """Test that Oracle constants are properly centralized"""
    print("\n🧪 Testing Oracle Constants...")
    
    try:
        from oracle.core import ERROR_CODES, SESSION_TYPES, ORACLE_VERSION
        
        # Test error codes
        assert isinstance(ERROR_CODES, dict), "ERROR_CODES should be a dictionary"
        assert 'SUCCESS' in ERROR_CODES, "ERROR_CODES should contain SUCCESS"
        assert 'SHARD_NOT_FOUND' in ERROR_CODES, "ERROR_CODES should contain SHARD_NOT_FOUND"
        print("  ✅ Error codes structure valid")
        
        # Test session types
        assert isinstance(SESSION_TYPES, set), "SESSION_TYPES should be a set"
        assert 'MIDNIGHT' in SESSION_TYPES, "SESSION_TYPES should contain MIDNIGHT"
        assert 'NY_AM' in SESSION_TYPES, "SESSION_TYPES should contain NY_AM"
        print("  ✅ Session types structure valid")
        
        # Test version
        assert isinstance(ORACLE_VERSION, str), "ORACLE_VERSION should be a string"
        assert len(ORACLE_VERSION.split('.')) >= 2, "ORACLE_VERSION should have version format"
        print(f"  ✅ Oracle version: {ORACLE_VERSION}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Constants test failed: {e}")
        traceback.print_exc()
        return False

def test_oracle_exceptions():
    """Test that Oracle exceptions work correctly"""
    print("\n🧪 Testing Oracle Exceptions...")
    
    try:
        from oracle.core import (
            OracleError, AuditError, SessionMappingError,
            create_audit_error, create_session_mapping_error
        )
        
        # Test base exception
        base_error = OracleError("Test error", error_code="TEST_ERROR")
        assert str(base_error) == "[TEST_ERROR] Test error"
        print("  ✅ Base OracleError works correctly")
        
        # Test specific exceptions
        audit_error = AuditError("Audit failed")
        assert isinstance(audit_error, OracleError)
        print("  ✅ AuditError inheritance works")
        
        mapping_error = SessionMappingError("Mapping failed")
        assert isinstance(mapping_error, OracleError)
        print("  ✅ SessionMappingError inheritance works")
        
        # Test factory functions
        factory_error = create_audit_error("TEST_CODE", "session_123", "Test details")
        assert factory_error.error_code == "TEST_CODE"
        assert "session_123" in str(factory_error)
        print("  ✅ Exception factory functions work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Exceptions test failed: {e}")
        traceback.print_exc()
        return False

def test_oracle_data_models():
    """Test that Oracle data models work correctly"""
    print("\n🧪 Testing Oracle Data Models...")
    
    try:
        import numpy as np
        from datetime import datetime
        from oracle.models import SessionMetadata, TrainingPair, AuditResult
        
        # Test SessionMetadata
        metadata = SessionMetadata(
            session_id="MIDNIGHT_2025-08-05",
            symbol="NQ",
            timeframe="M5",
            session_type="MIDNIGHT",
            session_date="2025-08-05",
            node_count=100,
            edge_count=200
        )
        assert metadata.quality_score == 2  # "fair" quality level
        assert metadata.session_type == "MIDNIGHT"
        print("  ✅ SessionMetadata model works")
        
        # Test TrainingPair
        embedding = np.random.rand(44)  # TGAT embedding dimension
        training_pair = TrainingPair(
            session_id="MIDNIGHT_2025-08-05",
            symbol="NQ",
            timeframe="M5",
            session_date="2025-08-05",
            early_pct=0.2,
            early_embedding=embedding,
            target_center=100.0,
            target_half_range=50.0
        )
        assert training_pair.target_low == 50.0
        assert training_pair.target_high == 150.0
        print("  ✅ TrainingPair model works")
        
        # Test AuditResult
        audit_result = AuditResult(
            session_id="MIDNIGHT_2025-08-05",
            status="SUCCESS",
            metadata=metadata
        )
        assert audit_result.is_success
        assert audit_result.is_processable
        print("  ✅ AuditResult model works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data models test failed: {e}")
        traceback.print_exc()
        return False

def test_oracle_utilities():
    """Test Oracle utility functions"""
    print("\n🧪 Testing Oracle Utilities...")
    
    try:
        from oracle.core import normalize_timeframe, get_error_description, ValidationRules
        
        # Test timeframe normalization
        assert normalize_timeframe("5") == "M5"
        assert normalize_timeframe("M5") == "M5"
        assert normalize_timeframe("m15") == "M15"
        print("  ✅ Timeframe normalization works")
        
        # Test error description
        desc = get_error_description("SUCCESS")
        assert "processable" in desc.lower()
        print("  ✅ Error description lookup works")
        
        # Test validation rules
        assert ValidationRules.validate_early_pct(0.2) == True
        assert ValidationRules.validate_early_pct(1.5) == False
        assert ValidationRules.validate_session_type("MIDNIGHT") == True
        assert ValidationRules.validate_session_type("INVALID") == False
        print("  ✅ Validation rules work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utilities test failed: {e}")
        traceback.print_exc()
        return False

def test_oracle_main_interface():
    """Test the main Oracle interface"""
    print("\n🧪 Testing Oracle Main Interface...")
    
    try:
        from oracle import (
            get_oracle_info, list_available_components,
            create_oracle_auditor, create_session_mapper
        )
        
        # Test info function
        info = get_oracle_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "components" in info
        print("  ✅ Oracle info function works")
        
        # Test component listing
        components = list_available_components()
        assert isinstance(components, list)
        assert len(components) > 0
        assert any("OracleAuditor" in comp for comp in components)
        print(f"  ✅ Found {len(components)} available components")
        
        # Test factory functions
        auditor = create_oracle_auditor(verbose=False)
        assert auditor is not None
        print("  ✅ Oracle auditor factory works")
        
        mapper = create_session_mapper()
        assert mapper is not None
        print("  ✅ Session mapper factory works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Main interface test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test backward compatibility with legacy imports"""
    print("\n🧪 Testing Backward Compatibility...")
    
    try:
        # Test that main Oracle imports still work
        from oracle import ORACLE_VERSION, ERROR_CODES, OracleAuditor
        
        # Test that we can access the same constants as before
        assert isinstance(ERROR_CODES, dict)
        assert 'SUCCESS' in ERROR_CODES
        print("  ✅ Legacy constant access works")
        
        # Test that we can create components as before
        auditor = OracleAuditor(verbose=False)
        assert hasattr(auditor, 'audit_oracle_training_data')
        print("  ✅ Legacy component creation works")
        
        # Test that error codes are accessible
        from oracle import get_error_description
        desc = get_error_description('SUCCESS')
        assert isinstance(desc, str)
        print("  ✅ Legacy utility functions work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Oracle refactoring tests"""
    print("🚀 Oracle System Refactor Test Suite")
    print("=" * 60)
    
    tests = [
        test_oracle_imports,
        test_oracle_constants,
        test_oracle_exceptions,
        test_oracle_data_models,
        test_oracle_utilities,
        test_oracle_main_interface,
        test_backward_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Oracle refactoring successful.")
        print("\n✅ Modular architecture is working correctly")
        print("✅ All constants and utilities are centralized")
        print("✅ Exception hierarchy is properly structured")
        print("✅ Data models are functioning correctly")
        print("✅ Backward compatibility is maintained")
    else:
        print("⚠️  Some tests failed. Review the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)