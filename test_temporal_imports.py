#!/usr/bin/env python3
"""
Import tests for the refactored Enhanced Temporal Query Engine
Tests that all imports work correctly without external dependencies
"""

import sys
import traceback
from unittest.mock import Mock, patch

def test_core_imports():
    """Test that core modules can be imported"""
    print("🧪 Testing Core Module Imports...")
    
    try:
        # Mock external dependencies
        sys.modules['pandas'] = Mock()
        sys.modules['numpy'] = Mock()
        sys.modules['matplotlib'] = Mock()
        sys.modules['matplotlib.pyplot'] = Mock()
        sys.modules['seaborn'] = Mock()
        
        # Test individual module imports
        from ironforge.temporal.session_manager import SessionDataManager
        print("  ✅ SessionDataManager imported")
        
        from ironforge.temporal.price_relativity import PriceRelativityEngine
        print("  ✅ PriceRelativityEngine imported")
        
        from ironforge.temporal.query_core import TemporalQueryCore
        print("  ✅ TemporalQueryCore imported")
        
        from ironforge.temporal.visualization import VisualizationManager
        print("  ✅ VisualizationManager imported")
        
        from ironforge.temporal.enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
        print("  ✅ EnhancedTemporalQueryEngine imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Core imports failed: {e}")
        traceback.print_exc()
        return False

def test_main_module_import():
    """Test main module import"""
    print("\n🧪 Testing Main Module Import...")
    
    try:
        # Mock external dependencies
        sys.modules['pandas'] = Mock()
        sys.modules['numpy'] = Mock()
        sys.modules['matplotlib'] = Mock()
        sys.modules['matplotlib.pyplot'] = Mock()
        sys.modules['seaborn'] = Mock()
        
        from ironforge.temporal import EnhancedTemporalQueryEngine
        print("  ✅ Main module import successful")
        
        # Test that we can access module functions
        from ironforge.temporal import get_module_info, list_available_queries
        print("  ✅ Module utility functions accessible")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Main module import failed: {e}")
        traceback.print_exc()
        return False

def test_class_instantiation():
    """Test that classes can be instantiated"""
    print("\n🧪 Testing Class Instantiation...")
    
    try:
        # Mock external dependencies
        sys.modules['pandas'] = Mock()
        sys.modules['numpy'] = Mock()
        sys.modules['matplotlib'] = Mock()
        sys.modules['matplotlib.pyplot'] = Mock()
        sys.modules['seaborn'] = Mock()
        
        # Mock file system operations
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal.session_manager import SessionDataManager
            session_manager = SessionDataManager()
            print("  ✅ SessionDataManager instantiated")
            
            from ironforge.temporal.price_relativity import PriceRelativityEngine
            price_engine = PriceRelativityEngine()
            print("  ✅ PriceRelativityEngine instantiated")
            
            from ironforge.temporal.query_core import TemporalQueryCore
            query_core = TemporalQueryCore(session_manager, price_engine)
            print("  ✅ TemporalQueryCore instantiated")
            
            from ironforge.temporal.visualization import VisualizationManager
            viz_manager = VisualizationManager()
            print("  ✅ VisualizationManager instantiated")
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            engine = EnhancedTemporalQueryEngine()
            print("  ✅ EnhancedTemporalQueryEngine instantiated")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Class instantiation failed: {e}")
        traceback.print_exc()
        return False

def test_method_existence():
    """Test that required methods exist"""
    print("\n🧪 Testing Method Existence...")
    
    try:
        # Mock external dependencies
        sys.modules['pandas'] = Mock()
        sys.modules['numpy'] = Mock()
        sys.modules['matplotlib'] = Mock()
        sys.modules['matplotlib.pyplot'] = Mock()
        sys.modules['seaborn'] = Mock()
        
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            engine = EnhancedTemporalQueryEngine()
            
            # Test core methods
            required_methods = [
                'ask',
                'get_enhanced_session_info',
                'list_sessions',
                'display_results',
                'plot_results',
                '_analyze_archaeological_zones',
                '_analyze_theory_b_patterns',
                '_analyze_post_rd40_sequences',
                '_analyze_relative_positioning',
                '_search_patterns',
                '_analyze_liquidity_sweeps'
            ]
            
            missing_methods = []
            for method_name in required_methods:
                if hasattr(engine, method_name):
                    print(f"  ✅ {method_name} method exists")
                else:
                    print(f"  ❌ {method_name} method missing")
                    missing_methods.append(method_name)
            
            if missing_methods:
                print(f"  ❌ Missing methods: {missing_methods}")
                return False
            
            return True
            
    except Exception as e:
        print(f"  ❌ Method existence test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility_interface():
    """Test backward compatibility interface"""
    print("\n🧪 Testing Backward Compatibility Interface...")
    
    try:
        # Mock external dependencies
        sys.modules['pandas'] = Mock()
        sys.modules['numpy'] = Mock()
        sys.modules['matplotlib'] = Mock()
        sys.modules['matplotlib.pyplot'] = Mock()
        sys.modules['seaborn'] = Mock()
        
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal import EnhancedTemporalQueryEngine
            engine = EnhancedTemporalQueryEngine()
            
            # Test backward compatibility attributes
            required_attributes = [
                'sessions',
                'graphs', 
                'metadata',
                'session_stats',
                'session_manager',
                'price_engine',
                'query_core',
                'visualization'
            ]
            
            missing_attributes = []
            for attr_name in required_attributes:
                if hasattr(engine, attr_name):
                    print(f"  ✅ {attr_name} attribute exists")
                else:
                    print(f"  ❌ {attr_name} attribute missing")
                    missing_attributes.append(attr_name)
            
            if missing_attributes:
                print(f"  ❌ Missing attributes: {missing_attributes}")
                return False
            
            return True
            
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling improvements"""
    print("\n🧪 Testing Error Handling...")
    
    try:
        # Mock external dependencies
        sys.modules['pandas'] = Mock()
        sys.modules['numpy'] = Mock()
        sys.modules['matplotlib'] = Mock()
        sys.modules['matplotlib.pyplot'] = Mock()
        sys.modules['seaborn'] = Mock()
        
        with patch('glob.glob', return_value=[]), \
             patch('pathlib.Path.exists', return_value=False):
            
            from ironforge.temporal.session_manager import SessionDataManager
            session_manager = SessionDataManager()
            
            # Test error handling for invalid inputs
            try:
                result = session_manager.get_session_data("")
                print("  ❌ Should have raised error for empty session ID")
                return False
            except ValueError:
                print("  ✅ Proper error handling for empty session ID")
            
            try:
                result = session_manager.get_enhanced_session_info("")
                if "error" in result:
                    print("  ✅ Proper error handling for invalid session info request")
                else:
                    print("  ❌ Should have returned error for empty session ID")
                    return False
            except Exception:
                print("  ❌ Unexpected exception for session info")
                return False
            
            return True
            
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_module_utilities():
    """Test module utility functions"""
    print("\n🧪 Testing Module Utilities...")
    
    try:
        # Mock external dependencies
        sys.modules['pandas'] = Mock()
        sys.modules['numpy'] = Mock()
        sys.modules['matplotlib'] = Mock()
        sys.modules['matplotlib.pyplot'] = Mock()
        sys.modules['seaborn'] = Mock()
        
        from ironforge.temporal import get_module_info, list_available_queries, get_example_usage
        
        # Test module info
        module_info = get_module_info()
        assert isinstance(module_info, dict), "Module info should be a dictionary"
        print("  ✅ get_module_info() works")
        
        # Test available queries
        queries = list_available_queries()
        assert isinstance(queries, list), "Available queries should be a list"
        print("  ✅ list_available_queries() works")
        
        # Test example usage
        example = get_example_usage()
        assert isinstance(example, str), "Example usage should be a string"
        print("  ✅ get_example_usage() works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Module utilities test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all import and basic functionality tests"""
    print("🚀 Enhanced Temporal Query Engine Import Test Suite")
    print("=" * 70)
    
    tests = [
        test_core_imports,
        test_main_module_import,
        test_class_instantiation,
        test_method_existence,
        test_backward_compatibility_interface,
        test_error_handling,
        test_module_utilities
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            
    print(f"\n📊 Import Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All import tests passed! Critical issues have been resolved.")
        print("\n✅ All modules can be imported successfully")
        print("✅ Classes can be instantiated without missing dependencies")
        print("✅ Required methods and attributes exist")
        print("✅ Backward compatibility interface is preserved")
        print("✅ Error handling improvements are working")
        print("✅ Module utilities are functional")
    else:
        print("⚠️  Some import tests failed. Review the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
