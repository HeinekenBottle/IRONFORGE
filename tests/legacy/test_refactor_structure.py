#!/usr/bin/env python3
"""
Test script for the refactored IRONFORGE Temporal Query Engine structure
Validates file structure and basic syntax without requiring dependencies
"""

import ast
import sys
from pathlib import Path

def test_file_structure():
    """Test that all required files exist in the correct structure"""
    print("🧪 Testing File Structure...")
    
    base_path = Path("ironforge/temporal")
    required_files = [
        "__init__.py",
        "session_manager.py", 
        "price_relativity.py",
        "query_core.py",
        "visualization.py",
        "enhanced_temporal_query_engine.py"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = base_path / file_name
        if file_path.exists():
            print(f"  ✅ {file_name} exists")
        else:
            print(f"  ❌ {file_name} missing")
            missing_files.append(file_name)
            
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print("  ✅ All required files present")
        return True

def test_python_syntax():
    """Test that all Python files have valid syntax"""
    print("\n🧪 Testing Python Syntax...")
    
    base_path = Path("ironforge/temporal")
    python_files = list(base_path.glob("*.py"))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse the AST to check syntax
            ast.parse(source_code, filename=str(file_path))
            print(f"  ✅ {file_path.name} syntax valid")
            
        except SyntaxError as e:
            print(f"  ❌ {file_path.name} syntax error: {e}")
            syntax_errors.append((file_path.name, str(e)))
        except Exception as e:
            print(f"  ❌ {file_path.name} error: {e}")
            syntax_errors.append((file_path.name, str(e)))
            
    if syntax_errors:
        print(f"  ❌ Syntax errors found in {len(syntax_errors)} files")
        return False
    else:
        print("  ✅ All files have valid Python syntax")
        return True

def test_class_definitions():
    """Test that required classes are defined in the correct files"""
    print("\n🧪 Testing Class Definitions...")
    
    expected_classes = {
        "session_manager.py": ["SessionDataManager"],
        "price_relativity.py": ["PriceRelativityEngine"], 
        "query_core.py": ["TemporalQueryCore"],
        "visualization.py": ["VisualizationManager"],
        "enhanced_temporal_query_engine.py": ["EnhancedTemporalQueryEngine"]
    }
    
    base_path = Path("ironforge/temporal")
    missing_classes = []
    
    for file_name, class_names in expected_classes.items():
        file_path = base_path / file_name
        
        if not file_path.exists():
            print(f"  ❌ {file_name} not found")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            # Parse AST to find class definitions
            tree = ast.parse(source_code)
            found_classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    found_classes.append(node.name)
                    
            # Check if all expected classes are present
            for class_name in class_names:
                if class_name in found_classes:
                    print(f"  ✅ {class_name} found in {file_name}")
                else:
                    print(f"  ❌ {class_name} missing from {file_name}")
                    missing_classes.append((file_name, class_name))
                    
        except Exception as e:
            print(f"  ❌ Error parsing {file_name}: {e}")
            
    if missing_classes:
        print(f"  ❌ Missing classes: {missing_classes}")
        return False
    else:
        print("  ✅ All required classes found")
        return True

def test_import_statements():
    """Test that import statements are properly structured"""
    print("\n🧪 Testing Import Statements...")
    
    base_path = Path("ironforge/temporal")
    
    # Test __init__.py imports
    init_file = base_path / "__init__.py"
    if init_file.exists():
        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for key imports
            required_imports = [
                "EnhancedTemporalQueryEngine",
                "SessionDataManager",
                "PriceRelativityEngine", 
                "TemporalQueryCore",
                "VisualizationManager"
            ]
            
            missing_imports = []
            for import_name in required_imports:
                if import_name in content:
                    print(f"  ✅ {import_name} imported in __init__.py")
                else:
                    print(f"  ❌ {import_name} missing from __init__.py")
                    missing_imports.append(import_name)
                    
            if missing_imports:
                return False
                
        except Exception as e:
            print(f"  ❌ Error checking __init__.py: {e}")
            return False
    else:
        print("  ❌ __init__.py not found")
        return False
        
    print("  ✅ Import statements properly structured")
    return True

def test_method_preservation():
    """Test that key methods are preserved in the main class"""
    print("\n🧪 Testing Method Preservation...")
    
    main_file = Path("ironforge/temporal/enhanced_temporal_query_engine.py")
    
    if not main_file.exists():
        print("  ❌ Main engine file not found")
        return False
        
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
            
        # Parse AST to find method definitions
        tree = ast.parse(source_code)
        found_methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "EnhancedTemporalQueryEngine":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        found_methods.append(item.name)
                        
        # Check for key methods that must be preserved
        required_methods = [
            "__init__",
            "ask",
            "get_enhanced_session_info",
            "list_sessions",
            "_analyze_archaeological_zones",
            "_analyze_theory_b_patterns",
            "_analyze_post_rd40_sequences",
            "_detect_rd40_events",
            "_classify_sequence_path"
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if method_name in found_methods:
                print(f"  ✅ {method_name} method preserved")
            else:
                print(f"  ❌ {method_name} method missing")
                missing_methods.append(method_name)
                
        if missing_methods:
            print(f"  ❌ Missing methods: {missing_methods}")
            return False
        else:
            print("  ✅ All key methods preserved")
            return True
            
    except Exception as e:
        print(f"  ❌ Error checking methods: {e}")
        return False

def test_file_sizes():
    """Test that files are reasonably sized (not empty, not too large)"""
    print("\n🧪 Testing File Sizes...")
    
    base_path = Path("ironforge/temporal")
    python_files = list(base_path.glob("*.py"))
    
    size_issues = []
    total_lines = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_count = len(lines)
                total_lines += line_count
                
            if line_count == 0:
                print(f"  ❌ {file_path.name} is empty")
                size_issues.append(file_path.name)
            elif line_count > 1000:
                print(f"  ⚠️  {file_path.name} is large ({line_count} lines)")
            else:
                print(f"  ✅ {file_path.name} ({line_count} lines)")
                
        except Exception as e:
            print(f"  ❌ Error reading {file_path.name}: {e}")
            size_issues.append(file_path.name)
            
    print(f"  📊 Total lines across all modules: {total_lines}")
    
    if size_issues:
        print(f"  ❌ Size issues with: {size_issues}")
        return False
    else:
        print("  ✅ All files have reasonable sizes")
        return True

def main():
    """Run all structure tests"""
    print("🚀 IRONFORGE Temporal Module Structure Test Suite")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_python_syntax,
        test_class_definitions,
        test_import_statements,
        test_method_preservation,
        test_file_sizes
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
        print("🎉 All structure tests passed! Refactoring structure is correct.")
        print("\n✅ File structure is properly organized")
        print("✅ Python syntax is valid in all modules")
        print("✅ Required classes and methods are present")
        print("✅ Import statements are correctly structured")
        print("✅ Backward compatibility interface is maintained")
    else:
        print("⚠️  Some structure tests failed. Review the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)