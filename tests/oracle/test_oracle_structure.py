#!/usr/bin/env python3
"""
Test script for Oracle refactoring structure
Validates file structure and basic syntax without requiring external dependencies
"""

import ast
import sys
from pathlib import Path

def test_oracle_file_structure():
    """Test that all required Oracle files exist in the correct structure"""
    print("🧪 Testing Oracle File Structure...")
    
    base_path = Path("oracle")
    required_structure = {
        "__init__.py": "Main Oracle module",
        "core/__init__.py": "Core module init",
        "core/constants.py": "Centralized constants",
        "core/exceptions.py": "Exception hierarchy",
        "models/__init__.py": "Models module init", 
        "models/session.py": "Data models",
        "data/__init__.py": "Data module init",
        "data/audit.py": "Audit functionality",
        "data/session_mapping.py": "Session mapping",
        "evaluation/__init__.py": "Evaluation module init",
        "evaluation/evaluator.py": "Model evaluator"
    }
    
    missing_files = []
    for file_path, description in required_structure.items():
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path} - {description}")
        else:
            print(f"  ❌ {file_path} missing - {description}")
            missing_files.append(file_path)
            
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print("  ✅ All required files present")
        return True

def test_oracle_python_syntax():
    """Test that all Oracle Python files have valid syntax"""
    print("\n🧪 Testing Oracle Python Syntax...")
    
    base_path = Path("oracle")
    python_files = []
    
    # Collect all Python files
    for pattern in ["*.py", "*/*.py", "*/*/*.py"]:
        python_files.extend(base_path.glob(pattern))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse the AST to check syntax
            ast.parse(source_code, filename=str(file_path))
            print(f"  ✅ {file_path.relative_to(base_path)} syntax valid")
            
        except SyntaxError as e:
            print(f"  ❌ {file_path.relative_to(base_path)} syntax error: {e}")
            syntax_errors.append((str(file_path), str(e)))
        except Exception as e:
            print(f"  ❌ {file_path.relative_to(base_path)} error: {e}")
            syntax_errors.append((str(file_path), str(e)))
            
    if syntax_errors:
        print(f"  ❌ Syntax errors found in {len(syntax_errors)} files")
        return False
    else:
        print("  ✅ All Oracle files have valid Python syntax")
        return True

def test_oracle_class_definitions():
    """Test that required Oracle classes are defined in the correct files"""
    print("\n🧪 Testing Oracle Class Definitions...")
    
    expected_classes = {
        "core/constants.py": ["ValidationRules", "ReturnCodes", "OracleStatus"],
        "core/exceptions.py": ["OracleError", "AuditError", "SessionMappingError"],
        "models/session.py": ["SessionMetadata", "TrainingPair", "AuditResult", "TrainingManifest"],
        "data/audit.py": ["OracleAuditor"],
        "data/session_mapping.py": ["SessionMapper"],
        "evaluation/evaluator.py": ["OracleEvaluator"]
    }
    
    base_path = Path("oracle")
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
        print("  ✅ All required Oracle classes found")
        return True

def test_oracle_constants_centralization():
    """Test that constants are properly centralized"""
    print("\n🧪 Testing Oracle Constants Centralization...")
    
    constants_file = Path("oracle/core/constants.py")
    
    if not constants_file.exists():
        print("  ❌ Constants file not found")
        return False
        
    try:
        with open(constants_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for key constants
        required_constants = [
            'ORACLE_VERSION',
            'ERROR_CODES',
            'SESSION_TYPES',
            'MIN_NODES_THRESHOLD',
            'MIN_EDGES_THRESHOLD',
            'DEFAULT_EARLY_PCT',
            'TGAT_EMBEDDING_DIM'
        ]
        
        missing_constants = []
        for constant in required_constants:
            if constant in content:
                print(f"  ✅ {constant} defined in constants.py")
            else:
                print(f"  ❌ {constant} missing from constants.py")
                missing_constants.append(constant)
                
        if missing_constants:
            return False
            
        # Check for ERROR_CODES structure
        if 'ERROR_CODES = {' in content:
            print("  ✅ ERROR_CODES properly structured as dictionary")
        else:
            print("  ❌ ERROR_CODES not properly structured")
            return False
            
        # Check for SESSION_TYPES structure
        if 'SESSION_TYPES:' in content and 'Set[str]' in content:
            print("  ✅ SESSION_TYPES properly typed as Set[str]")
        else:
            print("  ❌ SESSION_TYPES not properly typed")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking constants: {e}")
        return False

def test_oracle_import_structure():
    """Test that Oracle import statements are properly structured"""
    print("\n🧪 Testing Oracle Import Structure...")
    
    # Test main __init__.py imports
    main_init = Path("oracle/__init__.py")
    if main_init.exists():
        try:
            with open(main_init, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for key imports
            required_imports = [
                'from .core import',
                'from .models import',
                'from .data import',
                'from .evaluation import'
            ]
            
            missing_imports = []
            for import_stmt in required_imports:
                if import_stmt in content:
                    print(f"  ✅ {import_stmt} found in main __init__.py")
                else:
                    print(f"  ❌ {import_stmt} missing from main __init__.py")
                    missing_imports.append(import_stmt)
                    
            if missing_imports:
                return False
                
            # Check for __all__ definition
            if '__all__ = [' in content:
                print("  ✅ __all__ properly defined in main __init__.py")
            else:
                print("  ❌ __all__ missing from main __init__.py")
                return False
                
        except Exception as e:
            print(f"  ❌ Error checking main __init__.py: {e}")
            return False
    else:
        print("  ❌ Main __init__.py not found")
        return False
        
    print("  ✅ Oracle import structure properly organized")
    return True

def test_oracle_file_sizes():
    """Test that Oracle files are reasonably sized"""
    print("\n🧪 Testing Oracle File Sizes...")
    
    base_path = Path("oracle")
    python_files = []
    
    # Collect all Python files
    for pattern in ["*.py", "*/*.py"]:
        python_files.extend(base_path.glob(pattern))
    
    size_issues = []
    total_lines = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_count = len(lines)
                total_lines += line_count
                
            relative_path = file_path.relative_to(base_path)
            
            if line_count == 0:
                print(f"  ❌ {relative_path} is empty")
                size_issues.append(str(relative_path))
            elif line_count > 800:
                print(f"  ⚠️  {relative_path} is large ({line_count} lines)")
            else:
                print(f"  ✅ {relative_path} ({line_count} lines)")
                
        except Exception as e:
            print(f"  ❌ Error reading {file_path.relative_to(base_path)}: {e}")
            size_issues.append(str(file_path.relative_to(base_path)))
            
    print(f"  📊 Total lines across all Oracle modules: {total_lines}")
    
    if size_issues:
        print(f"  ❌ Size issues with: {size_issues}")
        return False
    else:
        print("  ✅ All Oracle files have reasonable sizes")
        return True

def main():
    """Run all Oracle structure tests"""
    print("🚀 Oracle System Structure Test Suite")
    print("=" * 60)
    
    tests = [
        test_oracle_file_structure,
        test_oracle_python_syntax,
        test_oracle_class_definitions,
        test_oracle_constants_centralization,
        test_oracle_import_structure,
        test_oracle_file_sizes
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
        print("🎉 All structure tests passed! Oracle refactoring structure is correct.")
        print("\n✅ File structure is properly organized into submodules")
        print("✅ Python syntax is valid in all Oracle modules")
        print("✅ Required classes are present in correct files")
        print("✅ Constants are properly centralized")
        print("✅ Import statements are correctly structured")
        print("✅ File sizes are reasonable and well-distributed")
    else:
        print("⚠️  Some structure tests failed. Review the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)