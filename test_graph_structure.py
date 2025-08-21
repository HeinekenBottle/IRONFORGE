#!/usr/bin/env python3
"""
Structure test for Enhanced Graph Builder refactoring
Validates file structure and imports without external dependencies
"""

import sys
import ast
from pathlib import Path
from unittest.mock import Mock

def test_graph_module_structure():
    """Test that graph module structure is correct"""
    print("🧪 Testing Graph Module Structure...")
    
    base_path = Path("ironforge/learning/graph")
    required_files = {
        "__init__.py": "Module initialization",
        "node_features.py": "45D node features with ICT semantics",
        "edge_features.py": "20D edge features with temporal relationships",
        "graph_builder.py": "Enhanced graph builder core logic",
        "feature_utils.py": "Shared utilities and calculations"
    }
    
    missing_files = []
    for file_name, description in required_files.items():
        file_path = base_path / file_name
        if file_path.exists():
            print(f"  ✅ {file_name} - {description}")
        else:
            print(f"  ❌ {file_name} missing - {description}")
            missing_files.append(file_name)
            
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print("  ✅ All required graph module files present")
        return True

def test_python_syntax():
    """Test that all Python files have valid syntax"""
    print("\n🧪 Testing Python Syntax...")
    
    base_path = Path("ironforge/learning")
    python_files = []
    
    # Collect Python files
    python_files.extend(base_path.glob("enhanced_graph_builder.py"))
    python_files.extend(base_path.glob("graph/*.py"))
    
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
        print("  ✅ All Python files have valid syntax")
        return True

def test_class_definitions():
    """Test that required classes are defined in correct files"""
    print("\n🧪 Testing Class Definitions...")
    
    expected_classes = {
        "ironforge/learning/graph/node_features.py": ["RichNodeFeature", "NodeFeatureProcessor"],
        "ironforge/learning/graph/edge_features.py": ["RichEdgeFeature", "EdgeFeatureProcessor"],
        "ironforge/learning/graph/graph_builder.py": ["EnhancedGraphBuilder"],
        "ironforge/learning/graph/feature_utils.py": ["FeatureUtils"]
    }
    
    missing_classes = []
    
    for file_name, class_names in expected_classes.items():
        file_path = Path(file_name)
        
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
                    print(f"  ✅ {class_name} found in {file_path.name}")
                else:
                    print(f"  ❌ {class_name} missing from {file_path.name}")
                    missing_classes.append((file_name, class_name))
                    
        except Exception as e:
            print(f"  ❌ Error parsing {file_name}: {e}")
            
    if missing_classes:
        print(f"  ❌ Missing classes: {missing_classes}")
        return False
    else:
        print("  ✅ All required classes found")
        return True

def test_import_structure():
    """Test that import statements are properly structured"""
    print("\n🧪 Testing Import Structure...")
    
    # Test main __init__.py imports
    init_file = Path("ironforge/learning/graph/__init__.py")
    if init_file.exists():
        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for key imports
            required_imports = [
                'from .graph_builder import EnhancedGraphBuilder',
                'from .node_features import RichNodeFeature',
                'from .edge_features import RichEdgeFeature',
                'from .feature_utils import FeatureUtils'
            ]
            
            missing_imports = []
            for import_stmt in required_imports:
                if import_stmt in content:
                    print(f"  ✅ {import_stmt}")
                else:
                    print(f"  ❌ Missing: {import_stmt}")
                    missing_imports.append(import_stmt)
                    
            if missing_imports:
                return False
                
            # Check for __all__ definition
            if '__all__' in content:
                print("  ✅ __all__ properly defined")
            else:
                print("  ❌ __all__ missing")
                return False
                
        except Exception as e:
            print(f"  ❌ Error checking __init__.py: {e}")
            return False
    else:
        print("  ❌ __init__.py not found")
        return False
        
    # Test backward compatibility file
    compat_file = Path("ironforge/learning/enhanced_graph_builder.py")
    if compat_file.exists():
        try:
            with open(compat_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for backward compatibility imports
            if 'from .graph import' in content:
                print("  ✅ Backward compatibility imports found")
            else:
                print("  ❌ Backward compatibility imports missing")
                return False
                
        except Exception as e:
            print(f"  ❌ Error checking enhanced_graph_builder.py: {e}")
            return False
    else:
        print("  ❌ enhanced_graph_builder.py not found")
        return False
        
    print("  ✅ Import structure properly organized")
    return True

def test_method_preservation():
    """Test that key methods are preserved in the refactored structure"""
    print("\n🧪 Testing Method Preservation...")
    
    # Check for key methods in graph_builder.py
    graph_builder_file = Path("ironforge/learning/graph/graph_builder.py")
    if graph_builder_file.exists():
        try:
            with open(graph_builder_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            required_methods = [
                'build_session_graph',
                'validate_theory_b_compliance',
                'extract_features_for_tgat',
                '_extract_session_context'
            ]
            
            missing_methods = []
            for method in required_methods:
                if f'def {method}' in content:
                    print(f"  ✅ {method} method found")
                else:
                    print(f"  ❌ {method} method missing")
                    missing_methods.append(method)
                    
            if missing_methods:
                return False
                
        except Exception as e:
            print(f"  ❌ Error checking graph_builder.py: {e}")
            return False
    else:
        print("  ❌ graph_builder.py not found")
        return False
        
    print("  ✅ Key methods preserved in refactored structure")
    return True

def test_feature_dimensions():
    """Test that feature dimensions are correctly specified"""
    print("\n🧪 Testing Feature Dimensions...")
    
    # Check node features (45D)
    node_file = Path("ironforge/learning/graph/node_features.py")
    if node_file.exists():
        try:
            with open(node_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'torch.zeros(45' in content:
                print("  ✅ 45D node features correctly specified")
            else:
                print("  ❌ 45D node features not found")
                return False
                
            if '8 semantic + 37 traditional' in content:
                print("  ✅ Node feature breakdown documented")
            else:
                print("  ❌ Node feature breakdown not documented")
                
        except Exception as e:
            print(f"  ❌ Error checking node_features.py: {e}")
            return False
    
    # Check edge features (20D)
    edge_file = Path("ironforge/learning/graph/edge_features.py")
    if edge_file.exists():
        try:
            with open(edge_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'torch.zeros(20' in content:
                print("  ✅ 20D edge features correctly specified")
            else:
                print("  ❌ 20D edge features not found")
                return False
                
            if '3 semantic + 17 traditional' in content:
                print("  ✅ Edge feature breakdown documented")
            else:
                print("  ❌ Edge feature breakdown not documented")
                
        except Exception as e:
            print(f"  ❌ Error checking edge_features.py: {e}")
            return False
            
    print("  ✅ Feature dimensions correctly preserved")
    return True

def main():
    """Run all structure tests"""
    print("🚀 Enhanced Graph Builder Structure Test Suite")
    print("=" * 70)
    
    tests = [
        test_graph_module_structure,
        test_python_syntax,
        test_class_definitions,
        test_import_structure,
        test_method_preservation,
        test_feature_dimensions
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            
    print(f"\n📊 Structure Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! Graph Builder refactoring is correctly structured.")
        print("\n✅ Modular file structure properly organized")
        print("✅ Python syntax valid in all modules")
        print("✅ Required classes present in correct files")
        print("✅ Import statements correctly structured")
        print("✅ Key methods preserved in refactored structure")
        print("✅ Feature dimensions (45D/20D) correctly preserved")
    else:
        print("⚠️  Some structure tests failed. Review the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
