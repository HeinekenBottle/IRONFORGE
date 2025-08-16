#!/usr/bin/env python3
"""
IRONFORGE Refactored Structure Test
==================================
Test script to verify the refactored directory structure and imports work correctly.
"""

import sys
import os
from pathlib import Path

def test_directory_structure():
    """Test that the new directory structure exists"""
    print("🏗️  Testing Directory Structure...")
    
    required_dirs = [
        'ironforge',
        'ironforge/learning',
        'ironforge/analysis', 
        'ironforge/synthesis',
        'ironforge/integration',
        'scripts',
        'scripts/analysis',
        'scripts/data_processing',
        'scripts/utilities',
        'data',
        'data/raw',
        'data/enhanced',
        'data/adapted',
        'data/discoveries',
        'tests',
        'tests/integration',
        'tests/unit',
        'reports',
        'docs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"  ✅ {dir_path}")
    
    if missing_dirs:
        print(f"  ❌ Missing directories: {missing_dirs}")
        return False
    
    print("  ✅ All required directories exist")
    return True

def test_package_imports():
    """Test that package imports work correctly"""
    print("\n📦 Testing Package Imports...")
    
    # Test iron_core import
    try:
        from iron_core import IRONContainer
        print("  ✅ iron_core import successful")
    except Exception as e:
        print(f"  ❌ iron_core import failed: {e}")
        return False
    
    # Test ironforge package import
    try:
        import ironforge
        print(f"  ✅ ironforge package import successful (v{ironforge.__version__})")
    except Exception as e:
        print(f"  ❌ ironforge package import failed: {e}")
        return False
    
    # Test integration container import
    try:
        from ironforge.integration.ironforge_container import IRONFORGEContainer
        print("  ✅ ironforge.integration.ironforge_container import successful")
    except Exception as e:
        print(f"  ❌ ironforge.integration.ironforge_container import failed: {e}")
        return False
    
    return True

def test_config_system():
    """Test the configuration system"""
    print("\n⚙️  Testing Configuration System...")
    
    try:
        from config import get_config
        config = get_config()
        
        # Test that config returns proper paths
        data_path = config.get_data_path()
        preservation_path = config.get_preservation_path()
        
        print(f"  ✅ Config system working")
        print(f"    Data path: {data_path}")
        print(f"    Preservation path: {preservation_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Config system failed: {e}")
        return False

def test_moved_files():
    """Test that files were moved correctly"""
    print("\n📁 Testing File Organization...")
    
    # Check that test files were moved
    test_files_moved = len(list(Path('tests/integration').glob('test_*.py'))) > 0
    if test_files_moved:
        print("  ✅ Test files moved to tests/integration/")
    else:
        print("  ❌ Test files not found in tests/integration/")
    
    # Check that scripts were moved
    analysis_scripts = len(list(Path('scripts/analysis').glob('*.py'))) > 0
    if analysis_scripts:
        print("  ✅ Analysis scripts moved to scripts/analysis/")
    else:
        print("  ❌ Analysis scripts not found in scripts/analysis/")
    
    # Check that data was moved
    data_files = len(list(Path('data/discoveries').glob('*.json'))) > 0
    if data_files:
        print("  ✅ Discovery data moved to data/discoveries/")
    else:
        print("  ❌ Discovery data not found in data/discoveries/")
    
    # Check that docs were moved
    docs_files = len(list(Path('docs').glob('*.md'))) > 0
    if docs_files:
        print("  ✅ Documentation moved to docs/")
    else:
        print("  ❌ Documentation not found in docs/")
    
    return test_files_moved and analysis_scripts and docs_files

def test_root_cleanup():
    """Test that root directory is clean"""
    print("\n🧹 Testing Root Directory Cleanup...")
    
    # Count loose files in root (excluding expected ones)
    expected_files = {
        'README.md', 'REFACTORING_PLAN.md', 'requirements.txt', 
        'config.py', 'orchestrator.py', '__init__.py',
        'test_refactored_structure.py'  # This test file
    }
    
    root_files = set()
    for item in Path('.').iterdir():
        if item.is_file() and not item.name.startswith('.'):
            root_files.add(item.name)
    
    unexpected_files = root_files - expected_files
    
    if unexpected_files:
        print(f"  ⚠️  Unexpected files in root: {unexpected_files}")
    else:
        print("  ✅ Root directory is clean")
    
    print(f"  📊 Root files: {len(root_files)} (expected: {len(expected_files)})")
    
    return len(unexpected_files) < 5  # Allow some flexibility

def main():
    """Run all tests"""
    print("🔬 IRONFORGE Refactored Structure Test Suite")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_package_imports,
        test_config_system,
        test_moved_files,
        test_root_cleanup
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test.__name__}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful!")
        return 0
    else:
        print("⚠️  Some tests failed. Review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
