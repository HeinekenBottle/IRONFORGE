#!/usr/bin/env python3
"""
Fix Test Import Paths Script
============================
Updates all test files to use the new ironforge package structure.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix import statements in a single file"""
    print(f"🔧 Fixing imports in {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix learning imports
    content = re.sub(
        r'from learning\.([a-zA-Z_]+) import',
        r'from ironforge.learning.\1 import',
        content
    )
    
    # Fix analysis imports
    content = re.sub(
        r'from analysis\.([a-zA-Z_]+) import',
        r'from ironforge.analysis.\1 import',
        content
    )
    
    # Fix synthesis imports
    content = re.sub(
        r'from synthesis\.([a-zA-Z_]+) import',
        r'from ironforge.synthesis.\1 import',
        content
    )
    
    # Fix integration imports
    content = re.sub(
        r'from integration\.([a-zA-Z_]+) import',
        r'from ironforge.integration.\1 import',
        content
    )
    
    # Fix reporting imports
    content = re.sub(
        r'from reporting\.([a-zA-Z_]+) import',
        r'from ironforge.reporting.\1 import',
        content
    )
    
    # Fix standalone module imports that moved to scripts
    content = re.sub(
        r'from performance_monitor import',
        r'from scripts.utilities.performance_monitor import',
        content
    )
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Updated {file_path}")
        return True
    else:
        print(f"  ⏭️  No changes needed in {file_path}")
        return False

def main():
    """Fix all test files"""
    print("🚨 CRITICAL FIX: Updating Test Import Paths")
    print("=" * 50)
    
    test_dir = Path("tests")
    if not test_dir.exists():
        print("❌ Tests directory not found!")
        return
    
    files_updated = 0
    total_files = 0
    
    # Find all Python files in tests directory
    for py_file in test_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        total_files += 1
        if fix_imports_in_file(py_file):
            files_updated += 1
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Total files scanned: {total_files}")
    print(f"  Files updated: {files_updated}")
    print(f"  Files unchanged: {total_files - files_updated}")
    
    if files_updated > 0:
        print("\n✅ Import fixes completed successfully!")
    else:
        print("\n⚠️  No files needed updating")

if __name__ == "__main__":
    main()
