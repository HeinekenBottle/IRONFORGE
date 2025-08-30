#!/usr/bin/env python3
"""
Advanced Code Consolidation Analysis
====================================

Identifies potential code consolidation opportunities.
"""

import os
from pathlib import Path
from collections import defaultdict
import re


def analyze_consolidation_opportunities():
    """Analyze code for consolidation opportunities."""
    print('🔍 Advanced Code Consolidation Analysis')
    print('='*50)

    # Look for similar function names that might be duplicates
    similar_functions = defaultdict(list)

    for py_file in Path('.').rglob('*.py'):
        # Skip protected paths
        path_str = str(py_file)
        if any(skip in path_str for skip in ['.git', 'venv', 'node_modules', '__pycache__']):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find function definitions
            func_matches = re.findall(r'def\s+(\w+)', content)
            for func_name in func_matches:
                # Group by similar patterns
                base_name = re.sub(r'_\d+$', '', func_name)  # Remove trailing numbers
                base_name = re.sub(r'_(test|example|demo)$', '', base_name)  # Remove suffixes
                
                if len(base_name) > 3:  # Skip very short names
                    similar_functions[base_name].append((func_name, str(py_file)))
        
        except Exception:
            continue

    # Find potential consolidation opportunities
    consolidation_candidates = []
    for base_name, functions in similar_functions.items():
        if len(functions) > 1:
            # Check if they're actually similar (not just coincidentally named)
            func_names = [f[0] for f in functions]
            if any(name != base_name for name in func_names):  # Has variations
                consolidation_candidates.append((base_name, functions))

    print(f'Potential consolidation opportunities: {len(consolidation_candidates)}')

    # Show top candidates
    for i, (base_name, functions) in enumerate(consolidation_candidates[:10]):
        print(f'\n{i+1}. Base pattern: {base_name}')
        for func_name, file_path in functions[:3]:  # Show first 3
            print(f'   - {func_name} in {file_path}')
        if len(functions) > 3:
            print(f'   ... and {len(functions) - 3} more')

    print(f'\n📊 Analysis complete. Found {len(consolidation_candidates)} consolidation opportunities.')
    
    return consolidation_candidates


def find_duplicate_imports():
    """Find duplicate import patterns."""
    print('\n🔍 Duplicate Import Analysis')
    print('='*40)
    
    import_patterns = defaultdict(list)
    
    for py_file in Path('.').rglob('*.py'):
        path_str = str(py_file)
        if any(skip in path_str for skip in ['.git', 'venv', 'node_modules', '__pycache__']):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find import statements
            import_matches = re.findall(r'^(import\s+\w+|from\s+\w+\s+import\s+\w+)', content, re.MULTILINE)
            for import_stmt in import_matches:
                import_patterns[import_stmt].append(str(py_file))
        
        except Exception:
            continue
    
    # Find commonly duplicated imports
    common_imports = [(stmt, files) for stmt, files in import_patterns.items() if len(files) > 5]
    common_imports.sort(key=lambda x: len(x[1]), reverse=True)
    
    print(f'Most common imports (top 5):')
    for i, (import_stmt, files) in enumerate(common_imports[:5]):
        print(f'{i+1}. "{import_stmt}" used in {len(files)} files')
    
    return common_imports


def find_large_files():
    """Find large files that might benefit from splitting."""
    print('\n🔍 Large File Analysis')
    print('='*30)
    
    large_files = []
    
    for py_file in Path('.').rglob('*.py'):
        path_str = str(py_file)
        if any(skip in path_str for skip in ['.git', 'venv', 'node_modules', '__pycache__']):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > 200:  # Files with more than 200 lines
                large_files.append((str(py_file), len(lines)))
        
        except Exception:
            continue
    
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f'Large files (>200 lines):')
    for i, (file_path, line_count) in enumerate(large_files[:10]):
        print(f'{i+1}. {file_path}: {line_count} lines')
    
    return large_files


if __name__ == "__main__":
    consolidation_candidates = analyze_consolidation_opportunities()
    common_imports = find_duplicate_imports()
    large_files = find_large_files()
    
    print(f'\n🎯 Summary:')
    print(f'- Consolidation opportunities: {len(consolidation_candidates)}')
    print(f'- Common import patterns: {len(common_imports)}')
    print(f'- Large files (>200 lines): {len(large_files)}')
