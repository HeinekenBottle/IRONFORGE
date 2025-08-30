#!/usr/bin/env python3
"""
IRONFORGE Unused Import Cleanup
===============================

Systematically removes unused imports while preserving Golden Invariants.
"""

import ast
import sys
from pathlib import Path
from typing import Set, List, Dict, Any
import argparse


class SafeImportCleaner:
    """Safe import cleaner that preserves Golden Invariants."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.protected_imports = {
            'EVENT_TYPES', 'EDGE_INTENTS', 'NODE_FEATURE_DIM_STANDARD',
            'NODE_FEATURE_DIM_HTF', 'EDGE_FEATURE_DIM', '__future__'
        }
        self.protected_modules = {
            'ironforge/analysis', 'ironforge/learning', 'ironforge/synthesis',
            'ironforge/validation', 'ironforge/utilities'
        }
        self.changes_made = 0
    
    def is_protected_file(self, file_path: Path) -> bool:
        """Check if file is protected from cleanup."""
        path_str = str(file_path.relative_to(self.project_root))
        
        # Protected directories
        protected_dirs = ['runs/', 'data/', 'models/', 'configs/', '.git/']
        for protected in protected_dirs:
            if path_str.startswith(protected):
                return True
        
        # Protected modules
        for protected_module in self.protected_modules:
            if path_str.startswith(protected_module):
                return True
        
        return False
    
    def analyze_imports(self, file_path: Path) -> Dict[str, Any]:
        """Analyze imports in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            imports = []
            import_lines = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.name
                        imports.append(import_name)
                        import_lines[import_name] = node.lineno
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            import_name = f"{node.module}.{alias.name}"
                            imports.append(import_name)
                            import_lines[import_name] = node.lineno
            
            return {
                'imports': imports,
                'import_lines': import_lines,
                'content': content,
                'tree': tree
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def is_import_used(self, import_name: str, content: str) -> bool:
        """Check if an import is actually used in the file content."""
        # Skip protected imports
        for protected in self.protected_imports:
            if protected in import_name:
                return True
        
        # Get the base name (last part after dot)
        base_name = import_name.split('.')[-1]
        
        # Skip special imports
        if base_name.startswith('_') or base_name in ['main', 'test']:
            return True
        
        # Count occurrences (should be > 1 if used beyond import)
        occurrences = content.count(base_name)
        
        # Also check for attribute access patterns
        attr_patterns = [f"{base_name}.", f"{base_name}(", f"{base_name}["]
        attr_usage = any(pattern in content for pattern in attr_patterns)
        
        return occurrences > 1 or attr_usage
    
    def clean_file_imports(self, file_path: Path, dry_run: bool = True) -> Dict[str, Any]:
        """Clean unused imports from a file."""
        if self.is_protected_file(file_path):
            return {'protected': True}
        
        analysis = self.analyze_imports(file_path)
        if 'error' in analysis:
            return analysis
        
        unused_imports = []
        for import_name in analysis['imports']:
            if not self.is_import_used(import_name, analysis['content']):
                unused_imports.append(import_name)
        
        if not unused_imports:
            return {'unused_imports': [], 'changes_made': False}
        
        if not dry_run:
            # Actually remove the imports (simplified approach)
            lines = analysis['content'].split('\n')
            lines_to_remove = set()
            
            for import_name in unused_imports:
                if import_name in analysis['import_lines']:
                    line_no = analysis['import_lines'][import_name] - 1  # 0-based
                    if line_no < len(lines):
                        # Simple check: if line contains only this import, remove it
                        line = lines[line_no].strip()
                        if import_name.split('.')[-1] in line and ('import' in line):
                            lines_to_remove.add(line_no)
            
            # Remove lines (in reverse order to maintain indices)
            for line_no in sorted(lines_to_remove, reverse=True):
                del lines[line_no]
                self.changes_made += 1
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        
        return {
            'unused_imports': unused_imports,
            'changes_made': not dry_run and len(unused_imports) > 0,
            'lines_removed': len(unused_imports) if not dry_run else 0
        }
    
    def clean_project(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean unused imports from entire project."""
        results = {
            'files_processed': 0,
            'files_changed': 0,
            'total_imports_removed': 0,
            'protected_files_skipped': 0,
            'errors': 0,
            'file_results': {}
        }
        
        # Process all Python files
        for py_file in self.project_root.rglob('*.py'):
            results['files_processed'] += 1
            
            file_result = self.clean_file_imports(py_file, dry_run)
            results['file_results'][str(py_file)] = file_result
            
            if file_result.get('protected'):
                results['protected_files_skipped'] += 1
            elif file_result.get('error'):
                results['errors'] += 1
            elif file_result.get('changes_made'):
                results['files_changed'] += 1
                results['total_imports_removed'] += file_result.get('lines_removed', 0)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Clean unused imports from IRONFORGE")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be changed without making changes")
    parser.add_argument("--execute", action="store_true",
                       help="Actually make the changes (overrides --dry-run)")
    parser.add_argument("--project-root", type=str, default=".",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    dry_run = not args.execute
    
    if not project_root.exists():
        print(f"❌ Project root not found: {project_root}")
        sys.exit(1)
    
    cleaner = SafeImportCleaner(project_root)
    
    print(f"🧹 IRONFORGE Import Cleanup")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"Project: {project_root}")
    print("=" * 50)
    
    results = cleaner.clean_project(dry_run)
    
    # Print summary
    print(f"\n📊 Cleanup Summary:")
    print(f"Files processed: {results['files_processed']}")
    print(f"Protected files skipped: {results['protected_files_skipped']}")
    print(f"Files with changes: {results['files_changed']}")
    print(f"Total imports removed: {results['total_imports_removed']}")
    print(f"Errors: {results['errors']}")
    
    # Show sample changes
    if results['files_changed'] > 0:
        print(f"\n🔍 Sample Changes:")
        count = 0
        for file_path, file_result in results['file_results'].items():
            if file_result.get('changes_made') and count < 5:
                unused = file_result.get('unused_imports', [])
                print(f"  {file_path}: {len(unused)} imports removed")
                for imp in unused[:3]:  # Show first 3
                    print(f"    - {imp}")
                if len(unused) > 3:
                    print(f"    ... and {len(unused) - 3} more")
                count += 1
    
    if dry_run:
        print(f"\n💡 Run with --execute to make actual changes")
    else:
        print(f"\n✅ Changes applied successfully!")


if __name__ == "__main__":
    main()
