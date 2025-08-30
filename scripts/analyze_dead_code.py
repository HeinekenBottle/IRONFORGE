#!/usr/bin/env python3
"""
IRONFORGE Dead Code Analysis
============================

Identifies unused imports, functions, classes, and configuration parameters
while preserving Golden Invariants and protected components.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import json
import argparse
from collections import defaultdict


class IRONFORGEDeadCodeAnalyzer:
    """IRONFORGE-aware dead code analyzer."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.protected_modules = {
            'ironforge/analysis',
            'ironforge/learning', 
            'ironforge/synthesis',
            'ironforge/validation',
            'ironforge/utilities'
        }
        self.golden_invariants = {
            'EVENT_TYPES', 'EDGE_INTENTS', 'NODE_FEATURE_DIM_STANDARD',
            'NODE_FEATURE_DIM_HTF', 'EDGE_FEATURE_DIM'
        }
        
        # Track usage across codebase
        self.imports_used = defaultdict(set)
        self.functions_defined = defaultdict(set)
        self.functions_called = defaultdict(set)
        self.classes_defined = defaultdict(set)
        self.classes_used = defaultdict(set)
        self.config_keys_defined = set()
        self.config_keys_used = set()
    
    def is_protected_path(self, path: Path) -> bool:
        """Check if path is protected from cleanup."""
        path_str = str(path.relative_to(self.project_root))
        
        # Protected directories
        protected_dirs = [
            'runs/', 'data/', 'models/', 'configs/', 'agents/',
            '.git/', '.claude/', 'tests/_golden/'
        ]
        
        for protected in protected_dirs:
            if path_str.startswith(protected):
                return True
        
        # Protected modules (per mypy exclusions)
        for protected_module in self.protected_modules:
            if path_str.startswith(protected_module):
                return True
        
        return False
    
    def analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file for dead code."""
        if self.is_protected_path(file_path):
            return {'protected': True}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            analysis = {
                'imports': [],
                'functions': [],
                'classes': [],
                'unused_imports': [],
                'protected': False
            }
            
            # Analyze AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                        self.imports_used[str(file_path)].add(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            import_name = f"{node.module}.{alias.name}"
                            analysis['imports'].append(import_name)
                            self.imports_used[str(file_path)].add(import_name)
                
                elif isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    analysis['functions'].append(func_name)
                    self.functions_defined[str(file_path)].add(func_name)
                
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    analysis['classes'].append(class_name)
                    self.classes_defined[str(file_path)].add(class_name)
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        self.functions_called[str(file_path)].add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            self.classes_used[str(file_path)].add(node.func.value.id)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'protected': False}
    
    def analyze_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze configuration file for unused parameters."""
        if self.is_protected_path(file_path):
            return {'protected': True}
        
        try:
            if file_path.suffix in ['.yml', '.yaml']:
                import yaml
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                def extract_keys(obj, prefix=''):
                    keys = set()
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            full_key = f"{prefix}.{key}" if prefix else key
                            keys.add(full_key)
                            self.config_keys_defined.add(full_key)
                            keys.update(extract_keys(value, full_key))
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            keys.update(extract_keys(item, f"{prefix}[{i}]"))
                    return keys
                
                return {
                    'config_keys': list(extract_keys(config)),
                    'protected': False
                }
            
        except Exception as e:
            return {'error': str(e), 'protected': False}
        
        return {'protected': False}
    
    def find_unused_imports(self) -> Dict[str, List[str]]:
        """Find unused imports across the codebase."""
        unused_imports = {}
        
        for file_path, imports in self.imports_used.items():
            file_unused = []
            
            # Read file content to check usage
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for import_name in imports:
                    # Simple heuristic: check if import name appears in content
                    # (excluding the import line itself)
                    import_parts = import_name.split('.')
                    base_name = import_parts[-1]
                    
                    # Skip Golden Invariants
                    if base_name in self.golden_invariants:
                        continue
                    
                    # Count occurrences (should be > 1 if used beyond import)
                    occurrences = content.count(base_name)
                    if occurrences <= 1:  # Only the import line
                        file_unused.append(import_name)
            
            except Exception:
                continue
            
            if file_unused:
                unused_imports[file_path] = file_unused
        
        return unused_imports
    
    def find_unused_functions(self) -> Dict[str, List[str]]:
        """Find functions that are defined but never called."""
        unused_functions = {}
        
        # Collect all function calls across all files
        all_calls = set()
        for calls in self.functions_called.values():
            all_calls.update(calls)
        
        for file_path, functions in self.functions_defined.items():
            file_unused = []
            
            for func_name in functions:
                # Skip special methods and protected functions
                if func_name.startswith('_') or func_name in ['main', 'test_']:
                    continue
                
                # Skip if called anywhere
                if func_name not in all_calls:
                    file_unused.append(func_name)
            
            if file_unused:
                unused_functions[file_path] = file_unused
        
        return unused_functions
    
    def find_unused_classes(self) -> Dict[str, List[str]]:
        """Find classes that are defined but never used."""
        unused_classes = {}
        
        # Collect all class usage across all files
        all_usage = set()
        for usage in self.classes_used.values():
            all_usage.update(usage)
        
        for file_path, classes in self.classes_defined.items():
            file_unused = []
            
            for class_name in classes:
                # Skip if used anywhere
                if class_name not in all_usage:
                    file_unused.append(class_name)
            
            if file_unused:
                unused_classes[file_path] = file_unused
        
        return unused_classes
    
    def analyze_feature_flags(self) -> Dict[str, Any]:
        """Analyze feature flags for cleanup opportunities."""
        feature_flags = {
            'archaeological_dag_weighting': {
                'config_key': 'dag.features.enable_archaeological_zone_weighting',
                'default': False,
                'usage_files': [],
                'cleanup_candidate': True
            },
            'htf_context_features': {
                'config_key': 'htf_enabled',
                'default': False,
                'usage_files': [],
                'cleanup_candidate': False  # Core feature
            },
            'oracle_integration': {
                'config_key': 'oracle_enabled',
                'default': False,
                'usage_files': [],
                'cleanup_candidate': False  # Core feature
            }
        }
        
        # Search for usage in codebase
        for flag_name, flag_info in feature_flags.items():
            config_key = flag_info['config_key']
            
            # Search Python files for usage
            for py_file in self.project_root.rglob('*.py'):
                if self.is_protected_path(py_file):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if config_key in content or flag_name in content:
                        flag_info['usage_files'].append(str(py_file))
                
                except Exception:
                    continue
        
        return feature_flags
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete dead code analysis."""
        print("🔍 Running IRONFORGE dead code analysis...")
        
        results = {
            'summary': {
                'files_analyzed': 0,
                'protected_files_skipped': 0,
                'errors': 0
            },
            'unused_imports': {},
            'unused_functions': {},
            'unused_classes': {},
            'feature_flags': {},
            'cleanup_recommendations': []
        }
        
        # Analyze Python files
        for py_file in self.project_root.rglob('*.py'):
            results['summary']['files_analyzed'] += 1
            
            analysis = self.analyze_python_file(py_file)
            
            if analysis.get('protected'):
                results['summary']['protected_files_skipped'] += 1
            elif analysis.get('error'):
                results['summary']['errors'] += 1
        
        # Analyze configuration files
        for config_file in self.project_root.rglob('*.yml'):
            self.analyze_config_file(config_file)
        
        # Find unused code
        results['unused_imports'] = self.find_unused_imports()
        results['unused_functions'] = self.find_unused_functions()
        results['unused_classes'] = self.find_unused_classes()
        results['feature_flags'] = self.analyze_feature_flags()
        
        # Generate recommendations
        results['cleanup_recommendations'] = self.generate_recommendations(results)
        
        return results
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cleanup recommendations."""
        recommendations = []
        
        # Unused imports
        if results['unused_imports']:
            recommendations.append({
                'type': 'unused_imports',
                'priority': 'low',
                'description': f"Remove {sum(len(imports) for imports in results['unused_imports'].values())} unused imports",
                'files_affected': len(results['unused_imports']),
                'risk': 'low'
            })
        
        # Unused functions
        if results['unused_functions']:
            recommendations.append({
                'type': 'unused_functions',
                'priority': 'medium',
                'description': f"Remove {sum(len(funcs) for funcs in results['unused_functions'].values())} unused functions",
                'files_affected': len(results['unused_functions']),
                'risk': 'medium'
            })
        
        # Feature flags
        cleanup_flags = [name for name, info in results['feature_flags'].items() 
                        if info['cleanup_candidate'] and len(info['usage_files']) == 0]
        
        if cleanup_flags:
            recommendations.append({
                'type': 'feature_flags',
                'priority': 'high',
                'description': f"Remove {len(cleanup_flags)} unused feature flags",
                'flags': cleanup_flags,
                'risk': 'low'
            })
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="IRONFORGE dead code analysis")
    parser.add_argument("--preserve-golden-invariants", action="store_true", default=True,
                       help="Preserve Golden Invariants (default: True)")
    parser.add_argument("--output", type=str, default="dead_code_analysis.json",
                       help="Output file for analysis results")
    parser.add_argument("--project-root", type=str, default=".",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    
    if not project_root.exists():
        print(f"❌ Project root not found: {project_root}")
        sys.exit(1)
    
    analyzer = IRONFORGEDeadCodeAnalyzer(project_root)
    results = analyzer.run_analysis()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"Files analyzed: {results['summary']['files_analyzed']}")
    print(f"Protected files skipped: {results['summary']['protected_files_skipped']}")
    print(f"Errors: {results['summary']['errors']}")
    
    print(f"\n🧹 Cleanup Opportunities:")
    for rec in results['cleanup_recommendations']:
        priority_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}[rec['priority']]
        print(f"{priority_emoji} {rec['description']} (Risk: {rec['risk']})")
    
    print(f"\n💾 Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
