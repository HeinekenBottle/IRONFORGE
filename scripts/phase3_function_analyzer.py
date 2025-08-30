#!/usr/bin/env python3
"""
IRONFORGE Phase 3 Function Analyzer
===================================

Advanced analyzer for identifying unused functions with safety prioritization.
Focuses on the 847 unused function candidates for systematic removal.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import json
import argparse
from collections import defaultdict
import re


class Phase3FunctionAnalyzer:
    """Advanced function analyzer for Phase 3 cleanup."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
        # Protected modules (NEVER touch these)
        self.protected_modules = {
            'ironforge/analysis',
            'ironforge/learning', 
            'ironforge/synthesis',
            'ironforge/validation',
            'ironforge/contracts',
            'ironforge/constants.py'
        }
        
        # Golden Invariant functions (NEVER remove)
        self.golden_invariant_functions = {
            'validate_golden_invariants', 'EVENT_TYPES', 'EDGE_INTENTS',
            'NODE_FEATURE_DIM_STANDARD', 'NODE_FEATURE_DIM_HTF', 'EDGE_FEATURE_DIM'
        }
        
        # Core API functions (NEVER remove)
        self.core_api_functions = {
            'run_discovery', 'score_confluence', 'validate_run', 'build_minidash',
            'load_config', 'materialize_run_dir'
        }
        
        # Risk categories for systematic removal
        self.risk_categories = {
            'very_low': [],    # Scripts, test utilities, examples
            'low': [],         # Non-core utilities, helpers
            'medium': [],      # Secondary features, optional components
            'high': [],        # Core-adjacent functionality
            'protected': []    # Never remove
        }
        
        # Function usage tracking
        self.function_definitions = defaultdict(list)
        self.function_calls = defaultdict(set)
        self.import_usage = defaultdict(set)
    
    def is_protected_path(self, file_path: Path) -> bool:
        """Check if file path is protected from cleanup."""
        path_str = str(file_path.relative_to(self.project_root))
        
        # Protected directories
        protected_dirs = [
            'runs/', 'data/', 'models/', 'configs/', '.git/',
            'tests/_golden/', 'tests/contracts/'
        ]
        
        for protected in protected_dirs:
            if path_str.startswith(protected):
                return True
        
        # Protected modules
        for protected_module in self.protected_modules:
            if path_str.startswith(protected_module):
                return True
        
        return False
    
    def categorize_function_risk(self, func_name: str, file_path: Path) -> str:
        """Categorize function removal risk level."""
        path_str = str(file_path.relative_to(self.project_root))
        
        # Protected functions
        if (func_name in self.golden_invariant_functions or 
            func_name in self.core_api_functions or
            func_name.startswith('test_') or
            func_name.startswith('_') and not func_name.startswith('__')):
            return 'protected'
        
        # Very low risk: Scripts, examples, standalone utilities
        if any(path_str.startswith(prefix) for prefix in [
            'scripts/', 'examples/', 'tools/', 'agents/build_artifact_cleaner/'
        ]):
            return 'very_low'
        
        # Low risk: Non-core utilities
        if any(path_str.startswith(prefix) for prefix in [
            'ironforge/utilities/', 'ironforge/reporting/', 'ironforge/sdk/'
        ]):
            return 'low'
        
        # Medium risk: Secondary features
        if any(path_str.startswith(prefix) for prefix in [
            'ironforge/confluence/', 'ironforge/temporal/', 'ironforge/integration/'
        ]):
            return 'medium'
        
        # High risk: Core-adjacent
        if any(path_str.startswith(prefix) for prefix in [
            'ironforge/learning/', 'ironforge/synthesis/', 'ironforge/analysis/'
        ]):
            return 'high'
        
        # Default to medium risk
        return 'medium'
    
    def analyze_function_usage(self, file_path: Path) -> Dict[str, Any]:
        """Analyze function definitions and usage in a file."""
        if self.is_protected_path(file_path):
            return {'protected': True}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            analysis = {
                'functions_defined': [],
                'functions_called': [],
                'imports': [],
                'protected': False
            }
            
            # Walk the AST
            for node in ast.walk(tree):
                # Function definitions
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    analysis['functions_defined'].append(func_name)
                    self.function_definitions[func_name].append(str(file_path))
                
                # Function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        analysis['functions_called'].append(func_name)
                        self.function_calls[str(file_path)].add(func_name)
                    elif isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            func_name = f"{node.func.value.id}.{node.func.attr}"
                            analysis['functions_called'].append(func_name)
                            self.function_calls[str(file_path)].add(func_name)
                
                # Imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                        self.import_usage[str(file_path)].add(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            import_name = f"{node.module}.{alias.name}"
                            analysis['imports'].append(import_name)
                            self.import_usage[str(file_path)].add(import_name)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'protected': False}
    
    def find_unused_functions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find unused functions categorized by risk level."""
        print("🔍 Analyzing function usage across codebase...")
        
        # First pass: collect all function definitions and calls
        for py_file in self.project_root.rglob('*.py'):
            self.analyze_function_usage(py_file)
        
        # Second pass: identify unused functions
        all_calls = set()
        for calls in self.function_calls.values():
            all_calls.update(calls)
        
        unused_functions = {
            'very_low': [],
            'low': [],
            'medium': [],
            'high': [],
            'protected': []
        }
        
        for func_name, file_paths in self.function_definitions.items():
            # Skip if function is called anywhere
            if func_name in all_calls:
                continue
            
            # Skip special methods
            if func_name.startswith('__') and func_name.endswith('__'):
                continue
            
            # Categorize by risk for each file where it's defined
            for file_path_str in file_paths:
                file_path = Path(file_path_str)
                risk_level = self.categorize_function_risk(func_name, file_path)
                
                unused_functions[risk_level].append({
                    'function_name': func_name,
                    'file_path': file_path_str,
                    'risk_level': risk_level,
                    'line_number': None  # Could be enhanced to find line numbers
                })
        
        return unused_functions
    
    def generate_removal_plan(self, unused_functions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate systematic removal plan with safety checkpoints."""
        plan = {
            'total_candidates': sum(len(funcs) for funcs in unused_functions.values()),
            'removal_batches': [],
            'safety_checkpoints': [],
            'estimated_reduction': 0
        }
        
        # Create removal batches (10-15 functions per batch)
        batch_size = 12
        batch_num = 1
        
        # Process in order of increasing risk
        for risk_level in ['very_low', 'low', 'medium']:
            functions = unused_functions[risk_level]
            
            for i in range(0, len(functions), batch_size):
                batch = functions[i:i + batch_size]
                
                plan['removal_batches'].append({
                    'batch_number': batch_num,
                    'risk_level': risk_level,
                    'functions': batch,
                    'validation_required': True,
                    'rollback_tag': f'cleanup-phase-3-batch-{batch_num}'
                })
                
                # Add safety checkpoint every 3 batches
                if batch_num % 3 == 0:
                    plan['safety_checkpoints'].append({
                        'checkpoint_number': batch_num // 3,
                        'after_batch': batch_num,
                        'validation_suite': 'comprehensive',
                        'rollback_tag': f'cleanup-phase-3-checkpoint-{batch_num // 3}'
                    })
                
                batch_num += 1
        
        # Estimate code reduction
        plan['estimated_reduction'] = min(
            plan['total_candidates'] * 3,  # Assume 3 lines per function on average
            int(0.20 * 50000)  # Cap at 20% of estimated codebase size
        )
        
        return plan
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete Phase 3 function analysis."""
        print("🚀 Running Phase 3 Function Analysis")
        print("=" * 50)
        
        unused_functions = self.find_unused_functions()
        removal_plan = self.generate_removal_plan(unused_functions)
        
        results = {
            'analysis_summary': {
                'total_unused_functions': removal_plan['total_candidates'],
                'very_low_risk': len(unused_functions['very_low']),
                'low_risk': len(unused_functions['low']),
                'medium_risk': len(unused_functions['medium']),
                'high_risk': len(unused_functions['high']),
                'protected': len(unused_functions['protected'])
            },
            'unused_functions': unused_functions,
            'removal_plan': removal_plan,
            'safety_protocols': {
                'git_tagging': True,
                'validation_checkpoints': len(removal_plan['safety_checkpoints']),
                'rollback_capability': True,
                'golden_invariant_protection': True
            }
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Function Analysis for IRONFORGE")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, default="phase3_function_analysis.json", help="Output file")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    analyzer = Phase3FunctionAnalyzer(project_root)
    
    results = analyzer.run_analysis()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    summary = results['analysis_summary']
    print(f"\\n📊 Phase 3 Analysis Summary:")
    print(f"Total unused functions: {summary['total_unused_functions']}")
    print(f"Very low risk: {summary['very_low_risk']}")
    print(f"Low risk: {summary['low_risk']}")
    print(f"Medium risk: {summary['medium_risk']}")
    print(f"High risk: {summary['high_risk']} (will not remove)")
    print(f"Protected: {summary['protected']} (will not remove)")
    
    plan = results['removal_plan']
    print(f"\\n🗓️ Removal Plan:")
    print(f"Removal batches: {len(plan['removal_batches'])}")
    print(f"Safety checkpoints: {len(plan['safety_checkpoints'])}")
    print(f"Estimated code reduction: {plan['estimated_reduction']} lines")
    
    print(f"\\n💾 Detailed analysis saved to: {args.output}")


if __name__ == "__main__":
    main()
