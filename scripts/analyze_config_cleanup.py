#!/usr/bin/env python3
"""
IRONFORGE Configuration Cleanup Analysis
========================================

Analyzes configuration files for cleanup opportunities while preserving Golden Invariants.
"""

import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import argparse


class ConfigurationAnalyzer:
    """Analyzes IRONFORGE configuration files for cleanup opportunities."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.golden_invariant_keys = {
            'EVENT_TYPES', 'EDGE_INTENTS', 'NODE_FEATURE_DIM', 'EDGE_FEATURE_DIM',
            'event_types', 'edge_intents', 'node_features', 'edge_features',
            'htf_enabled', 'session_isolation'
        }
        self.config_usage = {}
        self.deprecated_patterns = [
            r'.*_deprecated.*', r'.*_old.*', r'.*_legacy.*', r'.*_unused.*',
            r'.*_temp.*', r'.*_test.*', r'.*_debug.*'
        ]
    
    def find_config_files(self) -> List[Path]:
        """Find all configuration files in the project."""
        config_files = []
        
        # YAML configuration files
        for pattern in ['*.yml', '*.yaml']:
            config_files.extend(self.project_root.rglob(pattern))
        
        # JSON configuration files
        config_files.extend(self.project_root.rglob('*.json'))
        
        # Python configuration files
        for py_file in self.project_root.rglob('*config*.py'):
            config_files.append(py_file)
        
        # Filter out protected directories
        protected_dirs = ['runs/', 'data/', 'models/', '.git/', 'node_modules/']
        filtered_files = []
        
        for config_file in config_files:
            path_str = str(config_file.relative_to(self.project_root))
            if not any(path_str.startswith(protected) for protected in protected_dirs):
                filtered_files.append(config_file)
        
        return filtered_files
    
    def analyze_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Analyze a YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return {'empty': True}
            
            analysis = {
                'keys': self.extract_all_keys(config_data),
                'deprecated_keys': [],
                'duplicate_sections': [],
                'unused_keys': [],
                'golden_invariant_keys': [],
                'complexity_score': 0
            }
            
            # Find deprecated keys
            for key in analysis['keys']:
                if any(re.match(pattern, key, re.IGNORECASE) for pattern in self.deprecated_patterns):
                    analysis['deprecated_keys'].append(key)
            
            # Find Golden Invariant keys (must preserve)
            for key in analysis['keys']:
                if any(gi_key.lower() in key.lower() for gi_key in self.golden_invariant_keys):
                    analysis['golden_invariant_keys'].append(key)
            
            # Calculate complexity score
            analysis['complexity_score'] = len(analysis['keys'])
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def extract_all_keys(self, obj: Any, prefix: str = '') -> List[str]:
        """Extract all keys from nested configuration object."""
        keys = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.append(full_key)
                keys.extend(self.extract_all_keys(value, full_key))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                keys.extend(self.extract_all_keys(item, f"{prefix}[{i}]"))
        
        return keys
    
    def find_config_usage_in_code(self, config_keys: List[str]) -> Dict[str, List[str]]:
        """Find usage of configuration keys in Python code."""
        usage_map = {}
        
        for key in config_keys:
            usage_map[key] = []
            
            # Search for key usage in Python files
            for py_file in self.project_root.rglob('*.py'):
                # Skip protected directories
                path_str = str(py_file.relative_to(self.project_root))
                if any(path_str.startswith(protected) for protected in ['runs/', 'data/', 'models/', '.git/']):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for key usage patterns
                    key_patterns = [
                        f'"{key}"', f"'{key}'", f'.{key}', f'["{key}"]', f"['{key}']"
                    ]
                    
                    for pattern in key_patterns:
                        if pattern in content:
                            usage_map[key].append(str(py_file))
                            break
                
                except Exception:
                    continue
        
        return usage_map
    
    def identify_duplicate_sections(self, config_data: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Identify potentially duplicate configuration sections."""
        duplicates = []
        
        if not isinstance(config_data, dict):
            return duplicates
        
        # Look for similar section names
        section_names = list(config_data.keys())
        
        for i, section1 in enumerate(section_names):
            for section2 in section_names[i+1:]:
                # Check for similar names
                if self.are_sections_similar(section1, section2):
                    duplicates.append((section1, section2))
        
        return duplicates
    
    def are_sections_similar(self, name1: str, name2: str) -> bool:
        """Check if two section names are similar enough to be duplicates."""
        # Simple similarity check
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Check for common patterns
        similar_patterns = [
            (name1_lower.replace('_', ''), name2_lower.replace('_', '')),
            (name1_lower.replace('-', ''), name2_lower.replace('-', '')),
        ]
        
        for pattern1, pattern2 in similar_patterns:
            if pattern1 == pattern2:
                return True
        
        return False
    
    def analyze_all_configs(self) -> Dict[str, Any]:
        """Analyze all configuration files in the project."""
        config_files = self.find_config_files()
        
        results = {
            'total_files': len(config_files),
            'files_analyzed': 0,
            'total_keys': 0,
            'deprecated_keys': 0,
            'unused_keys': 0,
            'golden_invariant_keys': 0,
            'complexity_reduction_potential': 0,
            'file_analyses': {},
            'cleanup_recommendations': []
        }
        
        all_config_keys = []
        
        for config_file in config_files:
            if config_file.suffix in ['.yml', '.yaml']:
                analysis = self.analyze_yaml_config(config_file)
                
                if not analysis.get('error') and not analysis.get('empty'):
                    results['files_analyzed'] += 1
                    results['total_keys'] += len(analysis['keys'])
                    results['deprecated_keys'] += len(analysis['deprecated_keys'])
                    results['golden_invariant_keys'] += len(analysis['golden_invariant_keys'])
                    
                    all_config_keys.extend(analysis['keys'])
                    
                    results['file_analyses'][str(config_file)] = analysis
        
        # Find unused keys across all files
        if all_config_keys:
            usage_map = self.find_config_usage_in_code(all_config_keys)
            unused_keys = [key for key, usage in usage_map.items() if not usage]
            results['unused_keys'] = len(unused_keys)
        
        # Generate cleanup recommendations
        results['cleanup_recommendations'] = self.generate_cleanup_recommendations(results)
        
        return results
    
    def generate_cleanup_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cleanup recommendations based on analysis."""
        recommendations = []
        
        # Deprecated keys recommendation
        if analysis['deprecated_keys'] > 0:
            recommendations.append({
                'type': 'deprecated_keys',
                'priority': 'high',
                'description': f"Remove {analysis['deprecated_keys']} deprecated configuration keys",
                'risk': 'low',
                'estimated_reduction': f"{analysis['deprecated_keys']} keys"
            })
        
        # Unused keys recommendation
        if analysis['unused_keys'] > 0:
            recommendations.append({
                'type': 'unused_keys',
                'priority': 'medium',
                'description': f"Remove {analysis['unused_keys']} unused configuration keys",
                'risk': 'medium',
                'estimated_reduction': f"{analysis['unused_keys']} keys"
            })
        
        # Complexity reduction
        total_keys = analysis['total_keys']
        golden_keys = analysis['golden_invariant_keys']
        reducible_keys = total_keys - golden_keys
        
        if reducible_keys > 0:
            potential_reduction = min(reducible_keys * 0.15, analysis['deprecated_keys'] + analysis['unused_keys'])
            recommendations.append({
                'type': 'complexity_reduction',
                'priority': 'medium',
                'description': f"Potential to reduce configuration complexity by {potential_reduction:.0f} keys (10-15%)",
                'risk': 'medium',
                'estimated_reduction': f"{potential_reduction:.0f} keys"
            })
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analyze IRONFORGE configuration cleanup opportunities")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, default="config_cleanup_analysis.json", help="Output file")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    analyzer = ConfigurationAnalyzer(project_root)
    
    print("🔍 IRONFORGE Configuration Cleanup Analysis")
    print("=" * 50)
    
    results = analyzer.analyze_all_configs()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"Configuration files found: {results['total_files']}")
    print(f"Files analyzed: {results['files_analyzed']}")
    print(f"Total configuration keys: {results['total_keys']}")
    print(f"Deprecated keys: {results['deprecated_keys']}")
    print(f"Unused keys: {results['unused_keys']}")
    print(f"Golden Invariant keys (protected): {results['golden_invariant_keys']}")
    
    print(f"\n🧹 Cleanup Recommendations:")
    for rec in results['cleanup_recommendations']:
        priority_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}[rec['priority']]
        print(f"{priority_emoji} {rec['description']} (Risk: {rec['risk']})")
    
    print(f"\n💾 Detailed analysis saved to: {args.output}")


if __name__ == "__main__":
    main()
