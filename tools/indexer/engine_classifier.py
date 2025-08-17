#!/usr/bin/env python3
"""
Engine Classifier for IRONFORGE Semantic Indexer
================================================

Classifies components into IRONFORGE's multi-engine architecture:
- Analysis Engine: Pattern analysis and session adaptation
- Learning Engine: TGAT discovery and graph building
- Synthesis Engine: Pattern validation and graduation
- Integration Engine: Container system and configuration
- Validation Engine: Testing and quality assurance
- Utilities: Support tools and scripts
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Set
import logging


class EngineClassifier:
    """
    Classifies IRONFORGE components into their respective engines
    based on file paths, naming patterns, and functionality.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ironforge.indexer.classifier')
        
        # Define engine classification rules
        self.engine_rules = {
            'analysis': {
                'paths': ['ironforge/analysis', 'analysis/'],
                'keywords': ['session', 'pattern', 'timeframe', 'lattice', 'archaeological', 'spectrum'],
                'classes': ['SessionAdapter', 'Lattice', 'Archaeology', 'Analyzer'],
                'description': 'Pattern analysis and session adaptation components'
            },
            'learning': {
                'paths': ['ironforge/learning', 'learning/'],
                'keywords': ['graph', 'tgat', 'discovery', 'neural', 'clustering', 'regime'],
                'classes': ['GraphBuilder', 'Discovery', 'TGAT', 'Clustering'],
                'description': 'Machine learning, TGAT discovery, and graph building'
            },
            'synthesis': {
                'paths': ['ironforge/synthesis', 'synthesis/'],
                'keywords': ['graduation', 'validation', 'production', 'pattern'],
                'classes': ['Graduation', 'Validator'],
                'description': 'Pattern validation and production graduation'
            },
            'integration': {
                'paths': ['ironforge/integration', 'integration/', 'iron_core/'],
                'keywords': ['container', 'config', 'lazy', 'performance', 'injection'],
                'classes': ['Container', 'Config', 'Lazy'],
                'description': 'System integration, configuration, and dependency injection'
            },
            'validation': {
                'paths': ['ironforge/validation', 'validation/', 'tests/'],
                'keywords': ['test', 'validate', 'check', 'verify', 'benchmark'],
                'classes': ['Test', 'Validator', 'Benchmark'],
                'description': 'Testing, validation, and quality assurance'
            },
            'reporting': {
                'paths': ['ironforge/reporting', 'reporting/', 'reports/'],
                'keywords': ['report', 'output', 'visualization', 'summary'],
                'classes': ['Report', 'Generator', 'Visualizer'],
                'description': 'Report generation and data visualization'
            },
            'utilities': {
                'paths': ['ironforge/utilities', 'utilities/', 'scripts/', 'data_migration/'],
                'keywords': ['util', 'helper', 'tool', 'script', 'migration'],
                'classes': ['Helper', 'Util', 'Tool', 'Migration'],
                'description': 'Utility functions, scripts, and support tools'
            },
            'data': {
                'paths': ['data/', 'preservation/'],
                'keywords': ['data', 'preserve', 'store', 'cache'],
                'classes': [],
                'description': 'Data storage and preservation'
            }
        }
    
    def classify_files(self, file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify analyzed files into IRONFORGE engine architecture.
        
        Args:
            file_analyses: Dictionary of file analysis results
            
        Returns:
            Engine architecture mapping with components
        """
        self.logger.info("Classifying files into engine architecture...")
        
        engine_architecture = {}
        unclassified_files = []
        
        # Initialize engines
        for engine_name, rules in self.engine_rules.items():
            engine_architecture[engine_name] = {
                'description': rules['description'],
                'components': [],
                'file_count': 0,
                'total_lines': 0,
                'complexity_score': 0,
                'key_classes': [],
                'public_interfaces': []
            }
        
        # Classify each file
        for file_path, analysis in file_analyses.items():
            if 'error' in analysis:
                continue
            
            # Determine engine classification
            engine = self._classify_file(file_path, analysis)
            
            if engine:
                component_info = self._extract_component_info(file_path, analysis)
                engine_architecture[engine]['components'].append(component_info)
                
                # Update engine metrics
                metrics = analysis.get('metrics', {})
                engine_architecture[engine]['file_count'] += 1
                engine_architecture[engine]['total_lines'] += metrics.get('lines_of_code', 0)
                engine_architecture[engine]['complexity_score'] += metrics.get('complexity_total', 0)
                
                # Extract key classes and interfaces
                for class_info in analysis.get('classes', []):
                    if not class_info['name'].startswith('_'):  # Public classes
                        engine_architecture[engine]['key_classes'].append({
                            'name': class_info['name'],
                            'file': file_path,
                            'docstring': class_info.get('docstring', '')[:100] + '...' if class_info.get('docstring', '') else ''
                        })
                
                # Extract public functions as interfaces
                for func_info in analysis.get('functions', []):
                    if not func_info['name'].startswith('_'):  # Public functions
                        engine_architecture[engine]['public_interfaces'].append({
                            'name': func_info['name'],
                            'file': file_path,
                            'parameters': len(func_info.get('parameters', [])),
                            'docstring': func_info.get('docstring', '')[:100] + '...' if func_info.get('docstring', '') else ''
                        })
            else:
                unclassified_files.append(file_path)
        
        # Calculate average complexity for each engine
        for engine_name, engine_data in engine_architecture.items():
            if engine_data['file_count'] > 0:
                engine_data['avg_complexity'] = round(
                    engine_data['complexity_score'] / engine_data['file_count'], 2
                )
            else:
                engine_data['avg_complexity'] = 0
        
        # Add metadata
        engine_architecture['metadata'] = {
            'total_files_classified': sum(e['file_count'] for e in engine_architecture.values() if isinstance(e, dict) and 'file_count' in e),
            'unclassified_files': unclassified_files,
            'classification_coverage': self._calculate_coverage(file_analyses, unclassified_files),
            'engine_distribution': {
                name: data['file_count'] 
                for name, data in engine_architecture.items() 
                if isinstance(data, dict) and 'file_count' in data
            }
        }
        
        self.logger.info(f"Classified {engine_architecture['metadata']['total_files_classified']} files into {len(self.engine_rules)} engines")
        
        return engine_architecture
    
    def _classify_file(self, file_path: str, analysis: Dict[str, Any]) -> str:
        """
        Classify a single file into an engine category.
        
        Args:
            file_path: Path to the file
            analysis: Analysis results for the file
            
        Returns:
            Engine name or None if unclassified
        """
        file_path_lower = file_path.lower()
        
        # Score each engine based on classification rules
        engine_scores = {}
        
        for engine_name, rules in self.engine_rules.items():
            score = 0
            
            # Path-based scoring (highest weight)
            for path_pattern in rules['paths']:
                if path_pattern.lower() in file_path_lower:
                    score += 100
                    break
            
            # Keyword-based scoring (filename and content)
            for keyword in rules['keywords']:
                if keyword.lower() in file_path_lower:
                    score += 50
                
                # Check in docstrings and class names
                if self._keyword_in_analysis(keyword, analysis):
                    score += 30
            
            # Class name pattern scoring
            for class_pattern in rules['classes']:
                if self._class_pattern_matches(class_pattern, analysis):
                    score += 40
            
            if score > 0:
                engine_scores[engine_name] = score
        
        # Return engine with highest score, or None if no matches
        if engine_scores:
            return max(engine_scores, key=engine_scores.get)
        
        return None
    
    def _keyword_in_analysis(self, keyword: str, analysis: Dict[str, Any]) -> bool:
        """Check if keyword appears in analysis content."""
        keyword_lower = keyword.lower()
        
        # Check module docstring
        if analysis.get('docstring'):
            if keyword_lower in analysis['docstring'].lower():
                return True
        
        # Check class names and docstrings
        for class_info in analysis.get('classes', []):
            if keyword_lower in class_info['name'].lower():
                return True
            if class_info.get('docstring') and keyword_lower in class_info['docstring'].lower():
                return True
        
        # Check function names and docstrings
        for func_info in analysis.get('functions', []):
            if keyword_lower in func_info['name'].lower():
                return True
            if func_info.get('docstring') and keyword_lower in func_info['docstring'].lower():
                return True
        
        return False
    
    def _class_pattern_matches(self, pattern: str, analysis: Dict[str, Any]) -> bool:
        """Check if any class matches the pattern."""
        pattern_lower = pattern.lower()
        
        for class_info in analysis.get('classes', []):
            if pattern_lower in class_info['name'].lower():
                return True
        
        return False
    
    def _extract_component_info(self, file_path: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract component information for the engine."""
        component_name = Path(file_path).stem
        
        # Extract primary classes and functions
        primary_classes = [
            cls['name'] for cls in analysis.get('classes', [])
            if not cls['name'].startswith('_')
        ]
        
        primary_functions = [
            func['name'] for func in analysis.get('functions', [])
            if not func['name'].startswith('_')
        ]
        
        # Detect component patterns
        patterns = analysis.get('patterns', [])
        pattern_names = [p['pattern'] for p in patterns]
        
        # Calculate component complexity
        metrics = analysis.get('metrics', {})
        
        return {
            'name': component_name,
            'file': file_path,
            'primary_classes': primary_classes,
            'primary_functions': primary_functions,
            'patterns': pattern_names,
            'lines_of_code': metrics.get('lines_of_code', 0),
            'complexity': metrics.get('complexity_total', 0),
            'class_count': metrics.get('class_count', 0),
            'function_count': metrics.get('function_count', 0),
            'docstring': analysis.get('docstring', '')[:200] + '...' if analysis.get('docstring', '') else '',
            'imports': len(analysis.get('imports', [])),
            'decorators': analysis.get('decorators', [])
        }
    
    def _calculate_coverage(self, file_analyses: Dict[str, Any], unclassified_files: List[str]) -> float:
        """Calculate classification coverage percentage."""
        total_files = len(file_analyses)
        classified_files = total_files - len(unclassified_files)
        
        if total_files == 0:
            return 0.0
        
        return round((classified_files / total_files) * 100, 2)
    
    def get_engine_summary(self, engine_architecture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the engine architecture.
        
        Args:
            engine_architecture: Engine classification results
            
        Returns:
            Engine architecture summary
        """
        summary = {
            'total_engines': len(self.engine_rules),
            'engines': {},
            'top_engines_by_size': [],
            'top_engines_by_complexity': [],
            'architecture_health': {}
        }
        
        # Summarize each engine
        engine_sizes = []
        engine_complexities = []
        
        for engine_name, rules in self.engine_rules.items():
            if engine_name in engine_architecture:
                engine_data = engine_architecture[engine_name]
                
                engine_summary = {
                    'description': rules['description'],
                    'file_count': engine_data.get('file_count', 0),
                    'total_lines': engine_data.get('total_lines', 0),
                    'avg_complexity': engine_data.get('avg_complexity', 0),
                    'key_classes_count': len(engine_data.get('key_classes', [])),
                    'public_interfaces_count': len(engine_data.get('public_interfaces', [])),
                    'top_components': [
                        comp['name'] for comp in sorted(
                            engine_data.get('components', []),
                            key=lambda x: x.get('lines_of_code', 0),
                            reverse=True
                        )[:3]
                    ]
                }
                
                summary['engines'][engine_name] = engine_summary
                
                engine_sizes.append((engine_name, engine_data.get('file_count', 0)))
                engine_complexities.append((engine_name, engine_data.get('avg_complexity', 0)))
        
        # Top engines by size and complexity
        summary['top_engines_by_size'] = sorted(engine_sizes, key=lambda x: x[1], reverse=True)[:5]
        summary['top_engines_by_complexity'] = sorted(engine_complexities, key=lambda x: x[1], reverse=True)[:5]
        
        # Architecture health metrics
        total_files = sum(e.get('file_count', 0) for e in engine_architecture.values() if isinstance(e, dict))
        avg_complexity = sum(e.get('avg_complexity', 0) for e in engine_architecture.values() if isinstance(e, dict)) / len(self.engine_rules)
        
        summary['architecture_health'] = {
            'total_files': total_files,
            'average_complexity': round(avg_complexity, 2),
            'balanced_distribution': self._assess_distribution_balance(engine_sizes),
            'complexity_hotspots': len([e for _, e in engine_complexities if e > 10]),
            'classification_coverage': engine_architecture.get('metadata', {}).get('classification_coverage', 0)
        }
        
        return summary
    
    def _assess_distribution_balance(self, engine_sizes: List[tuple]) -> str:
        """Assess if the engine distribution is balanced."""
        if not engine_sizes:
            return 'unknown'
        
        sizes = [size for _, size in engine_sizes]
        max_size = max(sizes)
        min_size = min(sizes)
        
        if max_size == 0:
            return 'empty'
        
        ratio = max_size / min_size if min_size > 0 else float('inf')
        
        if ratio <= 3:
            return 'well_balanced'
        elif ratio <= 10:
            return 'moderately_balanced'
        else:
            return 'imbalanced'
    
    def suggest_refactoring_opportunities(self, engine_architecture: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest refactoring opportunities based on engine analysis.
        
        Args:
            engine_architecture: Engine classification results
            
        Returns:
            List of refactoring suggestions
        """
        suggestions = []
        
        # Check for large components that might need splitting
        for engine_name, engine_data in engine_architecture.items():
            if isinstance(engine_data, dict) and 'components' in engine_data:
                for component in engine_data['components']:
                    if component.get('lines_of_code', 0) > 500:
                        suggestions.append({
                            'type': 'split_large_component',
                            'component': component['name'],
                            'engine': engine_name,
                            'lines': component['lines_of_code'],
                            'suggestion': f"Consider splitting {component['name']} ({component['lines_of_code']} lines) into smaller modules"
                        })
                    
                    if component.get('complexity', 0) > 20:
                        suggestions.append({
                            'type': 'reduce_complexity',
                            'component': component['name'],
                            'engine': engine_name,
                            'complexity': component['complexity'],
                            'suggestion': f"High complexity in {component['name']} (complexity: {component['complexity']}). Consider refactoring."
                        })
        
        # Check for engines with too many or too few components
        for engine_name, engine_data in engine_architecture.items():
            if isinstance(engine_data, dict) and 'file_count' in engine_data:
                file_count = engine_data['file_count']
                
                if file_count > 20:
                    suggestions.append({
                        'type': 'engine_too_large',
                        'engine': engine_name,
                        'file_count': file_count,
                        'suggestion': f"Engine '{engine_name}' has {file_count} files. Consider splitting into sub-engines."
                    })
                elif file_count == 1 and engine_name not in ['data', 'reporting']:
                    suggestions.append({
                        'type': 'engine_too_small',
                        'engine': engine_name,
                        'file_count': file_count,
                        'suggestion': f"Engine '{engine_name}' has only {file_count} file. Consider merging with related engine."
                    })
        
        return suggestions