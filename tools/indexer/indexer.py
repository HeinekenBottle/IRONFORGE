#!/usr/bin/env python3
"""
IRONFORGE Semantic Codebase Indexer
===================================

Main orchestrator for deep semantic analysis of the IRONFORGE codebase.
Coordinates AST analysis, engine classification, and report generation.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analyzer import CodeAnalyzer
from .dependency_mapper import DependencyMapper
from .engine_classifier import EngineClassifier
from .report_generator import ReportGenerator


class IRONFORGEIndexer:
    """
    Main indexer class that orchestrates the semantic analysis of IRONFORGE codebase.
    
    Provides comprehensive analysis including:
    - AST-based code parsing
    - Engine architecture mapping
    - Dependency relationship tracking
    - Complexity analysis
    - Pattern detection
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the indexer with project root path."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.analyzer = CodeAnalyzer()
        self.classifier = EngineClassifier()
        self.dependency_mapper = DependencyMapper()
        self.report_generator = ReportGenerator()
        
        # Analysis results
        self.analysis_results = {}
        
        self.logger.info(f"Initialized IRONFORGE Indexer for: {self.project_root}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the indexer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('ironforge.indexer')
    
    def discover_python_files(self) -> List[Path]:
        """
        Discover all Python files in the project.
        
        Returns:
            List of Path objects for Python files
        """
        python_files = []
        
        # Directories to scan
        scan_dirs = [
            'ironforge',
            'iron_core', 
            'analysis',
            'scripts',
            'data_migration',
            'tests',
            'visualizations'
        ]
        
        # Include root-level Python files
        for file_path in self.project_root.glob('*.py'):
            if file_path.is_file():
                python_files.append(file_path)
        
        # Scan specific directories
        for scan_dir in scan_dirs:
            dir_path = self.project_root / scan_dir
            if dir_path.exists():
                for file_path in dir_path.rglob('*.py'):
                    if file_path.is_file() and not self._should_skip_file(file_path):
                        python_files.append(file_path)
        
        self.logger.info(f"Discovered {len(python_files)} Python files")
        return sorted(python_files)
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_patterns = [
            '__pycache__',
            '.pyc',
            'test_',
            '_test.py',
            'conftest.py',
            '.egg-info',
            'build/',
            'dist/',
            '.venv',
            'venv/',
            '.git'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def analyze_codebase(self, include_tests: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the IRONFORGE codebase.
        
        Args:
            include_tests: Whether to include test files in analysis
            
        Returns:
            Complete analysis results dictionary
        """
        start_time = time.time()
        self.logger.info("Starting comprehensive codebase analysis...")
        
        # Discover files
        python_files = self.discover_python_files()
        
        if not include_tests:
            python_files = [f for f in python_files if not self._is_test_file(f)]
        
        # Phase 1: Individual file analysis
        self.logger.info("Phase 1: Analyzing individual files...")
        file_analyses = {}
        
        for file_path in python_files:
            try:
                relative_path = file_path.relative_to(self.project_root)
                self.logger.debug(f"Analyzing: {relative_path}")
                
                analysis = self.analyzer.analyze_file(file_path)
                file_analyses[str(relative_path)] = analysis
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
                continue
        
        # Phase 2: Engine classification
        self.logger.info("Phase 2: Classifying engines and components...")
        engine_architecture = self.classifier.classify_files(file_analyses)
        
        # Phase 3: Dependency mapping
        self.logger.info("Phase 3: Mapping dependencies and relationships...")
        dependency_map = self.dependency_mapper.build_dependency_map(file_analyses)
        
        # Phase 4: Aggregate analysis
        self.logger.info("Phase 4: Generating aggregate metrics...")
        project_overview = self._generate_project_overview(file_analyses, python_files)
        complexity_analysis = self._generate_complexity_analysis(file_analyses)
        public_interfaces = self._extract_public_interfaces(file_analyses, engine_architecture)
        
        # Compile results
        self.analysis_results = {
            'project_overview': project_overview,
            'engine_architecture': engine_architecture,
            'dependency_map': dependency_map,
            'complexity_analysis': complexity_analysis,
            'public_interfaces': public_interfaces,
            'file_analyses': file_analyses,
            'metadata': {
                'analysis_timestamp': time.time(),
                'analysis_duration': time.time() - start_time,
                'indexer_version': '1.0.0',
                'files_analyzed': len(file_analyses)
            }
        }
        
        self.logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
        return self.analysis_results
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        return (
            'test' in file_path.name or
            file_path.name.startswith('test_') or
            '/tests/' in str(file_path)
        )
    
    def _generate_project_overview(self, file_analyses: Dict, python_files: List[Path]) -> Dict[str, Any]:
        """Generate high-level project overview."""
        total_lines = sum(
            analysis.get('metrics', {}).get('lines_of_code', 0)
            for analysis in file_analyses.values()
        )
        
        total_functions = sum(
            len(analysis.get('functions', []))
            for analysis in file_analyses.values()
        )
        
        total_classes = sum(
            len(analysis.get('classes', []))
            for analysis in file_analyses.values()
        )
        
        # Calculate average complexity
        complexities = []
        for analysis in file_analyses.values():
            for func in analysis.get('functions', []):
                if 'complexity' in func:
                    complexities.append(func['complexity'])
        
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0
        
        return {
            'project_name': 'IRONFORGE',
            'total_files': len(file_analyses),
            'total_python_files_discovered': len(python_files),
            'total_lines': total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'average_complexity': round(avg_complexity, 2),
            'architecture_type': 'Multi-Engine Archaeological Discovery System',
            'key_technologies': ['TGAT', 'PyTorch', 'NetworkX', 'NumPy', 'iron-core'],
            'analysis_scope': list(file_analyses.keys())[:10]  # Sample of analyzed files
        }
    
    def _generate_complexity_analysis(self, file_analyses: Dict) -> Dict[str, Any]:
        """Generate complexity analysis across the codebase."""
        complexity_by_file = {}
        complexity_hotspots = []
        
        for file_path, analysis in file_analyses.items():
            file_complexity = 0
            file_functions = 0
            
            for func in analysis.get('functions', []):
                if 'complexity' in func:
                    complexity = func['complexity']
                    file_complexity += complexity
                    file_functions += 1
                    
                    if complexity > 10:  # High complexity threshold
                        complexity_hotspots.append({
                            'file': file_path,
                            'function': func['name'],
                            'complexity': complexity,
                            'line_number': func.get('line_number', 0)
                        })
            
            if file_functions > 0:
                complexity_by_file[file_path] = {
                    'total_complexity': file_complexity,
                    'average_complexity': round(file_complexity / file_functions, 2),
                    'function_count': file_functions
                }
        
        # Sort hotspots by complexity
        complexity_hotspots.sort(key=lambda x: x['complexity'], reverse=True)
        
        return {
            'by_file': complexity_by_file,
            'hotspots': complexity_hotspots[:20],  # Top 20 complexity hotspots
            'summary': {
                'files_with_high_complexity': len([
                    f for f, data in complexity_by_file.items()
                    if data['average_complexity'] > 5
                ]),
                'total_hotspot_functions': len(complexity_hotspots)
            }
        }
    
    def _extract_public_interfaces(self, file_analyses: Dict, engine_architecture: Dict) -> Dict[str, Any]:
        """Extract public interfaces for each engine."""
        public_interfaces = {}
        
        for engine_name, engine_data in engine_architecture.items():
            if engine_name == 'metadata':
                continue
                
            engine_interfaces = {}
            
            for component in engine_data.get('components', []):
                file_path = component['file']
                if file_path in file_analyses:
                    analysis = file_analyses[file_path]
                    
                    # Extract public classes
                    for cls in analysis.get('classes', []):
                        if not cls['name'].startswith('_'):  # Public class
                            class_key = f"{component['name']}.{cls['name']}"
                            engine_interfaces[class_key] = {
                                'type': 'class',
                                'file': file_path,
                                'docstring': cls.get('docstring', ''),
                                'methods': [
                                    m['name'] for m in cls.get('methods', [])
                                    if not m['name'].startswith('_')
                                ],
                                'base_classes': cls.get('base_classes', [])
                            }
                    
                    # Extract public functions
                    for func in analysis.get('functions', []):
                        if not func['name'].startswith('_'):  # Public function
                            func_key = f"{component['name']}.{func['name']}"
                            engine_interfaces[func_key] = {
                                'type': 'function',
                                'file': file_path,
                                'docstring': func.get('docstring', ''),
                                'parameters': func.get('parameters', []),
                                'return_annotation': func.get('return_annotation')
                            }
            
            if engine_interfaces:
                public_interfaces[engine_name] = engine_interfaces
        
        return public_interfaces
    
    def generate_reports(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate JSON and Markdown reports.
        
        Args:
            output_dir: Directory to save reports (defaults to project root)
            
        Returns:
            Dictionary with paths to generated reports
        """
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_codebase() first.")
        
        output_path = Path(output_dir) if output_dir else self.project_root
        
        # Generate reports
        report_paths = self.report_generator.generate_reports(
            self.analysis_results,
            output_path
        )
        
        self.logger.info(f"Generated reports: {list(report_paths.keys())}")
        return report_paths
    
    def quick_analysis(self, focus: str = 'engines') -> Dict[str, Any]:
        """
        Perform quick analysis focused on specific aspects.
        
        Args:
            focus: Analysis focus ('engines', 'dependencies', 'complexity')
            
        Returns:
            Focused analysis results
        """
        self.logger.info(f"Starting quick analysis with focus: {focus}")
        
        # Minimal file discovery for quick analysis
        python_files = self.discover_python_files()
        
        # Quick file analysis (limited scope)
        file_analyses = {}
        for file_path in python_files[:50]:  # Limit for speed
            try:
                relative_path = file_path.relative_to(self.project_root)
                analysis = self.analyzer.analyze_file(file_path, quick_mode=True)
                file_analyses[str(relative_path)] = analysis
            except Exception:
                continue
        
        if focus == 'engines':
            return {
                'focus': 'engines',
                'engine_architecture': self.classifier.classify_files(file_analyses),
                'file_count': len(file_analyses)
            }
        elif focus == 'dependencies':
            return {
                'focus': 'dependencies',
                'dependency_map': self.dependency_mapper.build_dependency_map(file_analyses),
                'file_count': len(file_analyses)
            }
        elif focus == 'complexity':
            return {
                'focus': 'complexity',
                'complexity_analysis': self._generate_complexity_analysis(file_analyses),
                'file_count': len(file_analyses)
            }
        
        return {'error': f'Unknown focus: {focus}'}