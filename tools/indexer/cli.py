#!/usr/bin/env python3
"""
Command Line Interface for IRONFORGE Semantic Indexer
====================================================

Provides command-line access to the semantic indexer with multiple modes:
- Full analysis with all reports
- Quick summary generation
- Engine-focused analysis
- Dependency analysis
- Custom output formats
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional, List
import logging

from .indexer import IRONFORGEIndexer


class IndexerCLI:
    """
    Command line interface for the IRONFORGE semantic indexer.
    
    Supports multiple analysis modes and output formats optimized
    for different use cases from terminal workflows.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.indexer = None
    
    def _setup_logging(self, verbose: bool = False) -> logging.Logger:
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(levelname)s: %(message)s',
            stream=sys.stdout
        )
        return logging.getLogger('ironforge.indexer.cli')
    
    def main(self) -> int:
        """Main CLI entry point."""
        parser = self._create_parser()
        args = parser.parse_args()
        
        # Update logging if verbose
        if args.verbose:
            self._setup_logging(verbose=True)
        
        try:
            return self._execute_command(args)
        except KeyboardInterrupt:
            self.logger.info("\\nOperation cancelled by user")
            return 130
        except Exception as e:
            self.logger.error(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command line argument parser."""
        parser = argparse.ArgumentParser(
            description='IRONFORGE Semantic Codebase Indexer',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
Examples:
  %(prog)s                    # Full analysis with all reports
  %(prog)s summary           # Quick markdown summary only
  %(prog)s engines           # Engine architecture analysis
  %(prog)s dependencies      # Dependency mapping analysis
  %(prog)s --output ./reports # Custom output directory
  %(prog)s --verbose         # Detailed logging output
            '''
        )
        
        # Analysis mode selection
        parser.add_argument(
            'mode',
            nargs='?',
            default='full',
            choices=['full', 'summary', 'engines', 'dependencies', 'quick'],
            help='Analysis mode (default: full)'
        )
        
        # Project root directory
        parser.add_argument(
            '--project-root',
            type=str,
            default='.',
            help='Project root directory (default: current directory)'
        )
        
        # Output directory
        parser.add_argument(
            '--output',
            type=str,
            help='Output directory for reports (default: project root)'
        )
        
        # Include test files
        parser.add_argument(
            '--include-tests',
            action='store_true',
            help='Include test files in analysis'
        )
        
        # Output format selection
        parser.add_argument(
            '--format',
            choices=['json', 'markdown', 'both'],
            default='both',
            help='Output format (default: both)'
        )
        
        # Verbose output
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Verbose output with detailed logging'
        )
        
        # Quiet mode
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Quiet mode - minimal output'
        )
        
        # JSON output to stdout
        parser.add_argument(
            '--json-stdout',
            action='store_true',
            help='Output JSON results to stdout (useful for piping)'
        )
        
        return parser
    
    def _execute_command(self, args) -> int:
        """Execute the indexer command based on arguments."""
        start_time = time.time()
        
        if not args.quiet:
            self.logger.info(f"Starting IRONFORGE semantic analysis in '{args.mode}' mode...")
        
        # Initialize indexer
        project_root = Path(args.project_root).resolve()
        if not project_root.exists():
            self.logger.error(f"Project root does not exist: {project_root}")
            return 1
        
        self.indexer = IRONFORGEIndexer(str(project_root))
        
        # Determine output directory
        output_dir = Path(args.output) if args.output else project_root
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute analysis based on mode
        try:
            if args.mode == 'full':
                return self._execute_full_analysis(args, output_dir)
            elif args.mode == 'summary':
                return self._execute_summary_analysis(args, output_dir)
            elif args.mode == 'engines':
                return self._execute_engine_analysis(args, output_dir)
            elif args.mode == 'dependencies':
                return self._execute_dependency_analysis(args, output_dir)
            elif args.mode == 'quick':
                return self._execute_quick_analysis(args, output_dir)
        
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return 1
        
        finally:
            if not args.quiet:
                duration = time.time() - start_time
                self.logger.info(f"Analysis completed in {duration:.2f} seconds")
    
    def _execute_full_analysis(self, args, output_dir: Path) -> int:
        """Execute full comprehensive analysis."""
        if not args.quiet:
            self.logger.info("Performing comprehensive codebase analysis...")
        
        # Run full analysis
        results = self.indexer.analyze_codebase(include_tests=args.include_tests)
        
        # Generate reports
        if args.json_stdout:
            # Output simplified JSON to stdout for piping
            simplified_results = self._simplify_results_for_stdout(results)
            print(json.dumps(simplified_results, indent=2))
        else:
            # Generate file-based reports
            report_paths = self.indexer.generate_reports(str(output_dir))
            
            if not args.quiet:
                self._print_analysis_summary(results)
                self._print_report_paths(report_paths)
        
        return 0
    
    def _execute_summary_analysis(self, args, output_dir: Path) -> int:
        """Execute summary-only analysis."""
        if not args.quiet:
            self.logger.info("Generating project summary...")
        
        # Run full analysis (needed for comprehensive summary)
        results = self.indexer.analyze_codebase(include_tests=args.include_tests)
        
        if args.json_stdout:
            # Output summary JSON to stdout
            summary = self._extract_summary_data(results)
            print(json.dumps(summary, indent=2))
        else:
            # Generate markdown summary only
            report_paths = self.indexer.generate_reports(str(output_dir))
            
            if not args.quiet:
                self.logger.info(f"Summary generated: {report_paths.get('markdown', 'unknown')}")
                self._print_quick_stats(results)
        
        return 0
    
    def _execute_engine_analysis(self, args, output_dir: Path) -> int:
        """Execute engine-focused analysis."""
        if not args.quiet:
            self.logger.info("Analyzing engine architecture...")
        
        # Use quick analysis focused on engines
        results = self.indexer.quick_analysis(focus='engines')
        
        if args.json_stdout:
            print(json.dumps(results, indent=2))
        else:
            self._print_engine_summary(results)
            
            # Save detailed engine report
            engine_report_path = output_dir / 'ironforge_engines.json'
            with open(engine_report_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            if not args.quiet:
                self.logger.info(f"Engine report saved: {engine_report_path}")
        
        return 0
    
    def _execute_dependency_analysis(self, args, output_dir: Path) -> int:
        """Execute dependency-focused analysis."""
        if not args.quiet:
            self.logger.info("Analyzing dependencies and relationships...")
        
        # Use quick analysis focused on dependencies
        results = self.indexer.quick_analysis(focus='dependencies')
        
        if args.json_stdout:
            print(json.dumps(results, indent=2))
        else:
            self._print_dependency_summary(results)
            
            # Save detailed dependency report
            deps_report_path = output_dir / 'ironforge_dependencies.json'
            with open(deps_report_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            if not args.quiet:
                self.logger.info(f"Dependency report saved: {deps_report_path}")
        
        return 0
    
    def _execute_quick_analysis(self, args, output_dir: Path) -> int:
        """Execute quick analysis with minimal output."""
        # Quick complexity analysis
        results = self.indexer.quick_analysis(focus='complexity')
        
        if args.json_stdout:
            print(json.dumps(results, indent=2))
        else:
            self._print_quick_stats(results)
        
        return 0
    
    def _print_analysis_summary(self, results: dict) -> None:
        """Print comprehensive analysis summary."""
        overview = results.get('project_overview', {})
        engine_arch = results.get('engine_architecture', {})
        dependency_map = results.get('dependency_map', {})
        complexity = results.get('complexity_analysis', {})
        
        print("\\n" + "="*60)
        print("IRONFORGE SEMANTIC ANALYSIS SUMMARY")
        print("="*60)
        
        # Project overview
        print(f"\\n📊 PROJECT OVERVIEW")
        print(f"   Files Analyzed: {overview.get('total_files', 0):,}")
        print(f"   Lines of Code: {overview.get('total_lines', 0):,}")
        print(f"   Functions: {overview.get('total_functions', 0):,}")
        print(f"   Classes: {overview.get('total_classes', 0):,}")
        print(f"   Avg Complexity: {overview.get('average_complexity', 0):.1f}")
        
        # Engine summary
        print(f"\\n🏗️  ENGINE ARCHITECTURE")
        engine_count = len([e for e in engine_arch.keys() if e != 'metadata'])
        print(f"   Total Engines: {engine_count}")
        
        for engine_name, engine_data in engine_arch.items():
            if isinstance(engine_data, dict) and engine_name != 'metadata':
                file_count = engine_data.get('file_count', 0)
                if file_count > 0:
                    print(f"   {engine_name.title()}: {file_count} files")
        
        # Dependency summary
        print(f"\\n🔗 DEPENDENCIES")
        total_deps = dependency_map.get('dependency_statistics', {}).get('total_dependencies', 0)
        circular_deps = len(dependency_map.get('circular_dependencies', []))
        print(f"   Total Dependencies: {total_deps}")
        print(f"   Circular Dependencies: {circular_deps}")
        
        # Complexity summary
        print(f"\\n⚡ COMPLEXITY")
        hotspots = len(complexity.get('hotspots', []))
        high_complexity_files = complexity.get('summary', {}).get('files_with_high_complexity', 0)
        print(f"   Complexity Hotspots: {hotspots}")
        print(f"   High Complexity Files: {high_complexity_files}")
        
        print("\\n" + "="*60)
    
    def _print_engine_summary(self, results: dict) -> None:
        """Print engine architecture summary."""
        engine_arch = results.get('engine_architecture', {})
        
        print("\\n" + "="*50)
        print("IRONFORGE ENGINE ARCHITECTURE")
        print("="*50)
        
        for engine_name, engine_data in engine_arch.items():
            if isinstance(engine_data, dict) and engine_name != 'metadata':
                file_count = engine_data.get('file_count', 0)
                total_lines = engine_data.get('total_lines', 0)
                avg_complexity = engine_data.get('avg_complexity', 0)
                
                print(f"\\n🔧 {engine_name.upper()} ENGINE")
                print(f"   Files: {file_count}")
                print(f"   Lines: {total_lines:,}")
                print(f"   Avg Complexity: {avg_complexity:.1f}")
                print(f"   Description: {engine_data.get('description', 'No description')}")
                
                # Top components
                components = engine_data.get('components', [])[:3]
                if components:
                    print("   Top Components:")
                    for comp in components:
                        print(f"     - {comp.get('name', 'Unknown')} ({comp.get('lines_of_code', 0)} LOC)")
        
        print("\\n" + "="*50)
    
    def _print_dependency_summary(self, results: dict) -> None:
        """Print dependency analysis summary."""
        dependency_map = results.get('dependency_map', {})
        
        print("\\n" + "="*50)
        print("IRONFORGE DEPENDENCY ANALYSIS")
        print("="*50)
        
        # Statistics
        stats = dependency_map.get('dependency_statistics', {})
        print(f"\\n📊 STATISTICS")
        print(f"   Total Modules: {stats.get('total_modules', 0)}")
        print(f"   Total Dependencies: {stats.get('total_dependencies', 0)}")
        print(f"   Avg Dependencies/Module: {stats.get('average_dependencies_per_module', 0):.1f}")
        
        # Circular dependencies
        circular_deps = dependency_map.get('circular_dependencies', [])
        print(f"\\n🔄 CIRCULAR DEPENDENCIES: {len(circular_deps)}")
        for i, cycle in enumerate(circular_deps[:3], 1):
            cycle_path = " → ".join(cycle.get('cycle', [])[:3])  # First 3 modules
            print(f"   {i}. {cycle_path}... (length: {cycle.get('length', 0)})")
        
        # Hub modules
        hubs = dependency_map.get('hub_modules', [])[:5]
        if hubs:
            print(f"\\n🌐 HUB MODULES")
            for hub in hubs:
                print(f"   {hub.get('module', 'Unknown')} (centrality: {hub.get('centrality', 0)})")
        
        print("\\n" + "="*50)
    
    def _print_quick_stats(self, results: dict) -> None:
        """Print quick statistics."""
        if 'file_count' in results:
            print(f"\\n📊 Quick Analysis Complete")
            print(f"   Files Analyzed: {results.get('file_count', 0)}")
            print(f"   Focus: {results.get('focus', 'unknown').title()}")
        else:
            overview = results.get('project_overview', {})
            print(f"\\n📊 Analysis Complete")
            print(f"   Files: {overview.get('total_files', 0)}")
            print(f"   Functions: {overview.get('total_functions', 0)}")
            print(f"   Classes: {overview.get('total_classes', 0)}")
    
    def _print_report_paths(self, report_paths: dict) -> None:
        """Print generated report file paths."""
        print(f"\\n📄 GENERATED REPORTS")
        for report_type, path in report_paths.items():
            print(f"   {report_type.title()}: {path}")
    
    def _simplify_results_for_stdout(self, results: dict) -> dict:
        """Simplify results for stdout JSON output."""
        return {
            'project_overview': results.get('project_overview', {}),
            'engine_summary': {
                name: {
                    'file_count': data.get('file_count', 0),
                    'total_lines': data.get('total_lines', 0),
                    'avg_complexity': data.get('avg_complexity', 0)
                }
                for name, data in results.get('engine_architecture', {}).items()
                if isinstance(data, dict) and name != 'metadata'
            },
            'dependency_summary': {
                'total_dependencies': results.get('dependency_map', {}).get('dependency_statistics', {}).get('total_dependencies', 0),
                'circular_dependencies': len(results.get('dependency_map', {}).get('circular_dependencies', [])),
                'hub_modules_count': len(results.get('dependency_map', {}).get('hub_modules', []))
            },
            'complexity_summary': {
                'hotspots_count': len(results.get('complexity_analysis', {}).get('hotspots', [])),
                'high_complexity_files': results.get('complexity_analysis', {}).get('summary', {}).get('files_with_high_complexity', 0)
            }
        }
    
    def _extract_summary_data(self, results: dict) -> dict:
        """Extract summary data for summary mode."""
        return {
            'mode': 'summary',
            'project_overview': results.get('project_overview', {}),
            'engine_count': len([
                e for e in results.get('engine_architecture', {}).keys() 
                if e != 'metadata'
            ]),
            'total_dependencies': results.get('dependency_map', {}).get('dependency_statistics', {}).get('total_dependencies', 0),
            'complexity_hotspots': len(results.get('complexity_analysis', {}).get('hotspots', []))
        }


def main():
    """CLI entry point."""
    cli = IndexerCLI()
    return cli.main()


if __name__ == '__main__':
    sys.exit(main())