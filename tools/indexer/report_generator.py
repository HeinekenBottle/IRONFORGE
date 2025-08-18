#!/usr/bin/env python3
"""
Report Generator for IRONFORGE Semantic Indexer
==============================================

Generates comprehensive reports from codebase analysis:
- JSON report for AI assistants
- Markdown summary for human readers
- Specialized engine reports
- Dependency visualizations
- Architecture insights
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List


class ReportGenerator:
    """
    Generates multiple report formats from IRONFORGE analysis results.
    
    Provides:
    - JSON reports optimized for AI assistants
    - Human-readable Markdown summaries
    - Engine-specific reports
    - Dependency analysis reports
    - Architecture health assessments
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ironforge.indexer.report')
    
    def generate_reports(self, analysis_results: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
        """
        Generate all report types from analysis results.
        
        Args:
            analysis_results: Complete analysis results from indexer
            output_dir: Directory to save reports
            
        Returns:
            Dictionary mapping report types to file paths
        """
        self.logger.info("Generating reports...")
        
        report_paths = {}
        
        # Generate JSON report for AI assistants
        json_path = output_dir / 'ironforge_index.json'
        self._generate_json_report(analysis_results, json_path)
        report_paths['json'] = str(json_path)
        
        # Generate Markdown summary for humans
        md_path = output_dir / 'ironforge_summary.md'
        self._generate_markdown_summary(analysis_results, md_path)
        report_paths['markdown'] = str(md_path)
        
        # Generate engine-specific report
        engine_path = output_dir / 'ironforge_engines.json'
        self._generate_engine_report(analysis_results, engine_path)
        report_paths['engines'] = str(engine_path)
        
        # Generate dependency report
        deps_path = output_dir / 'ironforge_dependencies.json'
        self._generate_dependency_report(analysis_results, deps_path)
        report_paths['dependencies'] = str(deps_path)
        
        self.logger.info(f"Generated {len(report_paths)} reports in {output_dir}")
        return report_paths
    
    def _generate_json_report(self, analysis_results: Dict[str, Any], output_path: Path) -> None:
        """Generate comprehensive JSON report optimized for AI assistants."""
        # Extract key sections
        project_overview = analysis_results.get('project_overview', {})
        engine_architecture = analysis_results.get('engine_architecture', {})
        dependency_map = analysis_results.get('dependency_map', {})
        complexity_analysis = analysis_results.get('complexity_analysis', {})
        public_interfaces = analysis_results.get('public_interfaces', {})
        metadata = analysis_results.get('metadata', {})
        
        # Create AI-optimized report structure
        ai_report = {
            'ironforge_semantic_index': {
                'version': '1.0.0',
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                'analysis_metadata': metadata
            },
            
            'project_overview': {
                'name': project_overview.get('project_name', 'IRONFORGE'),
                'architecture_type': project_overview.get('architecture_type', 'Multi-Engine Archaeological Discovery System'),
                'total_files': project_overview.get('total_files', 0),
                'total_lines': project_overview.get('total_lines', 0),
                'total_functions': project_overview.get('total_functions', 0),
                'total_classes': project_overview.get('total_classes', 0),
                'average_complexity': project_overview.get('average_complexity', 0),
                'key_technologies': project_overview.get('key_technologies', []),
                'analysis_scope': project_overview.get('analysis_scope', [])
            },
            
            'engine_architecture': self._process_engine_architecture_for_ai(engine_architecture),
            
            'dependency_map': {
                'import_relationships': self._simplify_import_graph(dependency_map.get('import_graph', {})),
                'cross_engine_flows': dependency_map.get('cross_engine_flows', {}),
                'circular_dependencies': dependency_map.get('circular_dependencies', []),
                'coupling_metrics': dependency_map.get('coupling_metrics', {}),
                'critical_paths': dependency_map.get('critical_paths', []),
                'hub_modules': dependency_map.get('hub_modules', [])
            },
            
            'complexity_analysis': {
                'hotspots': complexity_analysis.get('hotspots', [])[:10],  # Top 10 hotspots
                'summary': complexity_analysis.get('summary', {}),
                'high_complexity_files': [
                    f for f, data in complexity_analysis.get('by_file', {}).items()
                    if data.get('average_complexity', 0) > 5
                ]
            },
            
            'public_interfaces': self._organize_public_interfaces(public_interfaces),
            
            'architecture_insights': self._generate_architecture_insights(analysis_results),
            
            'ai_assistant_guidance': self._generate_ai_guidance(analysis_results)
        }
        
        # Write JSON report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ai_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated AI-optimized JSON report: {output_path}")
    
    def _generate_markdown_summary(self, analysis_results: Dict[str, Any], output_path: Path) -> None:
        """Generate human-readable Markdown summary."""
        project_overview = analysis_results.get('project_overview', {})
        engine_architecture = analysis_results.get('engine_architecture', {})
        dependency_map = analysis_results.get('dependency_map', {})
        complexity_analysis = analysis_results.get('complexity_analysis', {})
        metadata = analysis_results.get('metadata', {})
        
        md_content = self._build_markdown_content(
            project_overview, engine_architecture, dependency_map, 
            complexity_analysis, metadata
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.info(f"Generated Markdown summary: {output_path}")
    
    def _build_markdown_content(self, project_overview: Dict, engine_architecture: Dict, 
                              dependency_map: Dict, complexity_analysis: Dict, 
                              metadata: Dict) -> str:
        """Build the Markdown content."""
        
        # Header and overview
        md = f"""# IRONFORGE Codebase Analysis Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
Analysis Duration: {metadata.get('analysis_duration', 0):.2f} seconds
Files Analyzed: {metadata.get('files_analyzed', 0)}

## Project Overview

**{project_overview.get('project_name', 'IRONFORGE')}** - {project_overview.get('architecture_type', 'Multi-Engine Archaeological Discovery System')}

### Key Metrics
- **Total Files**: {project_overview.get('total_files', 0):,}
- **Lines of Code**: {project_overview.get('total_lines', 0):,}
- **Functions**: {project_overview.get('total_functions', 0):,}
- **Classes**: {project_overview.get('total_classes', 0):,}
- **Average Complexity**: {project_overview.get('average_complexity', 0):.1f}

### Key Technologies
{self._format_list(project_overview.get('key_technologies', []))}

## Engine Architecture

IRONFORGE follows a multi-engine architecture pattern with clear separation of concerns:

"""
        
        # Engine details
        for engine_name, engine_data in engine_architecture.items():
            if engine_name == 'metadata' or not isinstance(engine_data, dict):
                continue
            
            md += f"""### {engine_name.title()} Engine

**Description**: {engine_data.get('description', 'No description available')}

**Metrics**:
- Files: {engine_data.get('file_count', 0)}
- Lines of Code: {engine_data.get('total_lines', 0):,}
- Average Complexity: {engine_data.get('avg_complexity', 0):.1f}

**Key Components**:
{self._format_components(engine_data.get('components', []))}

**Key Classes**:
{self._format_key_classes(engine_data.get('key_classes', []))}

---

"""
        
        # Dependency analysis
        md += f"""## Dependency Analysis

### Cross-Engine Flows
{self._format_cross_engine_flows(dependency_map.get('cross_engine_flows', {}))}

### Circular Dependencies
{self._format_circular_dependencies(dependency_map.get('circular_dependencies', []))}

### Hub Modules (High Centrality)
{self._format_hub_modules(dependency_map.get('hub_modules', []))}

## Complexity Analysis

### Complexity Hotspots
{self._format_complexity_hotspots(complexity_analysis.get('hotspots', []))}

### Summary
- **Files with High Complexity**: {complexity_analysis.get('summary', {}).get('files_with_high_complexity', 0)}
- **Total Hotspot Functions**: {complexity_analysis.get('summary', {}).get('total_hotspot_functions', 0)}

## Architecture Health Assessment

{self._generate_health_assessment(engine_architecture, dependency_map, complexity_analysis)}

## Recommendations

{self._generate_recommendations(engine_architecture, dependency_map, complexity_analysis)}

---

*Report generated by IRONFORGE Semantic Indexer v1.0.0*
"""
        
        return md
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list as Markdown bullet points."""
        if not items:
            return "- None"
        return "\n".join(f"- {item}" for item in items)
    
    def _format_components(self, components: List[Dict]) -> str:
        """Format components list."""
        if not components:
            return "- No components found"
        
        formatted = []
        for comp in components[:5]:  # Top 5 components
            formatted.append(f"- **{comp.get('name', 'Unknown')}** ({comp.get('lines_of_code', 0)} LOC, Complexity: {comp.get('complexity', 0)})")
        
        if len(components) > 5:
            formatted.append(f"- ... and {len(components) - 5} more components")
        
        return "\n".join(formatted)
    
    def _format_key_classes(self, classes: List[Dict]) -> str:
        """Format key classes list."""
        if not classes:
            return "- No public classes found"
        
        formatted = []
        for cls in classes[:3]:  # Top 3 classes
            formatted.append(f"- **{cls.get('name', 'Unknown')}**: {cls.get('docstring', 'No description')}")
        
        if len(classes) > 3:
            formatted.append(f"- ... and {len(classes) - 3} more classes")
        
        return "\n".join(formatted)
    
    def _format_cross_engine_flows(self, flows: Dict) -> str:
        """Format cross-engine dependency flows."""
        if not flows.get('flow_summary'):
            return "No significant cross-engine dependencies detected."
        
        formatted = []
        for flow, count in sorted(flows['flow_summary'].items(), key=lambda x: x[1], reverse=True):
            formatted.append(f"- **{flow}**: {count} dependencies")
        
        return "\n".join(formatted)
    
    def _format_circular_dependencies(self, cycles: List[Dict]) -> str:
        """Format circular dependencies."""
        if not cycles:
            return "✅ No circular dependencies detected."
        
        formatted = [f"⚠️ Found {len(cycles)} circular dependencies:"]
        for i, cycle in enumerate(cycles[:3], 1):  # Show first 3
            cycle_path = " → ".join(cycle.get('cycle', []))
            formatted.append(f"{i}. {cycle_path} (Length: {cycle.get('length', 0)})")
        
        if len(cycles) > 3:
            formatted.append(f"... and {len(cycles) - 3} more cycles")
        
        return "\n".join(formatted)
    
    def _format_hub_modules(self, hubs: List[Dict]) -> str:
        """Format hub modules."""
        if not hubs:
            return "No significant hub modules identified."
        
        formatted = []
        for hub in hubs[:5]:  # Top 5 hubs
            formatted.append(f"- **{hub.get('module', 'Unknown')}** (Centrality: {hub.get('centrality', 0)}, Type: {hub.get('hub_type', 'unknown')})")
        
        return "\n".join(formatted)
    
    def _format_complexity_hotspots(self, hotspots: List[Dict]) -> str:
        """Format complexity hotspots."""
        if not hotspots:
            return "No high-complexity functions detected."
        
        formatted = []
        for hotspot in hotspots[:5]:  # Top 5 hotspots
            formatted.append(f"- **{hotspot.get('function', 'Unknown')}** in {hotspot.get('file', 'unknown')} (Complexity: {hotspot.get('complexity', 0)})")
        
        return "\n".join(formatted)
    
    def _generate_health_assessment(self, engine_architecture: Dict, dependency_map: Dict, complexity_analysis: Dict) -> str:
        """Generate architecture health assessment."""
        assessments = []
        
        # Check engine balance
        engine_sizes = [
            data.get('file_count', 0) for name, data in engine_architecture.items() 
            if isinstance(data, dict) and name != 'metadata'
        ]
        
        if engine_sizes:
            max_size = max(engine_sizes)
            min_size = min([s for s in engine_sizes if s > 0])
            balance_ratio = max_size / min_size if min_size > 0 else float('inf')
            
            if balance_ratio <= 3:
                assessments.append("✅ **Engine Balance**: Well-balanced distribution of components across engines")
            elif balance_ratio <= 10:
                assessments.append("⚠️ **Engine Balance**: Moderate imbalance in engine sizes")
            else:
                assessments.append("❌ **Engine Balance**: Significant imbalance - some engines are much larger than others")
        
        # Check circular dependencies
        cycles = dependency_map.get('circular_dependencies', [])
        if not cycles:
            assessments.append("✅ **Dependency Health**: No circular dependencies detected")
        else:
            assessments.append(f"❌ **Dependency Health**: {len(cycles)} circular dependencies found")
        
        # Check complexity
        hotspots = complexity_analysis.get('hotspots', [])
        high_complexity_count = len([h for h in hotspots if h.get('complexity', 0) > 15])
        
        if high_complexity_count == 0:
            assessments.append("✅ **Complexity Health**: No extreme complexity hotspots")
        elif high_complexity_count <= 5:
            assessments.append(f"⚠️ **Complexity Health**: {high_complexity_count} high-complexity functions need attention")
        else:
            assessments.append(f"❌ **Complexity Health**: {high_complexity_count} high-complexity functions require refactoring")
        
        return "\n".join(assessments)
    
    def _generate_recommendations(self, engine_architecture: Dict, dependency_map: Dict, complexity_analysis: Dict) -> str:
        """Generate architecture improvement recommendations."""
        recommendations = []
        
        # Engine-specific recommendations
        for engine_name, engine_data in engine_architecture.items():
            if not isinstance(engine_data, dict) or engine_name == 'metadata':
                continue
            
            file_count = engine_data.get('file_count', 0)
            avg_complexity = engine_data.get('avg_complexity', 0)
            
            if file_count > 20:
                recommendations.append(f"🔄 **{engine_name.title()} Engine**: Consider splitting into sub-engines (currently {file_count} files)")
            
            if avg_complexity > 8:
                recommendations.append(f"🧹 **{engine_name.title()} Engine**: High average complexity ({avg_complexity:.1f}) - consider refactoring")
        
        # Dependency recommendations
        problematic_flows = dependency_map.get('cross_engine_flows', {}).get('problematic_flows', [])
        for flow in problematic_flows[:3]:  # Top 3 issues
            recommendations.append(f"⚡ **Dependency**: {flow.get('issue', 'Unknown issue')} in {flow.get('flow', 'unknown flow')}")
        
        # Complexity recommendations
        hotspots = complexity_analysis.get('hotspots', [])[:3]  # Top 3 hotspots
        for hotspot in hotspots:
            if hotspot.get('complexity', 0) > 15:
                recommendations.append(f"🔧 **Refactor**: {hotspot.get('function', 'Unknown')} in {hotspot.get('file', 'unknown')} (complexity: {hotspot.get('complexity', 0)})")
        
        if not recommendations:
            recommendations.append("✅ No specific recommendations - architecture appears healthy!")
        
        return "\n".join(recommendations)
    
    def _process_engine_architecture_for_ai(self, engine_architecture: Dict) -> Dict[str, Any]:
        """Process engine architecture data for AI consumption."""
        ai_engines = {}
        
        for engine_name, engine_data in engine_architecture.items():
            if engine_name == 'metadata' or not isinstance(engine_data, dict):
                continue
            
            ai_engines[engine_name] = {
                'description': engine_data.get('description', ''),
                'metrics': {
                    'file_count': engine_data.get('file_count', 0),
                    'total_lines': engine_data.get('total_lines', 0),
                    'avg_complexity': engine_data.get('avg_complexity', 0)
                },
                'key_components': [
                    {
                        'name': comp.get('name', ''),
                        'file': comp.get('file', ''),
                        'primary_classes': comp.get('primary_classes', []),
                        'complexity': comp.get('complexity', 0)
                    }
                    for comp in engine_data.get('components', [])[:5]  # Top 5 components
                ],
                'public_interfaces': [
                    {
                        'name': cls.get('name', ''),
                        'file': cls.get('file', ''),
                        'description': cls.get('docstring', '')[:100]
                    }
                    for cls in engine_data.get('key_classes', [])[:3]  # Top 3 classes
                ]
            }
        
        return ai_engines
    
    def _simplify_import_graph(self, import_graph: Dict) -> Dict[str, Any]:
        """Simplify import graph for AI consumption."""
        simplified = {}
        
        for module, data in import_graph.items():
            simplified[module] = {
                'imports': [imp.get('module', '') for imp in data.get('imports', [])],
                'imported_by': [imp.get('module', '') for imp in data.get('imported_by', [])],
                'coupling_score': data.get('import_count', 0) + data.get('imported_by_count', 0)
            }
        
        return simplified
    
    def _organize_public_interfaces(self, public_interfaces: Dict) -> Dict[str, Any]:
        """Organize public interfaces by engine for AI consumption."""
        organized = {}
        
        for engine_name, interfaces in public_interfaces.items():
            organized[engine_name] = {
                'classes': {},
                'functions': {}
            }
            
            for interface_name, interface_data in interfaces.items():
                if interface_data.get('type') == 'class':
                    organized[engine_name]['classes'][interface_name] = {
                        'file': interface_data.get('file', ''),
                        'description': interface_data.get('docstring', ''),
                        'methods': interface_data.get('methods', [])
                    }
                elif interface_data.get('type') == 'function':
                    organized[engine_name]['functions'][interface_name] = {
                        'file': interface_data.get('file', ''),
                        'description': interface_data.get('docstring', ''),
                        'parameters': interface_data.get('parameters', [])
                    }
        
        return organized
    
    def _generate_architecture_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level architecture insights."""
        insights = {
            'design_patterns': [],
            'architectural_smells': [],
            'strengths': [],
            'suggested_improvements': []
        }
        
        # Extract design patterns from file analyses
        file_analyses = analysis_results.get('file_analyses', {})
        pattern_counts = {}
        
        for analysis in file_analyses.values():
            for pattern in analysis.get('patterns', []):
                pattern_name = pattern.get('pattern', 'unknown')
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        
        insights['design_patterns'] = [
            {'pattern': pattern, 'usage_count': count}
            for pattern, count in pattern_counts.items()
        ]
        
        # Detect architectural smells
        dependency_map = analysis_results.get('dependency_map', {})
        circular_deps = dependency_map.get('circular_dependencies', [])
        
        if circular_deps:
            insights['architectural_smells'].append({
                'smell': 'circular_dependencies',
                'severity': 'high',
                'count': len(circular_deps)
            })
        
        # Identify strengths
        engine_architecture = analysis_results.get('engine_architecture', {})
        engine_count = len([e for e in engine_architecture.keys() if e != 'metadata'])
        
        if engine_count >= 5:
            insights['strengths'].append({
                'strength': 'well_organized_architecture',
                'description': f'Clear separation into {engine_count} distinct engines'
            })
        
        return insights
    
    def _generate_ai_guidance(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate guidance specifically for AI assistants."""
        guidance = {
            'primary_entry_points': [],
            'core_abstractions': [],
            'data_flow_patterns': [],
            'testing_patterns': [],
            'configuration_patterns': []
        }
        
        # Identify primary entry points
        file_analyses = analysis_results.get('file_analyses', {})
        for file_path, analysis in file_analyses.items():
            # Look for main functions or classes that might be entry points
            for func in analysis.get('functions', []):
                if func['name'] in ['main', 'run', 'execute', 'start']:
                    guidance['primary_entry_points'].append({
                        'file': file_path,
                        'function': func['name'],
                        'description': func.get('docstring', '')
                    })
        
        # Identify core abstractions (base classes, interfaces)
        for file_path, analysis in file_analyses.items():
            for cls in analysis.get('classes', []):
                if cls.get('is_abstract') or 'base' in cls['name'].lower() or 'interface' in cls['name'].lower():
                    guidance['core_abstractions'].append({
                        'file': file_path,
                        'class': cls['name'],
                        'description': cls.get('docstring', '')
                    })
        
        # Identify configuration patterns
        for file_path, analysis in file_analyses.items():
            if 'config' in file_path.lower():
                guidance['configuration_patterns'].append({
                    'file': file_path,
                    'description': analysis.get('docstring', ''),
                    'classes': [cls['name'] for cls in analysis.get('classes', [])]
                })
        
        return guidance
    
    def _generate_engine_report(self, analysis_results: Dict[str, Any], output_path: Path) -> None:
        """Generate engine-specific detailed report."""
        engine_architecture = analysis_results.get('engine_architecture', {})
        
        engine_report = {
            'engines': engine_architecture,
            'summary': {
                'total_engines': len([e for e in engine_architecture.keys() if e != 'metadata']),
                'total_components': sum(
                    len(data.get('components', [])) 
                    for data in engine_architecture.values() 
                    if isinstance(data, dict)
                ),
                'engine_distribution': {
                    name: data.get('file_count', 0)
                    for name, data in engine_architecture.items()
                    if isinstance(data, dict) and name != 'metadata'
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(engine_report, f, indent=2, ensure_ascii=False)
    
    def _generate_dependency_report(self, analysis_results: Dict[str, Any], output_path: Path) -> None:
        """Generate dependency-specific detailed report."""
        dependency_map = analysis_results.get('dependency_map', {})
        
        dependency_report = {
            'dependency_analysis': dependency_map,
            'summary': {
                'total_dependencies': dependency_map.get('dependency_statistics', {}).get('total_dependencies', 0),
                'circular_dependencies_count': len(dependency_map.get('circular_dependencies', [])),
                'hub_modules_count': len(dependency_map.get('hub_modules', [])),
                'critical_paths_count': len(dependency_map.get('critical_paths', []))
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dependency_report, f, indent=2, ensure_ascii=False)