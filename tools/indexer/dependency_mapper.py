#!/usr/bin/env python3
"""
Dependency Mapper for IRONFORGE Semantic Indexer
===============================================

Maps and analyzes dependencies between IRONFORGE components:
- Import relationships between modules
- Cross-engine dependency flows
- Circular dependency detection
- Data flow analysis
- Coupling metrics
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


class DependencyMapper:
    """
    Analyzes and maps dependencies between IRONFORGE components.
    
    Provides comprehensive dependency analysis including:
    - Import relationship mapping
    - Cross-engine data flows
    - Circular dependency detection
    - Coupling strength analysis
    - Dependency graph construction
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ironforge.indexer.dependency')
    
    def build_dependency_map(self, file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive dependency map from file analyses.
        
        Args:
            file_analyses: Dictionary of file analysis results
            
        Returns:
            Complete dependency mapping and analysis
        """
        self.logger.info("Building dependency map...")
        
        # Extract import relationships
        import_graph = self._build_import_graph(file_analyses)
        
        # Analyze dependency flows
        dependency_flows = self._analyze_dependency_flows(import_graph, file_analyses)
        
        # Detect circular dependencies
        circular_dependencies = self._detect_circular_dependencies(import_graph)
        
        # Calculate coupling metrics
        coupling_metrics = self._calculate_coupling_metrics(import_graph, file_analyses)
        
        # Find orphaned modules
        orphaned_modules = self._find_orphaned_modules(import_graph, file_analyses)
        
        # Analyze cross-engine flows
        cross_engine_flows = self._analyze_cross_engine_flows(import_graph, file_analyses)
        
        # Generate dependency statistics
        dependency_stats = self._generate_dependency_statistics(import_graph, file_analyses)
        
        return {
            'import_graph': import_graph,
            'dependency_flows': dependency_flows,
            'circular_dependencies': circular_dependencies,
            'coupling_metrics': coupling_metrics,
            'orphaned_modules': orphaned_modules,
            'cross_engine_flows': cross_engine_flows,
            'dependency_statistics': dependency_stats,
            'critical_paths': self._find_critical_paths(import_graph),
            'hub_modules': self._identify_hub_modules(import_graph),
            'leaf_modules': self._identify_leaf_modules(import_graph)
        }
    
    def _build_import_graph(self, file_analyses: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Build directed graph of import relationships.
        
        Returns:
            Graph where keys are modules and values contain dependency info
        """
        import_graph = defaultdict(lambda: {
            'imports': [],
            'imported_by': [],
            'import_count': 0,
            'imported_by_count': 0
        })
        
        # Process each file's imports
        for file_path, analysis in file_analyses.items():
            if 'error' in analysis:
                continue
            
            module_name = self._file_path_to_module(file_path)
            
            # Process imports
            for import_info in analysis.get('imports', []):
                imported_module = self._resolve_import_module(import_info, file_path)
                
                if imported_module and imported_module != module_name:
                    # Add import relationship
                    import_graph[module_name]['imports'].append({
                        'module': imported_module,
                        'type': import_info['type'],
                        'line_number': import_info.get('line_number', 0),
                        'alias': import_info.get('alias'),
                        'specific_import': import_info.get('name')
                    })
                    
                    import_graph[imported_module]['imported_by'].append({
                        'module': module_name,
                        'type': import_info['type'],
                        'line_number': import_info.get('line_number', 0)
                    })
        
        # Update counts
        for module_name, data in import_graph.items():
            data['import_count'] = len(data['imports'])
            data['imported_by_count'] = len(data['imported_by'])
        
        return dict(import_graph)
    
    def _file_path_to_module(self, file_path: str) -> str:
        """Convert file path to module name."""
        # Remove .py extension and convert to module notation
        module_path = file_path.replace('.py', '').replace('/', '.')
        
        # Remove leading ./ if present
        if module_path.startswith('./'):
            module_path = module_path[2:]
        
        return module_path
    
    def _resolve_import_module(self, import_info: Dict[str, Any], current_file: str) -> Optional[str]:
        """
        Resolve import to actual module name.
        
        Args:
            import_info: Import information from AST analysis
            current_file: Current file path for relative imports
            
        Returns:
            Resolved module name or None
        """
        if import_info['type'] == 'import':
            return import_info['module']
        elif import_info['type'] == 'from_import':
            base_module = import_info['module']
            
            # Handle relative imports
            if import_info.get('level', 0) > 0:
                current_module = self._file_path_to_module(current_file)
                module_parts = current_module.split('.')
                
                # Navigate up the module hierarchy
                levels_up = import_info['level']
                if levels_up <= len(module_parts):
                    base_parts = module_parts[:-levels_up] if levels_up > 0 else module_parts
                    if base_module:
                        return '.'.join(base_parts + [base_module])
                    else:
                        return '.'.join(base_parts) if base_parts else None
            
            return base_module
        
        return None
    
    def _analyze_dependency_flows(self, import_graph: Dict[str, Any], file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in dependency flows."""
        flows = {
            'linear_chains': [],
            'fan_out_patterns': [],
            'fan_in_patterns': [],
            'bottlenecks': [],
            'flow_depth': {}
        }
        
        # Find linear dependency chains
        for module, data in import_graph.items():
            if data['import_count'] == 1 and data['imported_by_count'] <= 1:
                chain = self._trace_linear_chain(module, import_graph)
                if len(chain) > 2:
                    flows['linear_chains'].append(chain)
        
        # Find fan-out patterns (one module imported by many)
        for module, data in import_graph.items():
            if data['imported_by_count'] > 5:
                flows['fan_out_patterns'].append({
                    'module': module,
                    'imported_by_count': data['imported_by_count'],
                    'importers': [imp['module'] for imp in data['imported_by']]
                })
        
        # Find fan-in patterns (one module importing many)
        for module, data in import_graph.items():
            if data['import_count'] > 10:
                flows['fan_in_patterns'].append({
                    'module': module,
                    'import_count': data['import_count'],
                    'imports': [imp['module'] for imp in data['imports']]
                })
        
        # Identify bottlenecks (high coupling in both directions)
        for module, data in import_graph.items():
            coupling_score = data['import_count'] + data['imported_by_count']
            if coupling_score > 15:
                flows['bottlenecks'].append({
                    'module': module,
                    'coupling_score': coupling_score,
                    'imports': data['import_count'],
                    'imported_by': data['imported_by_count']
                })
        
        # Calculate flow depth for each module
        for module in import_graph:
            flows['flow_depth'][module] = self._calculate_dependency_depth(module, import_graph)
        
        return flows
    
    def _trace_linear_chain(self, start_module: str, import_graph: Dict[str, Any]) -> List[str]:
        """Trace a linear dependency chain from a starting module."""
        chain = [start_module]
        current = start_module
        visited = {start_module}
        
        # Follow the chain backward (to dependencies)
        while current in import_graph:
            imports = import_graph[current]['imports']
            if len(imports) == 1:
                next_module = imports[0]['module']
                if next_module not in visited and next_module in import_graph:
                    if import_graph[next_module]['imported_by_count'] == 1:
                        chain.insert(0, next_module)
                        visited.add(next_module)
                        current = next_module
                        continue
            break
        
        # Follow the chain forward (to dependents)
        current = start_module
        while current in import_graph:
            imported_by = import_graph[current]['imported_by']
            if len(imported_by) == 1:
                next_module = imported_by[0]['module']
                if next_module not in visited and next_module in import_graph:
                    if import_graph[next_module]['import_count'] == 1:
                        chain.append(next_module)
                        visited.add(next_module)
                        current = next_module
                        continue
            break
        
        return chain
    
    def _calculate_dependency_depth(self, module: str, import_graph: Dict[str, Any]) -> int:
        """Calculate the maximum dependency depth from a module."""
        def dfs_depth(current: str, visited: Set[str]) -> int:
            if current not in import_graph or current in visited:
                return 0
            
            visited.add(current)
            max_depth = 0
            
            for import_info in import_graph[current]['imports']:
                imported_module = import_info['module']
                depth = 1 + dfs_depth(imported_module, visited.copy())
                max_depth = max(max_depth, depth)
            
            return max_depth
        
        return dfs_depth(module, set())
    
    def _detect_circular_dependencies(self, import_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect circular dependencies using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs_cycle_detection(module: str, path: List[str]) -> None:
            if module in rec_stack:
                # Found a cycle
                cycle_start = path.index(module)
                cycle = path[cycle_start:] + [module]
                cycles.append({
                    'cycle': cycle,
                    'length': len(cycle) - 1,
                    'severity': 'high' if len(cycle) <= 3 else 'medium'
                })
                return
            
            if module in visited or module not in import_graph:
                return
            
            visited.add(module)
            rec_stack.add(module)
            path.append(module)
            
            for import_info in import_graph[module]['imports']:
                imported_module = import_info['module']
                dfs_cycle_detection(imported_module, path.copy())
            
            rec_stack.remove(module)
        
        # Check each unvisited module
        for module in import_graph:
            if module not in visited:
                dfs_cycle_detection(module, [])
        
        return cycles
    
    def _calculate_coupling_metrics(self, import_graph: Dict[str, Any], file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate various coupling metrics."""
        metrics = {
            'afferent_coupling': {},  # How many modules depend on this one
            'efferent_coupling': {},  # How many modules this one depends on
            'instability': {},        # Efferent / (Afferent + Efferent)
            'coupling_distribution': {},
            'highly_coupled_modules': []
        }
        
        for module, data in import_graph.items():
            afferent = data['imported_by_count']  # Incoming dependencies
            efferent = data['import_count']       # Outgoing dependencies
            
            metrics['afferent_coupling'][module] = afferent
            metrics['efferent_coupling'][module] = efferent
            
            # Calculate instability (0 = stable, 1 = unstable)
            total_coupling = afferent + efferent
            instability = efferent / total_coupling if total_coupling > 0 else 0
            metrics['instability'][module] = round(instability, 3)
            
            # Identify highly coupled modules
            if total_coupling > 10:
                metrics['highly_coupled_modules'].append({
                    'module': module,
                    'total_coupling': total_coupling,
                    'afferent': afferent,
                    'efferent': efferent,
                    'instability': instability
                })
        
        # Calculate coupling distribution
        coupling_values = [
            data['import_count'] + data['imported_by_count']
            for data in import_graph.values()
        ]
        
        if coupling_values:
            metrics['coupling_distribution'] = {
                'mean': round(sum(coupling_values) / len(coupling_values), 2),
                'max': max(coupling_values),
                'min': min(coupling_values),
                'modules_with_high_coupling': len([c for c in coupling_values if c > 10])
            }
        
        return metrics
    
    def _find_orphaned_modules(self, import_graph: Dict[str, Any], file_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find modules with no dependencies (potential candidates for removal)."""
        orphaned = []
        
        for module, data in import_graph.items():
            if data['import_count'] == 0 and data['imported_by_count'] == 0:
                # Check if it's a script (has main block)
                file_path = module.replace('.', '/') + '.py'
                is_script = False
                
                if file_path in file_analyses:
                    analysis = file_analyses[file_path]
                    for func in analysis.get('functions', []):
                        if func['name'] == 'main' or 'main' in func['name'].lower():
                            is_script = True
                            break
                
                orphaned.append({
                    'module': module,
                    'file_path': file_path,
                    'is_script': is_script,
                    'recommendation': 'script' if is_script else 'potential_removal_candidate'
                })
        
        return orphaned
    
    def _analyze_cross_engine_flows(self, import_graph: Dict[str, Any], file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependencies that cross engine boundaries."""
        # Engine classification patterns
        engine_patterns = {
            'analysis': ['analysis', 'timeframe', 'session', 'lattice'],
            'learning': ['learning', 'tgat', 'discovery', 'graph'],
            'synthesis': ['synthesis', 'graduation', 'production'],
            'integration': ['integration', 'container', 'config'],
            'validation': ['validation', 'test'],
            'utilities': ['utilities', 'util', 'script', 'migration']
        }
        
        # Classify modules into engines
        module_engines = {}
        for module in import_graph:
            module_engines[module] = self._classify_module_engine(module, engine_patterns)
        
        # Find cross-engine flows
        cross_flows = defaultdict(list)
        flow_summary = defaultdict(int)
        
        for module, data in import_graph.items():
            module_engine = module_engines.get(module, 'unknown')
            
            for import_info in data['imports']:
                imported_module = import_info['module']
                imported_engine = module_engines.get(imported_module, 'unknown')
                
                if module_engine != imported_engine and module_engine != 'unknown' and imported_engine != 'unknown':
                    flow_key = f"{module_engine} -> {imported_engine}"
                    cross_flows[flow_key].append({
                        'from_module': module,
                        'to_module': imported_module,
                        'import_type': import_info['type']
                    })
                    flow_summary[flow_key] += 1
        
        return {
            'module_engine_classification': module_engines,
            'cross_engine_dependencies': dict(cross_flows),
            'flow_summary': dict(flow_summary),
            'problematic_flows': self._identify_problematic_flows(flow_summary),
            'recommended_flow_pattern': 'analysis -> learning -> synthesis'
        }
    
    def _classify_module_engine(self, module: str, engine_patterns: Dict[str, List[str]]) -> str:
        """Classify a module into an engine based on patterns."""
        module_lower = module.lower()
        
        for engine, patterns in engine_patterns.items():
            for pattern in patterns:
                if pattern in module_lower:
                    return engine
        
        return 'unknown'
    
    def _identify_problematic_flows(self, flow_summary: Dict[str, int]) -> List[Dict[str, Any]]:
        """Identify potentially problematic cross-engine flows."""
        problematic = []
        
        # Flows that go against the typical architecture pattern
        problematic_patterns = [
            'synthesis -> learning',  # Backward flow
            'synthesis -> analysis',  # Backward flow
            'learning -> analysis',   # Backward flow
        ]
        
        for flow, count in flow_summary.items():
            if any(pattern in flow for pattern in problematic_patterns):
                problematic.append({
                    'flow': flow,
                    'count': count,
                    'issue': 'backward_dependency',
                    'recommendation': 'Consider inverting dependency or using events/callbacks'
                })
            elif count > 10:
                problematic.append({
                    'flow': flow,
                    'count': count,
                    'issue': 'high_coupling',
                    'recommendation': 'Consider reducing cross-engine dependencies'
                })
        
        return problematic
    
    def _generate_dependency_statistics(self, import_graph: Dict[str, Any], file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall dependency statistics."""
        total_modules = len(import_graph)
        total_dependencies = sum(data['import_count'] for data in import_graph.values())
        
        # Calculate average dependencies per module
        avg_dependencies = total_dependencies / total_modules if total_modules > 0 else 0
        
        # Find modules with most dependencies
        most_dependencies = sorted(
            [(module, data['import_count']) for module, data in import_graph.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Find most depended upon modules
        most_depended_upon = sorted(
            [(module, data['imported_by_count']) for module, data in import_graph.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_modules': total_modules,
            'total_dependencies': total_dependencies,
            'average_dependencies_per_module': round(avg_dependencies, 2),
            'modules_with_most_dependencies': most_dependencies,
            'most_depended_upon_modules': most_depended_upon,
            'dependency_density': round(total_dependencies / (total_modules ** 2), 4) if total_modules > 0 else 0,
            'modules_with_zero_dependencies': len([m for m, d in import_graph.items() if d['import_count'] == 0])
        }
    
    def _find_critical_paths(self, import_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find critical dependency paths in the system."""
        critical_paths = []
        
        # Find longest dependency chains
        def find_longest_path(start_module: str, visited: Set[str] = None) -> List[str]:
            if visited is None:
                visited = set()
            
            if start_module in visited or start_module not in import_graph:
                return [start_module]
            
            visited.add(start_module)
            longest_path = [start_module]
            
            for import_info in import_graph[start_module]['imports']:
                imported_module = import_info['module']
                path = [start_module] + find_longest_path(imported_module, visited.copy())
                if len(path) > len(longest_path):
                    longest_path = path
            
            return longest_path
        
        # Find critical paths (longest chains)
        all_paths = []
        for module in import_graph:
            path = find_longest_path(module)
            if len(path) > 3:  # Only consider paths longer than 3 modules
                all_paths.append(path)
        
        # Sort by length and take top paths
        all_paths.sort(key=len, reverse=True)
        for i, path in enumerate(all_paths[:5]):  # Top 5 longest paths
            critical_paths.append({
                'rank': i + 1,
                'path': path,
                'length': len(path),
                'risk_level': 'high' if len(path) > 7 else 'medium'
            })
        
        return critical_paths
    
    def _identify_hub_modules(self, import_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify hub modules (high centrality in dependency graph)."""
        hubs = []
        
        for module, data in import_graph.items():
            # Calculate centrality as combination of imports and imported_by
            centrality = data['import_count'] + data['imported_by_count']
            
            if centrality > 8:  # Threshold for hub identification
                hubs.append({
                    'module': module,
                    'centrality': centrality,
                    'imports': data['import_count'],
                    'imported_by': data['imported_by_count'],
                    'hub_type': self._classify_hub_type(data)
                })
        
        return sorted(hubs, key=lambda x: x['centrality'], reverse=True)
    
    def _classify_hub_type(self, module_data: Dict[str, Any]) -> str:
        """Classify the type of hub based on dependency patterns."""
        imports = module_data['import_count']
        imported_by = module_data['imported_by_count']
        
        if imports > imported_by * 2:
            return 'consumer_hub'  # Imports much more than it's imported
        elif imported_by > imports * 2:
            return 'provider_hub'  # Imported much more than it imports
        else:
            return 'mediator_hub'  # Balanced import/export
    
    def _identify_leaf_modules(self, import_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify leaf modules (endpoints in dependency graph)."""
        leaves = []
        
        for module, data in import_graph.items():
            # Leaf modules have dependencies but nothing depends on them
            if data['import_count'] > 0 and data['imported_by_count'] == 0:
                leaves.append({
                    'module': module,
                    'import_count': data['import_count'],
                    'leaf_type': 'terminal'  # Has dependencies but no dependents
                })
            # Or modules that nothing depends on and have no dependencies
            elif data['import_count'] == 0 and data['imported_by_count'] == 0:
                leaves.append({
                    'module': module,
                    'import_count': 0,
                    'leaf_type': 'isolated'  # No connections at all
                })
        
        return leaves