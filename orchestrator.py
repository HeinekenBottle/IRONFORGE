"""
IRONFORGE Main Orchestrator
Coordinates learning, preservation, and synthesis
Enhanced with Performance Monitor for Sprint 2 tracking
"""
import os
import json
import pickle
import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Configuration system to eliminate hardcoded paths
from config import get_config

# Lazy loading imports to resolve timeout issues
from integration.ironforge_container import get_ironforge_container
from synthesis.pattern_graduation import PatternGraduation
from learning.simple_event_clustering import analyze_time_patterns

class IRONFORGE:
    """Main orchestrator for discovery system"""
    
    def __init__(self, data_path: Optional[str] = None, use_enhanced: bool = True,
                 enable_performance_monitoring: bool = True, config_file: Optional[str] = None):
        # Initialize configuration system (NO HARDCODED PATHS)
        self.config = get_config(config_file)

        # Use configured paths instead of hardcoded ones
        self.data_path = data_path or self.config.get_data_path()
        self.use_enhanced = use_enhanced
        self.enhanced_mode = use_enhanced

        # Initialize logging
        self.logger = logging.getLogger('ironforge.orchestrator')

        # Initialize lazy loading container
        self.container = get_ironforge_container()

        # Components will be loaded lazily on first access
        self._graph_builder = None
        self._discovery_engine = None
        self._graduation_pipeline = None

        # Preservation paths from configuration
        self.preservation_path = self.config.get_preservation_path()
        self.graphs_path = self.config.get_graphs_path()
        self.embeddings_path = self.config.get_embeddings_path()
        
        # Performance monitoring for Sprint 2
        self.performance_monitor: Optional['PerformanceMonitor'] = None
        if enable_performance_monitoring:
            try:
                from performance_monitor import PerformanceMonitor, create_graph_analysis
                self.performance_monitor = PerformanceMonitor(regression_threshold=0.15)
                self.performance_monitor.hook_into_orchestrator(self)
                print("🔗 Performance monitoring enabled for Sprint 2 tracking")
            except ImportError as e:
                print(f"⚠️ Performance monitoring not available: {e}")
                self.performance_monitor = None
    
    @property
    def graph_builder(self):
        """Lazy loading graph builder - NO FALLBACKS"""
        if self._graph_builder is None:
            if self.use_enhanced:
                try:
                    self._graph_builder = self.container.get_enhanced_graph_builder()
                except Exception as e:
                    # NO FALLBACKS: Fail explicitly with clear error message
                    raise RuntimeError(
                        f"ENHANCED MODE FAILURE - Enhanced graph builder failed to load: {e}\n"
                        f"SOLUTION: Fix the enhanced graph builder or set use_enhanced=False explicitly\n"
                        f"NO FALLBACKS: Enhanced mode must work or fail explicitly"
                    ) from e
            else:
                # Basic mode explicitly requested
                from learning.graph_builder import IRONFORGEGraphBuilder
                self._graph_builder = IRONFORGEGraphBuilder()
        return self._graph_builder
    
    @property 
    def discovery_engine(self):
        """Lazy loading TGAT discovery engine"""
        if self._discovery_engine is None:
            self._discovery_engine = self.container.get_tgat_discovery()
        return self._discovery_engine
    
    @property
    def graduation_pipeline(self):
        """Lazy loading graduation pipeline"""
        if self._graduation_pipeline is None:
            self._graduation_pipeline = PatternGraduation()
        return self._graduation_pipeline
        
    def process_sessions(self, session_files: List[str]) -> Dict:
        """Process multiple session files through IRONFORGE at full capability"""
        results = {
            'sessions_processed': 0,
            'patterns_discovered': [],
            'graphs_preserved': [],
            'skipped_sessions': [],
            'processing_errors': []
        }
        
        # NO ARTIFICIAL LIMITS - Process all sessions at full archaeological capability
        print(f"🏛️ Processing {len(session_files)} sessions with full IRONFORGE capability")
        
        session_graphs = []
        
        # Build graphs from sessions
        for i, session_file in enumerate(session_files):
            print(f"Processing session {i+1}/{len(session_files)}: {Path(session_file).name}")
            
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Build full-preservation graph (enhanced or basic)
                if self.enhanced_mode:
                    graph, session_metadata = self.graph_builder.build_rich_graph(session_data, session_file_path=session_file)
                else:
                    graph = self.graph_builder.build_graph(session_data)
                    session_metadata = {}
                
                # Skip empty graphs to prevent TGAT failures
                if graph.get('metadata', {}).get('total_nodes', 0) == 0:
                    print(f"⚠️ Skipping {Path(session_file).name}: empty graph (no valid nodes after validation)")
                    results['skipped_sessions'].append({'file': session_file, 'reason': 'empty_graph_after_validation'})
                    continue
                
                # Store graph for batch processing (constant filtering will be done globally)
                session_graphs.append(graph)
                
                # Analyze time patterns (Simple Event-Time Clustering + Cross-TF Mapping)
                time_patterns = self._analyze_time_patterns(graph, session_file)
                if 'time_patterns' not in session_metadata:
                    session_metadata['time_patterns'] = time_patterns
                
                # Preserve graph with session metadata (including time patterns)
                graph_with_metadata = graph.copy()
                graph_with_metadata['session_metadata'] = session_metadata
                pickle_path = self._preserve_graph(graph_with_metadata, session_file)
                results['graphs_preserved'].append(pickle_path)
                
                print(f"✅ Processed {Path(session_file).name}: {graph.get('metadata', {}).get('total_nodes', 0)} nodes")
                
            except ValueError as e:
                # Data validation errors - log and skip this session
                print(f"⚠️ Skipping {Path(session_file).name}: {e}")
                results['skipped_sessions'].append({'file': session_file, 'reason': str(e)})
                continue
            except Exception as e:
                # Other errors - fail fast to expose the real issue
                print(f"❌ FATAL ERROR processing {session_file}: {e}")
                results['processing_errors'].append({'file': session_file, 'error': str(e)})
                raise
        
        # Global constant feature filtering and TGAT format conversion
        print(f"🔧 Applying global constant filtering across {len(session_graphs)} sessions...")
        tgat_session_graphs = self._batch_process_to_tgat_format(session_graphs)
        
        # Train discovery engine
        print(f"Training TGAT on {len(tgat_session_graphs)} sessions...")
        self.discovery_engine.train_on_historical(tgat_session_graphs, epochs=20)
        
        # Save discoveries
        self.discovery_engine.save_discoveries(self.preservation_path)
        results['patterns_discovered'] = self.discovery_engine.discovered_patterns
        results['sessions_processed'] = len(session_files)
        
        return results
    
    def validate_discoveries(self, historical_sessions: List[str]) -> Dict:
        """Validate discovered patterns against baseline"""
        # Load historical data
        historical_data = []
        for session_file in historical_sessions:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            if self.enhanced_mode:
                graph = self.graph_builder.build_rich_graph(session_data, session_file_path=session_file)
            else:
                graph = self.graph_builder.build_graph(session_data)
            historical_data.append(graph)
        
        # Validate each discovered pattern
        validation_results = []
        for pattern in self.discovery_engine.discovered_patterns:
            result = self.graduation_pipeline.validate_pattern(pattern, historical_data)
            validation_results.append(result)
            
            if result['status'] == 'VALIDATED':
                print(f"✅ Pattern validated: {pattern['type']} "
                      f"(+{result['improvement']:.2%} improvement)")
        
        # Save validated patterns
        self.graduation_pipeline.save_validated_patterns(self.preservation_path)
        
        return {
            'total_patterns': len(self.discovery_engine.discovered_patterns),
            'validated': len([r for r in validation_results if r['status'] == 'VALIDATED']),
            'results': validation_results
        }
    
    def _preserve_graph(self, graph: Dict, source_file: str) -> str:
        """
        Preserve complete graph data
        
        Returns:
            str: Path to the saved pickle file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.basename(source_file).replace('.json', '')
        
        output_path = os.path.join(
            self.graphs_path,
            f"{filename}_graph_{timestamp}.pkl"
        )
        
        # Create pickle-serialized version (handles complex Python objects)
        try:
            success = self._safe_pickle_dump(graph, output_path, filename)
            if not success:
                # NO SILENT ERRORS: Log preservation failure as ERROR, not warning
                import logging
                logger = logging.getLogger('ironforge.preservation')
                logger.error(f"PRESERVATION FAILURE: Could not preserve graph {filename} to {output_path}")
                logger.error("IMPACT: Graph data will be lost, affecting future analysis")
                logger.error("ACTION REQUIRED: Check disk space, permissions, and file system health")
                raise RuntimeError(f"Graph preservation failed for {filename}")
            return output_path
        except Exception as e:
            # NO SILENT ERRORS: Log as ERROR and re-raise to make failure visible
            import logging
            logger = logging.getLogger('ironforge.preservation')
            logger.error(f"CRITICAL PRESERVATION ERROR: Graph preservation failed for {filename}: {e}")
            logger.error(f"OUTPUT PATH: {output_path}")
            logger.error("IMPACT: Complete loss of graph data for this session")
            raise RuntimeError(f"Graph preservation failed for {filename}: {e}") from e
    
    def freeze_for_production(self):
        """Prepare validated patterns for production use"""
        print("Freezing IRONFORGE for production...")
        
        # Freeze TGAT model
        self.discovery_engine.freeze_for_prediction()
        
        # Export validated patterns as simple features
        validated_features = []
        for pattern in self.graduation_pipeline.validated_patterns:
            feature = self.graduation_pipeline.convert_to_simple_feature(pattern)
            validated_features.append(feature)
        
        # Save for production
        output_path = os.path.join(
            self.preservation_path,
            'production_features.json'
        )
        with open(output_path, 'w') as f:
            json.dump(validated_features, f, indent=2)
        
        print(f"Exported {len(validated_features)} features for production")
        return validated_features
    
    def generate_performance_report(self, 
                                  orchestrator_results: Dict,
                                  validation_results: Optional[Dict] = None,
                                  processed_graphs: Optional[List] = None,
                                  save_report: bool = True) -> Optional[Dict]:
        """
        Generate comprehensive Sprint 2 performance report with regression analysis
        
        Args:
            orchestrator_results: Results from process_sessions()
            validation_results: Results from validate_discoveries()  
            processed_graphs: List of processed graph data for analysis
            save_report: Whether to save report to file
        
        Returns:
            Performance report dict, None if monitoring not enabled
        """
        if not self.performance_monitor:
            print("⚠️ Performance monitoring not enabled - cannot generate report")
            return None
        
        try:
            # Create graph analysis for Sprint 2 metrics
            if processed_graphs:
                from performance_monitor import create_graph_analysis
                graph_analysis = create_graph_analysis(self.graph_builder, processed_graphs)
            else:
                # Attempt to extract from recent processing
                graph_analysis = {
                    'total_edges': 0,
                    'structural_edges_created': 0, 
                    'feature_dimensions': 37 if self.enhanced_mode else 4,
                    'edge_type_count': 4 if self.enhanced_mode else 3,
                    'regime_clusters_identified': 0,
                    'precursor_patterns_detected': 0
                }
            
            # Collect comprehensive metrics
            metrics = self.performance_monitor.collect_metrics(
                orchestrator_results, 
                validation_results, 
                graph_analysis
            )
            
            # Generate detailed report with regression analysis
            report = self.performance_monitor.generate_performance_report(
                metrics, 
                include_regression=True
            )
            
            # Save report if requested
            if save_report:
                report_path = os.path.join(
                    self.preservation_path,
                    f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"📊 Performance report saved to {report_path}")
            
            # Print summary
            self._print_performance_summary(report)
            
            return report
            
        except Exception as e:
            print(f"❌ Failed to generate performance report: {e}")
            if self.performance_monitor:
                self.performance_monitor.logger.error(f"Performance report generation failed: {e}")
            return None
    
    def _print_performance_summary(self, report: Dict) -> None:
        """Print concise performance summary to console"""
        print("\n" + "="*60)
        print("🚀 IRONFORGE SPRINT 2 PERFORMANCE SUMMARY")
        print("="*60)
        
        # Processing performance
        monitoring = report.get('monitoring_session', {})
        print(f"⏱️  Total Processing Time: {monitoring.get('total_processing_time_sec', 0):.2f}s")
        print(f"💾 Peak Memory Usage: {monitoring.get('peak_memory_usage_mb', 0):.1f} MB")
        
        # Discovery performance
        discovery = report.get('discovery_performance', {})
        print(f"📊 Sessions Processed: {discovery.get('sessions_processed', 0)}")
        print(f"🔍 Patterns Discovered: {discovery.get('patterns_discovered', 0)}")
        print(f"📈 Discovery Rate: {discovery.get('discovery_rate_per_session', 0):.2f} patterns/session")
        print(f"✅ Validation Success: {discovery.get('validation_success_rate', 0):.1%}")
        
        # Sprint 2 enhancements
        enhancements = report.get('sprint_2_enhancements', {})
        print(f"🏗️  Structural Edges: {enhancements.get('structural_edges_created', 0)}")
        print(f"📊 Edge Ratio: {enhancements.get('structural_edge_ratio', 0):.1%}")
        print(f"🎯 Regime Clusters: {enhancements.get('regime_clusters_identified', 0)}")
        print(f"🔮 Precursor Patterns: {enhancements.get('precursor_patterns_detected', 0)}")
        
        # Quality gates
        quality = report.get('quality_gates', {})
        feature_status = quality.get('feature_dimensions', {}).get('status', 'UNKNOWN')
        edge_status = quality.get('edge_types', {}).get('status', 'UNKNOWN')
        validation_status = quality.get('validation_accuracy', {}).get('status', 'UNKNOWN')
        tgat_status = quality.get('tgat_compatibility', {}).get('status', 'UNKNOWN')
        
        print(f"🎯 Quality Gates: Features={feature_status} | Edges={edge_status} | "
              f"Validation={validation_status} | TGAT={tgat_status}")
        
        # Regression analysis
        regression = report.get('regression_analysis', {})
        if regression and regression.get('status') == 'pass':
            print("✅ PERFORMANCE REGRESSION CHECK: PASSED")
        elif regression and regression.get('status') == 'fail':
            print("❌ PERFORMANCE REGRESSION CHECK: FAILED")
            regressions = regression.get('regressions', [])
            if regressions:
                print("   Regressions detected:")
                for reg in regressions[:3]:  # Show first 3
                    print(f"   - {reg.get('metric', 'unknown')}: {reg.get('regression_pct', 0):.1f}% worse")
        
        print("="*60)
    
    def _analyze_time_patterns(self, graph: Dict, session_file: str) -> Dict:
        """
        Analyze time patterns in session events with cross-TF mapping
        
        Integrates Simple Event-Time Clustering + Cross-TF Mapping for temporal intelligence.
        Provides "when events cluster" + "what HTF context" analysis with <0.05s overhead.
        
        Args:
            graph: Enhanced graph data from graph builder
            session_file: Session file path for context
            
        Returns:
            Time pattern analysis results for session metadata
        """
        # NO FALLBACKS: If this fails, we need to fix the root cause
        return analyze_time_patterns(graph, session_file, time_bin_minutes=5)
    
    def _batch_process_to_tgat_format(self, session_graphs: List[Dict]) -> List[Tuple]:
        """
        Apply global constant feature filtering and convert to TGAT format
        
        This ensures all sessions have the same feature dimensionality by filtering
        features that are constant across ALL sessions, not per-session.
        
        Args:
            session_graphs: List of enhanced graphs from graph builder
            
        Returns:
            List of TGAT format tuples with consistent dimensionality
        """
        if not session_graphs:
            return []
        
        print(f"📊 Collecting features from {len(session_graphs)} sessions...")
        
        # Step 1: Collect all raw feature tensors
        all_raw_features = []
        graph_metadata = []
        
        for graph in session_graphs:
            if 'rich_node_features' in graph and graph['rich_node_features']:
                # Convert rich features to raw tensor
                features = graph['rich_node_features']
                X_raw = torch.stack([feature.to_tensor() for feature in features])
                all_raw_features.append(X_raw)
                graph_metadata.append(graph)
            else:
                print(f"⚠️ Skipping graph with no rich_node_features")
        
        if not all_raw_features:
            print(f"❌ No valid feature tensors found")
            return []
        
        # Step 2: Global constant feature detection
        print(f"🔍 Detecting globally constant features...")
        global_constant_mask = self._detect_global_constant_features(all_raw_features)
        non_constant_count = (~global_constant_mask).sum().item()
        constant_count = global_constant_mask.sum().item()
        
        print(f"   Original: {global_constant_mask.shape[0]}D")
        print(f"   Constant: {constant_count}D (filtered out)")
        print(f"   Variable: {non_constant_count}D (kept for training)")
        
        # Step 3: Apply global filtering and convert to TGAT format
        tgat_graphs = []
        for i, (graph, X_raw) in enumerate(zip(graph_metadata, all_raw_features)):
            # Apply global constant mask
            X_filtered = X_raw[:, ~global_constant_mask]
            
            # Convert to TGAT format with globally filtered features
            if self.enhanced_mode:
                _, edge_index, edge_times, metadata, edge_attr = self.graph_builder.to_tgat_format(graph)
                # Replace the features with globally filtered ones
                tgat_graphs.append((X_filtered, edge_index, edge_times, metadata, edge_attr))
            else:
                _, edge_index, edge_times, metadata = self.graph_builder.to_tgat_format(graph)
                tgat_graphs.append((X_filtered, edge_index, edge_times, metadata))
            
            # Verify consistent dimensions
            if i == 0:
                expected_features = X_filtered.shape[1]
            elif X_filtered.shape[1] != expected_features:
                raise RuntimeError(f"Inconsistent feature dimensions: session {i} has {X_filtered.shape[1]}D, expected {expected_features}D")
        
        print(f"✅ All sessions processed with consistent {expected_features}D features")
        return tgat_graphs
    
    def _detect_global_constant_features(self, all_raw_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Detect features that are constant across ALL sessions
        
        A feature is considered globally constant if it has zero variance
        when computed across all nodes from all sessions.
        
        Args:
            all_raw_features: List of feature tensors from all sessions
            
        Returns:
            Boolean mask where True indicates constant features
        """
        # Concatenate all features across all sessions
        all_features_combined = torch.cat(all_raw_features, dim=0)
        
        # Calculate global variance across all nodes from all sessions
        global_variances = torch.var(all_features_combined, dim=0, unbiased=False)
        
        # Features with zero variance are constant
        constant_mask = global_variances == 0.0
        
        return constant_mask
    
    def _safe_pickle_dump(self, graph, output_path: str, filename: str) -> bool:
        """Pickle graphs with proper memory management - NO CHUNKED WORKAROUNDS"""
        try:
            # Proper tensor serialization for archaeological graphs
            with open(output_path, 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            # NO SILENT ERRORS: Fail explicitly to expose root cause
            import logging
            logger = logging.getLogger('ironforge.preservation')
            logger.error(f"GRAPH PRESERVATION FAILED: {filename}: {e}")
            logger.error(f"Output path: {output_path}")
            logger.error("ROOT CAUSE ANALYSIS REQUIRED:")
            logger.error("1. Check graph structure for memory leaks")
            logger.error("2. Verify tensor cleanup in graph builder")
            logger.error("3. Check disk space and permissions")
            logger.error("4. NO FALLBACKS: Fix the root cause")
            raise RuntimeError(f"Graph preservation failed for {filename}: {e}") from e
    
    # CHUNKED SERIALIZATION REMOVED - Fixed root causes instead of symptoms


if __name__ == "__main__":
    # Example usage with Sprint 2 performance monitoring
    print("🚀 IRONFORGE Sprint 2 - Structural Intelligence with Performance Monitoring")
    print("=" * 70)
    
    # Initialize with performance monitoring enabled (NO HARDCODED PATHS)
    forge = IRONFORGE(enable_performance_monitoring=True)

    # Use configured session data path
    from pathlib import Path
    session_data_path = Path(forge.config.get_session_data_path())

    if session_data_path.exists():
        session_files = list(session_data_path.rglob('*.json'))[:20]  # Test with 20 sessions (recursive)
        print(f"Found {len(session_files)} session files for testing")
    else:
        # NO FALLBACKS: If session data path doesn't exist, fail explicitly
        raise RuntimeError(f"Session data path does not exist: {session_data_path}")
        print(f"ROOT CAUSE: Configure IRONFORGE_SESSION_DATA_PATH environment variable")
    
    if session_files and (isinstance(session_files[0], Path) or os.path.exists(session_files[0])):
        print(f"\n🔄 Processing {len(session_files)} sessions with performance monitoring...")
        
        # Process sessions - automatically monitored via hooks
        results = forge.process_sessions([str(f) for f in session_files])
        print(f"✅ Processed {results['sessions_processed']} sessions")
        print(f"🔍 Discovered {len(results['patterns_discovered'])} patterns")
        
        # Validate discoveries - automatically monitored via hooks
        validation = forge.validate_discoveries([str(f) for f in session_files])
        print(f"✅ Validated {validation['validated']}/{validation['total_patterns']} patterns")
        
        # Generate comprehensive performance report
        print(f"\n📊 Generating Sprint 2 performance report...")
        report = forge.generate_performance_report(results, validation)
        
        if report:
            print("\n🎯 Sprint 2 Enhancement Status:")
            enhancements = report.get('sprint_2_enhancements', {})
            effectiveness = enhancements.get('enhancement_effectiveness', {})
            
            for enhancement, status in effectiveness.items():
                status_icon = "✅" if status in ['EXCELLENT', 'GOOD'] else "⚠️" if status == 'MINIMAL' else "❌"
                print(f"   {status_icon} {enhancement.replace('_', ' ').title()}: {status}")
        
        # Freeze for production
        print(f"\n🧊 Freezing for production...")
        forge.freeze_for_production()
        
        print("\n✅ IRONFORGE Sprint 2 processing complete with performance monitoring!")
        
    else:
        print("⚠️  No valid session files found.")
        print("Expected session files in: /Users/jack/IRONPULSE/data/sessions/level_1/")
        print("\nTo test performance monitoring manually:")
        print("```python")
        print("from orchestrator import IRONFORGE")
        print("forge = IRONFORGE(enable_performance_monitoring=True)")
        print("results = forge.process_sessions(your_session_files)")
        print("validation = forge.validate_discoveries(your_session_files)")
        print("report = forge.generate_performance_report(results, validation)")
        print("```")
