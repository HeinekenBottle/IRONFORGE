#!/usr/bin/env python3
"""
IRONFORGE Full Session Analysis
===============================

Comprehensive analysis script to run IRONFORGE over all logged sessions
and extract patterns, data, timing analysis, and generate visualizations.

Usage:
    python3 run_full_session_analysis.py

Outputs:
    - session_analysis_results.json: Complete analysis data
    - session_patterns_summary.csv: Pattern summary table
    - visualizations/: Graphs and charts
    - reports/: Detailed analysis reports
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import IRONFORGE components
from learning.enhanced_graph_builder import EnhancedGraphBuilder
from pattern_correlation_visualizer import PatternCorrelationVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('session_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SessionAnalyzer:
    """Comprehensive session analysis with pattern extraction and visualization"""
    
    def __init__(self, sessions_dir: str = "enhanced_sessions_with_relativity"):
        self.sessions_dir = sessions_dir
        self.builder = EnhancedGraphBuilder()
        self.results = []
        self.patterns = []
        
        # Create output directories
        Path("results").mkdir(exist_ok=True)
        Path("visualizations").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        
    def analyze_all_sessions(self) -> Dict[str, Any]:
        """Run IRONFORGE analysis on all sessions"""
        
        logger.info("🚀 Starting comprehensive session analysis...")
        
        # Get all session files
        session_files = [f for f in os.listdir(self.sessions_dir) if f.endswith('.json')]
        logger.info(f"Found {len(session_files)} session files")
        
        total_processing_time = 0
        successful_sessions = 0
        failed_sessions = []
        
        for i, session_file in enumerate(session_files, 1):
            logger.info(f"Processing {i}/{len(session_files)}: {session_file}")
            
            try:
                start_time = datetime.now()
                
                # Load and analyze session
                session_data = self._load_session(session_file)
                analysis_result = self._analyze_single_session(session_file, session_data)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                analysis_result['processing_time'] = processing_time
                total_processing_time += processing_time
                
                self.results.append(analysis_result)
                successful_sessions += 1
                
                logger.info(f"✅ Completed {session_file} in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {session_file}: {str(e)}")
                failed_sessions.append((session_file, str(e)))
                continue
        
        # Generate summary statistics
        summary = {
            'total_sessions': len(session_files),
            'successful_sessions': successful_sessions,
            'failed_sessions': len(failed_sessions),
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / successful_sessions if successful_sessions > 0 else 0,
            'failed_session_details': failed_sessions,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 Analysis complete: {successful_sessions}/{len(session_files)} sessions processed")
        logger.info(f"⏱️ Total time: {total_processing_time:.2f}s, Average: {summary['average_processing_time']:.2f}s/session")
        
        return summary
    
    def _load_session(self, session_file: str) -> Dict[str, Any]:
        """Load session data from file"""
        with open(os.path.join(self.sessions_dir, session_file), 'r') as f:
            return json.load(f)
    
    def _analyze_single_session(self, session_file: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single session and extract all relevant data"""
        
        # Build rich graph
        graph, metadata = self.builder.build_rich_graph(session_data)
        
        # Convert to TGAT format (with constant feature filtering)
        X, edge_index, edge_times, enhanced_metadata, edge_attr = self.builder.to_tgat_format(graph)
        
        # Extract semantic patterns
        semantic_patterns = self._extract_semantic_patterns(graph, metadata)
        
        # Extract timing analysis
        timing_analysis = self._extract_timing_analysis(graph, metadata)
        
        # Extract market characteristics
        market_analysis = self._extract_market_analysis(graph, metadata, enhanced_metadata)
        
        return {
            'session_file': session_file,
            'session_metadata': metadata,
            'graph_statistics': {
                'total_nodes': len(graph['rich_node_features']),
                'total_edges': len(edge_times),
                'node_feature_dims': X.shape[1] if len(X.shape) > 1 else 0,
                'edge_feature_dims': edge_attr.shape[1] if len(edge_attr.shape) > 1 else 0,
                'constant_features_filtered': enhanced_metadata.get('constant_features', {}).get('metadata_only_count', 0)
            },
            'semantic_patterns': semantic_patterns,
            'timing_analysis': timing_analysis,
            'market_analysis': market_analysis,
            'enhanced_metadata': enhanced_metadata
        }
    
    def _extract_semantic_patterns(self, graph: Dict, metadata: Dict) -> Dict[str, Any]:
        """Extract semantic event patterns from the graph"""
        
        nodes = graph['rich_node_features']
        if not nodes:
            return {}
        
        # Count semantic events (enhanced with complete market cycle detection)
        fvg_events = sum(1 for node in nodes if node.fvg_redelivery_flag > 0)
        expansion_events = sum(1 for node in nodes if node.expansion_phase_flag > 0)
        consolidation_events = sum(1 for node in nodes if node.consolidation_flag > 0)
        retracement_events = sum(1 for node in nodes if node.retracement_flag > 0)
        reversal_events = sum(1 for node in nodes if node.reversal_flag > 0)
        liquidity_sweeps = sum(1 for node in nodes if node.liq_sweep_flag > 0)
        pd_array_interactions = sum(1 for node in nodes if node.pd_array_interaction_flag > 0)
        
        # Session phase distribution
        phase_distribution = {
            'open_phase': sum(1 for node in nodes if node.phase_open > 0.5),
            'mid_phase': sum(1 for node in nodes if node.phase_mid > 0.5),
            'close_phase': sum(1 for node in nodes if node.phase_close > 0.5)
        }
        
        # Event type distribution
        event_types = {}
        for node in nodes:
            event_types[node.event_type_id] = event_types.get(node.event_type_id, 0) + 1
        
        return {
            'semantic_events': {
                'fvg_redelivery': fvg_events,
                'expansion_phase': expansion_events,
                'consolidation': consolidation_events,
                'retracement': retracement_events,
                'reversal': reversal_events,
                'liquidity_sweeps': liquidity_sweeps,
                'pd_array_interactions': pd_array_interactions
            },
            'semantic_percentages': {
                'fvg_redelivery_pct': (fvg_events / len(nodes)) * 100,
                'expansion_phase_pct': (expansion_events / len(nodes)) * 100,
                'consolidation_pct': (consolidation_events / len(nodes)) * 100,
                'retracement_pct': (retracement_events / len(nodes)) * 100,
                'reversal_pct': (reversal_events / len(nodes)) * 100,
                'liquidity_sweeps_pct': (liquidity_sweeps / len(nodes)) * 100,
                'pd_array_interactions_pct': (pd_array_interactions / len(nodes)) * 100
            },
            'phase_distribution': phase_distribution,
            'event_type_distribution': event_types,
            'total_semantic_events': fvg_events + expansion_events + consolidation_events + retracement_events + reversal_events + liquidity_sweeps + pd_array_interactions
        }
    
    def _extract_timing_analysis(self, graph: Dict, metadata: Dict) -> Dict[str, Any]:
        """Extract timing and temporal analysis"""
        
        nodes = graph['rich_node_features']
        if not nodes:
            return {}
        
        # Time distribution
        times = [node.time_minutes for node in nodes]
        
        return {
            'session_duration_minutes': max(times) - min(times) if times else 0,
            'event_time_distribution': {
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'mean_time': np.mean(times) if times else 0,
                'std_time': np.std(times) if times else 0
            },
            'temporal_density': len(nodes) / (max(times) - min(times)) if times and max(times) != min(times) else 0,
            'session_timing': {
                'session_start': metadata.get('session_start', 'unknown'),
                'session_end': metadata.get('session_end', 'unknown'),
                'session_date': metadata.get('session_date', 'unknown')
            }
        }
    
    def _extract_market_analysis(self, graph: Dict, metadata: Dict, enhanced_metadata: Dict) -> Dict[str, Any]:
        """Extract market characteristics and analysis"""
        
        nodes = graph['rich_node_features']
        if not nodes:
            return {}
        
        # Price analysis
        prices = [node.normalized_price for node in nodes]
        price_changes = [node.pct_from_open for node in nodes]
        energy_states = [node.energy_state for node in nodes]
        
        return {
            'price_analysis': {
                'price_range': max(prices) - min(prices) if prices else 0,
                'mean_price': np.mean(prices) if prices else 0,
                'price_volatility': np.std(prices) if prices else 0,
                'max_price_change': max(price_changes) if price_changes else 0,
                'min_price_change': min(price_changes) if price_changes else 0
            },
            'energy_analysis': {
                'mean_energy': np.mean(energy_states) if energy_states else 0,
                'energy_volatility': np.std(energy_states) if energy_states else 0,
                'max_energy': max(energy_states) if energy_states else 0,
                'min_energy': min(energy_states) if energy_states else 0
            },
            'market_characteristics': metadata.get('market_characteristics', {}),
            'session_quality': metadata.get('session_quality', 'unknown'),
            'constant_features_analysis': enhanced_metadata.get('constant_features', {})
        }

    def generate_visualizations(self) -> None:
        """Generate enhanced pattern correlation visualizations from analysis results"""

        if not self.results:
            logger.warning("No results to visualize")
            return

        logger.info("📊 Generating enhanced pattern correlation visualizations...")

        # Use the new pattern correlation visualizer instead of generic charts
        visualizer = PatternCorrelationVisualizer()
        visualizer.generate_correlation_visualizations(self.results)

        # Also create a basic summary DataFrame for CSV export
        df_data = []
        for result in self.results:
            session_name = result['session_metadata'].get('session_name', 'unknown')
            session_date = result['session_metadata'].get('session_date', 'unknown')

            row = {
                'session_file': result['session_file'],
                'session_name': session_name,
                'session_date': session_date,
                'session_quality': result['market_analysis'].get('session_quality', 'unknown'),
                'total_nodes': result['graph_statistics']['total_nodes'],
                'total_edges': result['graph_statistics']['total_edges'],
                'processing_time': result.get('processing_time', 0),
                'fvg_events': result['semantic_patterns'].get('semantic_events', {}).get('fvg_redelivery', 0),
                'expansion_events': result['semantic_patterns'].get('semantic_events', {}).get('expansion_phase', 0),
                'consolidation_events': result['semantic_patterns'].get('semantic_events', {}).get('consolidation', 0),
                'retracement_events': result['semantic_patterns'].get('semantic_events', {}).get('retracement', 0),
                'reversal_events': result['semantic_patterns'].get('semantic_events', {}).get('reversal', 0),
                'fvg_pct': result['semantic_patterns'].get('semantic_percentages', {}).get('fvg_redelivery_pct', 0),
                'expansion_pct': result['semantic_patterns'].get('semantic_percentages', {}).get('expansion_phase_pct', 0),
                'retracement_pct': result['semantic_patterns'].get('semantic_percentages', {}).get('retracement_pct', 0),
                'reversal_pct': result['semantic_patterns'].get('semantic_percentages', {}).get('reversal_pct', 0),
                'price_volatility': result['market_analysis'].get('price_analysis', {}).get('price_volatility', 0),
                'mean_energy': result['market_analysis'].get('energy_analysis', {}).get('mean_energy', 0),
                'constant_features_filtered': result['graph_statistics']['constant_features_filtered']
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Save enhanced summary CSV with new semantic events
        df.to_csv('results/session_patterns_summary.csv', index=False)

        logger.info("📊 Enhanced pattern correlation visualizations generated")
        logger.info("📋 Enhanced summary CSV saved to results/session_patterns_summary.csv")

    def generate_reports(self) -> None:
        """Generate detailed analysis reports"""

        if not self.results:
            logger.warning("No results to report")
            return

        logger.info("📝 Generating analysis reports...")

        # Aggregate statistics
        total_nodes = sum(r['graph_statistics']['total_nodes'] for r in self.results)
        total_edges = sum(r['graph_statistics']['total_edges'] for r in self.results)
        total_fvg_events = sum(r['semantic_patterns'].get('semantic_events', {}).get('fvg_redelivery', 0) for r in self.results)
        total_expansion_events = sum(r['semantic_patterns'].get('semantic_events', {}).get('expansion_phase', 0) for r in self.results)

        # Session type analysis
        session_types = {}
        for result in self.results:
            session_name = result['session_metadata'].get('session_name', 'unknown')
            if session_name not in session_types:
                session_types[session_name] = {
                    'count': 0,
                    'total_nodes': 0,
                    'total_fvg_events': 0,
                    'total_expansion_events': 0,
                    'avg_processing_time': 0
                }

            session_types[session_name]['count'] += 1
            session_types[session_name]['total_nodes'] += result['graph_statistics']['total_nodes']
            session_types[session_name]['total_fvg_events'] += result['semantic_patterns'].get('semantic_events', {}).get('fvg_redelivery', 0)
            session_types[session_name]['total_expansion_events'] += result['semantic_patterns'].get('semantic_events', {}).get('expansion_phase', 0)
            session_types[session_name]['avg_processing_time'] += result.get('processing_time', 0)

        # Calculate averages
        for session_type in session_types:
            count = session_types[session_type]['count']
            session_types[session_type]['avg_nodes'] = session_types[session_type]['total_nodes'] / count
            session_types[session_type]['avg_fvg_events'] = session_types[session_type]['total_fvg_events'] / count
            session_types[session_type]['avg_expansion_events'] = session_types[session_type]['total_expansion_events'] / count
            session_types[session_type]['avg_processing_time'] = session_types[session_type]['avg_processing_time'] / count

        # Generate report
        report = f"""
IRONFORGE Comprehensive Session Analysis Report
==============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
Total Sessions Analyzed: {len(self.results)}
Total Nodes Processed: {total_nodes:,}
Total Edges Created: {total_edges:,}
Total FVG Redelivery Events: {total_fvg_events:,}
Total Expansion Phase Events: {total_expansion_events:,}

SEMANTIC DISCOVERY PERFORMANCE
------------------------------
Average FVG Events per Session: {total_fvg_events / len(self.results):.1f}
Average Expansion Events per Session: {total_expansion_events / len(self.results):.1f}
Semantic Event Discovery Rate: {((total_fvg_events + total_expansion_events) / total_nodes * 100):.2f}%

SESSION TYPE ANALYSIS
--------------------
"""

        for session_type, stats in session_types.items():
            report += f"""
{session_type.upper()} Sessions:
  - Count: {stats['count']} sessions
  - Avg Nodes: {stats['avg_nodes']:.1f}
  - Avg FVG Events: {stats['avg_fvg_events']:.1f}
  - Avg Expansion Events: {stats['avg_expansion_events']:.1f}
  - Avg Processing Time: {stats['avg_processing_time']:.2f}s
"""

        report += f"""
PERFORMANCE METRICS
------------------
Total Processing Time: {sum(r.get('processing_time', 0) for r in self.results):.2f} seconds
Average Processing Time: {sum(r.get('processing_time', 0) for r in self.results) / len(self.results):.2f} seconds/session
Nodes per Second: {total_nodes / sum(r.get('processing_time', 0) for r in self.results):.1f}

CONSTANT FEATURE FILTERING EFFICIENCY
------------------------------------
Total Features Filtered: {sum(r['graph_statistics']['constant_features_filtered'] for r in self.results)}
Average Features Filtered per Session: {sum(r['graph_statistics']['constant_features_filtered'] for r in self.results) / len(self.results):.1f}
Feature Reduction Efficiency: {(sum(r['graph_statistics']['constant_features_filtered'] for r in self.results) / (len(self.results) * 45)) * 100:.1f}%

ARCHAEOLOGICAL DISCOVERY INSIGHTS
---------------------------------
Sessions with High FVG Activity (>20 events): {sum(1 for r in self.results if r['semantic_patterns'].get('semantic_events', {}).get('fvg_redelivery', 0) > 20)}
Sessions with High Expansion Activity (>15 events): {sum(1 for r in self.results if r['semantic_patterns'].get('semantic_events', {}).get('expansion_phase', 0) > 15)}
Most Active Session Type: {max(session_types.keys(), key=lambda x: session_types[x]['avg_fvg_events'] + session_types[x]['avg_expansion_events'])}

RECOMMENDATIONS
--------------
1. Focus archaeological discovery on {max(session_types.keys(), key=lambda x: session_types[x]['avg_fvg_events'])} sessions for FVG patterns
2. Analyze {max(session_types.keys(), key=lambda x: session_types[x]['avg_expansion_events'])} sessions for expansion phase dynamics
3. Consider optimizing processing for sessions with >1000 nodes
4. Investigate sessions with unusually high constant feature filtering

END OF REPORT
=============
"""

        # Save report
        with open('reports/comprehensive_analysis_report.txt', 'w') as f:
            f.write(report)

        logger.info("📝 Comprehensive report saved to reports/comprehensive_analysis_report.txt")

if __name__ == "__main__":
    analyzer = SessionAnalyzer()
    summary = analyzer.analyze_all_sessions()

    # Generate visualizations and reports
    analyzer.generate_visualizations()
    analyzer.generate_reports()

    # Save results
    with open('results/session_analysis_results.json', 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': analyzer.results
        }, f, indent=2, default=str)

    logger.info("✅ Complete analysis finished!")
    logger.info("📊 Check visualizations/ for graphs")
    logger.info("📝 Check reports/ for detailed analysis")
    logger.info("📋 Check results/ for raw data and CSV summary")
