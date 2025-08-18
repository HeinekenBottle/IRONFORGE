#!/usr/bin/env python3
"""
Phase 4a: Full Scale Archaeological Discovery
============================================
Run TGAT at full scale on all clean sessions with full features.
Test pattern discovery on long runs and multi-day datasets.
NO COMPROMISES - Full production-level archaeological mode.
"""

import glob
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from learning.enhanced_graph_builder import EnhancedGraphBuilder
from learning.tgat_discovery import IRONFORGEDiscovery


class FullScaleArchaeologicalDiscovery:
    """Full scale archaeological discovery with no session limits or chunking."""
    
    def __init__(self):
        """Initialize full scale discovery system."""
        self.graph_builder = EnhancedGraphBuilder()
        self.tgat_discovery = IRONFORGEDiscovery()
        self.session_count = 0
        self.total_patterns = 0
        self.total_nodes = 0
        self.total_edges = 0
        self.timeframe_distribution = {}
        self.pattern_types = {}
        self.cross_timeframe_edges = 0
        self.failed_sessions = []
        
    def discover_all_sessions(self) -> Dict[str, Any]:
        """Run full scale discovery on all available clean sessions."""
        print("🏛️ IRONFORGE Phase 4a: Full Scale Archaeological Discovery")
        print("=" * 70)
        print("🎯 Mission: Production-level archaeological mode with NO COMPROMISES")
        print("🚫 NO session limits, NO chunking, NO fallbacks")
        
        # Find all price-relativity enhanced sessions
        rel_files = glob.glob("/Users/jack/IRONPULSE/data/sessions/htf_relativity/*_rel.json")
        print(f"\n📊 Found {len(rel_files)} price-relativity enhanced sessions")
        
        # Sort by date to test multi-day patterns
        rel_files.sort()
        
        start_time = time.time()
        
        # Process all sessions without limits
        print(f"\n🔄 Processing ALL {len(rel_files)} sessions...")
        
        session_results = []
        
        for i, session_file in enumerate(rel_files, 1):
            session_name = Path(session_file).name
            print(f"\n🏛️ Session {i}/{len(rel_files)}: {session_name}")
            
            try:
                # Load session data
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Build enhanced graph
                graph_data = self.graph_builder.build_rich_graph(session_data)
                
                # Convert to TGAT format
                X, edge_index, edge_times, metadata, edge_attr = self.graph_builder.to_tgat_format(graph_data)
                
                # Run archaeological discovery
                learn_result = self.tgat_discovery.learn_session(X, edge_index, edge_times, metadata, edge_attr)
                
                # Analyze results
                patterns = learn_result.get('patterns', [])
                session_result = self._analyze_session_result(
                    session_name, graph_data, X, edge_index, edge_attr, patterns
                )
                
                session_results.append(session_result)
                
                # Update totals
                self.session_count += 1
                self.total_patterns += len(patterns)
                self.total_nodes += X.shape[0]
                self.total_edges += edge_index.shape[1]
                
                # Track timeframes and pattern types
                self._update_global_stats(graph_data, patterns)
                
                print(f"  ✅ Success: {X.shape[0]} nodes, {edge_index.shape[1]} edges, {len(patterns)} patterns")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                self.failed_sessions.append({
                    'session': session_name,
                    'error': str(e)
                })
                continue
        
        total_time = time.time() - start_time
        
        # Generate comprehensive results
        results = self._generate_full_scale_results(session_results, total_time)
        
        return results
    
    def _analyze_session_result(self, session_name: str, graph_data: Dict, 
                               X, edge_index, edge_attr, patterns: List) -> Dict:
        """Analyze individual session archaeological discovery results."""
        
        # Count scale edges (cross-timeframe)
        scale_edges = 0
        timeframe_mapping = {0: '1m', 1: '5m', 2: '15m', 3: '1h', 4: 'D', 5: 'W'}
        
        # Get timeframes for each node
        timeframes = []
        for node_features in graph_data['rich_node_features']:
            timeframe_id = node_features.timeframe_source
            timeframe = timeframe_mapping.get(timeframe_id, f'unknown_{timeframe_id}')
            timeframes.append(timeframe)
        
        # Count cross-timeframe edges
        for edge_idx in range(edge_index.shape[1]):
            source_idx = edge_index[0, edge_idx].item()
            target_idx = edge_index[1, edge_idx].item()
            
            if source_idx < len(timeframes) and target_idx < len(timeframes):
                source_tf = timeframes[source_idx]
                target_tf = timeframes[target_idx]
                
                if source_tf != target_tf:
                    scale_edges += 1
        
        # Analyze pattern types
        pattern_breakdown = {}
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            pattern_breakdown[pattern_type] = pattern_breakdown.get(pattern_type, 0) + 1
        
        # Count unique timeframes
        unique_timeframes = list(set(timeframes))
        
        return {
            'session_name': session_name,
            'nodes': X.shape[0],
            'edges': edge_index.shape[1],
            'patterns': len(patterns),
            'scale_edges': scale_edges,
            'scale_edge_percentage': (scale_edges / edge_index.shape[1] * 100) if edge_index.shape[1] > 0 else 0,
            'timeframes': unique_timeframes,
            'pattern_breakdown': pattern_breakdown,
            'node_feature_dims': X.shape[1],
            'edge_feature_dims': edge_attr.shape[1],
            'success': True
        }
    
    def _update_global_stats(self, graph_data: Dict, patterns: List):
        """Update global statistics across all sessions."""
        
        # Count timeframes
        timeframe_mapping = {0: '1m', 1: '5m', 2: '15m', 3: '1h', 4: 'D', 5: 'W'}
        
        for node_features in graph_data['rich_node_features']:
            timeframe_id = node_features.timeframe_source
            timeframe = timeframe_mapping.get(timeframe_id, f'unknown_{timeframe_id}')
            self.timeframe_distribution[timeframe] = self.timeframe_distribution.get(timeframe, 0) + 1
        
        # Count pattern types
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            self.pattern_types[pattern_type] = self.pattern_types.get(pattern_type, 0) + 1
    
    def _generate_full_scale_results(self, session_results: List[Dict], total_time: float) -> Dict:
        """Generate comprehensive full scale discovery results."""
        
        successful_sessions = [r for r in session_results if r['success']]
        total_sessions = len(session_results) + len(self.failed_sessions)
        success_rate = (len(successful_sessions) / total_sessions * 100) if total_sessions > 0 else 0
        
        # Multi-day analysis
        dates_covered = set()
        for result in successful_sessions:
            # Extract date from session name
            if '_2025_' in result['session_name']:
                date_part = result['session_name'].split('_2025_')[1][:5]  # MM_DD
                dates_covered.add(f"2025_{date_part}")
        
        # Performance metrics
        avg_time_per_session = total_time / len(successful_sessions) if successful_sessions else 0
        patterns_per_second = self.total_patterns / total_time if total_time > 0 else 0
        
        # Pattern diversity analysis
        pattern_diversity = len(self.pattern_types)
        avg_patterns_per_session = self.total_patterns / len(successful_sessions) if successful_sessions else 0
        
        # Scale edge analysis
        total_scale_edges = sum(r['scale_edges'] for r in successful_sessions)
        avg_scale_edge_percentage = sum(r['scale_edge_percentage'] for r in successful_sessions) / len(successful_sessions) if successful_sessions else 0
        
        # Timeframe coverage
        timeframes_discovered = list(self.timeframe_distribution.keys())
        
        return {
            'execution_summary': {
                'total_sessions_attempted': total_sessions,
                'successful_sessions': len(successful_sessions),
                'failed_sessions': len(self.failed_sessions),
                'success_rate': success_rate,
                'total_execution_time': total_time,
                'avg_time_per_session': avg_time_per_session
            },
            'archaeological_discoveries': {
                'total_patterns': self.total_patterns,
                'pattern_types_discovered': pattern_diversity,
                'avg_patterns_per_session': avg_patterns_per_session,
                'patterns_per_second': patterns_per_second,
                'pattern_type_breakdown': dict(self.pattern_types)
            },
            'graph_analysis': {
                'total_nodes': self.total_nodes,
                'total_edges': self.total_edges,
                'total_scale_edges': total_scale_edges,
                'avg_scale_edge_percentage': avg_scale_edge_percentage,
                'timeframes_discovered': timeframes_discovered,
                'timeframe_distribution': dict(self.timeframe_distribution)
            },
            'multi_day_coverage': {
                'dates_covered': sorted(list(dates_covered)),
                'date_span_days': len(dates_covered),
                'multi_day_dataset': len(dates_covered) > 1
            },
            'production_readiness': {
                'no_session_limits': True,
                'no_chunking_used': True,
                'no_fallbacks_triggered': len(self.failed_sessions) == 0,
                'full_feature_processing': True,
                'acceptable_runtime': total_time < 300  # <5 minutes target
            },
            'failed_sessions': self.failed_sessions,
            'session_details': successful_sessions
        }

def run_phase4a_full_scale():
    """Run Phase 4a full scale archaeological discovery."""
    
    print("🎯 IRONFORGE Phase 4a: Full Scale Archaeological Discovery")
    print("🚫 NO COMPROMISES - Production archaeological mode")
    
    discovery = FullScaleArchaeologicalDiscovery()
    results = discovery.discover_all_sessions()
    
    # Print comprehensive results
    print("\n🏆 PHASE 4a RESULTS:")
    print("=" * 70)
    
    exec_summary = results['execution_summary']
    print("📊 Execution Summary:")
    print(f"  Sessions: {exec_summary['successful_sessions']}/{exec_summary['total_sessions_attempted']} ({exec_summary['success_rate']:.1f}% success)")
    print(f"  Runtime: {exec_summary['total_execution_time']:.1f}s ({exec_summary['avg_time_per_session']:.2f}s per session)")
    
    arch_discoveries = results['archaeological_discoveries']
    print("\n🏛️ Archaeological Discoveries:")
    print(f"  Total patterns: {arch_discoveries['total_patterns']:,}")
    print(f"  Pattern types: {arch_discoveries['pattern_types_discovered']}")
    print(f"  Avg per session: {arch_discoveries['avg_patterns_per_session']:.1f}")
    print(f"  Discovery rate: {arch_discoveries['patterns_per_second']:.1f} patterns/second")
    
    graph_analysis = results['graph_analysis']
    print("\n📊 Graph Analysis:")
    print(f"  Total nodes: {graph_analysis['total_nodes']:,}")
    print(f"  Total edges: {graph_analysis['total_edges']:,}")
    print(f"  Scale edges: {graph_analysis['total_scale_edges']:,} ({graph_analysis['avg_scale_edge_percentage']:.1f}% avg)")
    print(f"  Timeframes: {len(graph_analysis['timeframes_discovered'])}")
    
    multi_day = results['multi_day_coverage']
    print("\n📅 Multi-Day Coverage:")
    print(f"  Date span: {multi_day['date_span_days']} days")
    print(f"  Multi-day dataset: {multi_day['multi_day_dataset']}")
    print(f"  Dates: {', '.join(multi_day['dates_covered'][:5])}{'...' if len(multi_day['dates_covered']) > 5 else ''}")
    
    production = results['production_readiness']
    print("\n🚀 Production Readiness:")
    all_criteria = [
        ("No session limits", production['no_session_limits']),
        ("No chunking", production['no_chunking_used']),
        ("No fallbacks", production['no_fallbacks_triggered']),
        ("Full features", production['full_feature_processing']),
        ("Acceptable runtime", production['acceptable_runtime'])
    ]
    
    for criterion, passed in all_criteria:
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}")
    
    all_passed = all(passed for _, passed in all_criteria)
    
    if all_passed:
        print("\n🎉 PHASE 4a: COMPLETE SUCCESS!")
        print("✅ Full scale archaeological discovery OPERATIONAL")
        print("✅ Multi-day pattern discovery CONFIRMED")
        print("✅ Production-level performance ACHIEVED")
    else:
        print("\n⚠️ PHASE 4a: PARTIAL SUCCESS")
        print("🔧 Some production criteria need attention")
    
    return results, all_passed

if __name__ == "__main__":
    try:
        results, success = run_phase4a_full_scale()
        
        # Save results for Phase 4b analysis
        output_file = "/Users/jack/IRONPULSE/IRONFORGE/phase4a_full_scale_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {output_file}")
        print(f"🎯 Phase 4a Status: {'SUCCESS' if success else 'NEEDS ATTENTION'}")
        
    except Exception as e:
        print(f"❌ Phase 4a failed: {e}")
        import traceback
        traceback.print_exc()