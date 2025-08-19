#!/usr/bin/env python3
"""
IRONFORGE Quick Pattern Discovery
=================================

Fast pattern discovery script for immediate insights.
Analyzes top sessions and shows key patterns quickly.

Usage:
    python3 quick_pattern_discovery.py [--limit N] [--session-type TYPE]

Examples:
    python3 quick_pattern_discovery.py --limit 10
    python3 quick_pattern_discovery.py --session-type NY_PM
    python3 quick_pattern_discovery.py --session-type LONDON --limit 5
"""

import argparse
import json
import logging
import os

from learning.enhanced_graph_builder import EnhancedGraphBuilder

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for quick analysis

class QuickPatternDiscovery:
    """Fast pattern discovery for immediate insights"""
    
    def __init__(self):
        self.builder = EnhancedGraphBuilder()
        
    def discover_patterns(self, limit: int = None, session_type: str = None) -> None:
        """Quick pattern discovery with immediate results"""
        
        print("🔍 IRONFORGE Quick Pattern Discovery")
        print("=" * 50)
        
        # Get session files
        sessions_dir = "enhanced_sessions_with_relativity"
        session_files = [f for f in os.listdir(sessions_dir) if f.endswith('.json')]
        
        # Filter by session type if specified
        if session_type:
            session_files = [f for f in session_files if session_type.upper() in f.upper()]
            print(f"🎯 Filtering for {session_type.upper()} sessions")
        
        # Limit number of sessions
        if limit:
            session_files = session_files[:limit]
            print(f"📊 Analyzing {len(session_files)} sessions (limited to {limit})")
        else:
            print(f"📊 Analyzing {len(session_files)} sessions")
        
        print()
        
        # Quick analysis
        results = []
        total_fvg = 0
        total_expansion = 0
        total_consolidation = 0
        total_nodes = 0
        
        for i, session_file in enumerate(session_files, 1):
            print(f"Processing {i}/{len(session_files)}: {session_file.replace('enhanced_rel_', '').replace('.json', '')}")
            
            try:
                # Load and analyze
                with open(os.path.join(sessions_dir, session_file)) as f:
                    session_data = json.load(f)
                
                graph, metadata = self.builder.build_rich_graph(session_data)
                
                # Extract key patterns (enhanced with complete market cycle detection)
                nodes = graph['rich_node_features']
                fvg_count = sum(1 for node in nodes if node.fvg_redelivery_flag > 0)
                expansion_count = sum(1 for node in nodes if node.expansion_phase_flag > 0)
                consolidation_count = sum(1 for node in nodes if node.consolidation_flag > 0)
                retracement_count = sum(1 for node in nodes if node.retracement_flag > 0)
                reversal_count = sum(1 for node in nodes if node.reversal_flag > 0)

                total_fvg += fvg_count
                total_expansion += expansion_count
                total_consolidation += consolidation_count
                total_nodes += len(nodes)
                
                # Store result
                results.append({
                    'file': session_file,
                    'session_name': metadata.get('session_name', 'unknown'),
                    'session_date': metadata.get('session_date', 'unknown'),
                    'session_quality': metadata.get('session_quality', 'unknown'),
                    'nodes': len(nodes),
                    'fvg_events': fvg_count,
                    'expansion_events': expansion_count,
                    'consolidation_events': consolidation_count,
                    'retracement_events': retracement_count,
                    'reversal_events': reversal_count,
                    'fvg_pct': (fvg_count / len(nodes)) * 100 if nodes else 0,
                    'expansion_pct': (expansion_count / len(nodes)) * 100 if nodes else 0,
                    'retracement_pct': (retracement_count / len(nodes)) * 100 if nodes else 0,
                    'reversal_pct': (reversal_count / len(nodes)) * 100 if nodes else 0
                })

                # Show immediate feedback
                print(f"  ✅ {len(nodes)} nodes | FVG: {fvg_count} | Expansion: {expansion_count} | Retracement: {retracement_count} | Reversal: {reversal_count} | Quality: {metadata.get('session_quality', 'unknown')}")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                continue
        
        print()
        print("🎯 PATTERN DISCOVERY RESULTS")
        print("=" * 50)
        
        # Overall statistics
        print("📊 OVERALL STATISTICS:")
        print(f"   Total Sessions: {len(results)}")
        print(f"   Total Nodes: {total_nodes:,}")
        print(f"   Total FVG Events: {total_fvg:,}")
        print(f"   Total Expansion Events: {total_expansion:,}")
        print(f"   Total Consolidation Events: {total_consolidation:,}")
        print(f"   Average FVG per Session: {total_fvg / len(results):.1f}")
        print(f"   Average Expansion per Session: {total_expansion / len(results):.1f}")
        print(f"   Semantic Discovery Rate: {((total_fvg + total_expansion) / total_nodes * 100):.2f}%")
        print()
        
        # Top sessions by FVG activity
        print("🔥 TOP FVG REDELIVERY SESSIONS:")
        top_fvg = sorted(results, key=lambda x: x['fvg_events'], reverse=True)[:5]
        for i, session in enumerate(top_fvg, 1):
            print(f"   {i}. {session['session_name']} ({session['session_date']}) - {session['fvg_events']} events ({session['fvg_pct']:.1f}%)")
        print()
        
        # Top sessions by expansion activity
        print("📈 TOP EXPANSION PHASE SESSIONS:")
        top_expansion = sorted(results, key=lambda x: x['expansion_events'], reverse=True)[:5]
        for i, session in enumerate(top_expansion, 1):
            print(f"   {i}. {session['session_name']} ({session['session_date']}) - {session['expansion_events']} events ({session['expansion_pct']:.1f}%)")
        print()
        
        # Session type analysis
        session_types = {}
        for result in results:
            session_name = result['session_name']
            if session_name not in session_types:
                session_types[session_name] = {
                    'count': 0,
                    'total_fvg': 0,
                    'total_expansion': 0,
                    'total_nodes': 0
                }
            session_types[session_name]['count'] += 1
            session_types[session_name]['total_fvg'] += result['fvg_events']
            session_types[session_name]['total_expansion'] += result['expansion_events']
            session_types[session_name]['total_nodes'] += result['nodes']
        
        print("🎪 SESSION TYPE ANALYSIS:")
        for session_type, stats in session_types.items():
            avg_fvg = stats['total_fvg'] / stats['count']
            avg_expansion = stats['total_expansion'] / stats['count']
            avg_nodes = stats['total_nodes'] / stats['count']
            print(f"   {session_type}: {stats['count']} sessions | Avg FVG: {avg_fvg:.1f} | Avg Expansion: {avg_expansion:.1f} | Avg Nodes: {avg_nodes:.0f}")
        print()
        
        # Quality analysis
        quality_stats = {}
        for result in results:
            quality = result['session_quality']
            if quality not in quality_stats:
                quality_stats[quality] = {'count': 0, 'total_fvg': 0, 'total_expansion': 0}
            quality_stats[quality]['count'] += 1
            quality_stats[quality]['total_fvg'] += result['fvg_events']
            quality_stats[quality]['total_expansion'] += result['expansion_events']
        
        print("⭐ SESSION QUALITY ANALYSIS:")
        for quality, stats in quality_stats.items():
            avg_fvg = stats['total_fvg'] / stats['count']
            avg_expansion = stats['total_expansion'] / stats['count']
            print(f"   {quality.title()}: {stats['count']} sessions | Avg FVG: {avg_fvg:.1f} | Avg Expansion: {avg_expansion:.1f}")
        print()
        
        # Recommendations
        best_session_type = max(session_types.keys(), key=lambda x: session_types[x]['total_fvg'] + session_types[x]['total_expansion'])
        most_active_session = max(results, key=lambda x: x['fvg_events'] + x['expansion_events'])
        
        print("💡 QUICK INSIGHTS & RECOMMENDATIONS:")
        print(f"   🏆 Most Active Session Type: {best_session_type}")
        print(f"   🎯 Most Active Individual Session: {most_active_session['session_name']} on {most_active_session['session_date']}")
        print(f"   📊 Best FVG Discovery Rate: {max(results, key=lambda x: x['fvg_pct'])['session_name']} ({max(results, key=lambda x: x['fvg_pct'])['fvg_pct']:.1f}%)")
        print(f"   📈 Best Expansion Discovery Rate: {max(results, key=lambda x: x['expansion_pct'])['session_name']} ({max(results, key=lambda x: x['expansion_pct'])['expansion_pct']:.1f}%)")
        print()
        
        print("✅ Quick pattern discovery complete!")
        print("💡 For detailed analysis with graphs, run: python3 run_full_session_analysis.py")

def main():
    parser = argparse.ArgumentParser(description='IRONFORGE Quick Pattern Discovery')
    parser.add_argument('--limit', type=int, help='Limit number of sessions to analyze')
    parser.add_argument('--session-type', type=str, help='Filter by session type (NY_PM, LONDON, ASIA, etc.)')
    
    args = parser.parse_args()
    
    discoverer = QuickPatternDiscovery()
    discoverer.discover_patterns(limit=args.limit, session_type=args.session_type)

if __name__ == "__main__":
    main()
