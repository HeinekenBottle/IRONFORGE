#!/usr/bin/env python3
"""
Process ALL Sessions with Simple Event-Time Clustering + Cross-TF Mapping
=========================================================================
Complete analysis of the entire dataset with temporal intelligence
"""

from orchestrator import IRONFORGE
import glob
import time
from pathlib import Path

def process_all_sessions():
    """Process all available sessions with time pattern analysis"""
    
    print("🚀 COMPLETE SESSION PROCESSING")
    print("=" * 80)
    print("Processing ALL sessions with Simple Event-Time Clustering + Cross-TF Mapping")
    print("=" * 80)
    
    # Initialize IRONFORGE
    start_time = time.time()
    print("🔧 Initializing IRONFORGE...")
    forge = IRONFORGE(use_enhanced=True)
    init_time = time.time() - start_time
    print(f"✅ IRONFORGE initialized in {init_time:.2f}s")
    
    # Find all session files
    session_files = glob.glob('enhanced_sessions_with_relativity/*.json')
    print(f"\n📁 Session Discovery:")
    print(f"   Total sessions found: {len(session_files)}")
    
    if not session_files:
        print("❌ No session files found in enhanced_sessions_with_relativity/")
        return
    
    # Session type breakdown
    session_types = {}
    for session in session_files:
        session_name = Path(session).stem
        if 'NY_PM' in session_name:
            session_types['NY_PM'] = session_types.get('NY_PM', 0) + 1
        elif 'NY_AM' in session_name:
            session_types['NY_AM'] = session_types.get('NY_AM', 0) + 1
        elif 'LONDON' in session_name:
            session_types['LONDON'] = session_types.get('LONDON', 0) + 1
        elif 'ASIA' in session_name:
            session_types['ASIA'] = session_types.get('ASIA', 0) + 1
        elif 'PREMARKET' in session_name:
            session_types['PREMARKET'] = session_types.get('PREMARKET', 0) + 1
        else:
            session_types['OTHER'] = session_types.get('OTHER', 0) + 1
    
    print(f"\n📊 Session Type Distribution:")
    for session_type, count in sorted(session_types.items()):
        print(f"   {session_type}: {count} sessions")
    
    # Process all sessions
    print(f"\n🔄 Processing {len(session_files)} sessions...")
    processing_start = time.time()
    
    results = forge.process_sessions(session_files)
    
    processing_time = time.time() - processing_start
    print(f"\n✅ Processing complete in {processing_time:.2f}s")
    print(f"📊 Performance: {processing_time/len(session_files):.2f}s per session")
    
    # Results summary
    patterns_discovered = results.get('patterns_discovered', [])
    graphs_preserved = results.get('graphs_preserved', 0)
    
    print(f"\n📈 COMPLETE RESULTS:")
    print(f"   Sessions processed: {len(session_files)}")
    print(f"   Patterns discovered: {len(patterns_discovered)}")
    print(f"   Graphs preserved: {graphs_preserved}")
    print(f"   Total processing time: {processing_time:.2f}s")
    
    # Analyze time patterns from all preserved graphs
    analyze_all_time_patterns()
    
    print(f"\n🎯 MISSION ACCOMPLISHED!")
    print(f"✅ All {len(session_files)} sessions processed with temporal intelligence")
    print(f"💡 Rich time clustering and cross-TF mapping now available for analysis")

def analyze_all_time_patterns():
    """Analyze time patterns from all preserved graphs"""
    
    print(f"\n🕐 COMPLETE TIME PATTERN ANALYSIS")
    print("-" * 60)
    
    # Find all preserved graphs
    graph_files = glob.glob("IRONFORGE/preservation/full_graph_store/*2025_08*.pkl")
    graph_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    print(f"📁 Analyzing {len(graph_files)} preserved graphs...")
    
    total_events = 0
    total_clusters = 0
    total_cross_tf_links = 0
    session_summaries = []
    
    for i, graph_file in enumerate(graph_files):
        session_name = Path(graph_file).stem.replace('_graph_', '_').split('_202')[0]
        
        try:
            import pickle
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Extract time pattern data
            session_metadata = graph_data.get('session_metadata', {})
            time_patterns = session_metadata.get('time_patterns', {})
            
            if time_patterns:
                # Extract metrics
                event_clusters = time_patterns.get('event_clusters', {})
                cross_tf_mapping = time_patterns.get('cross_tf_mapping', {})
                clustering_stats = time_patterns.get('clustering_stats', {})
                
                events = clustering_stats.get('total_events_analyzed', 0)
                clusters = len(event_clusters) if isinstance(event_clusters, dict) else 0
                cross_tf_links = len(cross_tf_mapping.get('cross_tf_links', [])) if isinstance(cross_tf_mapping, dict) else 0
                
                total_events += events
                total_clusters += clusters
                total_cross_tf_links += cross_tf_links
                
                session_summaries.append({
                    'session': session_name,
                    'events': events,
                    'clusters': clusters,
                    'cross_tf_links': cross_tf_links
                })
                
                if i < 10:  # Show first 10 sessions details
                    print(f"  ✅ {session_name}: {events} events, {clusters} clusters, {cross_tf_links} cross-TF links")
            else:
                if i < 10:
                    print(f"  ⚠️ {session_name}: No time patterns found")
                    
        except Exception as e:
            if i < 10:
                print(f"  ❌ {session_name}: Error - {e}")
    
    # Summary statistics
    print(f"\n📊 AGGREGATE TIME PATTERN RESULTS:")
    print(f"   Sessions with time patterns: {len(session_summaries)}")
    print(f"   Total events analyzed: {total_events}")
    print(f"   Total event clusters: {total_clusters}")
    print(f"   Total cross-TF links: {total_cross_tf_links}")
    
    if session_summaries:
        avg_events = total_events / len(session_summaries)
        avg_clusters = total_clusters / len(session_summaries)
        avg_cross_tf = total_cross_tf_links / len(session_summaries)
        
        print(f"\n📈 AVERAGES PER SESSION:")
        print(f"   Events: {avg_events:.1f}")
        print(f"   Clusters: {avg_clusters:.1f}")
        print(f"   Cross-TF links: {avg_cross_tf:.1f}")
        
        # Show top sessions
        top_sessions = sorted(session_summaries, key=lambda x: x['events'], reverse=True)[:5]
        print(f"\n🏆 TOP 5 SESSIONS BY EVENT COUNT:")
        for session in top_sessions:
            print(f"   {session['session']}: {session['events']} events")

if __name__ == "__main__":
    process_all_sessions()