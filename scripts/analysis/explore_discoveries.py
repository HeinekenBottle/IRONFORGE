#!/usr/bin/env python3
"""
Explore IRONFORGE Discovery Results with Time Pattern Intelligence
================================================================
See what the Simple Event-Time Clustering + Cross-TF Mapping reveals
"""

import glob
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def load_time_patterns_from_graphs():
    """Load time patterns from preserved graphs"""
    
    print("🔍 Loading time patterns from preserved graphs...")
    
    # Find preserved graphs
    graph_files = glob.glob("/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/graphs/*.pkl")
    print(f"📁 Found {len(graph_files)} preserved graphs")
    
    all_time_patterns = []
    session_summaries = []
    
    for graph_file in graph_files:
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            session_meta = graph_data.get('session_metadata', {})
            time_patterns = session_meta.get('time_patterns', {})
            
            if time_patterns:
                session_name = Path(graph_file).stem.replace('_graph_', '_')
                
                # Extract key metrics
                clusters = time_patterns.get('event_clusters', {})
                cross_tf = time_patterns.get('cross_tf_mapping', {})
                stats = time_patterns.get('clustering_stats', {})
                
                summary = {
                    'session': session_name,
                    'event_clusters': len(clusters) if isinstance(clusters, dict) else len(clusters),
                    'cross_tf_links': len(cross_tf.get('cross_tf_links', [])) if isinstance(cross_tf, dict) else 0,
                    'total_events': stats.get('total_events', 0),
                    'time_patterns': time_patterns
                }
                
                all_time_patterns.append(time_patterns)
                session_summaries.append(summary)
                
                print(f"  ✅ {session_name}: {summary['event_clusters']} clusters, {summary['cross_tf_links']} cross-TF links")
            else:
                print(f"  ⚠️ {Path(graph_file).stem}: No time patterns found")
                
        except Exception as e:
            print(f"  ❌ {Path(graph_file).stem}: Error loading - {e}")
    
    return all_time_patterns, session_summaries

def analyze_discovered_patterns():
    """Load and analyze the 500 discovered patterns"""
    
    print("\n🧠 Analyzing discovered patterns...")
    
    patterns_file = "/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/discovered_patterns.json"
    
    try:
        with open(patterns_file, 'r') as f:
            patterns = json.load(f)
        
        print(f"📊 Loaded {len(patterns)} discovered patterns")
        
        # Analyze pattern types
        pattern_types = Counter()
        session_distribution = Counter()
        temporal_features = []
        
        for pattern in patterns:
            # Pattern type analysis
            pattern_type = pattern.get('type', 'unknown')
            pattern_types[pattern_type] += 1
            
            # Session distribution
            session = pattern.get('session', 'unknown')
            session_distribution[session] += 1
            
            # Temporal features
            if 'temporal_features' in pattern:
                temporal_features.append(pattern['temporal_features'])
        
        # Display analysis
        print("\n📈 Pattern Type Distribution:")
        for ptype, count in pattern_types.most_common():
            percentage = (count / len(patterns)) * 100
            print(f"  {ptype}: {count} patterns ({percentage:.1f}%)")
        
        print("\n🗓️ Session Distribution:")
        for session, count in session_distribution.most_common():
            print(f"  {session}: {count} patterns")
        
        return patterns, pattern_types, session_distribution
        
    except Exception as e:
        print(f"❌ Error loading patterns: {e}")
        return [], {}, {}

def explore_time_clustering_insights(time_patterns, session_summaries):
    """Analyze the time clustering discoveries"""
    
    print("\n🕐 TIME CLUSTERING INSIGHTS")
    print("=" * 50)
    
    if not time_patterns:
        print("❌ No time patterns found")
        return
    
    # Aggregate insights across all sessions
    all_clusters = defaultdict(list)
    all_cross_tf_links = []
    
    for patterns in time_patterns:
        clusters = patterns.get('event_clusters', {})
        cross_tf = patterns.get('cross_tf_mapping', {})
        
        # Handle different data structures
        if isinstance(clusters, dict):
            for event_type, bins in clusters.items():
                all_clusters[event_type].extend(bins)
        
        # Cross-TF links
        if isinstance(cross_tf, dict):
            links = cross_tf.get('cross_tf_links', [])
            all_cross_tf_links.extend(links)
    
    # Event clustering analysis
    print(f"📊 Event Types Found: {list(all_clusters.keys())}")
    
    for event_type, all_bins in all_clusters.items():
        if all_bins:
            # Aggregate time bins
            time_distribution = Counter()
            total_events = 0
            
            for bin_data in all_bins:
                if isinstance(bin_data, list) and len(bin_data) == 2:
                    time_bin, count = bin_data
                    time_distribution[time_bin] += count
                    total_events += count
            
            print(f"\n🔥 {event_type} Analysis:")
            print(f"   Total events: {total_events}")
            print(f"   Active time bins: {len(time_distribution)}")
            
            # Show peak activity times
            if time_distribution:
                peak_times = time_distribution.most_common(3)
                print("   Peak activity:")
                for time_bin, count in peak_times:
                    print(f"     {time_bin}: {count} events")
    
    # Cross-timeframe analysis
    if all_cross_tf_links:
        print("\n📈 CROSS-TIMEFRAME INTELLIGENCE:")
        print(f"   Total cross-TF links: {len(all_cross_tf_links)}")
        
        # Analyze HTF contexts
        htf_contexts = Counter()
        ltf_events = Counter()
        
        for link in all_cross_tf_links:
            if isinstance(link, dict):
                htf_tf = link.get('htf_timeframe', 'unknown')
                ltf_event_types = link.get('ltf_event_types', [])
                
                htf_contexts[htf_tf] += 1
                for event in ltf_event_types:
                    ltf_events[event] += 1
        
        print("   HTF contexts:")
        for context, count in htf_contexts.most_common():
            print(f"     {context}: {count} links")
        
        print("   LTF events linked:")
        for event, count in ltf_events.most_common():
            print(f"     {event}: {count} links")

def show_session_insights(session_summaries):
    """Show insights by session"""
    
    print("\n📊 SESSION-BY-SESSION INSIGHTS:")
    print("=" * 50)
    
    if not session_summaries:
        print("❌ No session summaries available")
        return
    
    for summary in session_summaries:
        session = summary['session']
        clusters = summary['event_clusters']
        cross_tf = summary['cross_tf_links']
        events = summary['total_events']
        
        print(f"\n📈 {session}:")
        print(f"   Event clusters: {clusters}")
        print(f"   Cross-TF links: {cross_tf}")
        print(f"   Total events: {events}")
        
        # Show activity level
        if events > 20:
            activity = "🔥 High"
        elif events > 10:
            activity = "📊 Medium"
        elif events > 0:
            activity = "📉 Low"
        else:
            activity = "💤 Minimal"
        
        print(f"   Activity level: {activity}")

def main():
    """Main exploration function"""
    
    print("🚀 IRONFORGE DISCOVERY EXPLORATION")
    print("=" * 60)
    print("Analyzing Simple Event-Time Clustering + Cross-TF Mapping results")
    print("=" * 60)
    
    # Load time patterns from graphs
    time_patterns, session_summaries = load_time_patterns_from_graphs()
    
    # Analyze discovered patterns
    patterns, pattern_types, session_dist = analyze_discovered_patterns()
    
    # Deep dive into time clustering
    explore_time_clustering_insights(time_patterns, session_summaries)
    
    # Session-by-session breakdown
    show_session_insights(session_summaries)
    
    # Summary
    print("\n🎯 DISCOVERY SUMMARY:")
    print("=" * 30)
    print(f"📊 Sessions analyzed: {len(session_summaries)}")
    print(f"🧠 Patterns discovered: {len(patterns)}")
    print(f"🕐 Sessions with time patterns: {len([s for s in session_summaries if s['total_events'] > 0])}")
    print(f"📈 Pattern types: {len(pattern_types)}")
    
    if pattern_types:
        top_pattern = max(pattern_types, key=pattern_types.get)
        print(f"🔥 Dominant pattern type: {top_pattern} ({pattern_types[top_pattern]} patterns)")
    
    print("\n💡 Your enhanced IRONFORGE now provides rich temporal intelligence!")
    print("🎯 Use this to understand 'when events cluster' + 'what HTF context'")

if __name__ == "__main__":
    main()