#!/usr/bin/env python3
"""
Quick test of Simple Event-Time Clustering + Cross-TF Mapping
"""

from orchestrator import IRONFORGE
import glob
import pickle

def test_time_patterns():
    """Test the new time pattern capabilities"""
    
    print("🚀 Testing Simple Event-Time Clustering + Cross-TF Mapping")
    print("=" * 60)
    
    # Initialize IRONFORGE
    forge = IRONFORGE(use_enhanced=True)
    
    # Get ALL session files (remove [:5] limit)
    sessions = glob.glob('enhanced_sessions_with_relativity/*.json')
    print(f"📁 Found {len(sessions)} session files (processing ALL)")
    
    if not sessions:
        print("❌ No session files found in enhanced_sessions_with_relativity/")
        return
    
    # Process sessions
    print(f"🔄 Processing {len(sessions)} sessions...")
    results = forge.process_sessions(sessions)
    
    # Results summary
    print("\n📊 Results Summary:")
    print(f"  Sessions processed: {results['sessions_processed']}/{len(sessions)}")
    print(f"  Patterns discovered: {len(results['patterns_discovered'])}")
    print(f"  Graphs preserved: {len(results.get('graphs_preserved', []))}")
    
    # Check time patterns in preserved graphs
    print("\n🕐 Time Pattern Analysis:")
    print("-" * 40)
    
    for i, graph_file in enumerate(results.get('graphs_preserved', [])[:3]):
        print(f"\n📈 Graph {i+1}: {graph_file.split('/')[-1]}")
        
        try:
            with open(graph_file, 'rb') as f:
                graph = pickle.load(f)
            
            session_meta = graph.get('session_metadata', {})
            time_patterns = session_meta.get('time_patterns', {})
            
            if time_patterns:
                # Event clusters (returned as a list, not dict)
                clusters = time_patterns.get('event_clusters', [])
                cross_tf_mapping = time_patterns.get('cross_tf_mapping', {})
                cross_tf_links = cross_tf_mapping.get('ltf_to_15m', []) + cross_tf_mapping.get('ltf_to_1h', [])
                stats = time_patterns.get('clustering_stats', {})
                
                print("  ✅ Time patterns found!")
                print(f"     Event cluster types: {len(clusters)}")
                print(f"     Cross-TF links: {len(cross_tf_links)}")
                print(f"     Total events: {stats.get('total_events', 0)}")
                
                # Show event distribution (clusters is a list of cluster objects)
                if clusters:
                    for i, cluster in enumerate(clusters):
                        if hasattr(cluster, 'events') and cluster.events:
                            print(f"     Cluster {i+1}: {len(cluster.events)} events")
                        elif isinstance(cluster, dict) and 'events' in cluster:
                            print(f"     Cluster {i+1}: {len(cluster['events'])} events")
                
                # Show sample cross-TF link
                if cross_tf_links:
                    sample_link = cross_tf_links[0]
                    print(f"     Sample link: {sample_link['ltf_time']}min {sample_link['ltf_event_types']} → {sample_link['htf_timeframe']} {sample_link['htf_structures']}")
                    
            else:
                print("  ⚠️ No time patterns found")
                
        except Exception as e:
            print(f"  ❌ Error loading graph: {e}")
    
    print("\n🎉 Time pattern analysis complete!")
    print("💡 Your sessions now include 'when events cluster' + 'what HTF context' intelligence")

if __name__ == "__main__":
    test_time_patterns()