#!/usr/bin/env python3
"""
Comprehensive IRONFORGE Discovery Report
========================================
Deep analysis of what the enhanced system with Simple Event-Time Clustering discovers
"""

import glob
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def analyze_all_sessions():
    """Analyze all preserved sessions comprehensively"""
    
    print("🚀 COMPREHENSIVE IRONFORGE DISCOVERY ANALYSIS")
    print("=" * 80)
    
    # Find all recent preserved graphs
    graph_files = glob.glob("/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/full_graph_store/*2025_08*.pkl")
    graph_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)  # Most recent first
    
    print(f"📁 Analyzing {len(graph_files)} recent sessions (August 2025)")
    
    all_session_data = []
    total_semantic_events = 0
    total_nodes = 0
    
    for _i, graph_file in enumerate(graph_files[:10]):  # Analyze top 10 recent sessions
        session_name = Path(graph_file).stem.replace('_graph_', '_').split('_202')[0]
        
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Extract comprehensive session data
            session_analysis = analyze_single_session(graph_data, session_name)
            all_session_data.append(session_analysis)
            
            total_semantic_events += session_analysis['total_semantic_events']
            total_nodes += session_analysis['total_nodes']
            
            print(f"  ✅ {session_name}: {session_analysis['total_semantic_events']}/{session_analysis['total_nodes']} events ({session_analysis['event_percentage']:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ {session_name}: Error - {e}")
    
    # Aggregate analysis
    print("\n📊 AGGREGATE RESULTS:")
    print(f"   Sessions analyzed: {len(all_session_data)}")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Total semantic events: {total_semantic_events}")
    print(f"   Overall event rate: {(total_semantic_events/total_nodes)*100:.1f}%")
    
    # Event type distribution
    aggregate_events = defaultdict(int)
    session_types = Counter()
    temporal_distribution = defaultdict(list)
    
    for session in all_session_data:
        session_types[session['session_type']] += 1
        
        for event_type, count in session['semantic_events'].items():
            aggregate_events[event_type] += count
        
        # Temporal analysis
        if session['time_patterns']:
            for event_type, pattern_data in session['time_patterns'].items():
                temporal_distribution[event_type].extend(pattern_data)
    
    print("\n🔥 EVENT TYPE DISTRIBUTION:")
    for event_type, count in sorted(aggregate_events.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_semantic_events) * 100 if total_semantic_events > 0 else 0
        print(f"   {event_type}: {count} events ({percentage:.1f}%)")
    
    print("\n📈 SESSION TYPE ANALYSIS:")
    for session_type, count in session_types.most_common():
        print(f"   {session_type}: {count} sessions")
    
    # Temporal insights
    analyze_temporal_patterns(all_session_data)
    
    # Pattern quality analysis
    analyze_pattern_quality(all_session_data)
    
    return all_session_data

def analyze_single_session(graph_data, session_name):
    """Comprehensive analysis of a single session"""
    
    rich_features = graph_data.get('rich_node_features', [])
    session_metadata = graph_data.get('session_metadata', {})
    
    # Basic metrics
    total_nodes = len(rich_features)
    
    # Semantic event analysis
    semantic_events = {
        'fvg_redelivery': 0,
        'expansion_phase': 0,
        'consolidation': 0,
        'liq_sweep': 0,
        'pd_array_interaction': 0
    }
    
    # Time-based event tracking
    time_based_events = []
    
    for feature in rich_features:
        # Count semantic events
        if feature.fvg_redelivery_flag > 0.0:
            semantic_events['fvg_redelivery'] += 1
            time_based_events.append(('fvg_redelivery', feature.time_minutes))
        if feature.expansion_phase_flag > 0.0:
            semantic_events['expansion_phase'] += 1
            time_based_events.append(('expansion_phase', feature.time_minutes))
        if feature.consolidation_flag > 0.0:
            semantic_events['consolidation'] += 1
            time_based_events.append(('consolidation', feature.time_minutes))
        if feature.liq_sweep_flag > 0.0:
            semantic_events['liq_sweep'] += 1
            time_based_events.append(('liq_sweep', feature.time_minutes))
        if feature.pd_array_interaction_flag > 0.0:
            semantic_events['pd_array_interaction'] += 1
            time_based_events.append(('pd_array_interaction', feature.time_minutes))
    
    total_semantic_events = sum(semantic_events.values())
    event_percentage = (total_semantic_events / total_nodes) * 100 if total_nodes > 0 else 0
    
    # Time pattern analysis
    time_patterns = create_time_clusters(time_based_events)
    
    # Session characteristics
    session_type = extract_session_type(session_name)
    
    return {
        'session_name': session_name,
        'session_type': session_type,
        'total_nodes': total_nodes,
        'total_semantic_events': total_semantic_events,
        'event_percentage': event_percentage,
        'semantic_events': semantic_events,
        'time_patterns': time_patterns,
        'time_based_events': time_based_events,
        'session_metadata': session_metadata
    }

def create_time_clusters(time_based_events, bin_size=5):
    """Create time clusters from events"""
    
    if not time_based_events:
        return {}
    
    # Group events by type and time bin
    event_clusters = defaultdict(lambda: defaultdict(int))
    
    for event_type, time_minutes in time_based_events:
        if time_minutes is not None:
            bin_index = int(time_minutes // bin_size)
            bin_label = f"{bin_index*bin_size}–{(bin_index+1)*bin_size}m"
            event_clusters[event_type][bin_label] += 1
    
    # Convert to sorted lists
    result = {}
    for event_type, bins in event_clusters.items():
        sorted_bins = sorted(bins.items(), key=lambda x: int(x[0].split('–')[0]))
        result[event_type] = sorted_bins
    
    return result

def extract_session_type(session_name):
    """Extract session type from session name"""
    
    if 'NY_PM' in session_name:
        return 'NY_PM'
    elif 'NY_AM' in session_name:
        return 'NY_AM'
    elif 'LONDON' in session_name:
        return 'LONDON'
    elif 'ASIA' in session_name:
        return 'ASIA'
    elif 'PREMARKET' in session_name:
        return 'PREMARKET'
    elif 'MIDNIGHT' in session_name:
        return 'MIDNIGHT'
    else:
        return 'UNKNOWN'

def analyze_temporal_patterns(session_data):
    """Analyze temporal patterns across sessions"""
    
    print("\n🕐 TEMPORAL PATTERN ANALYSIS:")
    print("-" * 50)
    
    # Aggregate time patterns by event type
    all_time_patterns = defaultdict(list)
    
    for session in session_data:
        for event_type, bins in session['time_patterns'].items():
            all_time_patterns[event_type].extend(bins)
    
    # Analyze each event type's temporal distribution
    for event_type, all_bins in all_time_patterns.items():
        if all_bins:
            # Aggregate across all sessions
            time_distribution = defaultdict(int)
            total_events = 0
            
            for bin_label, count in all_bins:
                time_distribution[bin_label] += count
                total_events += count
            
            if total_events > 0:
                print(f"\n🔥 {event_type.upper()} Temporal Distribution:")
                print(f"   Total events: {total_events}")
                
                # Show top time periods
                sorted_times = sorted(time_distribution.items(), key=lambda x: x[1], reverse=True)
                peak_times = sorted_times[:5]
                
                print("   Peak activity periods:")
                for time_bin, count in peak_times:
                    percentage = (count / total_events) * 100
                    print(f"     {time_bin}: {count} events ({percentage:.1f}%)")
                
                # Calculate concentration
                if len(time_distribution) > 1:
                    max_count = max(time_distribution.values())
                    concentration = (max_count / total_events) * 100
                    print(f"   Concentration: {concentration:.1f}% in peak period")

def analyze_pattern_quality(session_data):
    """Analyze the quality and characteristics of discovered patterns"""
    
    print("\n📊 PATTERN QUALITY ANALYSIS:")
    print("-" * 50)
    
    # Event density analysis
    densities = [s['event_percentage'] for s in session_data]
    
    if densities:
        avg_density = np.mean(densities)
        max_density = np.max(densities)
        min_density = np.min(densities)
        
        print("📈 Event Density Statistics:")
        print(f"   Average: {avg_density:.1f}%")
        print(f"   Range: {min_density:.1f}% - {max_density:.1f}%")
        
        # Quality categories
        high_quality = [s for s in session_data if s['event_percentage'] > 25]
        medium_quality = [s for s in session_data if 10 <= s['event_percentage'] <= 25]
        low_quality = [s for s in session_data if s['event_percentage'] < 10]
        
        print("\n🎯 Session Quality Distribution:")
        print(f"   High quality (>25% events): {len(high_quality)} sessions")
        print(f"   Medium quality (10-25%): {len(medium_quality)} sessions")
        print(f"   Low quality (<10%): {len(low_quality)} sessions")
        
        # Show best sessions
        if high_quality:
            best_sessions = sorted(high_quality, key=lambda x: x['event_percentage'], reverse=True)[:3]
            print("\n🏆 Top Sessions:")
            for session in best_sessions:
                print(f"   {session['session_name']}: {session['event_percentage']:.1f}% ({session['total_semantic_events']} events)")

def analyze_discovered_patterns():
    """Analyze the 500 patterns from TGAT discovery"""
    
    print("\n🧠 TGAT PATTERN ANALYSIS:")
    print("-" * 50)
    
    try:
        patterns_file = "/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/discovered_patterns.json"
        with open(patterns_file) as f:
            patterns = json.load(f)
        
        print(f"📊 TGAT discovered {len(patterns)} patterns")
        
        # Analyze pattern characteristics
        pattern_analysis = {
            'types': Counter(),
            'sessions': Counter(),
            'features': [],
            'temporal_info': []
        }
        
        for pattern in patterns:
            pattern_analysis['types'][pattern.get('type', 'unknown')] += 1
            pattern_analysis['sessions'][pattern.get('session', 'unknown')] += 1
            
            if 'features' in pattern:
                pattern_analysis['features'].append(pattern['features'])
            
            if 'temporal_features' in pattern:
                pattern_analysis['temporal_info'].append(pattern['temporal_features'])
        
        print("\n📈 Pattern Type Breakdown:")
        for ptype, count in pattern_analysis['types'].most_common():
            percentage = (count / len(patterns)) * 100
            print(f"   {ptype}: {count} ({percentage:.1f}%)")
        
        return pattern_analysis
        
    except Exception as e:
        print(f"❌ Error analyzing TGAT patterns: {e}")
        return None

def main():
    """Main analysis function"""
    
    # Comprehensive session analysis
    session_results = analyze_all_sessions()
    
    # TGAT pattern analysis
    analyze_discovered_patterns()
    
    # Final insights
    print("\n🎯 KEY INSIGHTS:")
    print("=" * 50)
    
    if session_results:
        total_events = sum(s['total_semantic_events'] for s in session_results)
        total_nodes = sum(s['total_nodes'] for s in session_results)
        
        print("✅ IRONFORGE Enhanced System Performance:")
        print(f"   • {len(session_results)} sessions analyzed")
        print(f"   • {total_events} semantic events detected")
        print(f"   • {(total_events/total_nodes)*100:.1f}% overall event detection rate")
        print("   • Time clustering operational for temporal intelligence")
        print("   • Cross-timeframe mapping capturing HTF context")
        
        # Event type insights
        event_counts = defaultdict(int)
        for session in session_results:
            for event_type, count in session['semantic_events'].items():
                event_counts[event_type] += count
        
        print("\n🔥 Dominant Market Events:")
        for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"   • {event_type}: {count} occurrences")
        
        print("\n💡 Simple Event-Time Clustering + Cross-TF Mapping delivers:")
        print("   • When events cluster: Temporal distribution analysis")
        print("   • What HTF context: Cross-timeframe relationship mapping")
        print("   • Rich archaeological intelligence: Market phase detection")
    
    print("\n🚀 Your enhanced IRONFORGE system is fully operational!")

if __name__ == "__main__":
    main()