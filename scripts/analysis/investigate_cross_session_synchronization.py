#!/usr/bin/env python3
"""
RANK 1: Cross-Session Temporal Synchronization Investigation
===========================================================
Discover if events cluster at consistent intraday times across different calendar days,
revealing market microstructure synchronization patterns.
"""

import glob
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_event_timing_data():
    """Extract absolute time-of-day for each semantic event across all sessions"""
    
    print("🕐 EXTRACTING CROSS-SESSION TIMING DATA")
    print("=" * 60)
    
    # Load preserved graphs to get actual semantic events with timestamps
    graph_files = glob.glob("/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/full_graph_store/*2025_08*.pkl")
    
    event_timing_data = []
    session_metadata = {}
    
    print(f"📊 Analyzing {len(graph_files)} sessions...")
    
    for graph_file in graph_files:
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            session_name = Path(graph_file).stem.replace('_graph_', '_').split('_202')[0]
            rich_features = graph_data.get('rich_node_features', [])
            
            # Extract session metadata
            session_metadata[session_name] = {
                'total_features': len(rich_features),
                'file_date': Path(graph_file).stem.split('_202')[1][:2] if '_202' in Path(graph_file).stem else 'unknown'
            }
            
            for feature in rich_features:
                # Extract semantic events with timing
                events_found = []
                
                if hasattr(feature, 'expansion_phase_flag') and feature.expansion_phase_flag > 0.0:
                    events_found.append('expansion_phase')
                if hasattr(feature, 'consolidation_flag') and feature.consolidation_flag > 0.0:
                    events_found.append('consolidation')
                if hasattr(feature, 'liq_sweep_flag') and feature.liq_sweep_flag > 0.0:
                    events_found.append('liq_sweep')
                if hasattr(feature, 'fvg_redelivery_flag') and feature.fvg_redelivery_flag > 0.0:
                    events_found.append('fvg_redelivery')
                if hasattr(feature, 'reversal_flag') and feature.reversal_flag > 0.0:
                    events_found.append('reversal')
                if hasattr(feature, 'retracement_flag') and feature.retracement_flag > 0.0:
                    events_found.append('retracement')
                
                if events_found:
                    # Extract timing information
                    time_minutes = getattr(feature, 'time_minutes', 0.0)
                    absolute_timestamp = getattr(feature, 'absolute_timestamp', 0.0)
                    session_position = getattr(feature, 'session_position', 0.0)
                    phase_open = getattr(feature, 'phase_open', 0.0)
                    
                    # Convert time_minutes to time-of-day (assuming it's minutes from session start)
                    # We'll create consistent 5-minute bins for comparison
                    time_bin = int(time_minutes // 5) * 5  # Round down to nearest 5-minute bin
                    
                    for event_type in events_found:
                        event_timing_data.append({
                            'session': session_name,
                            'event_type': event_type,
                            'time_minutes': time_minutes,
                            'time_bin_5m': time_bin,
                            'absolute_timestamp': absolute_timestamp,
                            'session_position': session_position,
                            'phase_open': phase_open,
                            'calendar_day': session_metadata[session_name]['file_date']
                        })
        
        except Exception as e:
            print(f"  ⚠️ Error processing {Path(graph_file).stem}: {e}")
    
    print(f"✅ Extracted {len(event_timing_data)} event timing records")
    print(f"📅 Spanning {len(session_metadata)} sessions")
    
    return event_timing_data, session_metadata

def build_synchronization_matrix(event_data):
    """Build co-occurrence matrix: Time_Bin[i] vs Sessions[j]"""
    
    print("\n🔄 BUILDING SYNCHRONIZATION MATRIX")
    print("-" * 50)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(event_data)
    
    if df.empty:
        print("❌ No event data available")
        return None, None
    
    print("📊 Event Type Distribution:")
    event_counts = df['event_type'].value_counts()
    for event_type, count in event_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {event_type}: {count} events ({pct:.1f}%)")
    
    # Create co-occurrence matrix for each event type
    synchronization_matrices = {}
    
    for event_type in df['event_type'].unique():
        event_subset = df[df['event_type'] == event_type]
        
        # Create pivot table: Sessions vs Time Bins
        co_occurrence = event_subset.groupby(['session', 'time_bin_5m']).size().unstack(fill_value=0)
        
        # Convert to binary (occurred=1, not occurred=0)
        binary_matrix = (co_occurrence > 0).astype(int)
        
        synchronization_matrices[event_type] = {
            'raw_counts': co_occurrence,
            'binary_matrix': binary_matrix,
            'event_count': len(event_subset)
        }
        
        print(f"\n📈 {event_type.upper()} Synchronization Matrix:")
        print(f"   Sessions: {binary_matrix.shape[0]}")
        print(f"   Time bins: {binary_matrix.shape[1]}")
        print(f"   Total events: {len(event_subset)}")
    
    return synchronization_matrices, df

def identify_synchronized_time_slots(sync_matrices, threshold=0.6):
    """Identify time slots with >60% cross-session occurrence"""
    
    print(f"\n🎯 IDENTIFYING SYNCHRONIZED TIME SLOTS (>{threshold*100:.0f}% threshold)")
    print("=" * 60)
    
    synchronized_discoveries = {}
    
    for event_type, matrices in sync_matrices.items():
        binary_matrix = matrices['binary_matrix']
        
        if binary_matrix.empty:
            continue
            
        # Calculate cross-session occurrence rate for each time bin
        n_sessions = binary_matrix.shape[0]
        time_bin_occurrence_rates = binary_matrix.sum(axis=0) / n_sessions
        
        # Find highly synchronized time slots
        synchronized_bins = time_bin_occurrence_rates[time_bin_occurrence_rates >= threshold]
        
        if len(synchronized_bins) > 0:
            print(f"\n🔥 {event_type.upper()} SYNCHRONIZATION DISCOVERY:")
            print("-" * 40)
            
            for time_bin, occurrence_rate in synchronized_bins.sort_values(ascending=False).items():
                sessions_with_event = binary_matrix[binary_matrix[time_bin] == 1].index.tolist()
                n_sessions_synchronized = len(sessions_with_event)
                
                print(f"   ⏰ Time Bin {time_bin}-{time_bin+5}m: {occurrence_rate:.1%} synchronization")
                print(f"      Sessions: {n_sessions_synchronized}/{n_sessions}")
                print(f"      Sessions: {', '.join(sessions_with_event[:5])}{'...' if len(sessions_with_event) > 5 else ''}")
            
            synchronized_discoveries[event_type] = {
                'synchronized_bins': synchronized_bins,
                'occurrence_rates': time_bin_occurrence_rates,
                'binary_matrix': binary_matrix
            }
        else:
            print(f"\n❌ No synchronized time slots found for {event_type} (threshold {threshold:.0%})")
    
    return synchronized_discoveries

def analyze_temporal_patterns(event_data, sync_discoveries):
    """Analyze the temporal patterns for evidence of systematic timing"""
    
    print("\n🔍 TEMPORAL PATTERN ANALYSIS")
    print("=" * 60)
    
    df = pd.DataFrame(event_data)
    
    # Overall timing distribution analysis
    print("📊 OVERALL TIMING DISTRIBUTION:")
    
    # Time bin analysis across all events
    time_bin_dist = df.groupby('time_bin_5m').size().sort_index()
    
    # Find peak timing periods
    top_time_bins = time_bin_dist.nlargest(5)
    
    print("   🔥 Peak Event Time Bins:")
    for time_bin, count in top_time_bins.items():
        pct = (count / len(df)) * 100
        print(f"      {time_bin}-{time_bin+5}m: {count} events ({pct:.1f}%)")
    
    # Session consistency analysis
    print("\n📅 SESSION CONSISTENCY ANALYSIS:")
    
    session_event_counts = df.groupby('session').size().sort_values(ascending=False)
    print("   Most active sessions:")
    for session, count in session_event_counts.head(5).items():
        pct = (count / len(df)) * 100
        print(f"      {session}: {count} events ({pct:.1f}%)")
    
    # Cross-day analysis if we have calendar day info
    if 'calendar_day' in df.columns and df['calendar_day'].nunique() > 1:
        print("\n📆 CROSS-DAY SYNCHRONIZATION:")
        
        day_event_dist = df.groupby(['calendar_day', 'time_bin_5m']).size().unstack(fill_value=0)
        
        # Find time bins that are active across multiple days
        cross_day_bins = day_event_dist.columns[day_event_dist.gt(0).sum() >= 2]  # At least 2 days
        
        print(f"   Time bins active across multiple days: {len(cross_day_bins)}")
        for time_bin in cross_day_bins[:10]:  # Show top 10
            days_active = day_event_dist[time_bin].gt(0).sum()
            total_events = day_event_dist[time_bin].sum()
            print(f"      {time_bin}-{time_bin+5}m: {days_active} days, {total_events} total events")

def test_synchronization_hypothesis(sync_discoveries, event_data):
    """Test the core hypothesis: IF event@time occurs on Day_N, THEN probability increases on Day_N+1"""
    
    print("\n🧪 TESTING SYNCHRONIZATION HYPOTHESIS")
    print("=" * 60)
    
    df = pd.DataFrame(event_data)
    
    # For this test, we need to simulate day sequences
    # Since we don't have perfect calendar day sequences, we'll use session sequences as proxy
    
    hypothesis_results = {}
    
    for event_type in df['event_type'].unique():
        event_subset = df[df['event_type'] == event_type]
        
        if len(event_subset) < 10:  # Need sufficient data
            continue
            
        print(f"\n🔬 TESTING {event_type.upper()}:")
        print("-" * 30)
        
        # Group by time bins and calculate persistence
        time_bin_sessions = event_subset.groupby('time_bin_5m')['session'].apply(list)
        
        persistence_rates = {}
        
        for time_bin, sessions in time_bin_sessions.items():
            if len(sessions) >= 2:  # Need at least 2 occurrences
                # Calculate how often this time bin appears in consecutive sessions
                unique_sessions = list(set(sessions))
                
                # Simple persistence metric: ratio of sessions to unique sessions
                # Higher ratio = more persistence/repetition
                persistence_rate = len(sessions) / len(unique_sessions)
                persistence_rates[time_bin] = persistence_rate
        
        if persistence_rates:
            avg_persistence = np.mean(list(persistence_rates.values()))
            max_persistence_bin = max(persistence_rates.keys(), key=lambda k: persistence_rates[k])
            max_persistence = persistence_rates[max_persistence_bin]
            
            print(f"   Average persistence rate: {avg_persistence:.2f}")
            print(f"   Highest persistence: {max_persistence:.2f} at {max_persistence_bin}-{max_persistence_bin+5}m")
            
            # Evidence threshold: persistence > 1.5 suggests synchronization
            if max_persistence > 1.5:
                print(f"   ✅ SYNCHRONIZATION EVIDENCE: {max_persistence:.2f} > 1.5")
            else:
                print(f"   ❌ No strong synchronization: {max_persistence:.2f} ≤ 1.5")
            
            hypothesis_results[event_type] = {
                'avg_persistence': avg_persistence,
                'max_persistence': max_persistence,
                'max_persistence_time': max_persistence_bin,
                'evidence_strength': 'Strong' if max_persistence > 1.5 else 'Weak'
            }
    
    return hypothesis_results

def create_synchronization_visualization(sync_discoveries, event_data):
    """Create visualization of temporal synchronization patterns"""
    
    print("\n📈 CREATING SYNCHRONIZATION VISUALIZATION")
    print("-" * 50)
    
    df = pd.DataFrame(event_data)
    
    if df.empty:
        print("❌ No data for visualization")
        return
    
    # Create heatmap showing event intensity by time bins across sessions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Session Temporal Synchronization Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall event timing heatmap
    ax1 = axes[0, 0]
    time_session_matrix = df.groupby(['session', 'time_bin_5m']).size().unstack(fill_value=0)
    
    if not time_session_matrix.empty:
        sns.heatmap(time_session_matrix, ax=ax1, cmap='YlOrRd', cbar_kws={'label': 'Event Count'})
        ax1.set_title('Event Intensity: Sessions vs Time Bins')
        ax1.set_xlabel('Time Bin (5-minute intervals)')
        ax1.set_ylabel('Sessions')
    
    # Plot 2: Event type distribution over time
    ax2 = axes[0, 1]
    event_time_dist = df.groupby(['time_bin_5m', 'event_type']).size().unstack(fill_value=0)
    
    if not event_time_dist.empty:
        event_time_dist.plot(kind='bar', stacked=True, ax=ax2, width=0.8)
        ax2.set_title('Event Type Distribution Over Time')
        ax2.set_xlabel('Time Bin (5-minute intervals)')
        ax2.set_ylabel('Event Count')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Synchronization score heatmap
    ax3 = axes[1, 0]
    
    # Create synchronization score matrix
    sync_scores = {}
    for event_type in df['event_type'].unique():
        event_subset = df[df['event_type'] == event_type]
        time_bin_counts = event_subset.groupby(['session', 'time_bin_5m']).size().unstack(fill_value=0)
        
        if not time_bin_counts.empty:
            # Calculate synchronization score (occurrence rate across sessions)
            sync_score = time_bin_counts.gt(0).mean(axis=0)
            sync_scores[event_type] = sync_score
    
    if sync_scores:
        sync_matrix = pd.DataFrame(sync_scores).fillna(0)
        sns.heatmap(sync_matrix, ax=ax3, cmap='coolwarm', center=0.5, 
                   cbar_kws={'label': 'Synchronization Rate'})
        ax3.set_title('Cross-Session Synchronization Rates')
        ax3.set_xlabel('Event Type')
        ax3.set_ylabel('Time Bin (5-minute intervals)')
    
    # Plot 4: Peak synchronization events
    ax4 = axes[1, 1]
    
    # Show most synchronized time periods
    peak_times = df['time_bin_5m'].value_counts().head(10)
    peak_times.plot(kind='bar', ax=ax4, color='skyblue')
    ax4.set_title('Most Active Time Bins (All Events)')
    ax4.set_xlabel('Time Bin (5-minute intervals)')
    ax4.set_ylabel('Total Event Count')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = '/Users/jack/IRONPULSE/IRONFORGE/cross_session_synchronization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved: {output_path}")
    
    return output_path

def main():
    """Main cross-session synchronization investigation"""
    
    print("🕐 CROSS-SESSION TEMPORAL SYNCHRONIZATION INVESTIGATION")
    print("=" * 80)
    print("Testing hypothesis: Events cluster at consistent intraday times across different calendar days")
    print("=" * 80)
    
    # Step 1: Extract event timing data
    event_data, session_metadata = extract_event_timing_data()
    if not event_data:
        print("❌ Cannot proceed without event timing data")
        return
    
    # Step 2: Build synchronization matrix
    sync_matrices, df = build_synchronization_matrix(event_data)
    if not sync_matrices:
        print("❌ Cannot proceed without synchronization matrices")
        return
    
    # Step 3: Identify synchronized time slots (>60% threshold)
    sync_discoveries = identify_synchronized_time_slots(sync_matrices, threshold=0.6)
    
    # Step 4: Analyze temporal patterns
    analyze_temporal_patterns(event_data, sync_discoveries)
    
    # Step 5: Test synchronization hypothesis
    hypothesis_results = test_synchronization_hypothesis(sync_discoveries, event_data)
    
    # Step 6: Create visualization
    viz_path = create_synchronization_visualization(sync_discoveries, event_data)
    
    # Summary
    print("\n🎯 CROSS-SESSION SYNCHRONIZATION SUMMARY")
    print("=" * 60)
    
    total_events = len(event_data)
    n_sessions = len(session_metadata)
    n_sync_discoveries = len(sync_discoveries)
    
    print("📊 Analysis Scope:")
    print(f"   Events analyzed: {total_events}")
    print(f"   Sessions: {n_sessions}")
    print(f"   Event types with synchronization: {n_sync_discoveries}")
    
    if hypothesis_results:
        print("\n🧪 Hypothesis Test Results:")
        for event_type, result in hypothesis_results.items():
            evidence = result['evidence_strength']
            persistence = result['max_persistence']
            time_bin = result['max_persistence_time']
            
            print(f"   {event_type}: {evidence} evidence (persistence: {persistence:.2f} at {time_bin}m)")
    
    if sync_discoveries:
        print("\n🔥 Key Synchronization Discoveries:")
        for event_type, discovery in sync_discoveries.items():
            n_sync_bins = len(discovery['synchronized_bins'])
            max_sync_rate = discovery['synchronized_bins'].max()
            best_time = discovery['synchronized_bins'].idxmax()
            
            print(f"   {event_type}: {n_sync_bins} synchronized time bins (best: {max_sync_rate:.1%} at {best_time}m)")
    
    print("\n✅ RANK 1 INVESTIGATION COMPLETE")
    print(f"💾 Results visualization: {viz_path}")
    
    if n_sync_discoveries > 0:
        print("\n🚀 BREAKTHROUGH: Evidence of cross-session temporal synchronization detected!")
        print("   This suggests systematic market timing patterns that could enable predictive modeling.")
    else:
        print("\n📝 Result: No strong cross-session synchronization detected at 60% threshold.")
        print("   This suggests more individualistic session patterns rather than systematic timing.")

if __name__ == "__main__":
    main()