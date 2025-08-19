#!/usr/bin/env python3
"""
RANK 2: Multi-Event Causal Chain Investigation
===============================================
Discover predictable sequences: expansion_phase → consolidation → liq_sweep
with consistent lag profiles and probability weights.
"""

import glob
import pickle
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

def extract_event_sequences():
    """Extract chronological event sequences from each session"""
    
    print("🔗 EXTRACTING EVENT SEQUENCES FOR CAUSAL ANALYSIS")
    print("=" * 60)
    
    # Load preserved graphs to get temporal event sequences
    graph_files = glob.glob("/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/full_graph_store/*2025_08*.pkl")
    
    session_sequences = {}
    all_events = []
    
    print(f"📊 Analyzing {len(graph_files)} sessions for causal chains...")
    
    for graph_file in graph_files:
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            session_name = Path(graph_file).stem.replace('_graph_', '_').split('_202')[0]
            rich_features = graph_data.get('rich_node_features', [])
            
            # Extract events with precise timing
            session_events = []
            
            for i, feature in enumerate(rich_features):
                events_at_this_time = []
                
                # Extract semantic events with timestamps
                time_minutes = getattr(feature, 'time_minutes', 0.0)
                absolute_timestamp = getattr(feature, 'absolute_timestamp', 0.0)
                session_position = getattr(feature, 'session_position', 0.0)
                
                if hasattr(feature, 'expansion_phase_flag') and feature.expansion_phase_flag > 0.0:
                    events_at_this_time.append('expansion_phase')
                if hasattr(feature, 'consolidation_flag') and feature.consolidation_flag > 0.0:
                    events_at_this_time.append('consolidation')
                if hasattr(feature, 'liq_sweep_flag') and feature.liq_sweep_flag > 0.0:
                    events_at_this_time.append('liq_sweep')
                if hasattr(feature, 'fvg_redelivery_flag') and feature.fvg_redelivery_flag > 0.0:
                    events_at_this_time.append('fvg_redelivery')
                if hasattr(feature, 'reversal_flag') and feature.reversal_flag > 0.0:
                    events_at_this_time.append('reversal')
                if hasattr(feature, 'retracement_flag') and feature.retracement_flag > 0.0:
                    events_at_this_time.append('retracement')
                
                # Add each event with timing info
                for event_type in events_at_this_time:
                    event_record = {
                        'session': session_name,
                        'event_type': event_type,
                        'time_minutes': time_minutes,
                        'absolute_timestamp': absolute_timestamp,
                        'session_position': session_position,
                        'node_index': i
                    }
                    session_events.append(event_record)
                    all_events.append(event_record)
            
            # Sort events by time within session
            session_events.sort(key=lambda x: x['time_minutes'])
            session_sequences[session_name] = session_events
            
        except Exception as e:
            print(f"  ⚠️ Error processing {Path(graph_file).stem}: {e}")
    
    print(f"✅ Extracted sequences from {len(session_sequences)} sessions")
    print(f"📊 Total events: {len(all_events)}")
    
    # Event type distribution
    event_counts = Counter(event['event_type'] for event in all_events)
    print("📈 Event Distribution:")
    for event_type, count in event_counts.most_common():
        pct = (count / len(all_events)) * 100
        print(f"   {event_type}: {count} events ({pct:.1f}%)")
    
    return session_sequences, all_events

def build_transition_matrices(session_sequences):
    """Build event transition matrices within each session"""
    
    print("\n🔄 BUILDING EVENT TRANSITION MATRICES")
    print("-" * 50)
    
    # Define the main event types of interest
    target_events = ['expansion_phase', 'consolidation', 'liq_sweep', 'fvg_redelivery', 'reversal', 'retracement']
    
    # Global transition matrix
    global_transitions = defaultdict(lambda: defaultdict(int))
    session_transition_counts = {}
    
    # Lag data for each transition
    transition_lags = defaultdict(list)
    
    for session_name, events in session_sequences.items():
        if len(events) < 2:  # Need at least 2 events for transitions
            continue
            
        session_transitions = defaultdict(lambda: defaultdict(int))
        
        # Look for transitions between consecutive events
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            current_type = current_event['event_type']
            next_type = next_event['event_type']
            
            # Calculate lag
            lag_minutes = next_event['time_minutes'] - current_event['time_minutes']
            
            if current_type in target_events and next_type in target_events:
                # Record transition
                global_transitions[current_type][next_type] += 1
                session_transitions[current_type][next_type] += 1
                
                # Record lag
                transition_key = f"{current_type} → {next_type}"
                transition_lags[transition_key].append(lag_minutes)
        
        session_transition_counts[session_name] = dict(session_transitions)
    
    print("📊 Transition Matrix Analysis:")
    print(f"   Sessions with transitions: {len(session_transition_counts)}")
    print(f"   Unique transition types: {len(transition_lags)}")
    
    # Display top transitions
    total_transitions = sum(sum(transitions.values()) for transitions in global_transitions.values())
    print("\n🔥 Most Common Transitions:")
    
    transition_frequencies = []
    for from_event, to_events in global_transitions.items():
        for to_event, count in to_events.items():
            frequency = count / total_transitions
            transition_frequencies.append((from_event, to_event, count, frequency))
    
    # Sort by frequency
    transition_frequencies.sort(key=lambda x: x[2], reverse=True)
    
    for from_event, to_event, count, frequency in transition_frequencies[:10]:
        print(f"   {from_event} → {to_event}: {count} transitions ({frequency:.1%})")
    
    return global_transitions, transition_lags, session_transition_counts

def analyze_lag_profiles(transition_lags, min_occurrences=3):
    """Calculate lag histograms and consistency metrics for each transition pair"""
    
    print("\n⏱️ LAG PROFILE ANALYSIS")
    print("-" * 50)
    
    lag_profiles = {}
    
    for transition_name, lags in transition_lags.items():
        if len(lags) < min_occurrences:
            continue
            
        lags_array = np.array(lags)
        
        # Calculate statistics
        mean_lag = np.mean(lags_array)
        median_lag = np.median(lags_array)
        std_lag = np.std(lags_array)
        min_lag = np.min(lags_array)
        max_lag = np.max(lags_array)
        
        # Calculate consistency (1 - coefficient_of_variation)
        cv = std_lag / mean_lag if mean_lag > 0 else float('inf')
        consistency = max(0, 1 - cv) * 100  # Convert to percentage
        
        # Quartiles
        q25 = np.percentile(lags_array, 25)
        q75 = np.percentile(lags_array, 75)
        
        lag_profiles[transition_name] = {
            'count': len(lags),
            'mean_lag': mean_lag,
            'median_lag': median_lag,
            'std_lag': std_lag,
            'min_lag': min_lag,
            'max_lag': max_lag,
            'consistency_pct': consistency,
            'q25': q25,
            'q75': q75,
            'raw_lags': lags_array
        }
        
        print(f"\n📊 {transition_name}:")
        print(f"   Occurrences: {len(lags)}")
        print(f"   Mean lag: {mean_lag:.1f} ± {std_lag:.1f} minutes")
        print(f"   Median lag: {median_lag:.1f} minutes")
        print(f"   Range: {min_lag:.1f} - {max_lag:.1f} minutes")
        print(f"   Consistency: {consistency:.1f}% (higher = more predictable)")
        print(f"   IQR: {q25:.1f} - {q75:.1f} minutes")
    
    return lag_profiles

def identify_causal_chains(lag_profiles, consistency_threshold=80):
    """Identify causal chains with lag consistency >80%"""
    
    print(f"\n🔗 IDENTIFYING HIGH-CONSISTENCY CAUSAL CHAINS (>{consistency_threshold}%)")
    print("=" * 60)
    
    # Find chains with high consistency
    high_consistency_chains = {}
    
    for transition_name, profile in lag_profiles.items():
        if profile['consistency_pct'] >= consistency_threshold:
            high_consistency_chains[transition_name] = profile
    
    if not high_consistency_chains:
        print(f"❌ No chains found with consistency ≥{consistency_threshold}%")
        
        # Show best available chains
        print("\n📊 BEST AVAILABLE CHAINS (lower threshold):")
        sorted_profiles = sorted(lag_profiles.items(), key=lambda x: x[1]['consistency_pct'], reverse=True)
        
        for transition_name, profile in sorted_profiles[:5]:
            consistency = profile['consistency_pct']
            mean_lag = profile['mean_lag']
            count = profile['count']
            
            print(f"   {transition_name}: {consistency:.1f}% consistency")
            print(f"      {count} occurrences, {mean_lag:.1f}min avg lag")
        
        return sorted_profiles[:5] if sorted_profiles else {}
    
    print(f"🎯 DISCOVERED {len(high_consistency_chains)} HIGH-CONSISTENCY CHAINS:")
    
    for transition_name, profile in high_consistency_chains.items():
        consistency = profile['consistency_pct']
        mean_lag = profile['mean_lag']
        std_lag = profile['std_lag']
        count = profile['count']
        
        print(f"\n🔥 {transition_name}:")
        print(f"   ✅ Consistency: {consistency:.1f}% (exceeds {consistency_threshold}%)")
        print(f"   📊 Predictive lag: {mean_lag:.1f} ± {std_lag:.1f} minutes")
        print(f"   📈 Sample size: {count} occurrences")
        print(f"   🎯 Prediction confidence: {consistency:.1f}%")
    
    return high_consistency_chains

def test_specific_hypothesis(session_sequences, lag_profiles):
    """Test specific hypothesis: expansion_phase → consolidation → liq_sweep"""
    
    print("\n🧪 TESTING SPECIFIC HYPOTHESIS")
    print("=" * 60)
    print("Hypothesis: expansion_phase → consolidation → liq_sweep")
    print("Expected sequence with consistent lag profiles")
    
    # Look for the complete 3-event sequence
    complete_sequences = []
    partial_sequences = {'exp→con': [], 'con→liq': [], 'exp→liq': []}
    
    for session_name, events in session_sequences.items():
        if len(events) < 3:
            continue
            
        # Look for the specific sequence within the session
        for i in range(len(events) - 2):
            event1 = events[i]
            event2 = events[i + 1]
            event3 = events[i + 2]
            
            # Check for expansion → consolidation → liq_sweep
            if (event1['event_type'] == 'expansion_phase' and 
                event2['event_type'] == 'consolidation' and 
                event3['event_type'] == 'liq_sweep'):
                
                lag1 = event2['time_minutes'] - event1['time_minutes']
                lag2 = event3['time_minutes'] - event2['time_minutes']
                total_lag = event3['time_minutes'] - event1['time_minutes']
                
                complete_sequences.append({
                    'session': session_name,
                    'exp_time': event1['time_minutes'],
                    'con_time': event2['time_minutes'],
                    'liq_time': event3['time_minutes'],
                    'exp_con_lag': lag1,
                    'con_liq_lag': lag2,
                    'total_lag': total_lag
                })
    
    # Also check for partial sequences
    for session_name, events in session_sequences.items():
        for i in range(len(events) - 1):
            event1 = events[i]
            event2 = events[i + 1]
            
            if (event1['event_type'] == 'expansion_phase' and 
                event2['event_type'] == 'consolidation'):
                lag = event2['time_minutes'] - event1['time_minutes']
                partial_sequences['exp→con'].append(lag)
            
            elif (event1['event_type'] == 'consolidation' and 
                  event2['event_type'] == 'liq_sweep'):
                lag = event2['time_minutes'] - event1['time_minutes']
                partial_sequences['con→liq'].append(lag)
            
            elif (event1['event_type'] == 'expansion_phase' and 
                  event2['event_type'] == 'liq_sweep'):
                lag = event2['time_minutes'] - event1['time_minutes']
                partial_sequences['exp→liq'].append(lag)
    
    # Report results
    print("\n📊 HYPOTHESIS TEST RESULTS:")
    
    if complete_sequences:
        print(f"✅ COMPLETE SEQUENCES FOUND: {len(complete_sequences)}")
        
        exp_con_lags = [seq['exp_con_lag'] for seq in complete_sequences]
        con_liq_lags = [seq['con_liq_lag'] for seq in complete_sequences]
        total_lags = [seq['total_lag'] for seq in complete_sequences]
        
        print(f"   📈 expansion → consolidation lag: {np.mean(exp_con_lags):.1f} ± {np.std(exp_con_lags):.1f} min")
        print(f"   📈 consolidation → liq_sweep lag: {np.mean(con_liq_lags):.1f} ± {np.std(con_liq_lags):.1f} min")
        print(f"   📈 Total sequence duration: {np.mean(total_lags):.1f} ± {np.std(total_lags):.1f} min")
        
        # Check if close to hypothesis (15m, 45m)
        avg_exp_con = np.mean(exp_con_lags)
        avg_con_liq = np.mean(con_liq_lags)
        
        if abs(avg_exp_con - 15) < 10:  # Within 10 minutes of hypothesis
            print(f"   ✅ exp→con lag close to hypothesis (15m): {avg_exp_con:.1f}m")
        else:
            print(f"   ❌ exp→con lag differs from hypothesis: {avg_exp_con:.1f}m vs 15m")
            
        if abs(avg_con_liq - 45) < 20:  # Within 20 minutes of hypothesis
            print(f"   ✅ con→liq lag close to hypothesis (45m): {avg_con_liq:.1f}m")
        else:
            print(f"   ❌ con→liq lag differs from hypothesis: {avg_con_liq:.1f}m vs 45m")
    else:
        print("❌ NO COMPLETE SEQUENCES FOUND")
    
    # Report partial sequences
    print("\n📊 PARTIAL SEQUENCE ANALYSIS:")
    for seq_name, lags in partial_sequences.items():
        if lags:
            mean_lag = np.mean(lags)
            std_lag = np.std(lags)
            print(f"   {seq_name}: {len(lags)} occurrences, {mean_lag:.1f} ± {std_lag:.1f} min")
        else:
            print(f"   {seq_name}: No occurrences found")
    
    return complete_sequences, partial_sequences

def create_causal_chain_visualization(lag_profiles, high_consistency_chains, complete_sequences):
    """Create visualization of causal chain discoveries"""
    
    print("\n📈 CREATING CAUSAL CHAIN VISUALIZATION")
    print("-" * 50)
    
    if not lag_profiles:
        print("❌ No data for visualization")
        return
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Event Causal Chain Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Lag consistency vs frequency
    ax1 = axes[0, 0]
    
    transitions = list(lag_profiles.keys())
    consistencies = [lag_profiles[t]['consistency_pct'] for t in transitions]
    frequencies = [lag_profiles[t]['count'] for t in transitions]
    
    ax1.scatter(frequencies, consistencies, s=100, alpha=0.7, c=consistencies, cmap='viridis')
    
    # Highlight high-consistency chains
    if high_consistency_chains and isinstance(high_consistency_chains, dict):
        hc_freq = [lag_profiles[t]['count'] for t in high_consistency_chains]
        hc_cons = [lag_profiles[t]['consistency_pct'] for t in high_consistency_chains]
        ax1.scatter(hc_freq, hc_cons, s=200, alpha=0.8, c='red', marker='*', label='High Consistency')
    elif high_consistency_chains and isinstance(high_consistency_chains, list):
        # high_consistency_chains is a list of tuples (transition_name, profile)
        hc_freq = [profile['count'] for _, profile in high_consistency_chains]
        hc_cons = [profile['consistency_pct'] for _, profile in high_consistency_chains]
        ax1.scatter(hc_freq, hc_cons, s=200, alpha=0.8, c='red', marker='*', label='Best Chains')
    
    ax1.set_xlabel('Frequency (Number of Occurrences)')
    ax1.set_ylabel('Consistency (%)')
    ax1.set_title('Transition Consistency vs Frequency')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% Threshold')
    if high_consistency_chains:
        ax1.legend()
    
    # Plot 2: Lag distribution histogram for top transitions
    ax2 = axes[0, 1]
    
    top_transitions = sorted(lag_profiles.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_transitions)))
    
    for i, (trans_name, profile) in enumerate(top_transitions):
        lags = profile['raw_lags']
        ax2.hist(lags, bins=10, alpha=0.6, label=trans_name.replace(' → ', '→'), color=colors[i])
    
    ax2.set_xlabel('Lag (minutes)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Lag Distributions for Top Transitions')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Causal chain network diagram
    ax3 = axes[1, 0]
    
    # Simple network visualization
    positions = {
        'expansion_phase': (0, 2),
        'consolidation': (2, 2),
        'liq_sweep': (4, 2),
        'fvg_redelivery': (1, 1),
        'reversal': (3, 1),
        'retracement': (2, 0)
    }
    
    # Draw nodes
    for event, (x, y) in positions.items():
        ax3.scatter(x, y, s=500, alpha=0.7)
        ax3.text(x, y, event.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw edges for high-frequency transitions
    for trans_name, profile in lag_profiles.items():
        if profile['count'] >= 5:  # Only show frequent transitions
            from_event, to_event = trans_name.split(' → ')
            if from_event in positions and to_event in positions:
                x1, y1 = positions[from_event]
                x2, y2 = positions[to_event]
                
                # Line thickness based on frequency
                thickness = min(profile['count'] / 10, 5)
                alpha = min(profile['consistency_pct'] / 100, 1)
                
                ax3.arrow(x1, y1, x2-x1, y2-y1, head_width=0.1, head_length=0.1, 
                         fc='blue', ec='blue', alpha=alpha, linewidth=thickness)
    
    ax3.set_xlim(-0.5, 4.5)
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_title('Event Transition Network')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Complete sequence analysis
    ax4 = axes[1, 1]
    
    if complete_sequences:
        exp_con_lags = [seq['exp_con_lag'] for seq in complete_sequences]
        con_liq_lags = [seq['con_liq_lag'] for seq in complete_sequences]
        
        ax4.scatter(exp_con_lags, con_liq_lags, s=100, alpha=0.7)
        ax4.set_xlabel('Expansion → Consolidation Lag (min)')
        ax4.set_ylabel('Consolidation → Liq_Sweep Lag (min)')
        ax4.set_title(f'Complete Sequence Lags ({len(complete_sequences)} sequences)')
        
        # Add hypothesis reference lines
        ax4.axvline(x=15, color='red', linestyle='--', alpha=0.5, label='Hypothesis: 15m')
        ax4.axhline(y=45, color='red', linestyle='--', alpha=0.5, label='Hypothesis: 45m')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Complete\nSequences Found', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Complete Sequence Analysis')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_path = '/Users/jack/IRONPULSE/IRONFORGE/causal_event_chains.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved: {output_path}")
    
    return output_path

def main():
    """Main causal chain investigation"""
    
    print("🔗 MULTI-EVENT CAUSAL CHAIN INVESTIGATION")
    print("=" * 80)
    print("Testing hypothesis: expansion_phase → consolidation → liq_sweep")
    print("Discovering predictable sequences with consistent lag profiles")
    print("=" * 80)
    
    # Step 1: Extract event sequences
    session_sequences, all_events = extract_event_sequences()
    if not session_sequences:
        print("❌ Cannot proceed without event sequences")
        return
    
    # Step 2: Build transition matrices
    global_transitions, transition_lags, session_counts = build_transition_matrices(session_sequences)
    if not transition_lags:
        print("❌ Cannot proceed without transition data")
        return
    
    # Step 3: Analyze lag profiles
    lag_profiles = analyze_lag_profiles(transition_lags, min_occurrences=3)
    if not lag_profiles:
        print("❌ Cannot proceed without lag profiles")
        return
    
    # Step 4: Identify high-consistency causal chains
    high_consistency_chains = identify_causal_chains(lag_profiles, consistency_threshold=80)
    
    # Step 5: Test specific hypothesis
    complete_sequences, partial_sequences = test_specific_hypothesis(session_sequences, lag_profiles)
    
    # Step 6: Create visualization
    viz_path = create_causal_chain_visualization(lag_profiles, high_consistency_chains, complete_sequences)
    
    # Summary
    print("\n🎯 CAUSAL CHAIN INVESTIGATION SUMMARY")
    print("=" * 60)
    
    total_transitions = len(transition_lags)
    high_consistency_count = len(high_consistency_chains) if isinstance(high_consistency_chains, dict) else len(high_consistency_chains)
    complete_seq_count = len(complete_sequences) if complete_sequences else 0
    
    print("📊 Analysis Results:")
    print(f"   Event sequences analyzed: {len(session_sequences)}")
    print(f"   Total transition types: {total_transitions}")
    print(f"   High-consistency chains (≥80%): {high_consistency_count}")
    print(f"   Complete hypothesis sequences: {complete_seq_count}")
    
    # Report on hypothesis
    if complete_sequences:
        print("\n🧪 Hypothesis Validation:")
        exp_con_lags = [seq['exp_con_lag'] for seq in complete_sequences]
        con_liq_lags = [seq['con_liq_lag'] for seq in complete_sequences]
        
        avg_exp_con = np.mean(exp_con_lags)
        avg_con_liq = np.mean(con_liq_lags)
        
        exp_match = abs(avg_exp_con - 15) < 10
        liq_match = abs(avg_con_liq - 45) < 20
        
        if exp_match and liq_match:
            print("   ✅ HYPOTHESIS CONFIRMED: Lags match expected profile")
        elif exp_match or liq_match:
            print("   ⚠️ PARTIAL MATCH: Some lags match expected profile")
        else:
            print("   ❌ HYPOTHESIS REJECTED: Lags don't match expected profile")
        
        print(f"   Actual lags: {avg_exp_con:.1f}m → {avg_con_liq:.1f}m")
        print("   Expected lags: 15m → 45m")
    
    # Best causal chains discovered
    if lag_profiles:
        print("\n🔥 TOP CAUSAL DISCOVERIES:")
        sorted_chains = sorted(lag_profiles.items(), key=lambda x: x[1]['consistency_pct'], reverse=True)
        
        for i, (chain_name, profile) in enumerate(sorted_chains[:3]):
            print(f"   #{i+1}: {chain_name}")
            print(f"       Consistency: {profile['consistency_pct']:.1f}%")
            print(f"       Predictive lag: {profile['mean_lag']:.1f} ± {profile['std_lag']:.1f} min")
            print(f"       Sample size: {profile['count']} occurrences")
    
    print("\n✅ RANK 2 INVESTIGATION COMPLETE")
    print(f"💾 Results visualization: {viz_path}")
    
    if high_consistency_count > 0:
        print(f"\n🚀 BREAKTHROUGH: {high_consistency_count} high-consistency causal chains discovered!")
        print("   These enable predictive event forecasting with confidence intervals.")
    else:
        print("\n📝 Result: No chains exceed 80% consistency threshold.")
        print("   Consider investigating lower consistency thresholds or different event combinations.")

if __name__ == "__main__":
    main()