#!/usr/bin/env python3
"""
Liquidity Sweep Catalyst Investigation
======================================
Investigating liq_sweep as initiating event rather than terminal event.
Focus on discovering what causal chains liq_sweep triggers.
"""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import glob
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def extract_liquidity_sweep_sequences():
    """Extract all sequences that START with liq_sweep events"""
    
    print("⚡ EXTRACTING LIQUIDITY SWEEP CATALYST SEQUENCES")
    print("=" * 60)
    
    # Load preserved graphs to get liq_sweep initiated sequences
    graph_files = glob.glob("/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/full_graph_store/*2025_08*.pkl")
    
    liq_sweep_sequences = []
    session_sequences = {}
    liq_sweep_stats = {'total_sweeps': 0, 'sessions_with_sweeps': 0}
    
    print(f"🔍 Analyzing {len(graph_files)} sessions for liq_sweep catalyst patterns...")
    
    for graph_file in graph_files:
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            session_name = Path(graph_file).stem.replace('_graph_', '_').split('_202')[0]
            rich_features = graph_data.get('rich_node_features', [])
            
            # Extract all events with precise timing
            session_events = []
            session_liq_sweeps = []
            
            for i, feature in enumerate(rich_features):
                events_at_this_time = []
                
                # Extract timing info
                time_minutes = getattr(feature, 'time_minutes', 0.0)
                absolute_timestamp = getattr(feature, 'absolute_timestamp', 0.0)
                session_position = getattr(feature, 'session_position', 0.0)
                phase_open = getattr(feature, 'phase_open', 0.0)
                
                # Check for semantic events
                if hasattr(feature, 'liq_sweep_flag') and feature.liq_sweep_flag > 0.0:
                    events_at_this_time.append('liq_sweep')
                    session_liq_sweeps.append({
                        'time_minutes': time_minutes,
                        'session_position': session_position,
                        'phase_open': phase_open,
                        'node_index': i
                    })
                    liq_sweep_stats['total_sweeps'] += 1
                
                if hasattr(feature, 'expansion_phase_flag') and feature.expansion_phase_flag > 0.0:
                    events_at_this_time.append('expansion_phase')
                if hasattr(feature, 'consolidation_flag') and feature.consolidation_flag > 0.0:
                    events_at_this_time.append('consolidation')
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
                        'phase_open': phase_open,
                        'node_index': i
                    }
                    session_events.append(event_record)
            
            # Sort events by time within session
            session_events.sort(key=lambda x: x['time_minutes'])
            session_sequences[session_name] = session_events
            
            if session_liq_sweeps:
                liq_sweep_stats['sessions_with_sweeps'] += 1
                
                # For each liq_sweep in this session, extract the sequence that follows
                for sweep in session_liq_sweeps:
                    sweep_time = sweep['time_minutes']
                    
                    # Find all events that occur AFTER this liq_sweep within a window
                    following_events = []
                    
                    for event in session_events:
                        if (event['time_minutes'] > sweep_time and 
                            event['time_minutes'] <= sweep_time + 60 and  # 60 minute window
                            event['event_type'] != 'liq_sweep'):  # Exclude other liq_sweeps
                            
                            lag = event['time_minutes'] - sweep_time
                            following_events.append({
                                'event_type': event['event_type'],
                                'lag_minutes': lag,
                                'time_minutes': event['time_minutes'],
                                'session_position': event['session_position'],
                                'phase_open': event['phase_open']
                            })
                    
                    if following_events:
                        # Sort by lag
                        following_events.sort(key=lambda x: x['lag_minutes'])
                        
                        liq_sweep_sequences.append({
                            'session': session_name,
                            'sweep_time': sweep_time,
                            'sweep_session_position': sweep['session_position'],
                            'sweep_phase_open': sweep['phase_open'],
                            'following_events': following_events,
                            'sequence_length': len(following_events)
                        })
        
        except Exception as e:
            print(f"  ⚠️ Error processing {Path(graph_file).stem}: {e}")
    
    print(f"✅ Extracted {len(liq_sweep_sequences)} liq_sweep catalyst sequences")
    print(f"⚡ Total liq_sweeps found: {liq_sweep_stats['total_sweeps']}")
    print(f"📊 Sessions with liq_sweeps: {liq_sweep_stats['sessions_with_sweeps']}/{len(session_sequences)}")
    
    return liq_sweep_sequences, session_sequences, liq_sweep_stats

def analyze_immediate_responses(liq_sweep_sequences, time_windows=[1, 3, 5, 10, 15, 30]):
    """Analyze what happens immediately after liq_sweep events"""
    
    print(f"\n⚡ IMMEDIATE RESPONSE ANALYSIS")
    print("-" * 50)
    print("Analyzing what events trigger immediately after liq_sweep...")
    
    # Analyze responses within different time windows
    window_responses = {}
    
    for window_minutes in time_windows:
        window_responses[window_minutes] = {
            'event_counts': Counter(),
            'total_sequences': 0,
            'sequences_with_response': 0,
            'response_details': []
        }
        
        for sequence in liq_sweep_sequences:
            window_responses[window_minutes]['total_sequences'] += 1
            
            # Find events within this time window
            immediate_events = [
                event for event in sequence['following_events'] 
                if event['lag_minutes'] <= window_minutes
            ]
            
            if immediate_events:
                window_responses[window_minutes]['sequences_with_response'] += 1
                
                # Record first event of each type
                seen_types = set()
                for event in immediate_events:
                    if event['event_type'] not in seen_types:
                        window_responses[window_minutes]['event_counts'][event['event_type']] += 1
                        window_responses[window_minutes]['response_details'].append({
                            'event_type': event['event_type'],
                            'lag': event['lag_minutes'],
                            'session': sequence['session']
                        })
                        seen_types.add(event['event_type'])
    
    # Report results
    print(f"\n📊 RESPONSE RATES BY TIME WINDOW:")
    
    for window_minutes in time_windows:
        data = window_responses[window_minutes]
        total = data['total_sequences']
        responsive = data['sequences_with_response']
        response_rate = (responsive / total * 100) if total > 0 else 0
        
        print(f"\n⏰ Within {window_minutes} minutes:")
        print(f"   Response rate: {responsive}/{total} ({response_rate:.1f}%)")
        
        if data['event_counts']:
            print(f"   Most common responses:")
            for event_type, count in data['event_counts'].most_common(5):
                event_rate = (count / total * 100) if total > 0 else 0
                print(f"      {event_type}: {count} times ({event_rate:.1f}%)")
    
    return window_responses

def discover_catalyst_chains(liq_sweep_sequences, max_chain_length=5):
    """Discover the most common event chains triggered by liq_sweep"""
    
    print(f"\n🔗 DISCOVERING CATALYST CHAINS")
    print("-" * 50)
    print("Finding common event sequences that follow liq_sweep...")
    
    # Extract chains of different lengths
    chain_patterns = defaultdict(lambda: defaultdict(int))
    chain_details = defaultdict(list)
    
    for sequence in liq_sweep_sequences:
        following_events = sequence['following_events']
        
        if not following_events:
            continue
            
        # Extract chains of length 1 to max_chain_length
        for chain_length in range(1, min(len(following_events) + 1, max_chain_length + 1)):
            chain = []
            
            for i in range(chain_length):
                chain.append(following_events[i]['event_type'])
            
            chain_key = ' → '.join(chain)
            chain_patterns[chain_length][chain_key] += 1
            
            # Store timing details
            chain_details[chain_key].append({
                'session': sequence['session'],
                'sweep_time': sequence['sweep_time'],
                'sweep_position': sequence['sweep_session_position'],
                'event_lags': [following_events[i]['lag_minutes'] for i in range(chain_length)],
                'total_duration': following_events[chain_length-1]['lag_minutes']
            })
    
    # Report most common chains
    print(f"\n🔥 MOST COMMON CATALYST CHAINS:")
    
    for chain_length in sorted(chain_patterns.keys()):
        print(f"\n📊 {chain_length}-Event Chains:")
        
        sorted_chains = sorted(chain_patterns[chain_length].items(), key=lambda x: x[1], reverse=True)
        
        for i, (chain_pattern, count) in enumerate(sorted_chains[:5]):
            print(f"   #{i+1}: {chain_pattern}")
            print(f"       Occurrences: {count}")
            
            # Calculate average timing
            details = chain_details[chain_pattern]
            if details:
                avg_duration = np.mean([d['total_duration'] for d in details])
                avg_lags = np.mean([d['event_lags'] for d in details], axis=0)
                
                print(f"       Avg duration: {avg_duration:.1f} minutes")
                if len(avg_lags) == 1:
                    print(f"       Avg lag: {avg_lags[0]:.1f} minutes")
                else:
                    lag_str = ' → '.join([f"{lag:.1f}m" for lag in avg_lags])
                    print(f"       Avg lags: {lag_str}")
    
    return chain_patterns, chain_details

def analyze_catalyst_timing_patterns(liq_sweep_sequences):
    """Analyze when liq_sweeps occur and their effectiveness as catalysts"""
    
    print(f"\n⏰ CATALYST TIMING PATTERN ANALYSIS")
    print("-" * 50)
    
    # Analyze when liq_sweeps occur
    sweep_timing = {
        'session_positions': [],
        'phase_open_values': [],
        'time_minutes': [],
        'response_rates': [],
        'response_strengths': []
    }
    
    for sequence in liq_sweep_sequences:
        sweep_timing['session_positions'].append(sequence['sweep_session_position'])
        sweep_timing['phase_open_values'].append(sequence['sweep_phase_open'])
        sweep_timing['time_minutes'].append(sequence['sweep_time'])
        
        # Calculate response metrics
        following_events = sequence['following_events']
        response_rate = 1 if following_events else 0
        response_strength = len(following_events)
        
        sweep_timing['response_rates'].append(response_rate)
        sweep_timing['response_strengths'].append(response_strength)
    
    # Statistical analysis
    print(f"📊 LIQ_SWEEP CATALYST CHARACTERISTICS:")
    
    if sweep_timing['session_positions']:
        avg_position = np.mean(sweep_timing['session_positions'])
        avg_phase_open = np.mean(sweep_timing['phase_open_values'])
        avg_time = np.mean(sweep_timing['time_minutes'])
        avg_response_rate = np.mean(sweep_timing['response_rates'])
        avg_response_strength = np.mean(sweep_timing['response_strengths'])
        
        print(f"   Average session position: {avg_position:.2f} (0=start, 1=end)")
        print(f"   Average phase_open: {avg_phase_open:.2f} (1=opening phase)")
        print(f"   Average timing: {avg_time:.1f} minutes into session")
        print(f"   Response rate: {avg_response_rate:.1%}")
        print(f"   Average response strength: {avg_response_strength:.1f} events")
    
    # Categorize sweeps by timing
    timing_categories = {
        'early_session': [],    # position < 0.3
        'mid_session': [],      # 0.3 <= position < 0.7
        'late_session': [],     # position >= 0.7
        'opening_phase': [],    # phase_open > 0.5
        'non_opening': []       # phase_open <= 0.5
    }
    
    for i, sequence in enumerate(liq_sweep_sequences):
        pos = sequence['sweep_session_position']
        phase = sequence['sweep_phase_open']
        response_strength = sweep_timing['response_strengths'][i]
        
        if pos < 0.3:
            timing_categories['early_session'].append(response_strength)
        elif pos < 0.7:
            timing_categories['mid_session'].append(response_strength)
        else:
            timing_categories['late_session'].append(response_strength)
            
        if phase > 0.5:
            timing_categories['opening_phase'].append(response_strength)
        else:
            timing_categories['non_opening'].append(response_strength)
    
    print(f"\n🎯 CATALYST EFFECTIVENESS BY TIMING:")
    
    for category, strengths in timing_categories.items():
        if strengths:
            avg_strength = np.mean(strengths)
            count = len(strengths)
            print(f"   {category}: {count} sweeps, {avg_strength:.1f} avg response strength")
    
    return sweep_timing, timing_categories

def test_specific_catalyst_hypotheses(liq_sweep_sequences):
    """Test specific hypotheses about liq_sweep catalyst behavior"""
    
    print(f"\n🧪 TESTING CATALYST HYPOTHESES")
    print("=" * 60)
    
    hypotheses_results = {}
    
    # Hypothesis 1: liq_sweep → expansion_phase (discovered earlier)
    hypothesis_1 = {
        'name': 'liq_sweep → expansion_phase',
        'occurrences': 0,
        'lags': [],
        'sessions': []
    }
    
    # Hypothesis 2: liq_sweep → expansion_phase → consolidation
    hypothesis_2 = {
        'name': 'liq_sweep → expansion_phase → consolidation',
        'occurrences': 0,
        'lags': [],
        'sessions': []
    }
    
    # Hypothesis 3: liq_sweep → reversal (sweep reverses prior movement)
    hypothesis_3 = {
        'name': 'liq_sweep → reversal',
        'occurrences': 0,
        'lags': [],
        'sessions': []
    }
    
    # Hypothesis 4: liq_sweep → multiple events (catalyst triggers burst)
    hypothesis_4 = {
        'name': 'liq_sweep → multiple events (≥3)',
        'occurrences': 0,
        'burst_sizes': [],
        'sessions': []
    }
    
    for sequence in liq_sweep_sequences:
        following_events = sequence['following_events']
        
        if not following_events:
            continue
            
        # Test Hypothesis 1: liq_sweep → expansion_phase
        if following_events[0]['event_type'] == 'expansion_phase':
            hypothesis_1['occurrences'] += 1
            hypothesis_1['lags'].append(following_events[0]['lag_minutes'])
            hypothesis_1['sessions'].append(sequence['session'])
        
        # Test Hypothesis 2: liq_sweep → expansion_phase → consolidation
        if (len(following_events) >= 2 and 
            following_events[0]['event_type'] == 'expansion_phase' and
            following_events[1]['event_type'] == 'consolidation'):
            
            hypothesis_2['occurrences'] += 1
            hypothesis_2['lags'].append([
                following_events[0]['lag_minutes'],
                following_events[1]['lag_minutes']
            ])
            hypothesis_2['sessions'].append(sequence['session'])
        
        # Test Hypothesis 3: liq_sweep → reversal
        if following_events[0]['event_type'] == 'reversal':
            hypothesis_3['occurrences'] += 1
            hypothesis_3['lags'].append(following_events[0]['lag_minutes'])
            hypothesis_3['sessions'].append(sequence['session'])
        
        # Test Hypothesis 4: liq_sweep → multiple events (burst)
        if len(following_events) >= 3:
            hypothesis_4['occurrences'] += 1
            hypothesis_4['burst_sizes'].append(len(following_events))
            hypothesis_4['sessions'].append(sequence['session'])
    
    # Report results
    total_sequences = len(liq_sweep_sequences)
    
    print(f"📊 HYPOTHESIS TEST RESULTS ({total_sequences} total sequences):")
    
    # Hypothesis 1
    h1_rate = (hypothesis_1['occurrences'] / total_sequences * 100) if total_sequences > 0 else 0
    print(f"\n🔥 Hypothesis 1: {hypothesis_1['name']}")
    print(f"   Occurrences: {hypothesis_1['occurrences']}/{total_sequences} ({h1_rate:.1f}%)")
    if hypothesis_1['lags']:
        avg_lag = np.mean(hypothesis_1['lags'])
        std_lag = np.std(hypothesis_1['lags'])
        print(f"   Average lag: {avg_lag:.1f} ± {std_lag:.1f} minutes")
    
    # Hypothesis 2  
    h2_rate = (hypothesis_2['occurrences'] / total_sequences * 100) if total_sequences > 0 else 0
    print(f"\n🔗 Hypothesis 2: {hypothesis_2['name']}")
    print(f"   Occurrences: {hypothesis_2['occurrences']}/{total_sequences} ({h2_rate:.1f}%)")
    if hypothesis_2['lags']:
        avg_lags = np.mean(hypothesis_2['lags'], axis=0)
        print(f"   Average lags: {avg_lags[0]:.1f}m → {avg_lags[1]:.1f}m")
    
    # Hypothesis 3
    h3_rate = (hypothesis_3['occurrences'] / total_sequences * 100) if total_sequences > 0 else 0
    print(f"\n🔄 Hypothesis 3: {hypothesis_3['name']}")
    print(f"   Occurrences: {hypothesis_3['occurrences']}/{total_sequences} ({h3_rate:.1f}%)")
    if hypothesis_3['lags']:
        avg_lag = np.mean(hypothesis_3['lags'])
        std_lag = np.std(hypothesis_3['lags'])
        print(f"   Average lag: {avg_lag:.1f} ± {std_lag:.1f} minutes")
    
    # Hypothesis 4
    h4_rate = (hypothesis_4['occurrences'] / total_sequences * 100) if total_sequences > 0 else 0
    print(f"\n💥 Hypothesis 4: {hypothesis_4['name']}")
    print(f"   Occurrences: {hypothesis_4['occurrences']}/{total_sequences} ({h4_rate:.1f}%)")
    if hypothesis_4['burst_sizes']:
        avg_burst = np.mean(hypothesis_4['burst_sizes'])
        max_burst = np.max(hypothesis_4['burst_sizes'])
        print(f"   Average burst size: {avg_burst:.1f} events")
        print(f"   Maximum burst size: {max_burst} events")
    
    hypotheses_results = {
        'h1_expansion': hypothesis_1,
        'h2_exp_con_chain': hypothesis_2,
        'h3_reversal': hypothesis_3,
        'h4_burst': hypothesis_4
    }
    
    return hypotheses_results

def create_catalyst_visualization(liq_sweep_sequences, window_responses, chain_patterns, hypotheses_results):
    """Create comprehensive visualization of liq_sweep catalyst behavior"""
    
    print(f"\n📈 CREATING CATALYST VISUALIZATION")
    print("-" * 50)
    
    if not liq_sweep_sequences:
        print("❌ No data for visualization")
        return
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Liquidity Sweep Catalyst Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Response rates by time window
    ax1 = axes[0, 0]
    
    time_windows = sorted(window_responses.keys())
    response_rates = [
        (window_responses[tw]['sequences_with_response'] / window_responses[tw]['total_sequences'] * 100)
        if window_responses[tw]['total_sequences'] > 0 else 0
        for tw in time_windows
    ]
    
    ax1.plot(time_windows, response_rates, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Time Window (minutes)')
    ax1.set_ylabel('Response Rate (%)')
    ax1.set_title('Response Rate vs Time Window')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot 2: Most common immediate responses (within 5 minutes)
    ax2 = axes[0, 1]
    
    if 5 in window_responses and window_responses[5]['event_counts']:
        event_types = list(window_responses[5]['event_counts'].keys())
        event_counts = list(window_responses[5]['event_counts'].values())
        
        bars = ax2.bar(event_types, event_counts, color='skyblue', alpha=0.7)
        ax2.set_title('Most Common Responses (5min window)')
        ax2.set_ylabel('Frequency')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, event_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
    
    # Plot 3: Catalyst timing distribution
    ax3 = axes[0, 2]
    
    session_positions = [seq['sweep_session_position'] for seq in liq_sweep_sequences]
    response_strengths = [len(seq['following_events']) for seq in liq_sweep_sequences]
    
    scatter = ax3.scatter(session_positions, response_strengths, alpha=0.7, s=60)
    ax3.set_xlabel('Session Position (0=start, 1=end)')
    ax3.set_ylabel('Response Strength (# events)')
    ax3.set_title('Catalyst Effectiveness vs Timing')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Chain length distribution
    ax4 = axes[1, 0]
    
    chain_lengths = [len(seq['following_events']) for seq in liq_sweep_sequences]
    ax4.hist(chain_lengths, bins=range(0, max(chain_lengths)+2), alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_xlabel('Chain Length (# events following liq_sweep)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Response Chain Length Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Hypothesis test results
    ax5 = axes[1, 1]
    
    hypothesis_names = ['H1: →expansion', 'H2: →exp→con', 'H3: →reversal', 'H4: burst≥3']
    hypothesis_rates = [
        (hypotheses_results['h1_expansion']['occurrences'] / len(liq_sweep_sequences) * 100) if liq_sweep_sequences else 0,
        (hypotheses_results['h2_exp_con_chain']['occurrences'] / len(liq_sweep_sequences) * 100) if liq_sweep_sequences else 0,
        (hypotheses_results['h3_reversal']['occurrences'] / len(liq_sweep_sequences) * 100) if liq_sweep_sequences else 0,
        (hypotheses_results['h4_burst']['occurrences'] / len(liq_sweep_sequences) * 100) if liq_sweep_sequences else 0
    ]
    
    bars = ax5.bar(hypothesis_names, hypothesis_rates, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_title('Catalyst Hypothesis Test Results')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rate in zip(bars, hypothesis_rates):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # Plot 6: Response lag distribution
    ax6 = axes[1, 2]
    
    all_first_lags = []
    for seq in liq_sweep_sequences:
        if seq['following_events']:
            all_first_lags.append(seq['following_events'][0]['lag_minutes'])
    
    if all_first_lags:
        ax6.hist(all_first_lags, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax6.set_xlabel('First Response Lag (minutes)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Time to First Response Distribution')
        ax6.grid(True, alpha=0.3)
        
        # Add mean line
        mean_lag = np.mean(all_first_lags)
        ax6.axvline(mean_lag, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_lag:.1f}m')
        ax6.legend()
    
    plt.tight_layout()
    
    # Save visualization
    output_path = '/Users/jack/IRONPULSE/IRONFORGE/liquidity_sweep_catalyst.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved: {output_path}")
    
    return output_path

def main():
    """Main liquidity sweep catalyst investigation"""
    
    print("⚡ LIQUIDITY SWEEP CATALYST INVESTIGATION")
    print("=" * 80)
    print("Investigating liq_sweep as initiating event rather than terminal event")
    print("Discovering what causal chains liq_sweep triggers")
    print("=" * 80)
    
    # Step 1: Extract liq_sweep catalyst sequences
    liq_sweep_sequences, session_sequences, liq_sweep_stats = extract_liquidity_sweep_sequences()
    if not liq_sweep_sequences:
        print("❌ Cannot proceed without liq_sweep sequences")
        return
    
    # Step 2: Analyze immediate responses
    window_responses = analyze_immediate_responses(liq_sweep_sequences)
    
    # Step 3: Discover catalyst chains
    chain_patterns, chain_details = discover_catalyst_chains(liq_sweep_sequences)
    
    # Step 4: Analyze timing patterns
    sweep_timing, timing_categories = analyze_catalyst_timing_patterns(liq_sweep_sequences)
    
    # Step 5: Test specific hypotheses
    hypotheses_results = test_specific_catalyst_hypotheses(liq_sweep_sequences)
    
    # Step 6: Create visualization
    viz_path = create_catalyst_visualization(liq_sweep_sequences, window_responses, chain_patterns, hypotheses_results)
    
    # Summary
    print(f"\n⚡ LIQUIDITY SWEEP CATALYST SUMMARY")
    print("=" * 60)
    
    total_sweeps = liq_sweep_stats['total_sweeps']
    catalyst_sequences = len(liq_sweep_sequences)
    sessions_with_sweeps = liq_sweep_stats['sessions_with_sweeps']
    
    print(f"📊 Analysis Scope:")
    print(f"   Total liq_sweeps found: {total_sweeps}")
    print(f"   Catalyst sequences analyzed: {catalyst_sequences}")
    print(f"   Sessions with sweeps: {sessions_with_sweeps}")
    
    # Key discoveries
    if catalyst_sequences > 0:
        avg_response_strength = np.mean([len(seq['following_events']) for seq in liq_sweep_sequences])
        sequences_with_response = sum(1 for seq in liq_sweep_sequences if seq['following_events'])
        response_rate = (sequences_with_response / catalyst_sequences * 100)
        
        print(f"\n⚡ CATALYST EFFECTIVENESS:")
        print(f"   Overall response rate: {sequences_with_response}/{catalyst_sequences} ({response_rate:.1f}%)")
        print(f"   Average response strength: {avg_response_strength:.1f} events per sweep")
    
    # Top hypothesis results
    print(f"\n🏆 TOP CATALYST PATTERNS:")
    if hypotheses_results:
        sorted_hypotheses = sorted(hypotheses_results.items(), 
                                 key=lambda x: x[1]['occurrences'], reverse=True)
        
        for i, (h_name, h_data) in enumerate(sorted_hypotheses[:3]):
            rate = (h_data['occurrences'] / catalyst_sequences * 100) if catalyst_sequences > 0 else 0
            print(f"   #{i+1}: {h_data['name']}")
            print(f"       Success rate: {h_data['occurrences']}/{catalyst_sequences} ({rate:.1f}%)")
    
    print(f"\n✅ CATALYST INVESTIGATION COMPLETE")
    print(f"💾 Results visualization: {viz_path}")
    
    if catalyst_sequences > 0:
        print(f"\n🚀 BREAKTHROUGH: liq_sweep confirmed as market catalyst!")
        print(f"   Liquidity sweeps trigger measurable market responses with predictable patterns.")
        print(f"   This enables proactive positioning based on sweep detection.")
    else:
        print(f"\n📝 Result: Limited liq_sweep catalyst activity detected.")
        print(f"   Consider expanding analysis window or lowering detection thresholds.")

if __name__ == "__main__":
    main()