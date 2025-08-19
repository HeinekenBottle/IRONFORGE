#!/usr/bin/env python3
"""
Simple Timing Pattern Discovery for IRONFORGE
Focus on timing first, before semantics and archetypes
"""

import json
from pathlib import Path

import pandas as pd


def analyze_session_timing(nodes_df, edges_df, output_dir=None):
    """Extract timing patterns from a session and save as auxiliary metadata"""
    
    # Sort by timestamp
    nodes = nodes_df.sort_values('t').copy()
    nodes['timestamp'] = pd.to_datetime(nodes['t'], unit='ms')
    
    # Basic session info
    session_id = nodes['session_id'].iloc[0]
    duration = (nodes['timestamp'].iloc[-1] - nodes['timestamp'].iloc[0]).total_seconds()
    
    # Time gaps between events
    time_gaps = nodes['t'].diff().dropna() / 1000  # seconds
    
    # Event type mapping
    event_types = {
        0: 'EXPANSION', 1: 'CONSOLIDATION', 2: 'RETRACEMENT', 
        3: 'REVERSAL', 4: 'LIQUIDITY_TAKEN', 5: 'REDELIVERY'
    }
    
    # Analyze timing patterns
    patterns = {
        'session_info': {
            'session_id': session_id,
            'total_events': len(nodes),
            'duration_minutes': duration / 60,
            'start_time': nodes['timestamp'].iloc[0].isoformat(),
            'end_time': nodes['timestamp'].iloc[-1].isoformat(),
        },
        
        'timing_stats': {
            'mean_gap_seconds': time_gaps.mean(),
            'median_gap_seconds': time_gaps.median(),
            'min_gap_seconds': time_gaps.min(),
            'max_gap_seconds': time_gaps.max(),
        },
        
        'event_frequency': {
            event_types.get(k, f'Type_{k}'): int(v) 
            for k, v in nodes['kind'].value_counts().sort_index().items()
        },
        
        'timing_behaviors': {
            'rapid_events': int(sum(time_gaps < 60)),  # < 1 minute
            'normal_events': int(sum((time_gaps >= 60) & (time_gaps <= 600))),  # 1-10 min
            'delayed_events': int(sum(time_gaps > 600)),  # > 10 minutes
            'immediate_events': int(sum(time_gaps == 0)),  # Simultaneous
        }
    }
    
    # Look for specific timing patterns
    patterns['timing_patterns'] = find_timing_patterns(nodes, time_gaps)
    
    # Export auxiliary metadata if output directory provided
    if output_dir:
        export_aux_metadata(nodes, time_gaps, patterns, output_dir)
    
    return patterns


def find_timing_patterns(nodes, time_gaps):
    """Find specific timing patterns in the sequence"""
    
    patterns = {}
    
    # Burst detection: 3+ events within 2 minutes
    bursts = []
    current_burst = []
    
    for i, gap in enumerate(time_gaps):
        if gap <= 120:  # 2 minutes
            if not current_burst:
                current_burst = [i, i+1]  # Start burst
            else:
                current_burst.append(i+1)
        else:
            if len(current_burst) >= 3:
                bursts.append(current_burst)
            current_burst = []
    
    if len(current_burst) >= 3:
        bursts.append(current_burst)
    
    patterns['bursts'] = {
        'count': len(bursts),
        'events_in_bursts': sum(len(burst) for burst in bursts),
        'burst_details': bursts[:5]  # First 5 for brevity
    }
    
    # Rhythm detection: consistent gaps
    gap_rounded = (time_gaps / 60).round()  # Round to nearest minute
    rhythm_counts = gap_rounded.value_counts()
    dominant_rhythm = rhythm_counts.index[0] if len(rhythm_counts) > 0 else None
    
    patterns['rhythm'] = {
        'dominant_gap_minutes': float(dominant_rhythm) if dominant_rhythm else None,
        'rhythm_strength': float(rhythm_counts.iloc[0] / len(time_gaps)) if len(rhythm_counts) > 0 else 0,
        'gap_distribution': rhythm_counts.head(5).to_dict()
    }
    
    # Event clustering by type
    event_transitions = []
    for i in range(len(nodes) - 1):
        from_type = nodes.iloc[i]['kind']
        to_type = nodes.iloc[i+1]['kind'] 
        gap = time_gaps.iloc[i]
        event_transitions.append((from_type, to_type, gap))
    
    # Most common transitions
    transition_counts = {}
    for from_t, to_t, gap in event_transitions:
        key = f"{from_t}→{to_t}"
        if key not in transition_counts:
            transition_counts[key] = {'count': 0, 'avg_gap': 0}
        transition_counts[key]['count'] += 1
        transition_counts[key]['avg_gap'] += gap
    
    for key in transition_counts:
        transition_counts[key]['avg_gap'] /= transition_counts[key]['count']
    
    patterns['transitions'] = dict(sorted(transition_counts.items(), 
                                        key=lambda x: x[1]['count'], reverse=True)[:5])
    
    return patterns


def export_aux_metadata(nodes, time_gaps, patterns, output_dir):
    """Export auxiliary timing metadata to run directory"""
    
    aux_dir = Path(output_dir) / "aux" / "timing"
    aux_dir.mkdir(parents=True, exist_ok=True)
    
    session_id = nodes['session_id'].iloc[0]
    
    # Extract burst information
    bursts = patterns['timing_patterns']['bursts']['burst_details']
    burst_summaries = []
    node_annotations = []
    
    # Process each burst
    for burst_id, burst_indices in enumerate(bursts):
        if len(burst_indices) < 2:
            continue
            
        burst_nodes = nodes.iloc[burst_indices]
        burst_gaps = time_gaps.iloc[burst_indices[:-1]] if len(burst_indices) > 1 else []
        
        # Burst summary
        burst_summaries.append({
            'session_id': session_id,
            'burst_id': burst_id,
            'burst_start_ts': int(burst_nodes.iloc[0]['t']),
            'burst_end_ts': int(burst_nodes.iloc[-1]['t']),
            'events_in_burst': len(burst_indices),
            'gap_s_mean': float(burst_gaps.mean()) if len(burst_gaps) > 0 else 0.0,
            'gap_s_p95': float(burst_gaps.quantile(0.95)) if len(burst_gaps) > 0 else 0.0
        })
        
        # Node annotations for this burst
        for rank, idx in enumerate(burst_indices):
            node_id = nodes.iloc[idx]['node_id']
            gap_prev = time_gaps.iloc[idx-1] if idx > 0 else 0.0
            
            node_annotations.append({
                'node_id': node_id,
                'burst_id': burst_id,
                'intra_burst_rank': rank,
                'gap_prev_s': float(gap_prev)
            })
    
    # Add non-burst nodes with burst_id = -1
    burst_node_indices = set()
    for burst_indices in bursts:
        burst_node_indices.update(burst_indices)
    
    for idx in range(len(nodes)):
        if idx not in burst_node_indices:
            node_id = nodes.iloc[idx]['node_id']
            gap_prev = time_gaps.iloc[idx-1] if idx > 0 else 0.0
            
            node_annotations.append({
                'node_id': node_id,
                'burst_id': -1,  # Non-burst
                'intra_burst_rank': -1,
                'gap_prev_s': float(gap_prev)
            })
    
    # Save to parquet
    if burst_summaries:
        burst_df = pd.DataFrame(burst_summaries)
        burst_df.to_parquet(aux_dir / "summary.parquet", index=False)
    
    if node_annotations:
        node_df = pd.DataFrame(node_annotations)
        node_df.to_parquet(aux_dir / "node_annotations.parquet", index=False)
    
    print(f"   AUX: Saved {len(burst_summaries)} bursts, {len(node_annotations)} node annotations")


def analyze_shard(shard_path, output_dir=None):
    """Analyze timing patterns for a single shard"""
    shard_path = Path(shard_path)
    
    nodes = pd.read_parquet(shard_path / "nodes.parquet")
    edges = pd.read_parquet(shard_path / "edges.parquet")
    
    return analyze_session_timing(nodes, edges, output_dir)


def main():
    """Run timing analysis on available shards"""
    
    shards_dir = Path("data/shards/NQ_M5")
    shard_dirs = [d for d in shards_dir.iterdir() if d.is_dir() and d.name.startswith("shard_")]
    
    print(f"🕐 Analyzing timing patterns across {len(shard_dirs)} shards...")
    
    all_results = {}
    
    for shard_dir in sorted(shard_dirs)[:5]:  # Analyze first 5 shards
        print(f"   Analyzing {shard_dir.name}...")
        try:
            patterns = analyze_shard(shard_dir)
            all_results[shard_dir.name] = patterns
            
            # Quick summary
            session = patterns['session_info']
            timing = patterns['timing_patterns']
            print(f"     {session['total_events']} events over {session['duration_minutes']:.1f}min, "
                  f"{timing['bursts']['count']} bursts, rhythm: {timing['rhythm']['dominant_gap_minutes']}min")
            
        except Exception as e:
            print(f"     Error: {e}")
    
    # Save results
    output_path = Path("timing_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Timing analysis complete! Results saved to {output_path}")
    
    # Summary statistics
    print("\n📊 SUMMARY ACROSS SESSIONS:")
    total_events = sum(r['session_info']['total_events'] for r in all_results.values())
    total_bursts = sum(r['timing_patterns']['bursts']['count'] for r in all_results.values())
    avg_duration = sum(r['session_info']['duration_minutes'] for r in all_results.values()) / len(all_results)
    
    print(f"   Total events analyzed: {total_events}")
    print(f"   Total burst patterns: {total_bursts}")
    print(f"   Average session duration: {avg_duration:.1f} minutes")
    
    return all_results


if __name__ == "__main__":
    main()