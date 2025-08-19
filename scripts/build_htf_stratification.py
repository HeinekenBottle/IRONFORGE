#!/usr/bin/env python3
"""
AUX: HTF-Phase Stratification
Find where patterns pay - phase, not seconds.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def stratify_by_htf_phase(trajectories, market_data):
    """Stratify zones by HTF phase characteristics."""
    
    # Merge trajectories with HTF features
    htf_data = market_data[['node_id', 'f47', 'f48', 'f49', 'f50']].copy()
    
    # Merge on center_node_id
    analysis_data = trajectories.merge(
        htf_data, 
        left_on='center_node_id', 
        right_on='node_id', 
        how='inner'
    )
    
    print(f"Merged {len(analysis_data)} zones with HTF features")
    
    # Create stratification buckets
    buckets = {}
    
    # 1. f50_htf_regime buckets (direct integer regimes)
    regime_buckets = analysis_data.groupby('f50')
    for regime, group in regime_buckets:
        bucket_name = f"regime_{int(regime)}"
        buckets[bucket_name] = create_bucket_stats(group, bucket_name)
    
    # 2. f47/f48 quintiles (bar position)
    for feature in ['f47', 'f48']:
        quintiles = pd.qcut(analysis_data[feature], q=5, labels=False, duplicates='drop')
        analysis_data[f'{feature}_quintile'] = quintiles
        
        quintile_buckets = analysis_data.groupby(f'{feature}_quintile')
        for quintile, group in quintile_buckets:
            if pd.notna(quintile):
                bucket_name = f"{feature}_q{int(quintile)}"
                buckets[bucket_name] = create_bucket_stats(group, bucket_name)
    
    # 3. f49 bins (distance to daily mid)
    bins = pd.cut(analysis_data['f49'], bins=5, labels=False)
    analysis_data['f49_bin'] = bins
    
    bin_buckets = analysis_data.groupby('f49_bin')
    for bin_idx, group in bin_buckets:
        if pd.notna(bin_idx):
            bucket_name = f"f49_bin_{int(bin_idx)}"
            buckets[bucket_name] = create_bucket_stats(group, bucket_name)
    
    return buckets

def create_bucket_stats(group, bucket_name):
    """Create statistics for a bucket of zones."""
    
    stats = {
        'bucket_name': bucket_name,
        'count': len(group),
        'zones': group['zone_id'].tolist()
    }
    
    # Forward return metrics (if available)
    for period in ['3b', '12b', '24b']:
        col = f'fwd_ret_{period}'
        if col in group.columns:
            valid_data = group[col].dropna()
            if len(valid_data) > 0:
                stats[f'mean_fwd_ret_{period}'] = float(valid_data.mean())
                stats[f'median_fwd_ret_{period}'] = float(valid_data.median())
                stats[f'std_fwd_ret_{period}'] = float(valid_data.std())
    
    # Hit probability metrics
    hit_targets = [50, 100, 200]
    for target in hit_targets:
        hit_col = f'hit_+{target}_12b'
        if hit_col in group.columns:
            stats[f'P_hit_+{target}_12b'] = float(group[hit_col].mean())
            
        time_col = f'time_to_+{target}_bars'
        if time_col in group.columns:
            valid_times = group[time_col].dropna()
            if len(valid_times) > 0:
                stats[f'median_time_to_+{target}_bars'] = float(valid_times.median())
    
    # HTF feature averages (for context)
    htf_features = ['f47', 'f48', 'f49', 'f50']
    for feature in htf_features:
        if feature in group.columns:
            stats[f'avg_{feature}'] = float(group[feature].mean())
    
    return stats

def main():
    run_path = Path("runs/2025-08-19")
    
    # Load trajectories
    trajectories_path = run_path / "aux" / "trajectories.parquet"
    if not trajectories_path.exists():
        print("Error: trajectories.parquet not found. Run build_trajectories.py first.")
        return False
        
    trajectories = pd.read_parquet(trajectories_path)
    print(f"Loaded {len(trajectories)} trajectory records")
    
    # Load market data with HTF features
    market_data_path = Path("data/shards/NQ_M5/shard_ASIA_2025-08-05/nodes.parquet")
    if not market_data_path.exists():
        print(f"Error: Market data not found at {market_data_path}")
        return False
        
    market_data = pd.read_parquet(market_data_path)
    print(f"Loaded {len(market_data)} market records")
    
    # Build HTF stratification
    buckets = stratify_by_htf_phase(trajectories, market_data)
    
    # Filter buckets with reasonable counts (>=2 zones)
    filtered_buckets = {k: v for k, v in buckets.items() if v['count'] >= 2}
    
    print(f"Created {len(filtered_buckets)} HTF phase buckets")
    
    # Save phase stats
    aux_path = run_path / "aux"
    aux_path.mkdir(exist_ok=True)
    
    phase_stats_path = aux_path / "phase_stats.json"
    with open(phase_stats_path, 'w') as f:
        json.dump(filtered_buckets, f, indent=2)
    
    print(f"Saved HTF phase stats to {phase_stats_path}")
    
    # Print summary
    print(f"\n=== HTF Phase Stratification Summary ===")
    print(f"Total buckets: {len(filtered_buckets)}")
    
    # Show top buckets by hit probability
    bucket_list = list(filtered_buckets.values())
    
    # Sort by P(hit_+100_12b) if available
    hit_100_buckets = [b for b in bucket_list if 'P_hit_+100_12b' in b]
    if hit_100_buckets:
        hit_100_buckets.sort(key=lambda x: x['P_hit_+100_12b'], reverse=True)
        print(f"\nTop buckets by P(hit_+100_12b):")
        for bucket in hit_100_buckets[:5]:
            print(f"  {bucket['bucket_name']}: {bucket['P_hit_+100_12b']:.2f} (n={bucket['count']})")
    
    # Show buckets with best median returns
    ret_12b_buckets = [b for b in bucket_list if 'median_fwd_ret_12b' in b]
    if ret_12b_buckets:
        ret_12b_buckets.sort(key=lambda x: abs(x['median_fwd_ret_12b']), reverse=True)
        print(f"\nTop buckets by |median_fwd_ret_12b|:")
        for bucket in ret_12b_buckets[:5]:
            print(f"  {bucket['bucket_name']}: {bucket['median_fwd_ret_12b']:.3f}% (n={bucket['count']})")
    
    # Create minidash ready signal
    print(f"\n✅ HTF strat ready - {len(filtered_buckets)} buckets with counts & metrics")
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)