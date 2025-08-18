#!/usr/bin/env python3
"""
AUX: Post-Zone Trajectories (trade horizons)
Build trader-relevant trajectories from TGAT zone discoveries.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def calculate_forward_returns(df, price_col='price', periods=[3, 12, 24]):
    """Calculate forward returns over specified periods (bars)."""
    results = {}
    
    for period in periods:
        # Forward maximum excursion (absolute)
        fwd_max = df[price_col].rolling(window=period, min_periods=1).max().shift(-period)
        fwd_min = df[price_col].rolling(window=period, min_periods=1).min().shift(-period)
        
        current_price = df[price_col]
        
        # Calculate returns as percentage
        max_ret = ((fwd_max - current_price) / current_price * 100).round(4)
        min_ret = ((fwd_min - current_price) / current_price * 100).round(4)
        
        # Take the maximum excursion (absolute value)
        results[f'fwd_ret_{period}b'] = np.where(
            np.abs(max_ret) > np.abs(min_ret), max_ret, min_ret
        )
    
    return pd.DataFrame(results, index=df.index)

def calculate_hit_targets(df, price_col='price', targets=[50, 100, 200], period=12):
    """Calculate if price hits specific tick targets within period."""
    results = {}
    
    for target in targets:
        # Rolling window check for hits
        price_series = df[price_col]
        
        hit_plus = []
        hit_minus = []
        time_to_plus = []
        time_to_minus = []
        
        for i in range(len(df)):
            if i + period >= len(df):
                hit_plus.append(False)
                hit_minus.append(False)
                time_to_plus.append(np.nan)
                time_to_minus.append(np.nan)
                continue
                
            current_price = price_series.iloc[i]
            future_window = price_series.iloc[i+1:i+period+1]
            
            # Check for +target hit
            plus_hits = future_window >= (current_price + target * 0.25)  # 0.25 per tick for NQ
            if plus_hits.any():
                hit_plus.append(True)
                time_to_plus.append(plus_hits.idxmax() - i)
            else:
                hit_plus.append(False)
                time_to_plus.append(np.nan)
                
            # Check for -target hit  
            minus_hits = future_window <= (current_price - target * 0.25)
            if minus_hits.any():
                hit_minus.append(True)
                time_to_minus.append(minus_hits.idxmax() - i)
            else:
                hit_minus.append(False)
                time_to_minus.append(np.nan)
        
        results[f'hit_+{target}_{period}b'] = hit_plus
        results[f'hit_-{target}_{period}b'] = hit_minus
        results[f'time_to_+{target}_bars'] = time_to_plus
        results[f'time_to_-{target}_bars'] = time_to_minus
    
    return pd.DataFrame(results, index=df.index)

def build_trajectories(run_path):
    """Build post-zone trajectories for trader analysis."""
    print("Building post-zone trajectories...")
    
    # Load zone discoveries
    zone_nodes_path = run_path / "confluence" / "zone_nodes.parquet"
    zone_nodes = pd.read_parquet(zone_nodes_path)
    
    # Load source market data - detect correct session
    zone_node_ids = zone_nodes.node_id.tolist()
    
    # Try different sessions to find matching node IDs
    sessions = ["ASIA", "NY", "LONDON", "LUNCH", "MIDNIGHT"]
    source_data_path = None
    
    for session in sessions:
        test_path = Path(f"data/shards/NQ_M5/shard_{session}_2025-08-05/nodes.parquet")
        if test_path.exists():
            try:
                test_data = pd.read_parquet(test_path)
                if any(nid in test_data.node_id.values for nid in zone_node_ids):
                    source_data_path = test_path
                    print(f"Found zone nodes in session: {session}")
                    break
            except:
                continue
    
    if source_data_path is None:
        print("Error: Could not find source data containing zone nodes")
        return None
        
    nodes = pd.read_parquet(source_data_path)
    print(f"Loaded {len(nodes)} market nodes")
    
    # Filter to zone nodes
    zone_node_ids = zone_nodes.node_id.tolist()
    zone_data = nodes[nodes.node_id.isin(zone_node_ids)].copy()
    
    if len(zone_data) == 0:
        print("Warning: No zone nodes found in source data")
        return None
        
    print(f"Found {len(zone_data)} zone nodes in market data")
    
    # Sort by time for trajectory calculation
    zone_data = zone_data.sort_values('t').reset_index(drop=True)
    
    # Calculate forward returns
    fwd_returns = calculate_forward_returns(zone_data)
    
    # Calculate hit targets (50, 100, 200 ticks)
    hit_targets = calculate_hit_targets(zone_data, targets=[50, 100, 200])
    
    # Combine results
    trajectories = pd.concat([
        zone_data[['node_id', 't', 'price']].rename(columns={'t': 'ts', 'price': 'price_c'}),
        fwd_returns,
        hit_targets
    ], axis=1)
    
    # Add zone_id mapping
    zone_mapping = dict(zip(zone_nodes.node_id, zone_nodes.zone_id))
    trajectories['zone_id'] = trajectories['node_id'].map(zone_mapping)
    trajectories['center_node_id'] = trajectories['node_id']
    
    # Reorder columns
    cols = ['zone_id', 'center_node_id', 'ts', 'price_c'] + [c for c in trajectories.columns if c not in ['zone_id', 'center_node_id', 'ts', 'price_c', 'node_id']]
    trajectories = trajectories[cols]
    
    # Calculate population rate
    populated_cols = ['fwd_ret_3b', 'fwd_ret_12b', 'fwd_ret_24b']
    populated_rate = trajectories[populated_cols].notna().any(axis=1).mean()
    print(f"Population rate: {populated_rate:.1%}")
    
    return trajectories

def main():
    run_path = Path("runs/real-tgat-fixed-2025-08-18")
    
    # Build trajectories
    trajectories = build_trajectories(run_path)
    
    if trajectories is not None:
        # Create aux directory
        aux_path = run_path / "aux"
        aux_path.mkdir(exist_ok=True)
        
        # Save trajectories
        output_path = aux_path / "trajectories.parquet"
        trajectories.to_parquet(output_path, index=False)
        print(f"Saved trajectories to {output_path}")
        
        # Print summary
        print(f"\n=== Trajectory Summary ===")
        print(f"Zones analyzed: {len(trajectories)}")
        print(f"Forward return columns: {[c for c in trajectories.columns if 'fwd_ret' in c]}")
        print(f"Hit target columns: {[c for c in trajectories.columns if 'hit_' in c][:6]}...")
        
        print(f"\n=== Sample Trajectories ===")
        print(trajectories[['zone_id', 'center_node_id', 'fwd_ret_12b', 'hit_+100_12b']].head())
        
        return True
    else:
        print("Failed to build trajectories")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)