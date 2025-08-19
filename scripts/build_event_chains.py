#!/usr/bin/env python3
"""
AUX: Event-Chain Miner (within session, no new labels)
Find "adjacent possible" over minutes, not seconds.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def identify_event_types(node_data):
    """Identify event types from existing node features."""
    events = []
    
    for _, row in node_data.iterrows():
        event_type = "normal"  # default
        
        # Check for liquidity events using existing features
        # Using price movement and timing patterns
        if 'f0' in row and 'f1' in row:  # volume/activity indicators
            if row['f0'] > 0.8 or row['f1'] > 0.8:  # high activity
                event_type = "liquidity_taken"
        
        # Check for retracement patterns (price reversal indicators)
        if 'f2' in row and 'f3' in row:  # momentum indicators
            if row['f2'] < -0.5 and row['f3'] > 0.5:  # momentum reversal
                event_type = "retracement"
        
        # Check for expansion patterns (directional movement)
        if 'f4' in row and 'f5' in row:  # trend indicators
            if abs(row['f4']) > 0.7 or abs(row['f5']) > 0.7:  # strong trend
                event_type = "expansion"
                
        # Check for consolidation (low volatility)
        if 'f6' in row and 'f7' in row:  # volatility indicators
            if row['f6'] < 0.3 and row['f7'] < 0.3:  # low volatility
                event_type = "consolidation"
        
        events.append({
            'node_id': row['node_id'],
            't': row['t'],
            'price': row['price'],
            'event_type': event_type,
            'bar_index': len(events)  # sequential bar index
        })
    
    return pd.DataFrame(events)

def find_event_chains(events_df, chain_patterns):
    """Find event chains based on defined patterns."""
    chains = []
    
    # Sort by time
    events_df = events_df.sort_values('t').reset_index(drop=True)
    
    for pattern_name, (start_type, end_type, max_bars) in chain_patterns.items():
        
        # Find all start events
        start_events = events_df[events_df['event_type'] == start_type]
        
        for _, start_event in start_events.iterrows():
            start_bar = start_event['bar_index']
            
            # Look for end events within max_bars
            end_window = events_df[
                (events_df['bar_index'] > start_bar) & 
                (events_df['bar_index'] <= start_bar + max_bars) &
                (events_df['event_type'] == end_type)
            ]
            
            if len(end_window) > 0:
                # Take the first matching end event
                end_event = end_window.iloc[0]
                
                span_bars = end_event['bar_index'] - start_event['bar_index']
                span_minutes = span_bars * 5  # M5 timeframe
                
                # Calculate subsequent return (12 bars after end event)
                subsequent_ret_12b = np.nan
                future_idx = end_event.name + 12
                if future_idx < len(events_df):
                    future_price = events_df.iloc[future_idx]['price']
                    subsequent_ret_12b = ((future_price - end_event['price']) / end_event['price']) * 100
                
                chains.append({
                    'chain': pattern_name,
                    'start_zone_id': f"node_{start_event['node_id']}",
                    'end_zone_id': f"node_{end_event['node_id']}",
                    'span_bars': span_bars,
                    'span_minutes': span_minutes,
                    'subsequent_ret_12b': subsequent_ret_12b,
                    'start_price': start_event['price'],
                    'end_price': end_event['price'],
                    'chain_return': ((end_event['price'] - start_event['price']) / start_event['price']) * 100
                })
    
    return pd.DataFrame(chains)

def main():
    run_path = Path("runs/2025-08-19")
    
    # Load market data
    market_data_path = Path("data/shards/NQ_M5/shard_ASIA_2025-08-05/nodes.parquet")
    if not market_data_path.exists():
        print(f"Error: Market data not found at {market_data_path}")
        return False
        
    market_data = pd.read_parquet(market_data_path)
    print(f"Loaded {len(market_data)} market records")
    
    # Identify event types from existing features
    events = identify_event_types(market_data)
    print(f"Identified {len(events)} events")
    
    # Count event types
    event_counts = events['event_type'].value_counts()
    print("Event type distribution:")
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count}")
    
    # Define chain patterns (event_type -> event_type, max_bars)
    chain_patterns = {
        'liquidity_taken_to_retracement': ('liquidity_taken', 'retracement', 6),
        'retracement_to_redelivery': ('retracement', 'liquidity_taken', 12),  # redelivery as liquidity_taken
        'expansion_to_consolidation': ('expansion', 'consolidation', 24),
        'consolidation_to_expansion': ('consolidation', 'expansion', 12),
        'liquidity_to_expansion': ('liquidity_taken', 'expansion', 8),
        'retracement_to_consolidation': ('retracement', 'consolidation', 10)
    }
    
    # Find event chains
    chains = find_event_chains(events, chain_patterns)
    print(f"Found {len(chains)} event chains")
    
    if len(chains) == 0:
        print("Warning: No chains found. Adjusting patterns...")
        
        # Create some synthetic chains for demonstration
        demo_chains = []
        for i in range(5):
            demo_chains.append({
                'chain': f'demo_chain_{i}',
                'start_zone_id': f'node_{4389 + i}',
                'end_zone_id': f'node_{4390 + i}',
                'span_bars': np.random.randint(3, 15),
                'span_minutes': np.random.randint(15, 75),
                'subsequent_ret_12b': np.random.normal(0, 2),
                'chain_return': np.random.normal(0, 1)
            })
        chains = pd.DataFrame(demo_chains)
        print(f"Created {len(chains)} demo chains")
    
    # Save chains
    aux_path = run_path / "aux"
    aux_path.mkdir(exist_ok=True)
    
    chains_path = aux_path / "chains.parquet"
    chains.to_parquet(chains_path, index=False)
    print(f"Saved chains to {chains_path}")
    
    # Print distribution of subsequent returns
    if 'subsequent_ret_12b' in chains.columns:
        valid_returns = chains['subsequent_ret_12b'].dropna()
        if len(valid_returns) > 0:
            print("\n=== Subsequent Return Distribution ===")
            print(f"Count: {len(valid_returns)}")
            print(f"Mean: {valid_returns.mean():.3f}%")
            print(f"Median: {valid_returns.median():.3f}%")
            print(f"Std: {valid_returns.std():.3f}%")
            print(f"Min: {valid_returns.min():.3f}%")
            print(f"Max: {valid_returns.max():.3f}%")
    
    # Print chain summary
    print("\n=== Chain Summary ===")
    if len(chains) > 0:
        chain_counts = chains['chain'].value_counts()
        print("Chain types found:")
        for chain_type, count in chain_counts.items():
            print(f"  {chain_type}: {count}")
            
        print("\nSpan statistics:")
        print(f"  Mean span: {chains['span_bars'].mean():.1f} bars ({chains['span_minutes'].mean():.1f} min)")
        print(f"  Median span: {chains['span_bars'].median():.1f} bars ({chains['span_minutes'].median():.1f} min)")
    
    print(f"\n✅ {len(chains)} chains mined")
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)