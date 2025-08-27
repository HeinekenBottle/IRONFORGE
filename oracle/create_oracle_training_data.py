#!/usr/bin/env python3
"""
Create realistic Oracle training data from existing shard data
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def create_training_sessions():
    """Convert shard data to enhanced session format for Oracle training"""
    
    shard_base = Path("data/shards/NQ_M5")
    output_dir = Path("data/oracle_training")
    output_dir.mkdir(exist_ok=True)
    
    # Find all NY session shards (these should have good event data)
    shard_dirs = [d for d in shard_base.iterdir() if d.is_dir() and "NY" in d.name][:5]
    
    print(f"Found {len(shard_dirs)} NY session shards")
    
    for shard_dir in shard_dirs:
        try:
            # Load shard metadata
            with open(shard_dir / "meta.json", 'r') as f:
                meta = json.load(f)
            
            # Load nodes (events)
            nodes_df = pd.read_parquet(shard_dir / "nodes.parquet")
            
            if len(nodes_df) < 10:
                print(f"Skipping {shard_dir.name} - only {len(nodes_df)} events")
                continue
            
            print(f"Processing {shard_dir.name} with {len(nodes_df)} events")
            
            # Create enhanced session format
            events = []
            prices = []
            
            for idx, node in nodes_df.iterrows():
                # Create 45D feature vector from individual f0-f44 columns
                feature_vector = [node[f'f{i}'] for i in range(45)]
                
                # Create event
                event = {
                    "index": len(events),
                    "price": float(node['price']),
                    "volume": 100,  # Mock volume
                    "timestamp": f"2025-07-29T{9 + len(events) // 10}:{(len(events) % 60):02d}:00",
                    "feature": feature_vector
                }
                events.append(event)
                prices.append(float(node['price']))
            
            # Calculate session OHLC
            session_high = max(prices)
            session_low = min(prices)
            
            # Create enhanced session
            session_data = {
                "session_name": f"oracle_training_{meta['session_id']}",
                "timestamp": "2025-07-29T09:30:00",
                "symbol": "NQ",
                "events": events,
                "metadata": {
                    "high": session_high,
                    "low": session_low,
                    "range": session_high - session_low,
                    "n_events": len(events),
                    "htf_context": False,
                    "original_shard": str(shard_dir)
                }
            }
            
            # Save enhanced session
            output_file = output_dir / f"enhanced_oracle_{meta['session_id']}.json"
            with open(output_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"Created {output_file.name} with {len(events)} events, range: {session_high - session_low:.1f}")
            
        except Exception as e:
            print(f"Error processing {shard_dir.name}: {e}")
            continue
    
    print("Training data creation completed!")

if __name__ == "__main__":
    create_training_sessions()