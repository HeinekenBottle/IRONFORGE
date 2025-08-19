#!/usr/bin/env python3
"""
Generate mock Oracle training pairs for testing the training pipeline
"""

import numpy as np
import pandas as pd
import pathlib
import json
from datetime import datetime, timedelta

def generate_mock_training_pairs(n_samples: int = 200, early_pct: float = 0.20):
    """Generate realistic mock training pairs for Oracle testing"""
    
    np.random.seed(7)  # Reproducible results
    
    rows = []
    base_date = datetime(2025, 7, 15)
    
    for i in range(n_samples):
        # Generate realistic NQ session parameters
        session_range = np.random.uniform(40, 180)  # Typical NQ session range
        half_range = session_range / 2
        
        # Center around realistic NQ levels (23000-24000)
        center_base = 23500 + np.random.uniform(-500, 500)
        center = center_base + np.random.uniform(-50, 50)
        
        # Generate 44D pooled embedding (realistic TGAT output)
        pooled = np.random.normal(0, 1, (44,))
        
        # Create session date
        session_date = (base_date + timedelta(days=i % 20)).strftime("%Y-%m-%d")
        
        # Build training row
        row = {
            "symbol": "NQ",
            "tf": 5,  # Numeric timeframe
            "session_date": session_date,
            "htf_mode": "45D",
            "early_pct": early_pct,
            "target_center": float(center),
            "target_half_range": float(half_range)
        }
        
        # Add 44D pooled embedding features
        for j in range(44):
            row[f"pooled_{j:02d}"] = float(pooled[j])
            
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    output_dir = pathlib.Path("data/oracle_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write training pairs
    output_file = output_dir / "training_pairs.parquet"
    df.to_parquet(output_file, index=False)
    
    # Write metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "n_samples": n_samples,
        "early_pct": early_pct,
        "features": {
            "embedding_dim": 44,
            "target_dim": 2,
            "symbols": ["NQ"],
            "timeframes": [5]
        },
        "data_quality": "mock_generated",
        "schema_version": "v1.0"
    }
    
    metadata_file = output_dir / "training_pairs_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Generated {n_samples} mock training pairs")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Metadata: {metadata_file}")
    print(f"🎯 Target range: {df['target_half_range'].describe()['mean']:.1f} ± {df['target_half_range'].std():.1f} points")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate mock Oracle training pairs")
    parser.add_argument("--samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--early-pct", type=float, default=0.20, help="Early batch percentage")
    
    args = parser.parse_args()
    
    generate_mock_training_pairs(args.samples, args.early_pct)