#!/usr/bin/env python3
"""
Test enhanced confluence scoring with zone_nodes.parquet emission
"""

from ironforge.confluence.scoring import score_confluence
from pathlib import Path

def main():
    print("🧪 Testing Enhanced Confluence Scoring")
    
    # Set up test parameters
    pattern_paths = [
        "runs/2025-08-18/embeddings/patterns.parquet",
        "runs/2025-08-18/embeddings/embeddings.parquet"
    ]
    out_dir = "runs/2025-08-18/confluence"
    threshold = 0.7
    
    print(f"   Pattern paths: {len(pattern_paths)}")
    print(f"   Output directory: {out_dir}")
    print(f"   Threshold: {threshold}")
    
    # Run enhanced scoring
    result_path = score_confluence(pattern_paths, out_dir, None, threshold)
    print(f"   Scores written to: {result_path}")
    
    # Check if zone_nodes.parquet was created
    zone_nodes_path = Path(out_dir) / "zone_nodes.parquet"
    if zone_nodes_path.exists():
        import pandas as pd
        zone_nodes = pd.read_parquet(zone_nodes_path)
        print(f"   ✅ zone_nodes.parquet created: {len(zone_nodes)} entries")
        print(f"   Columns: {list(zone_nodes.columns)}")
        print(f"   Sample data:")
        print(zone_nodes.head(3).to_string(index=False))
    else:
        print("   ❌ zone_nodes.parquet not created")
    
    print("✅ Enhanced scoring test complete!")

if __name__ == "__main__":
    main()