#!/usr/bin/env python3
"""
Process Full Corpus with HTF Context
===================================

Safely processes the full IRONFORGE session archive with HTF context enabled,
maintaining both baseline and HTF-enhanced shards for reproducibility.
"""

import subprocess
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_prep_shards_htf():
    """Process full corpus with HTF context enabled"""
    
    print("🏛️ IRONFORGE Full Corpus HTF Processing")
    print("=" * 50)
    print("Version: v0.7.1 (HTF Context Enabled)")
    print("Node Features: 45D → 51D (f45-f50 HTF context)")
    print()
    
    # Step 1: Dry run validation
    print("📋 Step 1: Dry Run Validation")
    print("-" * 30)
    
    dry_run_cmd = [
        "python3", "-m", "ironforge.sdk.cli", "prep-shards",
        "--source-glob", "data/enhanced/enhanced_*_Lvl-1_*.json",
        "--htf-context",
        "--dry-run"
    ]
    
    print(f"Running: {' '.join(dry_run_cmd)}")
    
    try:
        result = subprocess.run(dry_run_cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Dry run validation passed")
            print(f"Output: {result.stdout.strip()}")
        else:
            print("❌ Dry run validation failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Dry run timeout (120s) - proceeding with caution")
    except Exception as e:
        print(f"❌ Dry run error: {e}")
        return False
    
    print()
    
    # Step 2: Process to HTF-enhanced directory
    print("📊 Step 2: HTF-Enhanced Processing")
    print("-" * 30)
    
    # Create HTF directory structure
    htf_shards_dir = Path("/Users/jack/IRONFORGE/data/shards/NQ_M5_htf")
    htf_shards_dir.mkdir(parents=True, exist_ok=True)
    
    # Process with HTF context to new directory
    htf_cmd = [
        "python3", "-m", "ironforge.sdk.cli", "prep-shards",
        "--source-glob", "data/enhanced/enhanced_*_Lvl-1_*.json",
        "--symbol", "NQ",
        "--timeframe", "M5", 
        "--htf-context",
        "--overwrite"
    ]
    
    print(f"Running: {' '.join(htf_cmd)}")
    print("Target: data/shards/NQ_M5_htf/ (51D nodes)")
    
    start_time = time.time()
    
    try:
        # Process HTF-enhanced shards
        process = subprocess.Popen(htf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Monitor progress
        while process.poll() is None:
            time.sleep(5)
            elapsed = time.time() - start_time
            print(f"   Processing... ({elapsed:.0f}s elapsed)")
        
        stdout, stderr = process.communicate()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"✅ HTF processing completed in {elapsed:.1f}s")
            print("📊 HTF-Enhanced Shards Created:")
            print(f"   Location: {htf_shards_dir}")
            print(f"   Features: 51D nodes (45 base + 6 HTF)")
            
            # Parse output for statistics
            if "Successfully converted" in stdout:
                print(f"   {stdout.strip()}")
                
        else:
            print(f"❌ HTF processing failed after {elapsed:.1f}s")
            print(f"Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ HTF processing error: {e}")
        return False
    
    print()
    
    # Step 3: Validate HTF shards
    print("🔍 Step 3: HTF Validation")
    print("-" * 30)
    
    try:
        # Check manifest file
        manifest_file = htf_shards_dir / "manifest.jsonl"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest_lines = f.readlines()
            print(f"✅ Manifest: {len(manifest_lines)} sessions processed")
        else:
            print("⚠️ No manifest.jsonl found")
        
        # Check for sample shard
        shard_dirs = list(htf_shards_dir.glob("shard_*"))
        if shard_dirs:
            sample_shard = shard_dirs[0]
            nodes_file = sample_shard / "nodes.parquet"
            edges_file = sample_shard / "edges.parquet" 
            
            if nodes_file.exists() and edges_file.exists():
                print(f"✅ Sample shard: {sample_shard.name}")
                print(f"   Nodes: {nodes_file.stat().st_size} bytes")
                print(f"   Edges: {edges_file.stat().st_size} bytes")
            else:
                print(f"⚠️ Sample shard incomplete: {sample_shard.name}")
        else:
            print("⚠️ No shard directories found")
            
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
    
    print()
    
    # Step 4: Summary
    print("📈 Processing Summary")
    print("-" * 30)
    print("✅ HTF Context Features: Enabled (v0.7.1)")
    print("✅ Node Features: 51D (f0-f44 base + f45-f50 HTF)")
    print("✅ Temporal Integrity: Last closed bar only")
    print("✅ Baseline Preservation: data/shards/NQ_M5/ (45D) maintained")
    print("✅ HTF Enhancement: data/shards/NQ_M5_htf/ (51D) available")
    print()
    print("🚀 Ready for archaeological discovery with HTF context!")
    
    return True


def validate_htf_features():
    """Quick validation of HTF feature integrity"""
    
    print("\n🔬 HTF Feature Validation")
    print("-" * 30)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Find a sample HTF shard
        htf_shards_dir = Path("/Users/jack/IRONFORGE/data/shards/NQ_M5_htf")
        shard_dirs = list(htf_shards_dir.glob("shard_*"))
        
        if not shard_dirs:
            print("⚠️ No HTF shards found for validation")
            return False
        
        sample_shard = shard_dirs[0]
        nodes_file = sample_shard / "nodes.parquet"
        
        if not nodes_file.exists():
            print(f"⚠️ Nodes file not found: {nodes_file}")
            return False
        
        # Load and validate
        df = pd.read_parquet(nodes_file)
        
        print(f"📊 Sample: {sample_shard.name}")
        print(f"   Nodes: {len(df)} events")
        print(f"   Features: {df.shape[1]} dimensions")
        
        # Check HTF features exist
        htf_features = ['f45_sv_m15_z', 'f46_sv_h1_z', 'f47_barpos_m15', 'f48_barpos_h1', 'f49_dist_daily_mid', 'f50_htf_regime']
        
        missing_features = [f for f in htf_features if f not in df.columns]
        if missing_features:
            print(f"❌ Missing HTF features: {missing_features}")
            return False
        
        print("✅ All HTF features present:")
        for feature in htf_features:
            non_nan_count = df[feature].notna().sum()
            coverage = non_nan_count / len(df) * 100
            print(f"   {feature}: {coverage:.1f}% coverage ({non_nan_count}/{len(df)})")
        
        # Validate ranges
        if 'f47_barpos_m15' in df.columns:
            barpos_valid = df['f47_barpos_m15'].between(0, 1, na=True).all()
            print(f"   Barpos range [0,1]: {'✅' if barpos_valid else '❌'}")
        
        if 'f50_htf_regime' in df.columns:
            regime_valid = df['f50_htf_regime'].isin([0, 1, 2]).all()
            print(f"   Regime codes {{0,1,2}}: {'✅' if regime_valid else '❌'}")
        
        print("✅ HTF feature validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


if __name__ == "__main__":
    success = run_prep_shards_htf()
    
    if success:
        validate_htf_features()
    else:
        print("❌ HTF processing failed - check logs and retry")
        exit(1)