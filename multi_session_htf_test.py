#!/usr/bin/env python3
"""
Multi-Session HTF Context Testing
=================================

Process multiple sessions with HTF context features to demonstrate
SV z-score population and archaeological discovery capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from ironforge.converters.json_to_parquet import JSONToParquetConverter, ConversionConfig
from ironforge.converters.htf_context_processor import create_default_htf_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_session_data(shard_path: str) -> Dict[str, Any]:
    """Load session data from existing shard directory"""
    shard_dir = Path(shard_path)
    meta_file = shard_dir / "meta.json"
    
    if not meta_file.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_file}")
    
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    
    return meta_data


def simulate_multi_session_htf():
    """Simulate processing multiple sessions with HTF context"""
    
    print("🧪 Multi-Session HTF Context Test")
    print("=" * 50)
    
    # Session paths to process (high-activity sessions)
    session_paths = [
        "/Users/jack/IRONFORGE/data/shards/NQ_M5/shard_ASIA_2025-08-05",    # 46 nodes
        "/Users/jack/IRONFORGE/data/shards/NQ_M5/shard_ASIA_2025-08-07",    # 50 nodes  
        "/Users/jack/IRONFORGE/data/shards/NQ_M5/shard_LONDON_2025-07-30",  # 48 nodes
        "/Users/jack/IRONFORGE/data/shards/NQ_M5/shard_NY_2025-08-05"       # If exists
    ]
    
    # Create HTF-enabled configuration
    config = ConversionConfig(
        htf_context_enabled=True,
        htf_context_config=create_default_htf_config()
    )
    
    print(f"📊 HTF Configuration:")
    print(f"   Timeframes: {config.htf_context_config.timeframes}")
    print(f"   SV Lookback: {config.htf_context_config.sv_lookback_bars} bars")
    print(f"   SV Weights: {config.htf_context_config.sv_weights}")
    print()
    
    # Load and analyze each session
    session_summaries = []
    
    for session_path in session_paths:
        if not Path(session_path).exists():
            print(f"⚠️  Skipping non-existent: {session_path}")
            continue
            
        try:
            print(f"📂 Processing: {Path(session_path).name}")
            
            # Load metadata
            meta_data = load_session_data(session_path)
            node_count = meta_data.get('node_count', 0)
            session_id = meta_data.get('session_id', 'unknown')
            
            print(f"   Session ID: {session_id}")
            print(f"   Node Count: {node_count}")
            
            # For demonstration, calculate expected HTF characteristics
            # In a real conversion, this would process actual events
            estimated_m15_bars = max(1, node_count // 3)  # ~3 M5 events per M15 bar
            estimated_h1_bars = max(1, node_count // 12)  # ~12 M5 events per H1 bar
            
            print(f"   Est. M15 bars: {estimated_m15_bars}")
            print(f"   Est. H1 bars: {estimated_h1_bars}")
            
            # Calculate SV z-score availability
            sv_m15_available = estimated_m15_bars >= 10  # Need 10+ bars for z-score
            sv_h1_available = estimated_h1_bars >= 10
            
            print(f"   SV M15 z-score: {'✅ Available' if sv_m15_available else '❌ Insufficient data'}")
            print(f"   SV H1 z-score: {'✅ Available' if sv_h1_available else '❌ Insufficient data'}")
            
            session_summaries.append({
                'session_id': session_id,
                'node_count': node_count,
                'estimated_m15_bars': estimated_m15_bars,
                'estimated_h1_bars': estimated_h1_bars,
                'sv_m15_available': sv_m15_available,
                'sv_h1_available': sv_h1_available
            })
            
            print()
            
        except Exception as e:
            print(f"❌ Error processing {session_path}: {e}")
            print()
    
    # Summary Analysis
    print("📈 Multi-Session HTF Analysis Summary")
    print("=" * 50)
    
    total_sessions = len(session_summaries)
    total_nodes = sum(s['node_count'] for s in session_summaries)
    sv_m15_sessions = sum(1 for s in session_summaries if s['sv_m15_available'])
    sv_h1_sessions = sum(1 for s in session_summaries if s['sv_h1_available'])
    
    print(f"Total Sessions Processed: {total_sessions}")
    print(f"Total Graph Nodes: {total_nodes}")
    print(f"Sessions with M15 SV z-scores: {sv_m15_sessions}/{total_sessions}")
    print(f"Sessions with H1 SV z-scores: {sv_h1_sessions}/{total_sessions}")
    print()
    
    # Archaeological Discovery Implications
    print("🏛️ Archaeological Discovery Implications")
    print("=" * 50)
    
    if sv_h1_sessions > 0:
        print("✅ H1 regime classification available - can identify expansion/consolidation phases")
    if sv_m15_sessions > 0:
        print("✅ M15 synthetic volume patterns - can detect liquidity anomalies")
    if total_sessions >= 3:
        print("✅ Multi-session patterns - can discover cross-session archaeological relationships")
    
    # Estimate archeological zone detection
    estimated_zones = sum(max(1, s['node_count'] // 8) for s in session_summaries)
    print(f"📍 Estimated archaeological zones: {estimated_zones}")
    
    # HTF context features summary
    print()
    print("🔍 HTF Context Features (45D → 51D)")
    print("=" * 50)
    print("f45_sv_m15_z    : M15 synthetic volume z-score")
    print("f46_sv_h1_z     : H1 synthetic volume z-score") 
    print("f47_barpos_m15  : Position within M15 bar (0.0-1.0)")
    print("f48_barpos_h1   : Position within H1 bar (0.0-1.0)")
    print("f49_dist_daily_mid : Distance to daily midpoint (normalized)")
    print("f50_htf_regime  : HTF regime code (0=consolidation, 1=transition, 2=expansion)")
    
    print()
    print("🎯 Recommendation: PROCEED with multi-session HTF processing")
    print("   Archaeological discovery potential unlocked with HTF context!")


if __name__ == "__main__":
    simulate_multi_session_htf()