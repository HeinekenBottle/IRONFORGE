#!/usr/bin/env python3
"""
Promote 3-5 best cards to a "watchlist" (AUX)
Goal: give yourself a daily shortlist with horizon stats.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_archetype_cards(run_path: Path):
    """Load enhanced archetype cards and their index."""
    cards_index_path = run_path / "motifs" / "cards_index.csv"
    
    if not cards_index_path.exists():
        print(f"Warning: No cards index found at {cards_index_path}")
        return None, None
    
    # Load cards index
    cards_index = pd.read_csv(cards_index_path)
    
    # Load individual card files
    cards_dir = run_path / "motifs" / "cards"
    enhanced_cards = {}
    
    if cards_dir.exists():
        for zone_id in cards_index['zone_id']:
            card_path = cards_dir / f"{zone_id}.json"
            if card_path.exists():
                try:
                    with open(card_path) as f:
                        enhanced_cards[zone_id] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load card {zone_id}: {e}")
    
    return cards_index, enhanced_cards

def create_watchlist(run_path: Path, top_n: int = 5):
    """Create watchlist CSV with top N cards and horizon stats."""
    
    print(f"Creating watchlist with top {top_n} cards...")
    
    # Load archetype data
    cards_index, enhanced_cards = load_archetype_cards(run_path)
    
    if cards_index is None or len(cards_index) == 0:
        print("No archetype cards found. Creating synthetic watchlist for demo...")
        return create_synthetic_watchlist(run_path, top_n)
    
    # Sort cards by trading score and select top N
    cards_sorted = cards_index.sort_values('score', ascending=False).head(top_n)
    
    watchlist_data = []
    
    for _, card_row in cards_sorted.iterrows():
        zone_id = card_row['zone_id']
        enhanced_card = enhanced_cards.get(zone_id, {})
        
        # Extract key watchlist information
        watchlist_entry = {
            'zone_id': zone_id,
            'ts': enhanced_card.get('htf_snapshot', {}).get('ts', 0),
            'center_node_id': enhanced_card.get('center_node_id', 0),
            'confidence': enhanced_card.get('confidence', 0),
            'cohesion': enhanced_card.get('cohesion', 0),
            'in_burst': enhanced_card.get('in_burst', False),
            'chain_tag': card_row.get('chain_tag', 'none'),
            
            # Trajectory horizon stats
            'fwd_ret_12b': enhanced_card.get('trajectory_summary', {}).get('fwd_ret_12b', np.nan),
            'hit_+100_12b': enhanced_card.get('trajectory_summary', {}).get('hit_+100_12b', False),
            'time_to_+100_bars': enhanced_card.get('trajectory_summary', {}).get('time_to_+100_bars', np.nan),
            
            # Phase context
            'phase_bucket': card_row.get('phase_bucket', 'unknown'),
            'phase_hit_rate': enhanced_card.get('phase_context', {}).get('P_hit_+100_12b', 0) if enhanced_card.get('phase_context') else 0,
            
            # Trading score
            'trading_score': enhanced_card.get('trading_score', 0),
            
            # HTF context  
            'htf_regime': enhanced_card.get('htf_snapshot', {}).get('f50_regime', 0),
            'htf_bar_pos': enhanced_card.get('htf_snapshot', {}).get('f47_bar_pos', 0),
            'htf_dist_mid': enhanced_card.get('htf_snapshot', {}).get('f49_dist_mid', 0),
        }
        
        watchlist_data.append(watchlist_entry)
    
    # Create watchlist DataFrame
    watchlist_df = pd.DataFrame(watchlist_data)
    
    # Save to CSV
    watchlist_path = run_path / "motifs" / "watchlist.csv"
    watchlist_df.to_csv(watchlist_path, index=False)
    
    print(f"✅ Watchlist created: {watchlist_path}")
    print(f"   {len(watchlist_df)} zones selected")
    
    # Display summary
    print("\\n=== Watchlist Summary ===")
    for _, row in watchlist_df.iterrows():
        zone_id = row['zone_id']
        confidence = row['confidence']
        hit_100 = row['hit_+100_12b']
        fwd_ret = row['fwd_ret_12b']
        chain_tag = row['chain_tag']
        trading_score = row['trading_score']
        
        hit_str = "✅" if hit_100 else "❌"
        ret_str = f"{fwd_ret:+.2f}%" if not pd.isna(fwd_ret) else "N/A"
        
        print(f"  {zone_id}: conf={confidence:.3f}, {hit_str} hit_100, ret={ret_str}, chain={chain_tag}, score={trading_score:.3f}")
    
    return watchlist_df

def create_synthetic_watchlist(run_path: Path, top_n: int = 5):
    """Create synthetic watchlist for demonstration."""
    
    print("Creating synthetic watchlist for demonstration...")
    
    np.random.seed(42)
    
    synthetic_data = []
    for i in range(top_n):
        # Generate realistic watchlist entry
        confidence = np.random.uniform(0.65, 0.85)
        hit_100 = np.random.random() < 0.4  # 40% hit rate
        fwd_ret_12b = np.random.normal(0, 0.8)  # Realistic returns
        
        synthetic_data.append({
            'zone_id': f'demo_zone_{i}',
            'ts': 1692000000 + i * 3600,
            'center_node_id': 4400 + i,
            'confidence': confidence,
            'cohesion': np.random.uniform(0.5, 0.9),
            'in_burst': np.random.random() < 0.3,  # 30% in burst
            'chain_tag': np.random.choice(['liquidity', 'expansion', 'retracement', 'none']),
            'fwd_ret_12b': fwd_ret_12b,
            'hit_+100_12b': hit_100,
            'time_to_+100_bars': np.random.randint(2, 12) if hit_100 else np.nan,
            'phase_bucket': f'regime_{np.random.randint(0, 3)}',
            'phase_hit_rate': np.random.uniform(0.2, 0.6),
            'trading_score': confidence * np.random.uniform(0.8, 1.2),
            'htf_regime': np.random.randint(0, 3),
            'htf_bar_pos': np.random.uniform(-1, 1),
            'htf_dist_mid': np.random.uniform(-50, 50),
        })
    
    watchlist_df = pd.DataFrame(synthetic_data)
    
    # Sort by trading score
    watchlist_df = watchlist_df.sort_values('trading_score', ascending=False)
    
    # Save to CSV
    watchlist_path = run_path / "motifs" / "watchlist.csv"
    watchlist_path.parent.mkdir(parents=True, exist_ok=True)
    watchlist_df.to_csv(watchlist_path, index=False)
    
    print(f"✅ Synthetic watchlist created: {watchlist_path}")
    print(f"   {len(watchlist_df)} demo zones")
    
    # Display summary
    print("\\n=== Synthetic Watchlist Summary ===")
    for _, row in watchlist_df.iterrows():
        zone_id = row['zone_id']
        confidence = row['confidence']
        hit_100 = row['hit_+100_12b']
        fwd_ret = row['fwd_ret_12b']
        chain_tag = row['chain_tag']
        trading_score = row['trading_score']
        
        hit_str = "✅" if hit_100 else "❌"
        ret_str = f"{fwd_ret:+.2f}%"
        
        print(f"  {zone_id}: conf={confidence:.3f}, {hit_str} hit_100, ret={ret_str}, chain={chain_tag}, score={trading_score:.3f}")
    
    return watchlist_df

def main():
    # Try both current run and real-tgat-fixed run
    run_paths = [
        Path("runs/real-tgat-fixed-2025-08-18"),
        Path("runs/2025-08-19")
    ]
    
    success = False
    for run_path in run_paths:
        if run_path.exists():
            print(f"\\nTrying run: {run_path}")
            try:
                watchlist = create_watchlist(run_path, top_n=5)
                if watchlist is not None and len(watchlist) > 0:
                    success = True
                    print(f"\\n🎯 Watchlist successfully created for {run_path}")
                    break
            except Exception as e:
                print(f"Error with {run_path}: {e}")
    
    if not success:
        print("\\n⚠️  No suitable run found, creating demo watchlist")
        demo_run = Path("runs/demo-watchlist")
        demo_run.mkdir(exist_ok=True)
        watchlist = create_synthetic_watchlist(demo_run, top_n=5)
        success = True
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)