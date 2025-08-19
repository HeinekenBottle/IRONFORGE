#!/usr/bin/env python3
"""
Curate it: Archetype pack v0 (AUX only)
Promote top 5 motif candidates to enhanced cards with trader context.
"""
import json
from pathlib import Path

import pandas as pd


def load_aux_context(aux_dir: Path, zone_id: str, center_node_id: int):
    """Load AUX context for enhanced archetype cards."""
    context = {
        "trajectory": None,
        "phase_bucket": None,
        "chain_context": None
    }
    
    try:
        # Load trajectory data
        traj_path = aux_dir / "trajectories.parquet"
        if traj_path.exists():
            traj_df = pd.read_parquet(traj_path)
            zone_traj = traj_df[traj_df['zone_id'] == zone_id]
            
            if len(zone_traj) > 0:
                traj_row = zone_traj.iloc[0]
                context["trajectory"] = {
                    "fwd_ret_12b": float(traj_row.get('fwd_ret_12b', 0)) if pd.notna(traj_row.get('fwd_ret_12b')) else None,
                    "hit_+100_12b": bool(traj_row.get('hit_+100_12b', False)),
                    "hit_+200_12b": bool(traj_row.get('hit_+200_12b', False)),
                    "time_to_+100_bars": float(traj_row.get('time_to_+100_bars', 0)) if pd.notna(traj_row.get('time_to_+100_bars')) else None
                }
        
        # Load phase context
        phase_path = aux_dir / "phase_stats.json"
        if phase_path.exists():
            with open(phase_path) as f:
                phase_data = json.load(f)
            
            # Find which bucket this zone belongs to (simplified mapping)
            for bucket_name, bucket_stats in phase_data.items():
                zones_in_bucket = bucket_stats.get('zones', [])
                if zone_id in zones_in_bucket:
                    context["phase_bucket"] = {
                        "bucket_name": bucket_name,
                        "P_hit_+100_12b": bucket_stats.get('P_hit_+100_12b', 0),
                        "count": bucket_stats.get('count', 0),
                        "median_fwd_ret_12b": bucket_stats.get('median_fwd_ret_12b', 0)
                    }
                    break
        
        # Load chain context
        chains_path = aux_dir / "chains.parquet"
        if chains_path.exists():
            chains_df = pd.read_parquet(chains_path)
            
            # Find chains involving this node
            node_chains = []
            for _, chain in chains_df.iterrows():
                start_zone = chain.get('start_zone_id', '')
                end_zone = chain.get('end_zone_id', '')
                
                # Check if this node is involved
                node_str = f"node_{center_node_id}"
                if node_str in [start_zone, end_zone]:
                    role = "start" if node_str == start_zone else "end"
                    node_chains.append({
                        "chain_type": chain.get('chain', ''),
                        "role": role,
                        "span_bars": int(chain.get('span_bars', 0)),
                        "span_minutes": int(chain.get('span_minutes', 0)),
                        "subsequent_ret_12b": float(chain.get('subsequent_ret_12b', 0)) if pd.notna(chain.get('subsequent_ret_12b')) else None
                    })
            
            if node_chains:
                context["chain_context"] = node_chains[:3]  # Top 3 chains
    
    except Exception as e:
        print(f"Warning: Error loading AUX context for {zone_id}: {e}")
    
    return context

def load_htf_snapshot(market_data_path: Path, center_node_id: int):
    """Load HTF snapshot for archetype card."""
    try:
        market_df = pd.read_parquet(market_data_path)
        node_data = market_df[market_df['node_id'] == center_node_id]
        
        if len(node_data) > 0:
            node_row = node_data.iloc[0]
            return {
                "ts": int(node_row.get('t', 0)),
                "price": float(node_row.get('price', 0)),
                "f47_bar_pos": float(node_row.get('f47', 0)),
                "f48_bar_pos": float(node_row.get('f48', 0)),
                "f49_dist_mid": float(node_row.get('f49', 0)),
                "f50_regime": float(node_row.get('f50', 0))
            }
    except Exception as e:
        print(f"Warning: Error loading HTF snapshot for node {center_node_id}: {e}")
    
    return {}

def create_enhanced_archetype_card(zone_id: str, motif_data: dict, aux_dir: Path, market_data_path: Path, center_node_id: int = None):
    """Create enhanced archetype card with full trader context."""
    
    center_node_id = center_node_id or motif_data.get('center_node_id')
    
    # Load AUX context
    aux_context = load_aux_context(aux_dir, zone_id, center_node_id)
    
    # Load HTF snapshot
    htf_snapshot = load_htf_snapshot(market_data_path, center_node_id)
    
    # Build enhanced card
    card = {
        "zone_id": zone_id,
        "center_node_id": center_node_id,
        "confidence": motif_data.get('confidence', 0),
        "cohesion": motif_data.get('cohesion', 0),
        "in_burst": motif_data.get('in_burst', False),
        
        # Edge mix from original card
        "edge_mix": motif_data.get('edge_mix', {}),
        "liq_links_1_15m": motif_data.get('liq_links_1_15m', 0),
        
        # Enhanced: HTF context
        "htf_snapshot": htf_snapshot,
        
        # Enhanced: Trajectory summary
        "trajectory_summary": aux_context["trajectory"],
        
        # Enhanced: Phase context
        "phase_context": aux_context["phase_bucket"],
        
        # Enhanced: Chain context
        "chain_context": aux_context["chain_context"],
        
        # Trading score (confidence * cohesion weighted by AUX performance)
        "trading_score": calculate_trading_score(motif_data, aux_context)
    }
    
    return card

def calculate_trading_score(motif_data: dict, aux_context: dict) -> float:
    """Calculate composite trading score for archetype."""
    base_score = motif_data.get('confidence', 0) * abs(motif_data.get('cohesion', 0))
    
    # Trajectory multiplier
    traj_mult = 1.0
    if aux_context["trajectory"]:
        traj = aux_context["trajectory"]
        if traj.get("hit_+100_12b"):
            traj_mult += 0.3  # 30% bonus for hitting targets
        if traj.get("fwd_ret_12b") and traj["fwd_ret_12b"] > 0:
            traj_mult += 0.2  # 20% bonus for positive returns
    
    # Phase multiplier
    phase_mult = 1.0
    if aux_context["phase_bucket"]:
        phase = aux_context["phase_bucket"]
        hit_rate = phase.get("P_hit_+100_12b", 0)
        phase_mult = 1.0 + (hit_rate * 0.4)  # Up to 40% bonus
    
    # Chain multiplier
    chain_mult = 1.0
    if aux_context["chain_context"]:
        positive_chains = [c for c in aux_context["chain_context"] if c.get("subsequent_ret_12b", 0) > 0]
        if positive_chains:
            chain_mult += 0.15 * len(positive_chains)  # 15% per positive chain
    
    return float(base_score * traj_mult * phase_mult * chain_mult)

def build_archetype_pack(run_path: Path):
    """Build enhanced archetype pack v0."""
    print("Building archetype pack v0...")
    
    # Load motif candidates
    candidates_path = run_path / "motifs" / "candidates.csv"
    if not candidates_path.exists():
        print("Error: motifs/candidates.csv not found")
        return False
    
    candidates_df = pd.read_parquet(candidates_path) if candidates_path.suffix == '.parquet' else pd.read_csv(candidates_path)
    
    # Select top 5 candidates by confidence * cohesion
    candidates_df['selection_score'] = candidates_df['confidence'] * candidates_df['cohesion'].abs()
    top_candidates = candidates_df.nlargest(5, 'selection_score')
    
    print(f"Selected top 5 candidates: {top_candidates['motif_seed'].tolist()}")
    
    # Load existing card data if available
    cards_dir = run_path / "motifs" / "cards"
    existing_cards = {}
    
    if cards_dir.exists():
        for card_file in cards_dir.glob("*.json"):
            try:
                with open(card_file) as f:
                    card_data = json.load(f)
                existing_cards[card_data.get('zone_id')] = card_data
            except:
                pass
    
    # AUX and market data paths
    aux_dir = run_path / "aux"
    market_data_path = Path("data/shards/NQ_M5/shard_ASIA_2025-08-05/nodes.parquet")
    
    # Build enhanced cards
    enhanced_cards = []
    cards_index = []
    
    for _, candidate in top_candidates.iterrows():
        zone_id = candidate['motif_seed']
        center_node_id = candidate['center_node_id']
        
        # Get existing card data or create basic structure
        if zone_id in existing_cards:
            base_card = existing_cards[zone_id]
        else:
            base_card = {
                'confidence': candidate['confidence'],
                'cohesion': candidate['cohesion'],
                'in_burst': candidate.get('in_burst', False),
                'edge_mix': {
                    'TEMPORAL_NEXT': candidate.get('mix_TEMPORAL_NEXT', 0),
                    'MOVEMENT_TRANSITION': candidate.get('mix_MOVEMENT_TRANSITION', 0),
                    'LIQ_LINK': candidate.get('mix_LIQ_LINK', 0),
                    'CONTEXT': candidate.get('mix_CONTEXT', 0)
                },
                'liq_links_1_15m': candidate.get('liq_links_1_15m', 0)
            }
        
        # Create enhanced card
        enhanced_card = create_enhanced_archetype_card(zone_id, base_card, aux_dir, market_data_path, center_node_id)
        enhanced_cards.append(enhanced_card)
        
        # Determine chain tag
        chain_tag = "none"
        if enhanced_card.get("chain_context"):
            chains = enhanced_card["chain_context"]
            positive_chains = [c for c in chains if c.get("subsequent_ret_12b", 0) > 0]
            if positive_chains:
                chain_tag = positive_chains[0]["chain_type"].split('_')[0]  # e.g., "liquidity", "expansion"
        
        # Determine phase bucket
        phase_bucket = "unknown"
        if enhanced_card.get("phase_context"):
            phase_bucket = enhanced_card["phase_context"]["bucket_name"]
        
        # Add to index
        cards_index.append({
            "zone_id": zone_id,
            "score": enhanced_card["trading_score"],
            "in_burst": enhanced_card["in_burst"],
            "chain_tag": chain_tag,
            "phase_bucket": phase_bucket
        })
        
        # Save enhanced card
        card_path = cards_dir / f"{zone_id}.json"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(card_path, 'w') as f:
            json.dump(enhanced_card, f, indent=2)
        
        print(f"Enhanced card: {zone_id} -> trading_score={enhanced_card['trading_score']:.4f}")
    
    # Save cards index
    index_df = pd.DataFrame(cards_index)
    index_path = cards_dir.parent / "cards_index.csv"
    index_df.to_csv(index_path, index=False)
    
    print(f"Saved {len(enhanced_cards)} enhanced cards and index")
    print(f"Cards directory: {cards_dir}")
    print(f"Index file: {index_path}")
    
    return True

def main():
    run_path = Path("runs/real-tgat-fixed-2025-08-18")
    
    success = build_archetype_pack(run_path)
    
    if success:
        print("\n✅ Archetype pack v0 complete")
        
        # Show summary
        index_path = run_path / "motifs" / "cards_index.csv"
        if index_path.exists():
            index_df = pd.read_csv(index_path)
            print("\n=== Archetype Pack Summary ===")
            print(index_df.to_string(index=False))
    else:
        print("❌ Failed to build archetype pack")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)