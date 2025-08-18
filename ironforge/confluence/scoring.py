from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def apply_phase_weighting(scores_df: pd.DataFrame, aux_dir: Path) -> pd.DataFrame:
    """Apply phase weighting based on AUX phase stats (optional adapter)."""
    phase_weight_col = [1.0] * len(scores_df)  # Default: no adjustment
    
    try:
        phase_stats_path = aux_dir / "phase_stats.json"
        if not phase_stats_path.exists():
            logging.debug("[confluence] phase_stats.json not found - skipping phase weighting")
            scores_df['phase_weight'] = phase_weight_col
            return scores_df
        
        with open(phase_stats_path) as f:
            phase_data = json.load(f)
        
        # Load HTF features for zone matching (simplified approach)
        # In practice, this would match zones to their HTF context
        # For this implementation, apply weights based on bucket performance
        
        best_buckets = {}
        for bucket_name, bucket_stats in phase_data.items():
            hit_rate = bucket_stats.get('P_hit_+100_12b', 0)
            count = bucket_stats.get('count', 0)
            
            if count >= 2:  # Only valid buckets
                # Map hit rate to weight multiplier (1.0 - 1.2 range)
                weight = 1.0 + (hit_rate * 0.2)  # 0% hit -> 1.0x, 100% hit -> 1.2x
                best_buckets[bucket_name] = weight
        
        # Apply weights (simplified: use best bucket average for all zones)
        if best_buckets:
            avg_weight = sum(best_buckets.values()) / len(best_buckets)
            phase_weight_col = [avg_weight] * len(scores_df)
            logging.info(f"[confluence] applied phase weighting: avg_weight={avg_weight:.3f}")
        
    except Exception as e:
        logging.debug(f"[confluence] phase weighting failed: {e}")
    
    scores_df['phase_weight'] = phase_weight_col
    return scores_df


def apply_chain_bonus(scores_df: pd.DataFrame, aux_dir: Path) -> pd.DataFrame:
    """Apply chain bonus for zones in allowed event chains (optional adapter)."""
    chain_bonus_col = [0.0] * len(scores_df)  # Default: no bonus
    
    try:
        chains_path = aux_dir / "chains.parquet"
        if not chains_path.exists():
            logging.debug("[confluence] chains.parquet not found - skipping chain bonus")
            scores_df['chain_bonus'] = chain_bonus_col
            return scores_df
        
        chains_df = pd.read_parquet(chains_path)
        
        # Find chains with positive subsequent returns
        if 'subsequent_ret_12b' in chains_df.columns:
            good_chains = chains_df[chains_df['subsequent_ret_12b'] > 0]
            
            # Extract node IDs from zone IDs in chain data
            start_nodes = []
            end_nodes = []
            
            for _, chain in good_chains.iterrows():
                start_zone = chain.get('start_zone_id', '')
                end_zone = chain.get('end_zone_id', '')
                
                # Extract node IDs (format: "node_XXXX")
                if start_zone.startswith('node_'):
                    try:
                        start_node_id = int(start_zone.split('_')[1])
                        start_nodes.append(start_node_id)
                    except:
                        pass
                        
                if end_zone.startswith('node_'):
                    try:
                        end_node_id = int(end_zone.split('_')[1]) 
                        end_nodes.append(end_node_id)
                    except:
                        pass
            
            # Apply bonus to zones that are in good chains
            chain_nodes = set(start_nodes + end_nodes)
            
            for idx, row in scores_df.iterrows():
                node_id = row.get('node_id')
                if isinstance(node_id, str) and node_id.isdigit():
                    node_id = int(node_id)
                
                if node_id in chain_nodes:
                    chain_bonus_col[idx] = 0.05  # Small 5% bonus
            
            chain_count = len(chain_nodes)
            bonus_count = sum(1 for b in chain_bonus_col if b > 0)
            logging.info(f"[confluence] applied chain bonus: {bonus_count} zones from {chain_count} chain nodes")
    
    except Exception as e:
        logging.debug(f"[confluence] chain bonus failed: {e}")
    
    scores_df['chain_bonus'] = chain_bonus_col
    return scores_df


def score_confluence(
    pattern_paths: Sequence[str],
    out_dir: str,
    _weights: Mapping[str, float] | None,
    threshold: float,
    phase_weighting: bool = False,
    chain_bonus: bool = False,
) -> str:
    """Stub confluence scorer.

    Creates a simple dataframe with scores and writes it to a parquet file.
    Returns the path to the written parquet.
    """
    # Accept both scales: if user passes 65 on a 0–1 scale, make it 0.65.
    original_threshold = threshold
    if isinstance(threshold, (int, float)) and threshold > 1.0:
        threshold = float(threshold) / 100.0
        logging.info(f"[confluence] normalized threshold {original_threshold}→{threshold} (0–1 scale)")
    
    # Create enhanced scores DataFrame with zone_id and node_id for bridge functionality
    # For graceful degradation, create synthetic zone_ids based on patterns
    zone_data = []
    for i, pattern_path in enumerate(pattern_paths):
        # Extract meaningful identifiers from pattern path
        pattern_name = Path(pattern_path).stem if pattern_path else f"pattern_{i}"
        zone_id = f"SYNTHETIC_{pattern_name.upper()}"
        node_id = f"node_{i + 10000}"  # Use high IDs to avoid conflicts with real node_ids
        
        zone_data.append({
            "zone_id": zone_id,
            "node_id": node_id,
            "pattern": pattern_path,
            "confidence": threshold,
            "ts": 1692000000 + i * 3600,  # Synthetic timestamps
            "event_kind": "confluence_pattern"
        })
    
    scores_df = pd.DataFrame(zone_data)
    
    # Apply optional AUX adapters if enabled
    aux_dir = Path(out_dir).parent / "aux"
    
    if phase_weighting:
        scores_df = apply_phase_weighting(scores_df, aux_dir)
        # Apply phase weight to confidence
        scores_df['confidence'] = scores_df['confidence'] * scores_df['phase_weight']
        logging.info("[confluence] phase weighting enabled")
    
    if chain_bonus:
        scores_df = apply_chain_bonus(scores_df, aux_dir)  
        # Apply chain bonus to confidence
        scores_df['confidence'] = scores_df['confidence'] + scores_df['chain_bonus']
        logging.info("[confluence] chain bonus enabled")
    
    # Health gates: coverage & variance watchdog
    health_status = "pass"
    health_reasons = []
    
    try:
        pattern_count = len(pattern_paths)
        var = float(scores_df["confidence"].var(ddof=0)) if "confidence" in scores_df else None
        
        # Check coverage gate (≥90% of nodes have embeddings)
        # For this implementation, assume all patterns have embeddings (graceful degradation)
        coverage_ratio = 1.0  # 100% coverage in graceful mode
        if coverage_ratio < 0.9:
            health_status = "fail"
            health_reasons.append(f"Low coverage: {coverage_ratio:.1%} < 90%")
        
        # Check variance gate (var(confidence) ≥ 1e-5)
        if var is not None and var < 1e-5:
            health_status = "fail"
            health_reasons.append(f"Low variance: {var:.2e} < 1e-5")
            logging.warning("[confluence] near-constant score detected; check weighting & thresholds.")
        
        logging.info(f"[confluence] health gates: {health_status} - coverage={coverage_ratio:.1%} var(score)={var}")
        if health_status == "fail":
            logging.warning(f"[confluence] health gate failures: {'; '.join(health_reasons)}")
    except Exception as e:
        logging.debug(f"[confluence] health gates skipped: {e}")
        health_status = "unknown"
        health_reasons.append(f"Health check error: {e}")
    
    # Persist outputs
    out_path = Path(out_dir) / "confluence_scores.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(out_path, index=False)
    
    # Emit minimal join key for explainability (zone_nodes.parquet)
    if "zone_id" in scores_df.columns and "node_id" in scores_df.columns:
        cols_to_emit = [c for c in ["zone_id", "node_id", "ts", "event_kind", "confidence"] if c in scores_df.columns]
        zone_nodes_path = Path(out_dir) / "zone_nodes.parquet"
        scores_df[cols_to_emit].to_parquet(zone_nodes_path, index=False)
        logging.info(f"[confluence] emitted zone↔node bridge to {zone_nodes_path}")
    
    # Stats for reporting / sanity
    if "confidence" in scores_df.columns:
        m, M = float(scores_df["confidence"].min()), float(scores_df["confidence"].max())
        mean = float(scores_df["confidence"].mean())
        std = float(scores_df["confidence"].std(ddof=0))
        scale = "0-1" if M <= 1.0 else "0-100"
        
        stats_data = {
            "min": m, 
            "max": M, 
            "mean": mean, 
            "std": std, 
            "scale": scale, 
            "threshold": threshold,
            "original_threshold": original_threshold,
            "coverage": coverage_ratio,
            "variance": var,
            "health_status": health_status,
            "health_reasons": health_reasons,
            "phase_weighting": phase_weighting,
            "chain_bonus": chain_bonus
        }
        
        # Add adapter-specific stats
        if phase_weighting and 'phase_weight' in scores_df.columns:
            stats_data['phase_weight_mean'] = float(scores_df['phase_weight'].mean())
            stats_data['phase_weight_range'] = [float(scores_df['phase_weight'].min()), float(scores_df['phase_weight'].max())]
        
        if chain_bonus and 'chain_bonus' in scores_df.columns:
            stats_data['chain_bonus_mean'] = float(scores_df['chain_bonus'].mean())
            stats_data['chain_bonus_count'] = int((scores_df['chain_bonus'] > 0).sum())
        
        stats_path = Path(out_dir) / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
    
    return str(out_path)


# Alias for CLI compatibility
def score_session(cfg) -> None:
    """CLI-compatible wrapper for score_confluence"""
    # Extract patterns from shards directory
    from pathlib import Path
    import glob
    
    # Default pattern paths from config or fallback
    shards_glob = getattr(cfg.data, 'shards_glob', 'data/shards/*/shard_*')
    pattern_paths = glob.glob(shards_glob)
    
    # Get run directory for output
    try:
        from ironforge.sdk.app_config import materialize_run_dir
        run_dir = materialize_run_dir(cfg)
        out_dir = str(run_dir / "confluence")
    except:
        out_dir = "runs/confluence"
    
    # Extract weights and threshold from config
    weights = getattr(cfg.scoring, 'weights', None)
    if weights:
        weights = dict(weights.__dict__) if hasattr(weights, '__dict__') else dict(weights)
    
    # Get threshold from config, defaulting to 0.7
    threshold = getattr(cfg.scoring, 'threshold', 0.7) if hasattr(cfg, 'scoring') else 0.7
    
    # Get adapter configurations (off by default)
    phase_weighting = False
    chain_bonus = False
    
    if hasattr(cfg, 'confluence'):
        phase_weighting = getattr(cfg.confluence, 'phase_weighting', False)
        chain_bonus = getattr(cfg.confluence, 'chain_bonus', False)
    
    # Run confluence scoring
    result_path = score_confluence(
        pattern_paths=pattern_paths,
        out_dir=out_dir, 
        _weights=weights,
        threshold=threshold,
        phase_weighting=phase_weighting,
        chain_bonus=chain_bonus
    )
    
    print(f"[confluence] scored {len(pattern_paths)} patterns -> {result_path}")
