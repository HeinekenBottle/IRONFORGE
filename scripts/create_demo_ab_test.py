#!/usr/bin/env python3
"""
Create Demo A/B Test Data
Generate realistic synthetic data to demonstrate adapter validation methodology
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_demo_data():
    """Create realistic demo data for A/B testing."""
    
    np.random.seed(42)  # Reproducible results
    
    # Create 20 zone patterns with realistic characteristics
    n_zones = 20
    
    # Generate base confluence scores (realistic range 0.5-0.9)
    zones_data = []
    for i in range(n_zones):
        zones_data.append({
            'zone_id': f'zone_{i}',
            'node_id': 4389 + i,
            'pattern': f'demo_pattern_{i}',
            'confidence': np.random.normal(0.7, 0.1),  # Mean 0.7, std 0.1
            'ts': 1692000000 + i * 3600,  # Hourly intervals
            'event_kind': 'confluence_pattern'
        })
    
    scores_df = pd.DataFrame(zones_data)
    scores_df['confidence'] = scores_df['confidence'].clip(0.4, 0.95)  # Reasonable bounds
    
    # Create realistic trajectory data
    trajectories_data = []
    for i, zone_id in enumerate(scores_df['zone_id']):
        center_node_id = 4389 + i
        
        # Simulate hit rates based on confidence (higher confidence -> higher hit rate)
        base_confidence = scores_df.loc[i, 'confidence']
        hit_prob_100 = base_confidence * 0.4  # Scale to realistic hit rates
        
        # Generate realistic forward returns (small moves in NQ points)
        fwd_ret_12b = np.random.normal(0, 0.8)  # Mean 0, std 0.8%
        
        trajectories_data.append({
            'zone_id': zone_id,
            'center_node_id': center_node_id,
            'ts': 1692000000 + i * 3600,
            'price_c': 15000 + np.random.normal(0, 100),  # Realistic NQ price
            'fwd_ret_3b': np.random.normal(0, 0.4),
            'fwd_ret_12b': fwd_ret_12b,
            'fwd_ret_24b': np.random.normal(0, 1.2),
            'hit_+50_12b': np.random.random() < (hit_prob_100 * 1.5),  # 50 tick easier than 100
            'hit_+100_12b': np.random.random() < hit_prob_100,
            'hit_+200_12b': np.random.random() < (hit_prob_100 * 0.6),  # 200 tick harder
            'time_to_+50_bars': np.random.randint(1, 8) if np.random.random() < 0.7 else np.nan,
            'time_to_+100_bars': np.random.randint(2, 12) if np.random.random() < 0.5 else np.nan,
            'time_to_+200_bars': np.random.randint(4, 24) if np.random.random() < 0.3 else np.nan,
        })
    
    trajectories_df = pd.DataFrame(trajectories_data)
    
    return scores_df, trajectories_df

def run_demo_ab_test():
    """Run demo A/B test with realistic synthetic data."""
    
    print("=== Demo A/B Test: Adapter Validation Methodology ===")
    print("Using synthetic but realistic data to demonstrate validation approach")
    
    # Create demo data
    baseline_scores, trajectories = create_demo_data()
    
    print(f"Generated: {len(baseline_scores)} confluence scores, {len(trajectories)} trajectories")
    print(f"Base hit rate (100 ticks): {trajectories['hit_+100_12b'].mean():.1%}")
    print(f"Base forward returns: μ={trajectories['fwd_ret_12b'].mean():.3f}%, σ={trajectories['fwd_ret_12b'].std():.3f}%")
    
    # Simulate baseline (adapters OFF)
    baseline_merged = baseline_scores.merge(trajectories[["zone_id", "hit_+100_12b", "fwd_ret_12b"]], on="zone_id", how="inner")
    
    # Simulate enhanced run (adapters ON)
    enhanced_scores = baseline_scores.copy()
    
    # Apply phase weighting: zones with higher hit rates get confidence boost
    phase_multiplier = 1.0 + (trajectories['hit_+100_12b'].astype(int) * 0.15)  # 15% boost for zones that hit targets
    
    # Apply chain bonus: zones with positive returns get confidence boost  
    chain_multiplier = 1.0 + (trajectories['fwd_ret_12b'] > 0).astype(int) * 0.08  # 8% boost for positive returns
    
    # Combine adapter effects
    enhanced_scores['confidence'] = (enhanced_scores['confidence'] * phase_multiplier * chain_multiplier).clip(0, 1.0)
    
    enhanced_merged = enhanced_scores.merge(trajectories[["zone_id", "hit_+100_12b", "fwd_ret_12b"]], on="zone_id", how="inner")
    
    # Calculate metrics for both runs
    results = []
    
    # Baseline metrics
    results.append({
        "run": "baseline_adapters_OFF",
        "P(hit+100)": baseline_merged["hit_+100_12b"].mean(),
        "med_ret_12b": baseline_merged["fwd_ret_12b"].median(),
        "var(conf)": baseline_merged["confidence"].var(ddof=0),
        "threshold": 65,
        "zones": len(baseline_merged)
    })
    
    # Enhanced metrics  
    results.append({
        "run": "enhanced_adapters_ON",
        "P(hit+100)": enhanced_merged["hit_+100_12b"].mean(),
        "med_ret_12b": enhanced_merged["fwd_ret_12b"].median(),
        "var(conf)": enhanced_merged["confidence"].var(ddof=0),
        "threshold": 65,
        "zones": len(enhanced_merged)
    })
    
    # Create results DataFrame and display
    results_df = pd.DataFrame(results)
    
    print("\\n=== A/B Test Results ===")
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(results_df.to_string(index=False))
    
    # Analyze results
    baseline_row = results[0]
    enhanced_row = results[1]
    
    hit_rate_lift = enhanced_row["P(hit+100)"] - baseline_row["P(hit+100)"]
    ret_lift = enhanced_row["med_ret_12b"] - baseline_row["med_ret_12b"]
    var_change = enhanced_row["var(conf)"] - baseline_row["var(conf)"]
    
    print("\\n=== Performance Lift Analysis ===")
    print(f"Hit Rate Lift: {hit_rate_lift:+.4f} ({hit_rate_lift/baseline_row['P(hit+100)']*100:+.1f}%)")
    print(f"Return Lift: {ret_lift:+.4f} ({ret_lift/abs(baseline_row['med_ret_12b'])*100:+.1f}%)")
    print(f"Confidence Var Change: {var_change:+.2e}")
    
    # Acceptance criteria
    accept_hit_rate = hit_rate_lift > 0.01  # At least 1% improvement
    accept_returns = ret_lift > 0 or abs(ret_lift) < 0.02  # Positive or minimal degradation
    accept_variance = var_change < 0.01  # Variance doesn't increase dramatically
    
    print("\\n=== Acceptance Criteria ===")
    print(f"✅ Hit Rate Improvement: {accept_hit_rate} (lift: {hit_rate_lift:+.4f})")
    print(f"✅ Returns Acceptable: {accept_returns} (lift: {ret_lift:+.4f})")  
    print(f"✅ Variance Healthy: {accept_variance} (change: {var_change:+.2e})")
    
    overall_accept = accept_hit_rate and accept_returns and accept_variance
    print(f"\\n🎯 Overall Acceptance: {'✅ PASS' if overall_accept else '❌ FAIL'}")
    
    if overall_accept:
        print("\\n📊 Recommendation: Keep adapters ON - they show measurable improvement at trade horizons")
        print("   - Phase weighting effectively identifies high-performing HTF buckets")
        print("   - Chain bonus rewards zones with positive subsequent returns")
        print("   - Confidence variance remains stable")
    else:
        print("\\n📊 Recommendation: Keep adapters OFF - insufficient evidence of improvement")
    
    # Show comparison details
    print("\\n=== Detailed Comparison ===")
    print("Confidence score changes:")
    conf_changes = enhanced_scores['confidence'] - baseline_scores['confidence']
    print(f"  Mean change: {conf_changes.mean():+.4f}")
    print(f"  Max boost: {conf_changes.max():+.4f}")
    print(f"  Zones boosted: {(conf_changes > 0).sum()}/{len(conf_changes)}")
    
    # Save demo data for inspection
    demo_dir = Path("runs/demo-ab-test")
    demo_dir.mkdir(exist_ok=True)
    
    confluence_dir = demo_dir / "confluence"
    confluence_dir.mkdir(exist_ok=True)
    aux_dir = demo_dir / "aux"
    aux_dir.mkdir(exist_ok=True)
    
    # Save baseline and enhanced scores
    baseline_scores.to_parquet(confluence_dir / "baseline_scores.parquet", index=False)
    enhanced_scores.to_parquet(confluence_dir / "enhanced_scores.parquet", index=False)
    trajectories.to_parquet(aux_dir / "trajectories.parquet", index=False)
    
    # Save results summary
    with open(demo_dir / "ab_test_results.json", 'w') as f:
        json.dump({
            "baseline": baseline_row,
            "enhanced": enhanced_row,
            "lift_analysis": {
                "hit_rate_lift": hit_rate_lift,
                "return_lift": ret_lift,
                "variance_change": var_change
            },
            "acceptance": {
                "hit_rate_ok": bool(accept_hit_rate),
                "returns_ok": bool(accept_returns),
                "variance_ok": bool(accept_variance),
                "overall_pass": bool(overall_accept)
            }
        }, f, indent=2)
    
    print(f"\\n💾 Demo data saved to: {demo_dir}")
    
    return overall_accept

def main():
    success = run_demo_ab_test()
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)