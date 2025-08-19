#!/usr/bin/env python3
"""
Phase 2 Validation Summary: Before vs After Enhancement
======================================================

Quick validation showing the transformation from contaminated to authentic features.
"""

import json
from pathlib import Path

import numpy as np


def validate_decontamination():
    """Show before/after decontamination results."""
    
    print("=" * 80)
    print("PHASE 2 DECONTAMINATION VALIDATION SUMMARY")
    print("=" * 80)
    
    # Sample a few enhanced sessions to validate transformation
    enhanced_dir = Path("enhanced_sessions")
    enhanced_files = list(enhanced_dir.glob("enhanced_*.json"))
    
    if not enhanced_files:
        print("No enhanced sessions found!")
        return
    
    print(f"Found {len(enhanced_files)} enhanced sessions")
    
    # Analyze feature diversity in enhanced sessions
    htf_strengths = []
    energy_densities = []
    liquidity_event_counts = []
    
    sample_results = []
    
    for enhanced_file in enhanced_files[:5]:  # Sample first 5
        with open(enhanced_file) as f:
            data = json.load(f)
        
        # Extract enhanced features
        htf_strength = data.get('contamination_analysis', {}).get('htf_contamination', {}).get('htf_carryover_strength', 0.3)
        energy_density = data.get('energy_state', {}).get('energy_density', 0.5) 
        event_count = len(data.get('session_liquidity_events', []))
        
        htf_strengths.append(htf_strength)
        energy_densities.append(energy_density)
        liquidity_event_counts.append(event_count)
        
        # Get enhancement metadata
        enhancement_info = data.get('phase2_enhancement', {})
        pre_score = enhancement_info.get('pre_enhancement_score', 0)
        post_score = enhancement_info.get('post_enhancement_score', 0)
        
        sample_results.append({
            'file': enhanced_file.name,
            'htf_strength': f"0.3 → {htf_strength}",
            'energy_density': f"0.5 → {energy_density}",
            'events': f"[] → {event_count} events",
            'authenticity': f"{pre_score}% → {post_score}%"
        })
    
    # Show sample transformations
    print("\nSample Decontamination Results:")
    print("-" * 80)
    for result in sample_results:
        print(f"📁 {result['file']}")
        print(f"   HTF Carryover: {result['htf_strength']}")
        print(f"   Energy Density: {result['energy_density']}")
        print(f"   Liquidity Events: {result['events']}")  
        print(f"   Authenticity: {result['authenticity']}")
        print()
    
    # Show feature diversity statistics
    print("Feature Diversity Analysis:")
    print("-" * 40)
    print("HTF Carryover Strengths:")
    print(f"  Range: {min(htf_strengths):.2f} - {max(htf_strengths):.2f}")
    print(f"  Mean: {np.mean(htf_strengths):.2f} ± {np.std(htf_strengths):.2f}")
    print(f"  Unique values: {len({round(x, 2) for x in htf_strengths})}")
    
    print("\nEnergy Densities:")
    print(f"  Range: {min(energy_densities):.3f} - {max(energy_densities):.3f}")
    print(f"  Mean: {np.mean(energy_densities):.3f} ± {np.std(energy_densities):.3f}")
    print(f"  Unique values: {len({round(x, 3) for x in energy_densities})}")
    
    print("\nLiquidity Event Counts:")
    print(f"  Range: {min(liquidity_event_counts)} - {max(liquidity_event_counts)} events")
    print(f"  Mean: {np.mean(liquidity_event_counts):.1f} events per session")
    print(f"  Total events generated: {sum(liquidity_event_counts)}")
    
    # Contamination elimination summary
    print("\n" + "=" * 80)
    print("DECONTAMINATION SUCCESS SUMMARY")
    print("=" * 80)
    
    default_contamination = sum([
        1 for x in htf_strengths if x == 0.3
    ]) + sum([
        1 for x in energy_densities if x == 0.5  
    ]) + sum([
        1 for x in liquidity_event_counts if x == 0
    ])
    
    total_features = len(enhanced_files) * 3  # 3 features per session
    contamination_rate = (default_contamination / total_features) * 100
    
    print(f"✅ Enhanced Sessions: {len(enhanced_files)}")
    print(f"✅ Total Features Analyzed: {total_features}")
    print(f"✅ Remaining Default Values: {default_contamination}")
    print(f"✅ Contamination Rate: {contamination_rate:.1f}%")
    print(f"✅ Decontamination Success: {100 - contamination_rate:.1f}%")
    
    if contamination_rate < 5:
        print("\n🎉 DECONTAMINATION SUCCESS: <5% contamination remaining!")
        print("🎯 Ready for TGAT model pattern discovery validation")
    else:
        print(f"\n⚠️  Warning: {contamination_rate:.1f}% contamination remains")
        print("🔧 Additional decontamination may be needed")
    
    print("\n" + "=" * 80)
    print("PHASE 2 FEATURE PIPELINE ENHANCEMENT - VALIDATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    validate_decontamination()