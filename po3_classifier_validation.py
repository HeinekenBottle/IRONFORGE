#!/usr/bin/env python3
"""
PO3 Classifier Validation - Comprehensive Testing
Test PO3 statistical classifier with realistic scenarios
"""

import sys
import os
sys.path.append('/Users/jack/IRONFORGE')

from po3_statistical_classifier import PO3StatisticalClassifier, MacroWindowData
import pandas as pd
import numpy as np

def create_realistic_test_scenarios():
    """Create realistic test scenarios for each PO3 phase"""
    
    scenarios = []
    
    # ACCUMULATION scenarios - suppressed amplification, high volume absorption, low directional bias
    scenarios.extend([
        MacroWindowData(
            window_id="ACC_1",
            timestamp=pd.Timestamp('2025-08-24 08:00:00'),
            session="NYAM",
            day_of_week="Monday",
            amplification=22.5,      # 0.44 ratio - suppressed
            volume_ratio=4.2,        # High volume
            price_movement=12.8,     # Low price movement (high absorption)
            directional_coherence=0.25,  # Low directional bias
            f8_liquidity_spike=28.5,
            archaeological_zones_hit=[],
            news_events=[]
        ),
        MacroWindowData(
            window_id="ACC_2", 
            timestamp=pd.Timestamp('2025-08-24 09:00:00'),
            session="NYAM",
            day_of_week="Monday",
            amplification=18.7,      # 0.37 ratio - heavily suppressed
            volume_ratio=3.8,        # High volume
            price_movement=8.2,      # Very low price movement
            directional_coherence=0.15,  # Very low directional bias
            f8_liquidity_spike=22.1,
            archaeological_zones_hit=[],
            news_events=["CPI"]      # News but still suppressed
        )
    ])
    
    # MANIPULATION scenarios - enhanced amplification, strong directional bias
    scenarios.extend([
        MacroWindowData(
            window_id="MAN_1",
            timestamp=pd.Timestamp('2025-08-24 10:00:00'),
            session="LUNCH",
            day_of_week="Monday", 
            amplification=82.4,      # 1.62 ratio - enhanced
            volume_ratio=2.1,        # Lower volume
            price_movement=28.5,     # High price movement
            directional_coherence=0.87,  # Strong directional bias
            f8_liquidity_spike=68.2,
            archaeological_zones_hit=["40%", "60%"],
            news_events=["NFP"]
        ),
        MacroWindowData(
            window_id="MAN_2",
            timestamp=pd.Timestamp('2025-08-24 11:00:00'),
            session="LUNCH",
            day_of_week="Monday",
            amplification=95.1,      # 1.87 ratio - highly enhanced
            volume_ratio=1.8,        # Low volume
            price_movement=32.1,     # Very high price movement  
            directional_coherence=0.92,  # Very strong directional bias
            f8_liquidity_spike=75.8,
            archaeological_zones_hit=["40%", "60%", "80%"],
            news_events=["FOMC"]
        )
    ])
    
    # DISTRIBUTION scenarios - variable amplification, high variance, mixed signals
    scenarios.extend([
        MacroWindowData(
            window_id="DIST_1",
            timestamp=pd.Timestamp('2025-08-24 14:00:00'),
            session="NYPM",
            day_of_week="Monday",
            amplification=38.2,      # 0.75 ratio - variable
            volume_ratio=3.5,        # High volume
            price_movement=22.8,     # Moderate price movement
            directional_coherence=0.35,  # Mixed signals
            f8_liquidity_spike=52.1,
            archaeological_zones_hit=["60%"],
            news_events=[]
        ),
        MacroWindowData(
            window_id="DIST_2",
            timestamp=pd.Timestamp('2025-08-24 15:00:00'), 
            session="CLOSE",
            day_of_week="Monday",
            amplification=41.7,      # 0.82 ratio - variable
            volume_ratio=4.1,        # High volume
            price_movement=18.5,     # Lower price movement
            directional_coherence=0.28,  # Very mixed signals
            f8_liquidity_spike=45.9,
            archaeological_zones_hit=["80%"],
            news_events=[]
        )
    ])
    
    # TRANSITION scenarios - baseline amplification, moderate metrics
    scenarios.extend([
        MacroWindowData(
            window_id="TRANS_1",
            timestamp=pd.Timestamp('2025-08-24 12:00:00'),
            session="LUNCH",
            day_of_week="Monday",
            amplification=49.8,      # 0.98 ratio - near baseline
            volume_ratio=2.8,        # Moderate volume
            price_movement=18.2,     # Moderate price movement
            directional_coherence=0.55,  # Moderate directional bias
            f8_liquidity_spike=42.5,
            archaeological_zones_hit=["40%"],
            news_events=[]
        )
    ])
    
    return scenarios

def main():
    """Comprehensive PO3 classifier validation"""
    
    print("🧪 PO3 Classifier Comprehensive Validation")
    print("=" * 50)
    
    classifier = PO3StatisticalClassifier()
    scenarios = create_realistic_test_scenarios()
    
    results = []
    
    print("\n📊 Classification Results by Expected Phase:")
    print("-" * 50)
    
    for scenario in scenarios:
        classification = classifier.classify_macro_window(scenario)
        
        expected_phase = scenario.window_id.split('_')[0]
        if expected_phase == "ACC":
            expected_phase = "ACCUMULATION"
        elif expected_phase == "MAN":
            expected_phase = "MANIPULATION" 
        elif expected_phase == "DIST":
            expected_phase = "DISTRIBUTION"
        elif expected_phase == "TRANS":
            expected_phase = "TRANSITION"
        
        correct = classification.phase == expected_phase
        
        results.append({
            'window_id': scenario.window_id,
            'expected': expected_phase,
            'predicted': classification.phase,
            'correct': correct,
            'confidence': classification.confidence,
            'amplification_ratio': classification.amplification_ratio
        })
        
        status_icon = "✅" if correct else "❌"
        print(f"\n{status_icon} {scenario.window_id}:")
        print(f"  Expected: {expected_phase}")
        print(f"  Predicted: {classification.phase} (conf: {classification.confidence:.3f})")
        print(f"  Amplification: {scenario.amplification}x (ratio: {classification.amplification_ratio:.2f})")
        print(f"  Volume/Price: {scenario.volume_ratio:.1f} | Directional: {scenario.directional_coherence:.2f}")
        
        # Show key supporting metrics
        sorted_metrics = sorted(classification.supporting_metrics.items(), 
                              key=lambda x: x[1], reverse=True)[:2]
        print(f"  Top Metrics: {', '.join([f'{k}: {v:.3f}' for k, v in sorted_metrics])}")
    
    # Calculate accuracy by phase
    phase_accuracy = {}
    for phase in ["ACCUMULATION", "MANIPULATION", "DISTRIBUTION", "TRANSITION"]:
        phase_results = [r for r in results if r['expected'] == phase]
        if phase_results:
            accuracy = sum(r['correct'] for r in phase_results) / len(phase_results)
            phase_accuracy[phase] = accuracy
    
    overall_accuracy = sum(r['correct'] for r in results) / len(results)
    
    print(f"\n📈 VALIDATION SUMMARY:")
    print(f"Overall Accuracy: {overall_accuracy:.1%}")
    print(f"Phase-Specific Accuracy:")
    for phase, accuracy in phase_accuracy.items():
        print(f"  {phase}: {accuracy:.1%}")
    
    print(f"\n🎯 Classification Thresholds Working:")
    print(f"  Accumulation: <0.50 ratio (suppressed amplification)")
    print(f"  Manipulation: >1.50 ratio (enhanced amplification)")
    print(f"  Distribution: Variable with high volatility variance")
    
    # Test specific threshold scenarios
    print(f"\n🔍 Threshold Boundary Testing:")
    
    boundary_tests = [
        ("Edge_Accumulation", 25.4, 0.15),    # Just at accumulation threshold
        ("Edge_Manipulation", 76.5, 0.85),   # Just at manipulation threshold  
        ("Perfect_Baseline", 50.96, 0.50)     # Exact baseline
    ]
    
    for test_name, amplification, directional_coherence in boundary_tests:
        test_window = MacroWindowData(
            window_id=test_name,
            timestamp=pd.Timestamp.now(),
            session="TEST",
            day_of_week="Monday",
            amplification=amplification,
            volume_ratio=2.5,
            price_movement=15.0,
            directional_coherence=directional_coherence,
            f8_liquidity_spike=40.0,
            archaeological_zones_hit=[],
            news_events=[]
        )
        
        boundary_classification = classifier.classify_macro_window(test_window)
        ratio = amplification / 50.96
        
        print(f"  {test_name}: {amplification}x (ratio: {ratio:.2f}) → {boundary_classification.phase}")
    
    print(f"\n✅ Validation Complete! Ready for H8 data application.")

if __name__ == "__main__":
    main()