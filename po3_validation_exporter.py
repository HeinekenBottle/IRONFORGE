#!/usr/bin/env python3
"""
PO3 Framework Validation Results Exporter
Export comprehensive validation results to CSV format for analysis
"""

import sys
import os
sys.path.append('/Users/jack/IRONFORGE')

from po3_statistical_classifier import PO3StatisticalClassifier, MacroWindowData
from po3_testing_framework import PO3TestingFramework
import pandas as pd
import numpy as np
from datetime import datetime

def create_validation_test_scenarios():
    """Create comprehensive test scenarios for validation export"""
    
    scenarios = []
    
    # ACCUMULATION scenarios - suppressed amplification, high volume absorption
    accumulation_scenarios = [
        {"id": "ACC_1", "amp": 22.5, "vol": 4.2, "price": 12.8, "dir": 0.25, "f8": 28.5, "zones": [], "news": []},
        {"id": "ACC_2", "amp": 18.7, "vol": 3.8, "price": 8.2, "dir": 0.15, "f8": 22.1, "zones": [], "news": ["CPI"]},
        {"id": "ACC_3", "amp": 24.1, "vol": 5.1, "price": 10.5, "dir": 0.20, "f8": 31.2, "zones": [], "news": []},
        {"id": "ACC_4", "amp": 19.8, "vol": 4.5, "price": 9.1, "dir": 0.18, "f8": 25.8, "zones": [], "news": ["Jobless"]}
    ]
    
    # MANIPULATION scenarios - enhanced amplification, strong directional bias  
    manipulation_scenarios = [
        {"id": "MAN_1", "amp": 82.4, "vol": 2.1, "price": 28.5, "dir": 0.87, "f8": 68.2, "zones": ["40%", "60%"], "news": ["NFP"]},
        {"id": "MAN_2", "amp": 95.1, "vol": 1.8, "price": 32.1, "dir": 0.92, "f8": 75.8, "zones": ["40%", "60%", "80%"], "news": ["FOMC"]},
        {"id": "MAN_3", "amp": 78.9, "vol": 2.3, "price": 26.2, "dir": 0.83, "f8": 71.4, "zones": ["60%"], "news": []},
        {"id": "MAN_4", "amp": 89.7, "vol": 1.9, "price": 31.8, "dir": 0.89, "f8": 73.6, "zones": ["40%", "80%"], "news": ["PPI"]}
    ]
    
    # DISTRIBUTION scenarios - variable amplification, high variance, mixed signals
    distribution_scenarios = [
        {"id": "DIST_1", "amp": 38.2, "vol": 3.5, "price": 22.8, "dir": 0.35, "f8": 52.1, "zones": ["60%"], "news": []},
        {"id": "DIST_2", "amp": 41.7, "vol": 4.1, "price": 18.5, "dir": 0.28, "f8": 45.9, "zones": ["80%"], "news": []},
        {"id": "DIST_3", "amp": 44.3, "vol": 3.8, "price": 20.1, "dir": 0.32, "f8": 48.7, "zones": ["40%", "60%"], "news": []},
        {"id": "DIST_4", "amp": 36.8, "vol": 4.2, "price": 19.8, "dir": 0.29, "f8": 51.2, "zones": ["60%", "80%"], "news": ["Retail"]}
    ]
    
    # TRANSITION scenarios - baseline amplification, moderate metrics
    transition_scenarios = [
        {"id": "TRANS_1", "amp": 49.8, "vol": 2.8, "price": 18.2, "dir": 0.55, "f8": 42.5, "zones": ["40%"], "news": []},
        {"id": "TRANS_2", "amp": 52.1, "vol": 2.5, "price": 19.5, "dir": 0.58, "f8": 44.8, "zones": [], "news": []},
        {"id": "TRANS_3", "amp": 48.2, "vol": 2.9, "price": 17.8, "dir": 0.52, "f8": 41.2, "zones": ["40%"], "news": ["ISM"]}
    ]
    
    # Combine all scenarios
    all_scenarios = accumulation_scenarios + manipulation_scenarios + distribution_scenarios + transition_scenarios
    
    # Convert to MacroWindowData objects
    for i, scenario in enumerate(all_scenarios):
        expected_phase = scenario["id"].split('_')[0]
        if expected_phase == "ACC":
            expected_phase = "ACCUMULATION"
        elif expected_phase == "MAN":
            expected_phase = "MANIPULATION" 
        elif expected_phase == "DIST":
            expected_phase = "DISTRIBUTION"
        elif expected_phase == "TRANS":
            expected_phase = "TRANSITION"
        
        # Assign sessions based on expected phase
        session_mapping = {
            "ACCUMULATION": "NYAM",
            "MANIPULATION": "LUNCH", 
            "DISTRIBUTION": "NYPM",
            "TRANSITION": "CLOSE"
        }
        
        window = MacroWindowData(
            window_id=scenario["id"],
            timestamp=pd.Timestamp('2025-08-24 08:00:00') + pd.Timedelta(hours=i),
            session=session_mapping[expected_phase],
            day_of_week="Monday",
            amplification=scenario["amp"],
            volume_ratio=scenario["vol"],
            price_movement=scenario["price"],
            directional_coherence=scenario["dir"],
            f8_liquidity_spike=scenario["f8"],
            archaeological_zones_hit=scenario["zones"],
            news_events=scenario["news"]
        )
        
        scenarios.append({
            'window_data': window,
            'expected_phase': expected_phase
        })
    
    return scenarios

def run_comprehensive_validation_export():
    """Run comprehensive validation and export results to CSV"""
    
    print("📊 PO3 Framework Validation Results Exporter")
    print("=" * 60)
    
    # Initialize classifier
    classifier = PO3StatisticalClassifier()
    testing_framework = PO3TestingFramework()
    
    # Create test scenarios
    test_scenarios = create_validation_test_scenarios()
    
    # Run classifications and collect detailed results
    detailed_results = []
    
    print(f"\n🧪 Running validation on {len(test_scenarios)} test scenarios...")
    
    for scenario in test_scenarios:
        window_data = scenario['window_data']
        expected_phase = scenario['expected_phase']
        
        # Get PO3 classification
        classification = classifier.classify_macro_window(window_data)
        
        # Calculate amplification metrics for detailed analysis
        metrics = classifier.calculate_amplification_metrics(window_data)
        
        # Compile detailed result
        result = {
            # Basic identifiers
            'window_id': window_data.window_id,
            'timestamp': window_data.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'session': window_data.session,
            'day_of_week': window_data.day_of_week,
            
            # Input metrics
            'amplification': window_data.amplification,
            'amplification_ratio': classification.amplification_ratio,
            'volume_ratio': window_data.volume_ratio,
            'price_movement': window_data.price_movement,
            'directional_coherence': window_data.directional_coherence,
            'f8_liquidity_spike': window_data.f8_liquidity_spike,
            'archaeological_zones_count': len(window_data.archaeological_zones_hit),
            'news_events_count': len(window_data.news_events),
            
            # Classification results
            'expected_phase': expected_phase,
            'predicted_phase': classification.phase,
            'classification_correct': classification.phase == expected_phase,
            'confidence': classification.confidence,
            'statistical_significance': classification.statistical_significance,
            
            # Detailed metrics
            'volume_price_ratio': metrics['volume_price_ratio'],
            'volatility_variance': metrics['volatility_variance'],
            'zone_impact': metrics['zone_impact'],
            'news_impact': metrics['news_impact'],
            
            # Supporting evidence (top 3 metrics)
            'top_metric_1': max(classification.supporting_metrics, key=classification.supporting_metrics.get),
            'top_metric_1_value': max(classification.supporting_metrics.values()),
            'top_metric_2': sorted(classification.supporting_metrics.items(), key=lambda x: x[1], reverse=True)[1][0] if len(classification.supporting_metrics) > 1 else '',
            'top_metric_2_value': sorted(classification.supporting_metrics.values(), reverse=True)[1] if len(classification.supporting_metrics) > 1 else 0,
            'top_metric_3': sorted(classification.supporting_metrics.items(), key=lambda x: x[1], reverse=True)[2][0] if len(classification.supporting_metrics) > 2 else '',
            'top_metric_3_value': sorted(classification.supporting_metrics.values(), reverse=True)[2] if len(classification.supporting_metrics) > 2 else 0,
            
            # Phase-specific scores  
            'accumulation_score': 0.0,  # Will be calculated below
            'manipulation_score': 0.0,
            'distribution_score': 0.0
        }
        
        # Calculate individual phase scores for analysis
        acc_score, _ = classifier.classify_accumulation_phase(metrics)
        man_score, _ = classifier.classify_manipulation_phase(metrics)
        dist_score, _ = classifier.classify_distribution_phase(metrics)
        
        result['accumulation_score'] = acc_score
        result['manipulation_score'] = man_score
        result['distribution_score'] = dist_score
        
        detailed_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(detailed_results)
    
    # Calculate summary statistics
    overall_accuracy = results_df['classification_correct'].mean()
    phase_accuracy = results_df.groupby('expected_phase')['classification_correct'].mean()
    avg_confidence = results_df['confidence'].mean()
    
    print(f"\n📈 VALIDATION SUMMARY:")
    print(f"Overall Accuracy: {overall_accuracy:.1%}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Phase-Specific Accuracy:")
    for phase, accuracy in phase_accuracy.items():
        print(f"  {phase}: {accuracy:.1%}")
    
    # Export main results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_csv_path = f"/Users/jack/IRONFORGE/data/validation/po3_validation_results_{timestamp}.csv"
    
    # Create directory if it doesn't exist
    os.makedirs("/Users/jack/IRONFORGE/data/validation", exist_ok=True)
    
    results_df.to_csv(main_csv_path, index=False)
    print(f"\n✅ Main validation results exported to: {main_csv_path}")
    
    # Export summary statistics
    phase_accuracy_list = [phase_accuracy[phase] for phase in phase_accuracy.index]
    summary_data = {
        'metric': ['overall_accuracy', 'average_confidence', 'total_samples'] + [f'{phase}_accuracy' for phase in phase_accuracy.index],
        'value': [overall_accuracy, avg_confidence, len(results_df)] + phase_accuracy_list
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = f"/Users/jack/IRONFORGE/data/validation/po3_validation_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Summary statistics exported to: {summary_csv_path}")
    
    # Export confusion matrix
    confusion_data = []
    for expected in results_df['expected_phase'].unique():
        for predicted in results_df['predicted_phase'].unique():
            count = len(results_df[(results_df['expected_phase'] == expected) & 
                                 (results_df['predicted_phase'] == predicted)])
            confusion_data.append({
                'expected_phase': expected,
                'predicted_phase': predicted, 
                'count': count
            })
    
    confusion_df = pd.DataFrame(confusion_data)
    confusion_csv_path = f"/Users/jack/IRONFORGE/data/validation/po3_confusion_matrix_{timestamp}.csv"
    confusion_df.to_csv(confusion_csv_path, index=False)
    print(f"✅ Confusion matrix exported to: {confusion_csv_path}")
    
    # Export threshold boundary analysis
    boundary_tests = [
        ("Edge_Accumulation", 25.4, 0.15),    # Just at accumulation threshold
        ("Edge_Manipulation", 76.5, 0.85),   # Just at manipulation threshold  
        ("Perfect_Baseline", 50.96, 0.50)     # Exact baseline
    ]
    
    boundary_results = []
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
        
        boundary_results.append({
            'test_scenario': test_name,
            'amplification': amplification,
            'amplification_ratio': ratio,
            'directional_coherence': directional_coherence,
            'predicted_phase': boundary_classification.phase,
            'confidence': boundary_classification.confidence,
            'statistical_significance': boundary_classification.statistical_significance
        })
    
    boundary_df = pd.DataFrame(boundary_results)
    boundary_csv_path = f"/Users/jack/IRONFORGE/data/validation/po3_boundary_tests_{timestamp}.csv"
    boundary_df.to_csv(boundary_csv_path, index=False)
    print(f"✅ Boundary tests exported to: {boundary_csv_path}")
    
    print(f"\n🎯 EXPORT COMPLETE!")
    print(f"📁 Files exported to: /Users/jack/IRONFORGE/data/validation/")
    print(f"📊 Total validation samples: {len(results_df)}")
    print(f"📈 Ready for H8 data application and comparison analysis!")
    
    return {
        'main_results_path': main_csv_path,
        'summary_path': summary_csv_path,
        'confusion_matrix_path': confusion_csv_path,
        'boundary_tests_path': boundary_csv_path,
        'overall_accuracy': overall_accuracy,
        'sample_count': len(results_df)
    }

def main():
    """Main export function"""
    export_results = run_comprehensive_validation_export()
    
    print(f"\n📋 Export Summary:")
    print(f"  Main Results: {export_results['main_results_path']}")
    print(f"  Summary Stats: {export_results['summary_path']}")
    print(f"  Confusion Matrix: {export_results['confusion_matrix_path']}")
    print(f"  Boundary Tests: {export_results['boundary_tests_path']}")
    print(f"  Overall Accuracy: {export_results['overall_accuracy']:.1%}")
    print(f"  Sample Count: {export_results['sample_count']}")

if __name__ == "__main__":
    main()