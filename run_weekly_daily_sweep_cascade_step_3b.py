#!/usr/bin/env python3
"""
📈 IRONFORGE Weekly→Daily Liquidity Sweep Cascade Analysis (Step 3B) - Execution
================================================================================

Macro Driver Analysis: Weekly dominance verification through cascade patterns

Goal: Verify Weekly dominance by showing sweeps cascade to Daily, then down to PM executions 
with measurable lead/lag and hit-rates.

Key Tests:
1. Sweep Detection: Weekly & Daily events where wick pierces prior swing high/low
2. Cascade Linking: Weekly_sweep → Daily_reaction → PM_execution chains  
3. Quantification: Hit-rates, lead/lag histograms, directional consistency
4. Statistical Tests: Causal ordering, effect size, robustness analysis

Usage:
    python run_weekly_daily_sweep_cascade_step_3b.py
"""

import logging
import sys
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

from ironforge.analysis.weekly_daily_sweep_cascade_analyzer import WeeklyDailySweepCascadeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Execute Weekly→Daily Liquidity Sweep Cascade Analysis (Step 3B)"""
    
    print("📈 IRONFORGE Weekly→Daily Liquidity Sweep Cascade Analysis (Step 3B)")
    print("=" * 80)
    print("Macro Driver Analysis: Weekly dominance verification through cascade patterns")
    print("Goal: Verify Weekly dominance by showing sweeps cascade to Daily → PM executions")
    print()
    
    try:
        # Initialize cascade analyzer
        analyzer = WeeklyDailySweepCascadeAnalyzer()
        
        # Execute comprehensive cascade analysis
        logger.info("Starting Weekly→Daily Sweep Cascade Analysis (Step 3B)...")
        results = analyzer.analyze_weekly_daily_cascades()
        
        if 'error' in results:
            print(f"❌ Step 3B Analysis failed: {results['error']}")
            return 1
        
        print("✅ WEEKLY→DAILY SWEEP CASCADE ANALYSIS COMPLETE")
        print("=" * 55)
        
        # Display results summary
        metadata = results.get('metadata', {})
        results.get('sweep_detection_results', {})
        cascade_analysis = results.get('cascade_analysis', {})
        
        print(f"📊 Sessions Analyzed: {metadata.get('sessions_analyzed', 0)}")
        print(f"🔍 Weekly Sweeps Detected: {metadata.get('weekly_sweeps_detected', 0)}")
        print(f"📈 Daily Sweeps Detected: {metadata.get('daily_sweeps_detected', 0)}")
        print(f"⏰ PM Executions Detected: {metadata.get('pm_executions_detected', 0)}")
        print(f"🔗 Cascade Links Mapped: {metadata.get('cascade_links_mapped', 0)}")
        print()
        
        # Quantification Results
        quantification = cascade_analysis.get('quantification_results', {})
        hit_rates = quantification.get('hit_rates', {})
        
        if hit_rates:
            print("🎯 HIT-RATE ANALYSIS")
            print("-" * 20)
            print(f"P(PM execution | Weekly sweep): {hit_rates.get('pm_execution_given_weekly_sweep', 0):.3f}")
            print(f"P(Daily reaction | Weekly sweep): {hit_rates.get('daily_reaction_given_weekly_sweep', 0):.3f}")
            print(f"P(PM execution | Daily confirmation): {hit_rates.get('pm_execution_given_daily_confirmation', 0):.3f}")
            print(f"Baseline PM rate: {hit_rates.get('baseline_pm_rate', 0):.3f}")
            print()
        
        # Lead/Lag Analysis
        lead_lag = quantification.get('lead_lag_analysis', {})
        if lead_lag:
            print("⏰ LEAD/LAG ANALYSIS")
            print("-" * 20)
            
            weekly_to_daily = lead_lag.get('weekly_to_daily_delays', {})
            print("Weekly→Daily transmission delay:")
            print(f"  Mean: {weekly_to_daily.get('mean', 0):.1f} hours")
            print(f"  Median: {weekly_to_daily.get('median', 0):.1f} hours")
            print(f"  Std: {weekly_to_daily.get('std', 0):.1f} hours")
            
            daily_to_pm = lead_lag.get('daily_to_pm_delays', {})
            print("Daily→PM transmission delay:")
            print(f"  Mean: {daily_to_pm.get('mean', 0):.1f} hours")
            print(f"  Median: {daily_to_pm.get('median', 0):.1f} hours")
            print(f"  Std: {daily_to_pm.get('std', 0):.1f} hours")
            print()
        
        # Directional Consistency
        directional = quantification.get('directional_consistency', {})
        if directional:
            print("🧭 DIRECTIONAL CONSISTENCY")
            print("-" * 25)
            print(f"Consistent cascades: {directional.get('consistent_cascades', 0)}")
            print(f"Total measurable cascades: {directional.get('total_measurable_cascades', 0)}")
            print(f"Consistency rate: {directional.get('consistency_rate', 0):.3f}")
            print(f"Significance: {directional.get('consistency_significance', 'UNKNOWN')}")
            print()
        
        # Statistical Validation
        statistical_tests = cascade_analysis.get('statistical_validation', {})
        
        # Causal Ordering Test
        causal_test = statistical_tests.get('causal_ordering_test', {})
        if causal_test and 'error' not in causal_test:
            print("🧪 CAUSAL ORDERING TEST (Permutation)")
            print("-" * 35)
            print(f"Observed cascades: {causal_test.get('observed_cascades', 0)}")
            print(f"Permutation mean: {causal_test.get('permutation_mean', 0):.1f}")
            print(f"Permutation std: {causal_test.get('permutation_std', 0):.1f}")
            print(f"P-value: {causal_test.get('p_value', 1.0):.6f}")
            print(f"Significant: {'✓ YES' if causal_test.get('significant', False) else '✗ NO'}")
            print(f"Z-score: {causal_test.get('z_score', 0):.3f}")
            print()
        
        # Effect Size Analysis
        effect_size = statistical_tests.get('effect_size_analysis', {})
        if effect_size and 'error' not in effect_size:
            print("📊 EFFECT SIZE ANALYSIS (Cohen's h)")
            print("-" * 32)
            print(f"Hit rate with Weekly: {effect_size.get('hit_rate_with_weekly', 0):.3f}")
            print(f"Baseline hit rate: {effect_size.get('baseline_hit_rate', 0):.3f}")
            print(f"Cohen's h: {effect_size.get('cohens_h', 0):.3f}")
            print(f"Effect magnitude: {effect_size.get('effect_magnitude', 'unknown')}")
            print(f"Uplift ratio: {effect_size.get('uplift_ratio', 1.0):.2f}x")
            print(f"Absolute improvement: {effect_size.get('absolute_improvement', 0):.3f}")
            print()
        
        # Robustness Analysis
        robustness = statistical_tests.get('robustness_analysis', {})
        if robustness and 'error' not in robustness:
            print("🔍 ROBUSTNESS ANALYSIS")
            print("-" * 20)
            print(f"Parameter combinations tested: {robustness.get('parameter_combinations_tested', 0)}")
            
            cascade_range = robustness.get('cascade_count_range', {})
            print(f"Cascade count range: {cascade_range.get('min', 0)}-{cascade_range.get('max', 0)}")
            print(f"Mean: {cascade_range.get('mean', 0):.1f}")
            print(f"Stability coefficient: {robustness.get('stability_coefficient', 0):.3f}")
            print(f"Stability assessment: {robustness.get('stability_assessment', 'UNKNOWN')}")
            print()
        
        # Bootstrap Confidence Intervals
        bootstrap = statistical_tests.get('bootstrap_confidence_intervals', {})
        if bootstrap and 'error' not in bootstrap:
            print("🎯 BOOTSTRAP CONFIDENCE INTERVALS")
            print("-" * 33)
            
            hit_rate_ci = bootstrap.get('hit_rate_ci_95', {})
            print(f"Hit rate 95% CI: [{hit_rate_ci.get('lower', 0):.3f}, {hit_rate_ci.get('upper', 0):.3f}]")
            print(f"Mean: {hit_rate_ci.get('mean', 0):.3f}")
            
            strength_ci = bootstrap.get('cascade_strength_ci_95', {})
            print(f"Cascade strength 95% CI: [{strength_ci.get('lower', 0):.3f}, {strength_ci.get('upper', 0):.3f}]")
            print(f"Mean: {strength_ci.get('mean', 0):.3f}")
            print()
        
        # Discovery Insights
        insights = results.get('discovery_insights', {})
        if insights:
            print("🔍 DISCOVERY INSIGHTS")
            print("-" * 20)
            
            # Weekly Dominance Assessment
            dominance = insights.get('weekly_dominance_assessment', {})
            if dominance:
                print(f"Weekly dominance confirmed: {'✓ YES' if dominance.get('dominance_confirmed', False) else '✗ NO'}")
                print(f"Hit rate uplift: {dominance.get('hit_rate_uplift', 1.0):.2f}x")
                print(f"Causal ordering significant: {'✓ YES' if dominance.get('causal_ordering_significant', False) else '✗ NO'}")
                print(f"Strength category: {dominance.get('strength_category', 'UNKNOWN')}")
                print()
            
            # Cascade Transmission Efficiency
            transmission = insights.get('cascade_transmission_efficiency', {})
            if transmission:
                print("📡 CASCADE TRANSMISSION EFFICIENCY")
                print("-" * 33)
                print(f"Weekly→Daily success: {transmission.get('weekly_to_daily_success_rate', 0):.3f}")
                print(f"Daily→PM success: {transmission.get('daily_to_pm_success_rate', 0):.3f}")
                print(f"End-to-end transmission: {transmission.get('end_to_end_transmission', 0):.3f}")
                print(f"Transmission quality: {transmission.get('transmission_quality', 'UNKNOWN')}")
                print()
            
            # Statistical Validation Summary
            stat_summary = insights.get('statistical_validation_summary', {})
            if stat_summary:
                print("📊 STATISTICAL VALIDATION SUMMARY")
                print("-" * 34)
                print(f"Causal ordering p-value: {stat_summary.get('causal_ordering_p_value', 1.0):.6f}")
                print(f"Effect size magnitude: {stat_summary.get('effect_size_magnitude', 'unknown')}")
                print(f"Robustness confirmed: {'✓ YES' if stat_summary.get('robustness_confirmed', False) else '✗ NO'}")
                print(f"Overall significance: {stat_summary.get('overall_significance', 'UNKNOWN')}")
                print()
            
            # Key Discoveries
            key_discoveries = insights.get('key_discoveries', [])
            if key_discoveries:
                print("🏆 KEY DISCOVERIES")
                print("-" * 16)
                for i, discovery in enumerate(key_discoveries, 1):
                    print(f"  {i}. {discovery}")
                print()
        
        print("🎉 STEP 3B ANALYSIS COMPLETE")
        print("=" * 30)
        print("Weekly→Daily liquidity sweep cascade patterns analyzed with statistical validation")
        print("Macro driver verification: Weekly dominance through cascade transmission")
        print("Complementary to Step 3A micro-level FPFVG redelivery analysis")
        print()
        print("📁 Results saved to discoveries/ directory")
        print("🔬 Statistical tests completed: Causal ordering, effect size, robustness")
        print("📈 Comprehensive cascade framework now complete")
        
        return 0
        
    except Exception as e:
        logger.error(f"Step 3B cascade analysis execution failed: {e}")
        print(f"❌ Execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)