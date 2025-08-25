#!/usr/bin/env python3
"""
IRONFORGE Theory B Statistical Validator
========================================

Rigorous statistical analysis of Theory B claims:
- Archaeological zone event at 14:35:00 (23162.25)
- Session High: 13:58:00 (23252.0) 
- Session Low: 14:53:00 (23115.0)

Theory B Claims:
1. Event positioned 7.55 points from 40% of FINAL range
2. Current range theory shows 30.80 point accuracy
3. This represents statistically significant predictive improvement

Validation Requirements:
- Confidence Level: 0.95
- FDR Correction: q = 0.10
- Permutation Testing: 2000 iterations
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import bootstrap, permutation_test
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TheoryBTestCase:
    """Single archaeological zone test case for Theory B validation"""
    session_name: str
    event_time: str
    event_price: float
    session_high: float
    session_low: float
    session_high_time: str
    session_low_time: str
    claimed_final_accuracy: float
    claimed_current_accuracy: float

class TheoryBStatisticalValidator:
    """Statistical validator for Theory B dimensional positioning claims"""
    
    def __init__(self, confidence_level: float = 0.95, fdr_q: float = 0.10):
        self.confidence_level = confidence_level
        self.fdr_q = fdr_q
        self.alpha = 1 - confidence_level
        
        # TODO(human): Initialize critical session data for 2025-08-05 PM
        # We need to validate the exact measurements claimed
        
    def calculate_archaeological_zone_accuracy(self, test_case: TheoryBTestCase) -> Dict[str, float]:
        """Calculate positioning accuracy for both theories"""
        
        # Final range theory (Theory B)
        final_range = test_case.session_high - test_case.session_low
        final_40_percent_level = test_case.session_low + (0.4 * final_range)
        final_theory_accuracy = abs(test_case.event_price - final_40_percent_level)
        
        # Current range theory (at event time)
        # TODO(human): Implement current range calculation logic
        # This requires determining session range AT THE TIME of the event
        # not the final session range
        
        return {
            'final_theory_accuracy': final_theory_accuracy,
            'final_40_percent_level': final_40_percent_level,
            'final_range': final_range,
            'current_theory_accuracy': 0.0,  # To be implemented
            'current_40_percent_level': 0.0  # To be implemented
        }
    
    def paired_accuracy_test(self, final_errors: np.ndarray, current_errors: np.ndarray) -> Dict[str, float]:
        """Paired t-test for accuracy differences"""
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(current_errors, final_errors)
        
        # Effect size (Cohen's d for paired samples)
        differences = current_errors - final_errors
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxontest(current_errors, final_errors)
        
        return {
            'paired_t_statistic': t_stat,
            'paired_t_pvalue': p_value,
            'cohens_d': effect_size,
            'mean_improvement': np.mean(differences),
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_pvalue': wilcoxon_p
        }
    
    def bootstrap_confidence_intervals(self, final_errors: np.ndarray, current_errors: np.ndarray, 
                                     n_bootstrap: int = 10000) -> Dict[str, Tuple[float, float]]:
        """Bootstrap confidence intervals for effect sizes"""
        
        differences = current_errors - final_errors
        
        def mean_difference(x, axis):
            return np.mean(x, axis=axis)
        
        def effect_size_func(x, axis):
            return np.mean(x, axis=axis) / np.std(x, axis=axis, ddof=1)
        
        # Bootstrap for mean difference
        rng = np.random.default_rng(42)
        bootstrap_result_mean = bootstrap((differences,), mean_difference, 
                                        n_resamples=n_bootstrap, 
                                        confidence_level=self.confidence_level,
                                        random_state=rng)
        
        # Bootstrap for effect size
        bootstrap_result_effect = bootstrap((differences,), effect_size_func,
                                          n_resamples=n_bootstrap,
                                          confidence_level=self.confidence_level,
                                          random_state=rng)
        
        return {
            'mean_difference_ci': (bootstrap_result_mean.confidence_interval.low,
                                 bootstrap_result_mean.confidence_interval.high),
            'effect_size_ci': (bootstrap_result_effect.confidence_interval.low,
                             bootstrap_result_effect.confidence_interval.high)
        }
    
    def permutation_test_accuracy(self, final_errors: np.ndarray, current_errors: np.ndarray,
                                n_permutations: int = 2000) -> Dict[str, float]:
        """Permutation test for accuracy difference significance"""
        
        def statistic(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        
        rng = np.random.default_rng(42)
        perm_result = permutation_test((current_errors, final_errors), statistic,
                                     n_resamples=n_permutations,
                                     alternative='greater',  # Current > Final (improvement)
                                     random_state=rng)
        
        return {
            'permutation_statistic': perm_result.statistic,
            'permutation_pvalue': perm_result.pvalue,
            'n_permutations': n_permutations
        }
    
    def assess_statistical_power(self, final_errors: np.ndarray, current_errors: np.ndarray) -> Dict[str, float]:
        """Assess statistical power and minimum detectable effect"""
        
        n = len(final_errors)
        differences = current_errors - final_errors
        
        # Observed effect size
        observed_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Estimated power for observed effect (post-hoc)
        power_analysis = {
            'sample_size': n,
            'observed_cohens_d': observed_d,
            'observed_mean_difference': np.mean(differences),
            'difference_std': np.std(differences, ddof=1)
        }
        
        return power_analysis
    
    def identify_confounds_and_limitations(self, test_case: TheoryBTestCase) -> List[str]:
        """Identify potential confounds and statistical limitations"""
        
        limitations = []
        
        # Sample size concerns
        limitations.append("CRITICAL: N=1 case study - insufficient for statistical inference")
        
        # Temporal concerns
        event_time = test_case.event_time
        high_time = test_case.session_high_time
        low_time = test_case.session_low_time
        
        # Check temporal ordering
        if high_time < event_time:
            limitations.append(f"Session high established BEFORE event ({high_time} < {event_time})")
        
        if low_time > event_time:
            limitations.append(f"Session low established AFTER event ({low_time} > {event_time})")
        
        # Look-ahead bias
        limitations.append("LOOK-AHEAD BIAS: Theory B uses future information (final range)")
        
        # Multiple comparisons
        limitations.append("Multiple archaeological zones tested - requires FDR correction")
        
        # Selection bias
        limitations.append("Cherry-picking concern: Single favorable case highlighted")
        
        return limitations
    
    def generate_comprehensive_report(self, test_case: TheoryBTestCase) -> Dict:
        """Generate comprehensive statistical assessment"""
        
        # Calculate accuracy measurements
        accuracy_results = self.calculate_archaeological_zone_accuracy(test_case)
        
        # Identify limitations
        limitations = self.identify_confounds_and_limitations(test_case)
        
        # TODO(human): Complete statistical testing once current range calculation is implemented
        
        report = {
            'test_case_summary': {
                'session': test_case.session_name,
                'event_details': {
                    'time': test_case.event_time,
                    'price': test_case.event_price,
                    'claimed_accuracy_final': test_case.claimed_final_accuracy,
                    'claimed_accuracy_current': test_case.claimed_current_accuracy
                },
                'session_extremes': {
                    'high': test_case.session_high,
                    'high_time': test_case.session_high_time,
                    'low': test_case.session_low,
                    'low_time': test_case.session_low_time
                }
            },
            'calculated_measurements': accuracy_results,
            'statistical_limitations': limitations,
            'validation_status': 'INCOMPLETE - awaiting current range calculation',
            'red_team_assessment': {
                'primary_concerns': [
                    "Sample size N=1 prevents meaningful statistical inference",
                    "Look-ahead bias in Theory B using future session extremes", 
                    "Temporal causality violation if using post-event information",
                    "Cherry-picking bias in case selection"
                ],
                'confidence_in_claims': 'VERY LOW - insufficient evidence'
            }
        }
        
        return report

def main():
    """Execute Theory B statistical validation"""
    
    print("🔬 THEORY B STATISTICAL VALIDATOR")
    print("=" * 50)
    
    # Initialize validator
    validator = TheoryBStatisticalValidator()
    
    # Create test case from claims
    test_case = TheoryBTestCase(
        session_name="NY_PM_2025-08-05",
        event_time="14:35:00",
        event_price=23162.25,
        session_high=23252.0,
        session_low=23115.0,
        session_high_time="13:58:00",
        session_low_time="14:53:00",
        claimed_final_accuracy=7.55,
        claimed_current_accuracy=30.80
    )
    
    # Generate comprehensive assessment
    report = validator.generate_comprehensive_report(test_case)
    
    # Display results
    print(f"📊 Session: {report['test_case_summary']['session']}")
    print(f"⚡ Event: {report['test_case_summary']['event_details']['time']} @ {report['test_case_summary']['event_details']['price']}")
    print(f"📈 Session Range: {test_case.session_low} - {test_case.session_high} ({test_case.session_high - test_case.session_low:.2f} points)")
    
    print(f"\n🎯 ACCURACY ANALYSIS:")
    calc_results = report['calculated_measurements']
    print(f"Theory B (Final Range): {calc_results['final_theory_accuracy']:.2f} points")
    print(f"40% Final Level: {calc_results['final_40_percent_level']:.2f}")
    
    print(f"\n⚠️  STATISTICAL LIMITATIONS ({len(report['statistical_limitations'])}):")
    for i, limitation in enumerate(report['statistical_limitations'], 1):
        print(f"{i:2d}. {limitation}")
    
    print(f"\n🔴 RED TEAM ASSESSMENT:")
    for concern in report['red_team_assessment']['primary_concerns']:
        print(f"   • {concern}")
    
    print(f"\n📋 Status: {report['validation_status']}")
    print(f"🎯 Confidence: {report['red_team_assessment']['confidence_in_claims']}")

if __name__ == "__main__":
    main()