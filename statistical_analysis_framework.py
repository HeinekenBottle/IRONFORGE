#!/usr/bin/env python3
"""
Statistical Analysis Framework for Experiment E Day & News Analysis
Implements Wilson CI, Bootstrap CI, and sample size management
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
import math
from collections import defaultdict

class StatisticalAnalysisFramework:
    """Comprehensive statistical analysis for Experiment E patterns"""
    
    def __init__(self):
        self.ci_methods = ['wilson', 'bootstrap']
        self.min_sample_size = 5
        self.inconclusive_threshold = 0.30  # 30% CI width threshold
        self.bootstrap_samples = 1000
    
    def wilson_confidence_interval(self, successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Wilson confidence interval for binomial proportion
        More robust than normal approximation for small samples
        """
        if trials == 0:
            return (0.0, 0.0)
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = successes / trials
        n = trials
        
        # Wilson interval formula
        center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / (1 + z**2 / n)
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return (lower, upper)
    
    def bootstrap_confidence_interval(self, data: List[int], trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for proportion
        Useful for very small samples or edge cases
        """
        if not data or trials == 0:
            return (0.0, 0.0)
        
        # Create bootstrap samples
        bootstrap_proportions = []
        
        for _ in range(self.bootstrap_samples):
            # Sample with replacement
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            proportion = np.sum(bootstrap_sample) / trials
            bootstrap_proportions.append(proportion)
        
        # Calculate percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_proportions, lower_percentile)
        upper = np.percentile(bootstrap_proportions, upper_percentile)
        
        return (lower, upper)
    
    def calculate_confidence_intervals(self, successes: int, trials: int, 
                                     confidence: float = 0.95) -> Dict[str, Any]:
        """Calculate confidence intervals using multiple methods"""
        if trials == 0:
            return {
                'proportion': 0.0,
                'wilson_ci': (0.0, 0.0),
                'bootstrap_ci': None,
                'ci_width_pct': 0.0,
                'inconclusive_flag': False,
                'recommended_method': 'wilson'
            }
        
        proportion = successes / trials
        
        # Wilson CI (always calculate)
        wilson_ci = self.wilson_confidence_interval(successes, trials, confidence)
        wilson_width = wilson_ci[1] - wilson_ci[0]
        
        # Bootstrap CI for small samples or edge cases
        bootstrap_ci = None
        if trials < 20 or proportion < 0.05 or proportion > 0.95:
            # Create binary data for bootstrap
            binary_data = [1] * successes + [0] * (trials - successes)
            bootstrap_ci = self.bootstrap_confidence_interval(binary_data, trials, confidence)
        
        # Determine recommended method and inconclusive flag
        ci_width_pct = wilson_width
        inconclusive_flag = ci_width_pct > self.inconclusive_threshold
        recommended_method = 'bootstrap' if bootstrap_ci and trials < 10 else 'wilson'
        
        return {
            'proportion': proportion,
            'wilson_ci': wilson_ci,
            'bootstrap_ci': bootstrap_ci,
            'ci_width_pct': ci_width_pct,
            'inconclusive_flag': inconclusive_flag,
            'recommended_method': recommended_method,
            'sample_size': trials
        }
    
    def manage_sample_sizes(self, data_dict: Dict[str, Dict[str, int]], 
                           merge_threshold: int = None) -> Dict[str, Any]:
        """
        Manage sample sizes by merging small categories and flagging insufficient data
        """
        if merge_threshold is None:
            merge_threshold = self.min_sample_size
        
        managed_data = {}
        merge_candidates = {}
        flagged_insufficient = {}
        
        for category, counts in data_dict.items():
            total_count = sum(counts.values())
            
            if total_count >= merge_threshold:
                managed_data[category] = counts
            else:
                if total_count > 0:
                    merge_candidates[category] = counts
                    flagged_insufficient[category] = {
                        'reason': 'insufficient_sample_size',
                        'actual_size': total_count,
                        'required_size': merge_threshold
                    }
        
        # Create "Other" category for merge candidates if any exist
        if merge_candidates:
            other_counts = defaultdict(int)
            for category, counts in merge_candidates.items():
                for key, value in counts.items():
                    other_counts[key] += value
            
            if sum(other_counts.values()) >= merge_threshold:
                managed_data['Other'] = dict(other_counts)
        
        return {
            'managed_data': managed_data,
            'flagged_insufficient': flagged_insufficient,
            'merge_candidates': merge_candidates,
            'total_categories': len(data_dict),
            'viable_categories': len(managed_data)
        }
    
    def analyze_path_distribution(self, path_counts: Dict[str, int], 
                                 category_name: str = "Category") -> Dict[str, Any]:
        """Comprehensive analysis of path distribution with statistical rigor"""
        total_events = sum(path_counts.values())
        
        if total_events == 0:
            return {
                'category_name': category_name,
                'total_events': 0,
                'viable_analysis': False,
                'reason': 'no_events'
            }
        
        # Calculate proportions and confidence intervals
        path_analysis = {}
        
        for path_type, count in path_counts.items():
            ci_analysis = self.calculate_confidence_intervals(count, total_events)
            
            path_analysis[path_type] = {
                'count': count,
                'percentage': ci_analysis['proportion'] * 100,
                'wilson_ci_lower': ci_analysis['wilson_ci'][0] * 100,
                'wilson_ci_upper': ci_analysis['wilson_ci'][1][1] * 100,
                'ci_width_pct': ci_analysis['ci_width_pct'] * 100,
                'inconclusive_flag': ci_analysis['inconclusive_flag'],
                'recommended_method': ci_analysis['recommended_method']
            }
            
            # Add bootstrap CI if available
            if ci_analysis['bootstrap_ci']:
                path_analysis[path_type].update({
                    'bootstrap_ci_lower': ci_analysis['bootstrap_ci'][0] * 100,
                    'bootstrap_ci_upper': ci_analysis['bootstrap_ci'][1] * 100
                })
        
        # Identify dominant path
        dominant_path = max(path_counts.items(), key=lambda x: x[1])
        
        # Overall analysis quality assessment
        inconclusive_paths = sum(1 for path_data in path_analysis.values() 
                               if path_data['inconclusive_flag'])
        
        analysis_quality = {
            'total_events': total_events,
            'viable_analysis': total_events >= self.min_sample_size,
            'dominant_path': dominant_path[0],
            'dominant_path_percentage': (dominant_path[1] / total_events) * 100,
            'inconclusive_paths': inconclusive_paths,
            'analysis_confidence': 'high' if inconclusive_paths == 0 and total_events >= 20 else 
                                  'medium' if inconclusive_paths <= 1 and total_events >= 10 else 'low'
        }
        
        return {
            'category_name': category_name,
            'total_events': total_events,
            'path_analysis': path_analysis,
            'analysis_quality': analysis_quality,
            'viable_analysis': analysis_quality['viable_analysis']
        }
    
    def validate_hypotheses(self, hypothesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate specific hypotheses with statistical significance testing"""
        hypothesis_results = {}
        
        for hypothesis_name, data in hypothesis_data.items():
            target_path = data.get('target_path')
            target_percentage = data.get('target_percentage', 50.0)  # Default 50%
            actual_count = data.get('actual_count', 0)
            total_count = data.get('total_count', 0)
            
            if total_count == 0:
                hypothesis_results[hypothesis_name] = {
                    'hypothesis': hypothesis_name,
                    'target_path': target_path,
                    'target_percentage': target_percentage,
                    'actual_percentage': 0.0,
                    'validated': False,
                    'reason': 'no_data'
                }
                continue
            
            actual_percentage = (actual_count / total_count) * 100
            
            # Statistical test: one-sample proportion test
            target_proportion = target_percentage / 100
            
            # Use Wilson CI to test if target is within confidence interval
            ci_analysis = self.calculate_confidence_intervals(actual_count, total_count)
            wilson_lower = ci_analysis['wilson_ci'][0] * 100
            wilson_upper = ci_analysis['wilson_ci'][1] * 100
            
            # Hypothesis is validated if target is within CI or actual exceeds target
            target_in_ci = wilson_lower <= target_percentage <= wilson_upper
            exceeds_target = actual_percentage >= target_percentage
            
            # Z-test for proportion
            z_stat = (actual_count/total_count - target_proportion) / math.sqrt(
                target_proportion * (1 - target_proportion) / total_count
            ) if total_count > 0 else 0
            
            p_value = 1 - stats.norm.cdf(abs(z_stat))  # One-tailed test
            
            hypothesis_results[hypothesis_name] = {
                'hypothesis': hypothesis_name,
                'target_path': target_path,
                'target_percentage': target_percentage,
                'actual_percentage': actual_percentage,
                'actual_count': actual_count,
                'total_count': total_count,
                'wilson_ci': (wilson_lower, wilson_upper),
                'target_in_ci': target_in_ci,
                'exceeds_target': exceeds_target,
                'validated': exceeds_target or target_in_ci,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'confidence_level': ci_analysis['recommended_method']
            }
        
        return hypothesis_results
    
    def create_summary_table(self, analysis_results: Dict[str, Any]) -> str:
        """Create formatted summary table for analysis results"""
        lines = []
        lines.append("📊 STATISTICAL ANALYSIS SUMMARY")
        lines.append("=" * 80)
        
        # Header
        header = f"{'Category':<15} {'Events':<8} {'Path':<10} {'%':<6} {'CI Lower':<9} {'CI Upper':<9} {'Quality':<8}"
        lines.append(header)
        lines.append("-" * 80)
        
        for category_name, result in analysis_results.items():
            if not result.get('viable_analysis', False):
                continue
            
            total_events = result['total_events']
            quality = result['analysis_quality']['analysis_confidence']
            
            # Find dominant path
            path_analysis = result.get('path_analysis', {})
            
            for path_type, path_data in path_analysis.items():
                if path_data['count'] > 0:  # Only show paths with events
                    percentage = path_data['percentage']
                    ci_lower = path_data['wilson_ci_lower']
                    ci_upper = path_data['wilson_ci_upper']
                    
                    line = f"{category_name:<15} {total_events:<8} {path_type:<10} {percentage:<6.1f} {ci_lower:<9.1f} {ci_upper:<9.1f} {quality:<8}"
                    lines.append(line)
                    category_name = ""  # Only show category name once
        
        lines.append("=" * 80)
        lines.append("Quality: high (n≥20, narrow CIs), medium (n≥10), low (n<10 or wide CIs)")
        
        return "\n".join(lines)

def demo_statistical_framework():
    """Demonstrate the statistical analysis framework"""
    print("🧮 STATISTICAL ANALYSIS FRAMEWORK DEMO")
    print("=" * 60)
    
    framework = StatisticalAnalysisFramework()
    
    # Sample data for testing
    sample_data = {
        'Monday': {'E2_MR': 15, 'E3_ACCEL': 8, 'E1_CONT': 2, 'UNKNOWN': 3},
        'Tuesday': {'E2_MR': 6, 'E3_ACCEL': 18, 'E1_CONT': 1, 'UNKNOWN': 2},  
        'Wednesday': {'E2_MR': 12, 'E3_ACCEL': 11, 'E1_CONT': 0, 'UNKNOWN': 4},
        'Thursday': {'E2_MR': 4, 'E3_ACCEL': 2, 'E1_CONT': 0, 'UNKNOWN': 1},  # Small sample
        'Friday': {'E2_MR': 8, 'E3_ACCEL': 5, 'E1_CONT': 1, 'UNKNOWN': 2}
    }
    
    # Test sample size management
    print("\n📏 SAMPLE SIZE MANAGEMENT:")
    managed = framework.manage_sample_sizes(sample_data, merge_threshold=5)
    
    print(f"   Original categories: {managed['total_categories']}")
    print(f"   Viable categories: {managed['viable_categories']}")
    print(f"   Flagged insufficient: {list(managed['flagged_insufficient'].keys())}")
    
    # Analyze each viable category
    print("\n📈 PATH DISTRIBUTION ANALYSIS:")
    analysis_results = {}
    
    for category, path_counts in managed['managed_data'].items():
        result = framework.analyze_path_distribution(path_counts, category)
        analysis_results[category] = result
        
        print(f"\n🗓️ {category}:")
        print(f"   Total events: {result['total_events']}")
        print(f"   Analysis quality: {result['analysis_quality']['analysis_confidence']}")
        print(f"   Dominant path: {result['analysis_quality']['dominant_path']} "
              f"({result['analysis_quality']['dominant_path_percentage']:.1f}%)")
    
    # Create summary table
    print("\n" + framework.create_summary_table(analysis_results))
    
    # Test hypothesis validation
    print("\n🔬 HYPOTHESIS VALIDATION:")
    
    hypotheses = {
        'Tuesday_ACCEL_Bias': {
            'target_path': 'E3_ACCEL',
            'target_percentage': 60.0,
            'actual_count': sample_data['Tuesday']['E3_ACCEL'],
            'total_count': sum(sample_data['Tuesday'].values())
        },
        'Monday_MR_Bias': {
            'target_path': 'E2_MR', 
            'target_percentage': 50.0,
            'actual_count': sample_data['Monday']['E2_MR'],
            'total_count': sum(sample_data['Monday'].values())
        }
    }
    
    hypothesis_results = framework.validate_hypotheses(hypotheses)
    
    for hyp_name, result in hypothesis_results.items():
        validated = "✅ VALIDATED" if result['validated'] else "❌ NOT VALIDATED"
        print(f"   {hyp_name}: {validated}")
        print(f"      Target: {result['target_percentage']:.1f}%, "
              f"Actual: {result['actual_percentage']:.1f}%")
        print(f"      CI: [{result['wilson_ci'][0]:.1f}%, {result['wilson_ci'][1]:.1f}%]")
        print(f"      Significant: {'Yes' if result['significant'] else 'No'} (p={result['p_value']:.3f})")
    
    return analysis_results

if __name__ == "__main__":
    demo_statistical_framework()