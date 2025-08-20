#!/usr/bin/env python3
"""
Enhanced Statistical Framework - Phase 5
Sample-size rules, CI validation, and merge logic for explore-only analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
from collections import defaultdict
import warnings

@dataclass
class AnalysisSlice:
    """Statistical analysis slice with validation"""
    name: str
    count: int
    successes: int
    percentage: float
    wilson_ci: Tuple[float, float]
    bootstrap_ci: Optional[Tuple[float, float]]
    inconclusive_flag: bool
    merged_buckets: List[str]
    median_time_60: Optional[float] = None
    median_time_80: Optional[float] = None

class EnhancedStatisticalAnalyzer:
    """Enhanced statistical analyzer with Phase 5 specifications"""
    
    def __init__(self, min_sample_size: int = 5, ci_width_threshold: float = 0.30,
                 bootstrap_samples: int = 1000, confidence_level: float = 0.95):
        self.min_sample_size = min_sample_size
        self.ci_width_threshold = ci_width_threshold
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.merge_log = []
        
    def analyze_slice_with_validation(self, data: List[Dict], 
                                    slice_key: str, 
                                    success_condition: str = "e3_accel",
                                    time_fields: List[str] = ["time_to_60", "time_to_80"]) -> Dict[str, AnalysisSlice]:
        """Analyze data slices with merge rules and CI validation"""
        
        # Group data by slice_key
        grouped_data = defaultdict(list)
        for item in data:
            slice_value = str(item.get(slice_key, 'unknown'))
            grouped_data[slice_value].append(item)
        
        # Apply merge rules for small samples
        merged_groups = self._apply_merge_rules(grouped_data)
        
        # Generate analysis slices
        analysis_results = {}
        
        for group_name, group_data in merged_groups.items():
            count = len(group_data)
            
            # Count successes based on condition
            successes = sum(1 for item in group_data 
                          if self._evaluate_success_condition(item, success_condition))
            
            # Calculate percentage and CIs
            if count > 0:
                percentage = (successes / count) * 100
                wilson_ci = self._wilson_confidence_interval(successes, count)
                
                # Bootstrap CI for small samples or extreme percentages
                bootstrap_ci = None
                if count < 20 or percentage < 5 or percentage > 95:
                    bootstrap_ci = self._bootstrap_confidence_interval(group_data, success_condition)
                
                # Check if inconclusive
                ci_width = wilson_ci[1] - wilson_ci[0]
                inconclusive_flag = ci_width > self.ci_width_threshold
                
                # Calculate median times
                median_times = self._calculate_median_times(group_data, time_fields)
                
                # Determine merged buckets
                merged_buckets = [group_name] if group_name in grouped_data else \
                               [key for key in grouped_data.keys() if key not in merged_groups]
                
                analysis_slice = AnalysisSlice(
                    name=group_name,
                    count=count,
                    successes=successes,
                    percentage=percentage,
                    wilson_ci=wilson_ci,
                    bootstrap_ci=bootstrap_ci,
                    inconclusive_flag=inconclusive_flag,
                    merged_buckets=merged_buckets,
                    median_time_60=median_times.get("time_to_60"),
                    median_time_80=median_times.get("time_to_80")
                )
                
                analysis_results[group_name] = analysis_slice
            
        return analysis_results
    
    def _apply_merge_rules(self, grouped_data: Dict[str, List]) -> Dict[str, List]:
        """Apply merge rules for samples with n < min_sample_size"""
        
        merged_groups = {}
        small_buckets = []
        
        # Identify buckets that meet minimum sample size
        for bucket_name, bucket_data in grouped_data.items():
            if len(bucket_data) >= self.min_sample_size:
                merged_groups[bucket_name] = bucket_data
            else:
                small_buckets.append((bucket_name, bucket_data))
                
        # Merge small buckets
        if small_buckets:
            # Strategy 1: Merge to "Other" if multiple small buckets
            if len(small_buckets) > 1:
                other_data = []
                merged_names = []
                for bucket_name, bucket_data in small_buckets:
                    other_data.extend(bucket_data)
                    merged_names.append(bucket_name)
                    
                merged_groups["Other"] = other_data
                self.merge_log.append(f"Merged {merged_names} → 'Other' (n={len(other_data)})")
                
            # Strategy 2: Merge single small bucket to nearest large bucket
            elif len(small_buckets) == 1 and merged_groups:
                bucket_name, bucket_data = small_buckets[0]
                
                # Find nearest bucket by name similarity or merge to largest
                nearest_bucket = max(merged_groups.keys(), key=lambda x: len(merged_groups[x]))
                merged_groups[nearest_bucket].extend(bucket_data)
                
                self.merge_log.append(f"Merged '{bucket_name}' → '{nearest_bucket}' (n<{self.min_sample_size})")
                
            # Strategy 3: Keep single small bucket if no others exist
            else:
                bucket_name, bucket_data = small_buckets[0]
                merged_groups[bucket_name] = bucket_data
                self.merge_log.append(f"Kept '{bucket_name}' as-is (only bucket, n={len(bucket_data)})")
        
        return merged_groups
    
    def _evaluate_success_condition(self, item: Dict, condition: str) -> bool:
        """Evaluate success condition for statistical analysis"""
        
        if condition == "e3_accel":
            return item.get("path_classification", "").lower() in ["e3_accel", "accel", "acceleration"]
        elif condition == "e2_mr":
            return item.get("path_classification", "").lower() in ["e2_mr", "mr", "mean_revert"]
        elif condition == "e1_cont":
            return item.get("path_classification", "").lower() in ["e1_cont", "cont", "continuation"]
        elif condition == "h1_breakout":
            return bool(item.get("h1_breakout", False))
        elif condition == "second_rd":
            return bool(item.get("second_rd", False))
        else:
            # Custom field evaluation
            return bool(item.get(condition, False))
    
    def _wilson_confidence_interval(self, successes: int, trials: int) -> Tuple[float, float]:
        """Wilson confidence interval with enhanced precision"""
        if trials == 0:
            return (0.0, 0.0)
            
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        p = successes / trials
        n = trials
        
        # Wilson score interval formula
        center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / (1 + z**2 / n)
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return (lower, upper)
    
    def _bootstrap_confidence_interval(self, data: List[Dict], 
                                     success_condition: str) -> Tuple[float, float]:
        """Bootstrap confidence interval for small samples"""
        
        if len(data) < 2:
            return (0.0, 1.0)
        
        bootstrap_stats = []
        
        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            resampled = np.random.choice(len(data), size=len(data), replace=True)
            bootstrap_data = [data[i] for i in resampled]
            
            # Calculate success rate for this bootstrap sample
            successes = sum(1 for item in bootstrap_data 
                          if self._evaluate_success_condition(item, success_condition))
            success_rate = successes / len(bootstrap_data)
            bootstrap_stats.append(success_rate)
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        
        return (float(lower_bound), float(upper_bound))
    
    def _calculate_median_times(self, data: List[Dict], time_fields: List[str]) -> Dict[str, Optional[float]]:
        """Calculate median times for time-to-event analysis"""
        
        median_times = {}
        
        for field in time_fields:
            values = [item.get(field) for item in data if item.get(field) is not None]
            
            if values:
                median_times[field] = float(np.median(values))
            else:
                median_times[field] = None
                
        return median_times
    
    def generate_analysis_table(self, analysis_results: Dict[str, AnalysisSlice],
                              title: str = "Analysis Results") -> str:
        """Generate formatted analysis table with CI validation"""
        
        output = [f"\n📊 {title}"]
        output.append("=" * len(title))
        
        # Header
        header = f"{'Slice':<20} {'n':<5} {'%':<8} {'Wilson CI':<20} {'Bootstrap CI':<20} {'Flag':<6} {'Median t60':<12} {'Median t80':<12}"
        output.append(header)
        output.append("-" * len(header))
        
        # Sort by count (descending)
        sorted_slices = sorted(analysis_results.items(), 
                             key=lambda x: x[1].count, reverse=True)
        
        for slice_name, slice_data in sorted_slices:
            # Format percentage display
            if slice_data.count < self.min_sample_size:
                pct_display = f"({slice_data.percentage:.1f}%)"  # Parentheses for small n
            else:
                pct_display = f"{slice_data.percentage:.1f}%"
            
            # Format CIs
            wilson_ci_str = f"({slice_data.wilson_ci[0]:.3f}, {slice_data.wilson_ci[1]:.3f})"
            
            bootstrap_ci_str = ""
            if slice_data.bootstrap_ci:
                bootstrap_ci_str = f"({slice_data.bootstrap_ci[0]:.3f}, {slice_data.bootstrap_ci[1]:.3f})"
            else:
                bootstrap_ci_str = "—"
            
            # Format flag
            flag = "INC" if slice_data.inconclusive_flag else "—"
            
            # Format median times
            t60_str = f"{slice_data.median_time_60:.1f}m" if slice_data.median_time_60 else "—"
            t80_str = f"{slice_data.median_time_80:.1f}m" if slice_data.median_time_80 else "—"
            
            row = f"{slice_name:<20} {slice_data.count:<5} {pct_display:<8} {wilson_ci_str:<20} {bootstrap_ci_str:<20} {flag:<6} {t60_str:<12} {t80_str:<12}"
            output.append(row)
        
        # Add merge log
        if self.merge_log:
            output.append(f"\n📝 Merge Log:")
            for log_entry in self.merge_log:
                output.append(f"  • {log_entry}")
        
        return "\n".join(output)

class VolatilityCalculator:
    """Calculate volatility multipliers using session ATR baseline"""
    
    def __init__(self, atr_window: int = 14):
        self.atr_window = atr_window
        
    def calculate_volatility_multiplier(self, session_data: Dict, 
                                      rd40_event_time: str,
                                      analysis_window_mins: int = 90) -> float:
        """Calculate volatility multiplier for RD@40 event"""
        
        try:
            # Get session ATR (if available)
            session_atr = session_data.get("session_atr", 1.0)
            
            # Get price data around RD@40 event
            event_time = pd.to_datetime(rd40_event_time)
            
            # Extract price movements in analysis window
            # This would integrate with actual price data
            # For now, simulate based on energy_density
            energy_density = session_data.get("energy_density", 0.5)
            
            # Simulate realized volatility based on energy density
            baseline_vol = session_atr
            realized_vol = baseline_vol * (1 + energy_density)
            
            # Calculate multiplier
            volatility_multiplier = realized_vol / baseline_vol if baseline_vol > 0 else 1.0
            
            return float(volatility_multiplier)
            
        except Exception as e:
            print(f"⚠️ Volatility calculation error: {e}")
            return 1.0  # Default multiplier

def demo_enhanced_statistical_analysis():
    """Demonstrate enhanced statistical analysis with Phase 5 specifications"""
    
    print("🧪 DEMO: Enhanced Statistical Framework")
    print("=" * 50)
    
    # Sample data with various slice sizes
    sample_data = [
        {"slice_key": "monday", "path_classification": "e3_accel", "time_to_60": 12.5, "time_to_80": 25.0},
        {"slice_key": "monday", "path_classification": "e2_mr", "time_to_60": 8.0, "time_to_80": 15.0},
        {"slice_key": "tuesday", "path_classification": "e3_accel", "time_to_60": 15.0, "time_to_80": 30.0},
        {"slice_key": "tuesday", "path_classification": "e3_accel", "time_to_60": 18.0, "time_to_80": 35.0},
        {"slice_key": "tuesday", "path_classification": "e3_accel", "time_to_60": 20.0, "time_to_80": 40.0},
        {"slice_key": "tuesday", "path_classification": "e2_mr", "time_to_60": 10.0, "time_to_80": 20.0},
        {"slice_key": "wednesday", "path_classification": "e3_accel", "time_to_60": 14.0, "time_to_80": 28.0},
        {"slice_key": "thursday", "path_classification": "e2_mr", "time_to_60": 6.0, "time_to_80": 12.0},
        {"slice_key": "friday", "path_classification": "e3_accel", "time_to_60": 16.0, "time_to_80": 32.0},
        # Small samples for merge testing
        {"slice_key": "weekend", "path_classification": "e3_accel", "time_to_60": 22.0, "time_to_80": 45.0},
    ]
    
    # Test enhanced statistical analysis
    analyzer = EnhancedStatisticalAnalyzer(min_sample_size=3)
    
    results = analyzer.analyze_slice_with_validation(
        sample_data, 
        slice_key="slice_key", 
        success_condition="e3_accel"
    )
    
    # Generate analysis table
    table = analyzer.generate_analysis_table(results, "Day of Week E3_ACCEL Analysis")
    print(table)
    
    print(f"\n📈 Statistical Insights:")
    for slice_name, slice_data in results.items():
        ci_width = slice_data.wilson_ci[1] - slice_data.wilson_ci[0]
        print(f"  • {slice_name}: {slice_data.percentage:.1f}% (CI width: {ci_width:.3f})")
        
        if slice_data.inconclusive_flag:
            print(f"    ⚠️ Inconclusive (CI width > 30pp)")
        
        if slice_data.bootstrap_ci:
            print(f"    🔄 Bootstrap CI: ({slice_data.bootstrap_ci[0]:.3f}, {slice_data.bootstrap_ci[1]:.3f})")

if __name__ == "__main__":
    demo_enhanced_statistical_analysis()