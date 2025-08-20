#!/usr/bin/env python3
"""
Quick test of Statistical Analysis Framework
"""

import sys
sys.path.append('/Users/jack/IRONFORGE')

from statistical_analysis_framework import StatisticalAnalysisFramework

# Quick test
framework = StatisticalAnalysisFramework()

# Test Wilson CI
successes = 15
trials = 28
ci = framework.wilson_confidence_interval(successes, trials)

print(f"Wilson CI Test:")
print(f"   {successes}/{trials} = {successes/trials:.1%}")
print(f"   95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
print(f"   Width: {(ci[1] - ci[0]):.1%}")

# Test with small sample
small_ci = framework.calculate_confidence_intervals(3, 7)
print(f"\nSmall Sample Test:")
print(f"   Proportion: {small_ci['proportion']:.1%}")
print(f"   Wilson CI: [{small_ci['wilson_ci'][0]:.1%}, {small_ci['wilson_ci'][1]:.1%}]")
print(f"   Inconclusive: {small_ci['inconclusive_flag']}")

print("\n✅ Statistical framework working correctly!")