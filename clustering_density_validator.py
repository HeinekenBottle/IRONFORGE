#!/usr/bin/env python3
"""
Clustering Density Validator - Red-Team Analysis of 16-172x Claims
Validates the "16-172x clustering density vs background noise" claims with volatility controls
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy import stats
from scipy.stats import poisson, binom, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class ClusteringDensityValidator:
    """Red-team validator specifically for 16-172x clustering density claims"""
    
    def __init__(self):
        self.gauntlet_data_path = Path("data/gauntlet_analysis")
        self.results = {
            'validation_timestamp': datetime.now().isoformat(),
            'original_claims': {
                'range': '16-172x',
                'description': 'clustering density vs background noise',
                'source': 'gauntlet_resonance_framework.py'
            },
            'tests_performed': [],
            'verdict': None
        }
    
    def load_resonance_data(self) -> Dict[str, Any]:
        """Load the resonance analysis data"""
        resonance_file = self.gauntlet_data_path / "resonance_analysis_20250821_224136.json"
        
        if not resonance_file.exists():
            raise FileNotFoundError(f"Resonance analysis file not found: {resonance_file}")
        
        with open(resonance_file, 'r') as f:
            return json.load(f)
    
    def extract_clustering_metrics(self, resonance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actual clustering metrics from resonance data"""
        
        session_results = resonance_data.get('session_results', {})
        
        clustering_metrics = {
            'total_sessions': len(session_results),
            'sessions_with_gauntlets': 0,
            'total_gauntlet_events': 0,
            'total_resonance_events': 0,
            'session_densities': [],
            'event_distributions': {},
            'temporal_windows': {}
        }
        
        for session_name, session_data in session_results.items():
            complete_sequences = session_data.get('complete_sequences', 0)
            resonance_events = session_data.get('resonance_events', {})
            
            if complete_sequences > 0:
                clustering_metrics['sessions_with_gauntlets'] += 1
                clustering_metrics['total_gauntlet_events'] += complete_sequences
                
                # Count resonance events by category
                total_session_resonance = 0
                for category, events in resonance_events.items():
                    event_count = len(events) if isinstance(events, list) else 0
                    total_session_resonance += event_count
                    
                    if category not in clustering_metrics['event_distributions']:
                        clustering_metrics['event_distributions'][category] = []
                    clustering_metrics['event_distributions'][category].append(event_count)
                
                clustering_metrics['total_resonance_events'] += total_session_resonance
                
                # Calculate session-level density (events per gauntlet)
                if complete_sequences > 0:
                    session_density = total_session_resonance / complete_sequences
                    clustering_metrics['session_densities'].append(session_density)
        
        return clustering_metrics
    
    def calculate_background_baseline(self, clustering_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate expected background event rate (null hypothesis)"""
        
        total_sessions = clustering_metrics['total_sessions']
        sessions_with_gauntlets = clustering_metrics['sessions_with_gauntlets']
        total_resonance_events = clustering_metrics['total_resonance_events']
        
        if total_sessions == 0:
            return {'error': 'No sessions for baseline calculation'}
        
        # Background assumption: events distributed uniformly across all sessions
        expected_events_per_session = total_resonance_events / total_sessions
        
        # Expected events in gauntlet sessions under null hypothesis
        expected_gauntlet_session_events = expected_events_per_session * sessions_with_gauntlets
        
        # Background density: expected events per gauntlet under uniform distribution
        total_gauntlet_events = clustering_metrics['total_gauntlet_events']
        if total_gauntlet_events == 0:
            return {'error': 'No gauntlet events for baseline'}
        
        background_density = expected_gauntlet_session_events / total_gauntlet_events
        
        return {
            'background_density': background_density,
            'expected_events_per_session': expected_events_per_session,
            'total_sessions': total_sessions,
            'sessions_with_gauntlets': sessions_with_gauntlets
        }
    
    def test_clustering_significance(self, clustering_metrics: Dict[str, Any], baseline: Dict[str, float]) -> Dict[str, Any]:
        """Test statistical significance of observed clustering"""
        
        if 'error' in baseline:
            return {'error': 'Cannot test significance without valid baseline'}
        
        observed_densities = clustering_metrics.get('session_densities', [])
        background_density = baseline['background_density']
        
        if not observed_densities:
            return {'error': 'No observed densities to test'}
        
        # Calculate observed vs expected ratios
        density_ratios = [obs / background_density if background_density > 0 else 0 for obs in observed_densities]
        
        # TODO(human): Implement proper statistical test for clustering significance
        # This should test if observed event clustering is significantly higher than expected
        # Consider using Poisson test, chi-square goodness of fit, or similar
        # The null hypothesis is that events are uniformly distributed (no clustering)
        
        # Basic statistics
        mean_ratio = np.mean(density_ratios) if density_ratios else 0
        std_ratio = np.std(density_ratios) if len(density_ratios) > 1 else 0
        min_ratio = min(density_ratios) if density_ratios else 0
        max_ratio = max(density_ratios) if density_ratios else 0
        
        significance_test = {
            'test_implemented': False,
            'mean_density_ratio': mean_ratio,
            'std_density_ratio': std_ratio,
            'min_ratio': min_ratio,
            'max_ratio': max_ratio,
            'n_observations': len(density_ratios),
            'claimed_range': '16-172x',
            'observed_range': f"{min_ratio:.1f}-{max_ratio:.1f}x" if density_ratios else "0x",
            'p_value': None,
            'test_statistic': None,
            'verdict': 'NOT_IMPLEMENTED'
        }
        
        return significance_test
    
    def volatility_controlled_analysis(self, resonance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test if clustering patterns persist under volatility controls"""
        
        # TODO(human): Implement volatility-controlled clustering analysis
        # This should re-analyze clustering after controlling for volatility regimes
        # Steps:
        # 1. Identify high/low volatility periods in each session
        # 2. Test clustering within each volatility regime separately  
        # 3. Compare clustering ratios across volatility conditions
        # 4. Test if clustering effect survives volatility normalization
        
        volatility_analysis = {
            'test_implemented': False,
            'high_volatility_clustering': None,
            'low_volatility_clustering': None,
            'volatility_independence_test': {
                'p_value': None,
                'test_statistic': None,
                'verdict': 'NOT_IMPLEMENTED'
            },
            'clustering_survives_volatility_controls': None
        }
        
        return volatility_analysis
    
    def multiple_testing_correction(self, p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """Apply Benjamini-Hochberg FDR correction"""
        
        if not p_values or any(p is None for p in p_values):
            return {'error': 'Cannot apply correction to None p-values'}
        
        # Benjamini-Hochberg procedure
        n = len(p_values)
        sorted_p_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_p_indices]
        
        # Calculate BH critical values
        bh_critical_values = [(i + 1) / n * alpha for i in range(n)]
        
        # Find largest k where P(k) <= (k/n) * alpha
        significant_indices = []
        for i in range(n - 1, -1, -1):  # Start from largest p-value
            if sorted_p_values[i] <= bh_critical_values[i]:
                significant_indices = list(range(i + 1))  # All smaller p-values are significant
                break
        
        # Map back to original indices
        original_significant = [sorted_p_indices[i] for i in significant_indices] if significant_indices else []
        
        return {
            'original_p_values': p_values,
            'corrected_alpha': alpha,
            'n_tests': n,
            'n_significant_before_correction': sum(p < alpha for p in p_values),
            'n_significant_after_correction': len(original_significant),
            'significant_test_indices': original_significant,
            'fdr_controlled': True
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation of 16-172x clustering density claims"""
        
        print("🔍 CLUSTERING DENSITY VALIDATION: 16-172x Claims")
        print("=" * 70)
        print("Red-team testing: Are clustering density claims statistically valid?\\n")
        
        try:
            # Load data
            print("📊 Loading resonance analysis data...")
            resonance_data = self.load_resonance_data()
            
            # Extract clustering metrics
            print("📈 Extracting clustering metrics...")
            clustering_metrics = self.extract_clustering_metrics(resonance_data)
            
            print(f"   Sessions analyzed: {clustering_metrics['total_sessions']}")
            print(f"   Sessions with Gauntlets: {clustering_metrics['sessions_with_gauntlets']}")
            print(f"   Total resonance events: {clustering_metrics['total_resonance_events']}")
            
            # Calculate baseline
            print("\\n🎯 Calculating background baseline...")
            baseline = self.calculate_background_baseline(clustering_metrics)
            
            if 'error' not in baseline:
                print(f"   Background density: {baseline['background_density']:.2f} events/gauntlet")
            else:
                print(f"   ⚠️  Baseline error: {baseline['error']}")
            
            # Test significance
            print("\\n📊 Testing clustering significance...")
            significance_test = self.test_clustering_significance(clustering_metrics, baseline)
            
            if significance_test.get('test_implemented', False):
                print(f"   Mean density ratio: {significance_test['mean_density_ratio']:.1f}x")
                print(f"   Observed range: {significance_test['observed_range']}")
                print(f"   P-value: {significance_test['p_value']}")
            else:
                print(f"   ⚠️  Statistical test not implemented")
                print(f"   Observed range: {significance_test.get('observed_range', 'unknown')}")
                print(f"   Claimed range: {significance_test.get('claimed_range', '16-172x')}")
            
            # Volatility controls
            print("\\n🌊 Testing volatility controls...")
            volatility_analysis = self.volatility_controlled_analysis(resonance_data)
            
            if volatility_analysis.get('test_implemented', False):
                print(f"   Clustering survives volatility controls: {volatility_analysis['clustering_survives_volatility_controls']}")
            else:
                print(f"   ⚠️  Volatility control tests not implemented")
            
            # Compile results
            validation_results = {
                'clustering_metrics': clustering_metrics,
                'baseline_analysis': baseline,
                'significance_test': significance_test,
                'volatility_analysis': volatility_analysis
            }
            
            # Generate verdict
            tests_implemented = (
                significance_test.get('test_implemented', False) and 
                volatility_analysis.get('test_implemented', False)
            )
            
            if not tests_implemented:
                verdict = 'INCOMPLETE_TESTING'
            elif significance_test.get('p_value', 1) > 0.05:
                verdict = 'CLUSTERING_NOT_SIGNIFICANT'
            elif not volatility_analysis.get('clustering_survives_volatility_controls', False):
                verdict = 'VOLATILITY_ARTIFACTS_DETECTED'
            else:
                verdict = 'CLUSTERING_VALIDATED'
            
            validation_results['final_verdict'] = verdict
            
            print(f"\\n🏆 VALIDATION VERDICT: {verdict}")
            
            return validation_results
            
        except Exception as e:
            error_result = {'error': str(e), 'final_verdict': 'VALIDATION_FAILED'}
            print(f"\\n❌ VALIDATION FAILED: {e}")
            return error_result
    
    def save_validation_report(self, results: Dict[str, Any], output_path: Optional[Path] = None):
        """Save detailed validation report"""
        
        if output_path is None:
            output_path = Path("data/validation/clustering_density_validation.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report
        report = {
            'validation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'original_claims': '16-172x clustering density vs background noise',
                'validator_version': '1.0.0'
            },
            'validation_results': results,
            'interpretation': {
                'INCOMPLETE_TESTING': 'Critical tests not implemented - results unverified',
                'CLUSTERING_NOT_SIGNIFICANT': 'Clustering not statistically significant',
                'VOLATILITY_ARTIFACTS_DETECTED': 'Clustering likely due to volatility effects',
                'CLUSTERING_VALIDATED': 'Clustering statistically validated with controls',
                'VALIDATION_FAILED': 'Technical error prevented validation'
            },
            'next_steps': [
                'Complete statistical significance testing implementation',
                'Implement volatility-controlled clustering analysis', 
                'Apply multiple testing corrections (BH-FDR)',
                'Expand sample size for robust validation'
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\\n📁 Validation report saved: {output_path}")

def main():
    """Run clustering density validation"""
    validator = ClusteringDensityValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Save report
    validator.save_validation_report(results)

if __name__ == "__main__":
    main()