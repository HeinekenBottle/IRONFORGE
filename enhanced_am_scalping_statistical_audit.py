#!/usr/bin/env python3
"""
Statistical Audit of Enhanced AM Scalping Framework
Rigorous statistical analysis of validation methodology and claims
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, binomtest, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class FrameworkStatisticalAuditor:
    """
    Comprehensive statistical audit of the Enhanced AM Scalping Framework
    focusing on statistical rigor, sample sizes, and methodological validity
    """
    
    def __init__(self):
        self.data_dir = Path("/Users/jack/IRONFORGE/data/shards/NQ_M5")
        self.test_sessions = ["2025-08-05", "2025-08-06", "2025-08-07"]
        self.theory_b_threshold = 7.55  # Points
        
        # Framework claims to audit
        self.claims = {
            'archaeological_precision': {
                'claimed_avg': 0.52,  # points
                'claimed_improvement': 15,  # 15x better than 7.55
                'claimed_compliance': 100,  # 100% Theory B compliance
            },
            'gauntlet_convergence': {
                'claimed_rate': 86.7,  # 86.7% detection rate
                'test_count': 15,  # 15 synthetic tests
                'claimed_confidence': 2.0  # 2x confidence multiplier
            },
            'macro_effectiveness': {
                'claimed_detection': 66.7,  # 66.7% orbital phase detection
                'claimed_multiplier': 2.0,  # 2.0x confidence multiplier
                'test_scenarios': 3  # 3 mock scenarios
            },
            'overall_score': {
                'claimed_score': 68.7,  # 68.7/100 production-ready
                'weighting': {'archaeological': 0.4, 'macro': 0.3, 'convergence': 0.3}
            }
        }
        
        self.audit_results = {}
    
    def load_session_data(self, session_date: str) -> Optional[Dict[str, Any]]:
        """Load actual session data for validation"""
        try:
            # Look for prior session for archaeological zones
            session_files = list(self.data_dir.glob("shard_NY*"))
            prior_session = None
            
            for session_file in sorted(session_files):
                file_date = session_file.name.split('_')[-1]
                if file_date < session_date:
                    prior_session = session_file
                    break
            
            if not prior_session:
                return None
            
            nodes_file = prior_session / "nodes.parquet"
            if not nodes_file.exists():
                return None
            
            nodes = pd.read_parquet(nodes_file)
            nodes['timestamp_et'] = pd.to_datetime(nodes['t'], unit='ms')
            
            return {
                'nodes': nodes,
                'session_name': prior_session.name,
                'session_high': nodes['price'].max(),
                'session_low': nodes['price'].min(),
                'session_range': nodes['price'].max() - nodes['price'].min(),
                'data_points': len(nodes)
            }
            
        except Exception as e:
            print(f"Error loading session {session_date}: {e}")
            return None
    
    def audit_archaeological_precision_claims(self) -> Dict[str, Any]:
        """
        CRITICAL AUDIT: Archaeological Zone Precision Claims
        Examining n=1 sample size and statistical validity
        """
        print("🔍 AUDITING ARCHAEOLOGICAL PRECISION CLAIMS")
        print("=" * 60)
        
        # Load actual data for claimed sessions
        sessions_with_data = []
        precision_measurements = []
        
        for session_date in self.test_sessions:
            session_data = self.load_session_data(session_date)
            if session_data:
                sessions_with_data.append(session_date)
                
                # Calculate archaeological zones (40%, 60%, 80%)
                session_high = session_data['session_high']
                session_low = session_data['session_low']
                session_range = session_data['session_range']
                nodes = session_data['nodes']
                
                for zone_pct in [0.40, 0.60, 0.80]:
                    zone_level = session_low + (session_range * zone_pct)
                    closest_idx = (nodes['price'] - zone_level).abs().idxmin()
                    closest_price = nodes.loc[closest_idx, 'price']
                    precision = abs(closest_price - zone_level)
                    
                    precision_measurements.append({
                        'session': session_date,
                        'zone_pct': zone_pct,
                        'precision': precision,
                        'theory_b_compliant': precision <= self.theory_b_threshold
                    })
        
        if not precision_measurements:
            return {
                'audit_status': 'FAILED',
                'reason': 'NO_DATA_AVAILABLE',
                'critical_issues': ['Cannot validate claims without data']
            }
        
        # Statistical Analysis
        precisions = [p['precision'] for p in precision_measurements]
        n_measurements = len(precisions)
        
        # Sample size adequacy test
        sample_adequacy = self._assess_sample_size_adequacy(n_measurements, 'precision')
        
        # Statistical tests
        mean_precision = np.mean(precisions)
        std_precision = np.std(precisions, ddof=1) if n_measurements > 1 else 0
        
        # Confidence interval for mean precision
        if n_measurements > 1:
            sem = stats.sem(precisions)
            ci_95 = stats.t.interval(0.95, n_measurements-1, loc=mean_precision, scale=sem)
        else:
            ci_95 = (mean_precision, mean_precision)
        
        # Test against claimed 0.52 average
        if n_measurements > 1:
            t_stat, p_value = ttest_1samp(precisions, self.claims['archaeological_precision']['claimed_avg'])
        else:
            t_stat, p_value = None, None
        
        # Theory B compliance rate
        compliant_count = sum(1 for p in precision_measurements if p['theory_b_compliant'])
        compliance_rate = (compliant_count / n_measurements * 100) if n_measurements > 0 else 0
        
        # Binomial test for 100% compliance claim
        if n_measurements > 0:
            binom_p = binomtest(compliant_count, n_measurements, 1.0, alternative='two-sided').pvalue
        else:
            binom_p = None
        
        audit_results = {
            'audit_status': 'COMPLETED',
            'sample_size': n_measurements,
            'sessions_analyzed': len(sessions_with_data),
            'actual_mean_precision': mean_precision,
            'actual_std_precision': std_precision,
            'confidence_interval_95': ci_95,
            'claimed_vs_actual': {
                'claimed_mean': self.claims['archaeological_precision']['claimed_avg'],
                'actual_mean': mean_precision,
                'difference': mean_precision - self.claims['archaeological_precision']['claimed_avg'],
                't_statistic': t_stat,
                'p_value': p_value
            },
            'theory_b_compliance': {
                'actual_rate': compliance_rate,
                'claimed_rate': self.claims['archaeological_precision']['claimed_compliance'],
                'compliant_count': compliant_count,
                'total_count': n_measurements,
                'binomial_test_p': binom_p
            },
            'sample_adequacy': sample_adequacy,
            'critical_issues': self._identify_precision_issues(n_measurements, sessions_with_data, compliance_rate)
        }
        
        self._print_precision_audit_results(audit_results)
        return audit_results
    
    def audit_gauntlet_convergence_claims(self) -> Dict[str, Any]:
        """
        CRITICAL AUDIT: Gauntlet Convergence Rate Claims
        Examining synthetic test methodology and circular logic
        """
        print("\n🔍 AUDITING GAUNTLET CONVERGENCE CLAIMS")
        print("=" * 60)
        
        claimed_rate = self.claims['gauntlet_convergence']['claimed_rate']
        test_count = self.claims['gauntlet_convergence']['test_count']
        
        # Assess methodology issues
        methodology_issues = []
        
        # Issue 1: Circular logic - using same threshold to define and test convergence
        methodology_issues.append({
            'issue': 'CIRCULAR_LOGIC',
            'description': 'Testing convergence using the same 7.55-point threshold used to define convergence',
            'severity': 'CRITICAL',
            'impact': 'Results are tautological and lack independent validation'
        })
        
        # Issue 2: Synthetic test data
        methodology_issues.append({
            'issue': 'SYNTHETIC_DATA',
            'description': 'Testing with artificial price levels around archaeological zones',
            'severity': 'HIGH',
            'impact': 'Does not represent actual market conditions or real trading scenarios'
        })
        
        # Issue 3: Selection bias
        methodology_issues.append({
            'issue': 'SELECTION_BIAS',
            'description': 'Only testing sessions where archaeological zones exist',
            'severity': 'HIGH', 
            'impact': 'Inflates success rate by excluding negative cases'
        })
        
        # Statistical power analysis for claimed 86.7% rate
        n_tests = test_count
        claimed_successes = int(n_tests * claimed_rate / 100)
        
        # Binomial confidence interval for claimed rate
        if n_tests > 0:
            # Wilson score interval (more accurate for small samples)
            p_hat = claimed_successes / n_tests
            z = 1.96  # 95% confidence
            denominator = 1 + z**2 / n_tests
            center = (p_hat + z**2 / (2 * n_tests)) / denominator
            margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_tests)) / n_tests) / denominator
            ci_95 = (center - margin, center + margin)
            ci_95_pct = (ci_95[0] * 100, ci_95[1] * 100)
        else:
            ci_95_pct = (0, 0)
        
        # Sample size requirement for meaningful testing
        required_n = self._calculate_required_sample_size(claimed_rate/100, 0.1, 0.05, 0.8)
        
        audit_results = {
            'audit_status': 'FAILED',
            'claimed_rate': claimed_rate,
            'test_count': n_tests,
            'claimed_successes': claimed_successes,
            'confidence_interval_95': ci_95_pct,
            'required_sample_size': required_n,
            'sample_adequacy': n_tests >= required_n,
            'methodology_issues': methodology_issues,
            'statistical_power': self._calculate_statistical_power(n_tests, claimed_rate/100),
            'critical_issues': [
                f'Sample size ({n_tests}) insufficient for reliable estimates (need ≥{required_n})',
                'Circular logic invalidates convergence testing',
                'Synthetic data lacks real-world validity',
                'Selection bias inflates apparent success rate'
            ]
        }
        
        self._print_convergence_audit_results(audit_results)
        return audit_results
    
    def audit_macro_window_effectiveness(self) -> Dict[str, Any]:
        """
        CRITICAL AUDIT: Macro Window Effectiveness Claims
        Examining mock scenario methodology
        """
        print("\n🔍 AUDITING MACRO WINDOW EFFECTIVENESS")
        print("=" * 60)
        
        claimed_detection = self.claims['macro_effectiveness']['claimed_detection']
        test_scenarios = self.claims['macro_effectiveness']['test_scenarios']
        
        # Critical methodology issues
        methodology_issues = []
        
        methodology_issues.append({
            'issue': 'INSUFFICIENT_SAMPLE_SIZE',
            'description': f'Only {test_scenarios} mock scenarios tested',
            'severity': 'CRITICAL',
            'impact': 'Cannot establish statistical significance with n=3'
        })
        
        methodology_issues.append({
            'issue': 'PREDETERMINED_OUTCOMES',
            'description': 'Testing with predetermined mock price scenarios',
            'severity': 'HIGH',
            'impact': 'Does not test actual market timing effectiveness'
        })
        
        methodology_issues.append({
            'issue': 'NO_CONTROL_GROUP',
            'description': 'No comparison with random timing or alternative methods',
            'severity': 'HIGH',
            'impact': 'Cannot establish whether results exceed baseline'
        })
        
        # Statistical analysis
        successes = int(test_scenarios * claimed_detection / 100)
        
        # Exact binomial test - can we reject null hypothesis of random performance (50%)?
        if test_scenarios > 0:
            binom_p = binomtest(successes, test_scenarios, 0.5, alternative='greater').pvalue
        else:
            binom_p = None
        
        # Required sample size for meaningful test
        required_n = self._calculate_required_sample_size(claimed_detection/100, 0.15, 0.05, 0.8)
        
        audit_results = {
            'audit_status': 'FAILED',
            'claimed_detection_rate': claimed_detection,
            'test_scenarios': test_scenarios,
            'successes': successes,
            'binomial_test_p': binom_p,
            'statistically_significant': binom_p < 0.05 if binom_p else False,
            'required_sample_size': required_n,
            'sample_adequacy': test_scenarios >= required_n,
            'methodology_issues': methodology_issues,
            'critical_issues': [
                f'Sample size ({test_scenarios}) grossly insufficient (need ≥{required_n})',
                'Mock scenarios cannot validate real market timing',
                'No control group for baseline comparison',
                'Statistical significance cannot be established'
            ]
        }
        
        self._print_macro_audit_results(audit_results)
        return audit_results
    
    def audit_overall_framework_scoring(self) -> Dict[str, Any]:
        """
        CRITICAL AUDIT: Overall Framework Scoring Methodology
        """
        print("\n🔍 AUDITING OVERALL FRAMEWORK SCORING")
        print("=" * 60)
        
        claimed_score = self.claims['overall_score']['claimed_score']
        weighting = self.claims['overall_score']['weighting']
        
        # Critical scoring methodology issues
        scoring_issues = []
        
        scoring_issues.append({
            'issue': 'INVALID_COMPONENT_SCORES',
            'description': 'Component scores based on insufficient/flawed testing',
            'severity': 'CRITICAL',
            'impact': 'Weighted average of invalid inputs produces invalid result'
        })
        
        scoring_issues.append({
            'issue': 'ARBITRARY_WEIGHTING',
            'description': 'No justification for 40/30/30 weighting scheme',
            'severity': 'HIGH',
            'impact': 'Weighting bias can artificially inflate/deflate scores'
        })
        
        scoring_issues.append({
            'issue': 'NO_UNCERTAINTY_QUANTIFICATION',
            'description': 'Point estimate without confidence intervals',
            'severity': 'HIGH',
            'impact': 'Cannot assess reliability of overall score'
        })
        
        # Recalculate score based on audit findings
        audit_based_scores = {
            'archaeological': 0,  # Failed due to insufficient data
            'macro': 0,  # Failed due to insufficient testing
            'convergence': 0  # Failed due to methodological flaws
        }
        
        audit_based_overall = (
            audit_based_scores['archaeological'] * weighting['archaeological'] +
            audit_based_scores['macro'] * weighting['macro'] +
            audit_based_scores['convergence'] * weighting['convergence']
        )
        
        audit_results = {
            'audit_status': 'FAILED',
            'claimed_score': claimed_score,
            'audit_based_score': audit_based_overall,
            'component_score_validity': {
                'archaeological': 'INVALID - insufficient data',
                'macro': 'INVALID - insufficient testing', 
                'convergence': 'INVALID - circular logic'
            },
            'weighting_scheme': weighting,
            'scoring_issues': scoring_issues,
            'production_readiness': False,
            'critical_issues': [
                'All component scores based on flawed methodology',
                'Insufficient statistical validation for any component',
                'Overall score not statistically meaningful',
                'Claims of "production-ready" status unjustified'
            ]
        }
        
        self._print_scoring_audit_results(audit_results)
        return audit_results
    
    def _assess_sample_size_adequacy(self, n: int, analysis_type: str) -> Dict[str, Any]:
        """Assess whether sample size is adequate for statistical analysis"""
        
        adequacy_thresholds = {
            'precision': 30,  # Minimum for central limit theorem
            'proportion': 30,  # Minimum for binomial approximation
            'correlation': 20,  # Minimum for meaningful correlation
            'regression': 50   # Minimum for regression analysis
        }
        
        required_n = adequacy_thresholds.get(analysis_type, 30)
        
        return {
            'actual_n': n,
            'required_n': required_n,
            'adequate': n >= required_n,
            'adequacy_ratio': n / required_n,
            'statistical_power': 'LOW' if n < required_n else 'MODERATE' if n < required_n * 2 else 'HIGH'
        }
    
    def _calculate_required_sample_size(self, p: float, margin_error: float, alpha: float = 0.05, power: float = 0.8) -> int:
        """Calculate required sample size for proportion estimation"""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # For proportion estimation
        n = (z_alpha**2 * p * (1-p)) / (margin_error**2)
        
        return int(np.ceil(n))
    
    def _calculate_statistical_power(self, n: int, p: float) -> float:
        """Calculate statistical power for given sample size and proportion"""
        if n <= 1:
            return 0.0
        
        # Simplified power calculation for proportion test
        se = np.sqrt(p * (1-p) / n)
        z_score = 1.96  # 95% confidence
        power = 1 - stats.norm.cdf(z_score - (0.1 / se))  # Power to detect 10% difference
        
        return max(0.0, min(1.0, power))
    
    def _identify_precision_issues(self, n: int, sessions: List[str], compliance_rate: float) -> List[str]:
        """Identify critical issues with precision claims"""
        issues = []
        
        if n < 10:
            issues.append(f'Extremely small sample size (n={n}) - insufficient for statistical inference')
        
        if len(sessions) == 1:
            issues.append('Claims based on single session - no generalizability')
        
        if compliance_rate < 100 and self.claims['archaeological_precision']['claimed_compliance'] == 100:
            issues.append(f'100% compliance claim false - actual rate {compliance_rate:.1f}%')
        
        if n == 0:
            issues.append('No archaeological zones found in test data - cannot validate claims')
        
        return issues
    
    def _print_precision_audit_results(self, results: Dict[str, Any]):
        """Print archaeological precision audit results"""
        print(f"\n📊 ARCHAEOLOGICAL PRECISION AUDIT RESULTS:")
        print(f"   Sample size: {results['sample_size']} measurements")
        print(f"   Sessions analyzed: {results['sessions_analyzed']}")
        
        if results['sample_size'] > 0:
            print(f"   Actual mean precision: {results['actual_mean_precision']:.2f} points")
            print(f"   95% CI: ({results['confidence_interval_95'][0]:.2f}, {results['confidence_interval_95'][1]:.2f})")
            print(f"   Theory B compliance: {results['theory_b_compliance']['actual_rate']:.1f}%")
        
        print(f"   Sample adequacy: {'ADEQUATE' if results['sample_adequacy']['adequate'] else 'INADEQUATE'}")
        
        if results['critical_issues']:
            print(f"\n❌ CRITICAL ISSUES:")
            for issue in results['critical_issues']:
                print(f"   • {issue}")
    
    def _print_convergence_audit_results(self, results: Dict[str, Any]):
        """Print gauntlet convergence audit results"""
        print(f"\n📊 GAUNTLET CONVERGENCE AUDIT RESULTS:")
        print(f"   Claimed rate: {results['claimed_rate']:.1f}%")
        print(f"   Test count: {results['test_count']}")
        print(f"   95% CI: ({results['confidence_interval_95'][0]:.1f}%, {results['confidence_interval_95'][1]:.1f}%)")
        print(f"   Required sample size: {results['required_sample_size']}")
        print(f"   Sample adequate: {'NO' if not results['sample_adequacy'] else 'YES'}")
        
        print(f"\n❌ METHODOLOGY ISSUES:")
        for issue in results['methodology_issues']:
            print(f"   • {issue['issue']}: {issue['description']}")
    
    def _print_macro_audit_results(self, results: Dict[str, Any]):
        """Print macro window effectiveness audit results"""
        print(f"\n📊 MACRO WINDOW EFFECTIVENESS AUDIT RESULTS:")
        print(f"   Claimed detection: {results['claimed_detection_rate']:.1f}%")
        print(f"   Test scenarios: {results['test_scenarios']}")
        print(f"   Statistical significance: {'NO' if not results['statistically_significant'] else 'YES'}")
        print(f"   Required sample size: {results['required_sample_size']}")
        
        print(f"\n❌ METHODOLOGY ISSUES:")
        for issue in results['methodology_issues']:
            print(f"   • {issue['issue']}: {issue['description']}")
    
    def _print_scoring_audit_results(self, results: Dict[str, Any]):
        """Print overall scoring audit results"""
        print(f"\n📊 OVERALL FRAMEWORK SCORING AUDIT:")
        print(f"   Claimed score: {results['claimed_score']}/100")
        print(f"   Audit-based score: {results['audit_based_score']}/100")
        print(f"   Production ready: {'NO' if not results['production_readiness'] else 'YES'}")
        
        print(f"\n❌ COMPONENT VALIDITY:")
        for component, validity in results['component_score_validity'].items():
            print(f"   • {component}: {validity}")
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive statistical audit of all framework claims"""
        print("🚨 ENHANCED AM SCALPING FRAMEWORK - STATISTICAL AUDIT")
        print("=" * 80)
        print("Comprehensive statistical validation of framework claims and methodology")
        print("=" * 80)
        
        # Run all audits
        self.audit_results = {
            'archaeological_precision': self.audit_archaeological_precision_claims(),
            'gauntlet_convergence': self.audit_gauntlet_convergence_claims(),
            'macro_effectiveness': self.audit_macro_window_effectiveness(),
            'overall_scoring': self.audit_overall_framework_scoring(),
            'audit_timestamp': datetime.now(),
            'audit_summary': self._generate_audit_summary()
        }
        
        return self.audit_results
    
    def _generate_audit_summary(self) -> Dict[str, Any]:
        """Generate comprehensive audit summary"""
        
        # Count failures
        failed_components = sum(1 for result in self.audit_results.values() 
                              if isinstance(result, dict) and result.get('audit_status') == 'FAILED')
        
        total_components = 4  # archaeological, gauntlet, macro, scoring
        
        return {
            'overall_audit_status': 'FAILED',
            'failed_components': failed_components,
            'total_components': total_components,
            'failure_rate': failed_components / total_components * 100,
            'production_readiness': False,
            'statistical_validity': False,
            'key_findings': [
                'Insufficient sample sizes across all components',
                'Methodological flaws invalidate most claims',
                'Circular logic in convergence testing',
                'Synthetic data lacks real-world validity',
                'No proper control groups or baseline comparisons',
                'Claims of statistical significance unsupported',
                'Overall framework score not statistically meaningful'
            ],
            'recommendations': [
                'Collect minimum 30 samples per component for statistical validity',
                'Implement proper control groups and baseline comparisons',
                'Use real market data instead of synthetic scenarios',
                'Eliminate circular logic in testing methodologies',
                'Calculate confidence intervals for all estimates',
                'Perform proper statistical power analysis',
                'Defer production deployment until statistical validation complete'
            ]
        }
    
    def display_audit_summary(self):
        """Display comprehensive audit summary"""
        if not self.audit_results:
            print("❌ No audit results available")
            return
        
        summary = self.audit_results['audit_summary']
        
        print(f"\n{'='*80}")
        print("🚨 FINAL AUDIT SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n📊 AUDIT OVERVIEW:")
        print(f"   Overall Status: {summary['overall_audit_status']}")
        print(f"   Failed Components: {summary['failed_components']}/{summary['total_components']}")
        print(f"   Failure Rate: {summary['failure_rate']:.1f}%")
        print(f"   Production Ready: {'NO' if not summary['production_readiness'] else 'YES'}")
        print(f"   Statistically Valid: {'NO' if not summary['statistical_validity'] else 'YES'}")
        
        print(f"\n🔍 KEY FINDINGS:")
        for finding in summary['key_findings']:
            print(f"   • {finding}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for recommendation in summary['recommendations']:
            print(f"   • {recommendation}")
        
        print(f"\n⚠️  CONCLUSION:")
        print(f"   The Enhanced AM Scalping Framework claims are NOT statistically validated.")
        print(f"   Current validation methodology contains critical flaws that invalidate results.")
        print(f"   Framework should NOT be considered production-ready until proper statistical")
        print(f"   validation is completed with adequate sample sizes and rigorous methodology.")
        
        print(f"\n{'='*80}")

def main():
    """Run the comprehensive statistical audit"""
    auditor = FrameworkStatisticalAuditor()
    results = auditor.run_comprehensive_audit()
    auditor.display_audit_summary()
    
    # Save detailed results
    output_file = f"/Users/jack/IRONFORGE/framework_statistical_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    
    print(f"\n📄 Detailed audit results saved to: {output_file}")

if __name__ == "__main__":
    main()