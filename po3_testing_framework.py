#!/usr/bin/env python3
"""
PO3 Testing Framework - Phase III Implementation
Framework testing: PO3 predictions vs observed data patterns
Challenge assumptions and test competing hypotheses

IRONFORGE Research Classification: H9 Validation Framework
Statistical Testing and Alternative Hypothesis Development
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

from po3_statistical_classifier import PO3StatisticalClassifier, MacroWindowData, PO3Classification

@dataclass
class CompetingHypothesis:
    """Structure for alternative hypothesis testing"""
    hypothesis_name: str
    description: str
    prediction_method: callable
    validation_metrics: Dict[str, float]
    statistical_significance: float

class PO3TestingFramework:
    """
    Comprehensive testing framework for PO3 classification
    Tests competing hypotheses and challenges core assumptions
    """
    
    def __init__(self):
        self.po3_classifier = PO3StatisticalClassifier()
        self.test_results = {}
        self.competing_hypotheses = {}
        
    def load_actual_h8_data(self, data_path: str) -> List[MacroWindowData]:
        """
        Load actual H8 analysis results from DATA agent
        Convert to MacroWindowData format for PO3 testing
        """
        # TODO: Implement actual data loading once DATA agent completes
        # This will read the CSV/JSON results from H8 analysis
        
        # Placeholder structure - replace with actual data loading
        sample_data = []
        
        print("📊 Loading H8 analysis results from DATA agent...")
        print(f"📁 Expected data path: {data_path}")
        print("⏳ Waiting for DATA agent H8 analysis completion...")
        
        return sample_data
    
    def compare_po3_vs_observed_compound_type(self, macro_windows: List[MacroWindowData], 
                                            observed_compound_types: List[str]) -> Dict[str, any]:
        """
        Compare PO3 predictions against observed compound_type from H8 analysis
        Test core assumption: Do our PO3 classifications match observed behavior?
        """
        
        print("🔍 Testing Core Assumption: PO3 Classifications vs Observed Compound Types")
        print("=" * 70)
        
        # Generate PO3 predictions
        po3_predictions = []
        po3_confidences = []
        
        for window in macro_windows:
            classification = self.po3_classifier.classify_macro_window(window)
            po3_predictions.append(classification.phase)
            po3_confidences.append(classification.confidence)
        
        # Map compound_type to expected PO3 phases for comparison
        compound_to_po3_mapping = {
            'liquidity_grab': 'MANIPULATION',  # Should be manipulation
            'range_expansion': 'DISTRIBUTION', # Should be distribution  
            'consolidation': 'ACCUMULATION',   # Should be accumulation
            'breakout': 'MANIPULATION',        # Should be manipulation
            'reversal': 'DISTRIBUTION',        # Should be distribution
            'continuation': 'MANIPULATION'     # Should be manipulation
        }
        
        # Convert observed compound types to expected PO3 phases
        expected_po3_phases = []
        for compound_type in observed_compound_types:
            expected_phase = compound_to_po3_mapping.get(compound_type, 'TRANSITION')
            expected_po3_phases.append(expected_phase)
        
        # Calculate accuracy metrics
        overall_accuracy = accuracy_score(expected_po3_phases, po3_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            expected_po3_phases, po3_predictions, average='weighted'
        )
        
        # Phase-specific analysis
        phase_accuracy = {}
        unique_phases = list(set(expected_po3_phases + po3_predictions))
        
        for phase in unique_phases:
            phase_expected = [1 if p == phase else 0 for p in expected_po3_phases]
            phase_predicted = [1 if p == phase else 0 for p in po3_predictions]
            phase_accuracy[phase] = accuracy_score(phase_expected, phase_predicted)
        
        # Statistical significance test
        contingency_table = pd.crosstab(pd.Series(expected_po3_phases), pd.Series(po3_predictions))
        try:
            from scipy.stats import chi2_contingency
            chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency_table)
            statistical_significance = chi2_p_value < 0.05
        except:
            chi2_stat, chi2_p_value, statistical_significance = 0, 1, False
        
        results = {
            'overall_accuracy': overall_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'phase_specific_accuracy': phase_accuracy,
            'average_confidence': np.mean(po3_confidences),
            'chi_square_statistic': chi2_stat,
            'chi_square_p_value': chi2_p_value,
            'statistically_significant': statistical_significance,
            'confusion_matrix': contingency_table.values.tolist(),
            'compound_type_mapping_success': overall_accuracy > 0.6
        }
        
        print(f"📈 Overall Accuracy: {overall_accuracy:.3f}")
        print(f"📊 Average Confidence: {np.mean(po3_confidences):.3f}")
        print(f"🎯 Statistically Significant: {statistical_significance}")
        
        return results
    
    def test_range_position_hypothesis(self, macro_windows: List[MacroWindowData]) -> CompetingHypothesis:
        """
        Competing Hypothesis 1: Range Position Effects
        Test if position within session range better predicts behavior than PO3
        """
        
        def predict_by_range_position(window: MacroWindowData) -> str:
            """Predict based on position within session range"""
            
            # Simulate range position calculation (would use actual session high/low)
            # Early session = accumulation, middle = manipulation, late = distribution
            hour = window.timestamp.hour if window.timestamp else 9
            
            if hour <= 10:      # Early session
                return 'ACCUMULATION'
            elif hour <= 14:    # Mid session  
                return 'MANIPULATION'
            else:               # Late session
                return 'DISTRIBUTION'
        
        # Generate range position predictions
        range_predictions = [predict_by_range_position(w) for w in macro_windows]
        
        # Compare against PO3 predictions (as baseline)
        po3_predictions = [self.po3_classifier.classify_macro_window(w).phase for w in macro_windows]
        
        # Calculate agreement rate
        agreement_rate = np.mean([r == p for r, p in zip(range_predictions, po3_predictions)])
        
        # Validate against amplification patterns
        range_amplification_accuracy = 0.0
        for window, range_pred in zip(macro_windows, range_predictions):
            expected_amp_ratio = window.amplification / 50.96
            
            if range_pred == 'ACCUMULATION' and expected_amp_ratio < 0.5:
                range_amplification_accuracy += 1
            elif range_pred == 'MANIPULATION' and expected_amp_ratio > 1.5:
                range_amplification_accuracy += 1
            elif range_pred == 'DISTRIBUTION' and 0.4 <= expected_amp_ratio <= 1.6:
                range_amplification_accuracy += 1
        
        range_amplification_accuracy /= len(macro_windows)
        
        return CompetingHypothesis(
            hypothesis_name="Range Position Effects",
            description="Session range position predicts institutional behavior better than amplification signatures",
            prediction_method=predict_by_range_position,
            validation_metrics={
                'agreement_with_po3': agreement_rate,
                'amplification_accuracy': range_amplification_accuracy,
                'sample_size': len(macro_windows)
            },
            statistical_significance=0.1 if range_amplification_accuracy > 0.6 else 0.0
        )
    
    def test_energy_density_cycles_hypothesis(self, macro_windows: List[MacroWindowData]) -> CompetingHypothesis:
        """
        Competing Hypothesis 2: Energy Density Cycles
        Test if f8 liquidity cycles better predict behavior than PO3 amplification
        """
        
        def predict_by_energy_density(window: MacroWindowData) -> str:
            """Predict based on f8 liquidity energy density patterns"""
            
            f8_intensity = window.f8_liquidity_spike
            
            # Energy density thresholds (would be calibrated from actual data)
            if f8_intensity < 30:       # Low energy = accumulation
                return 'ACCUMULATION'
            elif f8_intensity > 70:     # High energy = manipulation
                return 'MANIPULATION' 
            else:                       # Medium energy = distribution
                return 'DISTRIBUTION'
        
        # Generate energy density predictions
        energy_predictions = [predict_by_energy_density(w) for w in macro_windows]
        
        # Compare against amplification-based predictions
        po3_predictions = [self.po3_classifier.classify_macro_window(w).phase for w in macro_windows]
        
        agreement_rate = np.mean([e == p for e, p in zip(energy_predictions, po3_predictions)])
        
        # Test correlation between f8 intensity and amplification
        f8_values = [w.f8_liquidity_spike for w in macro_windows]
        amplification_values = [w.amplification for w in macro_windows]
        
        try:
            correlation_coeff, correlation_p = stats.pearsonr(f8_values, amplification_values)
        except:
            correlation_coeff, correlation_p = 0, 1
        
        return CompetingHypothesis(
            hypothesis_name="Energy Density Cycles",
            description="f8 liquidity energy density cycles predict behavior better than news amplification",
            prediction_method=predict_by_energy_density,
            validation_metrics={
                'agreement_with_po3': agreement_rate,
                'f8_amplification_correlation': correlation_coeff,
                'correlation_p_value': correlation_p,
                'sample_size': len(macro_windows)
            },
            statistical_significance=correlation_p if correlation_p < 0.05 else 0.0
        )
    
    def challenge_core_assumptions(self, macro_windows: List[MacroWindowData]) -> Dict[str, any]:
        """
        Challenge core PO3-H8 assumptions:
        1. Do macro windows truly = manipulation?
        2. Does news always = 50.96x amplification?
        3. Are PO3 phases mutually exclusive?
        """
        
        print("🚨 CHALLENGING CORE ASSUMPTIONS")
        print("=" * 50)
        
        assumption_tests = {}
        
        # Test 1: Macro Windows = Manipulation assumption
        manipulation_during_macro = 0
        total_macro_windows = len(macro_windows)
        
        for window in macro_windows:
            classification = self.po3_classifier.classify_macro_window(window)
            if classification.phase == 'MANIPULATION':
                manipulation_during_macro += 1
        
        manipulation_rate = manipulation_during_macro / total_macro_windows
        assumption_tests['macro_equals_manipulation'] = {
            'manipulation_rate_during_macro': manipulation_rate,
            'assumption_supported': manipulation_rate > 0.6,
            'challenge_evidence': f"Only {manipulation_rate:.1%} of macro windows show manipulation behavior"
        }
        
        # Test 2: News = 50.96x amplification assumption
        news_windows = [w for w in macro_windows if len(w.news_events) > 0]
        non_news_windows = [w for w in macro_windows if len(w.news_events) == 0]
        
        if news_windows and non_news_windows:
            news_amplifications = [w.amplification for w in news_windows]
            non_news_amplifications = [w.amplification for w in non_news_windows]
            
            news_avg = np.mean(news_amplifications)
            non_news_avg = np.mean(non_news_amplifications)
            
            # Test if news significantly increases amplification
            try:
                t_stat, t_p_value = stats.ttest_ind(news_amplifications, non_news_amplifications)
                news_effect_significant = t_p_value < 0.05 and news_avg > non_news_avg
            except:
                t_stat, t_p_value, news_effect_significant = 0, 1, False
            
            assumption_tests['news_equals_amplification'] = {
                'news_average_amplification': news_avg,
                'non_news_average_amplification': non_news_avg,
                'amplification_difference': news_avg - non_news_avg,
                't_test_p_value': t_p_value,
                'assumption_supported': news_effect_significant,
                'challenge_evidence': f"News effect magnitude: {(news_avg/non_news_avg - 1)*100:.1f}% vs expected ~400% boost"
            }
        
        # Test 3: PO3 phases mutually exclusive assumption
        mixed_classifications = 0
        classification_uncertainties = []
        
        for window in macro_windows:
            classification = self.po3_classifier.classify_macro_window(window)
            classification_uncertainties.append(1 - classification.confidence)
            
            if classification.phase == 'TRANSITION':
                mixed_classifications += 1
        
        transition_rate = mixed_classifications / total_macro_windows
        avg_uncertainty = np.mean(classification_uncertainties)
        
        assumption_tests['phases_mutually_exclusive'] = {
            'transition_rate': transition_rate,
            'average_uncertainty': avg_uncertainty,
            'assumption_supported': transition_rate < 0.2 and avg_uncertainty < 0.3,
            'challenge_evidence': f"{transition_rate:.1%} windows show mixed signals, avg uncertainty {avg_uncertainty:.3f}"
        }
        
        print(f"🎯 Macro = Manipulation: {assumption_tests['macro_equals_manipulation']['assumption_supported']}")
        print(f"📰 News = 50.96x Boost: {assumption_tests['news_equals_amplification']['assumption_supported'] if 'news_equals_amplification' in assumption_tests else 'Insufficient data'}")  
        print(f"🔄 Phases Exclusive: {assumption_tests['phases_mutually_exclusive']['assumption_supported']}")
        
        return assumption_tests
    
    def generate_alternative_frameworks(self, test_results: Dict[str, any]) -> List[str]:
        """
        Generate alternative frameworks based on data patterns and assumption challenges
        """
        
        alternative_frameworks = []
        
        # Framework 1: Hybrid Energy-Amplification Model
        if 'energy_density_cycles' in self.competing_hypotheses:
            energy_hypothesis = self.competing_hypotheses['energy_density_cycles']
            if energy_hypothesis.validation_metrics.get('f8_amplification_correlation', 0) > 0.6:
                alternative_frameworks.append(
                    "Hybrid Energy-Amplification Model: Combine f8 liquidity energy density with "
                    "news amplification for multi-dimensional behavioral classification"
                )
        
        # Framework 2: Temporal Position Model
        macro_manipulation_rate = test_results.get('macro_equals_manipulation', {}).get('manipulation_rate_during_macro', 0)
        if macro_manipulation_rate < 0.5:
            alternative_frameworks.append(
                "Temporal Position Model: Session time position and range completion percentage "
                "may be stronger predictors than macro window timing"
            )
        
        # Framework 3: Probabilistic Phase Model
        transition_rate = test_results.get('phases_mutually_exclusive', {}).get('transition_rate', 0)
        if transition_rate > 0.3:
            alternative_frameworks.append(
                "Probabilistic Phase Model: Replace binary phase classification with "
                "probability distributions across multiple simultaneous institutional behaviors"
            )
        
        # Framework 4: News Context Dependency Model
        news_test = test_results.get('news_equals_amplification', {})
        if news_test and not news_test.get('assumption_supported', True):
            alternative_frameworks.append(
                "News Context Dependency Model: Amplification effects vary by news type, "
                "market regime, and institutional positioning context rather than fixed 50.96x multiplier"
            )
        
        # Framework 5: Multi-Scale Institutional Behavior Model
        if len(alternative_frameworks) >= 2:
            alternative_frameworks.append(
                "Multi-Scale Institutional Behavior Model: Integrate range position, energy density, "
                "temporal factors, and news context into unified behavioral prediction system"
            )
        
        return alternative_frameworks
    
    def run_comprehensive_testing(self, data_path: str) -> Dict[str, any]:
        """
        Run comprehensive testing framework
        Load data, test assumptions, compare hypotheses, generate alternatives
        """
        
        print("🧪 COMPREHENSIVE PO3 TESTING FRAMEWORK")
        print("=" * 70)
        
        # Step 1: Load actual H8 data (when available)
        macro_windows = self.load_actual_h8_data(data_path)
        
        if not macro_windows:
            print("⚠️  No H8 data available yet - using synthetic test data")
            # Generate synthetic data for framework testing
            macro_windows = self._generate_synthetic_test_data()
        
        comprehensive_results = {
            'data_source': data_path,
            'sample_size': len(macro_windows),
            'framework_version': 'Phase_III_v1.0'
        }
        
        # Step 2: Test competing hypotheses
        print("\n🎯 Testing Competing Hypotheses...")
        
        range_hypothesis = self.test_range_position_hypothesis(macro_windows)
        energy_hypothesis = self.test_energy_density_cycles_hypothesis(macro_windows)
        
        self.competing_hypotheses['range_position'] = range_hypothesis
        self.competing_hypotheses['energy_density_cycles'] = energy_hypothesis
        
        comprehensive_results['competing_hypotheses'] = {
            'range_position': range_hypothesis.__dict__,
            'energy_density_cycles': energy_hypothesis.__dict__
        }
        
        # Step 3: Challenge core assumptions
        print("\n🚨 Challenging Core Assumptions...")
        assumption_results = self.challenge_core_assumptions(macro_windows)
        comprehensive_results['assumption_challenges'] = assumption_results
        
        # Step 4: Generate alternative frameworks
        print("\n🔬 Generating Alternative Frameworks...")
        alternative_frameworks = self.generate_alternative_frameworks(assumption_results)
        comprehensive_results['alternative_frameworks'] = alternative_frameworks
        
        print(f"\n✅ Framework Testing Complete!")
        print(f"📊 Tested {len(macro_windows)} macro windows")
        print(f"🎯 Evaluated {len(self.competing_hypotheses)} competing hypotheses") 
        print(f"🔬 Generated {len(alternative_frameworks)} alternative frameworks")
        
        return comprehensive_results
    
    def _generate_synthetic_test_data(self) -> List[MacroWindowData]:
        """Generate synthetic test data for framework validation"""
        
        synthetic_data = []
        
        # Create diverse test scenarios
        test_scenarios = [
            # Accumulation scenarios (low amplification)
            {'amplification': 20.5, 'volume_ratio': 3.2, 'directional_coherence': 0.3, 'session': 'NYAM'},
            {'amplification': 15.8, 'volume_ratio': 2.8, 'directional_coherence': 0.25, 'session': 'NYAM'},
            
            # Manipulation scenarios (high amplification)
            {'amplification': 89.3, 'volume_ratio': 1.8, 'directional_coherence': 0.85, 'session': 'LUNCH'},
            {'amplification': 75.2, 'volume_ratio': 2.1, 'directional_coherence': 0.78, 'session': 'NYPM'},
            
            # Distribution scenarios (variable amplification)
            {'amplification': 35.7, 'volume_ratio': 2.5, 'directional_coherence': 0.45, 'session': 'NYPM'},
            {'amplification': 42.1, 'volume_ratio': 3.1, 'directional_coherence': 0.35, 'session': 'CLOSE'},
            
            # Transition scenarios
            {'amplification': 48.2, 'volume_ratio': 2.3, 'directional_coherence': 0.55, 'session': 'LUNCH'},
        ]
        
        for i, scenario in enumerate(test_scenarios):
            window = MacroWindowData(
                window_id=f"synthetic_{i}",
                timestamp=pd.Timestamp('2025-08-24 09:00:00') + pd.Timedelta(hours=i),
                session=scenario['session'],
                day_of_week='Monday',
                amplification=scenario['amplification'],
                volume_ratio=scenario['volume_ratio'],
                price_movement=15.2,
                directional_coherence=scenario['directional_coherence'],
                f8_liquidity_spike=np.random.uniform(20, 80),
                archaeological_zones_hit=['40%'] if np.random.random() > 0.5 else [],
                news_events=['CPI'] if np.random.random() > 0.7 else []
            )
            synthetic_data.append(window)
        
        return synthetic_data

def main():
    """Main testing function"""
    
    print("🧪 PO3 Testing Framework - Phase III Implementation")
    print("=" * 70)
    
    # Initialize testing framework
    testing_framework = PO3TestingFramework()
    
    # Run comprehensive testing (will use synthetic data until H8 analysis complete)
    results = testing_framework.run_comprehensive_testing("data/h8_analysis_results.csv")
    
    print("\n📋 TESTING SUMMARY:")
    print("-" * 30)
    print(f"Sample Size: {results['sample_size']} macro windows")
    print(f"Competing Hypotheses: {len(results['competing_hypotheses'])}")
    print(f"Alternative Frameworks: {len(results['alternative_frameworks'])}")
    
    if results['alternative_frameworks']:
        print("\n🔬 ALTERNATIVE FRAMEWORKS GENERATED:")
        for i, framework in enumerate(results['alternative_frameworks'], 1):
            print(f"{i}. {framework}")
    
    print("\n✅ Ready to apply to actual H8 data when DATA agent completes!")

if __name__ == "__main__":
    main()