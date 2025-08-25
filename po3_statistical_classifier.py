#!/usr/bin/env python3
"""
PO3 Statistical Classifier - Phase II Implementation
Statistical framework to classify macro windows by PO3 phases using 50.96x baseline

IRONFORGE Research Classification: H9 Implementation
Statistical Framework for Institutional Behavior Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Statistical Classification Thresholds Based on H9 Hypotheses
BASELINE_AMPLIFICATION = 50.96  # Validated H8 baseline
ACCUMULATION_THRESHOLD = 0.50   # 15-25x range = 0.30-0.49 of baseline
MANIPULATION_THRESHOLD = 1.50   # 75-100x range = 1.47-1.96 of baseline
DISTRIBUTION_VARIANCE = 0.40    # High variance coefficient for distribution detection

@dataclass
class MacroWindowData:
    """Data structure for macro window analysis"""
    window_id: str
    timestamp: pd.Timestamp
    session: str  # NYAM, LUNCH, NYPM, CLOSE
    day_of_week: str
    amplification: float
    volume_ratio: float
    price_movement: float
    directional_coherence: float
    f8_liquidity_spike: float
    archaeological_zones_hit: List[str]
    news_events: List[str]
    
@dataclass
class PO3Classification:
    """PO3 phase classification result"""
    phase: str  # ACCUMULATION, MANIPULATION, DISTRIBUTION, TRANSITION
    confidence: float
    amplification_ratio: float
    supporting_metrics: Dict[str, float]
    statistical_significance: float

class PO3StatisticalClassifier:
    """
    Statistical framework for classifying macro windows by PO3 phases
    using validated 50.96x amplification baseline
    """
    
    def __init__(self):
        self.baseline_amplification = BASELINE_AMPLIFICATION
        self.scaler = StandardScaler()
        self.ml_classifier = None
        self.classification_history = []
        
    def calculate_amplification_metrics(self, window_data: MacroWindowData) -> Dict[str, float]:
        """Calculate core amplification-based classification metrics"""
        
        # Primary amplification ratio against 50.96x baseline
        amplification_ratio = window_data.amplification / self.baseline_amplification
        
        # Volume-Price Absorption Ratio (higher = more accumulation-like)
        volume_price_ratio = window_data.volume_ratio / max(window_data.price_movement, 0.1)
        
        # Directional Coherence Score (higher = more manipulation-like)
        directional_score = window_data.directional_coherence
        
        # Volatility Variance (higher = more distribution-like)
        # Rolling variance calculation for detecting chaotic distribution behavior
        if hasattr(window_data, 'price_history') and len(getattr(window_data, 'price_history', [])) >= 5:
            # Use actual price history if available
            price_movements = np.array(window_data.price_history)
            rolling_variance = np.var(price_movements[-10:])  # Last 10 data points
            # Normalize to 0-1 range where higher = more erratic
            volatility_variance = min(rolling_variance / 100.0, 1.0)
        else:
            # Synthetic variance estimation using available metrics
            # Distribution phases show 20-80x range with high variance
            amp_deviation = abs(amplification_ratio - 1.0)  # Distance from baseline
            directional_chaos = 1.0 - window_data.directional_coherence  # Inversely related
            volume_price_discord = abs(window_data.volume_ratio - window_data.price_movement) / 50.0
            
            # Composite variance estimate (0-1 scale)
            volatility_variance = min((amp_deviation * 0.4 + directional_chaos * 0.4 + 
                                     volume_price_discord * 0.2), 1.0)
        
        # Archaeological Zone Impact Score
        zone_impact = len(window_data.archaeological_zones_hit) * 0.2
        
        # News Event Timing Score
        news_impact = len(window_data.news_events) * 1.5
        
        return {
            'amplification_ratio': amplification_ratio,
            'volume_price_ratio': volume_price_ratio, 
            'directional_coherence': directional_score,
            'volatility_variance': volatility_variance,
            'zone_impact': zone_impact,
            'news_impact': news_impact,
            'f8_liquidity': window_data.f8_liquidity_spike
        }
    
    def classify_accumulation_phase(self, metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Classify ACCUMULATION phase probability
        H9.1: Suppressed amplification (15-25x) with high volume absorption
        """
        
        # Core accumulation indicators
        suppressed_amplification = 1.0 if metrics['amplification_ratio'] < ACCUMULATION_THRESHOLD else 0.0
        high_volume_absorption = min(metrics['volume_price_ratio'] / 2.0, 1.0)
        low_directional_bias = 1.0 - metrics['directional_coherence']
        
        # Statistical weighting
        accumulation_score = (
            suppressed_amplification * 0.40 +  # Primary indicator
            high_volume_absorption * 0.35 +    # Volume absorption
            low_directional_bias * 0.25        # Minimal directional movement
        )
        
        supporting_evidence = {
            'suppressed_amplification': suppressed_amplification,
            'volume_absorption': high_volume_absorption,
            'directional_neutrality': low_directional_bias,
            'accumulation_strength': metrics['amplification_ratio']
        }
        
        return min(accumulation_score, 1.0), supporting_evidence
    
    def classify_manipulation_phase(self, metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Classify MANIPULATION phase probability  
        H9.2: Enhanced amplification (75-100x) with strong directional coherence
        """
        
        # Core manipulation indicators
        enhanced_amplification = 1.0 if metrics['amplification_ratio'] > MANIPULATION_THRESHOLD else 0.0
        strong_directional_bias = metrics['directional_coherence']
        archaeological_precision = min(metrics['zone_impact'] / 1.0, 1.0)
        
        # News event timing bonus (manipulation often uses news as cover)
        news_timing_bonus = min(metrics['news_impact'] / 3.0, 0.3)
        
        # Statistical weighting
        manipulation_score = (
            enhanced_amplification * 0.45 +     # Primary indicator
            strong_directional_bias * 0.30 +    # Directional consistency
            archaeological_precision * 0.25 +   # Zone targeting
            news_timing_bonus                   # News confluence bonus
        )
        
        supporting_evidence = {
            'enhanced_amplification': enhanced_amplification,
            'directional_strength': strong_directional_bias,
            'zone_precision': archaeological_precision,
            'news_confluence': news_timing_bonus,
            'manipulation_intensity': metrics['amplification_ratio']
        }
        
        return min(manipulation_score, 1.0), supporting_evidence
    
    def classify_distribution_phase(self, metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Classify DISTRIBUTION phase probability
        H9.3: Volatile amplification (20-80x range) with high variance and mixed signals
        """
        
        # Core distribution indicators
        amplification_in_range = 1.0 if 0.4 <= metrics['amplification_ratio'] <= 1.6 else 0.0
        high_volatility_variance = metrics['volatility_variance']  # Implemented by human
        mixed_directional_signals = 1.0 - abs(metrics['directional_coherence'] - 0.5) * 2
        
        # High volume with erratic price movement
        volume_price_disconnect = min(metrics['volume_price_ratio'] / 1.5, 1.0) * (1.0 - metrics['directional_coherence'])
        
        # Statistical weighting
        distribution_score = (
            amplification_in_range * 0.30 +      # Amplification range
            high_volatility_variance * 0.25 +    # Variance indicator
            mixed_directional_signals * 0.25 +   # Mixed signals
            volume_price_disconnect * 0.20       # Volume-price disconnect
        )
        
        supporting_evidence = {
            'amplification_range': amplification_in_range,
            'volatility_variance': high_volatility_variance,
            'signal_inconsistency': mixed_directional_signals,
            'volume_disconnect': volume_price_disconnect,
            'distribution_chaos': metrics['amplification_ratio']
        }
        
        return min(distribution_score, 1.0), supporting_evidence
    
    def classify_macro_window(self, window_data: MacroWindowData) -> PO3Classification:
        """
        Main classification function using statistical framework
        Returns PO3 phase classification with confidence scoring
        """
        
        # Calculate amplification-based metrics
        metrics = self.calculate_amplification_metrics(window_data)
        
        # Calculate phase probabilities
        accumulation_prob, acc_evidence = self.classify_accumulation_phase(metrics)
        manipulation_prob, man_evidence = self.classify_manipulation_phase(metrics)
        distribution_prob, dist_evidence = self.classify_distribution_phase(metrics)
        
        # Determine primary classification
        phase_probabilities = {
            'ACCUMULATION': accumulation_prob,
            'MANIPULATION': manipulation_prob, 
            'DISTRIBUTION': distribution_prob
        }
        
        # Handle transition states (low confidence in all phases)
        max_prob = max(phase_probabilities.values())
        if max_prob < 0.4:
            primary_phase = 'TRANSITION'
            confidence = 1.0 - max_prob  # Higher uncertainty = higher transition confidence
            supporting_metrics = {'transition_uncertainty': max_prob}
        else:
            primary_phase = max(phase_probabilities, key=phase_probabilities.get)
            confidence = max_prob
            
            # Select supporting evidence based on primary classification
            if primary_phase == 'ACCUMULATION':
                supporting_metrics = acc_evidence
            elif primary_phase == 'MANIPULATION':
                supporting_metrics = man_evidence
            else:  # DISTRIBUTION
                supporting_metrics = dist_evidence
        
        # Calculate statistical significance using t-test against baseline
        amplification_samples = [metrics['amplification_ratio'] * self.baseline_amplification]  # Single observation
        baseline_samples = [self.baseline_amplification] * 30  # Assumed baseline distribution
        
        try:
            t_stat, p_value = stats.ttest_ind(amplification_samples, baseline_samples)
            statistical_significance = 1.0 - p_value if p_value < 1.0 else 0.0
        except:
            statistical_significance = 0.0
        
        return PO3Classification(
            phase=primary_phase,
            confidence=confidence,
            amplification_ratio=metrics['amplification_ratio'],
            supporting_metrics=supporting_metrics,
            statistical_significance=statistical_significance
        )
    
    def analyze_session_sequence(self, windows: List[MacroWindowData]) -> Dict[str, any]:
        """
        Analyze sequential PO3 phase patterns across sessions
        H9.4: Sequential chain amplification cascades
        """
        
        classifications = [self.classify_macro_window(window) for window in windows]
        
        # Sequential pattern analysis
        phase_sequence = [c.phase for c in classifications]
        confidence_sequence = [c.confidence for c in classifications]
        
        # Detect common sequences (Accumulation → Manipulation → Distribution)
        amd_sequences = 0
        for i in range(len(phase_sequence) - 2):
            if (phase_sequence[i] == 'ACCUMULATION' and 
                phase_sequence[i+1] == 'MANIPULATION' and 
                phase_sequence[i+2] == 'DISTRIBUTION'):
                amd_sequences += 1
        
        # Session boundary effects
        session_transitions = {}
        for i in range(len(windows) - 1):
            current_session = windows[i].session
            next_session = windows[i+1].session
            if current_session != next_session:
                transition_key = f"{current_session}→{next_session}"
                if transition_key not in session_transitions:
                    session_transitions[transition_key] = []
                session_transitions[transition_key].append({
                    'from_phase': classifications[i].phase,
                    'to_phase': classifications[i+1].phase,
                    'amplification_change': classifications[i+1].amplification_ratio - classifications[i].amplification_ratio
                })
        
        return {
            'phase_sequence': phase_sequence,
            'confidence_sequence': confidence_sequence,
            'amd_sequences_detected': amd_sequences,
            'session_transitions': session_transitions,
            'average_confidence': np.mean(confidence_sequence),
            'sequence_coherence': self._calculate_sequence_coherence(phase_sequence)
        }
    
    def analyze_day_of_week_patterns(self, windows: List[MacroWindowData]) -> Dict[str, any]:
        """
        Analyze day-of-week PO3 behavioral patterns
        H9.5: Day-of-week PO3 amplification interactions  
        """
        
        # Group by day of week
        day_groups = {}
        for window in windows:
            dow = window.day_of_week
            if dow not in day_groups:
                day_groups[dow] = []
            day_groups[dow].append(window)
        
        # Analyze patterns by day
        day_patterns = {}
        for day, day_windows in day_groups.items():
            classifications = [self.classify_macro_window(w) for w in day_windows]
            
            # Phase distribution for this day
            phase_counts = {}
            amplification_by_phase = {}
            
            for classification in classifications:
                phase = classification.phase
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                
                if phase not in amplification_by_phase:
                    amplification_by_phase[phase] = []
                amplification_by_phase[phase].append(classification.amplification_ratio)
            
            # Calculate day-specific metrics
            day_patterns[day] = {
                'total_windows': len(day_windows),
                'phase_distribution': phase_counts,
                'dominant_phase': max(phase_counts, key=phase_counts.get) if phase_counts else 'NONE',
                'average_amplification': np.mean([c.amplification_ratio for c in classifications]),
                'amplification_by_phase': {phase: np.mean(ratios) for phase, ratios in amplification_by_phase.items()},
                'friday_distribution_boost': day == 'Friday' and phase_counts.get('DISTRIBUTION', 0) > phase_counts.get('ACCUMULATION', 0)
            }
        
        return day_patterns
    
    def validate_classification_accuracy(self, test_windows: List[MacroWindowData], true_labels: List[str]) -> Dict[str, any]:
        """
        Validate classification accuracy against known PO3 phases
        Statistical validation methodology implementation
        """
        
        # Generate predictions
        predictions = []
        confidences = []
        
        for window in test_windows:
            classification = self.classify_macro_window(window)
            predictions.append(classification.phase)
            confidences.append(classification.confidence)
        
        # Calculate accuracy metrics
        accuracy = np.mean([pred == true for pred, true in zip(predictions, true_labels)])
        
        # Phase-specific accuracy
        phase_accuracy = {}
        for phase in ['ACCUMULATION', 'MANIPULATION', 'DISTRIBUTION', 'TRANSITION']:
            phase_predictions = [pred for pred, true in zip(predictions, true_labels) if true == phase]
            phase_true = [true for true in true_labels if true == phase]
            
            if len(phase_true) > 0:
                phase_accuracy[phase] = np.mean([pred == true for pred, true in zip(phase_predictions, phase_true)])
        
        # Confusion matrix
        unique_labels = list(set(true_labels + predictions))
        conf_matrix = confusion_matrix(true_labels, predictions, labels=unique_labels)
        
        # Statistical significance tests
        # Chi-square test for classification vs random
        from scipy.stats import chi2_contingency
        contingency_table = pd.crosstab(pd.Series(true_labels), pd.Series(predictions))
        chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency_table)
        
        return {
            'overall_accuracy': accuracy,
            'phase_specific_accuracy': phase_accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'average_confidence': np.mean(confidences),
            'chi_square_statistic': chi2_stat,
            'chi_square_p_value': chi2_p_value,
            'statistical_significance': chi2_p_value < 0.05,
            'classification_report': classification_report(true_labels, predictions, output_dict=True)
        }
    
    def _calculate_sequence_coherence(self, phase_sequence: List[str]) -> float:
        """Calculate coherence score for phase sequence patterns"""
        
        if len(phase_sequence) < 3:
            return 0.0
        
        # Award points for logical sequences
        coherence_score = 0.0
        total_transitions = len(phase_sequence) - 1
        
        for i in range(len(phase_sequence) - 1):
            current = phase_sequence[i]
            next_phase = phase_sequence[i + 1]
            
            # Logical transitions get higher scores
            if current == 'ACCUMULATION' and next_phase in ['MANIPULATION', 'TRANSITION']:
                coherence_score += 1.0
            elif current == 'MANIPULATION' and next_phase in ['DISTRIBUTION', 'TRANSITION']:
                coherence_score += 1.0
            elif current == 'DISTRIBUTION' and next_phase in ['ACCUMULATION', 'TRANSITION']:
                coherence_score += 1.0
            elif current == 'TRANSITION':  # Transitions can go anywhere
                coherence_score += 0.5
            else:  # Illogical transitions
                coherence_score += 0.2
        
        return coherence_score / total_transitions if total_transitions > 0 else 0.0

def create_sample_macro_window(window_id: str, amplification: float, session: str = "NYAM") -> MacroWindowData:
    """Create sample macro window data for testing"""
    
    return MacroWindowData(
        window_id=window_id,
        timestamp=pd.Timestamp.now(),
        session=session,
        day_of_week="Monday",
        amplification=amplification,
        volume_ratio=2.5,
        price_movement=15.2,
        directional_coherence=0.75,
        f8_liquidity_spike=45.8,
        archaeological_zones_hit=["40%", "60%"],
        news_events=["CPI", "NFP"]
    )

def main():
    """Main function demonstrating PO3 statistical classification"""
    
    print("🎯 IRONFORGE PO3 Statistical Classifier - Phase II Implementation")
    print("=" * 70)
    
    # Initialize classifier
    classifier = PO3StatisticalClassifier()
    
    # Test classifications with different amplification scenarios
    test_scenarios = [
        ("Accumulation Test", 20.5),  # 20.5x = 0.40 ratio (suppressed)
        ("Manipulation Test", 89.3),  # 89.3x = 1.75 ratio (enhanced) 
        ("Distribution Test", 35.7),  # 35.7x = 0.70 ratio (moderate with variance)
        ("Baseline Test", 50.96)      # Exactly baseline
    ]
    
    print("\n📊 Statistical Classification Results:")
    print("-" * 50)
    
    for scenario_name, amplification in test_scenarios:
        sample_window = create_sample_macro_window(scenario_name.replace(" ", "_"), amplification)
        classification = classifier.classify_macro_window(sample_window)
        
        print(f"\n{scenario_name}:")
        print(f"  Amplification: {amplification}x (ratio: {classification.amplification_ratio:.2f})")
        print(f"  Classification: {classification.phase}")
        print(f"  Confidence: {classification.confidence:.3f}")
        print(f"  Statistical Significance: {classification.statistical_significance:.3f}")
        
        # Show top supporting metrics
        sorted_metrics = sorted(classification.supporting_metrics.items(), key=lambda x: x[1], reverse=True)
        print(f"  Key Supporting Metrics:")
        for metric, value in sorted_metrics[:3]:
            print(f"    {metric}: {value:.3f}")
    
    print(f"\n🎯 Baseline Amplification Reference: {BASELINE_AMPLIFICATION}x")
    print(f"📋 Classification Thresholds:")
    print(f"  Accumulation: <{ACCUMULATION_THRESHOLD:.2f} ratio ({ACCUMULATION_THRESHOLD * BASELINE_AMPLIFICATION:.1f}x)")
    print(f"  Manipulation: >{MANIPULATION_THRESHOLD:.2f} ratio ({MANIPULATION_THRESHOLD * BASELINE_AMPLIFICATION:.1f}x)")
    print(f"  Distribution: Variable with high variance")
    
    print("\n✅ Statistical Framework Ready for 89 Macro Window Analysis!")

if __name__ == "__main__":
    main()