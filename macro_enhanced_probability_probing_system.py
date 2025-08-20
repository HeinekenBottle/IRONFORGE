#!/usr/bin/env python3
"""
IRONFORGE Macro-Enhanced 70% Probability Probing System
Multi-dimensional pattern discovery engine incorporating ICT macro timing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ProbingResult:
    """Structure for probing system results"""
    pattern_id: str
    probability: float
    sample_size: int
    confidence_score: float
    trigger_conditions: Dict[str, Any]
    expected_outcome: str
    timing_window: Tuple[int, int]  # milliseconds
    macro_relevance: str
    validation_score: float

class MacroEnhancedProbingSystem:
    """70% probability pattern discovery with macro timing integration"""
    
    def __init__(self, sessions: Dict[str, pd.DataFrame], feature_stats: Dict[str, Dict]):
        self.sessions = sessions
        self.feature_stats = feature_stats
        self.probing_results = []
        
        # Macro window definitions
        self.macro_windows = {
            'macro_1': (470, 490, '07:50-08:10'),  # minutes from midnight
            'macro_2': (530, 550, '08:50-09:10'),
            'macro_3': (590, 610, '09:50-10:10'),
            'macro_4': (650, 670, '10:50-11:10'),
            'macro_5': (710, 730, '11:50-12:10')
        }
        
        # Feature importance hierarchy
        self.primary_features = ['f8', 'f9', 'f4', 'f1', 'f3']  # Top predictors
        self.semantic_features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']  # Market events
        self.secondary_features = ['f10', 'f11', 'f12', 'f13', 'f14']  # Supporting signals
        
        # Probability thresholds for multi-tier discovery
        self.probability_tiers = {
            'platinum': 0.80,  # 80%+ patterns
            'gold': 0.70,      # 70-79% patterns  
            'silver': 0.60,    # 60-69% patterns (for refinement)
            'bronze': 0.50     # 50-59% patterns (baseline)
        }
        
        print("🎯 Macro-Enhanced Probing System Initialized")
        print(f"Sessions: {len(sessions)} | Features: {len(self.primary_features)} primary")
    
    def run_comprehensive_probe(self) -> List[ProbingResult]:
        """Execute all probing strategies"""
        print("\n🔍 EXECUTING COMPREHENSIVE 70% PROBABILITY PROBE")
        print("=" * 60)
        
        all_results = []
        
        # Probe 1: Macro-timed feature spikes
        print("🕐 Probe 1: Macro-timed feature spike patterns...")
        macro_results = self._probe_macro_timed_patterns()
        all_results.extend(macro_results)
        
        # Probe 2: Sequential macro relationships
        print("🔗 Probe 2: Sequential macro relationship patterns...")
        sequence_results = self._probe_sequential_macro_patterns()
        all_results.extend(sequence_results)
        
        # Probe 3: Feature combination amplifiers
        print("🧬 Probe 3: Feature combination amplifier patterns...")
        combo_results = self._probe_feature_combination_patterns()
        all_results.extend(combo_results)
        
        # Probe 4: Time-distance correlation patterns
        print("📏 Probe 4: Time-distance correlation patterns...")
        distance_results = self._probe_time_distance_patterns()
        all_results.extend(distance_results)
        
        # Probe 5: Cross-session macro persistence
        print("🌐 Probe 5: Cross-session macro persistence patterns...")
        persistence_results = self._probe_cross_session_patterns()
        all_results.extend(persistence_results)
        
        # Probe 6: Semantic event confluence
        print("💭 Probe 6: Semantic event confluence patterns...")
        semantic_results = self._probe_semantic_confluence_patterns()
        all_results.extend(semantic_results)
        
        # Probe 7: Volume-price divergence
        print("📊 Probe 7: Volume-price divergence patterns...")
        divergence_results = self._probe_volume_price_divergence()
        all_results.extend(divergence_results)
        
        # Probe 8: Adaptive threshold optimization
        print("🎛️ Probe 8: Adaptive threshold optimization...")
        adaptive_results = self._probe_adaptive_thresholds()
        all_results.extend(adaptive_results)
        
        # Filter and rank results
        high_prob_results = [r for r in all_results if r.probability >= 0.70]
        high_prob_results.sort(key=lambda x: (x.probability, x.confidence_score), reverse=True)
        
        self.probing_results = high_prob_results
        
        print(f"\n✅ Probing Complete: {len(high_prob_results)} patterns ≥70% probability")
        return high_prob_results
    
    def _probe_macro_timed_patterns(self) -> List[ProbingResult]:
        """Probe 1: Feature spikes timed with macro windows"""
        results = []
        
        for feature in self.primary_features:
            if feature not in self.feature_stats:
                continue
                
            # Test multiple threshold levels
            for percentile in [85, 90, 95, 99]:
                threshold_key = f'q{percentile}'
                if threshold_key not in self.feature_stats[feature]:
                    continue
                    
                threshold = self.feature_stats[feature][threshold_key]
                
                # Test each macro window
                for macro_name, (start_min, end_min, display) in self.macro_windows.items():
                    patterns = self._analyze_macro_feature_pattern(
                        feature, threshold, macro_name, start_min, end_min, percentile
                    )
                    results.extend(patterns)
        
        return results
    
    def _probe_sequential_macro_patterns(self) -> List[ProbingResult]:
        """Probe 2: Patterns that span multiple macro windows"""
        results = []
        
        # Test macro sequences (e.g., Macro 2 → Macro 3)
        macro_pairs = [
            ('macro_2', 'macro_3', 'Early NY → London-NY Overlap'),
            ('macro_3', 'macro_5', 'Overlap → Lunch Approach'),
            ('macro_1', 'macro_2', 'London → Early NY')
        ]
        
        for first_macro, second_macro, description in macro_pairs:
            sequence_patterns = self._analyze_macro_sequence(first_macro, second_macro, description)
            results.extend(sequence_patterns)
        
        return results
    
    def _probe_feature_combination_patterns(self) -> List[ProbingResult]:
        """Probe 3: Multi-feature combinations for amplified signals"""
        results = []
        
        # Test feature pairs
        for f1, f2 in combinations(self.primary_features, 2):
            if f1 in self.feature_stats and f2 in self.feature_stats:
                combo_patterns = self._analyze_feature_combination(f1, f2)
                results.extend(combo_patterns)
        
        # Test semantic + quantitative combinations
        for semantic_f in self.semantic_features[:4]:  # Top 4 semantic
            for quant_f in ['f8', 'f9']:  # Primary quantitative
                if semantic_f in self.feature_stats and quant_f in self.feature_stats:
                    hybrid_patterns = self._analyze_hybrid_combination(semantic_f, quant_f)
                    results.extend(hybrid_patterns)
        
        return results
    
    def _probe_time_distance_patterns(self) -> List[ProbingResult]:
        """Probe 4: Time-distance from macro centers correlation"""
        results = []
        
        # Test different distance thresholds from macro centers
        distance_windows = [
            (0, 5, 'Macro Core'),      # Within 5 minutes of center
            (5, 10, 'Macro Edge'),     # 5-10 minutes from center
            (10, 15, 'Macro Halo'),    # 10-15 minutes from center
            (15, 30, 'Inter-Macro')    # 15-30 minutes between macros
        ]
        
        for min_dist, max_dist, zone_name in distance_windows:
            distance_patterns = self._analyze_distance_correlation(min_dist, max_dist, zone_name)
            results.extend(distance_patterns)
        
        return results
    
    def _probe_cross_session_patterns(self) -> List[ProbingResult]:
        """Probe 5: Patterns that persist across different sessions"""
        results = []
        
        # Group sessions by type
        session_groups = self._group_sessions_by_type()
        
        for group_name, session_list in session_groups.items():
            if len(session_list) >= 3:  # Need minimum sessions for validation
                cross_patterns = self._analyze_cross_session_persistence(group_name, session_list)
                results.extend(cross_patterns)
        
        return results
    
    def _probe_semantic_confluence_patterns(self) -> List[ProbingResult]:
        """Probe 6: Semantic event confluence with macro timing"""
        results = []
        
        # Test combinations of semantic events
        semantic_combos = [
            (['f0', 'f4'], 'FVG + Liquidity Sweep'),
            (['f1', 'f2'], 'Expansion + Consolidation'),
            (['f3', 'f6'], 'PD Array + HTF Confluence'),
            (['f5', 'f7'], 'Session Boundary + Structure Break')
        ]
        
        for features, description in semantic_combos:
            confluence_patterns = self._analyze_semantic_confluence(features, description)
            results.extend(confluence_patterns)
        
        return results
    
    def _probe_volume_price_divergence(self) -> List[ProbingResult]:
        """Probe 7: Volume-price divergence patterns"""
        results = []
        
        # Test f8 (liquidity) vs price movement divergences
        divergence_patterns = self._analyze_volume_price_divergence()
        results.extend(divergence_patterns)
        
        return results
    
    def _probe_adaptive_thresholds(self) -> List[ProbingResult]:
        """Probe 8: Adaptive threshold optimization"""
        results = []
        
        # Test session-specific adaptive thresholds
        for feature in ['f8', 'f9', 'f4']:
            adaptive_patterns = self._analyze_adaptive_thresholds(feature)
            results.extend(adaptive_patterns)
        
        return results
    
    # Core analysis methods
    def _analyze_macro_feature_pattern(self, feature: str, threshold: float, 
                                     macro_name: str, start_min: int, end_min: int, 
                                     percentile: int) -> List[ProbingResult]:
        """Analyze feature spike patterns during specific macro windows"""
        patterns = []
        
        outcomes = ['fpfvg_redelivery', 'expansion', 'retracement', 'consolidation']
        
        for outcome in outcomes:
            pattern_events = []
            
            for session_id, nodes in self.sessions.items():
                if feature not in nodes.columns or len(nodes) < 20:
                    continue
                
                session_patterns = self._find_macro_feature_events(
                    nodes, feature, threshold, start_min, end_min, outcome
                )
                pattern_events.extend(session_patterns)
            
            if len(pattern_events) >= 5:  # Minimum sample size
                success_count = sum(1 for p in pattern_events if p['outcome_occurred'])
                probability = success_count / len(pattern_events)
                
                if probability >= 0.50:  # Include silver tier for refinement
                    confidence = self._calculate_confidence_score(len(pattern_events), probability)
                    
                    pattern = ProbingResult(
                        pattern_id=f"{feature}_{percentile}p_{macro_name}_{outcome}",
                        probability=probability,
                        sample_size=len(pattern_events),
                        confidence_score=confidence,
                        trigger_conditions={
                            'feature': feature,
                            'threshold': threshold,
                            'percentile': percentile,
                            'macro_window': macro_name
                        },
                        expected_outcome=outcome,
                        timing_window=(300000, 900000),  # 5-15 minutes
                        macro_relevance=f"{macro_name}_timed",
                        validation_score=probability * confidence
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_macro_feature_events(self, nodes: pd.DataFrame, feature: str, threshold: float,
                                 start_min: int, end_min: int, outcome: str) -> List[Dict]:
        """Find feature events within macro windows and test outcomes"""
        events = []
        
        for idx, row in nodes.iterrows():
            if row[feature] <= threshold:
                continue
                
            # Check if in macro window
            event_time = datetime.fromtimestamp(row['t'] / 1000)
            event_minutes = event_time.hour * 60 + event_time.minute
            
            if not (start_min <= event_minutes <= end_min):
                continue
            
            # Test outcome
            outcome_occurred = self._test_outcome_occurrence(nodes, idx, outcome)
            
            events.append({
                'session_id': nodes.iloc[0].get('session_id', 'unknown'),
                'timestamp': row['t'],
                'feature_value': row[feature],
                'outcome_occurred': outcome_occurred,
                'macro_distance': min(abs(event_minutes - start_min), abs(event_minutes - end_min))
            })
        
        return events
    
    def _analyze_macro_sequence(self, first_macro: str, second_macro: str, 
                              description: str) -> List[ProbingResult]:
        """Analyze sequential macro patterns"""
        patterns = []
        
        first_window = self.macro_windows[first_macro]
        second_window = self.macro_windows[second_macro]
        
        # Look for patterns like: High f8 in first macro → Outcome in second macro
        sequence_events = []
        
        for session_id, nodes in self.sessions.items():
            if 'f8' not in nodes.columns:
                continue
                
            first_events = self._find_events_in_window(nodes, 'f8', first_window)
            second_events = self._find_events_in_window(nodes, 'f8', second_window)
            
            for first_event in first_events:
                for second_event in second_events:
                    if second_event['timestamp'] > first_event['timestamp']:
                        # Test if first event predicts second event characteristics
                        correlation_strength = self._calculate_sequence_correlation(
                            first_event, second_event
                        )
                        
                        if correlation_strength > 0.5:
                            sequence_events.append({
                                'first_event': first_event,
                                'second_event': second_event,
                                'correlation': correlation_strength
                            })
        
        if len(sequence_events) >= 5:
            high_correlation_events = [e for e in sequence_events if e['correlation'] > 0.7]
            probability = len(high_correlation_events) / len(sequence_events)
            
            if probability >= 0.50:
                confidence = self._calculate_confidence_score(len(sequence_events), probability)
                
                pattern = ProbingResult(
                    pattern_id=f"sequence_{first_macro}_{second_macro}",
                    probability=probability,
                    sample_size=len(sequence_events),
                    confidence_score=confidence,
                    trigger_conditions={
                        'first_macro': first_macro,
                        'second_macro': second_macro,
                        'description': description
                    },
                    expected_outcome='macro_sequence_correlation',
                    timing_window=(first_window[0]*60000, second_window[1]*60000),
                    macro_relevance='sequential_macros',
                    validation_score=probability * confidence
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_feature_combination(self, f1: str, f2: str) -> List[ProbingResult]:
        """Analyze feature combination amplification patterns"""
        patterns = []
        
        if f1 not in self.feature_stats or f2 not in self.feature_stats:
            return patterns
        
        f1_threshold = self.feature_stats[f1]['q90']
        f2_threshold = self.feature_stats[f2]['q90']
        
        outcomes = ['fpfvg_redelivery', 'expansion', 'retracement']
        
        for outcome in outcomes:
            combo_events = []
            
            for session_id, nodes in self.sessions.items():
                if f1 not in nodes.columns or f2 not in nodes.columns:
                    continue
                
                # Find simultaneous high values
                high_combo = nodes[
                    (nodes[f1] > f1_threshold) & 
                    (nodes[f2] > f2_threshold)
                ]
                
                for idx, row in high_combo.iterrows():
                    outcome_occurred = self._test_outcome_occurrence(nodes, idx, outcome)
                    
                    combo_events.append({
                        'session_id': session_id,
                        'f1_value': row[f1],
                        'f2_value': row[f2],
                        'outcome_occurred': outcome_occurred
                    })
            
            if len(combo_events) >= 5:
                success_count = sum(1 for e in combo_events if e['outcome_occurred'])
                probability = success_count / len(combo_events)
                
                if probability >= 0.50:
                    confidence = self._calculate_confidence_score(len(combo_events), probability)
                    
                    pattern = ProbingResult(
                        pattern_id=f"combo_{f1}_{f2}_{outcome}",
                        probability=probability,
                        sample_size=len(combo_events),
                        confidence_score=confidence,
                        trigger_conditions={
                            'feature_1': f1,
                            'feature_2': f2,
                            'f1_threshold': f1_threshold,
                            'f2_threshold': f2_threshold
                        },
                        expected_outcome=outcome,
                        timing_window=(180000, 600000),  # 3-10 minutes
                        macro_relevance='feature_amplification',
                        validation_score=probability * confidence
                    )
                    patterns.append(pattern)
        
        return patterns
    
    # Helper methods
    def _test_outcome_occurrence(self, nodes: pd.DataFrame, event_idx: int, outcome: str) -> bool:
        """Test if specific outcome occurs after event"""
        if event_idx >= len(nodes) - 5:
            return False
        
        event_time = nodes.iloc[event_idx]['t']
        event_price = nodes.iloc[event_idx]['price']
        
        # Look ahead 5-15 minutes
        future_start = event_time + 300000
        future_end = event_time + 900000
        
        future_events = nodes[
            (nodes['t'] >= future_start) & 
            (nodes['t'] <= future_end) &
            (nodes.index > event_idx)
        ]
        
        if len(future_events) == 0:
            return False
        
        if outcome == 'fpfvg_redelivery':
            # Price returns to event area
            tolerance = 15
            redelivery = future_events[abs(future_events['price'] - event_price) <= tolerance]
            return len(redelivery) > 0
        
        elif outcome == 'expansion':
            # Range expansion
            future_range = future_events['price'].max() - future_events['price'].min()
            recent_range = nodes.iloc[max(0, event_idx-10):event_idx]['price'].max() - \
                          nodes.iloc[max(0, event_idx-10):event_idx]['price'].min()
            return recent_range > 0 and future_range > recent_range * 1.5
        
        elif outcome == 'retracement':
            # Price retracement
            if event_idx < 5:
                return False
            trend_start_price = nodes.iloc[event_idx-5]['price']
            trend_move = abs(event_price - trend_start_price)
            if trend_move < 5:
                return False
            
            if event_price > trend_start_price:  # Uptrend
                min_future = future_events['price'].min()
                retracement = event_price - min_future
            else:  # Downtrend
                max_future = future_events['price'].max()
                retracement = max_future - event_price
            
            return retracement > trend_move * 0.3
        
        elif outcome == 'consolidation':
            # Range compression
            future_range = future_events['price'].max() - future_events['price'].min()
            recent_volatility = nodes.iloc[max(0, event_idx-10):event_idx]['price'].std()
            return recent_volatility > 0 and future_range < recent_volatility * 0.5
        
        return False
    
    def _calculate_confidence_score(self, sample_size: int, probability: float) -> float:
        """Calculate confidence score based on sample size and probability"""
        # Wilson score interval for confidence
        n = sample_size
        p = probability
        z = 1.96  # 95% confidence
        
        if n == 0:
            return 0.0
        
        confidence = (p + z*z/(2*n) - z * np.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)
        return max(0.0, min(1.0, confidence))
    
    # Placeholder methods for remaining probes
    def _analyze_distance_correlation(self, min_dist: int, max_dist: int, zone_name: str) -> List[ProbingResult]:
        """Placeholder for distance correlation analysis"""
        return []  # Implement based on specific requirements
    
    def _group_sessions_by_type(self) -> Dict[str, List[str]]:
        """Group sessions by type for cross-session analysis"""
        groups = {'LONDON': [], 'PREMARKET': [], 'NY_AM': [], 'NY_PM': []}
        
        for session_id in self.sessions.keys():
            if 'LONDON' in session_id:
                groups['LONDON'].append(session_id)
            elif 'PREMARKET' in session_id:
                groups['PREMARKET'].append(session_id)
            elif 'NY_AM' in session_id:
                groups['NY_AM'].append(session_id)
            elif 'NY_PM' in session_id:
                groups['NY_PM'].append(session_id)
        
        return groups
    
    def _analyze_cross_session_persistence(self, group_name: str, session_list: List[str]) -> List[ProbingResult]:
        """Placeholder for cross-session persistence analysis"""
        return []
    
    def _analyze_hybrid_combination(self, semantic_f: str, quant_f: str) -> List[ProbingResult]:
        """Placeholder for hybrid semantic-quantitative combinations"""
        return []
    
    def _analyze_semantic_confluence(self, features: List[str], description: str) -> List[ProbingResult]:
        """Placeholder for semantic confluence analysis"""
        return []
    
    def _analyze_volume_price_divergence(self) -> List[ProbingResult]:
        """Placeholder for volume-price divergence analysis"""
        return []
    
    def _analyze_adaptive_thresholds(self, feature: str) -> List[ProbingResult]:
        """Placeholder for adaptive threshold analysis"""
        return []
    
    def _find_events_in_window(self, nodes: pd.DataFrame, feature: str, window: Tuple) -> List[Dict]:
        """Find feature events within time window"""
        return []  # Simplified for placeholder
    
    def _calculate_sequence_correlation(self, first_event: Dict, second_event: Dict) -> float:
        """Calculate correlation between sequential events"""
        return 0.5  # Simplified for placeholder

def run_macro_probability_probe():
    """Main function to run the macro-enhanced probability probing system"""
    from predictive_condition_hunter import PredictiveConditionHunter
    
    print("🚀 INITIALIZING MACRO-ENHANCED PROBABILITY PROBING SYSTEM")
    print("=" * 70)
    
    # Initialize base system
    hunter = PredictiveConditionHunter()
    
    # Create probing system
    probing_system = MacroEnhancedProbingSystem(
        hunter.engine.sessions, 
        hunter.core_analyzer.feature_stats
    )
    
    # Run comprehensive probe
    results = probing_system.run_comprehensive_probe()
    
    # Display results
    print(f"\n🏆 TOP 70%+ PROBABILITY PATTERNS DISCOVERED:")
    print("=" * 60)
    
    for i, result in enumerate(results[:10], 1):  # Top 10
        print(f"\n{i}. {result.pattern_id}")
        print(f"   Probability: {result.probability:.1%}")
        print(f"   Sample Size: {result.sample_size}")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Expected: {result.expected_outcome}")
        print(f"   Macro Relevance: {result.macro_relevance}")
        print(f"   Validation Score: {result.validation_score:.3f}")
    
    # Summary by probability tier
    platinum = [r for r in results if r.probability >= 0.80]
    gold = [r for r in results if 0.70 <= r.probability < 0.80]
    
    print(f"\n📊 DISCOVERY SUMMARY:")
    print(f"🥇 Platinum (80%+): {len(platinum)} patterns")
    print(f"🥇 Gold (70-79%): {len(gold)} patterns")
    print(f"Total actionable patterns: {len(results)}")
    
    return results

if __name__ == "__main__":
    results = run_macro_probability_probe()