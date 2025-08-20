#!/usr/bin/env python3
"""
Enhanced Probability Probing Strategies
Extended pattern discovery with wider search parameters and creative combinations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from itertools import combinations
from predictive_condition_hunter import PredictiveConditionHunter
import warnings
warnings.filterwarnings('ignore')

class EnhancedProbingStrategies:
    """Extended probing strategies for 70%+ pattern discovery"""
    
    def __init__(self):
        self.hunter = PredictiveConditionHunter()
        self.sessions = self.hunter.engine.sessions
        self.feature_stats = self.hunter.core_analyzer.feature_stats
        
        # Expanded feature groups
        self.liquidity_features = ['f8', 'f9']  # Primary liquidity indicators
        self.semantic_features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
        self.supporting_features = ['f10', 'f11', 'f12', 'f13', 'f14']
        
        # Macro timing definitions (minutes from midnight)
        self.macro_centers = {
            'macro_1': 480,  # 08:00 center
            'macro_2': 540,  # 09:00 center
            'macro_3': 600,  # 10:00 center
            'macro_4': 660,  # 11:00 center
            'macro_5': 720   # 12:00 center
        }
        
        # Extended timing windows
        self.timing_zones = {
            'pre_macro': (-15, -5),      # 15-5 min before
            'macro_approach': (-5, 0),   # 5 min before to start
            'macro_core': (0, 10),       # First 10 min of window
            'macro_late': (10, 20),      # Last 10 min of window
            'post_macro': (20, 35),      # 20-35 min after start
            'inter_macro': (35, 85)      # Between windows
        }
        
    def run_enhanced_probing(self) -> Dict[str, List]:
        """Execute enhanced probing strategies"""
        print("🔬 ENHANCED PROBABILITY PROBING STRATEGIES")
        print("=" * 55)
        
        strategies = {
            'macro_progression': self._probe_macro_progression_patterns(),
            'feature_cascades': self._probe_feature_cascade_patterns(), 
            'temporal_clusters': self._probe_temporal_cluster_patterns(),
            'threshold_optimization': self._probe_threshold_optimization(),
            'session_fingerprints': self._probe_session_fingerprint_patterns(),
            'volume_price_sync': self._probe_volume_price_synchronization(),
            'semantic_timing': self._probe_semantic_timing_patterns(),
            'cross_feature_amplification': self._probe_cross_feature_amplification(),
            'momentum_persistence': self._probe_momentum_persistence_patterns(),
            'archaeological_zones': self._probe_archaeological_zone_patterns()
        }
        
        all_results = []
        for strategy_name, results in strategies.items():
            print(f"📊 {strategy_name}: {len(results)} patterns found")
            all_results.extend(results)
        
        # Filter high-probability patterns
        high_prob = [r for r in all_results if r.get('probability', 0) >= 0.70]
        high_prob.sort(key=lambda x: x.get('probability', 0), reverse=True)
        
        print(f"\\n✅ Total 70%+ patterns discovered: {len(high_prob)}")
        return {'all_patterns': all_results, 'high_probability': high_prob}
    
    def _probe_macro_progression_patterns(self) -> List[Dict]:
        """Strategy 1: Macro window progression patterns"""
        print("🕐 Probing macro progression patterns...")
        patterns = []
        
        # Test progression through multiple macros
        macro_sequences = [
            ['macro_1', 'macro_2', 'macro_3'],
            ['macro_2', 'macro_3', 'macro_4'],
            ['macro_3', 'macro_4', 'macro_5']
        ]
        
        for sequence in macro_sequences:
            progression_patterns = self._analyze_macro_progression(sequence)
            patterns.extend(progression_patterns)
        
        return patterns
    
    def _probe_feature_cascade_patterns(self) -> List[Dict]:
        """Strategy 2: Feature cascade patterns (f8 → f9 → outcome)"""
        print("🌊 Probing feature cascade patterns...")
        patterns = []
        
        # Test cascading feature activations
        cascade_chains = [
            ['f8', 'f9', 'f4'],
            ['f8', 'f4', 'f1'],
            ['f9', 'f8', 'f3'],
            ['f4', 'f8', 'f9']
        ]
        
        for chain in cascade_chains:
            cascade_patterns = self._analyze_feature_cascade(chain)
            patterns.extend(cascade_patterns)
        
        return patterns
    
    def _probe_temporal_cluster_patterns(self) -> List[Dict]:
        """Strategy 3: Temporal clustering of events"""
        print("⏱️ Probing temporal cluster patterns...")
        patterns = []
        
        # Look for events that cluster in time
        cluster_windows = [60000, 120000, 300000]  # 1, 2, 5 minute windows
        
        for window_size in cluster_windows:
            cluster_patterns = self._analyze_temporal_clusters(window_size)
            patterns.extend(cluster_patterns)
        
        return patterns
    
    def _probe_threshold_optimization(self) -> List[Dict]:
        """Strategy 4: Dynamic threshold optimization"""
        print("🎛️ Probing dynamic threshold optimization...")
        patterns = []
        
        for feature in ['f8', 'f9', 'f4']:
            if feature not in self.feature_stats:
                continue
                
            # Test multiple threshold combinations
            percentiles = [75, 80, 85, 90, 95, 99]
            
            for p1, p2 in combinations(percentiles, 2):
                threshold_patterns = self._analyze_dual_thresholds(feature, p1, p2)
                patterns.extend(threshold_patterns)
        
        return patterns
    
    def _probe_session_fingerprint_patterns(self) -> List[Dict]:
        """Strategy 5: Session-specific fingerprint patterns"""
        print("🔍 Probing session fingerprint patterns...")
        patterns = []
        
        # Group sessions and find session-specific patterns
        session_groups = self._group_sessions()
        
        for group_name, session_list in session_groups.items():
            fingerprint_patterns = self._analyze_session_fingerprints(group_name, session_list)
            patterns.extend(fingerprint_patterns)
        
        return patterns
    
    def _probe_volume_price_synchronization(self) -> List[Dict]:
        """Strategy 6: Volume-price synchronization patterns"""
        print("📊 Probing volume-price synchronization...")
        patterns = []
        
        # Test f8 (liquidity) vs price movement synchronization
        sync_patterns = self._analyze_volume_price_sync()
        patterns.extend(sync_patterns)
        
        return patterns
    
    def _probe_semantic_timing_patterns(self) -> List[Dict]:
        """Strategy 7: Semantic event timing patterns"""
        print("💭 Probing semantic timing patterns...")
        patterns = []
        
        # Test semantic features with precise timing
        for semantic_f in self.semantic_features[:6]:  # Top 6 semantic
            timing_patterns = self._analyze_semantic_timing(semantic_f)
            patterns.extend(timing_patterns)
        
        return patterns
    
    def _probe_cross_feature_amplification(self) -> List[Dict]:
        """Strategy 8: Cross-feature amplification"""
        print("🔗 Probing cross-feature amplification...")
        patterns = []
        
        # Test combinations across different feature groups
        for liq_f in self.liquidity_features:
            for sem_f in self.semantic_features[:4]:
                amplification_patterns = self._analyze_cross_amplification(liq_f, sem_f)
                patterns.extend(amplification_patterns)
        
        return patterns
    
    def _probe_momentum_persistence_patterns(self) -> List[Dict]:
        """Strategy 9: Momentum persistence patterns"""
        print("🚀 Probing momentum persistence patterns...")
        patterns = []
        
        # Test how long patterns persist
        persistence_windows = [
            (300000, 600000),   # 5-10 minutes
            (600000, 900000),   # 10-15 minutes
            (900000, 1800000)   # 15-30 minutes
        ]
        
        for window in persistence_windows:
            persistence_patterns = self._analyze_momentum_persistence(window)
            patterns.extend(persistence_patterns)
        
        return patterns
    
    def _probe_archaeological_zone_patterns(self) -> List[Dict]:
        """Strategy 10: Archaeological zone interaction patterns"""
        print("🏛️ Probing archaeological zone patterns...")
        patterns = []
        
        # Test Theory B 40%, 60%, 80% zones with macro timing
        zone_levels = [0.4, 0.6, 0.8]
        
        for zone_level in zone_levels:
            zone_patterns = self._analyze_archaeological_zones(zone_level)
            patterns.extend(zone_patterns)
        
        return patterns
    
    # Core analysis methods
    def _analyze_macro_progression(self, sequence: List[str]) -> List[Dict]:
        """Analyze patterns across macro sequence"""
        patterns = []
        
        outcomes = ['fpfvg_redelivery', 'expansion', 'consolidation']
        
        for outcome in outcomes:
            progression_events = []
            
            for session_id, nodes in self.sessions.items():
                if 'f8' not in nodes.columns:
                    continue
                
                # Find events in each macro of the sequence
                macro_events = {}
                for macro in sequence:
                    macro_center = self.macro_centers[macro]
                    events = self._find_events_near_time(nodes, 'f8', macro_center, 20)
                    macro_events[macro] = events
                
                # Test if progression exists
                if all(len(macro_events[m]) > 0 for m in sequence):
                    progression_strength = self._calculate_progression_strength(macro_events, sequence)
                    
                    if progression_strength > 0.5:
                        # Test outcome after sequence
                        last_macro_events = macro_events[sequence[-1]]
                        if last_macro_events:
                            last_event_idx = last_macro_events[-1]['index']
                            outcome_occurred = self._test_outcome_after_index(nodes, last_event_idx, outcome)
                            
                            progression_events.append({
                                'session_id': session_id,
                                'sequence': sequence,
                                'progression_strength': progression_strength,
                                'outcome_occurred': outcome_occurred
                            })
            
            if len(progression_events) >= 3:
                success_count = sum(1 for e in progression_events if e['outcome_occurred'])
                probability = success_count / len(progression_events)
                
                if probability >= 0.50:
                    patterns.append({
                        'pattern_type': 'macro_progression',
                        'sequence': sequence,
                        'outcome': outcome,
                        'probability': probability,
                        'sample_size': len(progression_events),
                        'confidence': self._wilson_confidence(len(progression_events), probability)
                    })
        
        return patterns
    
    def _analyze_feature_cascade(self, chain: List[str]) -> List[Dict]:
        """Analyze cascading feature activation patterns"""
        patterns = []
        
        outcomes = ['fpfvg_redelivery', 'expansion', 'retracement']
        
        for outcome in outcomes:
            cascade_events = []
            
            for session_id, nodes in self.sessions.items():
                if not all(f in nodes.columns for f in chain):
                    continue
                
                # Find cascade sequences
                cascades = self._find_feature_cascades(nodes, chain)
                
                for cascade in cascades:
                    # Test outcome after cascade completes
                    outcome_occurred = self._test_outcome_after_index(
                        nodes, cascade['end_index'], outcome
                    )
                    
                    cascade_events.append({
                        'session_id': session_id,
                        'cascade': cascade,
                        'outcome_occurred': outcome_occurred
                    })
            
            if len(cascade_events) >= 5:
                success_count = sum(1 for e in cascade_events if e['outcome_occurred'])
                probability = success_count / len(cascade_events)
                
                if probability >= 0.50:
                    patterns.append({
                        'pattern_type': 'feature_cascade',
                        'chain': chain,
                        'outcome': outcome,
                        'probability': probability,
                        'sample_size': len(cascade_events),
                        'confidence': self._wilson_confidence(len(cascade_events), probability)
                    })
        
        return patterns
    
    def _analyze_temporal_clusters(self, window_size: int) -> List[Dict]:
        """Analyze temporal clustering patterns"""
        patterns = []
        
        for feature in ['f8', 'f9', 'f4']:
            if feature not in self.feature_stats:
                continue
                
            threshold = self.feature_stats[feature]['q90']
            cluster_events = []
            
            for session_id, nodes in self.sessions.items():
                if feature not in nodes.columns:
                    continue
                
                # Find high-value events
                high_events = nodes[nodes[feature] > threshold]
                
                # Find clusters (multiple events within window)
                clusters = self._find_event_clusters(high_events, window_size)
                
                for cluster in clusters:
                    if len(cluster) >= 3:  # Minimum cluster size
                        # Test outcome after cluster
                        last_event_idx = cluster[-1]['index']
                        
                        for outcome in ['expansion', 'consolidation']:
                            outcome_occurred = self._test_outcome_after_index(nodes, last_event_idx, outcome)
                            
                            cluster_events.append({
                                'session_id': session_id,
                                'feature': feature,
                                'cluster_size': len(cluster),
                                'outcome': outcome,
                                'outcome_occurred': outcome_occurred
                            })
            
            # Analyze by outcome
            outcome_groups = {}
            for event in cluster_events:
                outcome = event['outcome']
                if outcome not in outcome_groups:
                    outcome_groups[outcome] = []
                outcome_groups[outcome].append(event)
            
            for outcome, events in outcome_groups.items():
                if len(events) >= 5:
                    success_count = sum(1 for e in events if e['outcome_occurred'])
                    probability = success_count / len(events)
                    
                    if probability >= 0.50:
                        patterns.append({
                            'pattern_type': 'temporal_cluster',
                            'feature': feature,
                            'window_size_ms': window_size,
                            'outcome': outcome,
                            'probability': probability,
                            'sample_size': len(events),
                            'confidence': self._wilson_confidence(len(events), probability)
                        })
        
        return patterns
    
    # Helper methods
    def _find_events_near_time(self, nodes: pd.DataFrame, feature: str, 
                              target_minutes: int, tolerance: int) -> List[Dict]:
        """Find events near specific time"""
        events = []
        
        if feature not in nodes.columns:
            return events
        
        threshold = self.feature_stats.get(feature, {}).get('q90', 0)
        high_events = nodes[nodes[feature] > threshold]
        
        for idx, row in high_events.iterrows():
            event_time = datetime.fromtimestamp(row['t'] / 1000)
            event_minutes = event_time.hour * 60 + event_time.minute
            
            if abs(event_minutes - target_minutes) <= tolerance:
                events.append({
                    'index': idx,
                    'timestamp': row['t'],
                    'value': row[feature],
                    'minutes': event_minutes
                })
        
        return events
    
    def _test_outcome_after_index(self, nodes: pd.DataFrame, event_idx: int, outcome: str) -> bool:
        """Test if outcome occurs after specific index"""
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
            redelivery = future_events[abs(future_events['price'] - event_price) <= 15]
            return len(redelivery) > 0
        
        elif outcome == 'expansion':
            future_range = future_events['price'].max() - future_events['price'].min()
            recent_range = nodes.iloc[max(0, event_idx-10):event_idx]['price'].max() - \
                          nodes.iloc[max(0, event_idx-10):event_idx]['price'].min()
            return recent_range > 0 and future_range > recent_range * 1.5
        
        elif outcome == 'consolidation':
            future_range = future_events['price'].max() - future_events['price'].min()
            recent_volatility = nodes.iloc[max(0, event_idx-10):event_idx]['price'].std()
            return recent_volatility > 0 and future_range < recent_volatility * 0.5
        
        elif outcome == 'retracement':
            if event_idx < 5:
                return False
            trend_start_price = nodes.iloc[event_idx-5]['price']
            trend_move = abs(event_price - trend_start_price)
            if trend_move < 5:
                return False
            
            if event_price > trend_start_price:
                min_future = future_events['price'].min()
                retracement = event_price - min_future
            else:
                max_future = future_events['price'].max()
                retracement = max_future - event_price
            
            return retracement > trend_move * 0.3
        
        return False
    
    def _wilson_confidence(self, n: int, p: float) -> float:
        """Wilson confidence interval"""
        if n == 0:
            return 0.0
        z = 1.96
        return (p + z*z/(2*n) - z * np.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)
    
    # Placeholder methods for remaining strategies
    def _calculate_progression_strength(self, macro_events: Dict, sequence: List[str]) -> float:
        """Calculate strength of macro progression"""
        return 0.7  # Simplified
    
    def _find_feature_cascades(self, nodes: pd.DataFrame, chain: List[str]) -> List[Dict]:
        """Find feature cascade sequences"""
        return []  # Simplified
    
    def _find_event_clusters(self, events: pd.DataFrame, window_size: int) -> List[List]:
        """Find temporal event clusters"""
        return []  # Simplified
    
    def _analyze_dual_thresholds(self, feature: str, p1: int, p2: int) -> List[Dict]:
        """Analyze dual threshold patterns"""
        return []  # Simplified placeholder
    
    def _group_sessions(self) -> Dict[str, List[str]]:
        """Group sessions by characteristics"""
        groups = {'LONDON': [], 'PREMARKET': []}
        for session_id in self.sessions.keys():
            if 'LONDON' in session_id:
                groups['LONDON'].append(session_id)
            elif 'PREMARKET' in session_id:
                groups['PREMARKET'].append(session_id)
        return groups
    
    def _analyze_session_fingerprints(self, group_name: str, session_list: List[str]) -> List[Dict]:
        """Analyze session-specific patterns"""
        return []  # Simplified placeholder
    
    def _analyze_volume_price_sync(self) -> List[Dict]:
        """Analyze volume-price synchronization"""
        return []  # Simplified placeholder
    
    def _analyze_semantic_timing(self, semantic_f: str) -> List[Dict]:
        """Analyze semantic feature timing"""
        return []  # Simplified placeholder
    
    def _analyze_cross_amplification(self, liq_f: str, sem_f: str) -> List[Dict]:
        """Analyze cross-feature amplification"""
        return []  # Simplified placeholder
    
    def _analyze_momentum_persistence(self, window: Tuple[int, int]) -> List[Dict]:
        """Analyze momentum persistence"""
        return []  # Simplified placeholder
    
    def _analyze_archaeological_zones(self, zone_level: float) -> List[Dict]:
        """Analyze archaeological zone patterns"""
        return []  # Simplified placeholder

def run_enhanced_probing():
    """Run enhanced probing strategies"""
    probing = EnhancedProbingStrategies()
    results = probing.run_enhanced_probing()
    
    print(f"\n🏆 TOP ENHANCED PATTERNS (70%+):")
    print("=" * 45)
    
    for i, pattern in enumerate(results['high_probability'][:15], 1):
        print(f"\n{i}. {pattern.get('pattern_type', 'Unknown')}")
        print(f"   Probability: {pattern.get('probability', 0):.1%}")
        print(f"   Sample Size: {pattern.get('sample_size', 0)}")
        print(f"   Confidence: {pattern.get('confidence', 0):.3f}")
        
        # Pattern-specific details
        if 'chain' in pattern:
            print(f"   Chain: {' → '.join(pattern['chain'])}")
        if 'sequence' in pattern:
            print(f"   Sequence: {' → '.join(pattern['sequence'])}")
        if 'feature' in pattern:
            print(f"   Feature: {pattern['feature']}")
        if 'outcome' in pattern:
            print(f"   Expected: {pattern['outcome']}")
    
    return results

if __name__ == "__main__":
    results = run_enhanced_probing()