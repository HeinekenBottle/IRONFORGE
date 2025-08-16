#!/usr/bin/env python3
"""
IRONFORGE Directional Move Detector
===================================

Advanced directional move detection and analysis system for PM event patterns.
Provides sophisticated post-event movement analysis with archaeological intelligence.

Capabilities:
1. Multi-scale move detection (1m, 5m, 15m horizons)
2. Volatility expansion analysis with baseline comparisons
3. Range position progression tracking
4. Move type classification (expansion, breakout, cascade, reversal)
5. Significance scoring with archaeological calibration
6. Momentum persistence analysis
7. Exhaustion pattern detection

Based on PM session archaeological discoveries:
- Events at minutes 126-129 (15:36-15:39 ET) 
- 2.5-3.5 minute event clusters
- Directional moves within 10-15 minutes post-cluster
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
from datetime import datetime, timedelta
from enum import Enum

class MoveType(Enum):
    """Types of directional moves"""
    EXPANSION = "expansion"
    BREAKOUT = "breakout" 
    CASCADE = "cascade"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    EXHAUSTION = "exhaustion"

class MoveSignificance(Enum):
    """Move significance levels"""
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    MAJOR = "major"
    EXTREME = "extreme"

@dataclass
class MoveCharacteristics:
    """Characteristics of a directional move"""
    price_range_change: float
    volatility_expansion_ratio: float
    momentum_persistence: float
    range_position_progression: List[float]
    event_density: float
    time_acceleration: float
    exhaustion_signals: List[str]

@dataclass
class DirectionalMoveSignature:
    """Advanced directional move signature"""
    move_id: str
    session_date: str
    move_start_minute: float
    move_end_minute: float
    move_duration: float
    move_type: MoveType
    significance: MoveSignificance
    characteristics: MoveCharacteristics
    archaeological_match: Optional[str]
    prediction_accuracy: float
    
    # Multi-scale analysis
    scale_1m_metrics: Dict[str, float]
    scale_5m_metrics: Dict[str, float] 
    scale_15m_metrics: Dict[str, float]
    
    # Context analysis
    pre_move_context: Dict[str, any]
    move_context: List[str]
    post_move_implications: List[str]

@dataclass
class MovePrediction:
    """Movement prediction based on archaeological intelligence"""
    predicted_move_type: MoveType
    predicted_significance: MoveSignificance
    confidence_score: float
    expected_duration_range: Tuple[float, float]
    expected_volatility_expansion: float
    archaeological_basis: str
    risk_factors: List[str]

class DirectionalMoveDetector:
    """
    Advanced directional move detection system
    """
    
    def __init__(self, archaeological_data_path: str = None):
        self.logger = logging.getLogger('directional_move_detector')
        
        # Archaeological intelligence from PM discoveries
        self.archaeological_patterns = self._load_archaeological_move_patterns()
        self.volatility_baselines = self._initialize_volatility_baselines()
        self.significance_thresholds = self._initialize_significance_thresholds()
        
        print(f"🎯 Directional Move Detector initialized")
        print(f"  Archaeological patterns loaded: {len(self.archaeological_patterns)}")
    
    def _load_archaeological_move_patterns(self) -> Dict[str, Dict]:
        """Load archaeological movement patterns from PM discoveries"""
        return {
            "pm_126_129_expansion": {
                "typical_duration": 8.5,  # minutes
                "volatility_expansion_avg": 2.8,
                "range_progression_avg": 0.15,  # 15% range movement
                "success_indicators": ["volatility_spike", "range_breakout", "momentum_acceleration"],
                "exhaustion_signals": ["volatility_decline", "range_stall", "reversal_patterns"],
                "archaeological_frequency": 0.73  # 73% of PM events show this pattern
            },
            "pm_cascade_pattern": {
                "typical_duration": 12.3,
                "volatility_expansion_avg": 4.2,
                "range_progression_avg": 0.25,
                "success_indicators": ["multi_wave_expansion", "persistent_directional_bias", "increasing_event_density"],
                "exhaustion_signals": ["wave_divergence", "momentum_fatigue", "consolidation_formation"],
                "archaeological_frequency": 0.45
            },
            "pm_breakout_signature": {
                "typical_duration": 6.8,
                "volatility_expansion_avg": 3.5,
                "range_progression_avg": 0.18,
                "success_indicators": ["range_level_breach", "volume_confirmation", "follow_through"],
                "exhaustion_signals": ["false_breakout", "immediate_reversion", "weak_follow_through"],
                "archaeological_frequency": 0.38
            },
            "pm_exhaustion_reversal": {
                "typical_duration": 4.2,
                "volatility_expansion_avg": 1.8,
                "range_progression_avg": 0.08,
                "success_indicators": ["momentum_divergence", "reversal_candlestick", "support_resistance_test"],
                "exhaustion_signals": ["trend_continuation", "breakout_failure", "consolidation_extension"],
                "archaeological_frequency": 0.28
            }
        }
    
    def _initialize_volatility_baselines(self) -> Dict[str, float]:
        """Initialize volatility baselines for different PM phases"""
        return {
            "pm_early": 0.008,      # 0.8% baseline volatility
            "pm_middle": 0.012,     # 1.2% baseline
            "pm_late": 0.018,       # 1.8% baseline (126-129 window)
            "pm_closing": 0.025     # 2.5% baseline
        }
    
    def _initialize_significance_thresholds(self) -> Dict[str, Dict]:
        """Initialize significance scoring thresholds"""
        return {
            "volatility_expansion": {
                MoveSignificance.MINOR: 1.2,
                MoveSignificance.MODERATE: 1.8,
                MoveSignificance.SIGNIFICANT: 2.5,
                MoveSignificance.MAJOR: 3.5,
                MoveSignificance.EXTREME: 5.0
            },
            "range_progression": {
                MoveSignificance.MINOR: 0.03,      # 3%
                MoveSignificance.MODERATE: 0.08,   # 8%
                MoveSignificance.SIGNIFICANT: 0.15, # 15%
                MoveSignificance.MAJOR: 0.25,      # 25%
                MoveSignificance.EXTREME: 0.40     # 40%
            },
            "duration_factor": {
                MoveSignificance.MINOR: 2.0,       # minutes
                MoveSignificance.MODERATE: 4.0,
                MoveSignificance.SIGNIFICANT: 8.0,
                MoveSignificance.MAJOR: 12.0,
                MoveSignificance.EXTREME: 18.0
            }
        }
    
    def detect_directional_move(self, events: List[Dict], cluster_end_minute: float, 
                              session_date: str, baseline_context: Dict) -> Optional[DirectionalMoveSignature]:
        """Detect and analyze directional move following event cluster"""
        
        # Define search window (10-15 minutes after cluster)
        search_start = cluster_end_minute + 10
        search_end = cluster_end_minute + 15
        
        # Find events in directional move window
        move_events = [
            event for event in events
            if search_start <= event.get('time_minutes', 0) <= search_end
        ]
        
        if len(move_events) < 3:  # Need minimum events for analysis
            return None
        
        # Sort events by time
        move_events.sort(key=lambda e: e.get('time_minutes', 0))
        
        # Analyze move characteristics
        characteristics = self._analyze_move_characteristics(move_events, baseline_context)
        
        # Determine move type
        move_type = self._classify_move_type(characteristics, move_events)
        
        # Calculate significance
        significance = self._calculate_move_significance(characteristics, move_type)
        
        # Multi-scale analysis
        scale_1m = self._analyze_1m_scale(move_events, characteristics)
        scale_5m = self._analyze_5m_scale(move_events, characteristics)
        scale_15m = self._analyze_15m_scale(move_events, characteristics)
        
        # Archaeological matching
        archaeological_match, prediction_accuracy = self._match_archaeological_pattern(
            characteristics, move_type, significance
        )
        
        # Context analysis
        pre_move_context = self._analyze_pre_move_context(baseline_context)
        move_context = self._extract_move_context(move_events)
        post_move_implications = self._generate_post_move_implications(
            move_type, significance, characteristics
        )
        
        signature = DirectionalMoveSignature(
            move_id=f"{session_date}_{cluster_end_minute:.1f}_move",
            session_date=session_date,
            move_start_minute=move_events[0].get('time_minutes', 0),
            move_end_minute=move_events[-1].get('time_minutes', 0),
            move_duration=move_events[-1].get('time_minutes', 0) - move_events[0].get('time_minutes', 0),
            move_type=move_type,
            significance=significance,
            characteristics=characteristics,
            archaeological_match=archaeological_match,
            prediction_accuracy=prediction_accuracy,
            scale_1m_metrics=scale_1m,
            scale_5m_metrics=scale_5m,
            scale_15m_metrics=scale_15m,
            pre_move_context=pre_move_context,
            move_context=move_context,
            post_move_implications=post_move_implications
        )
        
        return signature
    
    def _analyze_move_characteristics(self, move_events: List[Dict], baseline_context: Dict) -> MoveCharacteristics:
        """Analyze comprehensive move characteristics"""
        
        # Price range analysis
        price_levels = [event.get('price_level', 0) for event in move_events if event.get('price_level')]
        range_positions = [event.get('range_position', 0) for event in move_events if event.get('range_position') is not None]
        
        price_range_change = max(price_levels) - min(price_levels) if price_levels else 0.0
        
        # Volatility analysis
        volatilities = [event.get('volatility_window', 0) for event in move_events if event.get('volatility_window')]
        baseline_volatility = baseline_context.get('avg_volatility', 0.015)
        max_volatility = max(volatilities) if volatilities else 0.0
        volatility_expansion_ratio = max_volatility / baseline_volatility if baseline_volatility > 0 else 1.0
        
        # Momentum persistence
        momentum_persistence = self._calculate_momentum_persistence(move_events)
        
        # Event density (events per minute)
        duration = move_events[-1].get('time_minutes', 0) - move_events[0].get('time_minutes', 0)
        event_density = len(move_events) / duration if duration > 0 else 0.0
        
        # Time acceleration (increasing event frequency)
        time_acceleration = self._calculate_time_acceleration(move_events)
        
        # Exhaustion signals
        exhaustion_signals = self._detect_exhaustion_signals(move_events, volatilities)
        
        return MoveCharacteristics(
            price_range_change=price_range_change,
            volatility_expansion_ratio=volatility_expansion_ratio,
            momentum_persistence=momentum_persistence,
            range_position_progression=range_positions,
            event_density=event_density,
            time_acceleration=time_acceleration,
            exhaustion_signals=exhaustion_signals
        )
    
    def _calculate_momentum_persistence(self, move_events: List[Dict]) -> float:
        """Calculate momentum persistence score"""
        if len(move_events) < 3:
            return 0.0
        
        # Analyze price delta consistency
        deltas_1m = [event.get('price_delta_1m', 0) for event in move_events]
        
        # Calculate directional consistency
        positive_deltas = sum(1 for d in deltas_1m if d > 0)
        negative_deltas = sum(1 for d in deltas_1m if d < 0)
        total_deltas = len(deltas_1m)
        
        directional_bias = max(positive_deltas, negative_deltas) / total_deltas if total_deltas > 0 else 0.0
        
        # Calculate magnitude consistency (standard deviation of absolute deltas)
        abs_deltas = [abs(d) for d in deltas_1m if d != 0]
        if len(abs_deltas) < 2:
            magnitude_consistency = 0.0
        else:
            magnitude_consistency = 1.0 - (np.std(abs_deltas) / np.mean(abs_deltas)) if np.mean(abs_deltas) > 0 else 0.0
        
        return (directional_bias * 0.7 + magnitude_consistency * 0.3)
    
    def _calculate_time_acceleration(self, move_events: List[Dict]) -> float:
        """Calculate time acceleration factor"""
        if len(move_events) < 4:
            return 1.0
        
        # Split events into two halves
        mid_point = len(move_events) // 2
        first_half = move_events[:mid_point]
        second_half = move_events[mid_point:]
        
        # Calculate event density for each half
        first_duration = first_half[-1].get('time_minutes', 0) - first_half[0].get('time_minutes', 0)
        second_duration = second_half[-1].get('time_minutes', 0) - second_half[0].get('time_minutes', 0)
        
        if first_duration <= 0 or second_duration <= 0:
            return 1.0
        
        first_density = len(first_half) / first_duration
        second_density = len(second_half) / second_duration
        
        return second_density / first_density if first_density > 0 else 1.0
    
    def _detect_exhaustion_signals(self, move_events: List[Dict], volatilities: List[float]) -> List[str]:
        """Detect exhaustion signals in the move"""
        signals = []
        
        if not volatilities or len(volatilities) < 3:
            return signals
        
        # Declining volatility in final third
        final_third_start = len(volatilities) * 2 // 3
        final_volatilities = volatilities[final_third_start:]
        initial_volatilities = volatilities[:len(volatilities)//3]
        
        if final_volatilities and initial_volatilities:
            final_avg = np.mean(final_volatilities)
            initial_avg = np.mean(initial_volatilities)
            
            if final_avg < initial_avg * 0.7:  # 30% decline
                signals.append("volatility_decline")
        
        # Range stall detection
        range_positions = [event.get('range_position', 0) for event in move_events[-3:]]  # Last 3 events
        if len(set([round(pos, 2) for pos in range_positions])) == 1:  # Same range position
            signals.append("range_stall")
        
        # Momentum divergence
        deltas = [event.get('price_delta_1m', 0) for event in move_events]
        if len(deltas) >= 3:
            recent_deltas = deltas[-3:]
            if all(abs(d) < 0.001 for d in recent_deltas):  # Very small movements
                signals.append("momentum_fatigue")
        
        return signals
    
    def _classify_move_type(self, characteristics: MoveCharacteristics, move_events: List[Dict]) -> MoveType:
        """Classify the type of directional move"""
        
        # Extract context clues
        contexts = [event.get('context', '').lower() for event in move_events]
        combined_context = ' '.join(contexts)
        
        # Rule-based classification
        if 'expansion' in combined_context and characteristics.volatility_expansion_ratio > 2.0:
            return MoveType.EXPANSION
        elif 'breakout' in combined_context and characteristics.price_range_change > 0.1:
            return MoveType.BREAKOUT
        elif 'cascade' in combined_context or characteristics.time_acceleration > 1.5:
            return MoveType.CASCADE
        elif 'reversal' in combined_context or len(characteristics.exhaustion_signals) >= 2:
            return MoveType.REVERSAL
        elif len(characteristics.exhaustion_signals) >= 3:
            return MoveType.EXHAUSTION
        elif characteristics.volatility_expansion_ratio < 1.5 and characteristics.price_range_change < 0.05:
            return MoveType.CONSOLIDATION
        else:
            # Default classification based on characteristics
            if characteristics.volatility_expansion_ratio > 3.0:
                return MoveType.EXPANSION
            elif characteristics.momentum_persistence > 0.8:
                return MoveType.CASCADE
            else:
                return MoveType.CONSOLIDATION
    
    def _calculate_move_significance(self, characteristics: MoveCharacteristics, move_type: MoveType) -> MoveSignificance:
        """Calculate overall move significance"""
        
        # Volatility significance
        vol_thresholds = self.significance_thresholds["volatility_expansion"]
        vol_significance = MoveSignificance.MINOR
        for sig, threshold in vol_thresholds.items():
            if characteristics.volatility_expansion_ratio >= threshold:
                vol_significance = sig
        
        # Range significance  
        range_thresholds = self.significance_thresholds["range_progression"]
        range_significance = MoveSignificance.MINOR
        if characteristics.range_position_progression:
            max_range_change = max(characteristics.range_position_progression) - min(characteristics.range_position_progression)
            for sig, threshold in range_thresholds.items():
                if max_range_change >= threshold:
                    range_significance = sig
        
        # Combine significance scores (take higher)
        significance_order = [MoveSignificance.MINOR, MoveSignificance.MODERATE, 
                            MoveSignificance.SIGNIFICANT, MoveSignificance.MAJOR, MoveSignificance.EXTREME]
        
        vol_idx = significance_order.index(vol_significance)
        range_idx = significance_order.index(range_significance)
        
        final_significance = significance_order[max(vol_idx, range_idx)]
        
        # Move type modifiers
        type_modifiers = {
            MoveType.EXPANSION: 1.2,
            MoveType.CASCADE: 1.3,  
            MoveType.BREAKOUT: 1.1,
            MoveType.REVERSAL: 1.0,
            MoveType.CONSOLIDATION: 0.8,
            MoveType.EXHAUSTION: 0.9
        }
        
        modifier = type_modifiers.get(move_type, 1.0)
        
        # Apply modifier (can potentially upgrade significance)
        if modifier > 1.1 and final_significance != MoveSignificance.EXTREME:
            final_idx = significance_order.index(final_significance)
            if final_idx < len(significance_order) - 1:
                final_significance = significance_order[final_idx + 1]
        
        return final_significance
    
    def _analyze_1m_scale(self, move_events: List[Dict], characteristics: MoveCharacteristics) -> Dict[str, float]:
        """Analyze move at 1-minute scale"""
        deltas_1m = [event.get('price_delta_1m', 0) for event in move_events]
        
        return {
            'avg_delta': np.mean(deltas_1m) if deltas_1m else 0.0,
            'max_delta': max(deltas_1m) if deltas_1m else 0.0,
            'delta_consistency': 1.0 - (np.std(deltas_1m) / np.mean(np.abs(deltas_1m))) if deltas_1m and np.mean(np.abs(deltas_1m)) > 0 else 0.0,
            'directional_strength': characteristics.momentum_persistence
        }
    
    def _analyze_5m_scale(self, move_events: List[Dict], characteristics: MoveCharacteristics) -> Dict[str, float]:
        """Analyze move at 5-minute scale"""
        deltas_5m = [event.get('price_delta_5m', 0) for event in move_events]
        
        return {
            'avg_delta': np.mean(deltas_5m) if deltas_5m else 0.0,
            'max_delta': max(deltas_5m) if deltas_5m else 0.0,
            'volatility_persistence': characteristics.volatility_expansion_ratio,
            'trend_consistency': len([d for d in deltas_5m if abs(d) > 0.005]) / len(deltas_5m) if deltas_5m else 0.0
        }
    
    def _analyze_15m_scale(self, move_events: List[Dict], characteristics: MoveCharacteristics) -> Dict[str, float]:
        """Analyze move at 15-minute scale"""
        deltas_15m = [event.get('price_delta_15m', 0) for event in move_events]
        
        return {
            'avg_delta': np.mean(deltas_15m) if deltas_15m else 0.0,
            'max_delta': max(deltas_15m) if deltas_15m else 0.0,
            'range_progression': characteristics.price_range_change,
            'exhaustion_risk': len(characteristics.exhaustion_signals) / 5.0  # Normalized to 5 possible signals
        }
    
    def _match_archaeological_pattern(self, characteristics: MoveCharacteristics, 
                                    move_type: MoveType, significance: MoveSignificance) -> Tuple[Optional[str], float]:
        """Match move against archaeological patterns"""
        best_match = None
        best_accuracy = 0.0
        
        for pattern_name, pattern_data in self.archaeological_patterns.items():
            accuracy = 0.0
            
            # Duration matching
            expected_duration = pattern_data["typical_duration"]
            # Using event density as proxy for duration characteristics
            if characteristics.event_density > 0:
                duration_score = 1.0 - abs(expected_duration - (1/characteristics.event_density)) / expected_duration
                accuracy += duration_score * 0.3
            
            # Volatility expansion matching
            expected_vol = pattern_data["volatility_expansion_avg"]
            vol_score = 1.0 - abs(expected_vol - characteristics.volatility_expansion_ratio) / expected_vol
            accuracy += vol_score * 0.4
            
            # Range progression matching
            expected_range = pattern_data["range_progression_avg"]
            if characteristics.range_position_progression:
                actual_range_change = max(characteristics.range_position_progression) - min(characteristics.range_position_progression)
                range_score = 1.0 - abs(expected_range - actual_range_change) / expected_range
                accuracy += range_score * 0.3
            
            # Track best match
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_match = pattern_name
        
        return best_match, best_accuracy
    
    def _analyze_pre_move_context(self, baseline_context: Dict) -> Dict[str, any]:
        """Analyze pre-move context"""
        return {
            'baseline_volatility': baseline_context.get('avg_volatility', 0.0),
            'range_movement': baseline_context.get('range_movement', 0.0),
            'htf_confluence_count': baseline_context.get('htf_confluence_count', 0),
            'dominant_context': baseline_context.get('dominant_context', ''),
            'primary_event_type': baseline_context.get('primary_event_type', '')
        }
    
    def _extract_move_context(self, move_events: List[Dict]) -> List[str]:
        """Extract context from move events"""
        contexts = []
        for event in move_events:
            context = event.get('context', '').strip()
            if context and context not in contexts:
                contexts.append(context)
        return contexts
    
    def _generate_post_move_implications(self, move_type: MoveType, significance: MoveSignificance, 
                                       characteristics: MoveCharacteristics) -> List[str]:
        """Generate post-move implications and expectations"""
        implications = []
        
        # Type-based implications
        if move_type == MoveType.EXPANSION:
            implications.append("Expect continued volatility expansion if momentum maintains")
            if significance in [MoveSignificance.MAJOR, MoveSignificance.EXTREME]:
                implications.append("High probability of session high/low establishment")
        
        elif move_type == MoveType.CASCADE:
            implications.append("Monitor for multi-wave continuation patterns")
            implications.append("Expect momentum persistence with potential acceleration")
        
        elif move_type == MoveType.BREAKOUT:
            implications.append("Watch for follow-through confirmation or false breakout reversal")
            if characteristics.momentum_persistence > 0.7:
                implications.append("Strong follow-through probability based on momentum")
        
        elif move_type == MoveType.REVERSAL:
            implications.append("Expect counter-trend movement or consolidation phase")
            implications.append("Monitor for exhaustion completion and trend resumption")
        
        elif move_type == MoveType.EXHAUSTION:
            implications.append("High probability of consolidation or reversal")
            implications.append("Monitor for re-accumulation patterns")
        
        # Exhaustion-based implications
        if len(characteristics.exhaustion_signals) >= 2:
            implications.append("Multiple exhaustion signals - expect momentum fade")
        
        # Significance-based implications
        if significance in [MoveSignificance.MAJOR, MoveSignificance.EXTREME]:
            implications.append("High impact move - expect structural range implications")
        
        return implications
    
    def predict_next_move(self, current_signature: DirectionalMoveSignature) -> MovePrediction:
        """Predict next directional move based on current signature"""
        
        # Base prediction on archaeological patterns
        if current_signature.archaeological_match:
            pattern_data = self.archaeological_patterns.get(current_signature.archaeological_match, {})
            
            # Determine likely next move type
            if current_signature.move_type == MoveType.EXPANSION:
                predicted_type = MoveType.CASCADE if current_signature.characteristics.momentum_persistence > 0.8 else MoveType.CONSOLIDATION
            elif current_signature.move_type == MoveType.CASCADE:
                predicted_type = MoveType.EXHAUSTION if len(current_signature.characteristics.exhaustion_signals) >= 2 else MoveType.EXPANSION
            elif current_signature.move_type == MoveType.BREAKOUT:
                predicted_type = MoveType.EXPANSION if current_signature.significance != MoveSignificance.MINOR else MoveType.REVERSAL
            else:
                predicted_type = MoveType.CONSOLIDATION
            
            # Predict significance based on current momentum
            if current_signature.characteristics.momentum_persistence > 0.9:
                predicted_significance = current_signature.significance
            elif current_signature.characteristics.momentum_persistence > 0.7:
                # Maintain or slightly reduce significance
                significance_order = [MoveSignificance.MINOR, MoveSignificance.MODERATE, 
                                    MoveSignificance.SIGNIFICANT, MoveSignificance.MAJOR, MoveSignificance.EXTREME]
                current_idx = significance_order.index(current_signature.significance)
                predicted_significance = significance_order[max(0, current_idx - 1)]
            else:
                predicted_significance = MoveSignificance.MINOR
            
            # Calculate confidence based on archaeological match accuracy
            confidence = current_signature.prediction_accuracy * 0.8  # Conservative multiplier
            
            # Expected characteristics
            expected_duration = pattern_data.get("typical_duration", 8.0)
            expected_vol_expansion = pattern_data.get("volatility_expansion_avg", 2.0)
            
            # Risk factors
            risk_factors = []
            if len(current_signature.characteristics.exhaustion_signals) >= 2:
                risk_factors.append("Multiple exhaustion signals present")
            if current_signature.characteristics.momentum_persistence < 0.5:
                risk_factors.append("Weak momentum persistence")
            if current_signature.significance == MoveSignificance.EXTREME:
                risk_factors.append("Extreme move may trigger reversal")
            
            return MovePrediction(
                predicted_move_type=predicted_type,
                predicted_significance=predicted_significance,
                confidence_score=confidence,
                expected_duration_range=(expected_duration * 0.7, expected_duration * 1.3),
                expected_volatility_expansion=expected_vol_expansion,
                archaeological_basis=current_signature.archaeological_match,
                risk_factors=risk_factors
            )
        
        # Default prediction if no archaeological match
        return MovePrediction(
            predicted_move_type=MoveType.CONSOLIDATION,
            predicted_significance=MoveSignificance.MINOR,
            confidence_score=0.3,
            expected_duration_range=(3.0, 8.0),
            expected_volatility_expansion=1.5,
            archaeological_basis="No archaeological match",
            risk_factors=["Low confidence due to no historical pattern match"]
        )
    
    def generate_move_analysis_report(self, move_signatures: List[DirectionalMoveSignature]) -> Dict:
        """Generate comprehensive move analysis report"""
        
        if not move_signatures:
            return {"error": "No move signatures provided"}
        
        # Aggregate statistics
        move_types = [sig.move_type for sig in move_signatures]
        significances = [sig.significance for sig in move_signatures]
        archaeological_matches = [sig.archaeological_match for sig in move_signatures if sig.archaeological_match]
        
        # Calculate averages
        avg_duration = np.mean([sig.move_duration for sig in move_signatures])
        avg_volatility_expansion = np.mean([sig.characteristics.volatility_expansion_ratio for sig in move_signatures])
        avg_prediction_accuracy = np.mean([sig.prediction_accuracy for sig in move_signatures])
        
        report = {
            "analysis_metadata": {
                "total_moves_analyzed": len(move_signatures),
                "analysis_timestamp": datetime.now().isoformat(),
                "archaeological_matches": len(archaeological_matches)
            },
            
            "move_type_distribution": dict(Counter([mt.value for mt in move_types])),
            "significance_distribution": dict(Counter([sig.value for sig in significances])),
            "archaeological_pattern_matches": dict(Counter(archaeological_matches)),
            
            "performance_metrics": {
                "avg_move_duration": avg_duration,
                "avg_volatility_expansion": avg_volatility_expansion,
                "avg_prediction_accuracy": avg_prediction_accuracy,
                "high_significance_moves": len([s for s in significances if s in [MoveSignificance.MAJOR, MoveSignificance.EXTREME]]),
                "archaeological_match_rate": len(archaeological_matches) / len(move_signatures) if move_signatures else 0
            },
            
            "pattern_insights": self._generate_pattern_insights(move_signatures),
            
            "detailed_moves": [
                {
                    "move_id": sig.move_id,
                    "move_type": sig.move_type.value,
                    "significance": sig.significance.value,
                    "duration": sig.move_duration,
                    "volatility_expansion": sig.characteristics.volatility_expansion_ratio,
                    "archaeological_match": sig.archaeological_match,
                    "prediction_accuracy": sig.prediction_accuracy,
                    "exhaustion_signals": sig.characteristics.exhaustion_signals
                }
                for sig in move_signatures
            ]
        }
        
        return report
    
    def _generate_pattern_insights(self, move_signatures: List[DirectionalMoveSignature]) -> Dict:
        """Generate insights from move pattern analysis"""
        insights = {}
        
        # Most reliable archaeological patterns
        pattern_accuracies = defaultdict(list)
        for sig in move_signatures:
            if sig.archaeological_match:
                pattern_accuracies[sig.archaeological_match].append(sig.prediction_accuracy)
        
        if pattern_accuracies:
            insights["most_reliable_patterns"] = {
                pattern: np.mean(accuracies)
                for pattern, accuracies in pattern_accuracies.items()
            }
        
        # Exhaustion signal effectiveness
        exhausted_moves = [sig for sig in move_signatures if len(sig.characteristics.exhaustion_signals) >= 2]
        if exhausted_moves:
            insights["exhaustion_signal_frequency"] = len(exhausted_moves) / len(move_signatures)
        
        # Move type progression patterns
        move_sequences = []
        for i in range(len(move_signatures) - 1):
            current_type = move_signatures[i].move_type.value
            next_type = move_signatures[i + 1].move_type.value
            move_sequences.append(f"{current_type} → {next_type}")
        
        if move_sequences:
            insights["common_move_progressions"] = dict(Counter(move_sequences))
        
        return insights

if __name__ == "__main__":
    print("🎯 IRONFORGE Directional Move Detector")
    print("=" * 60)
    
    detector = DirectionalMoveDetector()
    
    # Demonstrate with sample data
    print("\n🔍 Directional Move Detection System Ready!")
    print(f"  Archaeological patterns loaded: {len(detector.archaeological_patterns)}")
    print(f"  Significance thresholds configured: {len(detector.significance_thresholds)}")
    print(f"  Volatility baselines initialized: {len(detector.volatility_baselines)}")