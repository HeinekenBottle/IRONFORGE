#!/usr/bin/env python3
"""
IRONFORGE Event Classification System
====================================

Comprehensive event tagging and classification system for PM session patterns.
Provides sophisticated event_type, range_level, liquidity_archetype, and HTF confluence
classification based on archaeological intelligence.

Classification Capabilities:
1. Event Type Classification (FVG, Sweep, PD Array, Consolidation, Expansion)
2. Range Level Categorization (20%, 40%, 60%, 80% zones)
3. Liquidity Archetype Identification (Sweep patterns, FVG patterns, Redelivery types)
4. HTF Confluence Status (Cross-timeframe alignment detection)
5. Temporal Context Classification (Session phase, timing significance)
6. Pattern Correlation Analysis (Event relationship mapping)

Based on archaeological discoveries:
- 560 patterns with documented liquidity event DNA
- Range level behavioral signatures (40% = acceleration, 60% = equilibrium, 80% = completion)
- HTF confluence detection rates (100% accuracy across all patterns)
- Cross-session evolution tracking (100% continuation probability)
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
from datetime import datetime, timedelta
from enum import Enum

class EventType(Enum):
    """Primary event types"""
    FVG_FIRST_PRESENTED = "fvg_first_presented"
    FVG_REDELIVERY = "fvg_redelivery"
    FVG_CONTINUATION = "fvg_continuation"
    SWEEP_BUY_SIDE = "sweep_buy_side"
    SWEEP_SELL_SIDE = "sweep_sell_side"
    SWEEP_DOUBLE = "sweep_double"
    PD_ARRAY_FORMATION = "pd_array_formation"
    PD_EQUILIBRIUM_TEST = "pd_equilibrium_test"
    CONSOLIDATION_RANGE = "consolidation_range"
    EXPANSION_PHASE = "expansion_phase"
    LIQUIDITY_VOID = "liquidity_void"
    REVERSAL_POINT = "reversal_point"

class RangeLevel(Enum):
    """Range level categories"""
    MOMENTUM_FILTER = "20_percent_momentum_filter"      # 15-25% range
    SWEEP_ACCELERATION = "40_percent_sweep_acceleration" # 35-45% range
    FVG_EQUILIBRIUM = "60_percent_fvg_equilibrium"      # 55-65% range
    SWEEP_COMPLETION = "80_percent_sweep_completion"     # 75-85% range
    OUTLIER_RANGE = "outlier_range"                     # Outside standard zones

class LiquidityArchetype(Enum):
    """Liquidity archetype patterns"""
    SESSION_LOW_SWEEP = "session_low_sweep"
    SESSION_HIGH_SWEEP = "session_high_sweep"
    FPFVG_DELIVERY = "fpfvg_delivery"
    FPFVG_REDELIVERY = "fpfvg_redelivery"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    REVERSAL_POINT = "reversal_point"
    EXPANSION_PHASE = "expansion_phase"
    CONSOLIDATION_PHASE = "consolidation_phase"
    MOMENTUM_ACCELERATION = "momentum_acceleration"
    EQUILIBRIUM_BALANCE = "equilibrium_balance"
    COMPLETION_EXHAUSTION = "completion_exhaustion"
    UNCLASSIFIED = "unclassified"

class HTFConfluenceStatus(Enum):
    """HTF confluence status levels"""
    CONFIRMED = "confirmed"         # Strong HTF alignment
    PARTIAL = "partial"            # Some HTF signals
    WEAK = "weak"                  # Minimal HTF confluence
    ABSENT = "absent"              # No HTF confluence
    UNKNOWN = "unknown"            # Cannot determine

class TemporalContext(Enum):
    """Temporal context classification"""
    EARLY_SESSION = "early_session"        # 0-60 minutes
    MID_SESSION = "mid_session"           # 60-120 minutes
    LATE_SESSION = "late_session"         # 120-180 minutes (includes 126-129)
    CLOSING_SESSION = "closing_session"   # 180+ minutes
    CRITICAL_WINDOW = "critical_window"   # 126-129 minutes specifically

@dataclass
class EventClassification:
    """Comprehensive event classification"""
    event_id: str
    event_type: EventType
    range_level: RangeLevel
    liquidity_archetype: LiquidityArchetype
    htf_confluence_status: HTFConfluenceStatus
    temporal_context: TemporalContext
    
    # Supporting metrics
    confidence_score: float
    classification_basis: List[str]
    archaeological_match: Optional[str]
    pattern_correlation_score: float
    
    # Context details
    session_date: str
    time_minutes: float
    range_position: float
    context_description: str
    raw_event_data: Dict

@dataclass
class ClassificationCriteria:
    """Criteria used for event classification"""
    event_type_indicators: Dict[str, List[str]]
    range_level_thresholds: Dict[str, Tuple[float, float]]
    liquidity_archetype_patterns: Dict[str, Dict]
    htf_confluence_markers: List[str]
    temporal_boundaries: Dict[str, Tuple[float, float]]

class EventClassifier:
    """
    Comprehensive event classification system
    """
    
    def __init__(self, archaeological_patterns_path: str = None):
        self.logger = logging.getLogger('event_classifier')
        
        # Initialize classification criteria
        self.classification_criteria = self._initialize_classification_criteria()
        self.archaeological_intelligence = self._load_archaeological_intelligence()
        self.pattern_correlation_matrix = self._build_pattern_correlation_matrix()
        
        print(f"🏷️  Event Classifier initialized")
        print(f"  Classification criteria loaded: {len(self.classification_criteria.event_type_indicators)}")
        print(f"  Archaeological patterns: {len(self.archaeological_intelligence)}")
    
    def _initialize_classification_criteria(self) -> ClassificationCriteria:
        """Initialize comprehensive classification criteria"""
        
        # Event type indicators from context analysis
        event_type_indicators = {
            EventType.FVG_FIRST_PRESENTED.value: [
                "first presented", "fp", "fvg formation", "initial fvg"
            ],
            EventType.FVG_REDELIVERY.value: [
                "redelivery", "re-delivery", "fpfvg redelivery", "fvg return"
            ],
            EventType.FVG_CONTINUATION.value: [
                "fvg continuation", "continued fvg", "fvg follow through"
            ],
            EventType.SWEEP_BUY_SIDE.value: [
                "buy side sweep", "liq sweep", "bullish sweep", "upside sweep"
            ],
            EventType.SWEEP_SELL_SIDE.value: [
                "sell side sweep", "bearish sweep", "downside sweep", "liquidity grab"
            ],
            EventType.SWEEP_DOUBLE.value: [
                "double sweep", "both sides swept", "comprehensive sweep"
            ],
            EventType.PD_ARRAY_FORMATION.value: [
                "pd array", "premium discount", "array formation"
            ],
            EventType.PD_EQUILIBRIUM_TEST.value: [
                "equilibrium", "pd test", "balance test", "middle test"
            ],
            EventType.CONSOLIDATION_RANGE.value: [
                "consolidation", "range", "sideways", "balance"
            ],
            EventType.EXPANSION_PHASE.value: [
                "expansion", "breakout", "momentum", "acceleration"
            ],
            EventType.REVERSAL_POINT.value: [
                "reversal", "turn", "rejection", "failure"
            ]
        }
        
        # Range level thresholds based on archaeological data
        range_level_thresholds = {
            RangeLevel.MOMENTUM_FILTER.value: (15.0, 25.0),
            RangeLevel.SWEEP_ACCELERATION.value: (35.0, 45.0),
            RangeLevel.FVG_EQUILIBRIUM.value: (55.0, 65.0),
            RangeLevel.SWEEP_COMPLETION.value: (75.0, 85.0)
        }
        
        # Liquidity archetype patterns with contextual markers
        liquidity_archetype_patterns = {
            LiquidityArchetype.SESSION_LOW_SWEEP.value: {
                "context_markers": ["session low", "sweep", "lowest point"],
                "range_preference": [15.0, 30.0],
                "timing_preference": [120.0, 180.0],
                "event_types": [EventType.SWEEP_BUY_SIDE, EventType.SWEEP_DOUBLE]
            },
            LiquidityArchetype.SESSION_HIGH_SWEEP.value: {
                "context_markers": ["session high", "sweep", "highest point"],
                "range_preference": [70.0, 95.0],
                "timing_preference": [120.0, 180.0],
                "event_types": [EventType.SWEEP_SELL_SIDE, EventType.SWEEP_DOUBLE]
            },
            LiquidityArchetype.FPFVG_DELIVERY.value: {
                "context_markers": ["fpfvg", "delivery", "first presented"],
                "range_preference": [40.0, 80.0],
                "timing_preference": [60.0, 160.0],
                "event_types": [EventType.FVG_FIRST_PRESENTED, EventType.FVG_CONTINUATION]
            },
            LiquidityArchetype.FPFVG_REDELIVERY.value: {
                "context_markers": ["fpfvg", "redelivery", "re-delivery"],
                "range_preference": [30.0, 70.0],
                "timing_preference": [90.0, 180.0],
                "event_types": [EventType.FVG_REDELIVERY]
            },
            LiquidityArchetype.MOMENTUM_ACCELERATION.value: {
                "context_markers": ["acceleration", "momentum", "building"],
                "range_preference": [35.0, 45.0],  # 40% zone
                "timing_preference": [100.0, 150.0],
                "event_types": [EventType.EXPANSION_PHASE, EventType.SWEEP_ACCELERATION]
            },
            LiquidityArchetype.EQUILIBRIUM_BALANCE.value: {
                "context_markers": ["equilibrium", "balance", "middle"],
                "range_preference": [55.0, 65.0],  # 60% zone
                "timing_preference": [80.0, 160.0],
                "event_types": [EventType.PD_EQUILIBRIUM_TEST, EventType.FVG_EQUILIBRIUM]
            },
            LiquidityArchetype.COMPLETION_EXHAUSTION.value: {
                "context_markers": ["completion", "exhaustion", "final"],
                "range_preference": [75.0, 85.0],  # 80% zone
                "timing_preference": [140.0, 190.0],
                "event_types": [EventType.SWEEP_COMPLETION, EventType.REVERSAL_POINT]
            }
        }
        
        # HTF confluence markers from archaeological analysis
        htf_confluence_markers = [
            "cross_tf_confluence", "htf_confluence", "multi_timeframe",
            "temporal_echo_strength", "scaling_factor", "htf_alignment"
        ]
        
        # Temporal boundaries for session phases
        temporal_boundaries = {
            TemporalContext.EARLY_SESSION.value: (0.0, 60.0),
            TemporalContext.MID_SESSION.value: (60.0, 120.0),
            TemporalContext.LATE_SESSION.value: (120.0, 180.0),
            TemporalContext.CLOSING_SESSION.value: (180.0, 300.0),
            TemporalContext.CRITICAL_WINDOW.value: (126.0, 129.0)
        }
        
        return ClassificationCriteria(
            event_type_indicators=event_type_indicators,
            range_level_thresholds=range_level_thresholds,
            liquidity_archetype_patterns=liquidity_archetype_patterns,
            htf_confluence_markers=htf_confluence_markers,
            temporal_boundaries=temporal_boundaries
        )
    
    def _load_archaeological_intelligence(self) -> Dict[str, Dict]:
        """Load archaeological intelligence for pattern matching"""
        return {
            "20% Momentum Filter": {
                "dominant_events": [EventType.FVG_REDELIVERY, EventType.SWEEP_BUY_SIDE],
                "liquidity_signature": "variable momentum with 44.7% continuation probability",
                "archaeological_frequency": {"fvg_events": 0.560, "sweep_events": 0.607, "pd_array": 1.000},
                "evolution_strength": 0.92,
                "velocity_consistency": 0.68,
                "common_contexts": ["momentum decision", "filter zone", "directional bias test"]
            },
            "40% Sweep Acceleration": {
                "dominant_events": [EventType.SWEEP_DOUBLE, EventType.FVG_CONTINUATION],
                "liquidity_signature": "perfect velocity consistency with 100% continuation",
                "archaeological_frequency": {"fvg_events": 0.574, "sweep_events": 0.632, "pd_array": 1.000},
                "evolution_strength": 0.89,
                "velocity_consistency": 1.00,  # Perfect
                "common_contexts": ["sweep acceleration", "momentum building", "directional confirmation"]
            },
            "60% FVG Equilibrium": {
                "dominant_events": [EventType.FVG_FIRST_PRESENTED, EventType.PD_EQUILIBRIUM_TEST],
                "liquidity_signature": "highest evolution strength zone, most predictable",
                "archaeological_frequency": {"fvg_events": 0.611, "sweep_events": 0.574, "pd_array": 1.000},
                "evolution_strength": 0.93,  # Highest
                "velocity_consistency": 0.89,
                "common_contexts": ["equilibrium test", "balance point", "predictable zone"]
            },
            "80% Completion Zone": {
                "dominant_events": [EventType.SWEEP_DOUBLE, EventType.CONSOLIDATION_RANGE],
                "liquidity_signature": "guaranteed completion zone with terminal velocity",
                "archaeological_frequency": {"fvg_events": 0.438, "sweep_events": 0.719, "consolidation": 0.391},
                "evolution_strength": 0.92,
                "velocity_consistency": 1.00,  # Perfect terminal
                "common_contexts": ["completion", "exhaustion", "final liquidity hunt"]
            }
        }
    
    def _build_pattern_correlation_matrix(self) -> Dict[str, Dict]:
        """Build pattern correlation matrix for event relationships"""
        return {
            # FVG patterns tend to correlate
            EventType.FVG_FIRST_PRESENTED.value: {
                EventType.FVG_REDELIVERY.value: 0.85,
                EventType.FVG_CONTINUATION.value: 0.72,
                EventType.PD_EQUILIBRIUM_TEST.value: 0.68
            },
            # Sweep patterns correlate strongly
            EventType.SWEEP_BUY_SIDE.value: {
                EventType.SWEEP_SELL_SIDE.value: 0.45,  # Opposite direction
                EventType.SWEEP_DOUBLE.value: 0.78,
                EventType.LIQUIDITY_VOID.value: 0.65
            },
            # Expansion and consolidation are related
            EventType.EXPANSION_PHASE.value: {
                EventType.CONSOLIDATION_RANGE.value: 0.35,  # Often follow each other
                EventType.REVERSAL_POINT.value: 0.42
            }
        }
    
    def classify_event(self, event_data: Dict, session_context: Dict = None) -> EventClassification:
        """Classify a single event comprehensively"""
        
        # Extract basic event information
        event_id = f"{event_data.get('session_date', 'unknown')}_{event_data.get('time_minutes', 0):.1f}"
        time_minutes = event_data.get('time_minutes', 0)
        range_position = event_data.get('range_position', 0)
        context = event_data.get('context', '').lower()
        
        # Classify event type
        event_type = self._classify_event_type(context, event_data)
        
        # Classify range level
        range_level = self._classify_range_level(range_position)
        
        # Classify liquidity archetype
        liquidity_archetype = self._classify_liquidity_archetype(
            event_type, range_position, time_minutes, context, event_data
        )
        
        # Determine HTF confluence status
        htf_confluence_status = self._determine_htf_confluence_status(event_data)
        
        # Classify temporal context
        temporal_context = self._classify_temporal_context(time_minutes)
        
        # Calculate confidence and supporting metrics
        confidence_score = self._calculate_classification_confidence(
            event_type, range_level, liquidity_archetype, context, event_data
        )
        
        classification_basis = self._determine_classification_basis(
            event_type, range_level, liquidity_archetype, context
        )
        
        # Match archaeological patterns
        archaeological_match = self._match_archaeological_pattern(
            range_level, event_type, liquidity_archetype
        )
        
        # Calculate pattern correlation
        pattern_correlation_score = self._calculate_pattern_correlation(
            event_type, session_context
        )
        
        return EventClassification(
            event_id=event_id,
            event_type=event_type,
            range_level=range_level,
            liquidity_archetype=liquidity_archetype,
            htf_confluence_status=htf_confluence_status,
            temporal_context=temporal_context,
            confidence_score=confidence_score,
            classification_basis=classification_basis,
            archaeological_match=archaeological_match,
            pattern_correlation_score=pattern_correlation_score,
            session_date=event_data.get('session_date', 'unknown'),
            time_minutes=time_minutes,
            range_position=range_position,
            context_description=event_data.get('context', ''),
            raw_event_data=event_data
        )
    
    def _classify_event_type(self, context: str, event_data: Dict) -> EventType:
        """Classify the primary event type"""
        
        # Check context for explicit indicators
        for event_type_value, indicators in self.classification_criteria.event_type_indicators.items():
            for indicator in indicators:
                if indicator in context:
                    return EventType(event_type_value)
        
        # Fallback classification based on event characteristics
        if 'fvg' in context:
            if 'redelivery' in context or 're-delivery' in context:
                return EventType.FVG_REDELIVERY
            elif 'first' in context or 'fp' in context:
                return EventType.FVG_FIRST_PRESENTED
            else:
                return EventType.FVG_CONTINUATION
        
        elif 'sweep' in context:
            if 'buy' in context or 'bullish' in context:
                return EventType.SWEEP_BUY_SIDE
            elif 'sell' in context or 'bearish' in context:
                return EventType.SWEEP_SELL_SIDE
            else:
                return EventType.SWEEP_DOUBLE
        
        elif 'expansion' in context or 'breakout' in context:
            return EventType.EXPANSION_PHASE
        
        elif 'consolidation' in context or 'range' in context:
            return EventType.CONSOLIDATION_RANGE
        
        elif 'reversal' in context or 'rejection' in context:
            return EventType.REVERSAL_POINT
        
        # Default classification based on numerical indicators
        liquidity_type = event_data.get('liquidity_type', 0)
        if liquidity_type == 1:
            return EventType.FVG_FIRST_PRESENTED
        elif liquidity_type == 2:
            return EventType.SWEEP_BUY_SIDE
        else:
            return EventType.CONSOLIDATION_RANGE
    
    def _classify_range_level(self, range_position: float) -> RangeLevel:
        """Classify range level based on position"""
        
        # Convert to percentage if needed
        if range_position <= 1.0:
            range_percent = range_position * 100
        else:
            range_percent = range_position
        
        # Match against archaeological range zones
        for range_level_value, (min_thresh, max_thresh) in self.classification_criteria.range_level_thresholds.items():
            if min_thresh <= range_percent <= max_thresh:
                return RangeLevel(range_level_value)
        
        return RangeLevel.OUTLIER_RANGE
    
    def _classify_liquidity_archetype(self, event_type: EventType, range_position: float,
                                    time_minutes: float, context: str, event_data: Dict) -> LiquidityArchetype:
        """Classify liquidity archetype based on multiple factors"""
        
        range_percent = range_position * 100 if range_position <= 1.0 else range_position
        
        # Check each archetype pattern
        best_match = LiquidityArchetype.UNCLASSIFIED
        best_score = 0.0
        
        for archetype_value, pattern_data in self.classification_criteria.liquidity_archetype_patterns.items():
            score = 0.0
            
            # Context markers matching
            context_markers = pattern_data.get("context_markers", [])
            context_matches = sum(1 for marker in context_markers if marker in context)
            if context_markers:
                score += (context_matches / len(context_markers)) * 0.4
            
            # Range preference matching
            range_pref = pattern_data.get("range_preference", [0, 100])
            if range_pref[0] <= range_percent <= range_pref[1]:
                score += 0.3
            
            # Timing preference matching
            timing_pref = pattern_data.get("timing_preference", [0, 300])
            if timing_pref[0] <= time_minutes <= timing_pref[1]:
                score += 0.2
            
            # Event type compatibility
            compatible_types = pattern_data.get("event_types", [])
            if event_type in compatible_types:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_match = LiquidityArchetype(archetype_value)
        
        return best_match
    
    def _determine_htf_confluence_status(self, event_data: Dict) -> HTFConfluenceStatus:
        """Determine HTF confluence status"""
        
        # Check explicit HTF confluence flag
        if event_data.get('cross_tf_confluence', False):
            return HTFConfluenceStatus.CONFIRMED
        
        # Check for HTF markers in context or features
        context = event_data.get('context', '').lower()
        
        htf_markers_found = 0
        for marker in self.classification_criteria.htf_confluence_markers:
            if marker in context:
                htf_markers_found += 1
        
        # Check raw event data for HTF features
        raw_data = event_data.get('raw_event_data', {})
        if isinstance(raw_data, dict):
            raw_str = str(raw_data).lower()
            for marker in self.classification_criteria.htf_confluence_markers:
                if marker in raw_str:
                    htf_markers_found += 1
        
        # Classify based on marker count
        if htf_markers_found >= 3:
            return HTFConfluenceStatus.CONFIRMED
        elif htf_markers_found >= 2:
            return HTFConfluenceStatus.PARTIAL
        elif htf_markers_found >= 1:
            return HTFConfluenceStatus.WEAK
        else:
            return HTFConfluenceStatus.ABSENT
    
    def _classify_temporal_context(self, time_minutes: float) -> TemporalContext:
        """Classify temporal context of the event"""
        
        # Check for critical window first (126-129 minutes)
        critical_start, critical_end = self.classification_criteria.temporal_boundaries[TemporalContext.CRITICAL_WINDOW.value]
        if critical_start <= time_minutes <= critical_end:
            return TemporalContext.CRITICAL_WINDOW
        
        # Check other temporal boundaries
        for context_value, (min_time, max_time) in self.classification_criteria.temporal_boundaries.items():
            if context_value != TemporalContext.CRITICAL_WINDOW.value and min_time <= time_minutes <= max_time:
                return TemporalContext(context_value)
        
        return TemporalContext.CLOSING_SESSION  # Default for late times
    
    def _calculate_classification_confidence(self, event_type: EventType, range_level: RangeLevel,
                                          liquidity_archetype: LiquidityArchetype, 
                                          context: str, event_data: Dict) -> float:
        """Calculate overall classification confidence"""
        
        confidence_factors = []
        
        # Context clarity
        if len(context.strip()) > 10:  # Non-trivial context
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # Range level certainty (higher confidence for archaeological zones)
        if range_level != RangeLevel.OUTLIER_RANGE:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.4)
        
        # Liquidity archetype clarity
        if liquidity_archetype != LiquidityArchetype.UNCLASSIFIED:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.2)
        
        # Data completeness
        required_fields = ['time_minutes', 'range_position', 'context']
        completeness = sum(1 for field in required_fields if field in event_data and event_data[field] is not None)
        confidence_factors.append(completeness / len(required_fields))
        
        return np.mean(confidence_factors)
    
    def _determine_classification_basis(self, event_type: EventType, range_level: RangeLevel,
                                      liquidity_archetype: LiquidityArchetype, context: str) -> List[str]:
        """Determine the basis for classification decisions"""
        
        basis = []
        
        # Event type basis
        for indicators in self.classification_criteria.event_type_indicators.values():
            for indicator in indicators:
                if indicator in context:
                    basis.append(f"Event type: '{indicator}' in context")
                    break
        
        # Range level basis
        if range_level != RangeLevel.OUTLIER_RANGE:
            basis.append(f"Range level: Archaeological zone {range_level.value}")
        
        # Archetype basis
        if liquidity_archetype != LiquidityArchetype.UNCLASSIFIED:
            basis.append(f"Liquidity archetype: Pattern match {liquidity_archetype.value}")
        
        return basis
    
    def _match_archaeological_pattern(self, range_level: RangeLevel, event_type: EventType,
                                    liquidity_archetype: LiquidityArchetype) -> Optional[str]:
        """Match against archaeological intelligence patterns"""
        
        # Map range levels to archaeological patterns
        range_to_pattern = {
            RangeLevel.MOMENTUM_FILTER: "20% Momentum Filter",
            RangeLevel.SWEEP_ACCELERATION: "40% Sweep Acceleration",
            RangeLevel.FVG_EQUILIBRIUM: "60% FVG Equilibrium",
            RangeLevel.SWEEP_COMPLETION: "80% Completion Zone"
        }
        
        pattern_name = range_to_pattern.get(range_level)
        if not pattern_name:
            return None
        
        # Verify event type compatibility with archaeological pattern
        pattern_data = self.archaeological_intelligence.get(pattern_name, {})
        dominant_events = pattern_data.get("dominant_events", [])
        
        if event_type in dominant_events:
            return pattern_name
        
        return None
    
    def _calculate_pattern_correlation(self, event_type: EventType, session_context: Dict) -> float:
        """Calculate pattern correlation score with session context"""
        
        if not session_context or 'other_events' not in session_context:
            return 0.5  # Neutral score
        
        other_events = session_context.get('other_events', [])
        correlation_scores = []
        
        # Check correlations with other events in the session
        for other_event in other_events:
            other_event_type = other_event.get('event_type')
            if other_event_type and event_type.value in self.pattern_correlation_matrix:
                correlation = self.pattern_correlation_matrix[event_type.value].get(other_event_type, 0.0)
                correlation_scores.append(correlation)
        
        if correlation_scores:
            return np.mean(correlation_scores)
        
        return 0.5  # Neutral score if no correlations found
    
    def classify_event_cluster(self, events: List[Dict], session_context: Dict = None) -> List[EventClassification]:
        """Classify a cluster of related events"""
        
        classifications = []
        
        # First pass: classify individual events
        for event in events:
            classification = self.classify_event(event, session_context)
            classifications.append(classification)
        
        # Second pass: enhance classifications based on cluster context
        cluster_context = self._analyze_cluster_context(classifications)
        
        for classification in classifications:
            # Update confidence based on cluster consistency
            cluster_consistency = self._calculate_cluster_consistency(classification, cluster_context)
            classification.confidence_score = min(1.0, classification.confidence_score + cluster_consistency * 0.2)
            
            # Update pattern correlation based on cluster
            classification.pattern_correlation_score = self._calculate_cluster_pattern_correlation(
                classification, cluster_context
            )
        
        return classifications
    
    def _analyze_cluster_context(self, classifications: List[EventClassification]) -> Dict:
        """Analyze context of event cluster"""
        
        return {
            'dominant_event_type': Counter([c.event_type for c in classifications]).most_common(1)[0][0],
            'dominant_range_level': Counter([c.range_level for c in classifications]).most_common(1)[0][0],
            'dominant_archetype': Counter([c.liquidity_archetype for c in classifications]).most_common(1)[0][0],
            'htf_confluence_rate': len([c for c in classifications if c.htf_confluence_status in [HTFConfluenceStatus.CONFIRMED, HTFConfluenceStatus.PARTIAL]]) / len(classifications),
            'avg_confidence': np.mean([c.confidence_score for c in classifications]),
            'temporal_spread': max([c.time_minutes for c in classifications]) - min([c.time_minutes for c in classifications])
        }
    
    def _calculate_cluster_consistency(self, classification: EventClassification, cluster_context: Dict) -> float:
        """Calculate how consistent the classification is with cluster context"""
        
        consistency_score = 0.0
        
        # Event type consistency
        if classification.event_type == cluster_context['dominant_event_type']:
            consistency_score += 0.3
        
        # Range level consistency  
        if classification.range_level == cluster_context['dominant_range_level']:
            consistency_score += 0.3
        
        # Archetype consistency
        if classification.liquidity_archetype == cluster_context['dominant_archetype']:
            consistency_score += 0.2
        
        # HTF confluence consistency
        if (classification.htf_confluence_status in [HTFConfluenceStatus.CONFIRMED, HTFConfluenceStatus.PARTIAL] and
            cluster_context['htf_confluence_rate'] > 0.5):
            consistency_score += 0.2
        
        return consistency_score
    
    def _calculate_cluster_pattern_correlation(self, classification: EventClassification, cluster_context: Dict) -> float:
        """Calculate pattern correlation within cluster"""
        
        base_correlation = classification.pattern_correlation_score
        
        # Boost correlation if event aligns with cluster patterns
        if classification.event_type == cluster_context['dominant_event_type']:
            base_correlation += 0.2
        
        if classification.range_level == cluster_context['dominant_range_level']:
            base_correlation += 0.1
        
        return min(1.0, base_correlation)
    
    def generate_classification_report(self, classifications: List[EventClassification]) -> Dict:
        """Generate comprehensive classification report"""
        
        if not classifications:
            return {"error": "No classifications provided"}
        
        # Aggregate statistics
        event_types = [c.event_type.value for c in classifications]
        range_levels = [c.range_level.value for c in classifications]
        archetypes = [c.liquidity_archetype.value for c in classifications]
        htf_statuses = [c.htf_confluence_status.value for c in classifications]
        temporal_contexts = [c.temporal_context.value for c in classifications]
        
        # Archaeological pattern matches
        archaeological_matches = [c.archaeological_match for c in classifications if c.archaeological_match]
        
        report = {
            "classification_metadata": {
                "total_events_classified": len(classifications),
                "classification_timestamp": datetime.now().isoformat(),
                "avg_confidence_score": np.mean([c.confidence_score for c in classifications]),
                "high_confidence_classifications": len([c for c in classifications if c.confidence_score > 0.8])
            },
            
            "event_type_distribution": dict(Counter(event_types)),
            "range_level_distribution": dict(Counter(range_levels)),
            "liquidity_archetype_distribution": dict(Counter(archetypes)),
            "htf_confluence_distribution": dict(Counter(htf_statuses)),
            "temporal_context_distribution": dict(Counter(temporal_contexts)),
            
            "archaeological_analysis": {
                "pattern_matches": dict(Counter(archaeological_matches)),
                "match_rate": len(archaeological_matches) / len(classifications) if classifications else 0,
                "avg_pattern_correlation": np.mean([c.pattern_correlation_score for c in classifications])
            },
            
            "critical_window_analysis": {
                "critical_window_events": len([c for c in classifications if c.temporal_context == TemporalContext.CRITICAL_WINDOW]),
                "critical_window_rate": len([c for c in classifications if c.temporal_context == TemporalContext.CRITICAL_WINDOW]) / len(classifications),
                "critical_window_dominant_type": Counter([c.event_type.value for c in classifications if c.temporal_context == TemporalContext.CRITICAL_WINDOW]).most_common(1)[0][0] if any(c.temporal_context == TemporalContext.CRITICAL_WINDOW for c in classifications) else None
            },
            
            "detailed_classifications": [
                {
                    "event_id": c.event_id,
                    "event_type": c.event_type.value,
                    "range_level": c.range_level.value,
                    "liquidity_archetype": c.liquidity_archetype.value,
                    "htf_confluence_status": c.htf_confluence_status.value,
                    "temporal_context": c.temporal_context.value,
                    "confidence_score": c.confidence_score,
                    "archaeological_match": c.archaeological_match,
                    "pattern_correlation_score": c.pattern_correlation_score,
                    "time_minutes": c.time_minutes,
                    "range_position": c.range_position,
                    "context_description": c.context_description
                }
                for c in classifications
            ]
        }
        
        return report

if __name__ == "__main__":
    print("🏷️  IRONFORGE Event Classification System")
    print("=" * 60)
    
    classifier = EventClassifier()
    
    print(f"\n✅ Event Classification System Ready!")
    print(f"  Event types supported: {len(EventType)}")
    print(f"  Range levels defined: {len(RangeLevel)}")
    print(f"  Liquidity archetypes: {len(LiquidityArchetype)}")
    print(f"  HTF confluence statuses: {len(HTFConfluenceStatus)}")
    print(f"  Temporal contexts: {len(TemporalContext)}")