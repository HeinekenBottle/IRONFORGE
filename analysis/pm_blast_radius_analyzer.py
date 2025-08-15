#!/usr/bin/env python3
"""
IRONFORGE PM Blast Radius Analyzer
==================================

Integrated PM phenomenon detection and blast radius analysis system.
Combines all discovery modules to identify recurring PM events (66-68 minutes, 14:36-14:38 PM),
analyze precursors, follow-on effects, and map cross-session evolution.

Workflow:
1. Core Event Scan (66-68 minute window, 2.5-3.5 minute duration)
2. Blast Radius Context (5 min before + 15 min after analysis)
3. Post-Event Impact Detection (directional moves across scales)
4. Archetype Classification (archaeological intelligence matching)
5. Cross-Session Evolution Tracking
6. Comprehensive Analysis Report Generation
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Import all analysis modules
from pm_event_scanner import PMEventScanner, PMEventPattern, PMEventCluster, DirectionalMove
from event_classifier import EventClassifier, EventClassification, EventType, RangeLevel, LiquidityArchetype
from directional_move_detector import DirectionalMoveDetector, DirectionalMoveSignature, MoveType, MoveSignificance

@dataclass
class BlastRadiusContext:
    """Context analysis for 5 minutes before and 15 minutes after core event"""
    pre_event_window: Dict[str, any]
    post_event_window: Dict[str, any]
    liquidity_sweeps: List[Dict]
    fvg_events: List[Dict]
    pd_array_transitions: List[Dict]
    htf_interactions: List[Dict]
    microstructural_transitions: List[str]

@dataclass
class PostEventImpact:
    """Post-event impact analysis across multiple scales"""
    directional_move: Optional[DirectionalMoveSignature]
    scale_1m_impact: Dict[str, float]
    scale_5m_impact: Dict[str, float]
    scale_15m_impact: Dict[str, float]
    magnitude_score: float
    velocity_score: float
    exhaustion_score: float
    move_classification: str  # impulse, continuation, false_break

@dataclass
class CrossSessionEvolution:
    """Cross-session evolution tracking"""
    session_sequence: List[str]
    recurrence_pattern: Dict[str, any]
    structural_context_evolution: List[str]
    predictive_tendencies: Dict[str, float]
    continuation_probability: float

@dataclass
class PMPhenomenonSignature:
    """Complete PM phenomenon signature with blast radius"""
    phenomenon_id: str
    core_event: PMEventPattern
    event_classification: EventClassification
    blast_radius_context: BlastRadiusContext
    post_event_impact: PostEventImpact
    cross_session_evolution: Optional[CrossSessionEvolution]
    significance_score: float
    archaeological_match_score: float
    predictive_confidence: float

class PMBlastRadiusAnalyzer:
    """
    Integrated PM blast radius analysis system
    """
    
    def __init__(self, sessions_path: str = None):
        self.logger = logging.getLogger('pm_blast_radius_analyzer')
        
        # Initialize all analysis modules
        self.pm_scanner = PMEventScanner(sessions_path)
        self.event_classifier = EventClassifier()
        self.move_detector = DirectionalMoveDetector()
        
        # Analysis parameters
        self.pre_event_window = 5   # minutes before
        self.post_event_window = 15 # minutes after
        self.cross_session_lookback = 2  # sessions to analyze for evolution
        
        print(f"💥 PM Blast Radius Analyzer initialized")
        print(f"  PM sessions available: {len(self.pm_scanner.pm_sessions)}")
        print(f"  Analysis window: -{self.pre_event_window}m to +{self.post_event_window}m")
        
    def execute_integrated_analysis(self) -> List[PMPhenomenonSignature]:
        """Execute complete integrated PM analysis workflow"""
        
        print(f"\n🔍 Starting Integrated PM Blast Radius Analysis...")
        print(f"{'='*70}")
        
        # Step 1: Core Event Scan
        print(f"\n1️⃣ Executing Core Event Scan (65-68 minutes, 14:35-14:38 PM)...")
        core_patterns = self.pm_scanner.scan_all_pm_sessions()
        print(f"   Core patterns found: {len(core_patterns)}")
        
        if not core_patterns:
            print(f"   ⚠️  No core patterns found - analysis complete")
            return []
        
        phenomena_signatures = []
        
        for i, pattern in enumerate(core_patterns):
            print(f"\n📊 Analyzing Pattern {i+1}/{len(core_patterns)}: {pattern.pattern_id}")
            
            try:
                # Step 2: Event Classification
                event_classification = self._classify_core_event(pattern)
                
                # Step 3: Blast Radius Context Analysis
                blast_context = self._analyze_blast_radius_context(pattern)
                
                # Step 4: Post-Event Impact Detection
                post_impact = self._detect_post_event_impact(pattern, blast_context)
                
                # Step 5: Cross-Session Evolution
                cross_evolution = self._track_cross_session_evolution(pattern)
                
                # Step 6: Calculate significance scores
                significance_score = self._calculate_phenomenon_significance(
                    pattern, event_classification, blast_context, post_impact
                )
                
                archaeological_match_score = self._calculate_archaeological_match_score(
                    event_classification, post_impact
                )
                
                predictive_confidence = self._calculate_predictive_confidence(
                    archaeological_match_score, cross_evolution, post_impact
                )
                
                # Create phenomenon signature
                phenomenon = PMPhenomenonSignature(
                    phenomenon_id=f"PM_{pattern.pattern_id}_blast",
                    core_event=pattern,
                    event_classification=event_classification,
                    blast_radius_context=blast_context,
                    post_event_impact=post_impact,
                    cross_session_evolution=cross_evolution,
                    significance_score=significance_score,
                    archaeological_match_score=archaeological_match_score,
                    predictive_confidence=predictive_confidence
                )
                
                phenomena_signatures.append(phenomenon)
                
                print(f"   ✅ Pattern analyzed - Significance: {significance_score:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing pattern {pattern.pattern_id}: {e}")
                continue
        
        print(f"\n🎯 Analysis Complete: {len(phenomena_signatures)} phenomena signatures generated")
        return phenomena_signatures
    
    def _classify_core_event(self, pattern: PMEventPattern) -> EventClassification:
        """Classify core event using event classifier"""
        
        # Convert pattern data to event classification format
        event_data = {
            'session_date': pattern.session_date,
            'time_minutes': pattern.event_cluster.cluster_start_minute,
            'range_position': np.mean([e.range_position for e in pattern.event_cluster.events]),
            'context': pattern.event_cluster.dominant_context,
            'price_level': np.mean([e.price_level for e in pattern.event_cluster.events]),
            'cross_tf_confluence': pattern.event_cluster.htf_confluence_count > 0,
            'liquidity_type': pattern.event_cluster.events[0].liquidity_type if pattern.event_cluster.events else 0,
            'volatility_window': pattern.event_cluster.avg_volatility,
            'raw_event_data': {
                'cluster_duration': pattern.event_cluster.cluster_duration,
                'primary_event_type': pattern.event_cluster.primary_event_type,
                'range_movement': pattern.event_cluster.range_movement
            }
        }
        
        return self.event_classifier.classify_event(event_data)
    
    def _analyze_blast_radius_context(self, pattern: PMEventPattern) -> BlastRadiusContext:
        """Analyze 5 minutes before and 15 minutes after core event"""
        
        # Load session data for context analysis
        session_file = None
        for pm_file in self.pm_scanner.pm_sessions:
            if pattern.session_date in pm_file.name:
                session_file = pm_file
                break
        
        if not session_file:
            return self._create_empty_blast_context()
        
        # Load session data
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Extract all events from session
        all_events = self.pm_scanner._extract_graph_events(session_data, pattern.session_date)
        all_events.extend(self.pm_scanner._extract_relativity_events(session_data, pattern.session_date))
        all_events = self.pm_scanner._deduplicate_events(all_events)
        
        # Define analysis windows
        cluster_start = pattern.event_cluster.cluster_start_minute
        pre_window_start = cluster_start - self.pre_event_window
        pre_window_end = cluster_start
        post_window_start = pattern.event_cluster.cluster_end_minute
        post_window_end = post_window_start + self.post_event_window
        
        # Extract context events
        pre_events = [e for e in all_events if pre_window_start <= e.time_minutes <= pre_window_end]
        post_events = [e for e in all_events if post_window_start <= e.time_minutes <= post_window_end]
        
        # Analyze context components
        liquidity_sweeps = self._identify_liquidity_sweeps(pre_events + post_events)
        fvg_events = self._identify_fvg_events(pre_events + post_events)
        pd_array_transitions = self._identify_pd_array_transitions(pre_events + post_events)
        htf_interactions = self._identify_htf_interactions(pre_events + post_events)
        microstructural_transitions = self._identify_microstructural_transitions(pre_events, post_events)
        
        return BlastRadiusContext(
            pre_event_window={
                'events_count': len(pre_events),
                'avg_volatility': np.mean([e.volatility_window for e in pre_events]) if pre_events else 0.0,
                'range_progression': [e.range_position for e in pre_events],
                'dominant_context': Counter([e.context for e in pre_events]).most_common(1)[0][0] if pre_events else ""
            },
            post_event_window={
                'events_count': len(post_events),
                'avg_volatility': np.mean([e.volatility_window for e in post_events]) if post_events else 0.0,
                'range_progression': [e.range_position for e in post_events],
                'dominant_context': Counter([e.context for e in post_events]).most_common(1)[0][0] if post_events else ""
            },
            liquidity_sweeps=liquidity_sweeps,
            fvg_events=fvg_events,
            pd_array_transitions=pd_array_transitions,
            htf_interactions=htf_interactions,
            microstructural_transitions=microstructural_transitions
        )
    
    def _detect_post_event_impact(self, pattern: PMEventPattern, 
                                blast_context: BlastRadiusContext) -> PostEventImpact:
        """Detect post-event impact using directional move detector"""
        
        # Load session events for move detection
        session_file = None
        for pm_file in self.pm_scanner.pm_sessions:
            if pattern.session_date in pm_file.name:
                session_file = pm_file
                break
        
        if not session_file:
            return self._create_empty_post_impact()
        
        # Load and process events
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        all_events = self.pm_scanner._extract_graph_events(session_data, pattern.session_date)
        all_events.extend(self.pm_scanner._extract_relativity_events(session_data, pattern.session_date))
        
        # Convert to format expected by move detector
        events_dict = []
        for event in all_events:
            events_dict.append({
                'time_minutes': event.time_minutes,
                'price_level': event.price_level,
                'range_position': event.range_position,
                'volatility_window': event.volatility_window,
                'price_delta_1m': event.price_delta_1m,
                'price_delta_5m': event.price_delta_5m,
                'price_delta_15m': event.price_delta_15m,
                'context': event.context
            })
        
        # Baseline context from cluster
        baseline_context = {
            'avg_volatility': pattern.event_cluster.avg_volatility,
            'range_movement': pattern.event_cluster.range_movement,
            'htf_confluence_count': pattern.event_cluster.htf_confluence_count,
            'dominant_context': pattern.event_cluster.dominant_context,
            'primary_event_type': pattern.event_cluster.primary_event_type
        }
        
        # Detect directional move
        directional_move = self.move_detector.detect_directional_move(
            events_dict, 
            pattern.event_cluster.cluster_end_minute,
            pattern.session_date,
            baseline_context
        )
        
        # Calculate impact scores
        magnitude_score = self._calculate_magnitude_score(directional_move, blast_context)
        velocity_score = self._calculate_velocity_score(directional_move)
        exhaustion_score = self._calculate_exhaustion_score(directional_move)
        move_classification = self._classify_move_type(directional_move)
        
        # Multi-scale analysis
        scale_1m = directional_move.scale_1m_metrics if directional_move else {}
        scale_5m = directional_move.scale_5m_metrics if directional_move else {}
        scale_15m = directional_move.scale_15m_metrics if directional_move else {}
        
        return PostEventImpact(
            directional_move=directional_move,
            scale_1m_impact=scale_1m,
            scale_5m_impact=scale_5m,
            scale_15m_impact=scale_15m,
            magnitude_score=magnitude_score,
            velocity_score=velocity_score,
            exhaustion_score=exhaustion_score,
            move_classification=move_classification
        )
    
    def _track_cross_session_evolution(self, pattern: PMEventPattern) -> Optional[CrossSessionEvolution]:
        """Track cross-session evolution patterns"""
        
        # Find similar patterns in adjacent sessions
        current_date = datetime.strptime(pattern.session_date, '%Y-%m-%d')
        similar_patterns = []
        
        # Look in next 1-2 sessions
        for days_ahead in range(1, self.cross_session_lookback + 1):
            future_date = current_date + timedelta(days=days_ahead)
            future_date_str = future_date.strftime('%Y-%m-%d')
            
            # Find session file for this date
            for pm_file in self.pm_scanner.pm_sessions:
                if future_date_str in pm_file.name:
                    # Scan this session for similar patterns
                    try:
                        session_patterns = self.pm_scanner._scan_single_pm_session(pm_file)
                        for sp in session_patterns:
                            if self._patterns_are_similar(pattern, sp):
                                similar_patterns.append(sp)
                    except Exception as e:
                        self.logger.debug(f"Error scanning {pm_file}: {e}")
                    break
        
        if not similar_patterns:
            return None
        
        # Analyze evolution
        session_sequence = [pattern.session_date] + [sp.session_date for sp in similar_patterns]
        
        # Calculate recurrence metrics
        recurrence_rate = len(similar_patterns) / self.cross_session_lookback
        
        # Analyze structural context evolution
        structural_contexts = [pattern.liquidity_archetype] + [sp.liquidity_archetype for sp in similar_patterns]
        
        # Predictive tendencies
        move_confirmations = [pattern.move_confirmed] + [sp.move_confirmed for sp in similar_patterns]
        continuation_probability = sum(move_confirmations) / len(move_confirmations)
        
        return CrossSessionEvolution(
            session_sequence=session_sequence,
            recurrence_pattern={
                'recurrence_rate': recurrence_rate,
                'total_occurrences': len(similar_patterns) + 1,
                'pattern_consistency': self._calculate_pattern_consistency(pattern, similar_patterns)
            },
            structural_context_evolution=structural_contexts,
            predictive_tendencies={
                'move_confirmation_rate': continuation_probability,
                'avg_pattern_strength': np.mean([pattern.pattern_strength] + [sp.pattern_strength for sp in similar_patterns]),
                'consistency_score': self._calculate_evolution_consistency(similar_patterns)
            },
            continuation_probability=continuation_probability
        )
    
    def _calculate_phenomenon_significance(self, pattern: PMEventPattern, 
                                         classification: EventClassification,
                                         blast_context: BlastRadiusContext,
                                         post_impact: PostEventImpact) -> float:
        """Calculate overall phenomenon significance score"""
        
        factors = []
        
        # Base pattern strength
        factors.append(pattern.pattern_strength * 0.25)
        
        # Classification confidence
        factors.append(classification.confidence_score * 0.20)
        
        # Blast radius richness
        blast_richness = (
            len(blast_context.liquidity_sweeps) * 0.1 +
            len(blast_context.fvg_events) * 0.1 +
            len(blast_context.pd_array_transitions) * 0.05 +
            len(blast_context.htf_interactions) * 0.15 +
            len(blast_context.microstructural_transitions) * 0.1
        )
        factors.append(min(blast_richness, 1.0) * 0.25)
        
        # Post-event impact
        if post_impact.directional_move:
            impact_score = (
                post_impact.magnitude_score * 0.4 +
                post_impact.velocity_score * 0.3 +
                (1.0 - post_impact.exhaustion_score) * 0.3  # Lower exhaustion = higher significance
            )
            factors.append(impact_score * 0.30)
        else:
            factors.append(0.0)
        
        return np.mean(factors)
    
    def _calculate_archaeological_match_score(self, classification: EventClassification,
                                            post_impact: PostEventImpact) -> float:
        """Calculate archaeological intelligence match score"""
        
        base_score = 0.0
        
        # Classification archaeological match
        if classification.archaeological_match:
            base_score += 0.5
        
        # Pattern correlation score
        base_score += classification.pattern_correlation_score * 0.3
        
        # Move detector archaeological match
        if post_impact.directional_move and post_impact.directional_move.archaeological_match:
            base_score += post_impact.directional_move.prediction_accuracy * 0.2
        
        return min(base_score, 1.0)
    
    def _calculate_predictive_confidence(self, archaeological_score: float,
                                       evolution: Optional[CrossSessionEvolution],
                                       post_impact: PostEventImpact) -> float:
        """Calculate predictive confidence score"""
        
        confidence_factors = [archaeological_score * 0.4]
        
        # Cross-session evolution boost
        if evolution:
            confidence_factors.append(evolution.continuation_probability * 0.3)
            confidence_factors.append(evolution.recurrence_pattern['pattern_consistency'] * 0.2)
        else:
            confidence_factors.append(0.1)  # Low confidence without evolution data
            confidence_factors.append(0.1)
        
        # Post-event impact consistency
        if post_impact.directional_move:
            move_confidence = 1.0 - post_impact.exhaustion_score  # Less exhaustion = higher confidence
            confidence_factors.append(move_confidence * 0.1)
        else:
            confidence_factors.append(0.0)
        
        return np.mean(confidence_factors)
    
    # Helper methods for context analysis
    def _identify_liquidity_sweeps(self, events: List) -> List[Dict]:
        """Identify liquidity sweep events"""
        sweeps = []
        for event in events:
            if 'sweep' in event.context.lower():
                sweeps.append({
                    'time_minutes': event.time_minutes,
                    'sweep_type': self._determine_sweep_type(event.context),
                    'range_position': event.range_position,
                    'context': event.context
                })
        return sweeps
    
    def _identify_fvg_events(self, events: List) -> List[Dict]:
        """Identify Fair Value Gap events"""
        fvg_events = []
        for event in events:
            if 'fvg' in event.context.lower() or 'fair value' in event.context.lower():
                fvg_events.append({
                    'time_minutes': event.time_minutes,
                    'fvg_type': self._determine_fvg_type(event.context),
                    'range_position': event.range_position,
                    'context': event.context
                })
        return fvg_events
    
    def _identify_pd_array_transitions(self, events: List) -> List[Dict]:
        """Identify Premium/Discount array transitions"""
        pd_transitions = []
        for event in events:
            if any(term in event.context.lower() for term in ['premium', 'discount', 'pd array', 'equilibrium']):
                pd_transitions.append({
                    'time_minutes': event.time_minutes,
                    'transition_type': self._determine_pd_transition(event.context),
                    'range_position': event.range_position,
                    'context': event.context
                })
        return pd_transitions
    
    def _identify_htf_interactions(self, events: List) -> List[Dict]:
        """Identify HTF structural interactions"""
        htf_events = []
        for event in events:
            if event.cross_tf_confluence or any(term in event.context.lower() for term in ['htf', 'timeframe', 'confluence']):
                htf_events.append({
                    'time_minutes': event.time_minutes,
                    'htf_type': 'confluence' if event.cross_tf_confluence else 'structural',
                    'range_position': event.range_position,
                    'context': event.context
                })
        return htf_events
    
    def _identify_microstructural_transitions(self, pre_events: List, post_events: List) -> List[str]:
        """Identify microstructural transitions between pre and post windows"""
        transitions = []
        
        if not pre_events or not post_events:
            return transitions
        
        # Analyze context transitions
        pre_contexts = [e.context.lower() for e in pre_events]
        post_contexts = [e.context.lower() for e in post_events]
        
        # Common transition patterns
        if any('range' in c or 'consolidation' in c for c in pre_contexts) and \
           any('breakout' in c or 'expansion' in c for c in post_contexts):
            transitions.append('range_to_breakout')
        
        if any('accumulation' in c for c in pre_contexts) and \
           any('expansion' in c for c in post_contexts):
            transitions.append('accumulation_to_expansion')
        
        if any('sweep' in c for c in pre_contexts) and \
           any('delivery' in c or 'redelivery' in c for c in post_contexts):
            transitions.append('sweep_to_delivery')
        
        # Volatility transitions
        pre_avg_vol = np.mean([e.volatility_window for e in pre_events]) if pre_events else 0
        post_avg_vol = np.mean([e.volatility_window for e in post_events]) if post_events else 0
        
        if post_avg_vol > pre_avg_vol * 1.5:
            transitions.append('volatility_expansion')
        elif post_avg_vol < pre_avg_vol * 0.7:
            transitions.append('volatility_compression')
        
        return transitions
    
    # Helper methods for classification
    def _determine_sweep_type(self, context: str) -> str:
        """Determine sweep type from context"""
        context_lower = context.lower()
        if 'buy' in context_lower or 'bullish' in context_lower:
            return 'buy_side'
        elif 'sell' in context_lower or 'bearish' in context_lower:
            return 'sell_side'
        elif 'double' in context_lower or 'both' in context_lower:
            return 'double_sweep'
        else:
            return 'liquidity_sweep'
    
    def _determine_fvg_type(self, context: str) -> str:
        """Determine FVG type from context"""
        context_lower = context.lower()
        if 'redelivery' in context_lower or 're-delivery' in context_lower:
            return 'redelivery'
        elif 'first' in context_lower or 'fp' in context_lower:
            return 'first_presented'
        elif 'continuation' in context_lower:
            return 'continuation'
        else:
            return 'fvg_formation'
    
    def _determine_pd_transition(self, context: str) -> str:
        """Determine PD array transition type"""
        context_lower = context.lower()
        if 'premium' in context_lower:
            return 'premium_zone'
        elif 'discount' in context_lower:
            return 'discount_zone'
        elif 'equilibrium' in context_lower:
            return 'equilibrium_test'
        else:
            return 'pd_array_formation'
    
    # Helper methods for impact analysis
    def _calculate_magnitude_score(self, move: Optional[DirectionalMoveSignature], 
                                 blast_context: BlastRadiusContext) -> float:
        """Calculate magnitude score"""
        if not move:
            return 0.0
        
        # Base magnitude from move characteristics
        base_magnitude = min(move.characteristics.price_range_change / 0.2, 1.0)  # Normalize to 20% range
        
        # Boost from blast context richness
        context_boost = min(len(blast_context.liquidity_sweeps) * 0.1 + 
                           len(blast_context.fvg_events) * 0.05, 0.3)
        
        return min(base_magnitude + context_boost, 1.0)
    
    def _calculate_velocity_score(self, move: Optional[DirectionalMoveSignature]) -> float:
        """Calculate velocity score"""
        if not move:
            return 0.0
        
        # Based on time acceleration and volatility expansion
        velocity = (move.characteristics.time_acceleration * 0.6 + 
                   min(move.characteristics.volatility_expansion_ratio / 3.0, 1.0) * 0.4)
        
        return min(velocity, 1.0)
    
    def _calculate_exhaustion_score(self, move: Optional[DirectionalMoveSignature]) -> float:
        """Calculate exhaustion score (higher = more exhausted)"""
        if not move:
            return 1.0  # No move = fully exhausted
        
        # Based on exhaustion signals and momentum persistence
        signal_score = len(move.characteristics.exhaustion_signals) / 5.0  # Max 5 signals
        momentum_exhaustion = 1.0 - move.characteristics.momentum_persistence
        
        return min((signal_score * 0.6 + momentum_exhaustion * 0.4), 1.0)
    
    def _classify_move_type(self, move: Optional[DirectionalMoveSignature]) -> str:
        """Classify move as impulse, continuation, or false_break"""
        if not move:
            return 'no_move'
        
        # Classification based on move characteristics
        if (move.characteristics.momentum_persistence > 0.8 and 
            move.characteristics.volatility_expansion_ratio > 2.5):
            return 'impulse'
        elif (move.characteristics.momentum_persistence > 0.6 and
              len(move.characteristics.exhaustion_signals) < 2):
            return 'continuation'
        else:
            return 'false_break'
    
    # Helper methods for evolution analysis
    def _patterns_are_similar(self, pattern1: PMEventPattern, pattern2: PMEventPattern) -> bool:
        """Check if two patterns are similar"""
        
        # Time window similarity (within 5 minutes)
        time_diff = abs(pattern1.event_cluster.cluster_start_minute - pattern2.event_cluster.cluster_start_minute)
        if time_diff > 5:
            return False
        
        # Range level similarity
        range1 = np.mean([e.range_position for e in pattern1.event_cluster.events])
        range2 = np.mean([e.range_position for e in pattern2.event_cluster.events])
        if abs(range1 - range2) > 0.2:  # 20% range difference
            return False
        
        # Liquidity archetype similarity
        if pattern1.liquidity_archetype != pattern2.liquidity_archetype:
            return False
        
        return True
    
    def _calculate_pattern_consistency(self, base_pattern: PMEventPattern, 
                                     similar_patterns: List[PMEventPattern]) -> float:
        """Calculate consistency score across similar patterns"""
        if not similar_patterns:
            return 0.0
        
        consistency_factors = []
        
        # Duration consistency
        base_duration = base_pattern.event_cluster.cluster_duration
        durations = [sp.event_cluster.cluster_duration for sp in similar_patterns]
        duration_std = np.std([base_duration] + durations)
        duration_consistency = max(0, 1.0 - (duration_std / np.mean([base_duration] + durations)))
        consistency_factors.append(duration_consistency)
        
        # Pattern strength consistency
        base_strength = base_pattern.pattern_strength
        strengths = [sp.pattern_strength for sp in similar_patterns]
        strength_std = np.std([base_strength] + strengths)
        strength_consistency = max(0, 1.0 - (strength_std / np.mean([base_strength] + strengths)))
        consistency_factors.append(strength_consistency)
        
        return np.mean(consistency_factors)
    
    def _calculate_evolution_consistency(self, patterns: List[PMEventPattern]) -> float:
        """Calculate evolution consistency score"""
        if len(patterns) < 2:
            return 0.5
        
        # Check if patterns show consistent behavior over time
        move_confirmations = [p.move_confirmed for p in patterns]
        confirmation_rate = sum(move_confirmations) / len(move_confirmations)
        
        return confirmation_rate
    
    # Empty object creators
    def _create_empty_blast_context(self) -> BlastRadiusContext:
        """Create empty blast radius context"""
        return BlastRadiusContext(
            pre_event_window={'events_count': 0, 'avg_volatility': 0.0, 'range_progression': [], 'dominant_context': ''},
            post_event_window={'events_count': 0, 'avg_volatility': 0.0, 'range_progression': [], 'dominant_context': ''},
            liquidity_sweeps=[],
            fvg_events=[],
            pd_array_transitions=[],
            htf_interactions=[],
            microstructural_transitions=[]
        )
    
    def _create_empty_post_impact(self) -> PostEventImpact:
        """Create empty post-event impact"""
        return PostEventImpact(
            directional_move=None,
            scale_1m_impact={},
            scale_5m_impact={},
            scale_15m_impact={},
            magnitude_score=0.0,
            velocity_score=0.0,
            exhaustion_score=1.0,
            move_classification='no_move'
        )
    
    def generate_blast_radius_report(self, phenomena: List[PMPhenomenonSignature]) -> Dict:
        """Generate comprehensive blast radius analysis report"""
        
        if not phenomena:
            return {"error": "No phenomena signatures provided"}
        
        # Event Table
        event_table = []
        for p in phenomena:
            event_table.append({
                'phenomenon_id': p.phenomenon_id,
                'session_date': p.core_event.session_date,
                'time_window': f"{p.core_event.event_cluster.cluster_start_minute:.1f}-{p.core_event.event_cluster.cluster_end_minute:.1f}",
                'duration': p.core_event.event_cluster.cluster_duration,
                'range_level': p.event_classification.range_level.value,
                'liquidity_archetype': p.event_classification.liquidity_archetype.value,
                'htf_confluence': p.event_classification.htf_confluence_status.value,
                'significance_score': p.significance_score,
                'archaeological_match': p.event_classification.archaeological_match,
                'move_confirmed': p.core_event.move_confirmed,
                'post_move_type': p.post_event_impact.move_classification,
                'predictive_confidence': p.predictive_confidence
            })
        
        # Blast Radius Map
        blast_radius_map = {
            'avg_pre_event_count': np.mean([len(p.blast_radius_context.liquidity_sweeps) + 
                                          len(p.blast_radius_context.fvg_events) for p in phenomena]),
            'avg_post_event_count': np.mean([len(p.blast_radius_context.pd_array_transitions) + 
                                           len(p.blast_radius_context.htf_interactions) for p in phenomena]),
            'common_microstructural_transitions': dict(Counter([
                transition for p in phenomena 
                for transition in p.blast_radius_context.microstructural_transitions
            ])),
            'liquidity_sweep_frequency': len([p for p in phenomena if p.blast_radius_context.liquidity_sweeps]) / len(phenomena),
            'fvg_event_frequency': len([p for p in phenomena if p.blast_radius_context.fvg_events]) / len(phenomena),
            'htf_interaction_frequency': len([p for p in phenomena if p.blast_radius_context.htf_interactions]) / len(phenomena)
        }
        
        # Evolution Summary
        evolution_phenomena = [p for p in phenomena if p.cross_session_evolution]
        evolution_summary = {
            'cross_session_patterns': len(evolution_phenomena),
            'avg_recurrence_rate': np.mean([p.cross_session_evolution.recurrence_pattern['recurrence_rate'] 
                                          for p in evolution_phenomena]) if evolution_phenomena else 0,
            'avg_continuation_probability': np.mean([p.cross_session_evolution.continuation_probability 
                                                   for p in evolution_phenomena]) if evolution_phenomena else 0,
            'predictive_tendencies': {
                'high_confidence_phenomena': len([p for p in phenomena if p.predictive_confidence > 0.7]),
                'archaeological_matches': len([p for p in phenomena if p.archaeological_match_score > 0.5]),
                'consistent_move_patterns': len([p for p in phenomena if p.post_event_impact.move_classification != 'no_move'])
            }
        }
        
        # Comprehensive report
        report = {
            'analysis_metadata': {
                'total_phenomena_analyzed': len(phenomena),
                'analysis_timestamp': datetime.now().isoformat(),
                'blast_radius_window': f"-{self.pre_event_window}m to +{self.post_event_window}m",
                'cross_session_lookback': self.cross_session_lookback
            },
            
            'event_table': event_table,
            'blast_radius_map': blast_radius_map,
            'evolution_summary': evolution_summary,
            
            'aggregate_insights': {
                'avg_significance_score': np.mean([p.significance_score for p in phenomena]),
                'avg_archaeological_match_score': np.mean([p.archaeological_match_score for p in phenomena]),
                'avg_predictive_confidence': np.mean([p.predictive_confidence for p in phenomena]),
                'most_common_archetype': Counter([p.event_classification.liquidity_archetype.value for p in phenomena]).most_common(1)[0][0],
                'most_common_range_level': Counter([p.event_classification.range_level.value for p in phenomena]).most_common(1)[0][0],
                'move_confirmation_rate': len([p for p in phenomena if p.core_event.move_confirmed]) / len(phenomena)
            },
            
            'detailed_phenomena': [
                {
                    'phenomenon_id': p.phenomenon_id,
                    'core_event': {
                        'session_date': p.core_event.session_date,
                        'cluster_start_minute': p.core_event.event_cluster.cluster_start_minute,
                        'cluster_duration': p.core_event.event_cluster.cluster_duration,
                        'primary_event_type': p.core_event.event_cluster.primary_event_type,
                        'pattern_strength': p.core_event.pattern_strength
                    },
                    'classification': {
                        'event_type': p.event_classification.event_type.value,
                        'range_level': p.event_classification.range_level.value,
                        'liquidity_archetype': p.event_classification.liquidity_archetype.value,
                        'htf_confluence_status': p.event_classification.htf_confluence_status.value,
                        'confidence_score': p.event_classification.confidence_score
                    },
                    'blast_radius': {
                        'pre_event_activities': len(p.blast_radius_context.liquidity_sweeps) + len(p.blast_radius_context.fvg_events),
                        'post_event_activities': len(p.blast_radius_context.pd_array_transitions) + len(p.blast_radius_context.htf_interactions),
                        'microstructural_transitions': p.blast_radius_context.microstructural_transitions
                    },
                    'post_impact': {
                        'move_classification': p.post_event_impact.move_classification,
                        'magnitude_score': p.post_event_impact.magnitude_score,
                        'velocity_score': p.post_event_impact.velocity_score,
                        'exhaustion_score': p.post_event_impact.exhaustion_score
                    },
                    'scores': {
                        'significance_score': p.significance_score,
                        'archaeological_match_score': p.archaeological_match_score,
                        'predictive_confidence': p.predictive_confidence
                    }
                }
                for p in phenomena
            ]
        }
        
        return report
    
    def save_blast_radius_analysis(self, output_path: str = None) -> str:
        """Execute analysis and save comprehensive results"""
        
        if output_path is None:
            output_path = '/Users/jack/IRONFORGE/analysis/pm_blast_radius_analysis.json'
        
        print(f"\n💥 Executing Complete PM Blast Radius Analysis...")
        
        # Execute integrated analysis
        phenomena_signatures = self.execute_integrated_analysis()
        
        # Generate comprehensive report
        report = self.generate_blast_radius_report(phenomena_signatures)
        
        # Save results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Blast Radius Analysis saved to: {output_path}")
        
        # Summary
        print(f"\n📊 Analysis Summary:")
        print(f"  Total phenomena analyzed: {len(phenomena_signatures)}")
        print(f"  High significance phenomena: {len([p for p in phenomena_signatures if p.significance_score > 0.7])}")
        print(f"  Archaeological matches: {len([p for p in phenomena_signatures if p.archaeological_match_score > 0.5])}")
        print(f"  Cross-session evolution patterns: {len([p for p in phenomena_signatures if p.cross_session_evolution])}")
        
        return output_path

if __name__ == "__main__":
    print("💥 IRONFORGE PM Blast Radius Analyzer")
    print("=" * 70)
    
    analyzer = PMBlastRadiusAnalyzer()
    output_file = analyzer.save_blast_radius_analysis()
    
    print(f"\n✅ Complete PM Blast Radius Analysis finished!")
    print(f"📊 Results saved to: {output_file}")