#!/usr/bin/env python3
"""
IRONFORGE PM Pattern Discovery Engine
====================================

Comprehensive pattern discovery and correlation analysis system for PM session events.
Integrates event scanning, directional move detection, and event classification to
discover, analyze, and correlate patterns across PM sessions.

Discovery Engine Capabilities:
1. End-to-End PM Pattern Discovery (126-129 minute events → directional moves)
2. Multi-dimensional Pattern Correlation (events, moves, classifications)
3. Archaeological Pattern Validation (560-pattern database matching)
4. Cross-Session Pattern Evolution Tracking
5. Predictive Pattern Identification
6. Comprehensive Discovery Reporting

Integration Components:
- PM Event Scanner: Event cluster detection in critical window
- Directional Move Detector: Post-cluster movement analysis
- Event Classifier: Comprehensive event tagging system
- Archaeological Intelligence: Pattern validation against discoveries

Based on PM Session Archaeological Intelligence:
- Critical window: 126-129 minutes (15:36-15:39 ET)
- Event cluster duration: 2.5-3.5 minutes
- Directional move window: 10-15 minutes post-cluster
- Pattern correlation scoring across multiple dimensions
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

# Import our specialized analysis modules
from pm_event_scanner import PMEventScanner, PMEventPattern, PMEventCluster, DirectionalMove
from directional_move_detector import DirectionalMoveDetector, DirectionalMoveSignature, MoveType, MoveSignificance
from event_classifier import EventClassifier, EventClassification, EventType, RangeLevel, LiquidityArchetype, HTFConfluenceStatus

class PatternStrength(Enum):
    """Pattern strength levels"""
    WEAK = "weak"                    # 0.0-0.3
    MODERATE = "moderate"            # 0.3-0.6
    STRONG = "strong"               # 0.6-0.8
    EXCEPTIONAL = "exceptional"      # 0.8-1.0

class DiscoveryStatus(Enum):
    """Pattern discovery status"""
    CANDIDATE = "candidate"          # Initial detection
    VALIDATED = "validated"         # Archaeological validation
    CORRELATED = "correlated"       # Cross-session correlation
    PREDICTIVE = "predictive"       # Predictive value confirmed

@dataclass
class IntegratedPattern:
    """Integrated pattern combining all analysis dimensions"""
    pattern_id: str
    session_date: str
    discovery_timestamp: datetime
    discovery_status: DiscoveryStatus
    pattern_strength: PatternStrength
    
    # Core components
    event_cluster: PMEventCluster
    directional_move: Optional[DirectionalMoveSignature]
    event_classifications: List[EventClassification]
    
    # Analysis results
    archaeological_validation: Dict[str, any]
    correlation_analysis: Dict[str, float]
    predictive_metrics: Dict[str, float]
    
    # Discovery metadata
    integration_confidence: float
    discovery_uniqueness: float
    cross_session_relevance: float
    pattern_tags: List[str]

@dataclass
class PatternEvolution:
    """Pattern evolution tracking across sessions"""
    evolution_id: str
    base_pattern: IntegratedPattern
    evolution_sessions: List[IntegratedPattern]
    evolution_strength: float
    consistency_score: float
    predictive_reliability: float
    evolution_insights: List[str]

@dataclass
class DiscoverySession:
    """Complete discovery session results"""
    session_id: str
    discovery_timestamp: datetime
    sessions_analyzed: int
    patterns_discovered: List[IntegratedPattern]
    pattern_evolutions: List[PatternEvolution]
    discovery_statistics: Dict[str, any]
    insights_generated: List[str]

class PMPatternDiscoverer:
    """
    Comprehensive PM pattern discovery and correlation engine
    """
    
    def __init__(self, sessions_path: str = None):
        self.logger = logging.getLogger('pm_pattern_discoverer')
        
        # Initialize analysis components
        self.pm_scanner = PMEventScanner(sessions_path)
        self.move_detector = DirectionalMoveDetector()
        self.event_classifier = EventClassifier()
        
        # Discovery configuration
        self.discovery_config = self._initialize_discovery_config()
        self.archaeological_validation = self._load_archaeological_validation()
        self.pattern_evolution_tracking = defaultdict(list)
        
        print(f"🔍 PM Pattern Discovery Engine initialized")
        print(f"  PM sessions available: {len(self.pm_scanner.pm_sessions)}")
        print(f"  Analysis components integrated: 3")
    
    def _initialize_discovery_config(self) -> Dict:
        """Initialize discovery configuration parameters"""
        return {
            "pattern_strength_thresholds": {
                PatternStrength.WEAK: (0.0, 0.3),
                PatternStrength.MODERATE: (0.3, 0.6),
                PatternStrength.STRONG: (0.6, 0.8),
                PatternStrength.EXCEPTIONAL: (0.8, 1.0)
            },
            "integration_confidence_threshold": 0.7,
            "archaeological_validation_threshold": 0.6,
            "cross_session_relevance_threshold": 0.5,
            "predictive_reliability_threshold": 0.8,
            "pattern_correlation_weights": {
                "temporal_correlation": 0.25,
                "event_type_correlation": 0.25,
                "range_level_correlation": 0.20,
                "directional_move_correlation": 0.20,
                "archaeological_match_correlation": 0.10
            }
        }
    
    def _load_archaeological_validation(self) -> Dict:
        """Load archaeological validation criteria"""
        return {
            "validated_patterns": {
                "20% Momentum Filter": {
                    "event_signature": [EventType.FVG_REDELIVERY, EventType.SWEEP_BUY_SIDE],
                    "range_signature": RangeLevel.MOMENTUM_FILTER,
                    "move_signature": [MoveType.CONSOLIDATION, MoveType.EXPANSION],
                    "continuation_probability": 0.447,
                    "evolution_strength": 0.92,
                    "validation_weight": 0.85
                },
                "40% Sweep Acceleration": {
                    "event_signature": [EventType.SWEEP_DOUBLE, EventType.FVG_CONTINUATION],
                    "range_signature": RangeLevel.SWEEP_ACCELERATION,
                    "move_signature": [MoveType.EXPANSION, MoveType.CASCADE],
                    "continuation_probability": 1.000,
                    "evolution_strength": 0.89,
                    "validation_weight": 0.95  # Perfect continuation
                },
                "60% FVG Equilibrium": {
                    "event_signature": [EventType.FVG_FIRST_PRESENTED, EventType.PD_EQUILIBRIUM_TEST],
                    "range_signature": RangeLevel.FVG_EQUILIBRIUM,
                    "move_signature": [MoveType.BREAKOUT, MoveType.EXPANSION],
                    "continuation_probability": 1.000,
                    "evolution_strength": 0.93,  # Highest
                    "validation_weight": 0.98
                },
                "80% Completion Zone": {
                    "event_signature": [EventType.SWEEP_DOUBLE, EventType.CONSOLIDATION_RANGE],
                    "range_signature": RangeLevel.SWEEP_COMPLETION,
                    "move_signature": [MoveType.EXHAUSTION, MoveType.REVERSAL],
                    "continuation_probability": 1.000,
                    "evolution_strength": 0.92,
                    "validation_weight": 0.92
                }
            },
            "pattern_evolution_markers": {
                "cross_session_continuation": 1.000,  # 100% documented
                "htf_confluence_detection": 1.000,    # 100% accuracy
                "velocity_consistency": {"20%": 0.68, "40%": 1.00, "60%": 0.89, "80%": 1.00}
            }
        }
    
    def discover_integrated_patterns(self) -> DiscoverySession:
        """Execute comprehensive pattern discovery across all PM sessions"""
        
        print(f"🚀 Starting integrated PM pattern discovery...")
        discovery_start = datetime.now()
        
        # Step 1: Scan all PM sessions for event clusters
        pm_patterns = self.pm_scanner.scan_all_pm_sessions()
        print(f"  📊 PM event patterns found: {len(pm_patterns)}")
        
        # Step 2: Integrate multi-dimensional analysis
        integrated_patterns = []
        
        for pm_pattern in pm_patterns:
            integrated = self._create_integrated_pattern(pm_pattern)
            if integrated and integrated.integration_confidence >= self.discovery_config["integration_confidence_threshold"]:
                integrated_patterns.append(integrated)
        
        print(f"  🔬 High-confidence integrated patterns: {len(integrated_patterns)}")
        
        # Step 3: Pattern evolution tracking
        pattern_evolutions = self._track_pattern_evolutions(integrated_patterns)
        print(f"  📈 Pattern evolutions identified: {len(pattern_evolutions)}")
        
        # Step 4: Generate discovery insights
        insights = self._generate_discovery_insights(integrated_patterns, pattern_evolutions)
        
        # Step 5: Compile discovery statistics
        statistics = self._compile_discovery_statistics(integrated_patterns, pattern_evolutions)
        
        discovery_session = DiscoverySession(
            session_id=f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            discovery_timestamp=discovery_start,
            sessions_analyzed=len(self.pm_scanner.pm_sessions),
            patterns_discovered=integrated_patterns,
            pattern_evolutions=pattern_evolutions,
            discovery_statistics=statistics,
            insights_generated=insights
        )
        
        print(f"✅ Pattern discovery complete! Duration: {(datetime.now() - discovery_start).total_seconds():.1f}s")
        
        return discovery_session
    
    def _create_integrated_pattern(self, pm_pattern: PMEventPattern) -> Optional[IntegratedPattern]:
        """Create integrated pattern from PM event pattern"""
        
        # Extract events from the cluster for classification
        cluster_events = []
        for event in pm_pattern.event_cluster.events:
            event_dict = {
                'session_date': event.session_date,
                'time_minutes': event.time_minutes,
                'range_position': event.range_position,
                'context': event.context,
                'cross_tf_confluence': event.cross_tf_confluence,
                'event_type_id': event.event_type_id,
                'liquidity_type': event.liquidity_type,
                'raw_event_data': event.raw_event_data
            }
            cluster_events.append(event_dict)
        
        # Classify all events in the cluster
        event_classifications = self.event_classifier.classify_event_cluster(cluster_events)
        
        # Enhanced directional move detection if basic move exists
        directional_move_signature = None
        if pm_pattern.directional_move:
            # Convert PM events to format expected by move detector
            all_session_events = []
            for event in pm_pattern.event_cluster.events:
                event_dict = {
                    'time_minutes': event.time_minutes,
                    'price_level': event.price_level,
                    'range_position': event.range_position,
                    'context': event.context,
                    'volatility_window': event.volatility_window,
                    'price_delta_1m': event.price_delta_1m,
                    'price_delta_5m': event.price_delta_5m,
                    'price_delta_15m': event.price_delta_15m
                }
                all_session_events.append(event_dict)
            
            # Create baseline context from cluster
            baseline_context = {
                'avg_volatility': pm_pattern.event_cluster.avg_volatility,
                'range_movement': pm_pattern.event_cluster.range_movement,
                'htf_confluence_count': pm_pattern.event_cluster.htf_confluence_count,
                'dominant_context': pm_pattern.event_cluster.dominant_context,
                'primary_event_type': pm_pattern.event_cluster.primary_event_type
            }
            
            # Detect enhanced directional move
            directional_move_signature = self.move_detector.detect_directional_move(
                all_session_events, 
                pm_pattern.event_cluster.cluster_end_minute,
                pm_pattern.session_date,
                baseline_context
            )
        
        # Archaeological validation
        archaeological_validation = self._validate_against_archaeological_patterns(
            pm_pattern, event_classifications, directional_move_signature
        )
        
        # Correlation analysis
        correlation_analysis = self._perform_correlation_analysis(
            pm_pattern, event_classifications, directional_move_signature
        )
        
        # Predictive metrics calculation
        predictive_metrics = self._calculate_predictive_metrics(
            pm_pattern, event_classifications, directional_move_signature, archaeological_validation
        )
        
        # Integration confidence calculation
        integration_confidence = self._calculate_integration_confidence(
            pm_pattern, event_classifications, directional_move_signature, 
            archaeological_validation, correlation_analysis
        )
        
        # Discovery uniqueness assessment
        discovery_uniqueness = self._assess_discovery_uniqueness(
            pm_pattern, event_classifications, directional_move_signature
        )
        
        # Cross-session relevance calculation
        cross_session_relevance = self._calculate_cross_session_relevance(
            pm_pattern, event_classifications, archaeological_validation
        )
        
        # Pattern strength determination
        pattern_strength = self._determine_pattern_strength(integration_confidence)
        
        # Discovery status assignment
        discovery_status = self._assign_discovery_status(
            archaeological_validation, integration_confidence, cross_session_relevance
        )
        
        # Generate pattern tags
        pattern_tags = self._generate_pattern_tags(
            pm_pattern, event_classifications, directional_move_signature
        )
        
        integrated_pattern = IntegratedPattern(
            pattern_id=pm_pattern.pattern_id,
            session_date=pm_pattern.session_date,
            discovery_timestamp=datetime.now(),
            discovery_status=discovery_status,
            pattern_strength=pattern_strength,
            event_cluster=pm_pattern.event_cluster,
            directional_move=directional_move_signature,
            event_classifications=event_classifications,
            archaeological_validation=archaeological_validation,
            correlation_analysis=correlation_analysis,
            predictive_metrics=predictive_metrics,
            integration_confidence=integration_confidence,
            discovery_uniqueness=discovery_uniqueness,
            cross_session_relevance=cross_session_relevance,
            pattern_tags=pattern_tags
        )
        
        return integrated_pattern
    
    def _validate_against_archaeological_patterns(self, pm_pattern: PMEventPattern,
                                                classifications: List[EventClassification],
                                                move_signature: Optional[DirectionalMoveSignature]) -> Dict[str, any]:
        """Validate pattern against archaeological intelligence"""
        
        validation_results = {
            "is_validated": False,
            "validation_score": 0.0,
            "matching_patterns": [],
            "validation_details": {}
        }
        
        if not classifications:
            return validation_results
        
        # Extract dominant characteristics
        dominant_event_types = [c.event_type for c in classifications]
        dominant_range_levels = [c.range_level for c in classifications]
        dominant_range_level = Counter(dominant_range_levels).most_common(1)[0][0]
        
        # Check against each archaeological pattern
        best_match_score = 0.0
        best_match_pattern = None
        
        for pattern_name, pattern_data in self.archaeological_validation["validated_patterns"].items():
            match_score = 0.0
            
            # Event signature matching
            expected_events = pattern_data["event_signature"]
            actual_events = set(dominant_event_types)
            event_overlap = len(set(expected_events).intersection(actual_events))
            if expected_events:
                event_match_score = event_overlap / len(expected_events)
                match_score += event_match_score * 0.4
            
            # Range signature matching
            expected_range = pattern_data["range_signature"]
            if dominant_range_level == expected_range:
                match_score += 0.3
            
            # Move signature matching (if available)
            if move_signature:
                expected_moves = pattern_data["move_signature"]
                if move_signature.move_type in expected_moves:
                    match_score += 0.2
            
            # Validation weight
            match_score *= pattern_data["validation_weight"]
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_pattern = pattern_name
        
        # Validation threshold check
        validation_threshold = self.discovery_config["archaeological_validation_threshold"]
        
        validation_results.update({
            "is_validated": best_match_score >= validation_threshold,
            "validation_score": best_match_score,
            "matching_patterns": [best_match_pattern] if best_match_pattern else [],
            "validation_details": {
                "best_match": best_match_pattern,
                "match_score": best_match_score,
                "threshold_met": best_match_score >= validation_threshold
            }
        })
        
        return validation_results
    
    def _perform_correlation_analysis(self, pm_pattern: PMEventPattern,
                                    classifications: List[EventClassification],
                                    move_signature: Optional[DirectionalMoveSignature]) -> Dict[str, float]:
        """Perform comprehensive correlation analysis"""
        
        correlation_weights = self.discovery_config["pattern_correlation_weights"]
        correlations = {}
        
        # Temporal correlation (timing consistency)
        temporal_correlation = self._calculate_temporal_correlation(pm_pattern)
        correlations["temporal_correlation"] = temporal_correlation
        
        # Event type correlation (event consistency)
        event_type_correlation = self._calculate_event_type_correlation(classifications)
        correlations["event_type_correlation"] = event_type_correlation
        
        # Range level correlation (range consistency)
        range_level_correlation = self._calculate_range_level_correlation(classifications)
        correlations["range_level_correlation"] = range_level_correlation
        
        # Directional move correlation (move consistency)
        directional_move_correlation = self._calculate_directional_move_correlation(move_signature)
        correlations["directional_move_correlation"] = directional_move_correlation
        
        # Archaeological match correlation (historical consistency)
        archaeological_correlation = self._calculate_archaeological_correlation(classifications)
        correlations["archaeological_match_correlation"] = archaeological_correlation
        
        # Weighted overall correlation
        overall_correlation = sum(
            correlations[key] * weight 
            for key, weight in correlation_weights.items()
        )
        correlations["overall_correlation"] = overall_correlation
        
        return correlations
    
    def _calculate_predictive_metrics(self, pm_pattern: PMEventPattern,
                                    classifications: List[EventClassification],
                                    move_signature: Optional[DirectionalMoveSignature],
                                    archaeological_validation: Dict) -> Dict[str, float]:
        """Calculate predictive metrics for the pattern"""
        
        metrics = {}
        
        # Base predictive power from pattern strength
        base_prediction = pm_pattern.pattern_strength
        metrics["base_predictive_power"] = base_prediction
        
        # Archaeological enhancement
        if archaeological_validation["is_validated"]:
            validation_boost = archaeological_validation["validation_score"] * 0.3
            metrics["archaeological_enhancement"] = validation_boost
        else:
            metrics["archaeological_enhancement"] = 0.0
        
        # Move confirmation enhancement
        if move_signature and move_signature.significance in [MoveSignificance.SIGNIFICANT, MoveSignificance.MAJOR, MoveSignificance.EXTREME]:
            move_boost = 0.2
            metrics["move_confirmation_enhancement"] = move_boost
        else:
            metrics["move_confirmation_enhancement"] = 0.0
        
        # HTF confluence enhancement
        htf_confluent_events = [c for c in classifications if c.htf_confluence_status in [HTFConfluenceStatus.CONFIRMED, HTFConfluenceStatus.PARTIAL]]
        htf_rate = len(htf_confluent_events) / len(classifications) if classifications else 0
        metrics["htf_confluence_enhancement"] = htf_rate * 0.15
        
        # Overall predictive score
        overall_predictive = min(1.0, sum([
            metrics["base_predictive_power"],
            metrics["archaeological_enhancement"],
            metrics["move_confirmation_enhancement"],
            metrics["htf_confluence_enhancement"]
        ]))
        
        metrics["overall_predictive_score"] = overall_predictive
        
        return metrics
    
    def _calculate_integration_confidence(self, pm_pattern: PMEventPattern,
                                        classifications: List[EventClassification],
                                        move_signature: Optional[DirectionalMoveSignature],
                                        archaeological_validation: Dict,
                                        correlation_analysis: Dict) -> float:
        """Calculate integration confidence score"""
        
        confidence_factors = []
        
        # Base pattern strength
        confidence_factors.append(pm_pattern.pattern_strength)
        
        # Classification quality
        if classifications:
            avg_classification_confidence = np.mean([c.confidence_score for c in classifications])
            confidence_factors.append(avg_classification_confidence)
        else:
            confidence_factors.append(0.3)
        
        # Move detection quality
        if move_signature:
            move_confidence = 0.8 if move_signature.significance != MoveSignificance.MINOR else 0.5
            confidence_factors.append(move_confidence)
        else:
            confidence_factors.append(0.4)  # Some penalty for no move detection
        
        # Archaeological validation
        validation_confidence = archaeological_validation.get("validation_score", 0.0)
        confidence_factors.append(validation_confidence)
        
        # Correlation strength
        overall_correlation = correlation_analysis.get("overall_correlation", 0.5)
        confidence_factors.append(overall_correlation)
        
        return np.mean(confidence_factors)
    
    def _assess_discovery_uniqueness(self, pm_pattern: PMEventPattern,
                                   classifications: List[EventClassification],
                                   move_signature: Optional[DirectionalMoveSignature]) -> float:
        """Assess uniqueness of the discovered pattern"""
        
        uniqueness_factors = []
        
        # Temporal uniqueness (126-129 minute window)
        if 126 <= pm_pattern.event_cluster.cluster_start_minute <= 129:
            uniqueness_factors.append(0.9)  # High uniqueness for critical window
        else:
            uniqueness_factors.append(0.5)
        
        # Event type rarity
        if classifications:
            rare_events = [c for c in classifications if c.event_type in [
                EventType.SWEEP_DOUBLE, EventType.LIQUIDITY_VOID, EventType.PD_ARRAY_FORMATION
            ]]
            rarity_score = len(rare_events) / len(classifications)
            uniqueness_factors.append(rarity_score)
        
        # Move signature uniqueness
        if move_signature:
            if move_signature.move_type in [MoveType.CASCADE, MoveType.EXHAUSTION]:
                uniqueness_factors.append(0.8)  # Rare move types
            elif move_signature.significance in [MoveSignificance.MAJOR, MoveSignificance.EXTREME]:
                uniqueness_factors.append(0.7)  # Significant moves
            else:
                uniqueness_factors.append(0.5)
        else:
            uniqueness_factors.append(0.3)
        
        return np.mean(uniqueness_factors)
    
    def _calculate_cross_session_relevance(self, pm_pattern: PMEventPattern,
                                         classifications: List[EventClassification],
                                         archaeological_validation: Dict) -> float:
        """Calculate cross-session relevance score"""
        
        relevance_score = 0.0
        
        # Archaeological patterns have documented cross-session continuation
        if archaeological_validation.get("is_validated", False):
            # 100% cross-session continuation documented
            relevance_score += 0.6
        
        # HTF confluence indicates cross-session relevance
        htf_confluent_events = [c for c in classifications if c.htf_confluence_status in [HTFConfluenceStatus.CONFIRMED, HTFConfluenceStatus.PARTIAL]]
        if htf_confluent_events:
            htf_rate = len(htf_confluent_events) / len(classifications)
            relevance_score += htf_rate * 0.3
        
        # Critical window events have higher cross-session relevance
        critical_events = [c for c in classifications if c.temporal_context.value == "critical_window"]
        if critical_events:
            critical_rate = len(critical_events) / len(classifications)
            relevance_score += critical_rate * 0.1
        
        return min(1.0, relevance_score)
    
    def _determine_pattern_strength(self, integration_confidence: float) -> PatternStrength:
        """Determine pattern strength based on integration confidence"""
        
        thresholds = self.discovery_config["pattern_strength_thresholds"]
        
        for strength, (min_thresh, max_thresh) in thresholds.items():
            if min_thresh <= integration_confidence <= max_thresh:
                return strength
        
        return PatternStrength.WEAK
    
    def _assign_discovery_status(self, archaeological_validation: Dict,
                               integration_confidence: float,
                               cross_session_relevance: float) -> DiscoveryStatus:
        """Assign discovery status based on analysis results"""
        
        # Predictive status (highest)
        if (archaeological_validation.get("is_validated", False) and 
            integration_confidence >= 0.8 and 
            cross_session_relevance >= self.discovery_config["cross_session_relevance_threshold"]):
            return DiscoveryStatus.PREDICTIVE
        
        # Correlated status
        elif (cross_session_relevance >= self.discovery_config["cross_session_relevance_threshold"] and
              integration_confidence >= 0.6):
            return DiscoveryStatus.CORRELATED
        
        # Validated status
        elif archaeological_validation.get("is_validated", False):
            return DiscoveryStatus.VALIDATED
        
        # Candidate status (default)
        else:
            return DiscoveryStatus.CANDIDATE
    
    def _generate_pattern_tags(self, pm_pattern: PMEventPattern,
                             classifications: List[EventClassification],
                             move_signature: Optional[DirectionalMoveSignature]) -> List[str]:
        """Generate descriptive tags for the pattern"""
        
        tags = []
        
        # Temporal tags
        tags.append(f"pm_session_{pm_pattern.session_date}")
        tags.append(f"minute_{pm_pattern.event_cluster.cluster_start_minute:.0f}")
        
        # Event cluster tags
        tags.append(f"cluster_duration_{pm_pattern.event_cluster.cluster_duration:.1f}min")
        tags.append(f"primary_event_{pm_pattern.event_cluster.primary_event_type}")
        
        # Classification tags
        if classifications:
            dominant_archetype = Counter([c.liquidity_archetype.value for c in classifications]).most_common(1)[0][0]
            tags.append(f"archetype_{dominant_archetype}")
            
            htf_confluent = any(c.htf_confluence_status in [HTFConfluenceStatus.CONFIRMED, HTFConfluenceStatus.PARTIAL] for c in classifications)
            if htf_confluent:
                tags.append("htf_confluent")
        
        # Move tags
        if move_signature:
            tags.append(f"move_{move_signature.move_type.value}")
            tags.append(f"significance_{move_signature.significance.value}")
            
            if move_signature.archaeological_match:
                tags.append(f"archaeological_{move_signature.archaeological_match.replace(' ', '_').lower()}")
        
        # Directional move confirmation
        if pm_pattern.move_confirmed:
            tags.append("directional_move_confirmed")
        
        return tags
    
    # Correlation analysis helper methods
    def _calculate_temporal_correlation(self, pm_pattern: PMEventPattern) -> float:
        """Calculate temporal correlation score"""
        # Patterns in 126-129 window have perfect temporal correlation
        if 126 <= pm_pattern.event_cluster.cluster_start_minute <= 129:
            return 1.0
        else:
            return 0.5
    
    def _calculate_event_type_correlation(self, classifications: List[EventClassification]) -> float:
        """Calculate event type correlation"""
        if not classifications:
            return 0.0
        
        # Higher correlation for consistent event types
        event_types = [c.event_type for c in classifications]
        dominant_type_count = Counter(event_types).most_common(1)[0][1]
        return dominant_type_count / len(classifications)
    
    def _calculate_range_level_correlation(self, classifications: List[EventClassification]) -> float:
        """Calculate range level correlation"""
        if not classifications:
            return 0.0
        
        # Higher correlation for consistent range levels
        range_levels = [c.range_level for c in classifications]
        dominant_range_count = Counter(range_levels).most_common(1)[0][1]
        return dominant_range_count / len(classifications)
    
    def _calculate_directional_move_correlation(self, move_signature: Optional[DirectionalMoveSignature]) -> float:
        """Calculate directional move correlation"""
        if not move_signature:
            return 0.3  # Neutral score for no move
        
        # Higher correlation for significant moves with archaeological matches
        base_score = 0.6
        
        if move_signature.significance in [MoveSignificance.SIGNIFICANT, MoveSignificance.MAJOR, MoveSignificance.EXTREME]:
            base_score += 0.2
        
        if move_signature.archaeological_match:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _calculate_archaeological_correlation(self, classifications: List[EventClassification]) -> float:
        """Calculate archaeological correlation"""
        if not classifications:
            return 0.0
        
        # Higher correlation for patterns with archaeological matches
        matched_classifications = [c for c in classifications if c.archaeological_match]
        return len(matched_classifications) / len(classifications)
    
    def _track_pattern_evolutions(self, integrated_patterns: List[IntegratedPattern]) -> List[PatternEvolution]:
        """Track pattern evolutions across sessions"""
        
        # Group patterns by similar characteristics for evolution tracking
        pattern_groups = defaultdict(list)
        
        for pattern in integrated_patterns:
            # Create grouping key based on dominant characteristics
            if pattern.event_classifications:
                dominant_archetype = Counter([c.liquidity_archetype.value for c in pattern.event_classifications]).most_common(1)[0][0]
                dominant_range = Counter([c.range_level.value for c in pattern.event_classifications]).most_common(1)[0][0]
                
                grouping_key = f"{dominant_archetype}_{dominant_range}"
                pattern_groups[grouping_key].append(pattern)
        
        evolutions = []
        
        for group_key, group_patterns in pattern_groups.items():
            if len(group_patterns) >= 2:  # Need at least 2 sessions for evolution
                # Sort by date
                group_patterns.sort(key=lambda p: p.session_date)
                
                base_pattern = group_patterns[0]
                evolution_sessions = group_patterns[1:]
                
                # Calculate evolution metrics
                evolution_strength = self._calculate_evolution_strength(base_pattern, evolution_sessions)
                consistency_score = self._calculate_evolution_consistency(base_pattern, evolution_sessions)
                predictive_reliability = self._calculate_evolution_predictive_reliability(evolution_sessions)
                
                # Generate evolution insights
                insights = self._generate_evolution_insights(base_pattern, evolution_sessions)
                
                evolution = PatternEvolution(
                    evolution_id=f"evolution_{group_key}_{len(evolution_sessions)}",
                    base_pattern=base_pattern,
                    evolution_sessions=evolution_sessions,
                    evolution_strength=evolution_strength,
                    consistency_score=consistency_score,
                    predictive_reliability=predictive_reliability,
                    evolution_insights=insights
                )
                
                evolutions.append(evolution)
        
        return evolutions
    
    def _calculate_evolution_strength(self, base_pattern: IntegratedPattern, evolution_sessions: List[IntegratedPattern]) -> float:
        """Calculate pattern evolution strength"""
        
        if not evolution_sessions:
            return 0.0
        
        strength_scores = []
        
        for evolved_pattern in evolution_sessions:
            # Compare integration confidence
            confidence_similarity = 1.0 - abs(base_pattern.integration_confidence - evolved_pattern.integration_confidence)
            
            # Compare pattern strength
            strength_match = 1.0 if base_pattern.pattern_strength == evolved_pattern.pattern_strength else 0.7
            
            # Compare discovery status
            status_progression = self._calculate_status_progression(base_pattern.discovery_status, evolved_pattern.discovery_status)
            
            session_strength = np.mean([confidence_similarity, strength_match, status_progression])
            strength_scores.append(session_strength)
        
        return np.mean(strength_scores)
    
    def _calculate_evolution_consistency(self, base_pattern: IntegratedPattern, evolution_sessions: List[IntegratedPattern]) -> float:
        """Calculate evolution consistency score"""
        
        if not evolution_sessions:
            return 0.0
        
        consistency_factors = []
        
        # Archaeological validation consistency
        base_validated = base_pattern.archaeological_validation.get("is_validated", False)
        evolution_validated = [p.archaeological_validation.get("is_validated", False) for p in evolution_sessions]
        validation_consistency = sum(v == base_validated for v in evolution_validated) / len(evolution_validated)
        consistency_factors.append(validation_consistency)
        
        # Cross-session relevance consistency
        base_relevance = base_pattern.cross_session_relevance
        relevance_variations = [abs(base_relevance - p.cross_session_relevance) for p in evolution_sessions]
        relevance_consistency = 1.0 - np.mean(relevance_variations)
        consistency_factors.append(max(0.0, relevance_consistency))
        
        return np.mean(consistency_factors)
    
    def _calculate_evolution_predictive_reliability(self, evolution_sessions: List[IntegratedPattern]) -> float:
        """Calculate predictive reliability of evolved patterns"""
        
        if not evolution_sessions:
            return 0.0
        
        # Check if patterns maintain or improve predictive metrics
        predictive_scores = []
        
        for pattern in evolution_sessions:
            overall_predictive = pattern.predictive_metrics.get("overall_predictive_score", 0.0)
            predictive_scores.append(overall_predictive)
        
        # Reliability based on consistency and improvement
        if len(predictive_scores) > 1:
            trend_slope = np.polyfit(range(len(predictive_scores)), predictive_scores, 1)[0]
            reliability = np.mean(predictive_scores) + max(0, trend_slope * 0.2)  # Bonus for improving trend
        else:
            reliability = predictive_scores[0] if predictive_scores else 0.0
        
        return min(1.0, reliability)
    
    def _calculate_status_progression(self, base_status: DiscoveryStatus, evolved_status: DiscoveryStatus) -> float:
        """Calculate status progression score"""
        
        status_order = [DiscoveryStatus.CANDIDATE, DiscoveryStatus.VALIDATED, DiscoveryStatus.CORRELATED, DiscoveryStatus.PREDICTIVE]
        
        try:
            base_idx = status_order.index(base_status)
            evolved_idx = status_order.index(evolved_status)
            
            if evolved_idx >= base_idx:
                return 1.0  # Same or improved
            else:
                return 0.7  # Regressed
        except ValueError:
            return 0.5  # Unknown status
    
    def _generate_evolution_insights(self, base_pattern: IntegratedPattern, evolution_sessions: List[IntegratedPattern]) -> List[str]:
        """Generate insights about pattern evolution"""
        
        insights = []
        
        if not evolution_sessions:
            return insights
        
        # Pattern persistence
        insights.append(f"Pattern persisted across {len(evolution_sessions)} sessions")
        
        # Strength evolution
        base_strength = base_pattern.pattern_strength.value
        final_strength = evolution_sessions[-1].pattern_strength.value
        
        if final_strength != base_strength:
            insights.append(f"Pattern strength evolved from {base_strength} to {final_strength}")
        
        # Archaeological validation consistency
        base_validated = base_pattern.archaeological_validation.get("is_validated", False)
        evolution_validated_count = sum(1 for p in evolution_sessions if p.archaeological_validation.get("is_validated", False))
        
        if base_validated and evolution_validated_count == len(evolution_sessions):
            insights.append("Archaeological validation consistently maintained across sessions")
        elif evolution_validated_count > 0:
            insights.append(f"Archaeological validation in {evolution_validated_count}/{len(evolution_sessions)} evolved sessions")
        
        # Predictive reliability trend
        predictive_scores = [p.predictive_metrics.get("overall_predictive_score", 0.0) for p in evolution_sessions]
        if len(predictive_scores) > 1:
            trend_slope = np.polyfit(range(len(predictive_scores)), predictive_scores, 1)[0]
            if trend_slope > 0.05:
                insights.append("Predictive reliability shows improving trend")
            elif trend_slope < -0.05:
                insights.append("Predictive reliability shows declining trend")
            else:
                insights.append("Predictive reliability remains stable")
        
        return insights
    
    def _generate_discovery_insights(self, integrated_patterns: List[IntegratedPattern], 
                                   pattern_evolutions: List[PatternEvolution]) -> List[str]:
        """Generate comprehensive discovery insights"""
        
        insights = []
        
        if not integrated_patterns:
            return ["No patterns discovered in current analysis"]
        
        # Overall discovery statistics
        total_patterns = len(integrated_patterns)
        validated_patterns = len([p for p in integrated_patterns if p.archaeological_validation.get("is_validated", False)])
        predictive_patterns = len([p for p in integrated_patterns if p.discovery_status == DiscoveryStatus.PREDICTIVE])
        
        insights.append(f"Discovered {total_patterns} integrated patterns across PM sessions")
        insights.append(f"Archaeological validation achieved for {validated_patterns}/{total_patterns} patterns ({validated_patterns/total_patterns:.1%})")
        
        if predictive_patterns > 0:
            insights.append(f"Identified {predictive_patterns} patterns with predictive value")
        
        # Critical window analysis
        critical_window_patterns = [
            p for p in integrated_patterns 
            if 126 <= p.event_cluster.cluster_start_minute <= 129
        ]
        
        if critical_window_patterns:
            insights.append(f"Critical window (126-129 min) contains {len(critical_window_patterns)} patterns")
            
            # Analyze critical window success rate
            critical_moves_confirmed = len([p for p in critical_window_patterns if p.directional_move])
            if critical_window_patterns:
                critical_success_rate = critical_moves_confirmed / len(critical_window_patterns)
                insights.append(f"Critical window directional move confirmation rate: {critical_success_rate:.1%}")
        
        # Pattern evolution insights
        if pattern_evolutions:
            insights.append(f"Identified {len(pattern_evolutions)} pattern evolution sequences")
            
            high_reliability_evolutions = [e for e in pattern_evolutions if e.predictive_reliability >= 0.8]
            if high_reliability_evolutions:
                insights.append(f"High predictive reliability in {len(high_reliability_evolutions)} evolution sequences")
        
        # Range level effectiveness
        range_effectiveness = defaultdict(list)
        for pattern in integrated_patterns:
            if pattern.event_classifications:
                dominant_range = Counter([c.range_level.value for c in pattern.event_classifications]).most_common(1)[0][0]
                range_effectiveness[dominant_range].append(pattern.integration_confidence)
        
        if range_effectiveness:
            best_range = max(range_effectiveness.items(), key=lambda x: np.mean(x[1]))[0]
            best_range_confidence = np.mean(range_effectiveness[best_range])
            insights.append(f"Most effective range level: {best_range} (avg confidence: {best_range_confidence:.2f})")
        
        return insights
    
    def _compile_discovery_statistics(self, integrated_patterns: List[IntegratedPattern], 
                                    pattern_evolutions: List[PatternEvolution]) -> Dict[str, any]:
        """Compile comprehensive discovery statistics"""
        
        if not integrated_patterns:
            return {"error": "No patterns to analyze"}
        
        # Basic statistics
        total_patterns = len(integrated_patterns)
        
        # Status distribution
        status_dist = dict(Counter([p.discovery_status.value for p in integrated_patterns]))
        
        # Strength distribution
        strength_dist = dict(Counter([p.pattern_strength.value for p in integrated_patterns]))
        
        # Archaeological validation statistics
        validated_count = len([p for p in integrated_patterns if p.archaeological_validation.get("is_validated", False)])
        validation_rate = validated_count / total_patterns
        
        # Integration confidence statistics
        confidence_scores = [p.integration_confidence for p in integrated_patterns]
        
        # Cross-session relevance statistics
        relevance_scores = [p.cross_session_relevance for p in integrated_patterns]
        
        # Move confirmation statistics
        move_confirmed_count = len([p for p in integrated_patterns if p.directional_move is not None])
        move_confirmation_rate = move_confirmed_count / total_patterns
        
        # Critical window statistics
        critical_window_count = len([
            p for p in integrated_patterns 
            if 126 <= p.event_cluster.cluster_start_minute <= 129
        ])
        
        # Evolution statistics
        evolution_stats = {}
        if pattern_evolutions:
            evolution_stats = {
                "total_evolutions": len(pattern_evolutions),
                "avg_evolution_strength": np.mean([e.evolution_strength for e in pattern_evolutions]),
                "avg_consistency_score": np.mean([e.consistency_score for e in pattern_evolutions]),
                "avg_predictive_reliability": np.mean([e.predictive_reliability for e in pattern_evolutions])
            }
        
        return {
            "discovery_overview": {
                "total_patterns_discovered": total_patterns,
                "archaeological_validation_rate": validation_rate,
                "move_confirmation_rate": move_confirmation_rate,
                "critical_window_patterns": critical_window_count,
                "critical_window_rate": critical_window_count / total_patterns
            },
            
            "distribution_analysis": {
                "discovery_status_distribution": status_dist,
                "pattern_strength_distribution": strength_dist
            },
            
            "quality_metrics": {
                "avg_integration_confidence": np.mean(confidence_scores),
                "avg_cross_session_relevance": np.mean(relevance_scores),
                "high_confidence_patterns": len([s for s in confidence_scores if s >= 0.8]),
                "high_relevance_patterns": len([s for s in relevance_scores if s >= 0.7])
            },
            
            "evolution_analysis": evolution_stats,
            
            "predictive_insights": {
                "predictive_patterns": len([p for p in integrated_patterns if p.discovery_status == DiscoveryStatus.PREDICTIVE]),
                "avg_predictive_score": np.mean([
                    p.predictive_metrics.get("overall_predictive_score", 0.0) 
                    for p in integrated_patterns
                ]),
                "htf_confluence_rate": np.mean([
                    len([c for c in p.event_classifications if c.htf_confluence_status in [HTFConfluenceStatus.CONFIRMED, HTFConfluenceStatus.PARTIAL]]) / len(p.event_classifications) if p.event_classifications else 0
                    for p in integrated_patterns
                ])
            }
        }
    
    def save_discovery_results(self, discovery_session: DiscoverySession, output_path: str = None) -> str:
        """Save comprehensive discovery results"""
        
        if output_path is None:
            output_path = '/Users/jack/IRONFORGE/analysis/pm_pattern_discovery_results.json'
        
        # Prepare serializable data
        results = {
            "discovery_session_metadata": {
                "session_id": discovery_session.session_id,
                "discovery_timestamp": discovery_session.discovery_timestamp.isoformat(),
                "sessions_analyzed": discovery_session.sessions_analyzed,
                "patterns_discovered": len(discovery_session.patterns_discovered),
                "pattern_evolutions": len(discovery_session.pattern_evolutions)
            },
            
            "discovery_statistics": discovery_session.discovery_statistics,
            "insights_generated": discovery_session.insights_generated,
            
            "integrated_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "session_date": p.session_date,
                    "discovery_status": p.discovery_status.value,
                    "pattern_strength": p.pattern_strength.value,
                    "integration_confidence": p.integration_confidence,
                    "discovery_uniqueness": p.discovery_uniqueness,
                    "cross_session_relevance": p.cross_session_relevance,
                    
                    "event_cluster_summary": {
                        "start_minute": p.event_cluster.cluster_start_minute,
                        "duration": p.event_cluster.cluster_duration,
                        "event_count": len(p.event_cluster.events),
                        "primary_event_type": p.event_cluster.primary_event_type,
                        "htf_confluence_count": p.event_cluster.htf_confluence_count
                    },
                    
                    "directional_move_summary": {
                        "move_detected": p.directional_move is not None,
                        "move_type": p.directional_move.move_type.value if p.directional_move else None,
                        "significance": p.directional_move.significance.value if p.directional_move else None,
                        "archaeological_match": p.directional_move.archaeological_match if p.directional_move else None
                    } if p.directional_move else {"move_detected": False},
                    
                    "classification_summary": {
                        "classified_events": len(p.event_classifications),
                        "dominant_archetype": Counter([c.liquidity_archetype.value for c in p.event_classifications]).most_common(1)[0][0] if p.event_classifications else None,
                        "dominant_range_level": Counter([c.range_level.value for c in p.event_classifications]).most_common(1)[0][0] if p.event_classifications else None,
                        "htf_confluent_events": len([c for c in p.event_classifications if c.htf_confluence_status in [HTFConfluenceStatus.CONFIRMED, HTFConfluenceStatus.PARTIAL]])
                    },
                    
                    "archaeological_validation": p.archaeological_validation,
                    "correlation_analysis": p.correlation_analysis,
                    "predictive_metrics": p.predictive_metrics,
                    "pattern_tags": p.pattern_tags
                }
                for p in discovery_session.patterns_discovered
            ],
            
            "pattern_evolutions": [
                {
                    "evolution_id": e.evolution_id,
                    "sessions_tracked": len(e.evolution_sessions) + 1,  # +1 for base pattern
                    "evolution_strength": e.evolution_strength,
                    "consistency_score": e.consistency_score,
                    "predictive_reliability": e.predictive_reliability,
                    "evolution_insights": e.evolution_insights,
                    "base_pattern_id": e.base_pattern.pattern_id,
                    "evolved_pattern_ids": [p.pattern_id for p in e.evolution_sessions]
                }
                for e in discovery_session.pattern_evolutions
            ]
        }
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 PM pattern discovery results saved to: {output_path}")
        
        # Save detailed patterns separately
        detailed_path = output_path.replace('.json', '_detailed_patterns.json')
        detailed_patterns = {
            "session_metadata": {
                "session_id": discovery_session.session_id,
                "discovery_timestamp": discovery_session.discovery_timestamp.isoformat()
            },
            "detailed_integrated_patterns": [
                {
                    "pattern_metadata": {
                        "pattern_id": p.pattern_id,
                        "session_date": p.session_date,
                        "discovery_status": p.discovery_status.value,
                        "pattern_strength": p.pattern_strength.value,
                        "integration_confidence": p.integration_confidence
                    },
                    "event_cluster_details": {
                        "cluster_start_minute": p.event_cluster.cluster_start_minute,
                        "cluster_end_minute": p.event_cluster.cluster_end_minute,
                        "cluster_duration": p.event_cluster.cluster_duration,
                        "primary_event_type": p.event_cluster.primary_event_type,
                        "dominant_context": p.event_cluster.dominant_context,
                        "events": [
                            {
                                "time_minutes": e.time_minutes,
                                "range_position": e.range_position,
                                "context": e.context,
                                "cross_tf_confluence": e.cross_tf_confluence
                            }
                            for e in p.event_cluster.events
                        ]
                    },
                    "directional_move_details": {
                        "move_start_minute": p.directional_move.move_start_minute,
                        "move_duration": p.directional_move.move_duration,
                        "move_type": p.directional_move.move_type.value,
                        "significance": p.directional_move.significance.value,
                        "characteristics": {
                            "volatility_expansion_ratio": p.directional_move.characteristics.volatility_expansion_ratio,
                            "price_range_change": p.directional_move.characteristics.price_range_change,
                            "momentum_persistence": p.directional_move.characteristics.momentum_persistence
                        }
                    } if p.directional_move else None,
                    "event_classifications_details": [
                        {
                            "event_id": c.event_id,
                            "event_type": c.event_type.value,
                            "range_level": c.range_level.value,
                            "liquidity_archetype": c.liquidity_archetype.value,
                            "htf_confluence_status": c.htf_confluence_status.value,
                            "confidence_score": c.confidence_score,
                            "archaeological_match": c.archaeological_match
                        }
                        for c in p.event_classifications
                    ]
                }
                for p in discovery_session.patterns_discovered
            ]
        }
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_patterns, f, indent=2, default=str)
        
        print(f"📊 Detailed pattern data saved to: {detailed_path}")
        
        return output_path

if __name__ == "__main__":
    print("🔍 IRONFORGE PM Pattern Discovery Engine")
    print("=" * 60)
    
    discoverer = PMPatternDiscoverer()
    
    # Execute comprehensive discovery
    discovery_session = discoverer.discover_integrated_patterns()
    
    # Save results
    output_file = discoverer.save_discovery_results(discovery_session)
    
    print(f"\n✅ PM Pattern Discovery Complete!")
    print(f"  Patterns discovered: {len(discovery_session.patterns_discovered)}")
    print(f"  Pattern evolutions: {len(discovery_session.pattern_evolutions)}")
    print(f"  Discovery insights: {len(discovery_session.insights_generated)}")
    print(f"📊 Results saved to: {output_file}")