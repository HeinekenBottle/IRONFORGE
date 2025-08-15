#!/usr/bin/env python3
"""
IRONFORGE Structural Link Analyzer
==================================

Advanced cross-timeframe relationship and cascade analysis system.
Identifies structural linkages between events across different timeframes,
analyzes causal chains, energy accumulation patterns, and cascade dynamics.

Features:
- Lead/lag relationship identification
- Cascade event chain analysis
- Energy accumulation and release patterns
- HTF → LTF structural inheritance mapping
- Cross-session structural resonance
- Predictive cascade modeling

Author: IRONFORGE Archaeological Discovery System
Date: August 15, 2025
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime, timedelta
from enum import Enum
import math
import networkx as nx

try:
    from .broad_spectrum_archaeology import ArchaeologicalEvent, TimeframeType, SessionPhase
    from .temporal_clustering_engine import TemporalCluster
except ImportError:
    # Fallback for direct execution
    from broad_spectrum_archaeology import ArchaeologicalEvent, TimeframeType, SessionPhase
    from temporal_clustering_engine import TemporalCluster


class LinkType(Enum):
    """Types of structural links"""
    LEAD_LAG = "lead_lag"
    CAUSAL_CHAIN = "causal_chain"
    RESONANCE = "resonance"
    CASCADE = "cascade"
    INHERITANCE = "inheritance"
    ENERGY_TRANSFER = "energy_transfer"
    STRUCTURAL_ECHO = "structural_echo"


class CascadeDirection(Enum):
    """Direction of cascade flow"""
    HTF_TO_LTF = "htf_to_ltf"    # Higher timeframe to lower
    LTF_TO_HTF = "ltf_to_htf"    # Lower timeframe to higher
    LATERAL = "lateral"          # Same timeframe
    BIDIRECTIONAL = "bidirectional"


@dataclass
class StructuralLink:
    """Structural relationship between events"""
    
    link_id: str
    link_type: LinkType
    
    # Source and target events
    source_event: ArchaeologicalEvent
    target_event: ArchaeologicalEvent
    
    # Link properties
    strength: float              # 0.0-1.0 relationship strength
    confidence: float           # 0.0-1.0 confidence in link
    temporal_distance: float    # Time difference (minutes)
    structural_distance: float  # Structural separation
    
    # Timeframe relationship
    source_timeframe_level: int
    target_timeframe_level: int
    timeframe_separation: int   # Levels apart
    cascade_direction: CascadeDirection
    
    # Causal properties
    causality_score: float      # Likelihood of causation
    lead_time: float           # Time lead (if applicable)
    delay_consistency: float    # How consistent the delay is
    
    # Energy dynamics
    energy_transfer: float      # Amount of energy transferred
    energy_amplification: float # Amplification factor
    momentum_preservation: float # How much momentum is preserved
    
    # Pattern properties
    pattern_similarity: float   # Similarity of event patterns
    structural_coherence: float # How structurally coherent the link is
    historical_precedent: float # How often this link pattern occurs
    
    # Validation metrics
    statistical_significance: float
    false_positive_probability: float
    supporting_evidence_count: int


@dataclass
class CascadeChain:
    """Chain of cascading events across timeframes"""
    
    chain_id: str
    chain_type: str             # linear, branching, converging
    
    # Chain structure
    events: List[ArchaeologicalEvent]
    links: List[StructuralLink]
    timeframe_sequence: List[TimeframeType]
    
    # Temporal properties
    total_duration: float       # Start to end time
    average_link_delay: float   # Average delay between links
    chain_velocity: float       # Speed of cascade propagation
    
    # Energy dynamics
    initial_energy: float       # Starting energy/magnitude
    final_energy: float         # Ending energy/magnitude
    energy_efficiency: float    # Energy preservation ratio
    amplification_factor: float # Overall amplification
    
    # Structural properties
    chain_coherence: float      # How coherent the chain is
    structural_integrity: float # How well structure is preserved
    pattern_consistency: float  # Consistency of patterns
    
    # Predictive properties
    completion_probability: float  # Likelihood of chain completing
    next_event_prediction: Optional[Dict] # Predicted next event
    risk_assessment: float      # Risk of cascade continuing


@dataclass
class EnergyAccumulation:
    """Energy accumulation pattern analysis"""
    
    accumulation_id: str
    
    # Spatial definition
    timeframe_levels: List[int]
    position_range: Tuple[float, float]
    
    # Energy properties
    accumulated_events: List[ArchaeologicalEvent]
    total_energy: float
    energy_density: float
    accumulation_rate: float
    
    # Temporal dynamics
    accumulation_duration: float
    peak_accumulation_time: float
    energy_release_events: List[ArchaeologicalEvent]
    
    # Release characteristics
    release_magnitude: float
    release_efficiency: float
    release_direction: CascadeDirection
    
    # Predictive indicators
    critical_threshold: float
    current_energy_level: float
    time_to_release: Optional[float]
    release_probability: float


@dataclass
class StructuralAnalysis:
    """Complete structural link analysis results"""
    
    analysis_id: str
    analysis_timestamp: str
    
    # Input data
    events_analyzed: List[ArchaeologicalEvent]
    total_events: int
    timeframes_covered: List[TimeframeType]
    sessions_analyzed: List[str]
    
    # Link analysis results
    structural_links: List[StructuralLink]
    link_network: nx.DiGraph
    cascade_chains: List[CascadeChain]
    energy_accumulations: List[EnergyAccumulation]
    
    # Network properties
    network_density: float
    average_path_length: float
    clustering_coefficient: float
    central_nodes: List[str]
    
    # Cascade analysis
    cascade_statistics: Dict[str, Any]
    energy_flow_patterns: Dict[str, Any]
    timeframe_interaction_matrix: np.ndarray
    
    # Predictive insights
    active_cascade_chains: List[CascadeChain]
    energy_hotspots: List[EnergyAccumulation]
    risk_assessment: Dict[str, float]


class StructuralLinkAnalyzer:
    """
    Analyzes structural relationships and cascades across timeframes
    """
    
    def __init__(self,
                 min_link_strength: float = 0.3,
                 cascade_threshold: float = 0.5,
                 energy_accumulation_threshold: float = 0.7):
        """
        Initialize the structural link analyzer
        
        Args:
            min_link_strength: Minimum strength for link detection
            cascade_threshold: Minimum strength for cascade detection
            energy_accumulation_threshold: Threshold for energy accumulation
        """
        
        self.logger = logging.getLogger('structural_link_analyzer')
        
        # Configuration
        self.min_link_strength = min_link_strength
        self.cascade_threshold = cascade_threshold
        self.energy_accumulation_threshold = energy_accumulation_threshold
        
        # Timeframe hierarchy
        self.timeframe_levels = {
            TimeframeType.MONTHLY: 0,
            TimeframeType.WEEKLY: 1,
            TimeframeType.DAILY: 2,
            TimeframeType.HOUR_1: 3,
            TimeframeType.MINUTE_50: 4,
            TimeframeType.MINUTE_15: 5,
            TimeframeType.MINUTE_5: 6,
            TimeframeType.MINUTE_1: 7
        }
        
        # Analysis results
        self.structural_links: List[StructuralLink] = []
        self.cascade_chains: List[CascadeChain] = []
        self.energy_accumulations: List[EnergyAccumulation] = []
        self.link_network: nx.DiGraph = nx.DiGraph()
        
        print(f"🔗 Structural Link Analyzer initialized")
        print(f"  Min link strength: {min_link_strength}")
        print(f"  Cascade threshold: {cascade_threshold}")
        print(f"  Energy threshold: {energy_accumulation_threshold}")
    
    def analyze_structural_relationships(self, events: List[ArchaeologicalEvent]) -> StructuralAnalysis:
        """
        Perform comprehensive structural relationship analysis
        
        Args:
            events: List of archaeological events to analyze
            
        Returns:
            Complete structural analysis with links, cascades, and energy patterns
        """
        
        print(f"\n🔗 Beginning structural relationship analysis...")
        print(f"  Events to analyze: {len(events)}")
        
        start_time = datetime.now()
        
        # Clear previous analysis
        self.structural_links.clear()
        self.cascade_chains.clear()
        self.energy_accumulations.clear()
        self.link_network.clear()
        
        # Step 1: Identify basic structural links
        print("  🔍 Identifying structural links...")
        self._identify_structural_links(events)
        
        # Step 2: Build network graph
        print("  🕸️  Building network graph...")
        self._build_link_network()
        
        # Step 3: Detect cascade chains
        print("  ⚡ Detecting cascade chains...")
        self._detect_cascade_chains(events)
        
        # Step 4: Analyze energy accumulation patterns
        print("  🔋 Analyzing energy accumulation...")
        self._analyze_energy_accumulation(events)
        
        # Step 5: Perform network analysis
        print("  📊 Performing network analysis...")
        network_properties = self._analyze_network_properties()
        
        # Step 6: Generate structural analysis
        print("  📋 Generating structural analysis...")
        analysis = self._create_structural_analysis(events, network_properties)
        
        elapsed = datetime.now() - start_time
        print(f"\n✅ Structural analysis complete!")
        print(f"  Links identified: {len(self.structural_links)}")
        print(f"  Cascade chains: {len(self.cascade_chains)}")
        print(f"  Energy accumulations: {len(self.energy_accumulations)}")
        print(f"  Analysis time: {elapsed.total_seconds():.1f} seconds")
        
        return analysis
    
    def _identify_structural_links(self, events: List[ArchaeologicalEvent]):
        """Identify structural links between events"""
        
        # Sort events by session and time
        session_groups = defaultdict(list)
        for event in events:
            session_groups[event.session_name].append(event)
        
        # Analyze each session for temporal relationships
        for session_name, session_events in session_groups.items():
            sorted_events = sorted(session_events, key=lambda e: e.session_minute)
            
            # Look for pairwise relationships
            for i, event1 in enumerate(sorted_events):
                for j, event2 in enumerate(sorted_events[i+1:], i+1):
                    
                    # Check temporal proximity (within reasonable window)
                    time_diff = event2.session_minute - event1.session_minute
                    if 0 < time_diff <= 60:  # Within 1 hour
                        
                        link = self._create_structural_link(event1, event2, time_diff)
                        
                        if link and link.strength >= self.min_link_strength:
                            self.structural_links.append(link)
        
        # Analyze cross-session relationships
        self._identify_cross_session_links(events)
        
        print(f"    Identified {len(self.structural_links)} structural links")
    
    def _create_structural_link(self, 
                              source_event: ArchaeologicalEvent, 
                              target_event: ArchaeologicalEvent,
                              time_diff: float) -> Optional[StructuralLink]:
        """Create a structural link between two events"""
        
        # Calculate basic properties
        source_level = self.timeframe_levels[source_event.timeframe]
        target_level = self.timeframe_levels[target_event.timeframe]
        timeframe_separation = abs(target_level - source_level)
        
        # Determine cascade direction
        if source_level < target_level:
            cascade_direction = CascadeDirection.HTF_TO_LTF
        elif source_level > target_level:
            cascade_direction = CascadeDirection.LTF_TO_HTF
        else:
            cascade_direction = CascadeDirection.LATERAL
        
        # Calculate link strength
        strength = self._calculate_link_strength(source_event, target_event, time_diff)
        
        if strength < self.min_link_strength:
            return None
        
        # Determine link type
        link_type = self._determine_link_type(source_event, target_event, time_diff, timeframe_separation)
        
        # Calculate additional properties
        confidence = self._calculate_link_confidence(source_event, target_event, link_type)
        causality_score = self._calculate_causality_score(source_event, target_event, time_diff)
        energy_transfer = self._calculate_energy_transfer(source_event, target_event)
        pattern_similarity = self._calculate_pattern_similarity(source_event, target_event)
        
        # Calculate structural metrics
        structural_distance = self._calculate_structural_distance(source_event, target_event)
        structural_coherence = self._calculate_structural_coherence(source_event, target_event)
        
        # Historical precedent
        historical_precedent = self._calculate_historical_precedent(source_event, target_event)
        
        # Create link
        link_id = f"link_{source_event.event_id}_{target_event.event_id}"
        
        return StructuralLink(
            link_id=link_id,
            link_type=link_type,
            source_event=source_event,
            target_event=target_event,
            strength=strength,
            confidence=confidence,
            temporal_distance=time_diff,
            structural_distance=structural_distance,
            source_timeframe_level=source_level,
            target_timeframe_level=target_level,
            timeframe_separation=timeframe_separation,
            cascade_direction=cascade_direction,
            causality_score=causality_score,
            lead_time=time_diff,
            delay_consistency=0.8,  # Placeholder
            energy_transfer=energy_transfer,
            energy_amplification=target_event.magnitude / max(source_event.magnitude, 0.01),
            momentum_preservation=min(target_event.velocity_signature / max(source_event.velocity_signature, 0.01), 2.0),
            pattern_similarity=pattern_similarity,
            structural_coherence=structural_coherence,
            historical_precedent=historical_precedent,
            statistical_significance=confidence,
            false_positive_probability=1.0 - confidence,
            supporting_evidence_count=1
        )
    
    def _calculate_link_strength(self, 
                               source_event: ArchaeologicalEvent, 
                               target_event: ArchaeologicalEvent,
                               time_diff: float) -> float:
        """Calculate the strength of a potential link"""
        
        # Base strength from event significance
        base_strength = (source_event.significance_score + target_event.significance_score) / 2
        
        # Temporal proximity bonus (closer = stronger)
        time_weight = max(0.1, 1.0 - time_diff / 60.0)  # Decay over 60 minutes
        
        # Event type compatibility
        type_compatibility = self._calculate_type_compatibility(source_event, target_event)
        
        # Magnitude relationship (amplification or preservation)
        magnitude_ratio = target_event.magnitude / max(source_event.magnitude, 0.01)
        magnitude_weight = 1.0 if 0.5 <= magnitude_ratio <= 2.0 else 0.7
        
        # HTF confluence bonus
        htf_bonus = 1.0
        if source_event.htf_confluence.value in ['confirmed', 'partial']:
            htf_bonus += 0.2
        if target_event.htf_confluence.value in ['confirmed', 'partial']:
            htf_bonus += 0.2
        
        # Combine factors
        strength = base_strength * time_weight * type_compatibility * magnitude_weight * htf_bonus
        
        return min(1.0, strength)
    
    def _calculate_type_compatibility(self, 
                                    source_event: ArchaeologicalEvent, 
                                    target_event: ArchaeologicalEvent) -> float:
        """Calculate compatibility between event types"""
        
        # Same type = high compatibility
        if source_event.event_type == target_event.event_type:
            return 1.0
        
        # Compatible patterns
        compatible_patterns = {
            'fvg_first_presented': ['fvg_redelivery', 'fvg_continuation'],
            'sweep_buy_side': ['expansion_phase', 'reversal_point'],
            'sweep_sell_side': ['expansion_phase', 'reversal_point'],
            'expansion_phase': ['consolidation_range'],
            'consolidation_range': ['expansion_phase', 'reversal_point']
        }
        
        source_type = source_event.event_type.value
        target_type = target_event.event_type.value
        
        if target_type in compatible_patterns.get(source_type, []):
            return 0.8
        
        # Same family
        source_family = source_event.pattern_family
        target_family = target_event.pattern_family
        
        if source_family == target_family:
            return 0.6
        
        return 0.4  # Base compatibility
    
    def _determine_link_type(self, 
                           source_event: ArchaeologicalEvent, 
                           target_event: ArchaeologicalEvent,
                           time_diff: float,
                           timeframe_separation: int) -> LinkType:
        """Determine the type of structural link"""
        
        # Short-term causality
        if time_diff < 5 and timeframe_separation <= 1:
            return LinkType.CAUSAL_CHAIN
        
        # Cross-timeframe cascade
        elif timeframe_separation > 0 and time_diff < 30:
            return LinkType.CASCADE
        
        # Lead-lag relationship
        elif 5 <= time_diff <= 30:
            return LinkType.LEAD_LAG
        
        # Energy transfer
        elif target_event.magnitude > source_event.magnitude * 1.2:
            return LinkType.ENERGY_TRANSFER
        
        # Resonance
        elif source_event.event_type == target_event.event_type:
            return LinkType.RESONANCE
        
        # Cross-session inheritance
        elif source_event.session_name != target_event.session_name:
            return LinkType.INHERITANCE
        
        else:
            return LinkType.STRUCTURAL_ECHO
    
    def _calculate_link_confidence(self, 
                                 source_event: ArchaeologicalEvent, 
                                 target_event: ArchaeologicalEvent,
                                 link_type: LinkType) -> float:
        """Calculate confidence in the link"""
        
        # Base confidence from event confidence
        base_confidence = (source_event.confidence_score + target_event.confidence_score) / 2
        
        # Link type specific adjustments
        type_confidence_map = {
            LinkType.CAUSAL_CHAIN: 0.9,
            LinkType.CASCADE: 0.8,
            LinkType.LEAD_LAG: 0.7,
            LinkType.ENERGY_TRANSFER: 0.8,
            LinkType.RESONANCE: 0.6,
            LinkType.INHERITANCE: 0.5,
            LinkType.STRUCTURAL_ECHO: 0.4
        }
        
        type_multiplier = type_confidence_map.get(link_type, 0.5)
        
        return min(1.0, base_confidence * type_multiplier)
    
    def _calculate_causality_score(self, 
                                 source_event: ArchaeologicalEvent, 
                                 target_event: ArchaeologicalEvent,
                                 time_diff: float) -> float:
        """Calculate likelihood of causation"""
        
        # Temporal causality (source must precede target)
        if time_diff <= 0:
            return 0.0
        
        # Closer in time = higher causality likelihood
        temporal_score = max(0.0, 1.0 - time_diff / 60.0)
        
        # Magnitude relationship (cause should influence effect size)
        magnitude_ratio = target_event.magnitude / max(source_event.magnitude, 0.01)
        magnitude_score = 1.0 if 0.3 <= magnitude_ratio <= 3.0 else 0.5
        
        # Event type logical progression
        type_progression_score = self._calculate_type_progression_score(source_event, target_event)
        
        return (temporal_score + magnitude_score + type_progression_score) / 3.0
    
    def _calculate_type_progression_score(self, 
                                        source_event: ArchaeologicalEvent, 
                                        target_event: ArchaeologicalEvent) -> float:
        """Calculate if event types follow logical progression"""
        
        # Define logical progressions
        progressions = {
            'fvg_first_presented': ['fvg_redelivery', 'fvg_continuation'],
            'sweep_buy_side': ['expansion_phase'],
            'sweep_sell_side': ['expansion_phase'],
            'expansion_phase': ['consolidation_range', 'reversal_point'],
            'consolidation_range': ['expansion_phase', 'sweep_buy_side', 'sweep_sell_side']
        }
        
        source_type = source_event.event_type.value
        target_type = target_event.event_type.value
        
        if target_type in progressions.get(source_type, []):
            return 1.0
        elif source_type == target_type:
            return 0.7  # Same type continuation
        else:
            return 0.3  # Weak progression
    
    def _calculate_energy_transfer(self, 
                                 source_event: ArchaeologicalEvent, 
                                 target_event: ArchaeologicalEvent) -> float:
        """Calculate energy transfer between events"""
        
        # Energy is related to magnitude and velocity
        source_energy = source_event.magnitude * source_event.velocity_signature
        target_energy = target_event.magnitude * target_event.velocity_signature
        
        # Transfer is the minimum of source and target energy
        transfer = min(source_energy, target_energy)
        
        # Normalize to 0-1 range
        return min(1.0, transfer)
    
    def _calculate_pattern_similarity(self, 
                                    source_event: ArchaeologicalEvent, 
                                    target_event: ArchaeologicalEvent) -> float:
        """Calculate pattern similarity between events"""
        
        similarity = 0.0
        
        # Event type similarity
        if source_event.event_type == target_event.event_type:
            similarity += 0.3
        elif source_event.pattern_family == target_event.pattern_family:
            similarity += 0.2
        
        # Archetype similarity
        if source_event.liquidity_archetype == target_event.liquidity_archetype:
            similarity += 0.2
        
        # Range level similarity
        if source_event.range_level == target_event.range_level:
            similarity += 0.2
        
        # Structural role similarity
        if source_event.structural_role == target_event.structural_role:
            similarity += 0.1
        
        # Session phase similarity
        if source_event.session_phase == target_event.session_phase:
            similarity += 0.1
        
        # HTF confluence similarity
        if source_event.htf_confluence == target_event.htf_confluence:
            similarity += 0.1
        
        return min(1.0, similarity)
    
    def _calculate_structural_distance(self, 
                                     source_event: ArchaeologicalEvent, 
                                     target_event: ArchaeologicalEvent) -> float:
        """Calculate structural distance between events"""
        
        # Timeframe distance
        source_level = self.timeframe_levels[source_event.timeframe]
        target_level = self.timeframe_levels[target_event.timeframe]
        timeframe_distance = abs(source_level - target_level) / len(self.timeframe_levels)
        
        # Position distance
        position_distance = abs(source_event.relative_cycle_position - target_event.relative_cycle_position)
        
        # Range position distance
        range_distance = abs(source_event.range_position_percent - target_event.range_position_percent) / 100.0
        
        # Combine distances
        structural_distance = math.sqrt(timeframe_distance**2 + position_distance**2 + range_distance**2)
        
        return min(1.0, structural_distance)
    
    def _calculate_structural_coherence(self, 
                                      source_event: ArchaeologicalEvent, 
                                      target_event: ArchaeologicalEvent) -> float:
        """Calculate structural coherence of the link"""
        
        coherence = 1.0
        
        # Timeframe coherence (HTF → LTF is more coherent)
        source_level = self.timeframe_levels[source_event.timeframe]
        target_level = self.timeframe_levels[target_event.timeframe]
        
        if source_level < target_level:  # HTF → LTF
            coherence *= 1.0
        elif source_level == target_level:  # Same timeframe
            coherence *= 0.8
        else:  # LTF → HTF (less coherent)
            coherence *= 0.6
        
        # Session coherence
        if source_event.session_name == target_event.session_name:
            coherence *= 1.0
        else:
            coherence *= 0.7
        
        # Pattern coherence
        pattern_similarity = self._calculate_pattern_similarity(source_event, target_event)
        coherence *= (0.5 + pattern_similarity * 0.5)
        
        return min(1.0, coherence)
    
    def _calculate_historical_precedent(self, 
                                      source_event: ArchaeologicalEvent, 
                                      target_event: ArchaeologicalEvent) -> float:
        """Calculate historical precedent for this type of link"""
        
        # Simplified calculation based on event properties
        precedent = 0.5  # Base precedent
        
        # High significance events have more precedent
        if source_event.significance_score > 0.7 and target_event.significance_score > 0.7:
            precedent += 0.2
        
        # HTF confluence adds precedent
        if source_event.htf_confluence.value in ['confirmed', 'partial']:
            precedent += 0.1
        
        # Historical matches add precedent
        if source_event.historical_matches or target_event.historical_matches:
            precedent += 0.1
        
        return min(1.0, precedent)
    
    def _identify_cross_session_links(self, events: List[ArchaeologicalEvent]):
        """Identify links that span across sessions"""
        
        # Group events by pattern signature
        pattern_groups = defaultdict(list)
        
        for event in events:
            # Create pattern signature
            signature = f"{event.event_type.value}_{event.range_level.value}_{event.absolute_time_signature}"
            pattern_groups[signature].append(event)
        
        # Look for cross-session inheritance patterns
        for signature, pattern_events in pattern_groups.items():
            sessions = set(e.session_name for e in pattern_events)
            
            if len(sessions) >= 2:  # Multi-session pattern
                
                # Create inheritance links
                for i, event1 in enumerate(pattern_events):
                    for event2 in pattern_events[i+1:]:
                        if event1.session_name != event2.session_name:
                            
                            # Calculate temporal distance between sessions
                            session_distance = self._calculate_session_distance(event1, event2)
                            
                            if session_distance <= 7:  # Within a week
                                
                                link = StructuralLink(
                                    link_id=f"inherit_{event1.event_id}_{event2.event_id}",
                                    link_type=LinkType.INHERITANCE,
                                    source_event=event1,
                                    target_event=event2,
                                    strength=0.6,  # Base inheritance strength
                                    confidence=0.7,
                                    temporal_distance=session_distance * 1440,  # Convert to minutes
                                    structural_distance=0.2,  # Low structural distance for inheritance
                                    source_timeframe_level=self.timeframe_levels[event1.timeframe],
                                    target_timeframe_level=self.timeframe_levels[event2.timeframe],
                                    timeframe_separation=0,
                                    cascade_direction=CascadeDirection.LATERAL,
                                    causality_score=0.4,
                                    lead_time=session_distance * 1440,
                                    delay_consistency=0.6,
                                    energy_transfer=0.3,
                                    energy_amplification=1.0,
                                    momentum_preservation=0.8,
                                    pattern_similarity=0.9,  # High similarity for inheritance
                                    structural_coherence=0.8,
                                    historical_precedent=0.7,
                                    statistical_significance=0.6,
                                    false_positive_probability=0.4,
                                    supporting_evidence_count=1
                                )
                                
                                self.structural_links.append(link)
    
    def _calculate_session_distance(self, event1: ArchaeologicalEvent, event2: ArchaeologicalEvent) -> float:
        """Calculate distance between sessions in days"""
        
        try:
            date1 = datetime.strptime(event1.session_date, '%Y-%m-%d')
            date2 = datetime.strptime(event2.session_date, '%Y-%m-%d')
            return abs((date2 - date1).days)
        except:
            return 1.0  # Default distance
    
    def _build_link_network(self):
        """Build network graph from structural links"""
        
        # Add nodes for each event
        all_events = set()
        for link in self.structural_links:
            all_events.add(link.source_event.event_id)
            all_events.add(link.target_event.event_id)
        
        for event_id in all_events:
            self.link_network.add_node(event_id)
        
        # Add edges for each link
        for link in self.structural_links:
            self.link_network.add_edge(
                link.source_event.event_id,
                link.target_event.event_id,
                weight=link.strength,
                link_type=link.link_type.value,
                strength=link.strength,
                confidence=link.confidence
            )
        
        print(f"    Network: {len(self.link_network.nodes)} nodes, {len(self.link_network.edges)} edges")
    
    def _detect_cascade_chains(self, events: List[ArchaeologicalEvent]):
        """Detect cascade chains in the link network"""
        
        # Find paths through the network
        strong_links = [link for link in self.structural_links if link.strength >= self.cascade_threshold]
        
        # Group links by session for cascade detection
        session_links = defaultdict(list)
        for link in strong_links:
            if link.source_event.session_name == link.target_event.session_name:
                session_links[link.source_event.session_name].append(link)
        
        # Detect cascades within each session
        for session_name, session_link_list in session_links.items():
            cascades = self._find_cascades_in_session(session_link_list)
            self.cascade_chains.extend(cascades)
        
        print(f"    Detected {len(self.cascade_chains)} cascade chains")
    
    def _find_cascades_in_session(self, links: List[StructuralLink]) -> List[CascadeChain]:
        """Find cascade chains within a session"""
        
        cascades = []
        
        # Build link graph for this session
        link_graph = nx.DiGraph()
        
        for link in links:
            link_graph.add_edge(
                link.source_event.event_id,
                link.target_event.event_id,
                link_obj=link
            )
        
        # Find paths of length 3 or more (cascades)
        for source in link_graph.nodes():
            for target in link_graph.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(link_graph, source, target, cutoff=5))
                        
                        for path in paths:
                            if len(path) >= 3:  # At least 3 events in cascade
                                cascade = self._create_cascade_chain(path, link_graph, links)
                                if cascade:
                                    cascades.append(cascade)
                    except:
                        continue
        
        return cascades
    
    def _create_cascade_chain(self, 
                            path: List[str], 
                            link_graph: nx.DiGraph,
                            all_links: List[StructuralLink]) -> Optional[CascadeChain]:
        """Create a cascade chain from a path"""
        
        if len(path) < 3:
            return None
        
        # Get events and links in the chain
        chain_events = []
        chain_links = []
        
        # Find events by ID
        event_lookup = {}
        for link in all_links:
            event_lookup[link.source_event.event_id] = link.source_event
            event_lookup[link.target_event.event_id] = link.target_event
        
        for event_id in path:
            if event_id in event_lookup:
                chain_events.append(event_lookup[event_id])
        
        # Get links between consecutive events
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            if link_graph.has_edge(source_id, target_id):
                link_obj = link_graph[source_id][target_id]['link_obj']
                chain_links.append(link_obj)
        
        if len(chain_events) < 3 or len(chain_links) < 2:
            return None
        
        # Calculate chain properties
        chain_id = f"cascade_{hash('_'.join(path)) % 10000:04d}"
        
        # Temporal properties
        start_time = min(event.session_minute for event in chain_events)
        end_time = max(event.session_minute for event in chain_events)
        total_duration = end_time - start_time
        
        link_delays = [link.temporal_distance for link in chain_links]
        average_link_delay = np.mean(link_delays) if link_delays else 0.0
        
        chain_velocity = len(chain_events) / max(total_duration, 1.0)  # Events per minute
        
        # Energy dynamics
        initial_energy = chain_events[0].magnitude
        final_energy = chain_events[-1].magnitude
        energy_efficiency = final_energy / max(initial_energy, 0.01)
        amplification_factor = final_energy / max(initial_energy, 0.01)
        
        # Chain quality
        chain_coherence = np.mean([link.structural_coherence for link in chain_links])
        structural_integrity = np.mean([link.strength for link in chain_links])
        pattern_consistency = np.mean([link.pattern_similarity for link in chain_links])
        
        # Predictive properties
        completion_probability = min(structural_integrity * 1.2, 1.0)
        
        # Timeframe sequence
        timeframe_sequence = [event.timeframe for event in chain_events]
        
        return CascadeChain(
            chain_id=chain_id,
            chain_type="linear",  # Simplified
            events=chain_events,
            links=chain_links,
            timeframe_sequence=timeframe_sequence,
            total_duration=total_duration,
            average_link_delay=average_link_delay,
            chain_velocity=chain_velocity,
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_efficiency=energy_efficiency,
            amplification_factor=amplification_factor,
            chain_coherence=chain_coherence,
            structural_integrity=structural_integrity,
            pattern_consistency=pattern_consistency,
            completion_probability=completion_probability,
            next_event_prediction=None,  # Would need more sophisticated prediction
            risk_assessment=1.0 - completion_probability
        )
    
    def _analyze_energy_accumulation(self, events: List[ArchaeologicalEvent]):
        """Analyze energy accumulation patterns"""
        
        # Group events by spatial regions
        spatial_groups = defaultdict(list)
        
        for event in events:
            # Create spatial key based on timeframe and position
            timeframe_level = self.timeframe_levels[event.timeframe]
            position_bucket = int(event.relative_cycle_position * 10)  # 10% buckets
            
            spatial_key = f"{timeframe_level}_{position_bucket}"
            spatial_groups[spatial_key].append(event)
        
        # Analyze each spatial region for energy accumulation
        for spatial_key, region_events in spatial_groups.items():
            if len(region_events) >= 3:  # Minimum events for accumulation
                
                accumulation = self._create_energy_accumulation(spatial_key, region_events)
                
                if accumulation and accumulation.energy_density >= self.energy_accumulation_threshold:
                    self.energy_accumulations.append(accumulation)
        
        print(f"    Identified {len(self.energy_accumulations)} energy accumulation zones")
    
    def _create_energy_accumulation(self, spatial_key: str, events: List[ArchaeologicalEvent]) -> Optional[EnergyAccumulation]:
        """Create energy accumulation analysis"""
        
        if len(events) < 3:
            return None
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda e: e.session_minute)
        
        # Calculate energy properties
        total_energy = sum(event.magnitude for event in events)
        energy_density = total_energy / len(events)
        
        # Temporal dynamics
        start_time = sorted_events[0].session_minute
        end_time = sorted_events[-1].session_minute
        accumulation_duration = end_time - start_time
        
        # Find peak accumulation time
        cumulative_energy = 0
        peak_time = start_time
        peak_energy = 0
        
        for event in sorted_events:
            cumulative_energy += event.magnitude
            if cumulative_energy > peak_energy:
                peak_energy = cumulative_energy
                peak_time = event.session_minute
        
        # Accumulation rate
        accumulation_rate = total_energy / max(accumulation_duration, 1.0)
        
        # Spatial properties
        timeframe_levels = list(set(self.timeframe_levels[event.timeframe] for event in events))
        positions = [event.relative_cycle_position for event in events]
        position_range = (min(positions), max(positions))
        
        # Look for release events (high magnitude events after accumulation)
        release_events = [event for event in sorted_events if event.magnitude > energy_density * 1.5]
        
        # Release characteristics
        if release_events:
            release_magnitude = max(event.magnitude for event in release_events)
            release_efficiency = release_magnitude / total_energy
        else:
            release_magnitude = 0.0
            release_efficiency = 0.0
        
        # Predictive indicators
        current_energy_level = total_energy
        critical_threshold = energy_density * len(events) * 1.2  # 20% above current
        
        accumulation_id = f"energy_{spatial_key}_{hash(str(sorted_events[0].event_id)) % 1000:03d}"
        
        return EnergyAccumulation(
            accumulation_id=accumulation_id,
            timeframe_levels=timeframe_levels,
            position_range=position_range,
            accumulated_events=events,
            total_energy=total_energy,
            energy_density=energy_density,
            accumulation_rate=accumulation_rate,
            accumulation_duration=accumulation_duration,
            peak_accumulation_time=peak_time,
            energy_release_events=release_events,
            release_magnitude=release_magnitude,
            release_efficiency=release_efficiency,
            release_direction=CascadeDirection.HTF_TO_LTF,  # Simplified
            critical_threshold=critical_threshold,
            current_energy_level=current_energy_level,
            time_to_release=None,  # Would need prediction model
            release_probability=min(current_energy_level / critical_threshold, 1.0)
        )
    
    def _analyze_network_properties(self) -> Dict[str, Any]:
        """Analyze network-level properties"""
        
        properties = {}
        
        if len(self.link_network.nodes) == 0:
            return {
                'network_density': 0.0,
                'average_path_length': 0.0,
                'clustering_coefficient': 0.0,
                'central_nodes': []
            }
        
        # Network density
        n_nodes = len(self.link_network.nodes)
        n_edges = len(self.link_network.edges)
        max_edges = n_nodes * (n_nodes - 1)
        properties['network_density'] = n_edges / max(max_edges, 1)
        
        # Average path length
        try:
            if nx.is_weakly_connected(self.link_network):
                properties['average_path_length'] = nx.average_shortest_path_length(self.link_network)
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.weakly_connected_components(self.link_network), key=len)
                subgraph = self.link_network.subgraph(largest_cc)
                properties['average_path_length'] = nx.average_shortest_path_length(subgraph)
        except:
            properties['average_path_length'] = 0.0
        
        # Clustering coefficient
        try:
            properties['clustering_coefficient'] = nx.average_clustering(self.link_network.to_undirected())
        except:
            properties['clustering_coefficient'] = 0.0
        
        # Central nodes (by degree centrality)
        try:
            centrality = nx.degree_centrality(self.link_network)
            sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            properties['central_nodes'] = [node for node, cent in sorted_centrality[:5]]
        except:
            properties['central_nodes'] = []
        
        return properties
    
    def _create_structural_analysis(self, events: List[ArchaeologicalEvent], network_properties: Dict[str, Any]) -> StructuralAnalysis:
        """Create comprehensive structural analysis"""
        
        # Basic statistics
        timeframes_covered = list(set(event.timeframe for event in events))
        sessions_analyzed = list(set(event.session_name for event in events))
        
        # Cascade statistics
        cascade_stats = {
            'total_cascades': len(self.cascade_chains),
            'average_cascade_length': np.mean([len(c.events) for c in self.cascade_chains]) if self.cascade_chains else 0,
            'longest_cascade': max([len(c.events) for c in self.cascade_chains]) if self.cascade_chains else 0,
            'cascade_efficiency': np.mean([c.energy_efficiency for c in self.cascade_chains]) if self.cascade_chains else 0,
            'high_coherence_cascades': sum(1 for c in self.cascade_chains if c.chain_coherence > 0.7)
        }
        
        # Energy flow patterns
        energy_flow = {
            'total_accumulations': len(self.energy_accumulations),
            'average_energy_density': np.mean([ea.energy_density for ea in self.energy_accumulations]) if self.energy_accumulations else 0,
            'high_density_zones': sum(1 for ea in self.energy_accumulations if ea.energy_density > 1.0),
            'release_probability': np.mean([ea.release_probability for ea in self.energy_accumulations]) if self.energy_accumulations else 0
        }
        
        # Timeframe interaction matrix
        n_timeframes = len(self.timeframe_levels)
        interaction_matrix = np.zeros((n_timeframes, n_timeframes))
        
        for link in self.structural_links:
            source_level = link.source_timeframe_level
            target_level = link.target_timeframe_level
            interaction_matrix[source_level][target_level] += link.strength
        
        # Active cascades (high completion probability)
        active_cascades = [c for c in self.cascade_chains if c.completion_probability > 0.7]
        
        # Energy hotspots (high release probability)
        energy_hotspots = [ea for ea in self.energy_accumulations if ea.release_probability > 0.8]
        
        # Risk assessment
        risk_assessment = {
            'cascade_risk': len(active_cascades) / max(len(events), 1),
            'energy_release_risk': len(energy_hotspots) / max(len(self.energy_accumulations), 1) if self.energy_accumulations else 0,
            'network_instability': 1.0 - network_properties.get('clustering_coefficient', 0.0),
            'overall_risk': 0.0  # Will be calculated
        }
        
        risk_assessment['overall_risk'] = (
            risk_assessment['cascade_risk'] + 
            risk_assessment['energy_release_risk'] + 
            risk_assessment['network_instability']
        ) / 3.0
        
        return StructuralAnalysis(
            analysis_id=f"structural_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            analysis_timestamp=datetime.now().isoformat(),
            events_analyzed=events,
            total_events=len(events),
            timeframes_covered=timeframes_covered,
            sessions_analyzed=sessions_analyzed,
            structural_links=self.structural_links,
            link_network=self.link_network.copy(),
            cascade_chains=self.cascade_chains,
            energy_accumulations=self.energy_accumulations,
            network_density=network_properties['network_density'],
            average_path_length=network_properties['average_path_length'],
            clustering_coefficient=network_properties['clustering_coefficient'],
            central_nodes=network_properties['central_nodes'],
            cascade_statistics=cascade_stats,
            energy_flow_patterns=energy_flow,
            timeframe_interaction_matrix=interaction_matrix,
            active_cascade_chains=active_cascades,
            energy_hotspots=energy_hotspots,
            risk_assessment=risk_assessment
        )
    
    def export_structural_analysis(self, analysis: StructuralAnalysis, output_path: str = "structural_analysis.json") -> str:
        """Export structural analysis to JSON file"""
        
        # Convert analysis to serializable format
        export_data = {
            'metadata': {
                'analysis_id': analysis.analysis_id,
                'analysis_timestamp': analysis.analysis_timestamp,
                'total_events': analysis.total_events,
                'timeframes_covered': [tf.value for tf in analysis.timeframes_covered],
                'sessions_analyzed': analysis.sessions_analyzed
            },
            'structural_links': [],
            'cascade_chains': [],
            'energy_accumulations': [],
            'network_properties': {
                'network_density': analysis.network_density,
                'average_path_length': analysis.average_path_length,
                'clustering_coefficient': analysis.clustering_coefficient,
                'central_nodes': analysis.central_nodes
            },
            'statistics': {
                'cascade_statistics': analysis.cascade_statistics,
                'energy_flow_patterns': analysis.energy_flow_patterns,
                'timeframe_interaction_matrix': analysis.timeframe_interaction_matrix.tolist()
            },
            'risk_assessment': analysis.risk_assessment
        }
        
        # Export structural links
        for link in analysis.structural_links:
            link_data = {
                'link_id': link.link_id,
                'link_type': link.link_type.value,
                'source_event_id': link.source_event.event_id,
                'target_event_id': link.target_event.event_id,
                'strength': link.strength,
                'confidence': link.confidence,
                'temporal_distance': link.temporal_distance,
                'structural_distance': link.structural_distance,
                'timeframe_relationship': {
                    'source_level': link.source_timeframe_level,
                    'target_level': link.target_timeframe_level,
                    'separation': link.timeframe_separation,
                    'cascade_direction': link.cascade_direction.value
                },
                'properties': {
                    'causality_score': link.causality_score,
                    'energy_transfer': link.energy_transfer,
                    'pattern_similarity': link.pattern_similarity,
                    'structural_coherence': link.structural_coherence,
                    'historical_precedent': link.historical_precedent
                }
            }
            export_data['structural_links'].append(link_data)
        
        # Export cascade chains
        for cascade in analysis.cascade_chains:
            cascade_data = {
                'chain_id': cascade.chain_id,
                'chain_type': cascade.chain_type,
                'event_count': len(cascade.events),
                'timeframe_sequence': [tf.value for tf in cascade.timeframe_sequence],
                'temporal_properties': {
                    'total_duration': cascade.total_duration,
                    'average_link_delay': cascade.average_link_delay,
                    'chain_velocity': cascade.chain_velocity
                },
                'energy_dynamics': {
                    'initial_energy': cascade.initial_energy,
                    'final_energy': cascade.final_energy,
                    'energy_efficiency': cascade.energy_efficiency,
                    'amplification_factor': cascade.amplification_factor
                },
                'quality_metrics': {
                    'chain_coherence': cascade.chain_coherence,
                    'structural_integrity': cascade.structural_integrity,
                    'pattern_consistency': cascade.pattern_consistency
                },
                'predictive_properties': {
                    'completion_probability': cascade.completion_probability,
                    'risk_assessment': cascade.risk_assessment
                },
                'event_ids': [event.event_id for event in cascade.events],
                'link_ids': [link.link_id for link in cascade.links]
            }
            export_data['cascade_chains'].append(cascade_data)
        
        # Export energy accumulations
        for accumulation in analysis.energy_accumulations:
            accumulation_data = {
                'accumulation_id': accumulation.accumulation_id,
                'spatial_definition': {
                    'timeframe_levels': accumulation.timeframe_levels,
                    'position_range': accumulation.position_range
                },
                'energy_properties': {
                    'total_energy': accumulation.total_energy,
                    'energy_density': accumulation.energy_density,
                    'accumulation_rate': accumulation.accumulation_rate
                },
                'temporal_dynamics': {
                    'accumulation_duration': accumulation.accumulation_duration,
                    'peak_accumulation_time': accumulation.peak_accumulation_time
                },
                'release_characteristics': {
                    'release_magnitude': accumulation.release_magnitude,
                    'release_efficiency': accumulation.release_efficiency,
                    'release_direction': accumulation.release_direction.value
                },
                'predictive_indicators': {
                    'current_energy_level': accumulation.current_energy_level,
                    'critical_threshold': accumulation.critical_threshold,
                    'release_probability': accumulation.release_probability
                },
                'event_ids': [event.event_id for event in accumulation.accumulated_events]
            }
            export_data['energy_accumulations'].append(accumulation_data)
        
        # Write to file
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"🔗 Structural analysis exported to {output_file}")
        return str(output_file)


if __name__ == "__main__":
    # Test the structural link analyzer
    print("🔗 Testing Structural Link Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = StructuralLinkAnalyzer()
    
    print("✅ Structural link analyzer initialized and ready for use")
    print("   Use analyze_structural_relationships() with ArchaeologicalEvent list to perform analysis")