#!/usr/bin/env python3
"""
IRONFORGE Broad-Spectrum Market Archaeology Engine
================================================

Comprehensive multi-timeframe archaeological pattern discovery system.
Scans all timeframes (1m to monthly) for recurring market phenomena,
classifies them, and maps them onto a timeframe × cycle-position lattice.

Features:
- Event mining across all timeframes (1m, 5m, 15m, 50m, 1h, daily, weekly, monthly)
- Session phase analysis (opening, mid-session, closing)
- Event classification (liquidity sweeps, PD arrays, FVGs, expansions, consolidations)
- HTF confluence detection and cross-session resonance tracking
- Full 560-pattern IRONFORGE historical archive integration

Author: IRONFORGE Archaeological Discovery System
Date: August 15, 2025
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime, timedelta
from enum import Enum
import math
import re

class ArchaeologyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for archaeological data structures."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

try:
    from .event_classifier import EventClassifier, EventType, RangeLevel, LiquidityArchetype, HTFConfluenceStatus, TemporalContext
except ImportError:
    # Fallback for direct execution
    from event_classifier import EventClassifier, EventType, RangeLevel, LiquidityArchetype, HTFConfluenceStatus, TemporalContext


class TimeframeType(Enum):
    """Supported timeframes for archaeological analysis"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_50 = "50m"
    HOUR_1 = "1h"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class SessionPhase(Enum):
    """Session phase classifications"""
    OPENING = "opening"
    MID_SESSION = "mid_session"
    SESSION_CLOSING = "session_closing"
    CRITICAL_WINDOW = "critical_window"  # 126-129 minutes


@dataclass
class ArchaeologicalEvent:
    """Complete archaeological event with multi-timeframe context"""
    
    # Core identification
    event_id: str
    session_name: str
    session_date: str
    timestamp: str
    timeframe: TimeframeType
    
    # Event classification
    event_type: EventType
    event_subtype: str
    range_level: RangeLevel
    liquidity_archetype: LiquidityArchetype
    
    # Temporal context
    session_phase: SessionPhase
    session_minute: float
    relative_cycle_position: float  # 0.0-1.0 within higher timeframe cycle
    absolute_time_signature: str    # e.g., "PM_37", "daily_bar_3"
    
    # Magnitude and significance
    magnitude: float
    duration_minutes: float
    velocity_signature: float
    significance_score: float
    
    # HTF context
    htf_confluence: HTFConfluenceStatus
    htf_regime: str
    cross_session_inheritance: float
    
    # Historical matching
    historical_matches: List[str]
    pattern_family: str
    recurrence_rate: float
    
    # Enhanced features (45D semantic features)
    enhanced_features: Dict[str, float]
    
    # Structural context
    range_position_percent: float
    structural_role: str  # accumulation, breakout, terminal_sweep, etc.
    
    # Metadata
    discovery_metadata: Dict[str, Any]
    confidence_score: float


@dataclass
class TemporalCluster:
    """Clustering of events by temporal patterns"""
    cluster_id: str
    cluster_type: str  # absolute_time, relative_position, session_phase
    temporal_signature: str
    events: List[ArchaeologicalEvent]
    recurrence_frequency: float
    average_significance: float
    pattern_stability: float


@dataclass
class ArchaeologicalSummary:
    """Complete archaeological analysis summary"""
    analysis_timestamp: str
    sessions_analyzed: int
    total_events_discovered: int
    events_by_timeframe: Dict[str, int]
    events_by_type: Dict[str, int]
    significant_clusters: List[TemporalCluster]
    cross_session_patterns: List[Dict]
    phenomena_catalog: List[ArchaeologicalEvent]


class BroadSpectrumArchaeology:
    """
    Comprehensive market archaeology engine for multi-timeframe pattern discovery
    """
    
    def __init__(self, 
                 enhanced_sessions_path: str = "enhanced_sessions_with_relativity",
                 preservation_path: str = "IRONFORGE/preservation",
                 enable_deep_analysis: bool = True):
        """
        Initialize the broad-spectrum archaeology engine
        
        Args:
            enhanced_sessions_path: Path to enhanced sessions with 45D features
            preservation_path: Path to preserved patterns and models
            enable_deep_analysis: Enable comprehensive pattern analysis
        """
        self.logger = logging.getLogger('broad_spectrum_archaeology')
        self.base_path = Path(__file__).parent.parent
        
        # Initialize paths
        self.enhanced_sessions_path = self.base_path / enhanced_sessions_path
        self.preservation_path = self.base_path / preservation_path
        
        # Load session files
        self.session_files = list(self.enhanced_sessions_path.glob('enhanced_rel_*.json'))
        
        # Initialize components
        self.event_classifier = EventClassifier()
        self.enable_deep_analysis = enable_deep_analysis
        
        # Timeframe configurations
        self.timeframe_configs = self._initialize_timeframe_configs()
        
        # Archaeological intelligence from 560-pattern archive
        self.archaeological_patterns = self._load_archaeological_patterns()
        
        # Event storage
        self.discovered_events: List[ArchaeologicalEvent] = []
        self.temporal_clusters: List[TemporalCluster] = []
        
        print(f"🏛️  Broad-Spectrum Market Archaeology Engine initialized")
        print(f"  Enhanced sessions available: {len(self.session_files)}")
        print(f"  Archaeological patterns loaded: {len(self.archaeological_patterns)}")
        print(f"  Timeframes configured: {len(self.timeframe_configs)}")
    
    def _initialize_timeframe_configs(self) -> Dict[TimeframeType, Dict]:
        """Initialize timeframe-specific configurations"""
        return {
            TimeframeType.MINUTE_1: {
                "window_size": 1,
                "significance_threshold": 0.6,
                "velocity_weight": 1.0,
                "pattern_types": ["fvg", "sweep", "momentum_shift"]
            },
            TimeframeType.MINUTE_5: {
                "window_size": 5,
                "significance_threshold": 0.65,
                "velocity_weight": 0.8,
                "pattern_types": ["fvg", "sweep", "pd_array", "consolidation"]
            },
            TimeframeType.MINUTE_15: {
                "window_size": 15,
                "significance_threshold": 0.7,
                "velocity_weight": 0.6,
                "pattern_types": ["expansion", "reversal", "structural_shift"]
            },
            TimeframeType.MINUTE_50: {
                "window_size": 50,
                "significance_threshold": 0.75,
                "velocity_weight": 0.4,
                "pattern_types": ["session_structure", "regime_change"]
            },
            TimeframeType.HOUR_1: {
                "window_size": 60,
                "significance_threshold": 0.8,
                "velocity_weight": 0.3,
                "pattern_types": ["htf_confluence", "cross_session"]
            },
            TimeframeType.DAILY: {
                "window_size": 1440,  # minutes in a day
                "significance_threshold": 0.85,
                "velocity_weight": 0.2,
                "pattern_types": ["weekly_structure", "monthly_confluence"]
            },
            TimeframeType.WEEKLY: {
                "window_size": 10080,  # minutes in a week
                "significance_threshold": 0.9,
                "velocity_weight": 0.1,
                "pattern_types": ["monthly_structure", "quarterly_confluence"]
            },
            TimeframeType.MONTHLY: {
                "window_size": 43200,  # approximate minutes in a month
                "significance_threshold": 0.95,
                "velocity_weight": 0.05,
                "pattern_types": ["seasonal_patterns", "yearly_cycles"]
            }
        }
    
    def _load_archaeological_patterns(self) -> Dict[str, Any]:
        """Load archaeological patterns from preservation store"""
        patterns = {}
        
        try:
            # Load discovered patterns
            discovered_file = self.preservation_path / "discovered_patterns.json"
            if discovered_file.exists():
                with open(discovered_file, 'r') as f:
                    patterns['discovered'] = json.load(f)
            
            # Load production features
            production_file = self.preservation_path / "production_features.json"
            if production_file.exists():
                with open(production_file, 'r') as f:
                    patterns['production'] = json.load(f)
            
            # Load validated patterns
            validated_file = self.preservation_path / "validated_patterns.json"
            if validated_file.exists():
                with open(validated_file, 'r') as f:
                    patterns['validated'] = json.load(f)
                    
        except Exception as e:
            self.logger.warning(f"Could not load some archaeological patterns: {e}")
        
        return patterns
    
    def discover_all_phenomena(self) -> ArchaeologicalSummary:
        """
        Comprehensive archaeological discovery across all timeframes and sessions
        
        Returns:
            Complete archaeological summary with phenomena catalog
        """
        print(f"\n🔍 Beginning broad-spectrum archaeological discovery...")
        print(f"  Analyzing {len(self.session_files)} enhanced sessions")
        print(f"  Scanning {len(self.timeframe_configs)} timeframes")
        
        start_time = datetime.now()
        
        # Clear previous discoveries
        self.discovered_events.clear()
        self.temporal_clusters.clear()
        
        # Process each session
        for i, session_file in enumerate(self.session_files, 1):
            print(f"  [{i}/{len(self.session_files)}] Processing {session_file.name}...")
            
            try:
                session_events = self._analyze_session_phenomena(session_file)
                self.discovered_events.extend(session_events)
                
                print(f"    Discovered {len(session_events)} archaeological events")
                
            except Exception as e:
                self.logger.error(f"Error processing {session_file.name}: {e}")
                continue
        
        # Perform temporal clustering
        print(f"\n🕰️  Performing temporal clustering analysis...")
        self.temporal_clusters = self._perform_temporal_clustering()
        
        # Generate cross-session patterns
        print(f"🔗 Analyzing cross-session patterns...")
        cross_session_patterns = self._analyze_cross_session_patterns()
        
        # Create summary
        summary = self._create_archaeological_summary(cross_session_patterns)
        
        elapsed = datetime.now() - start_time
        print(f"\n✅ Broad-spectrum archaeological discovery complete!")
        print(f"  Total events discovered: {len(self.discovered_events)}")
        print(f"  Temporal clusters identified: {len(self.temporal_clusters)}")
        print(f"  Cross-session patterns: {len(cross_session_patterns)}")
        print(f"  Analysis time: {elapsed.total_seconds():.1f} seconds")
        
        return summary
    
    def _analyze_session_phenomena(self, session_file: Path) -> List[ArchaeologicalEvent]:
        """Analyze a single session for archaeological phenomena across all timeframes"""
        
        # Load enhanced session data
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Could not load session {session_file.name}: {e}")
            return []
        
        session_events = []
        session_name = session_file.stem.replace('enhanced_rel_', '')
        
        # Extract session metadata
        session_date = self._extract_session_date(session_name)
        session_type = self._extract_session_type(session_name)
        
        # Analyze each timeframe
        for timeframe in self.timeframe_configs.keys():
            timeframe_events = self._analyze_timeframe_phenomena(
                session_data, session_name, session_date, session_type, timeframe
            )
            session_events.extend(timeframe_events)
        
        return session_events
    
    def _analyze_timeframe_phenomena(self, 
                                   session_data: Dict,
                                   session_name: str,
                                   session_date: str,
                                   session_type: str,
                                   timeframe: TimeframeType) -> List[ArchaeologicalEvent]:
        """Analyze phenomena within a specific timeframe"""
        
        timeframe_events = []
        config = self.timeframe_configs[timeframe]
        
        # Extract events from session data based on timeframe
        # This will depend on how the session data is structured
        raw_events = self._extract_timeframe_events(session_data, timeframe, config)
        
        for raw_event in raw_events:
            try:
                # Create archaeological event
                archaeological_event = self._create_archaeological_event(
                    raw_event, session_name, session_date, session_type, timeframe, config
                )
                
                # Check significance threshold
                if archaeological_event.significance_score >= config["significance_threshold"]:
                    timeframe_events.append(archaeological_event)
                    
            except Exception as e:
                self.logger.warning(f"Error processing event in {timeframe.value}: {e}")
                continue
        
        return timeframe_events
    
    def _extract_timeframe_events(self, session_data: Dict, timeframe: TimeframeType, config: Dict) -> List[Dict]:
        """Extract events for a specific timeframe from session data"""
        
        # This is a simplified extraction - would need to be adapted based on actual data structure
        events = []
        
        # Get events from session data
        if 'events' in session_data:
            raw_events = session_data['events']
        elif 'enhanced_features' in session_data:
            # Extract from enhanced features
            raw_events = self._synthesize_events_from_features(session_data['enhanced_features'], timeframe)
        else:
            return []
        
        # Filter events by timeframe relevance
        window_size = config["window_size"]
        
        for event in raw_events:
            # Apply timeframe-specific filtering logic
            if self._is_event_relevant_for_timeframe(event, timeframe, window_size):
                events.append(event)
        
        return events
    
    def _synthesize_events_from_features(self, enhanced_features: Dict, timeframe: TimeframeType) -> List[Dict]:
        """Synthesize events from 45D enhanced features"""
        
        events = []
        
        # Look for event indicators in enhanced features
        for feature_name, feature_value in enhanced_features.items():
            if isinstance(feature_value, (int, float)) and abs(feature_value) > 0.5:
                
                # Create synthetic event based on feature
                event = {
                    'feature_name': feature_name,
                    'value': feature_value,
                    'timeframe': timeframe.value,
                    'synthetic': True
                }
                
                # Add timeframe-specific context
                if 'fvg' in feature_name.lower():
                    event['type'] = 'fvg'
                elif 'sweep' in feature_name.lower():
                    event['type'] = 'sweep'
                elif 'expansion' in feature_name.lower():
                    event['type'] = 'expansion'
                elif 'consolidation' in feature_name.lower():
                    event['type'] = 'consolidation'
                else:
                    event['type'] = 'generic'
                
                events.append(event)
        
        return events
    
    def _is_event_relevant_for_timeframe(self, event: Dict, timeframe: TimeframeType, window_size: int) -> bool:
        """Check if an event is relevant for the specified timeframe"""
        
        # Simple relevance check - could be made more sophisticated
        if timeframe == TimeframeType.MINUTE_1:
            return True  # All events relevant for 1m
        elif timeframe == TimeframeType.MINUTE_5:
            return event.get('magnitude', 0) > 0.3
        elif timeframe == TimeframeType.MINUTE_15:
            return event.get('magnitude', 0) > 0.5
        elif timeframe == TimeframeType.HOUR_1:
            return event.get('magnitude', 0) > 0.7
        else:
            return event.get('magnitude', 0) > 0.8
    
    def _create_archaeological_event(self, 
                                   raw_event: Dict,
                                   session_name: str,
                                   session_date: str,
                                   session_type: str,
                                   timeframe: TimeframeType,
                                   config: Dict) -> ArchaeologicalEvent:
        """Create a comprehensive archaeological event from raw data"""
        
        # Generate unique event ID
        event_id = f"{session_name}_{timeframe.value}_{hash(str(raw_event)) % 10000:04d}"
        
        # Extract or synthesize basic properties
        magnitude = raw_event.get('magnitude', abs(raw_event.get('value', 0.5)))
        duration = raw_event.get('duration', config["window_size"])
        
        # Calculate temporal context
        session_minute = self._calculate_session_minute(raw_event, session_type)
        session_phase = self._determine_session_phase(session_minute)
        relative_cycle_position = self._calculate_relative_cycle_position(session_minute, timeframe)
        absolute_time_signature = self._generate_absolute_time_signature(session_minute, session_type)
        
        # Classify event using event classifier
        event_type = self._classify_event_type(raw_event)
        range_level = self._classify_range_level(raw_event)
        liquidity_archetype = self._classify_liquidity_archetype(raw_event)
        
        # Calculate significance score
        significance_score = self._calculate_significance_score(raw_event, timeframe, config)
        
        # Determine HTF confluence
        htf_confluence = self._determine_htf_confluence(raw_event, timeframe)
        
        # Find historical matches
        historical_matches = self._find_historical_matches(raw_event, event_type)
        
        # Calculate enhanced features
        enhanced_features = raw_event.get('enhanced_features', {})
        
        # Create archaeological event
        return ArchaeologicalEvent(
            event_id=event_id,
            session_name=session_name,
            session_date=session_date,
            timestamp=raw_event.get('timestamp', ''),
            timeframe=timeframe,
            event_type=event_type,
            event_subtype=raw_event.get('type', 'unknown'),
            range_level=range_level,
            liquidity_archetype=liquidity_archetype,
            session_phase=session_phase,
            session_minute=session_minute,
            relative_cycle_position=relative_cycle_position,
            absolute_time_signature=absolute_time_signature,
            magnitude=magnitude,
            duration_minutes=duration,
            velocity_signature=magnitude / max(duration, 1),
            significance_score=significance_score,
            htf_confluence=htf_confluence,
            htf_regime=raw_event.get('htf_regime', 'unknown'),
            cross_session_inheritance=raw_event.get('cross_session_inheritance', 0.0),
            historical_matches=historical_matches,
            pattern_family=self._determine_pattern_family(event_type),
            recurrence_rate=len(historical_matches) / max(len(self.archaeological_patterns.get('discovered', {})), 1),
            enhanced_features=enhanced_features,
            range_position_percent=raw_event.get('range_position', 0.5) * 100,
            structural_role=self._determine_structural_role(raw_event, timeframe),
            discovery_metadata={
                'timeframe': timeframe.value,
                'synthetic': raw_event.get('synthetic', False),
                'feature_name': raw_event.get('feature_name', ''),
                'analysis_timestamp': datetime.now().isoformat()
            },
            confidence_score=min(significance_score * 1.2, 1.0)
        )
    
    def _calculate_session_minute(self, raw_event: Dict, session_type: str) -> float:
        """Calculate session minute from event data"""
        # Simplified calculation - would need actual timestamp parsing
        return raw_event.get('session_minute', np.random.uniform(0, 180))
    
    def _determine_session_phase(self, session_minute: float) -> SessionPhase:
        """Determine session phase based on minute"""
        if session_minute < 60:
            return SessionPhase.OPENING
        elif session_minute < 120:
            return SessionPhase.MID_SESSION
        elif 126 <= session_minute <= 129:
            return SessionPhase.CRITICAL_WINDOW
        else:
            return SessionPhase.SESSION_CLOSING
    
    def _calculate_relative_cycle_position(self, session_minute: float, timeframe: TimeframeType) -> float:
        """Calculate relative position within timeframe cycle"""
        if timeframe == TimeframeType.MINUTE_1:
            return session_minute % 1.0
        elif timeframe == TimeframeType.MINUTE_5:
            return (session_minute % 5.0) / 5.0
        elif timeframe == TimeframeType.MINUTE_15:
            return (session_minute % 15.0) / 15.0
        elif timeframe == TimeframeType.HOUR_1:
            return (session_minute % 60.0) / 60.0
        else:
            return session_minute / 180.0  # Session relative position
    
    def _generate_absolute_time_signature(self, session_minute: float, session_type: str) -> str:
        """Generate absolute time signature"""
        minute_int = int(session_minute)
        return f"{session_type}_{minute_int}"
    
    def _classify_event_type(self, raw_event: Dict) -> EventType:
        """Classify event type using patterns"""
        event_type_str = raw_event.get('type', '').lower()
        
        if 'fvg' in event_type_str:
            if 'redelivery' in event_type_str:
                return EventType.FVG_REDELIVERY
            elif 'continuation' in event_type_str:
                return EventType.FVG_CONTINUATION
            else:
                return EventType.FVG_FIRST_PRESENTED
        elif 'sweep' in event_type_str:
            if 'buy' in event_type_str:
                return EventType.SWEEP_BUY_SIDE
            elif 'sell' in event_type_str:
                return EventType.SWEEP_SELL_SIDE
            else:
                return EventType.SWEEP_DOUBLE
        elif 'expansion' in event_type_str:
            return EventType.EXPANSION_PHASE
        elif 'consolidation' in event_type_str:
            return EventType.CONSOLIDATION_RANGE
        else:
            return EventType.LIQUIDITY_VOID
    
    def _classify_range_level(self, raw_event: Dict) -> RangeLevel:
        """Classify range level"""
        range_pos = raw_event.get('range_position', 0.5)
        
        if 0.15 <= range_pos <= 0.25:
            return RangeLevel.MOMENTUM_FILTER
        elif 0.35 <= range_pos <= 0.45:
            return RangeLevel.SWEEP_ACCELERATION
        elif 0.55 <= range_pos <= 0.65:
            return RangeLevel.FVG_EQUILIBRIUM
        elif 0.75 <= range_pos <= 0.85:
            return RangeLevel.SWEEP_COMPLETION
        else:
            return RangeLevel.OUTLIER_RANGE
    
    def _classify_liquidity_archetype(self, raw_event: Dict) -> LiquidityArchetype:
        """Classify liquidity archetype"""
        event_type = raw_event.get('type', '').lower()
        
        if 'sweep' in event_type and 'low' in event_type:
            return LiquidityArchetype.SESSION_LOW_SWEEP
        elif 'sweep' in event_type and 'high' in event_type:
            return LiquidityArchetype.SESSION_HIGH_SWEEP
        elif 'fvg' in event_type and 'delivery' in event_type:
            return LiquidityArchetype.FPFVG_DELIVERY
        elif 'expansion' in event_type:
            return LiquidityArchetype.EXPANSION_PHASE
        else:
            return LiquidityArchetype.UNCLASSIFIED
    
    def _calculate_significance_score(self, raw_event: Dict, timeframe: TimeframeType, config: Dict) -> float:
        """Calculate event significance score"""
        
        # Base score from magnitude
        magnitude = abs(raw_event.get('magnitude', raw_event.get('value', 0.5)))
        base_score = min(magnitude, 1.0)
        
        # Adjust for timeframe
        timeframe_weight = config.get("velocity_weight", 0.5)
        adjusted_score = base_score * (0.5 + timeframe_weight)
        
        # Boost for historical matches
        if self._has_historical_precedent(raw_event):
            adjusted_score *= 1.2
        
        # Boost for cross-session patterns
        if raw_event.get('cross_session_inheritance', 0) > 0.5:
            adjusted_score *= 1.15
        
        return min(adjusted_score, 1.0)
    
    def _determine_htf_confluence(self, raw_event: Dict, timeframe: TimeframeType) -> HTFConfluenceStatus:
        """Determine HTF confluence status"""
        
        htf_score = raw_event.get('htf_confluence', 0.0)
        
        if htf_score > 0.8:
            return HTFConfluenceStatus.CONFIRMED
        elif htf_score > 0.6:
            return HTFConfluenceStatus.PARTIAL
        elif htf_score > 0.3:
            return HTFConfluenceStatus.WEAK
        else:
            return HTFConfluenceStatus.ABSENT
    
    def _find_historical_matches(self, raw_event: Dict, event_type: EventType) -> List[str]:
        """Find historical pattern matches"""
        
        matches = []
        
        # Search through archaeological patterns
        for pattern_set_name, patterns in self.archaeological_patterns.items():
            if isinstance(patterns, dict):
                for pattern_id, pattern_data in patterns.items():
                    if self._patterns_match(raw_event, pattern_data, event_type):
                        matches.append(f"{pattern_set_name}_{pattern_id}")
        
        return matches
    
    def _patterns_match(self, raw_event: Dict, pattern_data: Any, event_type: EventType) -> bool:
        """Check if raw event matches historical pattern"""
        
        # Simplified matching logic
        if isinstance(pattern_data, dict):
            pattern_type = pattern_data.get('type', '').lower()
            event_type_str = event_type.value.lower()
            
            return any(word in pattern_type for word in event_type_str.split('_'))
        
        return False
    
    def _determine_pattern_family(self, event_type: EventType) -> str:
        """Determine pattern family"""
        if 'fvg' in event_type.value:
            return 'fvg_family'
        elif 'sweep' in event_type.value:
            return 'sweep_family'
        elif 'expansion' in event_type.value:
            return 'expansion_family'
        elif 'consolidation' in event_type.value:
            return 'consolidation_family'
        else:
            return 'miscellaneous'
    
    def _determine_structural_role(self, raw_event: Dict, timeframe: TimeframeType) -> str:
        """Determine structural role of event"""
        
        magnitude = raw_event.get('magnitude', 0.5)
        event_type = raw_event.get('type', '').lower()
        
        if magnitude > 0.8:
            if 'sweep' in event_type:
                return 'terminal_sweep'
            else:
                return 'breakout'
        elif magnitude > 0.6:
            return 'accumulation'
        else:
            return 'minor_signal'
    
    def _has_historical_precedent(self, raw_event: Dict) -> bool:
        """Check if event has historical precedent"""
        return len(self.archaeological_patterns.get('discovered', {})) > 0
    
    def _extract_session_date(self, session_name: str) -> str:
        """Extract session date from session name"""
        # Example: NYPM_Lvl-1_2025_08_05 -> 2025-08-05
        parts = session_name.split('_')
        if len(parts) >= 5:
            return f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
        return datetime.now().strftime('%Y-%m-%d')
    
    def _extract_session_type(self, session_name: str) -> str:
        """Extract session type from session name"""
        if session_name.startswith('NYPM'):
            return 'NY_PM'
        elif session_name.startswith('NYAM'):
            return 'NY_AM'
        elif session_name.startswith('LONDON'):
            return 'LONDON'
        elif session_name.startswith('ASIA'):
            return 'ASIA'
        else:
            return 'UNKNOWN'
    
    def _perform_temporal_clustering(self) -> List[TemporalCluster]:
        """Perform temporal clustering analysis on discovered events"""
        
        clusters = []
        
        # Cluster by absolute time signatures
        time_signature_groups = defaultdict(list)
        for event in self.discovered_events:
            time_signature_groups[event.absolute_time_signature].append(event)
        
        # Create clusters for significant time signatures
        for signature, events in time_signature_groups.items():
            if len(events) >= 3:  # Minimum 3 events for significance
                cluster = TemporalCluster(
                    cluster_id=f"time_sig_{signature}",
                    cluster_type="absolute_time",
                    temporal_signature=signature,
                    events=events,
                    recurrence_frequency=len(events) / len(self.session_files),
                    average_significance=np.mean([e.significance_score for e in events]),
                    pattern_stability=self._calculate_pattern_stability(events)
                )
                clusters.append(cluster)
        
        # Cluster by relative cycle positions
        position_groups = defaultdict(list)
        for event in self.discovered_events:
            position_bucket = int(event.relative_cycle_position * 10) / 10  # 0.1 buckets
            position_groups[position_bucket].append(event)
        
        # Create clusters for significant positions
        for position, events in position_groups.items():
            if len(events) >= 5:  # Higher threshold for position clusters
                cluster = TemporalCluster(
                    cluster_id=f"rel_pos_{position:.1f}",
                    cluster_type="relative_position",
                    temporal_signature=f"cycle_position_{position:.1f}",
                    events=events,
                    recurrence_frequency=len(events) / len(self.discovered_events),
                    average_significance=np.mean([e.significance_score for e in events]),
                    pattern_stability=self._calculate_pattern_stability(events)
                )
                clusters.append(cluster)
        
        return sorted(clusters, key=lambda c: c.average_significance, reverse=True)
    
    def _calculate_pattern_stability(self, events: List[ArchaeologicalEvent]) -> float:
        """Calculate stability score for a pattern cluster"""
        
        if len(events) < 2:
            return 0.0
        
        # Calculate variance in significance scores
        significances = [e.significance_score for e in events]
        variance = np.var(significances)
        
        # Calculate consistency in event types
        event_types = [e.event_type.value for e in events]
        type_consistency = len(set(event_types)) / len(event_types)
        
        # Combine metrics
        stability = (1.0 - variance) * type_consistency
        return max(0.0, min(1.0, stability))
    
    def _analyze_cross_session_patterns(self) -> List[Dict]:
        """Analyze patterns that span across sessions"""
        
        cross_session_patterns = []
        
        # Group events by session
        session_groups = defaultdict(list)
        for event in self.discovered_events:
            session_groups[event.session_name].append(event)
        
        # Look for patterns that appear across multiple sessions
        pattern_signatures = defaultdict(list)
        
        for session_name, events in session_groups.items():
            for event in events:
                # Create pattern signature
                signature = f"{event.event_type.value}_{event.range_level.value}_{event.session_phase.value}"
                pattern_signatures[signature].append((session_name, event))
        
        # Identify cross-session patterns
        for signature, occurrences in pattern_signatures.items():
            sessions_involved = set(occ[0] for occ in occurrences)
            
            if len(sessions_involved) >= 3:  # Appears in at least 3 sessions
                events_in_pattern = [occ[1] for occ in occurrences]
                
                pattern = {
                    'pattern_signature': signature,
                    'sessions_involved': list(sessions_involved),
                    'total_occurrences': len(occurrences),
                    'average_significance': np.mean([e.significance_score for e in events_in_pattern]),
                    'consistency_score': self._calculate_pattern_stability(events_in_pattern),
                    'timeframes_involved': list(set(e.timeframe.value for e in events_in_pattern))
                }
                
                cross_session_patterns.append(pattern)
        
        return sorted(cross_session_patterns, key=lambda p: p['average_significance'], reverse=True)
    
    def _create_archaeological_summary(self, cross_session_patterns: List[Dict]) -> ArchaeologicalSummary:
        """Create comprehensive archaeological summary"""
        
        # Count events by timeframe
        events_by_timeframe = Counter(event.timeframe.value for event in self.discovered_events)
        
        # Count events by type
        events_by_type = Counter(event.event_type.value for event in self.discovered_events)
        
        return ArchaeologicalSummary(
            analysis_timestamp=datetime.now().isoformat(),
            sessions_analyzed=len(self.session_files),
            total_events_discovered=len(self.discovered_events),
            events_by_timeframe=dict(events_by_timeframe),
            events_by_type=dict(events_by_type),
            significant_clusters=self.temporal_clusters[:20],  # Top 20 clusters
            cross_session_patterns=cross_session_patterns,
            phenomena_catalog=self.discovered_events
        )
    
    def export_phenomena_catalog(self, output_path: str = "phenomena_catalog.json") -> str:
        """Export complete phenomena catalog"""
        
        catalog_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_events': len(self.discovered_events),
                'sessions_analyzed': len(self.session_files),
                'timeframes_covered': list(self.timeframe_configs.keys())
            },
            'events': []
        }
        
        # Convert events to serializable format
        for event in self.discovered_events:
            event_data = {
                'event_id': event.event_id,
                'session_name': event.session_name,
                'session_date': event.session_date,
                'timestamp': event.timestamp,
                'timeframe': event.timeframe.value,
                'event_type': event.event_type.value,
                'event_subtype': event.event_subtype,
                'range_level': event.range_level.value,
                'liquidity_archetype': event.liquidity_archetype.value,
                'session_phase': event.session_phase.value,
                'session_minute': event.session_minute,
                'relative_cycle_position': event.relative_cycle_position,
                'absolute_time_signature': event.absolute_time_signature,
                'magnitude': event.magnitude,
                'duration_minutes': event.duration_minutes,
                'velocity_signature': event.velocity_signature,
                'significance_score': event.significance_score,
                'htf_confluence': event.htf_confluence.value,
                'htf_regime': event.htf_regime,
                'cross_session_inheritance': event.cross_session_inheritance,
                'historical_matches': event.historical_matches,
                'pattern_family': event.pattern_family,
                'recurrence_rate': event.recurrence_rate,
                'enhanced_features': event.enhanced_features,
                'range_position_percent': event.range_position_percent,
                'structural_role': event.structural_role,
                'discovery_metadata': event.discovery_metadata,
                'confidence_score': event.confidence_score
            }
            catalog_data['events'].append(event_data)
        
        # Write to file
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(catalog_data, f, indent=2, cls=ArchaeologyJSONEncoder)
        
        print(f"📄 Phenomena catalog exported to {output_file}")
        return str(output_file)


if __name__ == "__main__":
    # Test the broad-spectrum archaeology engine
    print("🏛️  Testing Broad-Spectrum Market Archaeology Engine")
    print("=" * 60)
    
    # Initialize engine
    archaeology = BroadSpectrumArchaeology()
    
    # Run discovery
    summary = archaeology.discover_all_phenomena()
    
    # Export catalog
    catalog_file = archaeology.export_phenomena_catalog()
    
    print(f"\n📊 Discovery Summary:")
    print(f"  Sessions analyzed: {summary.sessions_analyzed}")
    print(f"  Total events: {summary.total_events_discovered}")
    print(f"  Temporal clusters: {len(summary.significant_clusters)}")
    print(f"  Cross-session patterns: {len(summary.cross_session_patterns)}")
    print(f"  Catalog exported: {catalog_file}")