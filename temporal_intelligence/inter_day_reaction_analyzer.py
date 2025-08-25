#!/usr/bin/env python3
"""
INTER-DAY REACTION ANALYZER
===========================

Mission: Investigate multi-timeframe reactions after previous day 40% interactions.

Analysis Scope:
1. SAME SESSION events post-40% interaction
2. NEXT SESSION behavior changes  
3. Multi-session cascade patterns
4. Session-specific sensitivity (NY_AM, LUNCH, NY_PM)
5. Timing precision measurement across timeframes
6. Phenomena type classification

Temporal Intelligence: Track how 40% zone archaeological events create
cascading effects across sessions, days, and market structure phases.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Set
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ZoneInteraction:
    """40% Zone interaction event"""
    timestamp: str
    session_id: str
    price: float
    zone_type: str
    interaction_strength: float
    session_phase: str
    day: str

@dataclass
class ReactionEvent:
    """Post-interaction reaction event"""
    timestamp: str
    session_id: str
    price: float
    event_type: str
    distance_from_zone: float
    time_delta_minutes: int
    reaction_magnitude: float
    session_phase: str

@dataclass
class CascadePattern:
    """Multi-session cascade pattern"""
    trigger_interaction: ZoneInteraction
    same_session_reactions: List[ReactionEvent]
    next_session_reactions: List[ReactionEvent]
    cascade_duration_hours: float
    total_sessions_affected: int
    pattern_signature: str

@dataclass
class SessionSensitivity:
    """Session-specific sensitivity metrics"""
    session_type: str
    reaction_frequency: float
    avg_reaction_time_minutes: float
    avg_reaction_magnitude: float
    cascade_probability: float
    timing_precision_score: float

class InterDayReactionAnalyzer:
    """
    Advanced temporal analysis system for tracking multi-timeframe reactions
    after 40% archaeological zone interactions.
    """
    
    def __init__(self, data_path: str = "data", use_test_data: bool = False):
        if use_test_data:
            self.data_path = Path(data_path) / "test_zone_interactions"
        else:
            self.data_path = Path(data_path)
        self.sessions_data = {}
        self.zone_interactions = []
        self.reaction_events = []
        self.cascade_patterns = []
        self.session_sensitivities = {}
        
        # Analysis thresholds
        self.reaction_time_window_minutes = 180  # 3 hours post-interaction
        self.next_session_analysis_hours = 24    # Track next day sessions
        self.minimum_reaction_magnitude = 5.0    # Points
        self.cascade_threshold_sessions = 2      # Minimum sessions for cascade
        
        # Session type mappings
        self.session_types = {
            'NY_AM': ['09:30', '12:00'],
            'LUNCH': ['12:00', '14:00'], 
            'NY_PM': ['14:00', '16:00'],
            'AH': ['16:00', '20:00']  # After hours
        }

    def load_session_data(self) -> bool:
        """Load IRONFORGE Enhanced Session Adapter data"""
        try:
            # Check if using test data path directly
            if "test_zone_interactions" in str(self.data_path):
                session_files = list(self.data_path.glob("enhanced_test_*.json"))
            else:
                # Load from enhanced session data directory
                enhanced_path = self.data_path / "enhanced"
                if not enhanced_path.exists():
                    logger.warning("Enhanced session data directory not found")
                    return False
                session_files = list(enhanced_path.glob("enhanced_*.json"))
                
            if not session_files:
                logger.warning("No enhanced session data files found")
                return False
                
            for file_path in session_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Extract session identifier from filename
                    # e.g., enhanced_NY_PM_Lvl-1_2025_07_29.json -> NY_PM_2025_07_29
                    filename_parts = file_path.stem.replace('enhanced_', '').split('_')
                    if len(filename_parts) >= 4:
                        session_type = '_'.join(filename_parts[:-3])  # NY_PM
                        date_parts = filename_parts[-3:]  # [2025, 07, 29]
                        session_date = '-'.join(date_parts)  # 2025-07-29
                        session_id = f"{session_type}_{session_date}"
                    else:
                        session_id = file_path.stem.replace('enhanced_', '')
                    
                    self.sessions_data[session_id] = data
                    
            logger.info(f"Loaded {len(self.sessions_data)} enhanced session datasets")
            return True
            
        except Exception as e:
            logger.error(f"Error loading enhanced session data: {e}")
            return False

    def identify_40_percent_interactions(self) -> List[ZoneInteraction]:
        """Identify all 40% archaeological zone interactions across Enhanced Session Adapter data"""
        interactions = []
        
        for session_id, data in self.sessions_data.items():
            # Process session events from Enhanced Session Adapter structure
            session_events = self._extract_session_events(data)
            session_metadata = data.get('session_metadata', {})
            session_date = session_metadata.get('session_date', '')
            
            for event in session_events:
                if self._is_40_percent_zone_interaction(event):
                    # Build full timestamp from session date and event time
                    event_time = event.get('timestamp', event.get('time', ''))
                    if session_date and event_time and len(event_time.split()) == 1:
                        # Only add date if event_time is just time (not already full timestamp)
                        full_timestamp = f"{session_date} {event_time}"
                    else:
                        full_timestamp = event_time
                        
                    interaction = ZoneInteraction(
                        timestamp=full_timestamp,
                        session_id=session_id,
                        price=event.get('price', event.get('level', 0.0)),
                        zone_type=event.get('zone_type', event.get('type', 'dimensional')),
                        interaction_strength=event.get('strength', event.get('significance', 0.8)),
                        session_phase=session_metadata.get('session_type', 'unknown'),
                        day=session_date
                    )
                    interactions.append(interaction)
        
        self.zone_interactions = interactions
        logger.info(f"Identified {len(interactions)} 40% zone interactions from Enhanced Session Adapter data")
        return interactions

    def _is_40_percent_zone_interaction(self, event: dict) -> bool:
        """Determine if event is a 40% archaeological zone interaction using Theory B validation"""
        # Quick check for pre-validated archaeological zone interactions
        if event.get('is_archaeological_zone') and event.get('zone_percentage') == 40:
            return True
            
        # Core 40% zone criteria
        zone_percentage = event.get('zone_percentage')
        if zone_percentage != 40:
            return False
        
        # Must be dimensional archaeological zone type
        archaeological_zone_type = event.get('archaeological_zone_type')
        if archaeological_zone_type != 'dimensional':
            return False
        
        # Theory B validation: dimensional relationship to FINAL session range
        dimensional_relationship = event.get('dimensional_relationship')
        if dimensional_relationship != 'final_range':
            return False
        
        # Verify interaction type indicates zone engagement
        interaction_type = event.get('interaction_type')
        valid_interactions = {'zone_touch', 'zone_pierce', 'zone_rejection', 'zone_acceptance'}
        if interaction_type not in valid_interactions:
            return False
        
        # Temporal significance check - events that "know" future completion
        temporal_significance = event.get('temporal_significance', 0)
        if temporal_significance < 0.7:  # High temporal non-locality threshold
            return False
        
        # Zone precision validation - Theory B requires high precision to final range
        zone_precision = event.get('zone_precision', 0)
        if zone_precision < 7.55:  # Points precision threshold from empirical proof
            return False
        
        # Additional validation: forward-looking positioning
        forward_positioning = event.get('forward_positioning', False)
        if not forward_positioning:
            return False
        
        # All criteria met - this is an authentic 40% dimensional zone interaction
        return True

    def _extract_session_events(self, session_data: dict) -> List[dict]:
        """Extract all events from Enhanced Session Adapter data structure"""
        events = []
        
        # Extract from various Enhanced Session Adapter event categories
        event_sources = [
            'session_events',
            'fpfvg_events', 
            'session_patterns',
            'archaeological_zones',
            'temporal_events',
            'dimensional_events'
        ]
        
        for source in event_sources:
            if source in session_data:
                source_events = session_data[source]
                if isinstance(source_events, list):
                    events.extend(source_events)
                elif isinstance(source_events, dict):
                    # Handle nested event structures
                    for key, value in source_events.items():
                        if isinstance(value, list):
                            events.extend(value)
                        elif isinstance(value, dict) and 'events' in value:
                            events.extend(value['events'])
        
        # Also check session_features for archaeological zone events
        session_features = session_data.get('session_features', {})
        archaeological_zones = session_features.get('archaeological_zones', {})
        if isinstance(archaeological_zones, dict):
            for zone_type, zone_data in archaeological_zones.items():
                if isinstance(zone_data, dict) and 'interactions' in zone_data:
                    for interaction in zone_data['interactions']:
                        interaction['zone_type'] = zone_type
                        # Mark as validated archaeological zone interaction
                        interaction['is_archaeological_zone'] = True
                        events.append(interaction)
        
        # Add metadata to events for analysis
        session_metadata = session_data.get('session_metadata', {})
        for event in events:
            event['session_date'] = session_metadata.get('session_date', '')
            event['session_type'] = session_metadata.get('session_type', '')
            
        return events

    def track_same_session_reactions(self, interaction: ZoneInteraction) -> List[ReactionEvent]:
        """Track reaction events within the same session after 40% interaction"""
        reactions = []
        
        if interaction.session_id not in self.sessions_data:
            return reactions
            
        session_data = self.sessions_data[interaction.session_id]
        session_events = self._extract_session_events(session_data)
        interaction_time = pd.to_datetime(interaction.timestamp).tz_localize(None)
        
        # Look for events after the interaction within time window
        for event in session_events:
            # Build full event timestamp
            event_time_str = event.get('timestamp', event.get('time', ''))
            if not event_time_str:
                continue
                
            # Handle timestamp construction for Enhanced Session Adapter data
            if len(event_time_str.split()) == 1:  # Just time, need to add date
                session_date = event.get('session_date', interaction.day)
                full_timestamp = f"{session_date} {event_time_str}"
            else:
                full_timestamp = event_time_str
                
            try:
                event_time = pd.to_datetime(full_timestamp).tz_localize(None)
            except:
                continue
            
            if event_time <= interaction_time:
                continue
                
            time_delta = (event_time - interaction_time).total_seconds() / 60
            if time_delta > self.reaction_time_window_minutes:
                continue
                
            # Calculate reaction properties
            event_price = event.get('price', event.get('level', 0))
            price_distance = abs(event_price - interaction.price)
            if price_distance < self.minimum_reaction_magnitude:
                continue
                
            reaction = ReactionEvent(
                timestamp=full_timestamp,
                session_id=interaction.session_id,
                price=event_price,
                event_type=event.get('type', event.get('event_type', 'reaction')),
                distance_from_zone=price_distance,
                time_delta_minutes=int(time_delta),
                reaction_magnitude=price_distance,
                session_phase=event.get('session_type', interaction.session_phase)
            )
            reactions.append(reaction)
            
        logger.info(f"Found {len(reactions)} same-session reactions for {interaction.session_id}")
        return reactions

    def analyze_next_session_behavior(self, interaction: ZoneInteraction) -> List[ReactionEvent]:
        """Analyze behavior changes in subsequent sessions after 40% interaction"""
        reactions = []
        interaction_time = pd.to_datetime(interaction.timestamp).tz_localize(None)
        interaction_day = interaction_time.date()
        
        # Look for sessions in the next 24 hours
        for session_id, data in self.sessions_data.items():
            if session_id == interaction.session_id:
                continue
                
            # Check if this session is within next-day analysis window
            session_start = self._get_session_start_time(data)
            if not session_start:
                continue
                
            if session_start.date() != interaction_day:
                time_diff_hours = (session_start - interaction_time).total_seconds() / 3600
                if time_diff_hours < 0 or time_diff_hours > self.next_session_analysis_hours:
                    continue
            
            # Analyze events in this next session
            session_events = self._extract_session_events(data)
            for event in session_events:
                # Handle timestamp construction
                event_time_str = event.get('timestamp', event.get('time', ''))
                if not event_time_str:
                    continue
                    
                if len(event_time_str.split()) == 1:
                    session_date = event.get('session_date', '')
                    if session_date:
                        full_timestamp = f"{session_date} {event_time_str}"
                    else:
                        continue
                else:
                    full_timestamp = event_time_str
                    
                try:
                    event_time = pd.to_datetime(full_timestamp).tz_localize(None)
                except:
                    continue
                    
                time_since_interaction = (event_time - interaction_time).total_seconds() / 60
                
                # Look for significant events that might be reactions
                event_price = event.get('price', event.get('level', 0))
                price_distance = abs(event_price - interaction.price)
                if price_distance >= self.minimum_reaction_magnitude:
                    reaction = ReactionEvent(
                        timestamp=full_timestamp,
                        session_id=session_id,
                        price=event_price,
                        event_type=event.get('type', event.get('event_type', 'reaction')),
                        distance_from_zone=price_distance,
                        time_delta_minutes=int(time_since_interaction),
                        reaction_magnitude=price_distance,
                        session_phase=event.get('session_type', 'unknown')
                    )
                    reactions.append(reaction)
        
        logger.info(f"Found {len(reactions)} next-session reactions for {interaction.session_id}")
        return reactions

    def map_cascade_patterns(self) -> List[CascadePattern]:
        """Map multi-session cascade patterns triggered by 40% interactions"""
        cascade_patterns = []
        
        for interaction in self.zone_interactions:
            same_session_reactions = self.track_same_session_reactions(interaction)
            next_session_reactions = self.analyze_next_session_behavior(interaction)
            
            # Determine if this qualifies as a cascade pattern
            total_sessions = len(set([r.session_id for r in next_session_reactions]))
            if total_sessions >= self.cascade_threshold_sessions or len(same_session_reactions) >= 3:
                
                # Calculate cascade duration
                all_reactions = same_session_reactions + next_session_reactions
                if all_reactions:
                    start_time = pd.to_datetime(interaction.timestamp).tz_localize(None)
                    end_time = pd.to_datetime(max(r.timestamp for r in all_reactions)).tz_localize(None)
                    duration_hours = (end_time - start_time).total_seconds() / 3600
                else:
                    duration_hours = 0
                
                # Generate pattern signature
                signature = self._generate_pattern_signature(interaction, same_session_reactions, next_session_reactions)
                
                cascade = CascadePattern(
                    trigger_interaction=interaction,
                    same_session_reactions=same_session_reactions,
                    next_session_reactions=next_session_reactions,
                    cascade_duration_hours=duration_hours,
                    total_sessions_affected=total_sessions + 1,  # Include trigger session
                    pattern_signature=signature
                )
                cascade_patterns.append(cascade)
        
        self.cascade_patterns = cascade_patterns
        logger.info(f"Mapped {len(cascade_patterns)} cascade patterns")
        return cascade_patterns

    def measure_session_sensitivity(self) -> Dict[str, SessionSensitivity]:
        """Measure session-specific sensitivity to 40% zone interactions"""
        sensitivity_data = defaultdict(lambda: {
            'reaction_count': 0,
            'total_interactions': 0,
            'reaction_times': [],
            'reaction_magnitudes': [],
            'cascade_count': 0
        })
        
        # Collect data by session type
        for interaction in self.zone_interactions:
            session_type = self._classify_session_type(interaction.session_phase)
            sensitivity_data[session_type]['total_interactions'] += 1
            
            # Count reactions from this interaction
            same_session_reactions = self.track_same_session_reactions(interaction)
            next_session_reactions = self.analyze_next_session_behavior(interaction)
            
            if same_session_reactions or next_session_reactions:
                sensitivity_data[session_type]['reaction_count'] += 1
                
                # Collect timing and magnitude data
                for reaction in same_session_reactions + next_session_reactions:
                    sensitivity_data[session_type]['reaction_times'].append(reaction.time_delta_minutes)
                    sensitivity_data[session_type]['reaction_magnitudes'].append(reaction.reaction_magnitude)
            
            # Check for cascade
            total_sessions = len(set([r.session_id for r in next_session_reactions]))
            if total_sessions >= self.cascade_threshold_sessions:
                sensitivity_data[session_type]['cascade_count'] += 1
        
        # Calculate sensitivity metrics
        sensitivities = {}
        for session_type, data in sensitivity_data.items():
            if data['total_interactions'] == 0:
                continue
                
            sensitivity = SessionSensitivity(
                session_type=session_type,
                reaction_frequency=data['reaction_count'] / data['total_interactions'],
                avg_reaction_time_minutes=np.mean(data['reaction_times']) if data['reaction_times'] else 0,
                avg_reaction_magnitude=np.mean(data['reaction_magnitudes']) if data['reaction_magnitudes'] else 0,
                cascade_probability=data['cascade_count'] / data['total_interactions'],
                timing_precision_score=self._calculate_timing_precision(data['reaction_times'])
            )
            sensitivities[session_type] = sensitivity
        
        self.session_sensitivities = sensitivities
        logger.info(f"Calculated sensitivity metrics for {len(sensitivities)} session types")
        return sensitivities

    def _determine_session_phase(self, timestamp: str) -> str:
        """Determine session phase from timestamp"""
        try:
            dt = pd.to_datetime(timestamp)
            time_str = dt.strftime('%H:%M')
            
            for phase, (start, end) in self.session_types.items():
                if start <= time_str < end:
                    return phase
            return 'AH'  # After hours default
        except:
            return 'unknown'

    def _classify_session_type(self, session_phase: str) -> str:
        """Classify session type for sensitivity analysis"""
        phase_mapping = {
            'NY_AM': 'NY_AM',
            'LUNCH': 'LUNCH',
            'NY_PM': 'NY_PM',
            'AH': 'AH',
            'unknown': 'OTHER'
        }
        return phase_mapping.get(session_phase, 'OTHER')

    def _get_session_start_time(self, session_data: dict) -> Optional[pd.Timestamp]:
        """Extract session start time from session data"""
        try:
            if 'start_time' in session_data:
                return pd.to_datetime(session_data['start_time'])
            elif 'events' in session_data and session_data['events']:
                return pd.to_datetime(session_data['events'][0]['timestamp'])
            return None
        except:
            return None

    def _generate_pattern_signature(self, interaction: ZoneInteraction, 
                                  same_reactions: List[ReactionEvent],
                                  next_reactions: List[ReactionEvent]) -> str:
        """Generate unique signature for cascade pattern"""
        same_count = len(same_reactions)
        next_count = len(next_reactions)
        sessions_affected = len(set([r.session_id for r in next_reactions]))
        
        return f"{interaction.zone_type}_{interaction.session_phase}_{same_count}s_{next_count}n_{sessions_affected}sess"

    def _calculate_timing_precision(self, reaction_times: List[int]) -> float:
        """Calculate timing precision score based on reaction time consistency"""
        if not reaction_times:
            return 0.0
            
        # Higher precision = lower standard deviation of reaction times
        std_dev = np.std(reaction_times)
        mean_time = np.mean(reaction_times)
        
        if mean_time == 0:
            return 0.0
            
        # Precision score: inverse of coefficient of variation, scaled 0-100
        coefficient_of_variation = std_dev / mean_time
        precision_score = max(0, 100 - (coefficient_of_variation * 100))
        return min(precision_score, 100.0)

    def generate_analysis_report(self) -> dict:
        """Generate comprehensive inter-day reaction analysis report"""
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_sessions_analyzed': len(self.sessions_data),
                'total_40_percent_interactions': len(self.zone_interactions),
                'total_cascade_patterns': len(self.cascade_patterns),
                'analysis_time_window_minutes': self.reaction_time_window_minutes,
                'next_session_analysis_hours': self.next_session_analysis_hours
            },
            
            'zone_interactions_summary': {
                'total_interactions': len(self.zone_interactions),
                'by_session_phase': self._summarize_by_session_phase(),
                'by_zone_type': self._summarize_by_zone_type(),
                'temporal_distribution': self._analyze_temporal_distribution()
            },
            
            'cascade_patterns_analysis': {
                'total_cascades': len(self.cascade_patterns),
                'average_duration_hours': np.mean([c.cascade_duration_hours for c in self.cascade_patterns]) if self.cascade_patterns else 0,
                'average_sessions_affected': np.mean([c.total_sessions_affected for c in self.cascade_patterns]) if self.cascade_patterns else 0,
                'pattern_signatures': [c.pattern_signature for c in self.cascade_patterns],
                'most_common_signatures': self._get_most_common_signatures()
            },
            
            'session_sensitivity_analysis': {
                session_type: asdict(sensitivity) 
                for session_type, sensitivity in self.session_sensitivities.items()
            },
            
            'timing_precision_insights': self._analyze_timing_precision(),
            'phenomena_classification': self._classify_phenomena_types(),
            'predictive_indicators': self._identify_predictive_indicators()
        }
        
        return report

    def _summarize_by_session_phase(self) -> dict:
        """Summarize interactions by session phase"""
        phase_counts = defaultdict(int)
        for interaction in self.zone_interactions:
            phase_counts[interaction.session_phase] += 1
        return dict(phase_counts)

    def _summarize_by_zone_type(self) -> dict:
        """Summarize interactions by zone type"""
        type_counts = defaultdict(int)
        for interaction in self.zone_interactions:
            type_counts[interaction.zone_type] += 1
        return dict(type_counts)

    def _analyze_temporal_distribution(self) -> dict:
        """Analyze temporal distribution of interactions"""
        hours = []
        for interaction in self.zone_interactions:
            try:
                dt = pd.to_datetime(interaction.timestamp)
                hours.append(dt.hour)
            except:
                continue
        
        if not hours:
            return {}
            
        return {
            'peak_hour': max(set(hours), key=hours.count),
            'hour_distribution': {f"{hour:02d}:00": hours.count(hour) for hour in range(24) if hour in hours}
        }

    def _get_most_common_signatures(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most common cascade pattern signatures"""
        signature_counts = defaultdict(int)
        for cascade in self.cascade_patterns:
            signature_counts[cascade.pattern_signature] += 1
        
        return sorted(signature_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def _analyze_timing_precision(self) -> dict:
        """Analyze timing precision across different contexts"""
        precision_by_session = {}
        precision_by_zone_type = defaultdict(list)
        
        for session_type, sensitivity in self.session_sensitivities.items():
            precision_by_session[session_type] = sensitivity.timing_precision_score
        
        for cascade in self.cascade_patterns:
            zone_type = cascade.trigger_interaction.zone_type
            reaction_times = [r.time_delta_minutes for r in cascade.same_session_reactions + cascade.next_session_reactions]
            if reaction_times:
                precision = self._calculate_timing_precision(reaction_times)
                precision_by_zone_type[zone_type].append(precision)
        
        return {
            'by_session_type': precision_by_session,
            'by_zone_type': {zt: np.mean(precisions) for zt, precisions in precision_by_zone_type.items()},
            'overall_average': np.mean(list(precision_by_session.values())) if precision_by_session else 0
        }

    def _classify_phenomena_types(self) -> dict:
        """Classify different types of reaction phenomena"""
        phenomena = {
            'immediate_reactions': 0,      # < 30 min
            'delayed_reactions': 0,        # 30 min - 3 hours
            'next_session_echoes': 0,      # Following session
            'multi_session_cascades': 0,   # Multiple sessions
            'precision_events': 0          # High timing precision
        }
        
        for cascade in self.cascade_patterns:
            all_reactions = cascade.same_session_reactions + cascade.next_session_reactions
            
            # Classify reaction timing
            for reaction in cascade.same_session_reactions:
                if reaction.time_delta_minutes < 30:
                    phenomena['immediate_reactions'] += 1
                elif reaction.time_delta_minutes <= 180:
                    phenomena['delayed_reactions'] += 1
            
            if cascade.next_session_reactions:
                phenomena['next_session_echoes'] += 1
            
            if cascade.total_sessions_affected > 2:
                phenomena['multi_session_cascades'] += 1
            
            # Check timing precision
            reaction_times = [r.time_delta_minutes for r in all_reactions]
            if reaction_times and self._calculate_timing_precision(reaction_times) > 70:
                phenomena['precision_events'] += 1
        
        return phenomena

    def _identify_predictive_indicators(self) -> dict:
        """Identify patterns that serve as predictive indicators"""
        indicators = {
            'high_cascade_triggers': [],
            'precision_timing_zones': [],
            'session_transition_patterns': [],
            'magnitude_amplification_zones': []
        }
        
        # High cascade probability triggers
        for interaction in self.zone_interactions:
            cascade_reactions = sum(1 for c in self.cascade_patterns 
                                 if c.trigger_interaction.session_id == interaction.session_id)
            if cascade_reactions > 0:
                trigger_profile = f"{interaction.zone_type}_{interaction.session_phase}"
                indicators['high_cascade_triggers'].append(trigger_profile)
        
        # Precision timing zones (>80% timing precision)
        for session_type, sensitivity in self.session_sensitivities.items():
            if sensitivity.timing_precision_score > 80:
                indicators['precision_timing_zones'].append(session_type)
        
        # Session transition patterns (interactions near session boundaries)
        transition_window_minutes = 15  # 15 minutes before/after session boundaries
        for interaction in self.zone_interactions:
            interaction_time = pd.to_datetime(interaction.timestamp)
            hour_minute = interaction_time.time()
            
            # Check proximity to session boundaries
            for session_type, (start_str, end_str) in self.session_types.items():
                start_time = pd.to_datetime(start_str, format='%H:%M').time()
                end_time = pd.to_datetime(end_str, format='%H:%M').time()
                
                # Calculate minutes from boundaries
                start_minutes = abs((hour_minute.hour * 60 + hour_minute.minute) - 
                                  (start_time.hour * 60 + start_time.minute))
                end_minutes = abs((hour_minute.hour * 60 + hour_minute.minute) - 
                                (end_time.hour * 60 + end_time.minute))
                
                if start_minutes <= transition_window_minutes or end_minutes <= transition_window_minutes:
                    pattern = f"{interaction.zone_type}_{session_type}_boundary"
                    indicators['session_transition_patterns'].append(pattern)
                    break
        
        # Magnitude amplification (reactions larger than trigger)
        for cascade in self.cascade_patterns:
            avg_reaction_magnitude = np.mean([r.reaction_magnitude for r in 
                                           cascade.same_session_reactions + cascade.next_session_reactions])
            if avg_reaction_magnitude > cascade.trigger_interaction.interaction_strength * 1.5:
                indicators['magnitude_amplification_zones'].append(
                    f"{cascade.trigger_interaction.zone_type}_{cascade.trigger_interaction.session_phase}"
                )
        
        return indicators

    def run_full_analysis(self) -> dict:
        """Execute complete inter-day reaction analysis pipeline"""
        logger.info("Starting Inter-Day Reaction Analysis")
        
        # Load data
        if not self.load_session_data():
            logger.error("Failed to load session data")
            return {}
        
        # Execute analysis pipeline
        self.identify_40_percent_interactions()
        self.map_cascade_patterns()
        self.measure_session_sensitivity()
        
        # Generate comprehensive report
        report = self.generate_analysis_report()
        
        logger.info("Inter-Day Reaction Analysis Complete")
        return report

def main():
    """Execute Inter-Day Reaction Analysis"""
    # Use test dataset with generated 40% zone interactions
    analyzer = InterDayReactionAnalyzer(use_test_data=True)
    
    # Run comprehensive analysis
    analysis_report = analyzer.run_full_analysis()
    
    # Save results
    output_file = Path("data/inter_day_reaction_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_report, f, indent=2, default=str)
    
    # Display key findings
    print("\n" + "="*60)
    print("INTER-DAY REACTION ANALYSIS - KEY FINDINGS")
    print("="*60)
    
    if analysis_report:
        metadata = analysis_report.get('analysis_metadata', {})
        cascades = analysis_report.get('cascade_patterns_analysis', {})
        sensitivity = analysis_report.get('session_sensitivity_analysis', {})
        
        print(f"📊 Sessions Analyzed: {metadata.get('total_sessions_analyzed', 0)}")
        print(f"⚡ 40% Zone Interactions: {metadata.get('total_40_percent_interactions', 0)}")
        print(f"🌊 Cascade Patterns: {cascades.get('total_cascades', 0)}")
        print(f"⏱️  Average Cascade Duration: {cascades.get('average_duration_hours', 0):.1f} hours")
        print(f"📈 Sessions Affected per Cascade: {cascades.get('average_sessions_affected', 0):.1f}")
        
        print("\n🎯 Session Sensitivity Rankings:")
        for session_type, sens in sensitivity.items():
            print(f"  {session_type}: {sens.get('reaction_frequency', 0):.1%} reaction rate, "
                  f"{sens.get('timing_precision_score', 0):.0f} precision score")
        
        phenomena = analysis_report.get('phenomena_classification', {})
        if phenomena:
            print(f"\n🔬 Phenomena Types:")
            for phen_type, count in phenomena.items():
                print(f"  {phen_type.replace('_', ' ').title()}: {count}")
    
    print(f"\n💾 Full analysis saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()