#!/usr/bin/env python3
"""
IRONFORGE PM Event Scanner
==========================

Scans PM sessions for events occurring at minute_offset 126-129 (37-38 minutes into session)
that last 2.5-3.5 minutes and are followed by significant directional moves within 10-15 minutes.

Data Structure Mapping:
- PM sessions start at 13:30 ET (time_minutes=0)
- Target window: 126-129 minutes = 15:36-15:39 ET  
- Events stored in rich_node_features with time_minutes, raw_json context
- Sequential events allow duration calculation
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

@dataclass
class PMEvent:
    """Individual PM session event in target window"""
    session_date: str
    time_minutes: float
    timestamp: str
    price_level: float
    normalized_price: float
    range_position: float
    action_type: str
    context: str
    volatility_window: float
    price_delta_1m: float
    price_delta_5m: float
    price_delta_15m: float
    cross_tf_confluence: bool
    event_type_id: int
    liquidity_type: int
    session_position: float
    energy_state: float
    raw_event_data: Dict

@dataclass
class PMEventCluster:
    """Cluster of events in 126-129 minute window"""
    session_date: str
    cluster_start_minute: float
    cluster_end_minute: float
    cluster_duration: float
    events: List[PMEvent]
    primary_event_type: str
    primary_action: str
    dominant_context: str
    avg_volatility: float
    range_movement: float
    htf_confluence_count: int

@dataclass  
class DirectionalMove:
    """Directional move following PM event cluster"""
    move_start_minute: float
    move_end_minute: float
    move_duration: float
    volatility_expansion: float
    price_range_change: float
    range_position_change: float
    move_type: str  # expansion, breakout, cascade
    significance_score: float

@dataclass
class PMEventPattern:
    """Complete PM event pattern with directional move"""
    pattern_id: str
    session_date: str
    event_cluster: PMEventCluster
    directional_move: Optional[DirectionalMove]
    move_confirmed: bool
    time_to_move: Optional[float]
    pattern_strength: float
    liquidity_archetype: str
    range_level_category: str

class PMEventScanner:
    """
    Scans PM sessions for events in the critical 126-129 minute window
    """
    
    def __init__(self, sessions_path: str = None):
        self.logger = logging.getLogger('pm_event_scanner')
        
        if sessions_path is None:
            sessions_path = '/Users/jack/IRONFORGE/enhanced_sessions_with_relativity'
        
        self.sessions_path = Path(sessions_path)
        self.pm_sessions = self._discover_pm_sessions()
        
        # Analysis parameters
        self.target_window_start = 126  # minutes
        self.target_window_end = 129    # minutes
        self.min_event_duration = 2.5  # minutes
        self.max_event_duration = 3.5  # minutes
        self.directional_move_window = 15  # minutes to look ahead
        self.min_directional_move_window = 10  # minimum time before checking
        
        print(f"🕒 PM Event Scanner initialized")
        print(f"  PM sessions found: {len(self.pm_sessions)}")
        print(f"  Target window: {self.target_window_start}-{self.target_window_end} minutes")
    
    def _discover_pm_sessions(self) -> List[Path]:
        """Discover all PM session files"""
        pm_files = []
        
        if not self.sessions_path.exists():
            return pm_files
        
        # Look for NY_PM or NYPM session files
        pm_patterns = ['*NY_PM*.json', '*NYPM*.json']
        
        for pattern in pm_patterns:
            pm_files.extend(self.sessions_path.glob(pattern))
        
        return sorted(pm_files)
    
    def scan_all_pm_sessions(self) -> List[PMEventPattern]:
        """Scan all PM sessions for target events"""
        print(f"🔍 Scanning {len(self.pm_sessions)} PM sessions for 126-129 minute events...")
        
        all_patterns = []
        
        for session_file in self.pm_sessions:
            try:
                patterns = self._scan_single_pm_session(session_file)
                all_patterns.extend(patterns)
                
                if patterns:
                    print(f"  {session_file.name}: {len(patterns)} patterns found")
            except Exception as e:
                self.logger.error(f"Error scanning {session_file}: {e}")
        
        print(f"  ✅ Total patterns discovered: {len(all_patterns)}")
        return all_patterns
    
    def _scan_single_pm_session(self, session_file: Path) -> List[PMEventPattern]:
        """Scan a single PM session for target events"""
        
        # Load session data
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        session_date = self._extract_session_date(session_file.name)
        
        # Extract events from graph data if available
        graph_events = self._extract_graph_events(session_data, session_date)
        
        # Also check enhanced relativity data
        relativity_events = self._extract_relativity_events(session_data, session_date)
        
        # Combine and deduplicate events
        all_events = graph_events + relativity_events
        all_events = self._deduplicate_events(all_events)
        
        # Filter events in target window
        target_events = [
            event for event in all_events
            if self.target_window_start <= event.time_minutes <= self.target_window_end
        ]
        
        if not target_events:
            return []
        
        # Group events into clusters
        event_clusters = self._group_events_into_clusters(target_events, session_date)
        
        # Find directional moves following each cluster
        patterns = []
        for cluster in event_clusters:
            directional_move = self._detect_directional_move_after_cluster(cluster, all_events)
            
            pattern = PMEventPattern(
                pattern_id=f"{session_date}_{cluster.cluster_start_minute:.1f}",
                session_date=session_date,
                event_cluster=cluster,
                directional_move=directional_move,
                move_confirmed=directional_move is not None,
                time_to_move=directional_move.move_start_minute - cluster.cluster_end_minute if directional_move else None,
                pattern_strength=self._calculate_pattern_strength(cluster, directional_move),
                liquidity_archetype=self._classify_liquidity_archetype(cluster),
                range_level_category=self._classify_range_level(cluster)
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _extract_session_date(self, filename: str) -> str:
        """Extract session date from filename"""
        # Look for date pattern YYYY_MM_DD
        date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
        if date_match:
            year, month, day = date_match.groups()
            return f"{year}-{month}-{day}"
        return "unknown"
    
    def _extract_graph_events(self, session_data: Dict, session_date: str) -> List[PMEvent]:
        """Extract events from graph node features"""
        events = []
        
        # Look for rich_node_features in various locations
        rich_features = None
        
        if 'rich_node_features' in session_data:
            rich_features = session_data['rich_node_features']
        elif 'graph_data' in session_data and 'rich_node_features' in session_data['graph_data']:
            rich_features = session_data['graph_data']['rich_node_features']
        elif 'nodes' in session_data and 'rich_node_features' in session_data:
            rich_features = session_data['rich_node_features']
        
        if not rich_features:
            return events
        
        for feature_str in rich_features:
            if isinstance(feature_str, str) and 'RichNodeFeature' in feature_str:
                event = self._parse_rich_node_feature(feature_str, session_date)
                if event:
                    events.append(event)
        
        return events
    
    def _parse_rich_node_feature(self, feature_str: str, session_date: str) -> Optional[PMEvent]:
        """Parse a RichNodeFeature string into PMEvent"""
        try:
            # Extract time_minutes
            time_match = re.search(r'time_minutes=([-\d.]+)', feature_str)
            if not time_match:
                return None
            
            time_minutes = float(time_match.group(1))
            
            # Extract other numeric fields
            normalized_price = self._extract_float_field(feature_str, 'normalized_price')
            price_delta_1m = self._extract_float_field(feature_str, 'price_delta_1m')
            price_delta_5m = self._extract_float_field(feature_str, 'price_delta_5m') 
            price_delta_15m = self._extract_float_field(feature_str, 'price_delta_15m')
            volatility_window = self._extract_float_field(feature_str, 'volatility_window')
            cross_tf_confluence = self._extract_float_field(feature_str, 'cross_tf_confluence', 0.0) > 0.5
            event_type_id = int(self._extract_float_field(feature_str, 'event_type_id', 0))
            liquidity_type = int(self._extract_float_field(feature_str, 'liquidity_type', 0))
            session_position = self._extract_float_field(feature_str, 'session_position')
            energy_state = self._extract_float_field(feature_str, 'energy_state')
            
            # Extract raw_json
            raw_json_match = re.search(r"raw_json=(\{[^}]+\})", feature_str)
            raw_json = {}
            if raw_json_match:
                try:
                    # Clean up the string and parse
                    raw_str = raw_json_match.group(1)
                    raw_str = raw_str.replace("'", '"')
                    raw_json = json.loads(raw_str)
                except:
                    pass
            
            # Extract event details from raw_json
            timestamp = raw_json.get('timestamp', '00:00:00')
            price_level = raw_json.get('price_level', raw_json.get('price', 0))
            action_type = raw_json.get('action', raw_json.get('movement_type', 'unknown'))
            context = raw_json.get('context', '')
            
            # Calculate range position if available
            range_position = normalized_price  # Default fallback
            
            return PMEvent(
                session_date=session_date,
                time_minutes=time_minutes,
                timestamp=timestamp,
                price_level=price_level,
                normalized_price=normalized_price,
                range_position=range_position,
                action_type=action_type,
                context=context,
                volatility_window=volatility_window,
                price_delta_1m=price_delta_1m,
                price_delta_5m=price_delta_5m,
                price_delta_15m=price_delta_15m,
                cross_tf_confluence=cross_tf_confluence,
                event_type_id=event_type_id,
                liquidity_type=liquidity_type,
                session_position=session_position,
                energy_state=energy_state,
                raw_event_data=raw_json
            )
            
        except Exception as e:
            self.logger.debug(f"Failed to parse RichNodeFeature: {e}")
            return None
    
    def _extract_float_field(self, feature_str: str, field_name: str, default: float = 0.0) -> float:
        """Extract float field from feature string"""
        pattern = f'{field_name}=([-\d.e]+)'
        match = re.search(pattern, feature_str)
        if match:
            try:
                return float(match.group(1))
            except:
                return default
        return default
    
    def _extract_relativity_events(self, session_data: Dict, session_date: str) -> List[PMEvent]:
        """Extract events from price_movements relativity data"""
        events = []
        
        price_movements = session_data.get('price_movements', [])
        
        for movement in price_movements:
            # Calculate time_minutes from timestamp
            time_minutes = self._calculate_time_minutes_from_timestamp(movement.get('timestamp', ''))
            
            if time_minutes is None:
                continue
            
            event = PMEvent(
                session_date=session_date,
                time_minutes=time_minutes,
                timestamp=movement.get('timestamp', ''),
                price_level=movement.get('price_level', 0),
                normalized_price=movement.get('normalized_price', 0),
                range_position=movement.get('range_position', movement.get('normalized_price', 0)),
                action_type=movement.get('movement_type', 'unknown'),
                context=f"Price movement: {movement.get('movement_type', '')}",
                volatility_window=0.0,  # Not available in this data
                price_delta_1m=0.0,
                price_delta_5m=0.0,
                price_delta_15m=0.0,
                cross_tf_confluence=False,  # Not available
                event_type_id=0,
                liquidity_type=0,
                session_position=movement.get('normalized_time', 0),
                energy_state=0.0,
                raw_event_data=movement
            )
            
            events.append(event)
        
        return events
    
    def _calculate_time_minutes_from_timestamp(self, timestamp: str) -> Optional[float]:
        """Calculate minutes from session start (13:30) given timestamp"""
        if not timestamp or ':' not in timestamp:
            return None
        
        try:
            # Parse HH:MM:SS format
            time_parts = timestamp.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            
            # PM session starts at 13:30 (1:30 PM)
            session_start_hour = 13
            session_start_minute = 30
            
            # Convert to minutes from session start
            total_minutes = (hours * 60 + minutes) - (session_start_hour * 60 + session_start_minute)
            
            return float(total_minutes)
            
        except:
            return None
    
    def _deduplicate_events(self, events: List[PMEvent]) -> List[PMEvent]:
        """Remove duplicate events based on time and price"""
        seen = set()
        unique_events = []
        
        for event in events:
            key = (event.time_minutes, event.price_level, event.action_type)
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        return sorted(unique_events, key=lambda e: e.time_minutes)
    
    def _group_events_into_clusters(self, events: List[PMEvent], session_date: str) -> List[PMEventCluster]:
        """Group nearby events into clusters of 2.5-3.5 minute duration"""
        if not events:
            return []
        
        clusters = []
        current_cluster_events = [events[0]]
        
        for i in range(1, len(events)):
            current_event = events[i]
            cluster_start_time = current_cluster_events[0].time_minutes
            cluster_duration = current_event.time_minutes - cluster_start_time
            
            # If adding this event keeps duration within bounds, add to current cluster
            if cluster_duration <= self.max_event_duration:
                current_cluster_events.append(current_event)
            else:
                # Finalize current cluster if it meets duration requirements
                cluster_duration = current_cluster_events[-1].time_minutes - current_cluster_events[0].time_minutes
                if cluster_duration >= self.min_event_duration:
                    cluster = self._create_event_cluster(current_cluster_events, session_date)
                    clusters.append(cluster)
                
                # Start new cluster
                current_cluster_events = [current_event]
        
        # Check final cluster
        if len(current_cluster_events) > 1:
            cluster_duration = current_cluster_events[-1].time_minutes - current_cluster_events[0].time_minutes
            if cluster_duration >= self.min_event_duration:
                cluster = self._create_event_cluster(current_cluster_events, session_date)
                clusters.append(cluster)
        
        return clusters
    
    def _create_event_cluster(self, events: List[PMEvent], session_date: str) -> PMEventCluster:
        """Create an event cluster from a group of events"""
        
        start_minute = events[0].time_minutes
        end_minute = events[-1].time_minutes
        duration = end_minute - start_minute
        
        # Analyze cluster characteristics
        action_types = [event.action_type for event in events]
        contexts = [event.context for event in events]
        
        primary_action = Counter(action_types).most_common(1)[0][0]
        
        # Find most descriptive context
        dominant_context = max(contexts, key=len) if contexts else ""
        
        # Calculate averages
        avg_volatility = np.mean([event.volatility_window for event in events])
        range_movement = abs(events[-1].range_position - events[0].range_position)
        htf_confluence_count = sum(1 for event in events if event.cross_tf_confluence)
        
        # Determine primary event type
        primary_event_type = "unknown"
        if any("sweep" in context.lower() for context in contexts):
            primary_event_type = "sweep"
        elif any("delivery" in context.lower() or "redelivery" in context.lower() for context in contexts):
            primary_event_type = "redelivery"
        elif any("touch" in context.lower() for context in contexts):
            primary_event_type = "touch"
        elif any("expansion" in context.lower() for context in contexts):
            primary_event_type = "expansion"
        
        return PMEventCluster(
            session_date=session_date,
            cluster_start_minute=start_minute,
            cluster_end_minute=end_minute,
            cluster_duration=duration,
            events=events,
            primary_event_type=primary_event_type,
            primary_action=primary_action,
            dominant_context=dominant_context,
            avg_volatility=avg_volatility,
            range_movement=range_movement,
            htf_confluence_count=htf_confluence_count
        )
    
    def _detect_directional_move_after_cluster(self, cluster: PMEventCluster, all_events: List[PMEvent]) -> Optional[DirectionalMove]:
        """Detect significant directional move following the event cluster"""
        
        # Find events in the directional move window
        move_start_time = cluster.cluster_end_minute + self.min_directional_move_window
        move_end_time = cluster.cluster_end_minute + self.directional_move_window
        
        move_events = [
            event for event in all_events
            if move_start_time <= event.time_minutes <= move_end_time
        ]
        
        if len(move_events) < 2:
            return None
        
        # Calculate move characteristics
        baseline_volatility = cluster.avg_volatility
        max_volatility = max(event.volatility_window for event in move_events)
        volatility_expansion = max_volatility / baseline_volatility if baseline_volatility > 0 else 1.0
        
        # Calculate price range change
        start_range_pos = cluster.events[-1].range_position
        move_range_positions = [event.range_position for event in move_events]
        max_range_change = max(abs(pos - start_range_pos) for pos in move_range_positions)
        
        # Determine move type from contexts
        move_contexts = [event.context.lower() for event in move_events]
        move_type = "consolidation"  # default
        
        if any("expansion" in context for context in move_contexts):
            move_type = "expansion"
        elif any("breakout" in context for context in move_contexts):
            move_type = "breakout"
        elif any("cascade" in context for context in move_contexts):
            move_type = "cascade"
        
        # Calculate significance score
        significance_score = self._calculate_move_significance(
            volatility_expansion, max_range_change, move_type, len(move_events)
        )
        
        # Only return if move is significant
        if significance_score >= 0.3:  # Threshold for significant move
            return DirectionalMove(
                move_start_minute=move_events[0].time_minutes,
                move_end_minute=move_events[-1].time_minutes,
                move_duration=move_events[-1].time_minutes - move_events[0].time_minutes,
                volatility_expansion=volatility_expansion,
                price_range_change=max_range_change,
                range_position_change=max_range_change,
                move_type=move_type,
                significance_score=significance_score
            )
        
        return None
    
    def _calculate_move_significance(self, volatility_expansion: float, range_change: float, 
                                   move_type: str, event_count: int) -> float:
        """Calculate significance score for a directional move"""
        
        # Base score from volatility expansion
        volatility_score = min(volatility_expansion / 2.0, 1.0)  # Cap at 1.0
        
        # Range change score (significant if >5% range movement)
        range_score = min(range_change / 0.05, 1.0)  # 5% = 1.0 score
        
        # Move type multiplier
        type_multipliers = {
            "expansion": 1.2,
            "breakout": 1.3,
            "cascade": 1.5,
            "consolidation": 0.8
        }
        type_multiplier = type_multipliers.get(move_type, 1.0)
        
        # Event count bonus (more events = higher significance)
        event_bonus = min(event_count / 10.0, 0.3)
        
        significance = (volatility_score * 0.4 + range_score * 0.6) * type_multiplier + event_bonus
        
        return min(significance, 1.0)
    
    def _calculate_pattern_strength(self, cluster: PMEventCluster, move: Optional[DirectionalMove]) -> float:
        """Calculate overall pattern strength"""
        
        # Base strength from cluster characteristics
        cluster_strength = min(cluster.cluster_duration / 3.0, 1.0)  # 3min = max
        volatility_strength = min(cluster.avg_volatility / 0.05, 1.0)  # 5% = max
        htf_bonus = min(cluster.htf_confluence_count / len(cluster.events), 1.0)
        
        base_strength = cluster_strength * 0.3 + volatility_strength * 0.4 + htf_bonus * 0.3
        
        # Move confirmation bonus
        if move:
            move_bonus = move.significance_score * 0.5
            return min(base_strength + move_bonus, 1.0)
        
        return base_strength
    
    def _classify_liquidity_archetype(self, cluster: PMEventCluster) -> str:
        """Classify the liquidity archetype of the event cluster"""
        
        contexts = [event.context.lower() for event in cluster.events]
        combined_context = " ".join(contexts)
        
        if "sweep" in combined_context and "session low" in combined_context:
            return "session_low_sweep"
        elif "sweep" in combined_context:
            return "liquidity_sweep" 
        elif "redelivery" in combined_context and "fpfvg" in combined_context:
            return "fpfvg_redelivery"
        elif "delivery" in combined_context and "fpfvg" in combined_context:
            return "fpfvg_delivery"
        elif "reversal" in combined_context:
            return "reversal_point"
        elif "expansion" in combined_context:
            return "expansion_phase"
        elif "consolidation" in combined_context:
            return "consolidation_phase"
        else:
            return "unclassified"
    
    def _classify_range_level(self, cluster: PMEventCluster) -> str:
        """Classify range level category"""
        
        avg_range_position = np.mean([event.range_position for event in cluster.events])
        
        if avg_range_position < 0.25:
            return "lower_range"
        elif avg_range_position < 0.50:
            return "mid_lower_range"
        elif avg_range_position < 0.75:
            return "mid_upper_range"
        else:
            return "upper_range"
    
    def generate_discovery_report(self, patterns: List[PMEventPattern]) -> Dict:
        """Generate comprehensive discovery report"""
        
        print("📋 Generating PM Event Discovery Report...")
        
        # Filter successful patterns (with confirmed directional moves)
        successful_patterns = [p for p in patterns if p.move_confirmed]
        
        report = {
            "discovery_metadata": {
                "total_sessions_scanned": len(self.pm_sessions),
                "total_patterns_found": len(patterns),
                "successful_patterns": len(successful_patterns),
                "success_rate": len(successful_patterns) / len(patterns) if patterns else 0,
                "target_window": f"{self.target_window_start}-{self.target_window_end} minutes",
                "scan_timestamp": datetime.now().isoformat()
            },
            
            "pattern_analysis": {
                "liquidity_archetypes": dict(Counter([p.liquidity_archetype for p in patterns])),
                "range_level_distribution": dict(Counter([p.range_level_category for p in patterns])),
                "event_type_distribution": dict(Counter([p.event_cluster.primary_event_type for p in patterns])),
                "action_type_distribution": dict(Counter([p.event_cluster.primary_action for p in patterns])),
                "avg_pattern_strength": np.mean([p.pattern_strength for p in patterns]) if patterns else 0
            },
            
            "directional_move_analysis": {
                "move_types": dict(Counter([p.directional_move.move_type for p in successful_patterns])),
                "avg_time_to_move": np.mean([p.time_to_move for p in successful_patterns if p.time_to_move]) if successful_patterns else 0,
                "avg_volatility_expansion": np.mean([p.directional_move.volatility_expansion for p in successful_patterns]) if successful_patterns else 0,
                "avg_range_change": np.mean([p.directional_move.range_position_change for p in successful_patterns]) if successful_patterns else 0,
                "avg_significance_score": np.mean([p.directional_move.significance_score for p in successful_patterns]) if successful_patterns else 0
            },
            
            "temporal_patterns": {
                "cluster_duration_avg": np.mean([p.event_cluster.cluster_duration for p in patterns]) if patterns else 0,
                "cluster_duration_distribution": self._analyze_duration_distribution([p.event_cluster.cluster_duration for p in patterns]),
                "htf_confluence_correlation": self._analyze_htf_confluence_correlation(patterns)
            },
            
            "detailed_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "session_date": p.session_date,
                    "cluster_start_minute": p.event_cluster.cluster_start_minute,
                    "cluster_duration": p.event_cluster.cluster_duration,
                    "primary_event_type": p.event_cluster.primary_event_type,
                    "dominant_context": p.event_cluster.dominant_context,
                    "liquidity_archetype": p.liquidity_archetype,
                    "range_level_category": p.range_level_category,
                    "pattern_strength": p.pattern_strength,
                    "move_confirmed": p.move_confirmed,
                    "time_to_move": p.time_to_move,
                    "directional_move": {
                        "move_type": p.directional_move.move_type if p.directional_move else None,
                        "volatility_expansion": p.directional_move.volatility_expansion if p.directional_move else None,
                        "range_change": p.directional_move.range_position_change if p.directional_move else None,
                        "significance_score": p.directional_move.significance_score if p.directional_move else None
                    } if p.directional_move else None
                }
                for p in patterns
            ]
        }
        
        return report
    
    def _analyze_duration_distribution(self, durations: List[float]) -> Dict:
        """Analyze duration distribution"""
        if not durations:
            return {}
        
        return {
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_duration": np.mean(durations),
            "median_duration": np.median(durations),
            "std_duration": np.std(durations)
        }
    
    def _analyze_htf_confluence_correlation(self, patterns: List[PMEventPattern]) -> Dict:
        """Analyze HTF confluence correlation with success"""
        
        if not patterns:
            return {}
        
        # Patterns with HTF confluence
        htf_patterns = [p for p in patterns if p.event_cluster.htf_confluence_count > 0]
        htf_successful = [p for p in htf_patterns if p.move_confirmed]
        
        # Patterns without HTF confluence  
        no_htf_patterns = [p for p in patterns if p.event_cluster.htf_confluence_count == 0]
        no_htf_successful = [p for p in no_htf_patterns if p.move_confirmed]
        
        return {
            "patterns_with_htf_confluence": len(htf_patterns),
            "htf_success_rate": len(htf_successful) / len(htf_patterns) if htf_patterns else 0,
            "patterns_without_htf_confluence": len(no_htf_patterns),
            "no_htf_success_rate": len(no_htf_successful) / len(no_htf_patterns) if no_htf_patterns else 0,
            "htf_confluence_advantage": (len(htf_successful) / len(htf_patterns) if htf_patterns else 0) - 
                                      (len(no_htf_successful) / len(no_htf_patterns) if no_htf_patterns else 0)
        }
    
    def save_discovery_results(self, output_path: str = None) -> str:
        """Save PM event discovery results"""
        
        if output_path is None:
            output_path = '/Users/jack/IRONFORGE/analysis/pm_event_discovery.json'
        
        # Scan all patterns
        patterns = self.scan_all_pm_sessions()
        
        # Generate report
        report = self.generate_discovery_report(patterns)
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 PM event discovery results saved to: {output_path}")
        
        # Save patterns separately for detailed analysis
        patterns_path = output_path.replace('.json', '_patterns.json')
        patterns_data = [
            {
                "pattern_id": p.pattern_id,
                "session_date": p.session_date,
                "event_cluster": {
                    "start_minute": p.event_cluster.cluster_start_minute,
                    "end_minute": p.event_cluster.cluster_end_minute,
                    "duration": p.event_cluster.cluster_duration,
                    "primary_event_type": p.event_cluster.primary_event_type,
                    "primary_action": p.event_cluster.primary_action,
                    "dominant_context": p.event_cluster.dominant_context,
                    "avg_volatility": p.event_cluster.avg_volatility,
                    "range_movement": p.event_cluster.range_movement,
                    "htf_confluence_count": p.event_cluster.htf_confluence_count,
                    "events": [
                        {
                            "time_minutes": e.time_minutes,
                            "timestamp": e.timestamp,
                            "action_type": e.action_type,
                            "context": e.context,
                            "price_level": e.price_level,
                            "range_position": e.range_position,
                            "volatility_window": e.volatility_window,
                            "cross_tf_confluence": e.cross_tf_confluence
                        }
                        for e in p.event_cluster.events
                    ]
                },
                "directional_move": {
                    "move_start_minute": p.directional_move.move_start_minute,
                    "move_end_minute": p.directional_move.move_end_minute,
                    "move_duration": p.directional_move.move_duration,
                    "volatility_expansion": p.directional_move.volatility_expansion,
                    "range_position_change": p.directional_move.range_position_change,
                    "move_type": p.directional_move.move_type,
                    "significance_score": p.directional_move.significance_score
                } if p.directional_move else None,
                "pattern_metadata": {
                    "move_confirmed": p.move_confirmed,
                    "time_to_move": p.time_to_move,
                    "pattern_strength": p.pattern_strength,
                    "liquidity_archetype": p.liquidity_archetype,
                    "range_level_category": p.range_level_category
                }
            }
            for p in patterns
        ]
        
        with open(patterns_path, 'w') as f:
            json.dump(patterns_data, f, indent=2, default=str)
        
        print(f"📊 Detailed patterns saved to: {patterns_path}")
        
        return output_path

if __name__ == "__main__":
    print("🕒 IRONFORGE PM Event Scanner")
    print("=" * 60)
    
    scanner = PMEventScanner()
    output_file = scanner.save_discovery_results()
    
    print(f"\n✅ PM event discovery complete!")
    print(f"📊 Results saved to: {output_file}")