#!/usr/bin/env python3
"""
IRONFORGE Timeframe Lattice Mapper
==================================

Multi-dimensional lattice mapping system for visualizing market phenomena
across timeframes and cycle positions. Creates a structured coordinate system
where events can be positioned based on their temporal and structural context.

Features:
- Vertical axis: Timeframes (monthly → 1m)
- Horizontal axis: Relative cycle position (0% → 100%)
- Node properties: Event type, significance, structural role
- Connection mapping: Lead/lag relationships, causality chains
- Hot zone identification: High-frequency event clusters

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

try:
    from .broad_spectrum_archaeology import ArchaeologicalEvent, TimeframeType, SessionPhase
except ImportError:
    # Fallback for direct execution
    from broad_spectrum_archaeology import ArchaeologicalEvent, TimeframeType, SessionPhase


@dataclass
class LatticeCoordinate:
    """Coordinates in the timeframe × cycle-position lattice"""
    timeframe_level: int      # 0=monthly, 1=weekly, ..., 7=1m
    cycle_position: float     # 0.0-1.0 relative position in cycle
    absolute_timeframe: TimeframeType
    absolute_position: float


@dataclass
class LatticeNode:
    """Node in the lattice representing an event or cluster"""
    node_id: str
    coordinate: LatticeCoordinate
    
    # Event properties
    events: List[ArchaeologicalEvent]
    event_count: int
    average_significance: float
    dominant_event_type: str
    dominant_archetype: str
    
    # Visual properties
    color: str               # Based on event family
    size: float             # Based on significance/count
    shape: str              # Based on structural role
    opacity: float          # Based on confidence
    
    # Network properties
    incoming_connections: List[str]  # Node IDs that lead to this node
    outgoing_connections: List[str]  # Node IDs this node leads to
    connection_strengths: Dict[str, float]
    
    # Clustering properties
    cluster_id: Optional[str]
    hot_zone_member: bool
    recurrence_rate: float


@dataclass
class LatticeConnection:
    """Connection between lattice nodes"""
    connection_id: str
    source_node_id: str
    target_node_id: str
    
    # Connection properties
    connection_type: str     # lead_lag, causality, resonance, inheritance
    strength: float         # 0.0-1.0
    temporal_distance: float  # Time difference
    structural_distance: float  # Lattice distance
    
    # Evidence
    supporting_events: List[str]
    confidence_score: float
    
    # Visual properties
    line_style: str         # solid, dashed, dotted
    line_width: float
    arrow_type: str         # single, double, bidirectional


@dataclass
class HotZone:
    """High-frequency event cluster in the lattice"""
    zone_id: str
    zone_type: str          # temporal, spatial, hybrid
    
    # Spatial definition
    timeframe_range: Tuple[int, int]
    position_range: Tuple[float, float]
    center_coordinate: LatticeCoordinate
    
    # Statistical properties
    event_density: float
    average_significance: float
    dominant_pattern: str
    recurrence_frequency: float
    
    # Member nodes
    member_nodes: List[str]
    total_events: int
    
    # Temporal properties
    active_sessions: List[str]
    temporal_stability: float


@dataclass
class LatticeDataset:
    """Complete lattice dataset for visualization and analysis"""
    dataset_id: str
    creation_timestamp: str
    
    # Lattice structure
    nodes: Dict[str, LatticeNode]
    connections: Dict[str, LatticeConnection]
    hot_zones: Dict[str, HotZone]
    
    # Metadata
    timeframe_levels: Dict[int, TimeframeType]
    total_events_mapped: int
    sessions_covered: List[str]
    analysis_parameters: Dict[str, Any]
    
    # Statistics
    node_statistics: Dict[str, Any]
    connection_statistics: Dict[str, Any]
    hot_zone_statistics: Dict[str, Any]


class TimeframeLatticeMapper:
    """
    Maps archaeological events onto a structured timeframe × cycle-position lattice
    """
    
    def __init__(self, 
                 grid_resolution: int = 100,
                 min_node_events: int = 1,
                 hot_zone_threshold: float = 0.7):
        """
        Initialize the lattice mapper
        
        Args:
            grid_resolution: Number of position buckets (1-100)
            min_node_events: Minimum events required to create a node
            hot_zone_threshold: Minimum density for hot zone classification
        """
        self.logger = logging.getLogger('lattice_mapper')
        
        # Configuration
        self.grid_resolution = grid_resolution
        self.min_node_events = min_node_events
        self.hot_zone_threshold = hot_zone_threshold
        
        # Timeframe level mapping (0=highest timeframe, 7=lowest)
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
        
        # Visual styling
        self.event_type_colors = {
            'fvg_family': '#2E8B57',      # SeaGreen
            'sweep_family': '#DC143C',     # Crimson
            'expansion_family': '#4169E1', # RoyalBlue
            'consolidation_family': '#FF8C00', # DarkOrange
            'miscellaneous': '#9370DB'     # MediumPurple
        }
        
        self.structural_role_shapes = {
            'terminal_sweep': 'diamond',
            'breakout': 'triangle',
            'accumulation': 'circle',
            'minor_signal': 'square'
        }
        
        # Storage
        self.lattice_nodes: Dict[str, LatticeNode] = {}
        self.lattice_connections: Dict[str, LatticeConnection] = {}
        self.hot_zones: Dict[str, HotZone] = {}
        
        print(f"🗺️  Timeframe Lattice Mapper initialized")
        print(f"  Grid resolution: {grid_resolution}")
        print(f"  Timeframe levels: {len(self.timeframe_levels)}")
        print(f"  Hot zone threshold: {hot_zone_threshold}")
    
    def map_events_to_lattice(self, events: List[ArchaeologicalEvent]) -> LatticeDataset:
        """
        Map archaeological events to the timeframe lattice
        
        Args:
            events: List of archaeological events to map
            
        Returns:
            Complete lattice dataset with nodes, connections, and hot zones
        """
        print(f"\n🗺️  Mapping {len(events)} events to timeframe lattice...")
        
        # Clear previous mapping
        self.lattice_nodes.clear()
        self.lattice_connections.clear()
        self.hot_zones.clear()
        
        # Step 1: Create lattice coordinates for events
        print("  📍 Creating lattice coordinates...")
        event_coordinates = self._create_event_coordinates(events)
        
        # Step 2: Aggregate events into lattice nodes
        print("  🔗 Aggregating events into lattice nodes...")
        self._create_lattice_nodes(events, event_coordinates)
        
        # Step 3: Identify connections between nodes
        print("  🕸️  Identifying connections between nodes...")
        self._identify_lattice_connections(events)
        
        # Step 4: Detect hot zones
        print("  🔥 Detecting hot zones...")
        self._detect_hot_zones()
        
        # Step 5: Create complete dataset
        print("  📊 Creating lattice dataset...")
        dataset = self._create_lattice_dataset(events)
        
        print(f"✅ Lattice mapping complete!")
        print(f"  Nodes created: {len(self.lattice_nodes)}")
        print(f"  Connections identified: {len(self.lattice_connections)}")
        print(f"  Hot zones detected: {len(self.hot_zones)}")
        
        return dataset
    
    def _create_event_coordinates(self, events: List[ArchaeologicalEvent]) -> Dict[str, LatticeCoordinate]:
        """Create lattice coordinates for each event"""
        
        coordinates = {}
        
        for event in events:
            # Get timeframe level
            timeframe_level = self.timeframe_levels[event.timeframe]
            
            # Quantize cycle position to grid
            position_bucket = int(event.relative_cycle_position * self.grid_resolution) / self.grid_resolution
            
            coordinate = LatticeCoordinate(
                timeframe_level=timeframe_level,
                cycle_position=position_bucket,
                absolute_timeframe=event.timeframe,
                absolute_position=event.relative_cycle_position
            )
            
            coordinates[event.event_id] = coordinate
        
        return coordinates
    
    def _create_lattice_nodes(self, events: List[ArchaeologicalEvent], coordinates: Dict[str, LatticeCoordinate]):
        """Aggregate events into lattice nodes based on coordinates"""
        
        # Group events by coordinate
        coordinate_groups = defaultdict(list)
        
        for event in events:
            coord = coordinates[event.event_id]
            coord_key = f"{coord.timeframe_level}_{coord.cycle_position:.3f}"
            coordinate_groups[coord_key].append(event)
        
        # Create nodes for each coordinate group
        for coord_key, group_events in coordinate_groups.items():
            if len(group_events) >= self.min_node_events:
                
                # Get representative coordinate
                representative_coord = coordinates[group_events[0].event_id]
                
                # Calculate node properties
                node_id = f"node_{coord_key}"
                event_count = len(group_events)
                avg_significance = np.mean([e.significance_score for e in group_events])
                
                # Determine dominant characteristics
                event_types = [e.event_type.value for e in group_events]
                pattern_families = [e.pattern_family for e in group_events]
                archetypes = [e.liquidity_archetype.value for e in group_events]
                structural_roles = [e.structural_role for e in group_events]
                
                dominant_event_type = Counter(event_types).most_common(1)[0][0]
                dominant_pattern_family = Counter(pattern_families).most_common(1)[0][0]
                dominant_archetype = Counter(archetypes).most_common(1)[0][0]
                dominant_role = Counter(structural_roles).most_common(1)[0][0]
                
                # Calculate visual properties
                color = self.event_type_colors.get(dominant_pattern_family, '#9370DB')
                size = min(10 + event_count * 2, 50)  # Scale size with event count
                shape = self.structural_role_shapes.get(dominant_role, 'circle')
                opacity = min(0.3 + avg_significance * 0.7, 1.0)
                
                # Calculate recurrence rate
                sessions = set(e.session_name for e in group_events)
                recurrence_rate = len(sessions) / len(set(e.session_name for e in events))
                
                # Create lattice node
                lattice_node = LatticeNode(
                    node_id=node_id,
                    coordinate=representative_coord,
                    events=group_events,
                    event_count=event_count,
                    average_significance=avg_significance,
                    dominant_event_type=dominant_event_type,
                    dominant_archetype=dominant_archetype,
                    color=color,
                    size=size,
                    shape=shape,
                    opacity=opacity,
                    incoming_connections=[],
                    outgoing_connections=[],
                    connection_strengths={},
                    cluster_id=None,
                    hot_zone_member=False,
                    recurrence_rate=recurrence_rate
                )
                
                self.lattice_nodes[node_id] = lattice_node
    
    def _identify_lattice_connections(self, events: List[ArchaeologicalEvent]):
        """Identify connections between lattice nodes"""
        
        # Group events by session for temporal analysis
        session_events = defaultdict(list)
        for event in events:
            session_events[event.session_name].append(event)
        
        connection_count = 0
        
        # Analyze each session for temporal relationships
        for session_name, session_event_list in session_events.items():
            
            # Sort events by session minute
            sorted_events = sorted(session_event_list, key=lambda e: e.session_minute)
            
            # Look for lead-lag relationships
            for i, event1 in enumerate(sorted_events):
                for j, event2 in enumerate(sorted_events[i+1:], i+1):
                    
                    # Find nodes for these events
                    node1_id = self._find_node_for_event(event1)
                    node2_id = self._find_node_for_event(event2)
                    
                    if node1_id and node2_id and node1_id != node2_id:
                        
                        # Calculate connection properties
                        connection_strength = self._calculate_connection_strength(event1, event2)
                        
                        if connection_strength > 0.3:  # Threshold for significant connections
                            
                            connection_id = f"conn_{node1_id}_{node2_id}_{connection_count}"
                            connection_count += 1
                            
                            # Determine connection type
                            connection_type = self._determine_connection_type(event1, event2)
                            
                            # Calculate distances
                            temporal_distance = abs(event2.session_minute - event1.session_minute)
                            structural_distance = self._calculate_structural_distance(event1, event2)
                            
                            # Create connection
                            connection = LatticeConnection(
                                connection_id=connection_id,
                                source_node_id=node1_id,
                                target_node_id=node2_id,
                                connection_type=connection_type,
                                strength=connection_strength,
                                temporal_distance=temporal_distance,
                                structural_distance=structural_distance,
                                supporting_events=[event1.event_id, event2.event_id],
                                confidence_score=min(event1.confidence_score, event2.confidence_score),
                                line_style=self._get_line_style(connection_type),
                                line_width=max(1.0, connection_strength * 3.0),
                                arrow_type=self._get_arrow_type(connection_type)
                            )
                            
                            self.lattice_connections[connection_id] = connection
                            
                            # Update node connections
                            if node1_id in self.lattice_nodes:
                                self.lattice_nodes[node1_id].outgoing_connections.append(node2_id)
                                self.lattice_nodes[node1_id].connection_strengths[node2_id] = connection_strength
                            
                            if node2_id in self.lattice_nodes:
                                self.lattice_nodes[node2_id].incoming_connections.append(node1_id)
        
        # Look for cross-session inheritance patterns
        self._identify_cross_session_connections(events)
    
    def _find_node_for_event(self, event: ArchaeologicalEvent) -> Optional[str]:
        """Find the lattice node that contains this event"""
        
        for node_id, node in self.lattice_nodes.items():
            if event in node.events:
                return node_id
        
        return None
    
    def _calculate_connection_strength(self, event1: ArchaeologicalEvent, event2: ArchaeologicalEvent) -> float:
        """Calculate connection strength between two events"""
        
        # Base strength from significance scores
        base_strength = (event1.significance_score + event2.significance_score) / 2
        
        # Boost for similar event types
        if event1.event_type == event2.event_type:
            base_strength *= 1.2
        
        # Boost for similar archetypes
        if event1.liquidity_archetype == event2.liquidity_archetype:
            base_strength *= 1.1
        
        # Boost for HTF confluence
        if event1.htf_confluence.value in ['confirmed', 'partial'] and event2.htf_confluence.value in ['confirmed', 'partial']:
            base_strength *= 1.15
        
        # Adjust for temporal distance (closer events have stronger connections)
        temporal_distance = abs(event2.session_minute - event1.session_minute)
        temporal_weight = max(0.1, 1.0 - temporal_distance / 180.0)  # 180 minutes = full session
        
        return min(1.0, base_strength * temporal_weight)
    
    def _determine_connection_type(self, event1: ArchaeologicalEvent, event2: ArchaeologicalEvent) -> str:
        """Determine the type of connection between two events"""
        
        # Temporal precedence
        if event1.session_minute < event2.session_minute:
            if event2.session_minute - event1.session_minute < 5:
                return "immediate_causality"
            elif event2.session_minute - event1.session_minute < 30:
                return "short_term_lead"
            else:
                return "long_term_lead"
        
        # Same session phase
        elif event1.session_phase == event2.session_phase:
            return "phase_resonance"
        
        # Cross-session
        elif event1.session_name != event2.session_name:
            return "cross_session_inheritance"
        
        else:
            return "structural_resonance"
    
    def _calculate_structural_distance(self, event1: ArchaeologicalEvent, event2: ArchaeologicalEvent) -> float:
        """Calculate structural distance between events in lattice space"""
        
        level1 = self.timeframe_levels[event1.timeframe]
        level2 = self.timeframe_levels[event2.timeframe]
        
        pos1 = event1.relative_cycle_position
        pos2 = event2.relative_cycle_position
        
        # Euclidean distance in lattice space
        level_distance = abs(level1 - level2) / len(self.timeframe_levels)
        position_distance = abs(pos1 - pos2)
        
        return math.sqrt(level_distance**2 + position_distance**2)
    
    def _get_line_style(self, connection_type: str) -> str:
        """Get line style for connection type"""
        
        style_map = {
            "immediate_causality": "solid",
            "short_term_lead": "dashed",
            "long_term_lead": "dotted",
            "phase_resonance": "solid",
            "cross_session_inheritance": "dashdot",
            "structural_resonance": "dashed"
        }
        
        return style_map.get(connection_type, "solid")
    
    def _get_arrow_type(self, connection_type: str) -> str:
        """Get arrow type for connection"""
        
        if "lead" in connection_type or "causality" in connection_type:
            return "single"
        elif "resonance" in connection_type:
            return "bidirectional"
        else:
            return "single"
    
    def _identify_cross_session_connections(self, events: List[ArchaeologicalEvent]):
        """Identify connections that span across sessions"""
        
        # Group events by pattern signature
        pattern_groups = defaultdict(list)
        
        for event in events:
            pattern_signature = f"{event.event_type.value}_{event.range_level.value}_{event.absolute_time_signature}"
            pattern_groups[pattern_signature].append(event)
        
        # Look for patterns that repeat across sessions
        for pattern_signature, pattern_events in pattern_groups.items():
            sessions = set(e.session_name for e in pattern_events)
            
            if len(sessions) >= 2:  # Pattern appears in multiple sessions
                
                # Create connections between similar events in different sessions
                for i, event1 in enumerate(pattern_events):
                    for event2 in pattern_events[i+1:]:
                        if event1.session_name != event2.session_name:
                            
                            node1_id = self._find_node_for_event(event1)
                            node2_id = self._find_node_for_event(event2)
                            
                            if node1_id and node2_id and node1_id != node2_id:
                                
                                connection_strength = self._calculate_cross_session_strength(event1, event2)
                                
                                if connection_strength > 0.4:
                                    
                                    connection_id = f"cross_sess_{node1_id}_{node2_id}"
                                    
                                    connection = LatticeConnection(
                                        connection_id=connection_id,
                                        source_node_id=node1_id,
                                        target_node_id=node2_id,
                                        connection_type="cross_session_inheritance",
                                        strength=connection_strength,
                                        temporal_distance=self._calculate_session_distance(event1, event2),
                                        structural_distance=self._calculate_structural_distance(event1, event2),
                                        supporting_events=[event1.event_id, event2.event_id],
                                        confidence_score=(event1.confidence_score + event2.confidence_score) / 2,
                                        line_style="dashdot",
                                        line_width=max(1.0, connection_strength * 2.0),
                                        arrow_type="bidirectional"
                                    )
                                    
                                    self.lattice_connections[connection_id] = connection
    
    def _calculate_cross_session_strength(self, event1: ArchaeologicalEvent, event2: ArchaeologicalEvent) -> float:
        """Calculate strength of cross-session connection"""
        
        # Base strength from pattern similarity
        base_strength = 0.5
        
        # Boost for identical event types
        if event1.event_type == event2.event_type:
            base_strength += 0.2
        
        # Boost for similar range levels
        if event1.range_level == event2.range_level:
            base_strength += 0.15
        
        # Boost for similar time signatures
        if event1.absolute_time_signature == event2.absolute_time_signature:
            base_strength += 0.25
        
        # Boost for high individual significance
        significance_boost = (event1.significance_score + event2.significance_score) / 4
        
        return min(1.0, base_strength + significance_boost)
    
    def _calculate_session_distance(self, event1: ArchaeologicalEvent, event2: ArchaeologicalEvent) -> float:
        """Calculate temporal distance between sessions"""
        
        try:
            date1 = datetime.strptime(event1.session_date, '%Y-%m-%d')
            date2 = datetime.strptime(event2.session_date, '%Y-%m-%d')
            return abs((date2 - date1).days)
        except:
            return 1.0  # Default distance
    
    def _detect_hot_zones(self):
        """Detect high-density zones in the lattice"""
        
        # Calculate density grid
        density_grid = defaultdict(list)
        
        for node_id, node in self.lattice_nodes.items():
            grid_key = f"{node.coordinate.timeframe_level}_{int(node.coordinate.cycle_position * 10)}"
            density_grid[grid_key].append(node)
        
        # Identify hot zones
        zone_count = 0
        
        for grid_key, nodes in density_grid.items():
            if len(nodes) >= 3:  # Minimum nodes for hot zone
                
                total_events = sum(node.event_count for node in nodes)
                avg_significance = np.mean([node.average_significance for node in nodes])
                
                # Calculate density score
                density_score = total_events * avg_significance / len(nodes)
                
                if density_score >= self.hot_zone_threshold:
                    
                    zone_id = f"hot_zone_{zone_count}"
                    zone_count += 1
                    
                    # Determine zone boundaries
                    timeframe_levels = [node.coordinate.timeframe_level for node in nodes]
                    positions = [node.coordinate.cycle_position for node in nodes]
                    
                    timeframe_range = (min(timeframe_levels), max(timeframe_levels))
                    position_range = (min(positions), max(positions))
                    
                    # Calculate center coordinate
                    center_level = int(np.mean(timeframe_levels))
                    center_position = np.mean(positions)
                    
                    center_timeframe = None
                    for tf, level in self.timeframe_levels.items():
                        if level == center_level:
                            center_timeframe = tf
                            break
                    
                    if center_timeframe:
                        center_coordinate = LatticeCoordinate(
                            timeframe_level=center_level,
                            cycle_position=center_position,
                            absolute_timeframe=center_timeframe,
                            absolute_position=center_position
                        )
                        
                        # Determine dominant pattern
                        all_types = []
                        for node in nodes:
                            all_types.extend([e.event_type.value for e in node.events])
                        
                        dominant_pattern = Counter(all_types).most_common(1)[0][0] if all_types else "mixed"
                        
                        # Calculate temporal stability
                        all_sessions = set()
                        for node in nodes:
                            all_sessions.update(e.session_name for e in node.events)
                        
                        temporal_stability = len(all_sessions) / max(1, total_events)
                        
                        # Create hot zone
                        hot_zone = HotZone(
                            zone_id=zone_id,
                            zone_type="hybrid",
                            timeframe_range=timeframe_range,
                            position_range=position_range,
                            center_coordinate=center_coordinate,
                            event_density=density_score,
                            average_significance=avg_significance,
                            dominant_pattern=dominant_pattern,
                            recurrence_frequency=len(all_sessions) / 20,  # Assuming ~20 total sessions
                            member_nodes=[node.node_id for node in nodes],
                            total_events=total_events,
                            active_sessions=list(all_sessions),
                            temporal_stability=temporal_stability
                        )
                        
                        self.hot_zones[zone_id] = hot_zone
                        
                        # Mark nodes as hot zone members
                        for node in nodes:
                            node.hot_zone_member = True
                            node.cluster_id = zone_id
    
    def _create_lattice_dataset(self, events: List[ArchaeologicalEvent]) -> LatticeDataset:
        """Create complete lattice dataset"""
        
        # Calculate statistics
        node_stats = {
            'total_nodes': len(self.lattice_nodes),
            'average_events_per_node': np.mean([node.event_count for node in self.lattice_nodes.values()]) if self.lattice_nodes else 0,
            'average_significance': np.mean([node.average_significance for node in self.lattice_nodes.values()]) if self.lattice_nodes else 0,
            'timeframe_distribution': Counter(node.coordinate.absolute_timeframe.value for node in self.lattice_nodes.values()),
            'hot_zone_nodes': sum(1 for node in self.lattice_nodes.values() if node.hot_zone_member)
        }
        
        connection_stats = {
            'total_connections': len(self.lattice_connections),
            'average_strength': np.mean([conn.strength for conn in self.lattice_connections.values()]) if self.lattice_connections else 0,
            'connection_types': Counter(conn.connection_type for conn in self.lattice_connections.values()),
            'strong_connections': sum(1 for conn in self.lattice_connections.values() if conn.strength > 0.7)
        }
        
        hot_zone_stats = {
            'total_hot_zones': len(self.hot_zones),
            'average_events_per_zone': np.mean([zone.total_events for zone in self.hot_zones.values()]) if self.hot_zones else 0,
            'average_density': np.mean([zone.event_density for zone in self.hot_zones.values()]) if self.hot_zones else 0,
            'dominant_patterns': Counter(zone.dominant_pattern for zone in self.hot_zones.values())
        }
        
        # Get session coverage
        sessions_covered = list(set(event.session_name for event in events))
        
        # Create analysis parameters
        analysis_params = {
            'grid_resolution': self.grid_resolution,
            'min_node_events': self.min_node_events,
            'hot_zone_threshold': self.hot_zone_threshold,
            'timeframe_levels': {level: tf.value for tf, level in self.timeframe_levels.items()}
        }
        
        # Create reverse mapping for timeframe levels
        reverse_timeframe_levels = {level: tf for tf, level in self.timeframe_levels.items()}
        
        return LatticeDataset(
            dataset_id=f"lattice_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            creation_timestamp=datetime.now().isoformat(),
            nodes=self.lattice_nodes.copy(),
            connections=self.lattice_connections.copy(),
            hot_zones=self.hot_zones.copy(),
            timeframe_levels=reverse_timeframe_levels,
            total_events_mapped=len(events),
            sessions_covered=sessions_covered,
            analysis_parameters=analysis_params,
            node_statistics=node_stats,
            connection_statistics=connection_stats,
            hot_zone_statistics=hot_zone_stats
        )
    
    def export_lattice_dataset(self, dataset: LatticeDataset, output_path: str = "lattice_dataset.json") -> str:
        """Export lattice dataset to JSON file"""
        
        # Convert dataset to serializable format
        export_data = {
            'metadata': {
                'dataset_id': dataset.dataset_id,
                'creation_timestamp': dataset.creation_timestamp,
                'total_events_mapped': dataset.total_events_mapped,
                'sessions_covered': dataset.sessions_covered,
                'analysis_parameters': dataset.analysis_parameters
            },
            'nodes': {},
            'connections': {},
            'hot_zones': {},
            'statistics': {
                'nodes': dataset.node_statistics,
                'connections': dataset.connection_statistics,
                'hot_zones': dataset.hot_zone_statistics
            },
            'timeframe_levels': {str(level): tf.value for level, tf in dataset.timeframe_levels.items()}
        }
        
        # Export nodes
        for node_id, node in dataset.nodes.items():
            export_data['nodes'][node_id] = {
                'node_id': node.node_id,
                'coordinate': {
                    'timeframe_level': node.coordinate.timeframe_level,
                    'cycle_position': node.coordinate.cycle_position,
                    'absolute_timeframe': node.coordinate.absolute_timeframe.value,
                    'absolute_position': node.coordinate.absolute_position
                },
                'event_count': node.event_count,
                'average_significance': node.average_significance,
                'dominant_event_type': node.dominant_event_type,
                'dominant_archetype': node.dominant_archetype,
                'visual_properties': {
                    'color': node.color,
                    'size': node.size,
                    'shape': node.shape,
                    'opacity': node.opacity
                },
                'network_properties': {
                    'incoming_connections': node.incoming_connections,
                    'outgoing_connections': node.outgoing_connections,
                    'connection_strengths': node.connection_strengths
                },
                'clustering_properties': {
                    'cluster_id': node.cluster_id,
                    'hot_zone_member': node.hot_zone_member,
                    'recurrence_rate': node.recurrence_rate
                },
                'event_ids': [event.event_id for event in node.events]
            }
        
        # Export connections
        for conn_id, conn in dataset.connections.items():
            export_data['connections'][conn_id] = {
                'connection_id': conn.connection_id,
                'source_node_id': conn.source_node_id,
                'target_node_id': conn.target_node_id,
                'connection_type': conn.connection_type,
                'strength': conn.strength,
                'temporal_distance': conn.temporal_distance,
                'structural_distance': conn.structural_distance,
                'supporting_events': conn.supporting_events,
                'confidence_score': conn.confidence_score,
                'visual_properties': {
                    'line_style': conn.line_style,
                    'line_width': conn.line_width,
                    'arrow_type': conn.arrow_type
                }
            }
        
        # Export hot zones
        for zone_id, zone in dataset.hot_zones.items():
            export_data['hot_zones'][zone_id] = {
                'zone_id': zone.zone_id,
                'zone_type': zone.zone_type,
                'spatial_definition': {
                    'timeframe_range': zone.timeframe_range,
                    'position_range': zone.position_range,
                    'center_coordinate': {
                        'timeframe_level': zone.center_coordinate.timeframe_level,
                        'cycle_position': zone.center_coordinate.cycle_position,
                        'absolute_timeframe': zone.center_coordinate.absolute_timeframe.value
                    }
                },
                'statistical_properties': {
                    'event_density': zone.event_density,
                    'average_significance': zone.average_significance,
                    'dominant_pattern': zone.dominant_pattern,
                    'recurrence_frequency': zone.recurrence_frequency
                },
                'member_nodes': zone.member_nodes,
                'total_events': zone.total_events,
                'temporal_properties': {
                    'active_sessions': zone.active_sessions,
                    'temporal_stability': zone.temporal_stability
                }
            }
        
        # Write to file
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"🗺️  Lattice dataset exported to {output_file}")
        return str(output_file)


if __name__ == "__main__":
    # Test the lattice mapper
    print("🗺️  Testing Timeframe Lattice Mapper")
    print("=" * 50)
    
    # This would normally be called with actual archaeological events
    print("✅ Lattice mapper initialized and ready for use")
    print("   Use map_events_to_lattice() with ArchaeologicalEvent list to create lattice")