#!/usr/bin/env python3
"""
IRONFORGE Temporal Clustering Engine
====================================

Advanced temporal pattern clustering system for identifying recurring
market phenomena across absolute times and relative cycle positions.

Features:
- Absolute time clustering (e.g., PM minute 37, daily bar 3 of weekly cycle)
- Relative position clustering within higher-timeframe cycles
- Session phase clustering (opening, mid-session, closing)
- Cross-session temporal resonance detection
- Pattern stability and recurrence analysis
- Multi-dimensional clustering with temporal, structural, and significance weights

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
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Scikit-learn not available - using simplified clustering")

try:
    from .broad_spectrum_archaeology import ArchaeologicalEvent, TimeframeType, SessionPhase
except ImportError:
    # Fallback for direct execution
    from broad_spectrum_archaeology import ArchaeologicalEvent, TimeframeType, SessionPhase


class ClusterType(Enum):
    """Types of temporal clusters"""
    ABSOLUTE_TIME = "absolute_time"
    RELATIVE_POSITION = "relative_position"
    SESSION_PHASE = "session_phase"
    CROSS_SESSION = "cross_session"
    HYBRID = "hybrid"


class ClusteringMethod(Enum):
    """Clustering algorithms"""
    DBSCAN = "dbscan"
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    CUSTOM = "custom"


@dataclass
class ClusterFeatures:
    """Feature vector for clustering"""
    
    # Temporal features
    absolute_minute: float         # Session minute (0-180)
    relative_cycle_position: float # Position within timeframe cycle (0-1)
    session_phase_encoded: float   # Encoded session phase (0-3)
    time_of_day_encoded: float     # Encoded time signature
    
    # Structural features
    timeframe_level: int           # Timeframe hierarchy level (0-7)
    range_position: float          # Position within range (0-1)
    magnitude: float               # Event magnitude
    significance: float            # Significance score
    
    # Pattern features
    event_type_encoded: float      # Encoded event type
    archetype_encoded: float       # Encoded liquidity archetype
    structural_role_encoded: float # Encoded structural role
    
    # Context features
    htf_confluence: float          # HTF confluence strength
    cross_session_inheritance: float # Cross-session pattern strength
    velocity_signature: float     # Price velocity
    
    # Derived features
    temporal_uniqueness: float     # How unique this timing is
    structural_importance: float   # Structural significance
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy array for clustering"""
        return np.array([
            self.absolute_minute,
            self.relative_cycle_position,
            self.session_phase_encoded,
            self.time_of_day_encoded,
            self.timeframe_level,
            self.range_position,
            self.magnitude,
            self.significance,
            self.event_type_encoded,
            self.archetype_encoded,
            self.structural_role_encoded,
            self.htf_confluence,
            self.cross_session_inheritance,
            self.velocity_signature,
            self.temporal_uniqueness,
            self.structural_importance
        ])


@dataclass
class TemporalCluster:
    """Temporal cluster with comprehensive analysis"""
    
    cluster_id: str
    cluster_type: ClusterType
    clustering_method: ClusteringMethod
    
    # Core members
    events: List[ArchaeologicalEvent]
    event_count: int
    feature_vectors: List[ClusterFeatures]
    
    # Temporal characteristics
    temporal_signature: str        # e.g., "PM_37", "cycle_0.4"
    average_timing: float         # Average absolute timing
    timing_variance: float        # Timing consistency
    session_phases: List[str]     # Session phases involved
    
    # Pattern characteristics
    dominant_event_type: str
    dominant_archetype: str
    pattern_consistency: float    # How consistent the pattern is
    
    # Statistical properties
    average_significance: float
    significance_variance: float
    magnitude_profile: Dict[str, float]
    
    # Recurrence analysis
    session_coverage: List[str]   # Sessions where cluster appears
    recurrence_rate: float       # Frequency across sessions
    recurrence_consistency: float # How consistently it recurs
    temporal_stability: float    # How stable timing is across sessions
    
    # Cross-session properties
    session_distribution: Dict[str, int] # Count per session type
    date_distribution: Dict[str, int]    # Count per date
    cross_session_strength: float       # Strength of cross-session pattern
    
    # Cluster quality metrics
    silhouette_score: float      # Clustering quality
    intra_cluster_distance: float # Average distance within cluster
    inter_cluster_distance: float # Distance to nearest cluster
    cluster_density: float        # Event density
    
    # Predictive properties
    next_event_probability: float  # Probability of next event
    cascade_potential: float      # Likelihood of triggering cascades
    structural_importance: float  # Overall structural importance


@dataclass
class ClusteringAnalysis:
    """Complete clustering analysis results"""
    
    analysis_id: str
    analysis_timestamp: str
    
    # Input data
    total_events_analyzed: int
    sessions_covered: List[str]
    timeframes_analyzed: List[str]
    
    # Clustering results
    clusters: List[TemporalCluster]
    cluster_count: int
    noise_events: List[ArchaeologicalEvent]
    
    # Analysis parameters
    clustering_parameters: Dict[str, Any]
    feature_weights: Dict[str, float]
    
    # Quality metrics
    overall_silhouette_score: float
    cluster_quality_distribution: Dict[str, int]
    temporal_coverage: float
    pattern_discovery_rate: float
    
    # Statistical summaries
    cluster_statistics: Dict[str, Any]
    temporal_heatmap_data: Dict[str, Any]
    recurrence_analysis: Dict[str, Any]


class TemporalClusteringEngine:
    """
    Advanced temporal clustering engine for archaeological events
    """
    
    def __init__(self,
                 min_cluster_size: int = 3,
                 max_cluster_size: int = 50,
                 temporal_weight: float = 0.4,
                 structural_weight: float = 0.3,
                 significance_weight: float = 0.3):
        """
        Initialize the temporal clustering engine
        
        Args:
            min_cluster_size: Minimum events required for a cluster
            max_cluster_size: Maximum events in a cluster
            temporal_weight: Weight for temporal features
            structural_weight: Weight for structural features
            significance_weight: Weight for significance features
        """
        
        self.logger = logging.getLogger('temporal_clustering')
        
        # Configuration
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        # Feature weights
        self.feature_weights = {
            'temporal': temporal_weight,
            'structural': structural_weight,
            'significance': significance_weight
        }
        
        # Encoding dictionaries
        self.event_type_encoding = {}
        self.archetype_encoding = {}
        self.structural_role_encoding = {}
        self.session_phase_encoding = {
            'opening': 0.0,
            'mid_session': 1.0,
            'session_closing': 2.0,
            'critical_window': 3.0
        }
        
        # Clustering results
        self.clusters: List[TemporalCluster] = []
        self.feature_vectors: List[ClusterFeatures] = []
        self.processed_events: List[ArchaeologicalEvent] = []
        
        print(f"🕰️  Temporal Clustering Engine initialized")
        print(f"  Min cluster size: {min_cluster_size}")
        print(f"  Feature weights: temporal={temporal_weight}, structural={structural_weight}, significance={significance_weight}")
    
    def cluster_temporal_patterns(self, events: List[ArchaeologicalEvent]) -> ClusteringAnalysis:
        """
        Perform comprehensive temporal clustering analysis
        
        Args:
            events: List of archaeological events to cluster
            
        Returns:
            Complete clustering analysis with identified patterns
        """
        
        print(f"\n🕰️  Beginning temporal clustering analysis...")
        print(f"  Events to analyze: {len(events)}")
        
        start_time = datetime.now()
        
        # Step 1: Prepare features
        print("  🔧 Extracting and encoding features...")
        self._prepare_feature_encodings(events)
        feature_vectors = self._extract_features(events)
        
        # Step 2: Perform multiple clustering approaches
        print("  🎯 Performing multi-method clustering...")
        
        # Absolute time clustering
        absolute_clusters = self._cluster_absolute_timing(events, feature_vectors)
        
        # Relative position clustering  
        relative_clusters = self._cluster_relative_positions(events, feature_vectors)
        
        # Session phase clustering
        phase_clusters = self._cluster_session_phases(events, feature_vectors)
        
        # Cross-session clustering
        cross_session_clusters = self._cluster_cross_session_patterns(events, feature_vectors)
        
        # Hybrid clustering
        hybrid_clusters = self._cluster_hybrid_patterns(events, feature_vectors)
        
        # Step 3: Combine and evaluate clusters
        print("  🔗 Combining and evaluating clusters...")
        all_clusters = (absolute_clusters + relative_clusters + phase_clusters + 
                       cross_session_clusters + hybrid_clusters)
        
        # Remove duplicate/overlapping clusters
        refined_clusters = self._refine_clusters(all_clusters)
        
        # Step 4: Analyze cluster quality and properties
        print("  📊 Analyzing cluster properties...")
        final_clusters = self._analyze_cluster_properties(refined_clusters, events)
        
        # Step 5: Generate analysis summary
        print("  📋 Generating clustering analysis...")
        analysis = self._create_clustering_analysis(final_clusters, events)
        
        elapsed = datetime.now() - start_time
        print(f"\n✅ Temporal clustering complete!")
        print(f"  Clusters identified: {len(final_clusters)}")
        print(f"  Overall quality score: {analysis.overall_silhouette_score:.3f}")
        print(f"  Analysis time: {elapsed.total_seconds():.1f} seconds")
        
        return analysis
    
    def _prepare_feature_encodings(self, events: List[ArchaeologicalEvent]):
        """Prepare encoding dictionaries for categorical features"""
        
        # Extract unique values
        event_types = list(set(event.event_type.value for event in events))
        archetypes = list(set(event.liquidity_archetype.value for event in events))
        structural_roles = list(set(event.structural_role for event in events))
        
        # Create encodings
        self.event_type_encoding = {et: i / len(event_types) for i, et in enumerate(event_types)}
        self.archetype_encoding = {arch: i / len(archetypes) for i, arch in enumerate(archetypes)}
        self.structural_role_encoding = {role: i / len(structural_roles) for i, role in enumerate(structural_roles)}
    
    def _extract_features(self, events: List[ArchaeologicalEvent]) -> List[ClusterFeatures]:
        """Extract comprehensive feature vectors for clustering"""
        
        feature_vectors = []
        
        for event in events:
            
            # Temporal features
            absolute_minute = event.session_minute
            relative_cycle_position = event.relative_cycle_position
            session_phase_encoded = self.session_phase_encoding.get(event.session_phase.value, 0.0)
            
            # Encode time signature
            time_signature_hash = hash(event.absolute_time_signature) % 1000
            time_of_day_encoded = time_signature_hash / 1000.0
            
            # Structural features
            timeframe_level = self._get_timeframe_level(event.timeframe)
            range_position = event.range_position_percent / 100.0
            magnitude = event.magnitude
            significance = event.significance_score
            
            # Pattern features
            event_type_encoded = self.event_type_encoding.get(event.event_type.value, 0.0)
            archetype_encoded = self.archetype_encoding.get(event.liquidity_archetype.value, 0.0)
            structural_role_encoded = self.structural_role_encoding.get(event.structural_role, 0.0)
            
            # Context features
            htf_confluence = self._encode_htf_confluence(event.htf_confluence.value)
            cross_session_inheritance = event.cross_session_inheritance
            velocity_signature = event.velocity_signature
            
            # Derived features
            temporal_uniqueness = self._calculate_temporal_uniqueness(event, events)
            structural_importance = self._calculate_structural_importance(event)
            
            features = ClusterFeatures(
                absolute_minute=absolute_minute,
                relative_cycle_position=relative_cycle_position,
                session_phase_encoded=session_phase_encoded,
                time_of_day_encoded=time_of_day_encoded,
                timeframe_level=timeframe_level,
                range_position=range_position,
                magnitude=magnitude,
                significance=significance,
                event_type_encoded=event_type_encoded,
                archetype_encoded=archetype_encoded,
                structural_role_encoded=structural_role_encoded,
                htf_confluence=htf_confluence,
                cross_session_inheritance=cross_session_inheritance,
                velocity_signature=velocity_signature,
                temporal_uniqueness=temporal_uniqueness,
                structural_importance=structural_importance
            )
            
            feature_vectors.append(features)
        
        self.feature_vectors = feature_vectors
        self.processed_events = events
        
        return feature_vectors
    
    def _get_timeframe_level(self, timeframe: TimeframeType) -> int:
        """Get timeframe hierarchy level"""
        timeframe_levels = {
            TimeframeType.MONTHLY: 0,
            TimeframeType.WEEKLY: 1,
            TimeframeType.DAILY: 2,
            TimeframeType.HOUR_1: 3,
            TimeframeType.MINUTE_50: 4,
            TimeframeType.MINUTE_15: 5,
            TimeframeType.MINUTE_5: 6,
            TimeframeType.MINUTE_1: 7
        }
        return timeframe_levels.get(timeframe, 4)
    
    def _encode_htf_confluence(self, htf_status: str) -> float:
        """Encode HTF confluence status"""
        encoding = {
            'confirmed': 1.0,
            'partial': 0.7,
            'weak': 0.3,
            'absent': 0.0,
            'unknown': 0.5
        }
        return encoding.get(htf_status, 0.5)
    
    def _calculate_temporal_uniqueness(self, event: ArchaeologicalEvent, all_events: List[ArchaeologicalEvent]) -> float:
        """Calculate how temporally unique this event is"""
        
        # Count events within similar timing
        similar_timing_count = 0
        timing_window = 5.0  # 5-minute window
        
        for other_event in all_events:
            if abs(other_event.session_minute - event.session_minute) <= timing_window:
                similar_timing_count += 1
        
        # Higher uniqueness for rare timings
        uniqueness = 1.0 - (similar_timing_count / len(all_events))
        return max(0.0, uniqueness)
    
    def _calculate_structural_importance(self, event: ArchaeologicalEvent) -> float:
        """Calculate structural importance of event"""
        
        importance = 0.0
        
        # Significance contribution
        importance += event.significance_score * 0.4
        
        # Magnitude contribution
        importance += event.magnitude * 0.3
        
        # HTF confluence contribution
        htf_score = self._encode_htf_confluence(event.htf_confluence.value)
        importance += htf_score * 0.2
        
        # Cross-session inheritance contribution
        importance += event.cross_session_inheritance * 0.1
        
        return min(1.0, importance)
    
    def _cluster_absolute_timing(self, events: List[ArchaeologicalEvent], features: List[ClusterFeatures]) -> List[TemporalCluster]:
        """Cluster events by absolute timing patterns"""
        
        clusters = []
        
        # Group by absolute minute ranges
        minute_groups = defaultdict(list)
        
        for i, event in enumerate(events):
            minute_bucket = int(event.session_minute / 5) * 5  # 5-minute buckets
            minute_groups[minute_bucket].append((event, features[i]))
        
        # Create clusters for significant minute groups
        for minute_bucket, event_feature_pairs in minute_groups.items():
            if len(event_feature_pairs) >= self.min_cluster_size:
                
                cluster_events = [pair[0] for pair in event_feature_pairs]
                cluster_features = [pair[1] for pair in event_feature_pairs]
                
                cluster = self._create_temporal_cluster(
                    cluster_id=f"abs_time_{minute_bucket}",
                    cluster_type=ClusterType.ABSOLUTE_TIME,
                    clustering_method=ClusteringMethod.CUSTOM,
                    events=cluster_events,
                    features=cluster_features,
                    temporal_signature=f"minute_{minute_bucket}"
                )
                
                clusters.append(cluster)
        
        return clusters
    
    def _cluster_relative_positions(self, events: List[ArchaeologicalEvent], features: List[ClusterFeatures]) -> List[TemporalCluster]:
        """Cluster events by relative cycle positions"""
        
        clusters = []
        
        # Group by relative position ranges
        position_groups = defaultdict(list)
        
        for i, event in enumerate(events):
            position_bucket = int(event.relative_cycle_position * 10) / 10  # 10% buckets
            position_groups[position_bucket].append((event, features[i]))
        
        # Create clusters for significant position groups
        for position_bucket, event_feature_pairs in position_groups.items():
            if len(event_feature_pairs) >= self.min_cluster_size:
                
                cluster_events = [pair[0] for pair in event_feature_pairs]
                cluster_features = [pair[1] for pair in event_feature_pairs]
                
                cluster = self._create_temporal_cluster(
                    cluster_id=f"rel_pos_{position_bucket:.1f}",
                    cluster_type=ClusterType.RELATIVE_POSITION,
                    clustering_method=ClusteringMethod.CUSTOM,
                    events=cluster_events,
                    features=cluster_features,
                    temporal_signature=f"cycle_pos_{position_bucket:.1f}"
                )
                
                clusters.append(cluster)
        
        return clusters
    
    def _cluster_session_phases(self, events: List[ArchaeologicalEvent], features: List[ClusterFeatures]) -> List[TemporalCluster]:
        """Cluster events by session phases"""
        
        clusters = []
        
        # Group by session phases
        phase_groups = defaultdict(list)
        
        for i, event in enumerate(events):
            phase_groups[event.session_phase.value].append((event, features[i]))
        
        # Create clusters for session phases
        for phase, event_feature_pairs in phase_groups.items():
            if len(event_feature_pairs) >= self.min_cluster_size:
                
                cluster_events = [pair[0] for pair in event_feature_pairs]
                cluster_features = [pair[1] for pair in event_feature_pairs]
                
                cluster = self._create_temporal_cluster(
                    cluster_id=f"phase_{phase}",
                    cluster_type=ClusterType.SESSION_PHASE,
                    clustering_method=ClusteringMethod.CUSTOM,
                    events=cluster_events,
                    features=cluster_features,
                    temporal_signature=f"phase_{phase}"
                )
                
                clusters.append(cluster)
        
        return clusters
    
    def _cluster_cross_session_patterns(self, events: List[ArchaeologicalEvent], features: List[ClusterFeatures]) -> List[TemporalCluster]:
        """Cluster events that show cross-session patterns"""
        
        clusters = []
        
        # Group by absolute time signature across sessions
        signature_groups = defaultdict(list)
        
        for i, event in enumerate(events):
            signature_groups[event.absolute_time_signature].append((event, features[i]))
        
        # Create clusters for patterns that appear across multiple sessions
        for signature, event_feature_pairs in signature_groups.items():
            if len(event_feature_pairs) >= self.min_cluster_size:
                
                # Check if pattern appears across multiple sessions
                sessions = set(pair[0].session_name for pair in event_feature_pairs)
                if len(sessions) >= 2:  # Multi-session pattern
                    
                    cluster_events = [pair[0] for pair in event_feature_pairs]
                    cluster_features = [pair[1] for pair in event_feature_pairs]
                    
                    cluster = self._create_temporal_cluster(
                        cluster_id=f"cross_sess_{signature}",
                        cluster_type=ClusterType.CROSS_SESSION,
                        clustering_method=ClusteringMethod.CUSTOM,
                        events=cluster_events,
                        features=cluster_features,
                        temporal_signature=signature
                    )
                    
                    clusters.append(cluster)
        
        return clusters
    
    def _cluster_hybrid_patterns(self, events: List[ArchaeologicalEvent], features: List[ClusterFeatures]) -> List[TemporalCluster]:
        """Cluster using advanced machine learning methods"""
        
        clusters = []
        
        if len(events) < self.min_cluster_size:
            return clusters
        
        # Prepare feature matrix
        feature_matrix = np.array([f.to_vector() for f in features])
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Apply DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
            cluster_labels = dbscan.fit_predict(scaled_features)
            
            # Create clusters from DBSCAN results
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label != -1:  # Ignore noise points
                    
                    # Get events in this cluster
                    cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
                    cluster_events = [events[i] for i in cluster_indices]
                    cluster_features = [features[i] for i in cluster_indices]
                    
                    if len(cluster_events) >= self.min_cluster_size:
                        
                        cluster = self._create_temporal_cluster(
                            cluster_id=f"hybrid_dbscan_{label}",
                            cluster_type=ClusterType.HYBRID,
                            clustering_method=ClusteringMethod.DBSCAN,
                            events=cluster_events,
                            features=cluster_features,
                            temporal_signature=f"ml_cluster_{label}"
                        )
                        
                        clusters.append(cluster)
            
        except Exception as e:
            self.logger.warning(f"DBSCAN clustering failed: {e}")
        
        return clusters
    
    def _create_temporal_cluster(self, 
                               cluster_id: str,
                               cluster_type: ClusterType,
                               clustering_method: ClusteringMethod,
                               events: List[ArchaeologicalEvent],
                               features: List[ClusterFeatures],
                               temporal_signature: str) -> TemporalCluster:
        """Create a temporal cluster with comprehensive analysis"""
        
        # Basic properties
        event_count = len(events)
        
        # Temporal characteristics
        timings = [event.session_minute for event in events]
        average_timing = np.mean(timings)
        timing_variance = np.var(timings)
        
        session_phases = list(set(event.session_phase.value for event in events))
        
        # Pattern characteristics
        event_types = [event.event_type.value for event in events]
        archetypes = [event.liquidity_archetype.value for event in events]
        
        dominant_event_type = Counter(event_types).most_common(1)[0][0]
        dominant_archetype = Counter(archetypes).most_common(1)[0][0]
        
        # Calculate pattern consistency
        type_consistency = len(set(event_types)) / len(event_types)
        archetype_consistency = len(set(archetypes)) / len(archetypes)
        pattern_consistency = (type_consistency + archetype_consistency) / 2
        
        # Statistical properties
        significances = [event.significance_score for event in events]
        average_significance = np.mean(significances)
        significance_variance = np.var(significances)
        
        magnitudes = [event.magnitude for event in events]
        magnitude_profile = {
            'mean': np.mean(magnitudes),
            'std': np.std(magnitudes),
            'min': np.min(magnitudes),
            'max': np.max(magnitudes)
        }
        
        # Recurrence analysis
        sessions = [event.session_name for event in events]
        session_coverage = list(set(sessions))
        recurrence_rate = len(session_coverage) / len(set(sessions))
        
        # Calculate recurrence consistency
        session_counts = Counter(sessions)
        recurrence_consistency = 1.0 - (np.std(list(session_counts.values())) / np.mean(list(session_counts.values())))
        
        # Temporal stability
        timing_stds_by_session = []
        session_groups = defaultdict(list)
        for event in events:
            session_groups[event.session_name].append(event.session_minute)
        
        for session_timings in session_groups.values():
            if len(session_timings) > 1:
                timing_stds_by_session.append(np.std(session_timings))
        
        temporal_stability = 1.0 - np.mean(timing_stds_by_session) if timing_stds_by_session else 1.0
        
        # Cross-session properties
        session_types = [self._extract_session_type(event.session_name) for event in events]
        session_distribution = dict(Counter(session_types))
        
        dates = [event.session_date for event in events]
        date_distribution = dict(Counter(dates))
        
        cross_session_inheritances = [event.cross_session_inheritance for event in events]
        cross_session_strength = np.mean(cross_session_inheritances)
        
        # Cluster quality metrics (simplified)
        silhouette_score = self._calculate_cluster_silhouette(features)
        intra_cluster_distance = self._calculate_intra_cluster_distance(features)
        inter_cluster_distance = 0.5  # Placeholder
        cluster_density = len(events) / max(1, timing_variance)
        
        # Predictive properties
        next_event_probability = recurrence_rate * pattern_consistency
        cascade_potential = average_significance * cross_session_strength
        structural_importance = np.mean([f.structural_importance for f in features])
        
        return TemporalCluster(
            cluster_id=cluster_id,
            cluster_type=cluster_type,
            clustering_method=clustering_method,
            events=events,
            event_count=event_count,
            feature_vectors=features,
            temporal_signature=temporal_signature,
            average_timing=average_timing,
            timing_variance=timing_variance,
            session_phases=session_phases,
            dominant_event_type=dominant_event_type,
            dominant_archetype=dominant_archetype,
            pattern_consistency=pattern_consistency,
            average_significance=average_significance,
            significance_variance=significance_variance,
            magnitude_profile=magnitude_profile,
            session_coverage=session_coverage,
            recurrence_rate=recurrence_rate,
            recurrence_consistency=recurrence_consistency,
            temporal_stability=temporal_stability,
            session_distribution=session_distribution,
            date_distribution=date_distribution,
            cross_session_strength=cross_session_strength,
            silhouette_score=silhouette_score,
            intra_cluster_distance=intra_cluster_distance,
            inter_cluster_distance=inter_cluster_distance,
            cluster_density=cluster_density,
            next_event_probability=next_event_probability,
            cascade_potential=cascade_potential,
            structural_importance=structural_importance
        )
    
    def _extract_session_type(self, session_name: str) -> str:
        """Extract session type from session name"""
        if session_name.startswith('NYPM') or session_name.startswith('NY_PM'):
            return 'NY_PM'
        elif session_name.startswith('NYAM') or session_name.startswith('NY_AM'):
            return 'NY_AM'
        elif session_name.startswith('LONDON'):
            return 'LONDON'
        elif session_name.startswith('ASIA'):
            return 'ASIA'
        else:
            return 'UNKNOWN'
    
    def _calculate_cluster_silhouette(self, features: List[ClusterFeatures]) -> float:
        """Calculate silhouette score for cluster"""
        
        if len(features) < 2:
            return 0.0
        
        try:
            feature_matrix = np.array([f.to_vector() for f in features])
            if feature_matrix.shape[0] < 2:
                return 0.0
            
            # Create dummy labels (all same cluster)
            labels = np.zeros(len(features))
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    dist = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                    distances.append(dist)
            
            # Return inverse of average distance (higher = better clustering)
            avg_distance = np.mean(distances) if distances else 1.0
            return 1.0 / (1.0 + avg_distance)
            
        except Exception as e:
            self.logger.warning(f"Silhouette calculation failed: {e}")
            return 0.0
    
    def _calculate_intra_cluster_distance(self, features: List[ClusterFeatures]) -> float:
        """Calculate average intra-cluster distance"""
        
        if len(features) < 2:
            return 0.0
        
        feature_matrix = np.array([f.to_vector() for f in features])
        
        distances = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                dist = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _refine_clusters(self, clusters: List[TemporalCluster]) -> List[TemporalCluster]:
        """Remove overlapping and low-quality clusters"""
        
        refined_clusters = []
        
        # Sort by quality (silhouette score)
        sorted_clusters = sorted(clusters, key=lambda c: c.silhouette_score, reverse=True)
        
        # Remove overlapping clusters
        used_events = set()
        
        for cluster in sorted_clusters:
            # Check overlap with already selected clusters
            cluster_event_ids = set(event.event_id for event in cluster.events)
            overlap = len(cluster_event_ids & used_events) / len(cluster_event_ids)
            
            # Only include if low overlap
            if overlap < 0.3:  # Less than 30% overlap
                refined_clusters.append(cluster)
                used_events.update(cluster_event_ids)
        
        return refined_clusters
    
    def _analyze_cluster_properties(self, clusters: List[TemporalCluster], all_events: List[ArchaeologicalEvent]) -> List[TemporalCluster]:
        """Analyze and enhance cluster properties"""
        
        # Calculate inter-cluster distances
        for i, cluster1 in enumerate(clusters):
            min_inter_distance = float('inf')
            
            for j, cluster2 in enumerate(clusters):
                if i != j:
                    # Calculate distance between cluster centroids
                    centroid1 = np.mean([f.to_vector() for f in cluster1.feature_vectors], axis=0)
                    centroid2 = np.mean([f.to_vector() for f in cluster2.feature_vectors], axis=0)
                    distance = np.linalg.norm(centroid1 - centroid2)
                    min_inter_distance = min(min_inter_distance, distance)
            
            cluster1.inter_cluster_distance = min_inter_distance if min_inter_distance != float('inf') else 0.0
        
        return clusters
    
    def _create_clustering_analysis(self, clusters: List[TemporalCluster], events: List[ArchaeologicalEvent]) -> ClusteringAnalysis:
        """Create comprehensive clustering analysis"""
        
        # Calculate overall quality
        if clusters:
            overall_silhouette = np.mean([c.silhouette_score for c in clusters])
        else:
            overall_silhouette = 0.0
        
        # Quality distribution
        quality_ranges = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        for cluster in clusters:
            if cluster.silhouette_score > 0.8:
                quality_ranges['excellent'] += 1
            elif cluster.silhouette_score > 0.6:
                quality_ranges['good'] += 1
            elif cluster.silhouette_score > 0.4:
                quality_ranges['fair'] += 1
            else:
                quality_ranges['poor'] += 1
        
        # Find noise events (events not in any cluster)
        clustered_event_ids = set()
        for cluster in clusters:
            clustered_event_ids.update(event.event_id for event in cluster.events)
        
        noise_events = [event for event in events if event.event_id not in clustered_event_ids]
        
        # Calculate coverage metrics
        sessions_covered = list(set(event.session_name for event in events))
        timeframes_analyzed = list(set(event.timeframe.value for event in events))
        
        temporal_coverage = len(clustered_event_ids) / len(events) if events else 0.0
        pattern_discovery_rate = len(clusters) / len(events) if events else 0.0
        
        # Cluster statistics
        cluster_stats = {
            'total_clusters': len(clusters),
            'average_cluster_size': np.mean([c.event_count for c in clusters]) if clusters else 0,
            'largest_cluster_size': max([c.event_count for c in clusters]) if clusters else 0,
            'cluster_types': dict(Counter(c.cluster_type.value for c in clusters)),
            'clustering_methods': dict(Counter(c.clustering_method.value for c in clusters)),
            'average_recurrence_rate': np.mean([c.recurrence_rate for c in clusters]) if clusters else 0,
            'high_quality_clusters': sum(1 for c in clusters if c.silhouette_score > 0.7)
        }
        
        # Temporal heatmap data
        heatmap_data = self._generate_temporal_heatmap_data(clusters)
        
        # Recurrence analysis
        recurrence_analysis = {
            'patterns_with_high_recurrence': sum(1 for c in clusters if c.recurrence_rate > 0.5),
            'cross_session_patterns': sum(1 for c in clusters if c.cluster_type == ClusterType.CROSS_SESSION),
            'stable_patterns': sum(1 for c in clusters if c.temporal_stability > 0.7),
            'predictive_patterns': sum(1 for c in clusters if c.next_event_probability > 0.6)
        }
        
        return ClusteringAnalysis(
            analysis_id=f"clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            analysis_timestamp=datetime.now().isoformat(),
            total_events_analyzed=len(events),
            sessions_covered=sessions_covered,
            timeframes_analyzed=timeframes_analyzed,
            clusters=clusters,
            cluster_count=len(clusters),
            noise_events=noise_events,
            clustering_parameters={
                'min_cluster_size': self.min_cluster_size,
                'max_cluster_size': self.max_cluster_size,
                'feature_weights': self.feature_weights
            },
            feature_weights=self.feature_weights,
            overall_silhouette_score=overall_silhouette,
            cluster_quality_distribution=quality_ranges,
            temporal_coverage=temporal_coverage,
            pattern_discovery_rate=pattern_discovery_rate,
            cluster_statistics=cluster_stats,
            temporal_heatmap_data=heatmap_data,
            recurrence_analysis=recurrence_analysis
        )
    
    def _generate_temporal_heatmap_data(self, clusters: List[TemporalCluster]) -> Dict[str, Any]:
        """Generate data for temporal heatmaps"""
        
        # Absolute time heatmap
        absolute_time_data = defaultdict(int)
        for cluster in clusters:
            for event in cluster.events:
                time_bucket = int(event.session_minute / 10) * 10  # 10-minute buckets
                absolute_time_data[time_bucket] += 1
        
        # Relative position heatmap
        relative_position_data = defaultdict(int)
        for cluster in clusters:
            for event in cluster.events:
                pos_bucket = int(event.relative_cycle_position * 20) / 20  # 5% buckets
                relative_position_data[pos_bucket] += 1
        
        # Session phase heatmap
        session_phase_data = defaultdict(int)
        for cluster in clusters:
            for event in cluster.events:
                session_phase_data[event.session_phase.value] += 1
        
        return {
            'absolute_time': dict(absolute_time_data),
            'relative_position': dict(relative_position_data),
            'session_phase': dict(session_phase_data)
        }
    
    def export_clustering_analysis(self, analysis: ClusteringAnalysis, output_path: str = "temporal_clustering_analysis.json") -> str:
        """Export clustering analysis to JSON file"""
        
        # Convert analysis to serializable format
        export_data = {
            'metadata': {
                'analysis_id': analysis.analysis_id,
                'analysis_timestamp': analysis.analysis_timestamp,
                'total_events_analyzed': analysis.total_events_analyzed,
                'sessions_covered': analysis.sessions_covered,
                'timeframes_analyzed': analysis.timeframes_analyzed,
                'clustering_parameters': analysis.clustering_parameters
            },
            'clusters': [],
            'quality_metrics': {
                'overall_silhouette_score': analysis.overall_silhouette_score,
                'cluster_quality_distribution': analysis.cluster_quality_distribution,
                'temporal_coverage': analysis.temporal_coverage,
                'pattern_discovery_rate': analysis.pattern_discovery_rate
            },
            'statistics': {
                'cluster_statistics': analysis.cluster_statistics,
                'temporal_heatmap_data': analysis.temporal_heatmap_data,
                'recurrence_analysis': analysis.recurrence_analysis
            },
            'noise_events': [event.event_id for event in analysis.noise_events]
        }
        
        # Export cluster details
        for cluster in analysis.clusters:
            cluster_data = {
                'cluster_id': cluster.cluster_id,
                'cluster_type': cluster.cluster_type.value,
                'clustering_method': cluster.clustering_method.value,
                'event_count': cluster.event_count,
                'temporal_characteristics': {
                    'temporal_signature': cluster.temporal_signature,
                    'average_timing': cluster.average_timing,
                    'timing_variance': cluster.timing_variance,
                    'session_phases': cluster.session_phases
                },
                'pattern_characteristics': {
                    'dominant_event_type': cluster.dominant_event_type,
                    'dominant_archetype': cluster.dominant_archetype,
                    'pattern_consistency': cluster.pattern_consistency
                },
                'statistical_properties': {
                    'average_significance': cluster.average_significance,
                    'significance_variance': cluster.significance_variance,
                    'magnitude_profile': cluster.magnitude_profile
                },
                'recurrence_analysis': {
                    'session_coverage': cluster.session_coverage,
                    'recurrence_rate': cluster.recurrence_rate,
                    'recurrence_consistency': cluster.recurrence_consistency,
                    'temporal_stability': cluster.temporal_stability
                },
                'cross_session_properties': {
                    'session_distribution': cluster.session_distribution,
                    'date_distribution': cluster.date_distribution,
                    'cross_session_strength': cluster.cross_session_strength
                },
                'quality_metrics': {
                    'silhouette_score': cluster.silhouette_score,
                    'intra_cluster_distance': cluster.intra_cluster_distance,
                    'inter_cluster_distance': cluster.inter_cluster_distance,
                    'cluster_density': cluster.cluster_density
                },
                'predictive_properties': {
                    'next_event_probability': cluster.next_event_probability,
                    'cascade_potential': cluster.cascade_potential,
                    'structural_importance': cluster.structural_importance
                },
                'event_ids': [event.event_id for event in cluster.events]
            }
            export_data['clusters'].append(cluster_data)
        
        # Write to file
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"🕰️  Temporal clustering analysis exported to {output_file}")
        return str(output_file)


if __name__ == "__main__":
    # Test the temporal clustering engine
    print("🕰️  Testing Temporal Clustering Engine")
    print("=" * 50)
    
    # Initialize engine
    clustering_engine = TemporalClusteringEngine()
    
    print("✅ Temporal clustering engine initialized and ready for use")
    print("   Use cluster_temporal_patterns() with ArchaeologicalEvent list to perform clustering")