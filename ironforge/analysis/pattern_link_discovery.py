#!/usr/bin/env python3
"""
IRONFORGE Pattern Link Discovery System
======================================

Analyzes temporal, structural, and session-based relationships from discovered patterns.
Focuses on cross-session pattern evolution and recurring market structures.

Features:
- Temporal relationship analysis across sessions
- Structural pattern similarity detection  
- Cross-session evolution tracking
- Recurring market structure identification
- Pattern confluence analysis
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
from pathlib import Path
import logging

@dataclass
class PatternLink:
    """Represents a discovered link between patterns"""
    source_pattern_id: str
    target_pattern_id: str
    link_type: str
    strength: float
    temporal_distance: Optional[float] = None
    structural_similarity: Optional[float] = None
    semantic_overlap: Optional[float] = None
    session_relationship: Optional[str] = None
    evolution_stage: Optional[str] = None

@dataclass
class PatternCluster:
    """Represents a cluster of related patterns"""
    cluster_id: str
    pattern_ids: List[str]
    cluster_type: str
    centroid_features: Dict
    temporal_span: Tuple[str, str]
    session_coverage: Set[str]
    significance_score: float

class PatternLinkDiscovery:
    """
    Discovers and analyzes relationships between archaeological patterns
    """
    
    def __init__(self, patterns_file: str = None, session_data_path: str = None):
        self.logger = logging.getLogger('pattern_link_discovery')
        
        # Load discovered patterns
        if patterns_file is None:
            patterns_file = '/Users/jack/IRONFORGE/IRONFORGE/preservation/discovered_patterns.json'
        
        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)
        
        self.session_data_path = session_data_path or '/Users/jack/IRONFORGE/enhanced_sessions_with_relativity'
        
        # Initialize analysis structures
        self.pattern_links: List[PatternLink] = []
        self.pattern_clusters: List[PatternCluster] = []
        self.session_timeline = self._build_session_timeline()
        self.pattern_index = self._build_pattern_index()
        
        print(f"🔍 Pattern Link Discovery initialized with {len(self.patterns)} patterns")
    
    def _build_session_timeline(self) -> Dict:
        """Build timeline from enhanced session filenames"""
        timeline = {}
        session_path = Path(self.session_data_path)
        
        if session_path.exists():
            for session_file in session_path.glob('*.json'):
                # Parse session info from filename
                # Format: enhanced_rel_SESSION_Lvl-1_YYYY_MM_DD.json
                filename = session_file.name
                match = re.search(r'enhanced_rel_(\w+)_Lvl-1_(\d{4})_(\d{2})_(\d{2})', filename)
                
                if match:
                    session_type, year, month, day = match.groups()
                    date_str = f"{year}-{month}-{day}"
                    
                    if date_str not in timeline:
                        timeline[date_str] = {}
                    timeline[date_str][session_type] = filename
        
        return timeline
    
    def _build_pattern_index(self) -> Dict:
        """Build searchable index of patterns by various attributes"""
        index = {
            'by_type': defaultdict(list),
            'by_phase': defaultdict(list),
            'by_session': defaultdict(list),
            'by_range_position': defaultdict(list),
            'by_semantic_events': defaultdict(list)
        }
        
        for i, pattern in enumerate(self.patterns):
            pattern_idx = f"pattern_{i}"
            
            # Index by type
            index['by_type'][pattern.get('type', 'unknown')].append(pattern_idx)
            
            # Index by phase
            phase_info = pattern.get('phase_information', {})
            phase = phase_info.get('primary_phase', 'unknown')
            index['by_phase'][phase].append(pattern_idx)
            
            # Index by session (if we can extract it)
            session = pattern.get('session_name', 'unknown')
            index['by_session'][session].append(pattern_idx)
            
            # Index by range position (extract from description)
            desc = pattern.get('description', '')
            range_match = re.search(r'(\d+\.?\d*)% of range', desc)
            if range_match:
                range_pct = float(range_match.group(1))
                range_bucket = f"{int(range_pct//20)*20}-{int(range_pct//20)*20+20}%"
                index['by_range_position'][range_bucket].append(pattern_idx)
            
            # Index by semantic events
            semantic_context = pattern.get('semantic_context', {})
            event_types = semantic_context.get('event_types', [])
            for event in event_types:
                index['by_semantic_events'][event].append(pattern_idx)
        
        return index
    
    def analyze_temporal_relationships(self) -> List[PatternLink]:
        """Analyze temporal relationships between patterns"""
        print("🕐 Analyzing temporal relationships...")
        temporal_links = []
        
        # Group patterns by session timing
        session_groups = self._group_patterns_by_session_timing()
        
        # Analyze within-session temporal progression
        for session_key, pattern_indices in session_groups.items():
            if len(pattern_indices) > 1:
                # Sort by session position
                sorted_patterns = self._sort_by_session_position(pattern_indices)
                
                # Find temporal progression links
                for i in range(len(sorted_patterns) - 1):
                    current_idx = sorted_patterns[i]
                    next_idx = sorted_patterns[i + 1]
                    
                    temporal_dist = self._calculate_temporal_distance(current_idx, next_idx)
                    
                    if temporal_dist is not None and temporal_dist < 1.0:  # Within same session
                        link = PatternLink(
                            source_pattern_id=f"pattern_{current_idx}",
                            target_pattern_id=f"pattern_{next_idx}",
                            link_type="temporal_progression",
                            strength=1.0 - temporal_dist,
                            temporal_distance=temporal_dist,
                            session_relationship="intra_session"
                        )
                        temporal_links.append(link)
        
        # Analyze cross-session temporal relationships
        cross_session_links = self._analyze_cross_session_temporal()
        temporal_links.extend(cross_session_links)
        
        print(f"  Found {len(temporal_links)} temporal relationships")
        return temporal_links
    
    def analyze_structural_relationships(self) -> List[PatternLink]:
        """Analyze structural similarity between patterns"""
        print("🏗️ Analyzing structural relationships...")
        structural_links = []
        
        # Analyze range position similarities
        range_links = self._analyze_range_position_similarity()
        structural_links.extend(range_links)
        
        # Analyze pattern type relationships
        type_links = self._analyze_pattern_type_relationships()
        structural_links.extend(type_links)
        
        # Analyze phase significance relationships
        phase_links = self._analyze_phase_relationships()
        structural_links.extend(phase_links)
        
        print(f"  Found {len(structural_links)} structural relationships")
        return structural_links
    
    def analyze_cross_session_evolution(self) -> List[PatternLink]:
        """Track pattern evolution across different sessions"""
        print("🔄 Analyzing cross-session pattern evolution...")
        evolution_links = []
        
        # Group similar patterns across sessions
        pattern_families = self._identify_pattern_families()
        
        for family_id, family_patterns in pattern_families.items():
            if len(family_patterns) > 1:
                # Sort by session date/time
                sorted_family = self._sort_by_session_date(family_patterns)
                
                # Create evolution chain
                for i in range(len(sorted_family) - 1):
                    current = sorted_family[i]
                    next_pattern = sorted_family[i + 1]
                    
                    evolution_stage = self._determine_evolution_stage(current, next_pattern, i, len(sorted_family))
                    similarity = self._calculate_pattern_similarity(current, next_pattern)
                    
                    if similarity > 0.7:  # High similarity threshold
                        link = PatternLink(
                            source_pattern_id=f"pattern_{current}",
                            target_pattern_id=f"pattern_{next_pattern}",
                            link_type="cross_session_evolution",
                            strength=similarity,
                            structural_similarity=similarity,
                            evolution_stage=evolution_stage,
                            session_relationship="cross_session"
                        )
                        evolution_links.append(link)
        
        print(f"  Found {len(evolution_links)} evolution relationships")
        return evolution_links
    
    def identify_recurring_structures(self) -> List[PatternCluster]:
        """Identify recurring market structures across sessions"""
        print("🔁 Identifying recurring market structures...")
        recurring_clusters = []
        
        # Cluster by structural similarity
        structural_clusters = self._cluster_by_structural_features()
        
        # Cluster by temporal patterns
        temporal_clusters = self._cluster_by_temporal_features()
        
        # Cluster by semantic similarity
        semantic_clusters = self._cluster_by_semantic_features()
        
        # Merge and validate clusters
        all_clusters = structural_clusters + temporal_clusters + semantic_clusters
        validated_clusters = self._validate_clusters(all_clusters)
        
        print(f"  Identified {len(validated_clusters)} recurring structures")
        return validated_clusters
    
    def discover_pattern_links(self) -> Dict:
        """Main method to discover all pattern relationships"""
        print("🚀 Starting comprehensive pattern link discovery...")
        
        # Analyze different types of relationships
        temporal_links = self.analyze_temporal_relationships()
        structural_links = self.analyze_structural_relationships()
        evolution_links = self.analyze_cross_session_evolution()
        
        # Combine all links
        self.pattern_links = temporal_links + structural_links + evolution_links
        
        # Identify recurring structures
        self.pattern_clusters = self.identify_recurring_structures()
        
        # Build comprehensive analysis report
        results = {
            'total_patterns': len(self.patterns),
            'temporal_links': len(temporal_links),
            'structural_links': len(structural_links),  
            'evolution_links': len(evolution_links),
            'total_links': len(self.pattern_links),
            'recurring_clusters': len(self.pattern_clusters),
            'link_density': len(self.pattern_links) / len(self.patterns) if self.patterns else 0,
            'analysis_summary': self._generate_analysis_summary()
        }
        
        print(f"✅ Pattern link discovery complete!")
        print(f"  Total links discovered: {results['total_links']}")
        print(f"  Recurring structures: {results['recurring_clusters']}")
        print(f"  Link density: {results['link_density']:.2f} links per pattern")
        
        return results
    
    # Helper methods for analysis
    def _group_patterns_by_session_timing(self) -> Dict:
        """Group patterns by session timing characteristics"""
        groups = defaultdict(list)
        
        for i, pattern in enumerate(self.patterns):
            session_name = pattern.get('session_name', 'unknown')
            session_start = pattern.get('session_start', '00:00:00')
            session_end = pattern.get('session_end', '23:59:59')
            
            key = f"{session_name}_{session_start}_{session_end}"
            groups[key].append(i)
        
        return groups
    
    def _sort_by_session_position(self, pattern_indices: List[int]) -> List[int]:
        """Sort patterns by their position within session"""
        def get_session_position(idx):
            pattern = self.patterns[idx]
            phase_info = pattern.get('phase_information', {})
            return phase_info.get('session_position', 0)
        
        return sorted(pattern_indices, key=get_session_position)
    
    def _calculate_temporal_distance(self, idx1: int, idx2: int) -> Optional[float]:
        """Calculate temporal distance between two patterns"""
        pattern1 = self.patterns[idx1]
        pattern2 = self.patterns[idx2]
        
        pos1 = pattern1.get('phase_information', {}).get('session_position', 0)
        pos2 = pattern2.get('phase_information', {}).get('session_position', 0)
        
        if pos1 is not None and pos2 is not None:
            return abs(pos2 - pos1)
        
        return None
    
    def _analyze_cross_session_temporal(self) -> List[PatternLink]:
        """Analyze temporal relationships across different sessions"""
        links = []
        
        # Group patterns by day and analyze day-to-day evolution
        daily_patterns = defaultdict(list)
        
        for i, pattern in enumerate(self.patterns):
            # Extract date from session context if available
            # For now, use session timing as proxy
            session_start = pattern.get('session_start', '09:30:00')
            daily_patterns[session_start].append(i)
        
        # Find cross-day temporal links
        for session_time, pattern_indices in daily_patterns.items():
            if len(pattern_indices) > 1:
                # Create links between similar patterns at same session time
                for i in range(len(pattern_indices) - 1):
                    for j in range(i + 1, len(pattern_indices)):
                        idx1, idx2 = pattern_indices[i], pattern_indices[j]
                        similarity = self._calculate_pattern_similarity(idx1, idx2)
                        
                        if similarity > 0.8:  # High similarity threshold
                            link = PatternLink(
                                source_pattern_id=f"pattern_{idx1}",
                                target_pattern_id=f"pattern_{idx2}",
                                link_type="cross_session_temporal",
                                strength=similarity,
                                session_relationship="cross_session",
                                temporal_distance=0.0  # Same session time
                            )
                            links.append(link)
        
        return links
    
    def _analyze_range_position_similarity(self) -> List[PatternLink]:
        """Analyze patterns with similar range positions"""
        links = []
        
        # Group patterns by range position buckets
        for range_bucket, pattern_indices in self.pattern_index['by_range_position'].items():
            if len(pattern_indices) > 1:
                # Create structural similarity links within range buckets
                for i in range(len(pattern_indices) - 1):
                    for j in range(i + 1, len(pattern_indices)):
                        idx1 = int(pattern_indices[i].split('_')[1])
                        idx2 = int(pattern_indices[j].split('_')[1])
                        
                        structural_sim = self._calculate_structural_similarity(idx1, idx2)
                        
                        if structural_sim > 0.6:
                            link = PatternLink(
                                source_pattern_id=pattern_indices[i],
                                target_pattern_id=pattern_indices[j],
                                link_type="range_position_similarity",
                                strength=structural_sim,
                                structural_similarity=structural_sim
                            )
                            links.append(link)
        
        return links
    
    def _analyze_pattern_type_relationships(self) -> List[PatternLink]:
        """Analyze relationships between different pattern types"""
        links = []
        
        # Define pattern type relationships
        type_relationships = {
            ('range_position_confluence', 'session_open_relationship'): 'complementary',
            ('session_open_relationship', 'structural_context_confluence'): 'progressive',
            ('range_position_confluence', 'structural_context_confluence'): 'convergent'
        }
        
        for (type1, type2), relationship in type_relationships.items():
            patterns1 = self.pattern_index['by_type'][type1]
            patterns2 = self.pattern_index['by_type'][type2]
            
            # Find related patterns of different types
            for p1 in patterns1[:10]:  # Limit for performance
                for p2 in patterns2[:10]:
                    if p1 != p2:
                        idx1 = int(p1.split('_')[1])
                        idx2 = int(p2.split('_')[1])
                        
                        semantic_overlap = self._calculate_semantic_overlap(idx1, idx2)
                        
                        if semantic_overlap > 0.5:
                            link = PatternLink(
                                source_pattern_id=p1,
                                target_pattern_id=p2,
                                link_type=f"type_relationship_{relationship}",
                                strength=semantic_overlap,
                                semantic_overlap=semantic_overlap
                            )
                            links.append(link)
        
        return links
    
    def _analyze_phase_relationships(self) -> List[PatternLink]:
        """Analyze relationships between patterns in different phases"""
        links = []
        
        # Analyze phase transitions and relationships
        phase_patterns = self.pattern_index['by_phase']
        
        for phase, pattern_list in phase_patterns.items():
            if len(pattern_list) > 1:
                # Create intra-phase links for strong patterns
                for i in range(len(pattern_list) - 1):
                    for j in range(i + 1, min(i + 5, len(pattern_list))):  # Limit comparisons
                        p1, p2 = pattern_list[i], pattern_list[j]
                        idx1 = int(p1.split('_')[1])
                        idx2 = int(p2.split('_')[1])
                        
                        phase_similarity = self._calculate_phase_similarity(idx1, idx2)
                        
                        if phase_similarity > 0.7:
                            link = PatternLink(
                                source_pattern_id=p1,
                                target_pattern_id=p2,
                                link_type="phase_similarity",
                                strength=phase_similarity,
                                structural_similarity=phase_similarity
                            )
                            links.append(link)
        
        return links
    
    def _identify_pattern_families(self) -> Dict[str, List[int]]:
        """Identify families of similar patterns across sessions"""
        families = defaultdict(list)
        
        # Group by pattern type first
        for pattern_type, pattern_list in self.pattern_index['by_type'].items():
            type_patterns = [int(p.split('_')[1]) for p in pattern_list]
            
            # Sub-group by range position similarity
            range_groups = defaultdict(list)
            for idx in type_patterns:
                pattern = self.patterns[idx]
                desc = pattern.get('description', '')
                range_match = re.search(r'(\d+\.?\d*)% of range', desc)
                
                if range_match:
                    range_pct = float(range_match.group(1))
                    range_bucket = int(range_pct // 10) * 10  # 10% buckets
                    range_groups[range_bucket].append(idx)
            
            # Create families from range groups
            for range_bucket, indices in range_groups.items():
                if len(indices) > 1:
                    family_id = f"{pattern_type}_{range_bucket}pct"
                    families[family_id] = indices
        
        return families
    
    def _sort_by_session_date(self, pattern_indices: List[int]) -> List[int]:
        """Sort patterns by session date (approximate from session timing)"""
        # For now, sort by session position as proxy for temporal ordering
        def get_sort_key(idx):
            pattern = self.patterns[idx]
            phase_info = pattern.get('phase_information', {})
            session_pos = phase_info.get('session_position', 0)
            return session_pos
        
        return sorted(pattern_indices, key=get_sort_key)
    
    def _determine_evolution_stage(self, idx1: int, idx2: int, position: int, total: int) -> str:
        """Determine evolution stage between two patterns"""
        progress = position / max(1, total - 1)
        
        if progress < 0.33:
            return "early_evolution"
        elif progress < 0.67:
            return "mid_evolution"
        else:
            return "late_evolution"
    
    def _calculate_pattern_similarity(self, idx1: int, idx2: int) -> float:
        """Calculate overall similarity between two patterns"""
        pattern1 = self.patterns[idx1]
        pattern2 = self.patterns[idx2]
        
        similarity_score = 0.0
        
        # Type similarity
        if pattern1.get('type') == pattern2.get('type'):
            similarity_score += 0.3
        
        # Range position similarity
        desc1 = pattern1.get('description', '')
        desc2 = pattern2.get('description', '')
        
        range1 = self._extract_range_percent(desc1)
        range2 = self._extract_range_percent(desc2)
        
        if range1 is not None and range2 is not None:
            range_diff = abs(range1 - range2)
            range_similarity = max(0, 1.0 - range_diff / 100.0)
            similarity_score += 0.3 * range_similarity
        
        # Phase similarity
        phase1 = pattern1.get('phase_information', {}).get('primary_phase', '')
        phase2 = pattern2.get('phase_information', {}).get('primary_phase', '')
        
        if phase1 == phase2:
            similarity_score += 0.2
        
        # Semantic similarity
        semantic_sim = self._calculate_semantic_overlap(idx1, idx2)
        similarity_score += 0.2 * semantic_sim
        
        return min(1.0, similarity_score)
    
    def _calculate_structural_similarity(self, idx1: int, idx2: int) -> float:
        """Calculate structural similarity between patterns"""
        pattern1 = self.patterns[idx1]
        pattern2 = self.patterns[idx2]
        
        structural_score = 0.0
        
        # Phase significance similarity
        phase1 = pattern1.get('phase_information', {})
        phase2 = pattern2.get('phase_information', {})
        
        sig1 = phase1.get('phase_significance', 0)
        sig2 = phase2.get('phase_significance', 0)
        
        if sig1 > 0 and sig2 > 0:
            sig_similarity = 1.0 - abs(sig1 - sig2)
            structural_score += 0.4 * sig_similarity
        
        # Structural context similarity
        struct1 = pattern1.get('semantic_context', {}).get('structural_context', {})
        struct2 = pattern2.get('semantic_context', {}).get('structural_context', {})
        
        strength1 = struct1.get('pattern_strength', 0)
        strength2 = struct2.get('pattern_strength', 0)
        
        if strength1 > 0 and strength2 > 0:
            strength_similarity = 1.0 - abs(strength1 - strength2)
            structural_score += 0.3 * strength_similarity
        
        # Type similarity
        if pattern1.get('type') == pattern2.get('type'):
            structural_score += 0.3
        
        return min(1.0, structural_score)
    
    def _calculate_semantic_overlap(self, idx1: int, idx2: int) -> float:
        """Calculate semantic overlap between patterns"""
        pattern1 = self.patterns[idx1]
        pattern2 = self.patterns[idx2]
        
        semantic1 = pattern1.get('semantic_context', {})
        semantic2 = pattern2.get('semantic_context', {})
        
        events1 = set(semantic1.get('event_types', []))
        events2 = set(semantic2.get('event_types', []))
        
        if not events1 and not events2:
            return 1.0  # Both have no events
        
        if not events1 or not events2:
            return 0.0  # One has events, other doesn't
        
        intersection = events1.intersection(events2)
        union = events1.union(events2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_phase_similarity(self, idx1: int, idx2: int) -> float:
        """Calculate phase-based similarity"""
        pattern1 = self.patterns[idx1]
        pattern2 = self.patterns[idx2]
        
        phase1 = pattern1.get('phase_information', {})
        phase2 = pattern2.get('phase_information', {})
        
        similarity = 0.0
        
        # Primary phase match
        if phase1.get('primary_phase') == phase2.get('primary_phase'):
            similarity += 0.5
        
        # Session position similarity
        pos1 = phase1.get('session_position', 0)
        pos2 = phase2.get('session_position', 0)
        
        if pos1 > 0 and pos2 > 0:
            pos_similarity = 1.0 - abs(pos1 - pos2) / max(pos1, pos2)
            similarity += 0.3 * pos_similarity
        
        # Phase significance similarity  
        sig1 = phase1.get('phase_significance', 0)
        sig2 = phase2.get('phase_significance', 0)
        
        if sig1 > 0 and sig2 > 0:
            sig_similarity = 1.0 - abs(sig1 - sig2)
            similarity += 0.2 * sig_similarity
        
        return min(1.0, similarity)
    
    def _extract_range_percent(self, description: str) -> Optional[float]:
        """Extract range percentage from pattern description"""
        match = re.search(r'(\d+\.?\d*)% of range', description)
        return float(match.group(1)) if match else None
    
    def _cluster_by_structural_features(self) -> List[PatternCluster]:
        """Cluster patterns by structural features"""
        clusters = []
        
        # Group by pattern type and range position
        type_range_groups = defaultdict(list)
        
        for i, pattern in enumerate(self.patterns):
            pattern_type = pattern.get('type', 'unknown')
            desc = pattern.get('description', '')
            range_pct = self._extract_range_percent(desc)
            
            if range_pct is not None:
                range_bucket = int(range_pct // 20) * 20  # 20% buckets
                key = f"{pattern_type}_{range_bucket}pct"
                type_range_groups[key].append(i)
        
        # Create clusters from groups
        for cluster_id, pattern_indices in type_range_groups.items():
            if len(pattern_indices) >= 3:  # Minimum cluster size
                cluster = PatternCluster(
                    cluster_id=f"structural_{cluster_id}",
                    pattern_ids=[f"pattern_{i}" for i in pattern_indices],
                    cluster_type="structural_similarity",
                    centroid_features=self._calculate_cluster_centroid(pattern_indices),
                    temporal_span=self._calculate_temporal_span(pattern_indices),
                    session_coverage=self._calculate_session_coverage(pattern_indices),
                    significance_score=len(pattern_indices) / len(self.patterns)
                )
                clusters.append(cluster)
        
        return clusters
    
    def _cluster_by_temporal_features(self) -> List[PatternCluster]:
        """Cluster patterns by temporal features"""
        clusters = []
        
        # Group by phase and session position ranges
        temporal_groups = defaultdict(list)
        
        for i, pattern in enumerate(self.patterns):
            phase_info = pattern.get('phase_information', {})
            phase = phase_info.get('primary_phase', 'unknown')
            position = phase_info.get('session_position', 0)
            
            # Create position buckets
            if position > 0:
                pos_bucket = int(position) if position < 10 else "high"
                key = f"{phase}_{pos_bucket}"
                temporal_groups[key].append(i)
        
        # Create temporal clusters
        for cluster_id, pattern_indices in temporal_groups.items():
            if len(pattern_indices) >= 3:
                cluster = PatternCluster(
                    cluster_id=f"temporal_{cluster_id}",
                    pattern_ids=[f"pattern_{i}" for i in pattern_indices],
                    cluster_type="temporal_similarity",
                    centroid_features=self._calculate_cluster_centroid(pattern_indices),
                    temporal_span=self._calculate_temporal_span(pattern_indices),
                    session_coverage=self._calculate_session_coverage(pattern_indices),
                    significance_score=len(pattern_indices) / len(self.patterns)
                )
                clusters.append(cluster)
        
        return clusters
    
    def _cluster_by_semantic_features(self) -> List[PatternCluster]:
        """Cluster patterns by semantic features"""
        clusters = []
        
        # Group by semantic event combinations
        semantic_groups = defaultdict(list)
        
        for i, pattern in enumerate(self.patterns):
            semantic_context = pattern.get('semantic_context', {})
            event_types = semantic_context.get('event_types', [])
            
            # Create event signature
            event_signature = "_".join(sorted(event_types)) if event_types else "no_events"
            semantic_groups[event_signature].append(i)
        
        # Create semantic clusters
        for cluster_id, pattern_indices in semantic_groups.items():
            if len(pattern_indices) >= 5:  # Higher threshold for semantic clusters
                cluster = PatternCluster(
                    cluster_id=f"semantic_{cluster_id}",
                    pattern_ids=[f"pattern_{i}" for i in pattern_indices],
                    cluster_type="semantic_similarity",
                    centroid_features=self._calculate_cluster_centroid(pattern_indices),
                    temporal_span=self._calculate_temporal_span(pattern_indices),
                    session_coverage=self._calculate_session_coverage(pattern_indices),
                    significance_score=len(pattern_indices) / len(self.patterns)
                )
                clusters.append(cluster)
        
        return clusters
    
    def _validate_clusters(self, clusters: List[PatternCluster]) -> List[PatternCluster]:
        """Validate and filter clusters based on significance"""
        validated = []
        
        for cluster in clusters:
            # Validate cluster significance
            if cluster.significance_score > 0.02:  # At least 2% of patterns
                if len(cluster.pattern_ids) >= 3:  # Minimum cluster size
                    validated.append(cluster)
        
        # Sort by significance
        validated.sort(key=lambda c: c.significance_score, reverse=True)
        
        return validated
    
    def _calculate_cluster_centroid(self, pattern_indices: List[int]) -> Dict:
        """Calculate centroid features for cluster"""
        if not pattern_indices:
            return {}
        
        # Extract common features
        types = [self.patterns[i].get('type', 'unknown') for i in pattern_indices]
        phases = [self.patterns[i].get('phase_information', {}).get('primary_phase', 'unknown') 
                 for i in pattern_indices]
        
        # Calculate averages for numerical features
        positions = [self.patterns[i].get('phase_information', {}).get('session_position', 0) 
                    for i in pattern_indices]
        significances = [self.patterns[i].get('phase_information', {}).get('phase_significance', 0) 
                        for i in pattern_indices]
        
        return {
            'dominant_type': Counter(types).most_common(1)[0][0],
            'dominant_phase': Counter(phases).most_common(1)[0][0],
            'avg_session_position': np.mean([p for p in positions if p > 0]) if any(p > 0 for p in positions) else 0,
            'avg_phase_significance': np.mean([s for s in significances if s > 0]) if any(s > 0 for s in significances) else 0,
            'cluster_size': len(pattern_indices)
        }
    
    def _calculate_temporal_span(self, pattern_indices: List[int]) -> Tuple[str, str]:
        """Calculate temporal span of cluster"""
        # For now, use session timing as proxy
        start_times = []
        end_times = []
        
        for idx in pattern_indices:
            pattern = self.patterns[idx]
            start = pattern.get('session_start', '09:30:00')
            end = pattern.get('session_end', '16:00:00')
            start_times.append(start)
            end_times.append(end)
        
        return (min(start_times), max(end_times))
    
    def _calculate_session_coverage(self, pattern_indices: List[int]) -> Set[str]:
        """Calculate session coverage of cluster"""
        sessions = set()
        
        for idx in pattern_indices:
            pattern = self.patterns[idx]
            session = pattern.get('session_name', 'unknown')
            sessions.add(session)
        
        return sessions
    
    def _generate_analysis_summary(self) -> Dict:
        """Generate comprehensive analysis summary"""
        # Analyze link types
        link_types = Counter(link.link_type for link in self.pattern_links)
        
        # Analyze cluster types  
        cluster_types = Counter(cluster.cluster_type for cluster in self.pattern_clusters)
        
        # Calculate network metrics
        network_density = len(self.pattern_links) / (len(self.patterns) * (len(self.patterns) - 1) / 2) if len(self.patterns) > 1 else 0
        
        return {
            'link_type_distribution': dict(link_types),
            'cluster_type_distribution': dict(cluster_types),
            'network_density': network_density,
            'avg_links_per_pattern': len(self.pattern_links) / len(self.patterns) if self.patterns else 0,
            'top_clusters': [
                {
                    'id': cluster.cluster_id,
                    'type': cluster.cluster_type,
                    'size': len(cluster.pattern_ids),
                    'significance': cluster.significance_score
                }
                for cluster in sorted(self.pattern_clusters, key=lambda c: c.significance_score, reverse=True)[:5]
            ]
        }
    
    def save_results(self, output_path: str = None):
        """Save link discovery results to file"""
        if output_path is None:
            output_path = '/Users/jack/IRONFORGE/analysis/pattern_links_analysis.json'
        
        results = {
            'discovery_metadata': {
                'total_patterns_analyzed': len(self.patterns),
                'analysis_timestamp': datetime.now().isoformat(),
                'total_links_discovered': len(self.pattern_links),
                'total_clusters_identified': len(self.pattern_clusters)
            },
            'pattern_links': [
                {
                    'source': link.source_pattern_id,
                    'target': link.target_pattern_id,
                    'type': link.link_type,
                    'strength': link.strength,
                    'temporal_distance': link.temporal_distance,
                    'structural_similarity': link.structural_similarity,
                    'semantic_overlap': link.semantic_overlap,
                    'session_relationship': link.session_relationship,
                    'evolution_stage': link.evolution_stage
                }
                for link in self.pattern_links
            ],
            'pattern_clusters': [
                {
                    'cluster_id': cluster.cluster_id,
                    'pattern_ids': cluster.pattern_ids,
                    'cluster_type': cluster.cluster_type,
                    'centroid_features': cluster.centroid_features,
                    'temporal_span': cluster.temporal_span,
                    'session_coverage': list(cluster.session_coverage),
                    'significance_score': cluster.significance_score
                }
                for cluster in self.pattern_clusters
            ],
            'analysis_summary': self._generate_analysis_summary()
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Pattern link analysis saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    # Run pattern link discovery analysis
    print("🚀 IRONFORGE Pattern Link Discovery System")
    print("=" * 60)
    
    discovery = PatternLinkDiscovery()
    results = discovery.discover_pattern_links()
    
    # Save comprehensive results
    output_file = discovery.save_results()
    
    print("\n🏛️ Pattern Link Discovery Complete!")
    print(f"📊 Analysis saved to: {output_file}")