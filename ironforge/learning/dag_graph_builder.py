"""
DAG Graph Builder for Dual Graph Views
Extends EnhancedGraphBuilder with directed acyclic graph construction
Maintains temporal causality through (timestamp, seq_idx) ordering
"""

import logging
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.algorithms.dag import is_directed_acyclic_graph, topological_sort

from .enhanced_graph_builder import EnhancedGraphBuilder, RichNodeFeature, RichEdgeFeature
from .m1_event_detector import M1EventDetector, M1Event
from .cross_scale_edge_builder import CrossScaleEdgeBuilder
from ..temporal.archaeological_workflows import (
    compute_archaeological_zones, 
    compute_edge_archaeological_zone_score,
    ArchaeologicalZone
)

logger = logging.getLogger(__name__)


class DAGEdgeFeature:
    """20D DAG edge feature with causal relationship encoding"""
    
    def __init__(self):
        # 20D total: 3 semantic + 17 traditional (same as RichEdgeFeature)
        self.features = torch.zeros(20, dtype=torch.float32)
        
        # Semantic relationship indices (causal-specific)
        self.semantic_indices = {
            "causal_strength": 0,       # How strong is the causal link (0-1)
            "temporal_proximity": 1,    # Temporal closeness (0-1)
            "event_causality": 2,       # Type of causal relationship
        }
        
        # Traditional features start at index 3
        self.traditional_start = 3
        
    def set_causal_relationship(self, rel_type: str, value: float):
        """Set causal relationship strength"""
        if rel_type in self.semantic_indices:
            self.features[self.semantic_indices[rel_type]] = value
            
    def set_traditional_features(self, features: torch.Tensor):
        """Set traditional DAG features (17D)"""
        if features.size(0) == 17:
            self.features[self.traditional_start:] = features
        else:
            logger.warning(f"Expected 17D traditional features, got {features.size(0)}D")


class DAGGraphBuilder(EnhancedGraphBuilder):
    """
    DAG Graph Builder extending Enhanced Graph Builder
    Creates directed acyclic graphs with guaranteed temporal causality
    """
    
    def __init__(self, dag_config: dict = None):
        super().__init__()
        
        # DAG construction parameters
        self.dag_config = dag_config or {
            'k_successors': 4,         # Number of forward connections per node
            'dt_min_minutes': 1,       # Minimum time delta (minutes)
            'dt_max_minutes': 120,     # Maximum time delta (minutes) 
            'enabled': True,           # DAG construction enabled
            'predicate': 'WINDOW_KNN',  # Connection strategy
            'causality_weights': {     # Enhanced Archaeological DAG Weighting config
                'archaeological_zone_influence': 0.85  # Archaeological zone influence factor
            },
            'features': {
                'enable_archaeological_zone_weighting': False  # Flag-gated feature (default safe)
            }
        }
        
        # Initialize M1 components for dual graph views
        self.m1_detector = M1EventDetector()
        self.cross_scale_builder = CrossScaleEdgeBuilder()
        self.m1_integration_enabled = dag_config.get('m1_integration', True) if dag_config else True
        
        self.logger.info(f"DAG Graph Builder initialized: M1 integration {'enabled' if self.m1_integration_enabled else 'disabled'}")
        
    def build_dual_view_graphs(self, session_data: dict[str, Any]) -> Tuple[nx.Graph, nx.DiGraph]:
        """
        Build both undirected temporal and directed DAG views
        
        Args:
            session_data: Session JSON with events and metadata
            
        Returns:
            Tuple of (undirected_graph, dag_graph)
        """
        # Build standard undirected temporal graph
        temporal_graph = self.build_session_graph(session_data)
        
        # Build DAG if enabled
        if self.dag_config.get('enabled', True):
            dag_graph = self.build_dag_from_session(session_data)
        else:
            dag_graph = nx.DiGraph()  # Empty DAG
            
        return temporal_graph, dag_graph
        
    def build_dag_from_session(self, session_data: dict[str, Any]) -> nx.DiGraph:
        """
        Build directed acyclic graph from session data
        Guarantees acyclicity via (timestamp, seq_idx) ordering
        
        Args:
            session_data: Session JSON with events and metadata
            
        Returns:
            NetworkX DiGraph with DAG structure
        """
        try:
            dag = nx.DiGraph()
            
            # Extract and validate session data
            session_name = session_data.get("session_name", "unknown")
            events = session_data.get("events", [])
            
            if not events:
                self.logger.warning(f"No events in session {session_name}")
                return dag
                
            # Sort events by (timestamp, seq_idx) to guarantee acyclicity
            sorted_events = self._sort_events_for_acyclicity(events)
            
            # Extract session context
            session_context = self._extract_session_context(session_data, sorted_events)
            
            self.logger.info(f"Building DAG for session {session_name} with {len(sorted_events)} events")
            
            # Add nodes with same rich features as temporal graph
            for i, event in enumerate(sorted_events):
                node_feature = self._create_node_feature(event, session_context)
                dag.add_node(
                    i,
                    feature=node_feature.features,
                    raw_data=event,
                    session_name=session_name,
                    timestamp=event.get('timestamp', 0),
                    seq_idx=event.get('seq_idx', i)
                )
                
            # Add directed edges with temporal causality
            dag_edges = self._build_dag_edges(sorted_events)
            
            for src, dst, edge_data in dag_edges:
                dag_edge_feature = self._create_dag_edge_feature(
                    sorted_events[src], sorted_events[dst], edge_data
                )
                dag.add_edge(
                    src, dst,
                    feature=dag_edge_feature.features,
                    dt_minutes=edge_data['dt_minutes'],
                    reason=edge_data['reason'],
                    weight=edge_data.get('weight', 1.0)
                )
                
            # Validate DAG properties
            self._validate_dag_properties(dag, session_name)
            
            self.logger.info(
                f"DAG built: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges"
            )
            
            return dag
            
        except Exception as e:
            self.logger.error(f"Failed to build DAG for session {session_name}: {e}")
            raise
            
    def _sort_events_for_acyclicity(self, events: List[dict]) -> List[dict]:
        """
        Sort events by (timestamp, seq_idx) to guarantee DAG acyclicity
        This ordering ensures u→v implies t(u) < t(v) or (t(u) = t(v) and seq(u) < seq(v))
        """
        def sort_key(event):
            timestamp = event.get('timestamp', 0)
            seq_idx = event.get('seq_idx', 0)
            return (timestamp, seq_idx)
            
        return sorted(events, key=sort_key)
        
    def _build_dag_edges(self, sorted_events: List[dict]) -> List[Tuple[int, int, dict]]:
        """
        Build directed edges following DAG constraints
        For each node, connect to next K successors within time window
        """
        edges = []
        k = self.dag_config.get('k_successors', 4)
        dt_min = self.dag_config.get('dt_min_minutes', 1) * 60  # Convert to seconds
        dt_max = self.dag_config.get('dt_max_minutes', 120) * 60
        
        for i, src_event in enumerate(sorted_events):
            src_timestamp = src_event.get('timestamp', 0)
            connections_made = 0
            
            # Look forward for successors (maintaining temporal order)
            for j in range(i + 1, len(sorted_events)):
                if connections_made >= k:
                    break
                    
                dst_event = sorted_events[j]
                dst_timestamp = dst_event.get('timestamp', 0)
                
                dt_seconds = dst_timestamp - src_timestamp
                
                # Check time window constraints
                if dt_min <= dt_seconds <= dt_max:
                    weight = self._calculate_edge_weight(src_event, dst_event, dt_seconds, sorted_events)
                    
                    edge_data = {
                        'dt_minutes': dt_seconds / 60.0,
                        'reason': self.dag_config.get('predicate', 'WINDOW_KNN'),
                        'weight': weight,
                        'temporal_distance': j - i
                    }
                    
                    edges.append((i, j, edge_data))
                    connections_made += 1
                    
                elif dt_seconds > dt_max:
                    # Events are sorted, so no more valid connections for this source
                    break
                    
        return edges
        
    def _calculate_edge_weight(self, src_event: dict, dst_event: dict, dt_seconds: float, sorted_events: List[dict]) -> float:
        """
        Calculate edge weight based on temporal proximity and event similarity
        Enhanced with archaeological zone weighting when enabled
        Higher weight = stronger causal relationship
        """
        # Temporal proximity component (closer in time = higher weight)
        dt_max = self.dag_config.get('dt_max_minutes', 120) * 60
        temporal_weight = 1.0 - (dt_seconds / dt_max)  # 1.0 at dt=0, 0.0 at dt=max
        
        # Event type similarity component
        src_type = src_event.get('event_type', '')
        dst_type = dst_event.get('event_type', '')
        
        # ICT event causality patterns
        causality_weight = self._calculate_ict_causality(src_type, dst_type)
        
        # Price proximity component
        src_price = src_event.get('price', 0)
        dst_price = dst_event.get('price', 0)
        price_diff = abs(dst_price - src_price) if src_price > 0 else float('inf')
        price_weight = 1.0 / (1.0 + price_diff / max(src_price, 1.0))  # Normalized price proximity
        
        # Combined base weight (temporal=0.5, causality=0.3, price=0.2)
        base_weight = (
            0.5 * temporal_weight +
            0.3 * causality_weight +
            0.2 * price_weight
        )
        
        # Apply archaeological zone weighting if enabled (flag-gated)
        final_weight = self._apply_archaeological_zone_weighting(
            base_weight, src_event, dst_event, sorted_events
        )
        
        return max(0.1, min(1.0, final_weight))  # Clamp to [0.1, 1.0]
        
    def _calculate_ict_causality(self, src_type: str, dst_type: str) -> float:
        """
        Calculate ICT-based causality weight between event types
        Based on Inner Circle Trader market structure concepts
        """
        # ICT causality patterns (src → dst likelihood)
        causality_patterns = {
            ('fvg_formation', 'fvg_redelivery'): 0.9,
            ('liquidity_sweep', 'reversal'): 0.8,
            ('expansion', 'retracement'): 0.7,
            ('retracement', 'continuation'): 0.6,
            ('premium', 'discount'): 0.5,
            ('discount', 'premium'): 0.5,
            ('consolidation', 'expansion'): 0.6,
        }
        
        # Check for direct pattern match
        pattern_key = (src_type.lower(), dst_type.lower())
        if pattern_key in causality_patterns:
            return causality_patterns[pattern_key]
            
        # Check for partial matches (substring matching)
        for (src_pattern, dst_pattern), weight in causality_patterns.items():
            if src_pattern in src_type.lower() and dst_pattern in dst_type.lower():
                return weight * 0.7  # Reduced weight for partial match
                
        # Default neutral causality
        return 0.3
        
    def _apply_archaeological_zone_weighting(
        self, 
        base_weight: float, 
        src_event: dict, 
        dst_event: dict, 
        sorted_events: List[dict]
    ) -> float:
        """
        Apply archaeological zone weighting to edge strength (flag-gated feature).
        Multiply edge strength by archaeological zone factor when feature is enabled.
        
        Args:
            base_weight: Base edge weight before zone weighting
            src_event: Source event data
            dst_event: Target event data  
            sorted_events: All events in session for range calculation
            
        Returns:
            Enhanced weight with archaeological zone influence
        """
        # Check if archaeological zone weighting is enabled (flag-gated)
        features_config = self.dag_config.get('features', {})
        zone_weighting_enabled = features_config.get('enable_archaeological_zone_weighting', False)
        
        if not zone_weighting_enabled:
            # Feature disabled - return base weight unchanged (safe default)
            return base_weight
            
        # Extract archaeological configuration
        causality_weights = self.dag_config.get('causality_weights', {})
        zone_influence = causality_weights.get('archaeological_zone_influence', 0.85)
        
        if zone_influence <= 0:
            # Weight disabled (zone_influence=0) - return base weight
            return base_weight
            
        # Calculate session range from sorted events (last-closed HTF only)
        session_range = self._calculate_session_range(sorted_events)
        if not session_range:
            # No valid session range - return base weight
            return base_weight
            
        # Get zone percentages from archaeological config (research-agnostic)
        # Check both archaeological config and fallback to defaults
        archaeological_config = self.dag_config.get('archaeological', {})
        zone_percentages = archaeological_config.get('zone_percentages', [0.236, 0.382, 0.40, 0.618])
        
        # Compute archaeological zones for this session
        archaeological_zones = compute_archaeological_zones(session_range, zone_percentages)
        
        if not archaeological_zones:
            # No valid zones computed - return base weight
            return base_weight
            
        # Compute zone score for the edge (both endpoints)
        zone_score = compute_edge_archaeological_zone_score(
            src_event, dst_event, archaeological_zones, session_range
        )
        
        # Apply zone influence: edge_strength *= zone_score ** archaeological_zone_influence
        zone_factor = zone_score ** zone_influence
        enhanced_weight = base_weight * zone_factor
        
        self.logger.debug(f"Archaeological zone weighting: base={base_weight:.3f}, zone_score={zone_score:.3f}, "
                         f"influence={zone_influence:.2f}, enhanced={enhanced_weight:.3f}")
        
        return enhanced_weight
    
    def _calculate_session_range(self, sorted_events: List[dict]) -> Dict[str, float]:
        """
        Calculate final session range from sorted events (last-closed HTF only).
        
        Args:
            sorted_events: Events sorted by (timestamp, seq_idx)
            
        Returns:
            Dictionary with 'high' and 'low' keys or empty dict if invalid
        """
        if not sorted_events:
            return {}
            
        # Extract all valid prices from events
        prices = []
        for event in sorted_events:
            price = event.get('price', 0)
            if price > 0:  # Valid price
                prices.append(price)
                
        if len(prices) < 2:
            # Need at least 2 prices for a range
            return {}
            
        session_high = max(prices)
        session_low = min(prices)
        
        if session_high <= session_low:
            # Invalid range
            return {}
            
        return {
            'high': session_high,
            'low': session_low
        }
    def _create_dag_edge_feature(self, src_event: dict, dst_event: dict, edge_data: dict) -> DAGEdgeFeature:
        """Create 20D DAG edge feature from event pair"""
        feature = DAGEdgeFeature()
        
        # Set semantic causal relationships
        dt_minutes = edge_data.get('dt_minutes', 0)
        weight = edge_data.get('weight', 0.5)
        
        feature.set_causal_relationship('causal_strength', weight)
        feature.set_causal_relationship('temporal_proximity', 1.0 / (1.0 + dt_minutes / 60.0))
        feature.set_causal_relationship('event_causality', self._encode_causality_type(src_event, dst_event))
        
        # Generate traditional features (17D) - reuse from parent class edge feature logic
        traditional = torch.zeros(17)
        
        # Temporal features
        traditional[0] = dt_minutes  # Time delta
        traditional[1] = edge_data.get('temporal_distance', 0)  # Position distance
        traditional[2] = weight  # Connection strength
        
        # Price movement features
        src_price = src_event.get('price', 0)
        dst_price = dst_event.get('price', 0)
        price_change = (dst_price - src_price) / max(src_price, 1.0) if src_price > 0 else 0.0
        
        traditional[3] = price_change  # Price change %
        traditional[4] = abs(price_change)  # Price change magnitude
        traditional[5] = 1.0 if price_change > 0 else -1.0  # Direction
        
        # Volume relationship
        src_volume = src_event.get('volume', 0)
        dst_volume = dst_event.get('volume', 0)
        volume_ratio = dst_volume / max(src_volume, 1.0) if src_volume > 0 else 1.0
        traditional[6] = volume_ratio
        
        # Fill remaining with computed features...
        for i in range(7, 17):
            traditional[i] = np.random.normal(0, 0.1)  # Placeholder for additional features
            
        feature.set_traditional_features(traditional)
        
        return feature
        

class M1EnhancedNodeFeature(RichNodeFeature):
    """53D node feature vector with M1-derived features (45D base + 8D M1)"""
    
    def __init__(self):
        super().__init__()
        # Extend to 53D: 45D base + 8D M1-derived features
        self.features = torch.zeros(53, dtype=torch.float32)
        
        # Copy over base RichNodeFeature structure (45D)
        self.semantic_indices = {
            "fvg_redelivery_flag": 0,
            "expansion_phase_flag": 1,
            "consolidation_flag": 2,
            "retracement_flag": 3,
            "reversal_flag": 4,
            "liq_sweep_flag": 5,
            "pd_array_interaction_flag": 6,
            "semantic_reserved": 7,
        }
        self.traditional_start = 8
        
        # M1-derived features (indices 45-52) 
        self.m1_feature_indices = {
            "m1_event_density": 45,           # Events per minute in vicinity
            "m1_micro_volatility": 46,        # Micro-timeframe volatility measure
            "m1_coherence_score": 47,         # Cross-timeframe pattern coherence
            "m1_dominant_event_type": 48,     # Encoded dominant M1 event type
            "m1_volume_intensity": 49,        # Volume-weighted event intensity
            "m1_temporal_clustering": 50,     # Event clustering coefficient
            "m1_causality_strength": 51,      # Strength of M1→M5 relationships
            "m1_pattern_confidence": 52,      # Average confidence of M1 events
        }
        
    def set_m1_feature(self, feature_name: str, value: float):
        """Set M1-derived feature value"""
        if feature_name in self.m1_feature_indices:
            self.features[self.m1_feature_indices[feature_name]] = value
        else:
            logger.warning(f"Unknown M1 feature: {feature_name}")
    
    def set_traditional_features(self, features: torch.Tensor):
        """Set traditional features (37D), preserving M1 features"""
        if features.size(0) == 37:
            self.features[self.traditional_start:45] = features
        else:
            logger.warning(f"Expected 37D traditional features, got {features.size(0)}D")


    def _encode_causality_type(self, src_event: dict, dst_event: dict) -> float:
        """Encode the type of causal relationship as a float [0-1]"""
        src_type = src_event.get('event_type', '').lower()
        dst_type = dst_event.get('event_type', '').lower()
        
        # Map causality types to numeric encoding
        if 'fvg' in src_type and 'fvg' in dst_type:
            return 0.9  # FVG causality
        elif 'sweep' in src_type and 'reversal' in dst_type:
            return 0.8  # Liquidity sweep causality
        elif 'expansion' in src_type and 'retracement' in dst_type:
            return 0.7  # Market phase causality
        elif any(term in src_type for term in ['premium', 'discount']) and any(term in dst_type for term in ['premium', 'discount']):
            return 0.6  # PD array causality
        else:
            return 0.4  # Generic temporal causality
            
    def _validate_dag_properties(self, dag: nx.DiGraph, session_name: str):
        """Validate DAG properties and log statistics"""
        if not is_directed_acyclic_graph(dag):
            raise ValueError(f"Generated graph is not a DAG for session {session_name}")
            
        # Calculate DAG statistics
        num_nodes = dag.number_of_nodes()
        num_edges = dag.number_of_edges()
        
        if num_nodes > 0:
            avg_out_degree = num_edges / num_nodes
            
            # Check degree distribution
            out_degrees = [dag.out_degree(n) for n in dag.nodes()]
            max_out_degree = max(out_degrees) if out_degrees else 0
            
            # Calculate temporal span
            timestamps = [dag.nodes[n].get('timestamp', 0) for n in dag.nodes()]
            temporal_span = (max(timestamps) - min(timestamps)) / 60 if timestamps else 0  # Minutes
            
            self.logger.info(
                f"DAG validation passed for {session_name}: "
                f"avg_out_degree={avg_out_degree:.2f}, max_out_degree={max_out_degree}, "
                f"temporal_span={temporal_span:.1f}min"
            )
            
            # Validate topological order exists
            try:
                topo_order = list(topological_sort(dag))
                self.logger.debug(f"Topological order length: {len(topo_order)}")
            except nx.NetworkXError as e:
                raise ValueError(f"Topological sort failed for {session_name}: {e}")
        else:
            self.logger.warning(f"Empty DAG generated for session {session_name}")
            
    def save_dag_edges_parquet(self, dag: nx.DiGraph, output_path: Path, session_id: str = None):
        """
        Save DAG edges to parquet format for efficient storage
        Schema: [src, dst, dt_minutes, reason, weight, features...]
        """
        if dag.number_of_edges() == 0:
            self.logger.warning("No edges to save - DAG is empty")
            return
            
        edges_data = []
        
        for src, dst, edge_data in dag.edges(data=True):
            # Extract edge information
            dt_minutes = edge_data.get('dt_minutes', 0.0)
            reason = edge_data.get('reason', 'UNKNOWN')
            weight = edge_data.get('weight', 1.0)
            features = edge_data.get('feature', torch.zeros(20)).numpy()
            
            # Create record
            record = {
                'src': src,
                'dst': dst, 
                'dt_minutes': dt_minutes,
                'reason': reason,
                'weight': weight,
                'session_id': session_id or 'unknown'
            }
            
            # Add feature columns
            for i, feat_val in enumerate(features):
                record[f'feature_{i:02d}'] = float(feat_val)
                
            edges_data.append(record)
            
        # Convert to DataFrame and save
        df = pd.DataFrame(edges_data)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with ZSTD compression for performance (based on Context7 Parquet docs)
        df.to_parquet(
            output_path,
            compression='zstd',    # High compression ratio
            row_group_size=10000,  # Optimize for read performance
            engine='pyarrow'
        )
        
        self.logger.info(f"Saved {len(edges_data)} DAG edges to {output_path}")
        
    def load_dag_from_parquet(self, edges_path: Path, nodes_path: Path = None) -> nx.DiGraph:
        """
        Load DAG from parquet files
        
        Args:
            edges_path: Path to edges parquet file
            nodes_path: Optional path to nodes parquet file
            
        Returns:
            Reconstructed NetworkX DiGraph
        """
        try:
            # Load edges
            edges_df = pd.read_parquet(edges_path)
            
            dag = nx.DiGraph()
            
            # Add edges from DataFrame
            for _, row in edges_df.iterrows():
                src, dst = int(row['src']), int(row['dst'])
                
                # Reconstruct features tensor
                feature_cols = [col for col in row.index if col.startswith('feature_')]
                features = torch.tensor([row[col] for col in sorted(feature_cols)], dtype=torch.float32)
                
                edge_data = {
                    'dt_minutes': row['dt_minutes'],
                    'reason': row['reason'],
                    'weight': row['weight'],
                    'feature': features
                }
                
                dag.add_edge(src, dst, **edge_data)
                
            # Load nodes if provided
            if nodes_path and nodes_path.exists():
                nodes_df = pd.read_parquet(nodes_path)
                for _, row in nodes_df.iterrows():
                    node_id = int(row['node_id'])
                    
                    # Reconstruct node features
                    feature_cols = [col for col in row.index if col.startswith('feature_')]
                    features = torch.tensor([row[col] for col in sorted(feature_cols)], dtype=torch.float32)
                    
                    node_data = {
                        'feature': features,
                        'session_name': row.get('session_name', 'unknown'),
                        'timestamp': row.get('timestamp', 0),
                        'seq_idx': row.get('seq_idx', node_id)
                    }
                    
                    dag.add_node(node_id, **node_data)
                    
            self.logger.info(f"Loaded DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
            
            return dag
            
        except Exception as e:
            self.logger.error(f"Failed to load DAG from {edges_path}: {e}")
            raise
    
    # === M1 Integration Methods ===
    
    def _create_node_feature(
        self, event: dict[str, Any], session_context: dict[str, Any] = None, m1_events: List[M1Event] = None
    ) -> 'M1EnhancedNodeFeature':
        """Create enhanced 53D node feature with M1-derived features"""
        
        # Start with base 45D features from parent class
        base_feature = super()._create_node_feature(event, session_context)
        
        # Create enhanced feature and copy base features
        enhanced_feature = M1EnhancedNodeFeature()
        enhanced_feature.features[:45] = base_feature.features
        
        # Add M1-derived features if available
        if self.m1_integration_enabled and m1_events:
            m1_features = self._compute_m1_derived_features(event, m1_events, session_context)
            for feature_name, value in m1_features.items():
                enhanced_feature.set_m1_feature(feature_name, value)
        
        return enhanced_feature
    
    def _compute_m1_derived_features(
        self, event: dict[str, Any], m1_events: List[M1Event], session_context: dict[str, Any] = None
    ) -> Dict[str, float]:
        """Compute M1-derived features for M5 event node"""
        
        event_timestamp = event.get('timestamp', 0)
        event_price = event.get('price', 0)
        
        # Time window for M1 event analysis (±5 minutes from M5 event)
        time_window = 5 * 60  # 5 minutes in seconds
        
        # Filter M1 events within time window
        relevant_m1_events = [
            m1_event for m1_event in m1_events 
            if abs(m1_event.timestamp - event_timestamp) <= time_window
        ]
        
        if not relevant_m1_events:
            # Return zero features if no M1 events in vicinity
            return {
                "m1_event_density": 0.0,
                "m1_micro_volatility": 0.0,
                "m1_coherence_score": 0.0,
                "m1_dominant_event_type": 0.0,
                "m1_volume_intensity": 0.0,
                "m1_temporal_clustering": 0.0,
                "m1_causality_strength": 0.0,
                "m1_pattern_confidence": 0.0,
            }
        
        # Calculate M1-derived features
        features = {}
        
        # 1. Event density (events per minute)
        features["m1_event_density"] = len(relevant_m1_events) / (time_window / 60.0)
        
        # 2. Micro-volatility measure
        prices = [m1_event.price for m1_event in relevant_m1_events if m1_event.price > 0]
        if len(prices) > 1:
            price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            features["m1_micro_volatility"] = np.mean(price_changes) * 100  # Percentage
        else:
            features["m1_micro_volatility"] = 0.0
        
        # 3. Cross-timeframe coherence score
        features["m1_coherence_score"] = self._calculate_coherence_score(event, relevant_m1_events)
        
        # 4. Dominant event type (encoded as float)
        features["m1_dominant_event_type"] = self._encode_dominant_m1_event_type(relevant_m1_events)
        
        # 5. Volume-weighted event intensity
        total_volume = sum(getattr(m1_event, 'volume', 1.0) for m1_event in relevant_m1_events)
        avg_confidence = np.mean([m1_event.confidence for m1_event in relevant_m1_events])
        features["m1_volume_intensity"] = total_volume * avg_confidence
        
        # 6. Temporal clustering coefficient
        features["m1_temporal_clustering"] = self._calculate_temporal_clustering(relevant_m1_events)
        
        # 7. M1→M5 causality strength
        features["m1_causality_strength"] = self._calculate_causality_strength(event, relevant_m1_events)
        
        # 8. Average pattern confidence
        features["m1_pattern_confidence"] = avg_confidence
        
        return features
    
    def _calculate_coherence_score(self, m5_event: dict, m1_events: List[M1Event]) -> float:
        """Calculate coherence between M5 event and M1 micro-structure"""
        if not m1_events:
            return 0.0
            
        m5_direction = self._get_event_direction(m5_event)
        if m5_direction == 0:  # Neutral
            return 0.5
            
        # Calculate M1 directional bias
        m1_directions = [self._get_m1_event_direction(m1_event) for m1_event in m1_events]
        m1_bias = np.mean([d for d in m1_directions if d != 0]) if m1_directions else 0
        
        # Coherence = alignment between M5 and M1 directional bias
        coherence = (m5_direction * m1_bias + 1.0) / 2.0  # Normalize to [0,1]
        return max(0.0, min(1.0, coherence))
    
    def _get_event_direction(self, event: dict) -> float:
        """Get directional bias of M5 event: 1.0=bullish, -1.0=bearish, 0.0=neutral"""
        event_type = event.get('event_type', '').lower()
        
        if any(term in event_type for term in ['bullish', 'long', 'buy', 'upward']):
            return 1.0
        elif any(term in event_type for term in ['bearish', 'short', 'sell', 'downward']):
            return -1.0
        else:
            # Use price movement if available
            price_change = event.get('price_change_pct', 0)
            return np.sign(price_change) if abs(price_change) > 0.1 else 0.0
    
    def _get_m1_event_direction(self, m1_event: M1Event) -> float:
        """Get directional bias of M1 event"""
        if m1_event.event_type in ['micro_sweep_up', 'micro_impulse_up', 'imbalance_burst_up']:
            return 1.0
        elif m1_event.event_type in ['micro_sweep_down', 'micro_impulse_down', 'imbalance_burst_down']:
            return -1.0
        else:
            return 0.0  # Neutral events like vwap_touch, wick_extreme
    
    def _encode_dominant_m1_event_type(self, m1_events: List[M1Event]) -> float:
        """Encode dominant M1 event type as float [0-1]"""
        if not m1_events:
            return 0.0
            
        # Count event types
        type_counts = {}
        for event in m1_events:
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
        
        # Find dominant type
        dominant_type = max(type_counts.keys(), key=type_counts.get)
        
        # Encode as float
        type_encoding = {
            'micro_fvg_fill': 0.1,
            'micro_sweep': 0.2,
            'micro_impulse': 0.3,
            'vwap_touch': 0.4,
            'imbalance_burst': 0.5,
            'wick_extreme': 0.6
        }
        
        return type_encoding.get(dominant_type, 0.0)
    
    def _calculate_temporal_clustering(self, m1_events: List[M1Event]) -> float:
        """Calculate temporal clustering coefficient of M1 events"""
        if len(m1_events) < 2:
            return 0.0
            
        # Calculate time deltas between consecutive events
        timestamps = sorted([event.timestamp for event in m1_events])
        deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not deltas:
            return 0.0
            
        # Clustering = inverse of coefficient of variation
        # High clustering = low variation in time deltas
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        
        if mean_delta == 0:
            return 1.0
            
        cv = std_delta / mean_delta
        clustering = 1.0 / (1.0 + cv)  # Normalize to [0,1]
        
        return clustering
    
    def _calculate_causality_strength(self, m5_event: dict, m1_events: List[M1Event]) -> float:
        """Calculate strength of M1→M5 causal relationships"""
        if not m1_events:
            return 0.0
            
        m5_timestamp = m5_event.get('timestamp', 0)
        
        # Count M1 events that precede M5 event (potential causality)
        preceding_events = [
            event for event in m1_events 
            if event.timestamp < m5_timestamp and (m5_timestamp - event.timestamp) <= 300  # 5 minutes
        ]
        
        if not preceding_events:
            return 0.0
            
        # Calculate causality strength based on:
        # 1. Number of preceding events
        # 2. Temporal proximity
        # 3. Pattern coherence
        
        count_factor = min(len(preceding_events) / 5.0, 1.0)  # Normalize by max 5 events
        
        # Temporal proximity factor (closer events have stronger causality)
        proximities = [(m5_timestamp - event.timestamp) / 300.0 for event in preceding_events]
        proximity_factor = np.mean([1.0 - p for p in proximities])  # Invert: closer = stronger
        
        # Pattern coherence factor
        coherence_factor = self._calculate_coherence_score(m5_event, preceding_events)
        
        # Combined causality strength
        causality = (count_factor + proximity_factor + coherence_factor) / 3.0
        
        return causality
    
    def build_dag_with_m1_integration(
        self, session_data: dict[str, Any], m1_ohlcv_data: pd.DataFrame = None
    ) -> nx.DiGraph:
        """
        Build DAG with M1 integration enabled
        
        Args:
            session_data: Session JSON with events and metadata
            m1_ohlcv_data: Optional M1 OHLCV data for event detection
            
        Returns:
            NetworkX DiGraph with M1-enhanced node features
        """
        try:
            # Detect M1 events if data provided
            m1_events = []
            if self.m1_integration_enabled and m1_ohlcv_data is not None:
                session_id = session_data.get("session_name", "unknown")
                m5_bars = session_data.get("bars", pd.DataFrame())  # M5 bars if available
                
                self.logger.info(f"Detecting M1 events for session {session_id}")
                m1_events = self.m1_detector.detect_events(m1_ohlcv_data, session_id, m5_bars)
                self.logger.info(f"Detected {len(m1_events)} M1 events")
            
            # Build DAG with M1-enhanced features
            dag = nx.DiGraph()
            session_name = session_data.get("session_name", "unknown")
            events = session_data.get("events", [])
            
            if not events:
                self.logger.warning(f"No events in session {session_name}")
                return dag
                
            # Sort events for acyclicity
            sorted_events = self._sort_events_for_acyclicity(events)
            session_context = self._extract_session_context(session_data, sorted_events)
            
            self.logger.info(f"Building M1-enhanced DAG for session {session_name} with {len(sorted_events)} events")
            
            # Add nodes with M1-enhanced features
            for i, event in enumerate(sorted_events):
                node_feature = self._create_node_feature(event, session_context, m1_events)
                dag.add_node(
                    i,
                    feature=node_feature.features,  # 53D features
                    raw_data=event,
                    session_name=session_name,
                    timestamp=event.get('timestamp', 0),
                    seq_idx=event.get('seq_idx', i),
                    m1_enhanced=True
                )
            
            # Add directed edges with temporal causality
            dag_edges = self._build_dag_edges(sorted_events)
            for src, dst, edge_data in dag_edges:
                dag_edge_feature = self._create_dag_edge_feature(
                    sorted_events[src], sorted_events[dst], edge_data
                )
                dag.add_edge(
                    src, dst,
                    feature=dag_edge_feature.features,
                    dt_minutes=edge_data['dt_minutes'],
                    reason=edge_data['reason'],
                    weight=edge_data.get('weight', 1.0)
                )
            
            # Validate DAG properties
            self._validate_dag_properties(dag, session_name)
            
            return dag
            
        except Exception as e:
            self.logger.error(f"Failed to build M1-enhanced DAG: {e}")
            raise


def build_dag_edges(nodes: List[dict], dt_min: int = 1, dt_max: int = 120, k: int = 4) -> List[Tuple[int, int, dict]]:
    """
    Standalone function to build DAG edges from node list
    
    Args:
        nodes: List of node dictionaries with timestamp and seq_idx
        dt_min: Minimum time delta in minutes
        dt_max: Maximum time delta in minutes  
        k: Number of forward connections per node
        
    Returns:
        List of (src, dst, edge_data) tuples
    """
    dag_builder = DAGGraphBuilder({
        'k_successors': k,
        'dt_min_minutes': dt_min,
        'dt_max_minutes': dt_max,
        'predicate': 'WINDOW_KNN'
    })
    
    return dag_builder._build_dag_edges(nodes)