"""
Cross-Scale Edge Builder
Creates edges between different timeframe scales (M1 events ↔ M5 bars)
Enables multi-scale graph analysis across temporal hierarchies
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import torch

from .m1_event_detector import M1Event

logger = logging.getLogger(__name__)


@dataclass
class CrossScaleEdge:
    """Cross-scale edge data structure"""
    edge_id: str
    source_id: str
    target_id: str
    source_type: str  # 'M1_EVENT' or 'M5_BAR'
    target_type: str  # 'M1_EVENT' or 'M5_BAR'
    relationship: str  # 'CONTAINED_IN', 'INFLUENCES', 'PRECEDES'
    temporal_distance_ms: int
    strength: float
    features: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class CrossScaleConfig:
    """Configuration for cross-scale edge building"""
    # M1 → M5 containment parameters
    containment_tolerance_ms: int = 30000  # 30 seconds tolerance
    
    # M1 → M1 DAG parameters  
    m1_dag_k: int = 2  # Fewer connections for M1 events
    m1_dag_dt_min_ms: int = 60000   # 1 minute minimum
    m1_dag_dt_max_ms: int = 1200000 # 20 minutes maximum
    
    # Influence edge parameters
    influence_window_ms: int = 300000  # 5 minutes
    min_influence_strength: float = 0.2
    
    # Feature engineering
    enable_temporal_features: bool = True
    enable_volume_features: bool = True
    enable_price_features: bool = True


class CrossScaleEdgeBuilder:
    """
    Cross-Scale Edge Builder for Multi-Timeframe Graph Analysis
    
    Creates three types of edges:
    1. M1_EVENT → M5_BAR (CONTAINED_IN): Events contained within bars
    2. M1_EVENT → M1_EVENT (DAG): Directed temporal relationships
    3. M1_EVENT → M5_BAR (INFLUENCES): Events that influence subsequent bars
    """
    
    def __init__(self, config: CrossScaleConfig = None):
        self.config = config or CrossScaleConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Cross-Scale Edge Builder initialized: {self.config}")
        
    def build_cross_scale_edges(
        self,
        m1_events: List[M1Event],
        m5_bars: pd.DataFrame,
        session_id: str
    ) -> List[CrossScaleEdge]:
        """
        Build all cross-scale edges for a session
        
        Args:
            m1_events: List of M1 events
            m5_bars: DataFrame of M5 bars with columns [seq_idx, timestamp, open, high, low, close, volume]
            session_id: Session identifier
            
        Returns:
            List of CrossScaleEdge objects
        """
        all_edges = []
        
        self.logger.info(f"Building cross-scale edges: {len(m1_events)} M1 events, {len(m5_bars)} M5 bars")
        
        # 1. Build M1 → M5 containment edges
        containment_edges = self._build_containment_edges(m1_events, m5_bars, session_id)
        all_edges.extend(containment_edges)
        self.logger.debug(f"Built {len(containment_edges)} containment edges")
        
        # 2. Build M1 → M1 DAG edges
        m1_dag_edges = self._build_m1_dag_edges(m1_events, session_id)
        all_edges.extend(m1_dag_edges)
        self.logger.debug(f"Built {len(m1_dag_edges)} M1 DAG edges")
        
        # 3. Build M1 → M5 influence edges
        influence_edges = self._build_influence_edges(m1_events, m5_bars, session_id)
        all_edges.extend(influence_edges)
        self.logger.debug(f"Built {len(influence_edges)} influence edges")
        
        self.logger.info(f"Total cross-scale edges built: {len(all_edges)}")
        
        return all_edges
        
    def _build_containment_edges(
        self,
        m1_events: List[M1Event],
        m5_bars: pd.DataFrame,
        session_id: str
    ) -> List[CrossScaleEdge]:
        """Build M1_EVENT → M5_BAR containment edges"""
        edges = []
        
        # Create timestamp index for M5 bars for efficient lookup
        m5_timestamps = {}
        for idx, row in m5_bars.iterrows():
            m5_start = row['timestamp']
            m5_end = m5_start + 300000  # 5 minutes in milliseconds
            m5_timestamps[idx] = (m5_start, m5_end, row)
            
        for event in m1_events:
            event_timestamp = event.timestamp_ms
            
            # Find containing M5 bar
            containing_bar = None
            containing_idx = None
            
            for m5_idx, (m5_start, m5_end, m5_row) in m5_timestamps.items():
                # Check if event falls within M5 bar timeframe (with tolerance)
                if (m5_start - self.config.containment_tolerance_ms <= 
                    event_timestamp < 
                    m5_end + self.config.containment_tolerance_ms):
                    containing_bar = m5_row
                    containing_idx = m5_idx
                    break
                    
            if containing_bar is not None:
                # Calculate containment features
                temporal_offset = event_timestamp - containing_bar['timestamp']
                relative_position = temporal_offset / 300000.0  # Position within 5-min bar (0-1)
                
                # Price position within M5 bar
                m5_range = containing_bar['high'] - containing_bar['low']
                if m5_range > 0:
                    price_position = (event.price - containing_bar['low']) / m5_range
                else:
                    price_position = 0.5
                    
                # Volume relationship
                volume_ratio = event.volume / max(containing_bar['volume'], 1.0)
                
                features = {
                    'temporal_offset_ms': float(temporal_offset),
                    'relative_position': relative_position,
                    'price_position': price_position,
                    'volume_ratio': volume_ratio,
                    'event_confidence': event.confidence
                }
                
                # Add event-specific features
                for feat_name, feat_value in event.features.items():
                    features[f'event_{feat_name}'] = feat_value
                    
                edge = CrossScaleEdge(
                    edge_id=f"{session_id}_contain_{event.event_id}_to_{containing_idx}",
                    source_id=event.event_id,
                    target_id=f"{session_id}_m5_{containing_idx}",
                    source_type='M1_EVENT',
                    target_type='M5_BAR',
                    relationship='CONTAINED_IN',
                    temporal_distance_ms=abs(temporal_offset),
                    strength=1.0 - abs(relative_position - 0.5),  # Stronger if closer to center
                    features=features,
                    metadata={
                        'containing_bar_idx': containing_idx,
                        'event_type': event.event_kind,
                        'm5_bar_range_pips': m5_range * 10000
                    }
                )
                edges.append(edge)
                
        return edges
        
    def _build_m1_dag_edges(
        self,
        m1_events: List[M1Event], 
        session_id: str
    ) -> List[CrossScaleEdge]:
        """Build M1_EVENT → M1_EVENT directed edges"""
        edges = []
        
        # Sort events by timestamp for DAG construction
        sorted_events = sorted(m1_events, key=lambda e: e.timestamp_ms)
        
        for i, src_event in enumerate(sorted_events):
            connections_made = 0
            
            # Look forward for successor events
            for j in range(i + 1, len(sorted_events)):
                if connections_made >= self.config.m1_dag_k:
                    break
                    
                dst_event = sorted_events[j]
                dt_ms = dst_event.timestamp_ms - src_event.timestamp_ms
                
                # Check temporal window constraints
                if self.config.m1_dag_dt_min_ms <= dt_ms <= self.config.m1_dag_dt_max_ms:
                    
                    # Calculate edge strength
                    strength = self._calculate_m1_edge_strength(src_event, dst_event, dt_ms)
                    
                    if strength > 0.1:  # Only create meaningful connections
                        
                        features = self._create_m1_edge_features(src_event, dst_event, dt_ms)
                        
                        edge = CrossScaleEdge(
                            edge_id=f"{session_id}_m1dag_{src_event.event_id}_to_{dst_event.event_id}",
                            source_id=src_event.event_id,
                            target_id=dst_event.event_id,
                            source_type='M1_EVENT',
                            target_type='M1_EVENT',
                            relationship='PRECEDES',
                            temporal_distance_ms=dt_ms,
                            strength=strength,
                            features=features,
                            metadata={
                                'src_event_type': src_event.event_kind,
                                'dst_event_type': dst_event.event_kind,
                                'causality_type': self._classify_event_causality(src_event, dst_event)
                            }
                        )
                        edges.append(edge)
                        connections_made += 1
                        
                elif dt_ms > self.config.m1_dag_dt_max_ms:
                    # Events are sorted, so no more valid connections
                    break
                    
        return edges
        
    def _build_influence_edges(
        self,
        m1_events: List[M1Event],
        m5_bars: pd.DataFrame,
        session_id: str
    ) -> List[CrossScaleEdge]:
        """Build M1_EVENT → M5_BAR influence edges"""
        edges = []
        
        for event in m1_events:
            event_timestamp = event.timestamp_ms
            
            # Look for M5 bars that occur after this event (within influence window)
            for idx, m5_bar in m5_bars.iterrows():
                m5_timestamp = m5_bar['timestamp']
                dt_ms = m5_timestamp - event_timestamp
                
                # Check if M5 bar is within influence window
                if 0 < dt_ms <= self.config.influence_window_ms:
                    
                    # Calculate influence strength
                    influence_strength = self._calculate_influence_strength(event, m5_bar, dt_ms)
                    
                    if influence_strength >= self.config.min_influence_strength:
                        
                        features = self._create_influence_features(event, m5_bar, dt_ms)
                        
                        edge = CrossScaleEdge(
                            edge_id=f"{session_id}_influence_{event.event_id}_to_{idx}",
                            source_id=event.event_id,
                            target_id=f"{session_id}_m5_{idx}",
                            source_type='M1_EVENT',
                            target_type='M5_BAR',
                            relationship='INFLUENCES',
                            temporal_distance_ms=dt_ms,
                            strength=influence_strength,
                            features=features,
                            metadata={
                                'event_type': event.event_kind,
                                'influence_type': self._classify_influence_type(event, m5_bar),
                                'm5_bar_idx': idx
                            }
                        )
                        edges.append(edge)
                        
        return edges
        
    def _calculate_m1_edge_strength(self, src_event: M1Event, dst_event: M1Event, dt_ms: int) -> float:
        """Calculate edge strength between two M1 events"""
        
        # Temporal proximity component (closer = stronger)
        temporal_weight = 1.0 - (dt_ms / self.config.m1_dag_dt_max_ms)
        
        # Event type compatibility
        causality_weight = self._get_event_causality_weight(src_event.event_kind, dst_event.event_kind)
        
        # Confidence product
        confidence_weight = src_event.confidence * dst_event.confidence
        
        # Price proximity component
        price_diff = abs(dst_event.price - src_event.price)
        price_weight = 1.0 / (1.0 + price_diff / max(src_event.price, 1.0))
        
        # Combined strength
        strength = (
            0.4 * temporal_weight +
            0.3 * causality_weight +
            0.2 * confidence_weight +
            0.1 * price_weight
        )
        
        return max(0.0, min(1.0, strength))
        
    def _get_event_causality_weight(self, src_type: str, dst_type: str) -> float:
        """Get causality weight between event types"""
        
        # Define causality patterns for M1 events
        causality_patterns = {
            ('micro_fvg_formation', 'micro_fvg_fill'): 0.9,
            ('micro_sweep', 'micro_impulse'): 0.8,
            ('vwap_touch', 'micro_impulse'): 0.7,
            ('imbalance_burst', 'wick_extreme'): 0.6,
            ('micro_impulse', 'vwap_touch'): 0.5,
            ('wick_extreme', 'micro_sweep'): 0.4,
        }
        
        # Check direct patterns
        pattern_key = (src_type, dst_type)
        if pattern_key in causality_patterns:
            return causality_patterns[pattern_key]
            
        # Same event type has moderate causality
        if src_type == dst_type:
            return 0.4
            
        # Default weak causality
        return 0.2
        
    def _classify_event_causality(self, src_event: M1Event, dst_event: M1Event) -> str:
        """Classify the type of causality between events"""
        
        if src_event.event_kind == 'micro_fvg_formation' and dst_event.event_kind == 'micro_fvg_fill':
            return 'fvg_completion'
        elif src_event.event_kind == 'micro_sweep' and dst_event.event_kind == 'micro_impulse':
            return 'sweep_reaction'
        elif src_event.event_kind == 'vwap_touch' and dst_event.event_kind == 'micro_impulse':
            return 'level_rejection'
        elif abs(src_event.price - dst_event.price) / max(src_event.price, 1.0) > 0.001:  # 10 pips
            return 'price_continuation'
        else:
            return 'temporal_sequence'
            
    def _calculate_influence_strength(self, event: M1Event, m5_bar: pd.Series, dt_ms: int) -> float:
        """Calculate how much an M1 event influences a subsequent M5 bar"""
        
        # Temporal decay (influence weakens over time)
        temporal_factor = 1.0 - (dt_ms / self.config.influence_window_ms)
        
        # Event confidence factor
        confidence_factor = event.confidence
        
        # Price impact factor (did the M5 bar move in the direction suggested by the event?)
        price_impact_factor = self._calculate_price_impact(event, m5_bar)
        
        # Volume relationship factor
        volume_factor = min(1.0, event.volume / max(m5_bar['volume'], 1.0))
        
        # Event type influence weights
        event_influence_weights = {
            'micro_sweep': 0.8,
            'micro_impulse': 0.9,
            'micro_fvg_fill': 0.7,
            'imbalance_burst': 0.6,
            'vwap_touch': 0.5,
            'wick_extreme': 0.4
        }
        
        event_weight = event_influence_weights.get(event.event_kind, 0.3)
        
        # Combined influence strength
        influence = (
            0.3 * temporal_factor +
            0.2 * confidence_factor +
            0.3 * price_impact_factor +
            0.1 * volume_factor +
            0.1 * event_weight
        )
        
        return max(0.0, min(1.0, influence))
        
    def _calculate_price_impact(self, event: M1Event, m5_bar: pd.Series) -> float:
        """Calculate price impact score"""
        
        # Get event direction from features
        event_direction = 0.0
        
        if 'sweep_direction' in event.features:
            event_direction = event.features['sweep_direction']
        elif 'impulse_direction' in event.features:
            event_direction = event.features['impulse_direction']
        elif 'imbalance_direction' in event.features:
            event_direction = event.features['imbalance_direction']
        elif 'gap_direction' in event.features:
            event_direction = event.features['gap_direction']
        else:
            # Infer direction from price relative to event
            if m5_bar['close'] > event.price:
                event_direction = 1.0
            elif m5_bar['close'] < event.price:
                event_direction = -1.0
            else:
                event_direction = 0.0
                
        # Calculate M5 bar direction
        m5_direction = 1.0 if m5_bar['close'] > m5_bar['open'] else -1.0
        
        # Direction alignment score
        if event_direction * m5_direction > 0:
            return 0.8  # Strong alignment
        elif event_direction * m5_direction < 0:
            return 0.2  # Opposite direction
        else:
            return 0.5  # Neutral
            
    def _classify_influence_type(self, event: M1Event, m5_bar: pd.Series) -> str:
        """Classify the type of influence"""
        
        price_change = (m5_bar['close'] - m5_bar['open']) / m5_bar['open'] * 100
        
        if event.event_kind == 'micro_sweep':
            if abs(price_change) > 0.05:  # 5 pips
                return 'sweep_momentum'
            else:
                return 'sweep_exhaustion'
        elif event.event_kind == 'micro_impulse':
            if abs(price_change) > 0.03:  # 3 pips
                return 'impulse_continuation'
            else:
                return 'impulse_pause'
        elif event.event_kind == 'vwap_touch':
            return 'level_influence'
        else:
            return 'general_influence'
            
    def _create_m1_edge_features(self, src_event: M1Event, dst_event: M1Event, dt_ms: int) -> Dict[str, float]:
        """Create feature dictionary for M1 → M1 edges"""
        
        features = {
            'dt_minutes': dt_ms / 60000.0,
            'dt_seconds': dt_ms / 1000.0,
            'price_change': dst_event.price - src_event.price,
            'price_change_pct': (dst_event.price - src_event.price) / max(src_event.price, 1.0) * 100,
            'volume_ratio': dst_event.volume / max(src_event.volume, 1.0),
            'confidence_product': src_event.confidence * dst_event.confidence,
            'confidence_avg': (src_event.confidence + dst_event.confidence) / 2,
            'same_event_type': 1.0 if src_event.event_kind == dst_event.event_kind else 0.0
        }
        
        # Add event-specific feature interactions
        for src_feat, src_val in src_event.features.items():
            for dst_feat, dst_val in dst_event.features.items():
                if src_feat == dst_feat:
                    features[f'feature_diff_{src_feat}'] = dst_val - src_val
                    
        return features
        
    def _create_influence_features(self, event: M1Event, m5_bar: pd.Series, dt_ms: int) -> Dict[str, float]:
        """Create feature dictionary for M1 → M5 influence edges"""
        
        m5_range = m5_bar['high'] - m5_bar['low']
        m5_body = abs(m5_bar['close'] - m5_bar['open'])
        
        features = {
            'dt_minutes': dt_ms / 60000.0,
            'event_confidence': event.confidence,
            'price_distance': abs(event.price - m5_bar['open']),
            'price_distance_pct': abs(event.price - m5_bar['open']) / max(m5_bar['open'], 1.0) * 100,
            'volume_ratio': event.volume / max(m5_bar['volume'], 1.0),
            'm5_range_pips': m5_range * 10000,
            'm5_body_pips': m5_body * 10000,
            'm5_direction': 1.0 if m5_bar['close'] > m5_bar['open'] else -1.0,
            'event_in_m5_range': 1.0 if m5_bar['low'] <= event.price <= m5_bar['high'] else 0.0
        }
        
        # Add event features
        for feat_name, feat_value in event.features.items():
            features[f'event_{feat_name}'] = feat_value
            
        return features
        
    def save_cross_scale_edges_parquet(
        self, 
        edges: List[CrossScaleEdge], 
        output_path: Path
    ):
        """Save cross-scale edges to parquet format"""
        
        if not edges:
            self.logger.warning("No cross-scale edges to save")
            return
            
        records = []
        
        for edge in edges:
            record = {
                'edge_id': edge.edge_id,
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'source_type': edge.source_type,
                'target_type': edge.target_type,
                'relationship': edge.relationship,
                'temporal_distance_ms': edge.temporal_distance_ms,
                'strength': edge.strength
            }
            
            # Add features as columns
            for feat_name, feat_value in edge.features.items():
                record[f'feature_{feat_name}'] = feat_value
                
            # Add metadata as JSON
            import json
            record['metadata'] = json.dumps(edge.metadata)
            
            records.append(record)
            
        df = pd.DataFrame(records)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with ZSTD compression
        df.to_parquet(
            output_path,
            compression='zstd',
            row_group_size=10000,
            engine='pyarrow'
        )
        
        self.logger.info(f"Saved {len(edges)} cross-scale edges to {output_path}")
        
    def create_mixed_scale_graph(
        self,
        m1_events: List[M1Event],
        m5_bars: pd.DataFrame,
        cross_scale_edges: List[CrossScaleEdge],
        session_id: str
    ) -> nx.MultiDiGraph:
        """
        Create a unified multi-scale graph containing both M1 events and M5 bars
        
        Returns:
            NetworkX MultiDiGraph with both node types and cross-scale edges
        """
        
        graph = nx.MultiDiGraph()
        
        # Add M1 event nodes
        for event in m1_events:
            node_attrs = {
                'node_type': 'M1_EVENT',
                'event_kind': event.event_kind,
                'timestamp_ms': event.timestamp_ms,
                'price': event.price,
                'volume': event.volume,
                'confidence': event.confidence,
                'session_id': session_id
            }
            
            # Add event features as node attributes
            for feat_name, feat_value in event.features.items():
                node_attrs[f'event_{feat_name}'] = feat_value
                
            graph.add_node(event.event_id, **node_attrs)
            
        # Add M5 bar nodes
        for idx, m5_bar in m5_bars.iterrows():
            node_id = f"{session_id}_m5_{idx}"
            
            node_attrs = {
                'node_type': 'M5_BAR',
                'seq_idx': idx,
                'timestamp_ms': m5_bar['timestamp'],
                'open': m5_bar['open'],
                'high': m5_bar['high'],
                'low': m5_bar['low'],
                'close': m5_bar['close'],
                'volume': m5_bar['volume'],
                'session_id': session_id
            }
            
            graph.add_node(node_id, **node_attrs)
            
        # Add cross-scale edges
        for edge in cross_scale_edges:
            edge_attrs = {
                'edge_type': f"{edge.source_type}_to_{edge.target_type}",
                'relationship': edge.relationship,
                'temporal_distance_ms': edge.temporal_distance_ms,
                'strength': edge.strength
            }
            
            # Add edge features
            for feat_name, feat_value in edge.features.items():
                edge_attrs[f'edge_{feat_name}'] = feat_value
                
            graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge.relationship,  # Allow multiple edge types between same nodes
                **edge_attrs
            )
            
        self.logger.info(
            f"Created mixed-scale graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        
        return graph


def build_session_cross_scale_edges(
    m1_events_path: Path,
    m5_bars_path: Path,
    session_id: str,
    output_dir: Path,
    config: CrossScaleConfig = None
) -> List[CrossScaleEdge]:
    """
    Convenience function to build cross-scale edges for a session
    
    Args:
        m1_events_path: Path to M1 events parquet file
        m5_bars_path: Path to M5 bars parquet/CSV file
        session_id: Session identifier
        output_dir: Output directory for results
        config: Cross-scale configuration
        
    Returns:
        List of CrossScaleEdge objects
    """
    
    config = config or CrossScaleConfig()
    builder = CrossScaleEdgeBuilder(config)
    
    # Load M1 events
    try:
        m1_events_df = pd.read_parquet(m1_events_path)
        
        # Convert DataFrame back to M1Event objects
        m1_events = []
        for _, row in m1_events_df.iterrows():
            
            # Extract features from columns
            features = {}
            for col in row.index:
                if col.startswith('feature_'):
                    feat_name = col.replace('feature_', '')
                    features[feat_name] = row[col]
                    
            # Parse metadata if available
            metadata = {}
            if 'metadata' in row and pd.notna(row['metadata']):
                try:
                    import json
                    metadata = json.loads(row['metadata'])
                except:
                    pass
                    
            event = M1Event(
                event_id=row['event_id'],
                session_id=row['session_id'],
                timestamp_ms=row['timestamp_ms'],
                parent_m5_seq_idx=row.get('parent_m5_seq_idx', -1),
                event_kind=row['event_kind'],
                price=row['price'],
                volume=row['volume'],
                features=features,
                confidence=row['confidence'],
                metadata=metadata
            )
            m1_events.append(event)
            
    except Exception as e:
        logger.error(f"Failed to load M1 events from {m1_events_path}: {e}")
        return []
        
    # Load M5 bars
    try:
        if m5_bars_path.suffix.lower() == '.parquet':
            m5_bars = pd.read_parquet(m5_bars_path)
        else:
            m5_bars = pd.read_csv(m5_bars_path)
    except Exception as e:
        logger.error(f"Failed to load M5 bars from {m5_bars_path}: {e}")
        return []
        
    # Build edges
    cross_scale_edges = builder.build_cross_scale_edges(m1_events, m5_bars, session_id)
    
    # Save results
    if cross_scale_edges:
        output_path = output_dir / f"{session_id}_cross_scale_edges.parquet"
        builder.save_cross_scale_edges_parquet(cross_scale_edges, output_path)
        
        # Also create and save the mixed-scale graph
        mixed_graph = builder.create_mixed_scale_graph(
            m1_events, m5_bars, cross_scale_edges, session_id
        )
        
        graph_path = output_dir / f"{session_id}_mixed_scale_graph.pkl"
        import pickle
        with open(graph_path, 'wb') as f:
            pickle.dump(mixed_graph, f)
            
        logger.info(f"Saved mixed-scale graph to {graph_path}")
        
    return cross_scale_edges


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Build cross-scale edges")
    parser.add_argument("--m1-events", required=True, help="Path to M1 events parquet")
    parser.add_argument("--m5-bars", required=True, help="Path to M5 bars data")
    parser.add_argument("--session-id", required=True, help="Session identifier")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--influence-window", type=int, default=300000, help="Influence window in milliseconds")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure cross-scale building
    config = CrossScaleConfig(influence_window_ms=args.influence_window)
    
    # Build cross-scale edges
    edges = build_session_cross_scale_edges(
        Path(args.m1_events),
        Path(args.m5_bars),
        args.session_id,
        Path(args.output_dir),
        config
    )
    
    print(f"Built {len(edges)} cross-scale edges")
    
    # Print edge type summary
    edge_type_counts = {}
    for edge in edges:
        edge_type = edge.relationship
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
    for edge_type, count in sorted(edge_type_counts.items()):
        print(f"- {edge_type}: {count}")