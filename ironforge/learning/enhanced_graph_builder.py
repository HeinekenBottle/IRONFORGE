"""
Enhanced Graph Builder for 45D/20D architecture
Archaeological discovery graph construction with semantic features
"""

import torch
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class RichNodeFeature:
    """45D node feature vector with semantic preservation"""
    
    def __init__(self):
        # 45D total: 8 semantic + 37 traditional
        self.features = torch.zeros(45, dtype=torch.float32)
        
        # Semantic event flags (first 8 dimensions)
        self.semantic_indices = {
            'fvg_redelivery_flag': 0,
            'expansion_phase_flag': 1, 
            'consolidation_flag': 2,
            'retracement_flag': 3,
            'reversal_flag': 4,
            'liq_sweep_flag': 5,
            'pd_array_interaction_flag': 6,
            'semantic_reserved': 7
        }
        
        # Traditional features (indices 8-44)
        self.traditional_start = 8
    
    def set_semantic_event(self, event_type: str, value: float = 1.0):
        """Set semantic event flag"""
        if event_type in self.semantic_indices:
            self.features[self.semantic_indices[event_type]] = value
    
    def set_traditional_features(self, features: torch.Tensor):
        """Set traditional features (37D)"""
        if features.size(0) == 37:
            self.features[self.traditional_start:] = features
        else:
            logger.warning(f"Expected 37D traditional features, got {features.size(0)}D")

class RichEdgeFeature:
    """20D edge feature vector with semantic relationships"""
    
    def __init__(self):
        # 20D total: 3 semantic + 17 traditional
        self.features = torch.zeros(20, dtype=torch.float32)
        
        # Semantic relationship indices (first 3 dimensions)
        self.semantic_indices = {
            'semantic_event_link': 0,
            'event_causality': 1,
            'relationship_type': 2
        }
        
        # Traditional features (indices 3-19)
        self.traditional_start = 3
    
    def set_semantic_relationship(self, rel_type: str, value: float):
        """Set semantic relationship strength"""
        if rel_type in self.semantic_indices:
            self.features[self.semantic_indices[rel_type]] = value

class EnhancedGraphBuilder:
    """
    Enhanced graph builder for archaeological discovery
    Transforms JSON session data into rich 45D/20D graph representations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Graph Builder initialized")
        
    def build_session_graph(self, session_data: Dict[str, Any]) -> nx.Graph:
        """
        Build enhanced graph from session JSON data
        
        Args:
            session_data: Session JSON with events and metadata
            
        Returns:
            NetworkX graph with rich 45D/20D features
        """
        try:
            graph = nx.Graph()
            
            # Extract session metadata
            session_name = session_data.get('session_name', 'unknown')
            events = session_data.get('events', [])
            
            self.logger.info(f"Building graph for session {session_name} with {len(events)} events")
            
            # Add nodes with rich features
            for i, event in enumerate(events):
                node_feature = self._create_node_feature(event)
                graph.add_node(i, 
                             feature=node_feature.features,
                             raw_data=event,
                             session_name=session_name)
            
            # Add edges with rich features  
            for i in range(len(events)):
                for j in range(i + 1, min(i + 5, len(events))):  # Connect to next 4 events
                    edge_feature = self._create_edge_feature(events[i], events[j])
                    graph.add_edge(i, j, 
                                 feature=edge_feature.features,
                                 temporal_distance=j - i)
            
            self.logger.info(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            self.logger.error(f"Failed to build graph: {e}")
            raise
    
    def _create_node_feature(self, event: Dict[str, Any]) -> RichNodeFeature:
        """Create 45D node feature from event data"""
        feature = RichNodeFeature()
        
        # Set semantic events based on event data
        event_type = event.get('event_type', '')
        semantic_context = event.get('semantic_context', {})
        
        if 'fvg_redelivery' in event_type.lower():
            feature.set_semantic_event('fvg_redelivery_flag', 1.0)
        if 'expansion' in semantic_context.get('phase', '').lower():
            feature.set_semantic_event('expansion_phase_flag', 1.0)
        if 'consolidation' in semantic_context.get('phase', '').lower():
            feature.set_semantic_event('consolidation_flag', 1.0)
        if 'liquidity_sweep' in event_type.lower():
            feature.set_semantic_event('liq_sweep_flag', 1.0)
            
        # Generate traditional features (37D)
        traditional = torch.zeros(37)
        
        # Price features
        price = event.get('price', 0.0)
        volume = event.get('volume', 0.0)
        timestamp = event.get('timestamp', 0)
        
        traditional[0] = float(price)
        traditional[1] = float(volume) 
        traditional[2] = float(timestamp % 86400)  # Time of day
        
        # Fill remaining with derived features
        for i in range(3, 37):
            traditional[i] = torch.randn(1).item() * 0.1  # Placeholder
        
        feature.set_traditional_features(traditional)
        return feature
    
    def _create_edge_feature(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> RichEdgeFeature:
        """Create 20D edge feature from event pair"""
        feature = RichEdgeFeature()
        
        # Set semantic relationships
        type1 = event1.get('event_type', '')
        type2 = event2.get('event_type', '')
        
        # Detect causal relationships
        if 'liquidity_sweep' in type1 and 'redelivery' in type2:
            feature.set_semantic_relationship('event_causality', 0.8)
            feature.set_semantic_relationship('semantic_event_link', 1.0)
        
        # Generate traditional edge features (17D)
        traditional = torch.zeros(17)
        
        # Temporal distance
        time1 = event1.get('timestamp', 0)
        time2 = event2.get('timestamp', 0)
        traditional[0] = abs(time2 - time1)
        
        # Price distance  
        price1 = event1.get('price', 0.0)
        price2 = event2.get('price', 0.0)
        traditional[1] = abs(price2 - price1)
        
        # Fill remaining
        for i in range(2, 17):
            traditional[i] = torch.randn(1).item() * 0.1
            
        feature.features[feature.traditional_start:] = traditional
        return feature
    
    def extract_features_for_tgat(self, graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract feature tensors for TGAT processing
        
        Returns:
            node_features: [N, 45] tensor
            edge_features: [E, 20] tensor  
        """
        nodes = list(graph.nodes())
        edges = list(graph.edges())
        
        # Extract node features
        node_features = torch.stack([
            graph.nodes[node]['feature'] for node in nodes
        ])
        
        # Extract edge features
        edge_features = torch.stack([
            graph.edges[edge]['feature'] for edge in edges  
        ])
        
        self.logger.info(f"Extracted features: nodes {node_features.shape}, edges {edge_features.shape}")
        return node_features, edge_features