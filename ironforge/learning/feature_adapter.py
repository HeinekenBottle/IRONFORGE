"""
IRONFORGE Feature Adapter
========================

Converts shard columns (f0-f44/f45-f50, e0-e19) to TGAT-compatible tensors.
Handles both 45D (standard) and 51D (HTF-enabled) node features as defined in constants.py.
"""

import logging
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import torch

from ironforge.constants import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM_HTF,
    NODE_FEATURE_DIM_STANDARD,
)

logger = logging.getLogger(__name__)


class FeatureAdapter:
    """
    Adapter to convert shard parquet columns to TGAT-compatible tensors.
    
    Handles:
    - Node features: f0-f44 (45D standard) or f0-f50 (51D HTF-enabled)
    - Edge features: e0-e19 (20D standard)
    - Dimension validation and error handling
    """
    
    def __init__(self, htf_enabled: bool = False):
        """
        Initialize feature adapter.
        
        Args:
            htf_enabled: Whether to use HTF (High Timeframe) features (51D vs 45D)
        """
        self.htf_enabled = htf_enabled
        self.node_dim = NODE_FEATURE_DIM_HTF if htf_enabled else NODE_FEATURE_DIM_STANDARD
        self.edge_dim = EDGE_FEATURE_DIM
        
        logger.info(f"FeatureAdapter initialized: node_dim={self.node_dim}, edge_dim={self.edge_dim}")
    
    def validate_shard_dimensions(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> bool:
        """
        Validate that shard dataframes have expected feature columns.
        
        Args:
            nodes_df: Nodes dataframe from shard
            edges_df: Edges dataframe from shard
            
        Returns:
            True if dimensions match expectations, False otherwise
        """
        # Check node features
        expected_node_cols = [f"f{i}" for i in range(self.node_dim)]
        missing_node_cols = [col for col in expected_node_cols if col not in nodes_df.columns]
        
        if missing_node_cols:
            logger.error(f"Missing node feature columns: {missing_node_cols}")
            return False
        
        # Check edge features
        expected_edge_cols = [f"e{i}" for i in range(self.edge_dim)]
        missing_edge_cols = [col for col in expected_edge_cols if col not in edges_df.columns]
        
        if missing_edge_cols:
            logger.error(f"Missing edge feature columns: {missing_edge_cols}")
            return False
        
        logger.debug(f"Shard dimensions validated: {len(nodes_df)} nodes, {len(edges_df)} edges")
        return True
    
    def extract_node_features(self, nodes_df: pd.DataFrame) -> torch.Tensor:
        """
        Extract node features from dataframe and convert to tensor.
        
        Args:
            nodes_df: Nodes dataframe with f0-f44/f50 columns
            
        Returns:
            Tensor of shape [N, node_dim] with node features
        """
        feature_cols = [f"f{i}" for i in range(self.node_dim)]
        # Align columns, insert missing with 0.0, coerce to numeric, replace invalids with 0.0
        features_df = (
            nodes_df.reindex(columns=feature_cols, fill_value=0.0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        features_np = features_df.to_numpy(dtype=np.float32, copy=False)
        return torch.from_numpy(features_np)
    
    def extract_edge_features(self, edges_df: pd.DataFrame) -> torch.Tensor:
        """
        Extract edge features from dataframe and convert to tensor.
        
        Args:
            edges_df: Edges dataframe with e0-e19 columns
            
        Returns:
            Tensor of shape [E, edge_dim] with edge features
        """
        feature_cols = [f"e{i}" for i in range(self.edge_dim)]
        features_df = (
            edges_df.reindex(columns=feature_cols, fill_value=0.0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        features_np = features_df.to_numpy(dtype=np.float32, copy=False)
        return torch.from_numpy(features_np)
    
    def adapt_shard_to_graph(
        self, 
        nodes_df: pd.DataFrame, 
        edges_df: pd.DataFrame
    ) -> nx.Graph:
        """
        Convert shard dataframes to NetworkX graph with TGAT-compatible features.
        
        Args:
            nodes_df: Nodes dataframe from shard
            edges_df: Edges dataframe from shard
            
        Returns:
            NetworkX graph with 'feature' tensors on nodes and edges
        """
        # Validate dimensions
        if not self.validate_shard_dimensions(nodes_df, edges_df):
            raise ValueError("Shard dimensions do not match adapter configuration")
        
        # Create graph
        graph = nx.Graph()
        
        # Add nodes with features
        node_features = self.extract_node_features(nodes_df)
        for i, (_, node_row) in enumerate(nodes_df.iterrows()):
            node_id = node_row['node_id']  # Use actual node_id from parquet
            graph.add_node(
                node_id,
                feature=node_features[i],
                **{k: v for k, v in node_row.items() if not k.startswith('f')}
            )
        
        # Add edges with features
        edge_features = self.extract_edge_features(edges_df)
        for i, (_, edge_row) in enumerate(edges_df.iterrows()):
            source = edge_row['src']  # Use actual src from parquet
            target = edge_row['dst']  # Use actual dst from parquet

            # Calculate temporal distance if dt is available
            dt = edge_row.get('dt')
            temporal_distance = float(dt) / 60000.0 if dt is not None else 1.0

            graph.add_edge(
                source,
                target,
                feature=edge_features[i],
                temporal_distance=temporal_distance,
                **{k: v for k, v in edge_row.items() if not k.startswith('e')}
            )
        
        logger.info(f"Graph adapted: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph
    
    def validate_graph_features(self, graph: nx.Graph) -> bool:
        """
        Validate that graph has properly formatted feature tensors.
        
        Args:
            graph: NetworkX graph to validate
            
        Returns:
            True if all features are valid, False otherwise
        """
        # Check node features
        for node_id, attrs in graph.nodes(data=True):
            if 'feature' not in attrs:
                logger.error(f"Node {node_id} missing feature tensor")
                return False
            
            feature = attrs['feature']
            if not isinstance(feature, torch.Tensor):
                logger.error(f"Node {node_id} feature is not a tensor: {type(feature)}")
                return False
            
            if feature.shape != (self.node_dim,):
                logger.error(f"Node {node_id} feature has wrong shape: {feature.shape}, expected ({self.node_dim},)")
                return False
        
        # Check edge features
        for u, v, attrs in graph.edges(data=True):
            if 'feature' not in attrs:
                logger.error(f"Edge ({u}, {v}) missing feature tensor")
                return False
            
            feature = attrs['feature']
            if not isinstance(feature, torch.Tensor):
                logger.error(f"Edge ({u}, {v}) feature is not a tensor: {type(feature)}")
                return False
            
            if feature.shape != (self.edge_dim,):
                logger.error(f"Edge ({u}, {v}) feature has wrong shape: {feature.shape}, expected ({self.edge_dim},)")
                return False
        
        logger.debug("Graph feature validation passed")
        return True


def create_feature_adapter(htf_enabled: bool = False) -> FeatureAdapter:
    """
    Factory function to create feature adapter.
    
    Args:
        htf_enabled: Whether to use HTF (High Timeframe) features
        
    Returns:
        Configured FeatureAdapter instance
    """
    return FeatureAdapter(htf_enabled=htf_enabled)


def adapt_shard_features(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    htf_enabled: bool = False
) -> nx.Graph:
    """
    Convenience function to adapt shard features to TGAT-compatible graph.
    
    Args:
        nodes_df: Nodes dataframe from shard
        edges_df: Edges dataframe from shard
        htf_enabled: Whether to use HTF features
        
    Returns:
        NetworkX graph with TGAT-compatible features
    """
    adapter = create_feature_adapter(htf_enabled=htf_enabled)
    return adapter.adapt_shard_to_graph(nodes_df, edges_df)
