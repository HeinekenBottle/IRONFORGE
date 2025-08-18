"""
TGAT Discovery Engine for Archaeological Pattern Discovery
Temporal Graph Attention Network for market pattern archaeology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TemporalAttentionLayer(nn.Module):
    """Multi-head temporal attention for 45D node features"""

    def __init__(self, input_dim=45, hidden_dim=44, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Project 45D to 44D for multi-head processing
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Multi-head attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Temporal encoding
        self.temporal_encoding = nn.Linear(1, hidden_dim)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        logger.info(f"Temporal Attention Layer: {input_dim}D -> {hidden_dim}D, {num_heads} heads")

    def forward(self, node_features, edge_features, temporal_distances):
        """
        Forward pass with temporal attention

        Args:
            node_features: [N, 45] node feature tensor
            edge_features: [E, 20] edge feature tensor
            temporal_distances: [E, 1] temporal distance tensor

        Returns:
            attended_features: [N, 44] attended node features
            attention_weights: [N, N] attention weight matrix
        """
        batch_size, seq_len = node_features.size(0), node_features.size(0)

        # Project 45D to 44D
        projected_features = self.input_projection(node_features)  # [N, 44]

        # Generate Q, K, V
        Q = self.query(projected_features)  # [N, 44]
        K = self.key(projected_features)  # [N, 44]
        V = self.value(projected_features)  # [N, 44]

        # Reshape for multi-head attention
        Q = Q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)  # [4, N, 11]
        K = K.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)  # [4, N, 11]
        V = V.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)  # [4, N, 11]

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(
            self.head_dim
        )  # [4, N, N]

        # Apply temporal encoding if temporal distances provided
        if temporal_distances is not None and len(temporal_distances) > 0:
            temporal_encoding = self.temporal_encoding(
                temporal_distances.float().unsqueeze(-1)
            )  # [E, 44]
            # This is a simplified temporal bias - in practice would be more sophisticated
            temporal_bias = torch.zeros(seq_len, seq_len, device=node_features.device)
            attention_scores = attention_scores + temporal_bias.unsqueeze(0)  # [4, N, N]

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # [4, N, N]

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [4, N, 11]

        # Concatenate heads
        attended = attended.transpose(0, 1).contiguous().view(seq_len, -1)  # [N, 44]

        # Output projection
        output = self.output_projection(attended)  # [N, 44]

        # Return average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=0)  # [N, N]

        return output, avg_attention


class IRONFORGEDiscovery(nn.Module):
    """
    IRONFORGE Discovery Engine using TGAT
    Archaeological pattern discovery through temporal graph attention
    """

    def __init__(self, node_dim=45, edge_dim=20, hidden_dim=44, num_layers=2):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Temporal attention layers
        self.attention_layers = nn.ModuleList(
            [
                TemporalAttentionLayer(node_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

        # Edge processing network
        self.edge_processor = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # Pattern discovery head
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 16),  # 16 pattern types
            nn.Sigmoid(),
        )

        # Archaeological significance scorer
        self.significance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"IRONFORGE Discovery initialized: {node_dim}D nodes, {edge_dim}D edges, {num_layers} layers"
        )

    def forward(self, graph: nx.Graph) -> Dict[str, torch.Tensor]:
        """
        Discover archaeological patterns in session graph

        Args:
            graph: NetworkX graph with rich features

        Returns:
            Dictionary containing:
            - pattern_scores: [N, 16] pattern type probabilities
            - attention_weights: [N, N] attention weight matrix
            - significance_scores: [N, 1] archaeological significance
            - node_embeddings: [N, 44] final node embeddings
        """
        try:
            # Extract features from graph
            nodes = list(graph.nodes())
            edges = list(graph.edges())

            if len(nodes) == 0:
                logger.warning("Empty graph provided")
                return self._empty_result()

            # Get node features [N, 45]
            node_features = torch.stack([graph.nodes[node]["feature"] for node in nodes])

            # Get edge features [E, 20]
            edge_features = (
                torch.stack([graph.edges[edge]["feature"] for edge in edges])
                if edges
                else torch.zeros(0, self.edge_dim)
            )

            # Get temporal distances
            temporal_distances = (
                torch.tensor([graph.edges[edge].get("temporal_distance", 1.0) for edge in edges])
                if edges
                else torch.zeros(0)
            )

            logger.info(f"Processing graph: {len(nodes)} nodes, {len(edges)} edges")

            # Apply temporal attention layers
            current_features = node_features
            attention_weights = None

            for i, layer in enumerate(self.attention_layers):
                current_features, attention_weights = layer(
                    current_features, edge_features, temporal_distances
                )
                logger.debug(f"Layer {i+1} output shape: {current_features.shape}")

            # Discover patterns
            pattern_scores = self.pattern_classifier(current_features)  # [N, 16]

            # Score archaeological significance
            significance_scores = self.significance_scorer(current_features)  # [N, 1]

            # Prepare results
            results = {
                "pattern_scores": pattern_scores,
                "attention_weights": attention_weights,
                "significance_scores": significance_scores,
                "node_embeddings": current_features,
                "session_name": nodes[0] if nodes else "unknown",
            }

            logger.info(
                f"Discovery complete: found {(pattern_scores > 0.5).sum().item()} significant patterns"
            )
            return results

        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return self._empty_result()

    def _empty_result(self) -> Dict[str, torch.Tensor]:
        """Return empty result structure"""
        return {
            "pattern_scores": torch.zeros(0, 16),
            "attention_weights": torch.zeros(0, 0),
            "significance_scores": torch.zeros(0, 1),
            "node_embeddings": torch.zeros(0, self.hidden_dim),
            "session_name": "empty",
        }

    def discover_session_patterns(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-level interface for session pattern discovery

        Args:
            session_data: Session JSON data

        Returns:
            Archaeological discovery results with interpretable patterns
        """
        try:
            # Build graph from session data
            from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

            graph_builder = EnhancedGraphBuilder()
            graph = graph_builder.build_session_graph(session_data)

            # Discover patterns
            results = self.forward(graph)

            # Interpret results
            interpreted_results = self._interpret_discoveries(results, session_data)

            return interpreted_results

        except Exception as e:
            logger.error(f"Session discovery failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "session_name": session_data.get("session_name", "unknown"),
            }

    def _interpret_discoveries(
        self, results: Dict[str, torch.Tensor], session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Interpret neural network outputs into archaeological insights"""

        session_name = session_data.get("session_name", "unknown")

        # Extract significant patterns (threshold > 0.5)
        pattern_scores = results["pattern_scores"]
        significance_scores = results["significance_scores"]
        attention_weights = results["attention_weights"]

        significant_patterns = []
        if pattern_scores.size(0) > 0:
            significant_mask = (pattern_scores > 0.5).any(dim=1)
            significant_indices = torch.where(significant_mask)[0]

            for idx in significant_indices:
                pattern_types = torch.where(pattern_scores[idx] > 0.5)[0]
                significance = significance_scores[idx].item()

                significant_patterns.append(
                    {
                        "event_index": idx.item(),
                        "pattern_types": pattern_types.tolist(),
                        "pattern_scores": pattern_scores[idx].tolist(),
                        "archaeological_significance": significance,
                        "attention_received": (
                            attention_weights[idx].sum().item()
                            if attention_weights.size(0) > 0
                            else 0.0
                        ),
                    }
                )

        # Calculate session-level metrics
        session_metrics = {
            "total_events": pattern_scores.size(0),
            "significant_patterns": len(significant_patterns),
            "average_significance": (
                significance_scores.mean().item() if significance_scores.size(0) > 0 else 0.0
            ),
            "pattern_density": len(significant_patterns) / max(pattern_scores.size(0), 1),
            "attention_entropy": self._calculate_attention_entropy(attention_weights),
        }

        return {
            "status": "success",
            "session_name": session_name,
            "session_metrics": session_metrics,
            "significant_patterns": significant_patterns,
            "raw_results": {
                "pattern_scores": pattern_scores.tolist(),
                "significance_scores": significance_scores.tolist(),
                "attention_weights": (
                    attention_weights.tolist() if attention_weights.size(0) > 0 else []
                ),
            },
        }

    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Calculate entropy of attention distribution"""
        if attention_weights.size(0) == 0:
            return 0.0

        # Normalize attention weights
        attention_dist = F.softmax(attention_weights.flatten(), dim=0)

        # Calculate entropy
        entropy = -torch.sum(attention_dist * torch.log(attention_dist + 1e-10))
        return entropy.item()


def infer_shard_embeddings(data, out_dir: str, loader_cfg):  # type: ignore[no-untyped-def]
    """Minimal stub that writes placeholder embeddings and patterns.

    Parameters
    ----------
    data : Any
        Ignored graph data.
    out_dir : str
        Base output directory for run.
    loader_cfg : Any
        Configuration for the loader (unused).

    Returns
    -------
    tuple[str, str]
        Paths to the embeddings and patterns parquet files.
    """
    from pathlib import Path
    import pandas as pd

    run_path = Path(out_dir)
    emb_dir = run_path / "embeddings"
    patt_dir = run_path / "patterns"
    emb_dir.mkdir(parents=True, exist_ok=True)
    patt_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_dir / "embeddings.parquet"
    patt_path = patt_dir / "patterns.parquet"
    pd.DataFrame().to_parquet(emb_path)
    pd.DataFrame().to_parquet(patt_path)
    return str(emb_path), str(patt_path)
