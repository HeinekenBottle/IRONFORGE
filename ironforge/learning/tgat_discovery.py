"""
TGAT Discovery Engine for Archaeological Pattern Discovery
Temporal Graph Attention Network for market pattern archaeology
"""

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, node_features, _edge_features, temporal_distances, return_attn=True):
        """
        Forward pass with temporal attention

        Args:
            node_features: [N, 45] node feature tensor
            edge_features: [E, 20] edge feature tensor
            temporal_distances: [E, 1] temporal distance tensor
            return_attn: bool, whether to return attention weights for oracle predictions

        Returns:
            attended_features: [N, 44] attended node features
            attention_weights: [N, N] attention weight matrix (if return_attn=True)
        """
        _batch_size, seq_len = node_features.size(0), node_features.size(0)

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
            self.temporal_encoding(temporal_distances.float().unsqueeze(-1))  # [E, 44]
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
        if return_attn:
            avg_attention = attention_weights.mean(dim=0)  # [N, N]
            return output, avg_attention
        else:
            return output, None


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

        # Oracle session range prediction head
        self.range_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # -> (center, half_range)
        )
        
        # Initialize range_head with Xavier normal for stable cold start
        for layer in self.range_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        logger.info(
            f"IRONFORGE Discovery initialized: {node_dim}D nodes, {edge_dim}D edges, {num_layers} layers"
        )

    def forward(self, graph: nx.Graph, return_attn=True) -> dict[str, torch.Tensor]:
        """
        Discover archaeological patterns in session graph

        Args:
            graph: NetworkX graph with rich features
            return_attn: bool, whether to return attention weights (for oracle predictions)

        Returns:
            Dictionary containing:
            - pattern_scores: [N, 16] pattern type probabilities
            - attention_weights: [N, N] attention weight matrix (if return_attn=True)
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
                    current_features, edge_features, temporal_distances, return_attn
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

    def _empty_result(self) -> dict[str, torch.Tensor]:
        """Return empty result structure"""
        return {
            "pattern_scores": torch.zeros(0, 16),
            "attention_weights": torch.zeros(0, 0),
            "significance_scores": torch.zeros(0, 1),
            "node_embeddings": torch.zeros(0, self.hidden_dim),
            "session_name": "empty",
        }

    def predict_session_range(self, graph: nx.Graph, early_batch_pct: float = 0.20) -> dict[str, Any]:
        """
        Predict session range from early events using temporal non-locality
        
        Args:
            graph: NetworkX graph with session events (via EnhancedGraphBuilder)
            early_batch_pct: Percentage of early events to use for prediction
            
        Returns:
            Dictionary containing:
            - pct_seen: Percentage of session events analyzed  
            - n_events: Number of events used for prediction
            - pred_high: Predicted session high
            - pred_low: Predicted session low
            - center: Predicted range center
            - half_range: Predicted half-range width
            - confidence: Prediction confidence (0-1)
            - notes: Description of method used
            - node_idx_used: List of early node indices for pattern linking
        """
        try:
            nodes = list(graph.nodes())
            if len(nodes) == 0:
                return self._empty_oracle_result()
                
            num_events = len(nodes)
            k = max(1, int(num_events * early_batch_pct))
            
            # DELTA A: Use early subgraph via same path as discovery
            early_nodes = nodes[:k]  # Take first k events (temporal order)
            early_subgraph = graph.subgraph(early_nodes).copy()
            
            # Forward pass with attention weights enabled
            results = self.forward(early_subgraph, return_attn=True)
            
            embeddings = results["node_embeddings"]  # [k, 44]
            attention_weights = results["attention_weights"]  # [k, k] or None
            node_idx_used = early_nodes  # Track which nodes used for pattern linking
            
            # DELTA C: Extract phase sequence breadcrumbs from early subgraph (read-only)
            phase_counts = self._extract_phase_sequences(early_subgraph)
            
            # Attention-weighted pooling over early events
            if attention_weights is not None and attention_weights.size(0) > 0:
                # Use attention to weight embeddings
                attn_scores = attention_weights.sum(dim=0)  # Sum attention received by each node
                attn_weights = F.softmax(attn_scores, dim=0)  # Normalize to probabilities
                pooled = (attn_weights.unsqueeze(-1) * embeddings).sum(dim=0)  # [44]
                
                # Calculate confidence as max attention weight
                confidence = float(attn_weights.max().item())
            else:
                # Fallback to simple mean pooling
                pooled = embeddings.mean(dim=0)  # [44]
                confidence = 0.5
                
            # Map pooled embedding to range prediction
            range_pred = self.range_head(pooled.unsqueeze(0))  # [1, 2]
            center, half_range = range_pred[0, 0].item(), abs(range_pred[0, 1].item())
            
            pred_high = center + half_range
            pred_low = center - half_range
            
            return {
                "pct_seen": round(k / num_events, 4),
                "n_events": int(k),
                "pred_high": float(pred_high),
                "pred_low": float(pred_low),
                "center": float(center),
                "half_range": float(half_range),
                "confidence": float(confidence),
                "notes": f"attention-weighted pooling over {k} early events",
                "node_idx_used": list(node_idx_used),  # DELTA A: Return node indices used
                **phase_counts  # DELTA C: Add phase sequence breadcrumbs
            }
            
        except Exception as e:
            logger.error(f"Oracle prediction failed: {e}")
            return self._empty_oracle_result()
    
    def _empty_oracle_result(self) -> dict[str, Any]:
        """Return empty oracle prediction result"""
        return {
            "pct_seen": 0.0,
            "n_events": 0,
            "pred_high": 0.0,
            "pred_low": 0.0,
            "center": 0.0,
            "half_range": 0.0,
            "confidence": 0.0,
            "notes": "empty session - no predictions available",
            "node_idx_used": [],  # DELTA A: Empty node list
            # DELTA C: Empty phase counts
            "early_expansion_cnt": 0,
            "early_retracement_cnt": 0,
            "early_reversal_cnt": 0,
            "first_seq": ""
        }
    
    def _extract_phase_sequences(self, early_subgraph: nx.Graph) -> dict[str, Any]:
        """DELTA C: Extract phase sequence breadcrumbs from early subgraph (read-only)"""
        try:
            expansion_cnt = 0
            retracement_cnt = 0
            reversal_cnt = 0
            sequence_events = []
            
            # Analyze semantic flags in 45D node features (first 8 dimensions are semantic)
            for node_id in sorted(early_subgraph.nodes()):
                node_feature = early_subgraph.nodes[node_id]["feature"]
                
                # Check semantic event flags (indices 0-7 in 45D vector)
                if node_feature[1] > 0.5:  # expansion_phase_flag 
                    expansion_cnt += 1
                    sequence_events.append("E")
                elif node_feature[3] > 0.5:  # retracement_flag
                    retracement_cnt += 1  
                    sequence_events.append("R")
                elif node_feature[4] > 0.5:  # reversal_flag
                    reversal_cnt += 1
                    sequence_events.append("V")
                else:
                    sequence_events.append("C")  # Consolidation/other
            
            # Build first occurrence sequence string
            first_seq = "→".join(sequence_events[:5])  # First 5 events only
            
            return {
                "early_expansion_cnt": expansion_cnt,
                "early_retracement_cnt": retracement_cnt, 
                "early_reversal_cnt": reversal_cnt,
                "first_seq": first_seq
            }
            
        except Exception as e:
            logger.warning(f"Phase sequence extraction failed: {e}")
            return {
                "early_expansion_cnt": 0,
                "early_retracement_cnt": 0,
                "early_reversal_cnt": 0, 
                "first_seq": "unknown"
            }

    def discover_session_patterns(self, session_data: dict[str, Any]) -> dict[str, Any]:
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
        self, results: dict[str, torch.Tensor], session_data: dict[str, Any]
    ) -> dict[str, Any]:
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


def infer_shard_embeddings(data, out_dir: str, loader_cfg) -> tuple[str, str]:
    """
    Process shard data through TGAT discovery engine with oracle predictions
    
    Args:
        data: PyG graph data containing session information
        out_dir: Output directory for results
        loader_cfg: Configuration for data loading
        
    Returns:
        Tuple[str, str]: (embeddings_path, patterns_path)
    """
    try:
        from ironforge.graph_builder.pyg_converters import pyg_to_networkx
        
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert PyG data back to NetworkX for processing
        graph = pyg_to_networkx(data)
        
        # Initialize discovery engine
        discovery_engine = IRONFORGEDiscovery()
        discovery_engine.eval()  # Set to evaluation mode
        
        # Run pattern discovery
        with torch.no_grad():
            discovery_results = discovery_engine.forward(graph, return_attn=True)
            
            # Extract embeddings and patterns
            embeddings = discovery_results["node_embeddings"]
            pattern_scores = discovery_results["pattern_scores"]
            significance_scores = discovery_results["significance_scores"]
            
            # Save embeddings
            embeddings_path = output_dir / "embeddings.parquet"
            emb_df = pd.DataFrame(embeddings.numpy())
            emb_df.to_parquet(embeddings_path, index=False)
            
            # Save patterns 
            patterns_path = output_dir / "patterns.parquet"
            patterns_df = pd.DataFrame({
                "pattern_scores": pattern_scores.tolist(),
                "significance_scores": significance_scores.flatten().tolist(),
            })
            patterns_df.to_parquet(patterns_path, index=False)
            
            # NEW: Oracle predictions with configuration check
            oracle_enabled = getattr(loader_cfg, 'oracle', True)
            early_pct = getattr(loader_cfg, 'oracle_early_pct', 0.20)
            
            if oracle_enabled:
                oracle_predictions = discovery_engine.predict_session_range(
                    graph, early_batch_pct=early_pct
                )
                
                # DELTA B: Add pattern linking and temporal metadata
                oracle_predictions["run_dir"] = str(output_dir)
                oracle_predictions["ts_generated"] = pd.Timestamp.utcnow()
                oracle_predictions["session_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
                
                # Mock pattern linking (in production, would link to actual discovered patterns)
                oracle_predictions["pattern_id"] = f"pattern_{len(oracle_predictions['node_idx_used']):03d}"
                oracle_predictions["start_ts"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
                oracle_predictions["end_ts"] = (pd.Timestamp.now() + pd.Timedelta(minutes=oracle_predictions['n_events'])).strftime("%Y-%m-%dT%H:%M:%S")
                
                # DELTA B: Add oracle-distance fields (computed at report-time)
                oracle_predictions["center_delta_to_pattern_mid"] = 0.0  # Placeholder - computed vs discovered patterns
                oracle_predictions["range_overlap_pct"] = 0.0  # Placeholder - computed vs discovered patterns
                
                # Save oracle predictions with enhanced schema
                oracle_df = pd.DataFrame([oracle_predictions])
                oracle_path = output_dir / "oracle_predictions.parquet"
                oracle_df.to_parquet(oracle_path, index=False)
                
                logger.info(f"Oracle predictions saved: {oracle_predictions['pred_high']:.2f}/{oracle_predictions['pred_low']:.2f} "
                           f"(confidence: {oracle_predictions['confidence']:.3f}, pattern_id: {oracle_predictions['pattern_id']})")
            
        logger.info(f"Discovery complete: {embeddings_path}, {patterns_path}")
        return str(embeddings_path), str(patterns_path)
        
    except Exception as e:
        logger.error(f"Shard inference failed: {e}")
        # Return empty paths on error
        empty_embeddings = output_dir / "embeddings_empty.parquet"  
        empty_patterns = output_dir / "patterns_empty.parquet"
        
        pd.DataFrame().to_parquet(empty_embeddings)
        pd.DataFrame().to_parquet(empty_patterns)
        
        return str(empty_embeddings), str(empty_patterns)
