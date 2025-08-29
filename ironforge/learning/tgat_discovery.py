"""
TGAT Discovery Engine for Archaeological Pattern Discovery
Temporal Graph Attention Network for market pattern archaeology
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Callable

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Import SDPA - fail fast if not available (no fallbacks allowed)
import math
from torch.nn.functional import scaled_dot_product_attention as sdpa
_HAS_SDPA = True


def graph_attention(q, k, v, *, edge_mask_bool=None, time_bias=None,
                    dropout_p=0.0, is_causal=False, impl="sdpa", training=True):
    """
    Unified graph attention with masking and temporal bias support
    
    Args:
        q,k,v: [B,H,L,D] query, key, value tensors
        edge_mask_bool: [B,1,L,S] boolean mask where True = block attention
        time_bias: [B,1,L,S] additive bias to attention logits (0 where none)
        dropout_p: Dropout probability for attention weights
        is_causal: Apply causal masking
        impl: "sdpa" | "manual" - attention implementation
        training: Whether model is in training mode
        
    Returns:
        (out, attn_probs): Attention output and probabilities (None for SDPA)
    """
    B, H, L, D = q.shape
    S = k.shape[-2]

    # Build float mask for SDPA: 0 for allowed, -1e9 for blocked
    attn_mask_float = None
    if edge_mask_bool is not None:
        attn_mask_float = edge_mask_bool.to(q.dtype) * (-1e9)

    if time_bias is not None:
        if attn_mask_float is None:
            attn_mask_float = torch.zeros(B, 1, L, S, dtype=q.dtype, device=q.device)
        attn_mask_float = attn_mask_float + time_bias

    if impl == "sdpa":
        out = sdpa(q, k, v,
                   attn_mask=attn_mask_float,  # float mask works on PyTorch ≥2.0
                   dropout_p=dropout_p if training else 0.0,
                   is_causal=is_causal)
        # Optional: recover attention for logging via manual softmax on a small probe batch if needed
        return out, None

    # Manual implementation for debugging/validation
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)   # [B,H,L,S]
    if time_bias is not None:
        scores = scores + time_bias
    if edge_mask_bool is not None:
        scores = scores.masked_fill(edge_mask_bool, -1e9)
    if is_causal:
        causal = torch.ones(L, S, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(causal, -1e9)
    attn = F.softmax(scores, dim=-1)
    if training and dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    out = torch.matmul(attn, v)
    return out, attn


def build_edge_mask(edge_index, L, *, device, batch_ptr=None, allow_self=False):
    """
    Build edge mask from graph connectivity
    
    Args:
        edge_index: [2,E] directed edge indices (u->v)
        L: Number of nodes in graph
        device: Target device for mask
        batch_ptr: Optional list of (start,end) per session for block-diagonal masks
        allow_self: Allow self-attention connections
        
    Returns:
        mask: [B,1,L,L] boolean mask where True blocks attention
    """
    mask = torch.ones(1, 1, L, L, dtype=torch.bool, device=device)  # start fully blocked
    u, v = edge_index
    mask[0, 0, u, v] = False  # Allow attention along edges
    if allow_self:
        idx = torch.arange(L, device=device)
        mask[0, 0, idx, idx] = False  # Allow self-attention
    return mask


def build_time_bias(dt_minutes, L, *, device, scale=0.1, buckets=None):
    """
    Build temporal bias from time differences
    
    Args:
        dt_minutes: Dense [L,L] matrix of time differences (fill 0 where no edge)
        L: Number of nodes
        device: Target device for bias
        scale: Scale parameter for smooth bias
        buckets: List of (lo, hi, weight) tuples for bucketed bias
        
    Returns:
        tb: [1,1,L,L] float bias tensor
    """
    tb = torch.zeros(1, 1, L, L, dtype=torch.float32, device=device)
    if buckets:
        for lo, hi, w in buckets:
            m = (dt_minutes >= lo) & (dt_minutes <= hi)
            tb[0, 0][m] += w
    else:
        # smooth preference for nearer neighbors
        tb = -(torch.as_tensor(dt_minutes, device=device, dtype=torch.float32) / (scale + 1e-6)).sqrt().unsqueeze(0).unsqueeze(0)
    return tb


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

    def forward(self, node_features: torch.Tensor, _edge_features: torch.Tensor, 
                temporal_distances: torch.Tensor | None, return_attn: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
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


class EnhancedTemporalAttentionLayer(nn.Module):
    """
    Enhanced Temporal Attention with Masked Attention and Temporal Bias
    Optional TGAT patch for dual graph views with DAG constraints
    """

    def __init__(self, input_dim=45, hidden_dim=44, num_heads=4, cfg=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Use default config if none provided
        if cfg is None:
            from ironforge.learning.dual_graph_config import TGATConfig
            cfg = TGATConfig()
        
        self.cfg = cfg

        # Project 45D/53D (M1-enhanced) to 44D for multi-head processing
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Multi-head attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Enhanced temporal encoding with learnable parameters
        self.temporal_encoding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Temporal bias network for enhanced temporal relationships
        self.temporal_bias_network = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),  # 2D input: [dt_minutes, temporal_distance]
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Tanh()  # Bounded bias values
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # Optional: Positional encoding for DAG structure
        self.dag_positional_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)  # Max 1000 nodes
        
        logger.info(f"Enhanced Temporal Attention: {input_dim}D -> {hidden_dim}D, {num_heads} heads, impl={cfg.attention_impl}, edge_mask={cfg.use_edge_mask}, time_bias={cfg.use_time_bias}")


    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor = None, 
                temporal_data: torch.Tensor = None, dag: nx.DiGraph = None, 
                return_attn: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Enhanced forward pass with masked attention and temporal bias
        
        Args:
            node_features: [N, input_dim] node feature tensor (45D or 53D for M1-enhanced)
            edge_features: [E, 20] edge feature tensor (optional)
            temporal_data: [N, N, 2] temporal relationship data [dt_minutes, temporal_distance]
            dag: NetworkX DiGraph for causal masking (optional)
            return_attn: bool, whether to return attention weights
            
        Returns:
            attended_features: [N, 44] attended node features
            attention_weights: [N, N] attention weights (if requested)
        """
        batch_size, seq_len = 1, node_features.size(0)  # Single graph processing
        device = node_features.device

        # Project input features
        projected_features = self.input_projection(node_features)  # [N, 44]
        
        # Add DAG positional encoding if available
        if seq_len <= self.dag_positional_encoding.size(1):
            projected_features = projected_features + self.dag_positional_encoding[0, :seq_len, :]

        # Generate Q, K, V
        Q = self.query(projected_features).view(1, seq_len, self.num_heads, self.head_dim)  # [1, N, 4, 11]
        K = self.key(projected_features).view(1, seq_len, self.num_heads, self.head_dim)    # [1, N, 4, 11]
        V = self.value(projected_features).view(1, seq_len, self.num_heads, self.head_dim)  # [1, N, 4, 11]

        # Create edge mask and temporal bias using new unified system
        edge_mask = None
        time_bias = None
        
        if self.cfg.use_edge_mask and dag is not None:
            # Convert DAG to edge_index format for mask building
            edges = list(dag.edges())
            if edges:
                edge_index = torch.tensor(edges, device=device).t()  # [2, E]
                edge_mask = build_edge_mask(edge_index, seq_len, device=device, allow_self=True)
        
        if self.cfg.use_time_bias != "none" and temporal_data is not None:
            # Use temporal data to build time bias
            dt_minutes = temporal_data[:, :, 0]  # Extract dt_minutes from [N, N, 2]
            
            if self.cfg.use_time_bias == "bucket":
                # Define buckets for different time ranges with different bias weights
                buckets = [
                    (0, 5, 0.2),      # 0-5 minutes: strong positive bias
                    (5, 15, 0.1),     # 5-15 minutes: moderate positive bias  
                    (15, 60, 0.0),    # 15-60 minutes: neutral
                    (60, 240, -0.1),  # 1-4 hours: slight negative bias
                ]
                time_bias = build_time_bias(dt_minutes, seq_len, device=device, buckets=buckets)
            elif self.cfg.use_time_bias == "rbf":
                time_bias = build_time_bias(dt_minutes, seq_len, device=device, scale=0.1)
        
        # Reshape for attention: [1, N, H, D] -> [1, H, N, D] 
        Q_t = Q.transpose(1, 2)  # [1, 4, N, 11]
        K_t = K.transpose(1, 2)  # [1, 4, N, 11]
        V_t = V.transpose(1, 2)  # [1, 4, N, 11]
        
        # Apply unified graph attention
        attended, attention_weights = graph_attention(
            Q_t, K_t, V_t,
            edge_mask_bool=edge_mask,
            time_bias=time_bias,
            dropout_p=0.1,
            is_causal=self.cfg.is_causal,
            impl=self.cfg.attention_impl,
            training=self.training
        )
        
        # Reshape back: [1, H, N, D] -> [1, N, H, D]
        attended = attended.transpose(1, 2)  # [1, N, 4, 11]

        # Concatenate heads and project output
        attended = attended.contiguous().view(seq_len, -1)  # [N, 44]
        output = self.output_projection(attended)  # [N, 44]

        # Return attention weights if requested (average across heads)
        if return_attn and attention_weights is not None:
            avg_attention = attention_weights.mean(dim=1).squeeze(0)  # [N, N]
            return output, avg_attention
        else:
            return output, None


def create_attention_layer(
    input_dim: int = 45, 
    hidden_dim: int = 44, 
    num_heads: int = 4, 
    enhanced: bool = False,
    cfg=None
) -> nn.Module:
    """
    Factory function to create appropriate attention layer
    
    Args:
        input_dim: Input feature dimension (45D standard, 53D for M1-enhanced)
        hidden_dim: Hidden dimension for attention processing
        num_heads: Number of attention heads
        enhanced: Use enhanced TGAT with masked attention and temporal bias
        cfg: TGATConfig instance with attention settings
        
    Returns:
        Appropriate attention layer instance
    """
    if enhanced:
        logger.info("Creating Enhanced Temporal Attention Layer with DAG masking and temporal bias")
        return EnhancedTemporalAttentionLayer(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_heads=num_heads,
            cfg=cfg
        )
    else:
        logger.info("Creating Standard Temporal Attention Layer") 
        return TemporalAttentionLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )


class IRONFORGEDiscovery(nn.Module):
    """
    IRONFORGE Discovery Engine using TGAT
    Archaeological pattern discovery through temporal graph attention
    """

    def __init__(self, node_dim=45, edge_dim=20, hidden_dim=44, num_layers=2, 
                 enhanced_tgat=False, cfg=None):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.enhanced_tgat = enhanced_tgat

        # Use default config if none provided
        if cfg is None:
            from ironforge.learning.dual_graph_config import TGATConfig
            cfg = TGATConfig()
        self.cfg = cfg

        # Temporal attention layers with optional enhancement
        self.attention_layers = nn.ModuleList([
            create_attention_layer(
                input_dim=node_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                enhanced=enhanced_tgat,
                cfg=cfg
            ) for i in range(num_layers)
        ])
        
        logger.info(f"IRONFORGE Discovery: {num_layers} layers, enhanced_tgat={enhanced_tgat}")

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

    def load_calibrated_oracle_weights(self, weights_dir: Path | str) -> bool:
        """
        Load calibrated Oracle weights for enhanced temporal predictions
        
        Args:
            weights_dir: Directory containing calibrated Oracle weights
            
        Returns:
            bool: True if weights loaded successfully
        """
        weights_dir = Path(weights_dir)
        weights_path = weights_dir / "weights.pt"  # Updated to match trainer output
        manifest_path = weights_dir / "training_manifest.json"
        
        if not weights_path.exists():
            # Fallback to old naming convention
            old_weights_path = weights_dir / "oracle_range_head.pth"
            if old_weights_path.exists():
                weights_path = old_weights_path
            else:
                logger.warning(f"No calibrated Oracle weights found at {weights_path}")
                return False
        
        try:
            # Validate dimensions if manifest exists
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                
                # Validate model architecture compatibility
                model_arch = manifest.get("model_architecture", {})
                expected_input = model_arch.get("input_dim", 44)
                expected_output = model_arch.get("output_dim", 2)
                
                # Check TGAT architecture compatibility
                actual_hidden = self.hidden_dim
                if actual_hidden != expected_input:
                    logger.error(f"Architecture mismatch: TGAT hidden_dim={actual_hidden}, "
                               f"Oracle expects input_dim={expected_input}")
                    return False
                
                # Check range_head output dimension
                range_head_out = self.range_head[-1].out_features if hasattr(self.range_head[-1], 'out_features') else 2
                if range_head_out != expected_output:
                    logger.error(f"Range head mismatch: actual output_dim={range_head_out}, "
                               f"expected output_dim={expected_output}")
                    return False
                
                logger.info(f"Model architecture validated: {expected_input}→{expected_output}")
            
            # Load weights
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Validate state dict keys
            expected_keys = set(self.range_head.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            
            if expected_keys != loaded_keys:
                missing_keys = expected_keys - loaded_keys
                unexpected_keys = loaded_keys - expected_keys
                
                if missing_keys:
                    logger.error(f"Missing keys in state dict: {missing_keys}")
                    return False
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")
            
            # Load state dict
            self.range_head.load_state_dict(state_dict)
            
            # Verify loading worked by checking parameter counts
            total_params = sum(p.numel() for p in self.range_head.parameters())
            logger.info(f"✅ Calibrated Oracle weights loaded from {weights_path}")
            logger.info(f"   Model: {expected_input if manifest_path.exists() else 44}→{expected_output if manifest_path.exists() else 2}, {total_params} parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibrated Oracle weights: {e}")
            return False

    def forward(self, graph: nx.Graph, dag: nx.DiGraph = None, return_attn: bool = False) -> dict[str, torch.Tensor]:
        """
        Discover archaeological patterns in session graph

        Args:
            graph: NetworkX graph with rich features
            dag: Optional DAG for enhanced TGAT masking (used when enhanced_tgat=True)
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

            # Create temporal data for enhanced layers if needed
            temporal_data = None
            if self.enhanced_tgat and edges:
                # Create [N, N, 2] temporal data matrix for enhanced layers (O(E) scatter)
                n_nodes = len(nodes)
                temporal_data = torch.zeros((n_nodes, n_nodes, 2), dtype=torch.float32)

                node_to_idx = {node: idx for idx, node in enumerate(nodes)}
                src_idx: list[int] = []
                dst_idx: list[int] = []
                dt_vals: list[float] = []
                td_vals: list[float] = []

                for (u, v) in edges:
                    src_idx.append(node_to_idx[u])
                    dst_idx.append(node_to_idx[v])
                    edge_data = graph.edges[u, v]
                    dt_vals.append(float(edge_data.get('dt_minutes', 0.0)))
                    td_vals.append(float(edge_data.get('temporal_distance', 0.0)))

                if src_idx:
                    si = torch.tensor(src_idx, dtype=torch.long)
                    di = torch.tensor(dst_idx, dtype=torch.long)
                    temporal_data[si, di, 0] = torch.tensor(dt_vals, dtype=torch.float32)
                    temporal_data[si, di, 1] = torch.tensor(td_vals, dtype=torch.float32)

            for i, layer in enumerate(self.attention_layers):
                if self.enhanced_tgat and hasattr(layer, 'create_dag_mask'):
                    # Enhanced layer with DAG masking and temporal bias
                    current_features, attention_weights = layer(
                        current_features, edge_features, temporal_data, dag, return_attn
                    )
                else:
                    # Standard layer (backwards compatibility)
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
                f"Discovery complete: found {(pattern_scores > 0.1).sum().item()} significant patterns"
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
            early_batch_pct: Percentage of early events to use for prediction (0, 0.5]
            
        Returns:
            Dictionary containing oracle predictions and metadata
        """
        try:
            # Validate early_batch_pct
            if not (0 < early_batch_pct <= 0.5):
                raise ValueError(f"early_batch_pct must be in (0, 0.5], got {early_batch_pct}")
            
            nodes = list(graph.nodes())
            if len(nodes) == 0:
                return self._empty_oracle_result()
            
            # Guard for tiny sessions (need at least 3 events)
            if len(nodes) < 3:
                logger.warning(f"Session too small for oracle predictions: {len(nodes)} events < 3 minimum")
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
            
        except ValueError as e:
            # Re-raise validation errors
            raise e
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
            significant_mask = (pattern_scores > 0.1).any(dim=1)
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
    from ironforge.graph_builder.pyg_converters import pyg_to_networkx

    # Prepare output directories using documented run layout
    run_dir = Path(out_dir)
    embeddings_dir = run_dir / "embeddings"
    patterns_dir = run_dir / "patterns"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.mkdir(parents=True, exist_ok=True)

    # Convert PyG data back to NetworkX for processing
    graph = pyg_to_networkx(data)

    # Ensure required feature tensors exist on nodes/edges (adapt from f*/e* attributes)
    def _ensure_graph_features(g: nx.Graph) -> None:
        import math as _math
        # Nodes: build 45D tensor from f0..f44 (pad with 0.0 if missing)
        for node_id, attrs in g.nodes(data=True):
            if "feature" not in attrs:
                values: list[float] = []
                for i in range(45):
                    key = f"f{i}"
                    val = attrs.get(key, 0.0)
                    try:
                        fv = float(val) if val is not None and not (isinstance(val, float) and _math.isnan(val)) else 0.0
                    except Exception:
                        fv = 0.0
                        
                    values.append(fv)
                g.nodes[node_id]["feature"] = torch.tensor(values, dtype=torch.float32)

        # Edges: build 20D tensor from e0..e19; add temporal_distance from dt if present
        for u, v, attrs in g.edges(data=True):
            if "feature" not in attrs:
                e_values: list[float] = []
                for i in range(20):
                    key = f"e{i}"
                    val = attrs.get(key, 0.0)
                    try:
                        ev = float(val) if val is not None else 0.0
                    except Exception:
                        ev = 0.0
                    e_values.append(ev)
                graph.edges[u, v]["feature"] = torch.tensor(e_values, dtype=torch.float32)
            if "temporal_distance" not in attrs:
                dt = attrs.get("dt")
                graph.edges[u, v]["temporal_distance"] = float(dt) / 60000.0 if dt is not None else 1.0

    _ensure_graph_features(graph)

    # Determine a stable session identifier for file naming
    session_id = None
    for _, attrs in graph.nodes(data=True):
        sid = attrs.get("session_id") or attrs.get("session_name")
        if sid:
            session_id = str(sid)
            break
    if not session_id:
        session_id = "session"
    # Sanitize session_id for filesystem
    safe_session_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_id)

    # Initialize discovery engine
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()  # Set to evaluation mode

    # Run pattern discovery
    with torch.no_grad():
        results = discovery_engine.forward(graph, return_attn=True)

        # Extract embeddings and patterns
        embeddings = results["node_embeddings"]
        pattern_scores = results["pattern_scores"]
        significance_scores = results["significance_scores"]

        # Save embeddings (per-session file) under embeddings/
        embeddings_path = embeddings_dir / f"node_embeddings_{safe_session_id}.parquet"
        pd.DataFrame(embeddings.numpy()).to_parquet(embeddings_path, index=False)

        # Save patterns (per-session file) under patterns/
        patterns_path = patterns_dir / f"patterns_{safe_session_id}.parquet"
        patterns_df = pd.DataFrame({
            "pattern_scores": pattern_scores.tolist(),
            "significance_scores": significance_scores.flatten().tolist(),
        })
        patterns_df.to_parquet(patterns_path, index=False)

        # Optional: Oracle predictions with configuration check
        # Resolve oracle flags from various loader_cfg shapes (dc, dict, flat attrs)
        oracle_enabled = True
        early_pct = 0.20
        try:
            # Dataclass-style: cfg.oracle.enabled, cfg.oracle.early_pct
            if hasattr(loader_cfg, 'oracle'):
                oracle_section = getattr(loader_cfg, 'oracle')
                if isinstance(oracle_section, dict):
                    oracle_enabled = bool(oracle_section.get('enabled', True))
                    early_pct = float(oracle_section.get('early_pct', early_pct))
                else:
                    # object with attributes
                    if hasattr(oracle_section, 'enabled'):
                        oracle_enabled = bool(getattr(oracle_section, 'enabled'))
                    if hasattr(oracle_section, 'early_pct'):
                        early_pct = float(getattr(oracle_section, 'early_pct'))
            else:
                # Flat attributes
                if hasattr(loader_cfg, 'oracle_enabled'):
                    oracle_enabled = bool(getattr(loader_cfg, 'oracle_enabled'))
                if hasattr(loader_cfg, 'oracle_early_pct'):
                    early_pct = float(getattr(loader_cfg, 'oracle_early_pct'))
        except Exception:
            # Keep defaults on errors
            pass

        if oracle_enabled:
            oracle_predictions = discovery_engine.predict_session_range(
                graph, early_batch_pct=early_pct
            )

            # Add required schema v0 fields in exact order
            oracle_predictions["run_dir"] = str(run_dir)
            oracle_predictions["session_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
            oracle_predictions["pattern_id"] = f"pattern_{len(oracle_predictions['node_idx_used']):03d}"
            oracle_predictions["start_ts"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
            oracle_predictions["end_ts"] = (
                pd.Timestamp.now() + pd.Timedelta(minutes=oracle_predictions['n_events'])
            ).strftime("%Y-%m-%dT%H:%M:%S")

            schema_v0_data = {
                "run_dir": oracle_predictions["run_dir"],
                "session_date": oracle_predictions["session_date"],
                "pct_seen": oracle_predictions["pct_seen"],
                "n_events": oracle_predictions["n_events"],
                "pred_low": oracle_predictions["pred_low"],
                "pred_high": oracle_predictions["pred_high"],
                "pred_center": oracle_predictions["center"],
                "pred_half_range": oracle_predictions["half_range"],
                "confidence": oracle_predictions["confidence"],
                "pattern_id": oracle_predictions["pattern_id"],
                "start_ts": oracle_predictions["start_ts"],
                "end_ts": oracle_predictions["end_ts"],
                "early_expansion_cnt": oracle_predictions["early_expansion_cnt"],
                "early_retracement_cnt": oracle_predictions["early_retracement_cnt"],
                "early_reversal_cnt": oracle_predictions["early_reversal_cnt"],
                "first_seq": oracle_predictions["first_seq"],
            }

            oracle_df = pd.DataFrame([schema_v0_data])
            oracle_path = run_dir / "oracle_predictions.parquet"
            oracle_df.to_parquet(oracle_path, index=False)

            logger.debug(
                f"Oracle predictions: {oracle_predictions['pattern_id']}, confidence: {oracle_predictions['confidence']:.3f}"
            )

    logger.info(f"Discovery complete: {embeddings_path}, {patterns_path}")
    return str(embeddings_path), str(patterns_path)
