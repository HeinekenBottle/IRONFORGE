"""
Context7-Optimized TGAT Discovery Engine
Enhanced implementation with Context7 performance optimizations

Key optimizations:
1. AMP (Automatic Mixed Precision) support
2. Flash attention via SDPA backend selection
3. Block-sparse attention patterns
4. Time bias caching
5. FP16 precision options
6. Vectorized operations
"""

import math
import logging
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import autocast
except ImportError:
    try:
        from torch.autocast import autocast
    except ImportError:
        from contextlib import nullcontext as autocast
import networkx as nx
import numpy as np

from .dual_graph_config import TGATConfig
from .tgat_discovery import graph_attention, build_edge_mask, build_time_bias

logger = logging.getLogger(__name__)

# Import SDPA - fail fast if not available (no fallbacks)
from torch.nn.functional import scaled_dot_product_attention as sdpa


@dataclass 
class OptimizedTGATConfig:
    """Extended configuration for Context7 optimizations"""
    
    # Base TGAT config
    base_config: TGATConfig
    
    # Performance optimizations (Context7 recommendations)
    enable_amp: bool = False
    enable_flash_attention: bool = False
    enable_block_sparse_mask: bool = False
    enable_time_bias_caching: bool = True
    use_fp16: bool = False
    
    # Advanced attention optimizations
    use_content_defined_chunking: bool = False
    enable_fused_ops: bool = False
    sparse_attention_density: float = 0.1
    
    # Memory optimizations
    gradient_checkpointing: bool = False
    memory_efficient_attention: bool = True


class TimeBiasCache:
    """LRU cache for time bias computations with Context7 optimizations"""
    
    def __init__(self, maxsize: int = 128):
        self.cache = {}
        self.maxsize = maxsize
        self.access_order = []
        
    def get_or_compute(self, dt_minutes: torch.Tensor, L: int, device: torch.device, 
                      scale: float = 0.1, buckets: Optional[list] = None) -> torch.Tensor:
        """Get cached time bias or compute and cache it"""
        
        # Create cache key from tensor hash
        cache_key = (
            hash(dt_minutes.cpu().numpy().tobytes()),
            L, scale, str(buckets) if buckets else None
        )
        
        if cache_key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key].to(device)
            
        # Compute new time bias
        time_bias = build_time_bias(dt_minutes, L, device=device, scale=scale, buckets=buckets)
        
        # Add to cache
        self._add_to_cache(cache_key, time_bias.cpu())
        
        return time_bias
        
    def _add_to_cache(self, key: tuple, value: torch.Tensor):
        """Add item to cache with LRU eviction"""
        
        if len(self.cache) >= self.maxsize:
            # Remove least recently used
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            
        self.cache[key] = value
        self.access_order.append(key)


class OptimizedGraphAttention(nn.Module):
    """
    Context7-optimized graph attention with advanced performance features
    """
    
    def __init__(self, config: OptimizedTGATConfig):
        super().__init__()
        self.config = config
        self.time_bias_cache = TimeBiasCache() if config.enable_time_bias_caching else None
        
        # Enable CUDA optimizations if available
        if torch.cuda.is_available() and config.enable_flash_attention:
            # Context7 recommendation: Enable all SDPA backends for optimal selection
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True) 
            torch.backends.cuda.enable_math_sdp(True)
            
            if config.use_fp16:
                # Enable reduced precision reductions
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
                
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                dt_minutes: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Context7 optimizations
        
        Args:
            q, k, v: [B,H,L,D] query, key, value tensors
            edge_index: [2,E] edge connectivity 
            dt_minutes: [L,L] time differences matrix
            training: Whether in training mode
            
        Returns:
            (output, attention_weights): Attention output and optional weights
        """
        B, H, L, D = q.shape
        
        # Build optimized attention mask
        edge_mask = None
        if edge_index is not None:
            if self.config.enable_block_sparse_mask:
                edge_mask = self._build_block_sparse_mask(edge_index, L, q.device)
            else:
                edge_mask = build_edge_mask(edge_index, L, device=q.device, allow_self=True)
                
        # Build optimized time bias
        time_bias = None
        if dt_minutes is not None:
            if self.config.enable_time_bias_caching and self.time_bias_cache:
                time_bias = self.time_bias_cache.get_or_compute(
                    dt_minutes, L, q.device, scale=0.1
                )
            else:
                time_bias = build_time_bias(dt_minutes, L, device=q.device, scale=0.1)
                
        # Apply Context7 optimizations
        if self.config.enable_amp and torch.cuda.is_available():
            return self._forward_with_amp(q, k, v, edge_mask, time_bias, training)
        elif self.config.use_fp16:
            return self._forward_with_fp16(q, k, v, edge_mask, time_bias, training)
        else:
            return self._forward_standard(q, k, v, edge_mask, time_bias, training)
            
    def _forward_with_amp(self, q, k, v, edge_mask, time_bias, training) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with Automatic Mixed Precision"""
        
        with autocast('cuda'):
            return graph_attention(
                q, k, v,
                edge_mask_bool=edge_mask,
                time_bias=time_bias,
                impl="sdpa",
                training=training
            )
            
    def _forward_with_fp16(self, q, k, v, edge_mask, time_bias, training) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with FP16 precision"""
        
        # Convert to FP16
        q_fp16 = q.half()
        k_fp16 = k.half()
        v_fp16 = v.half()
        time_bias_fp16 = time_bias.half() if time_bias is not None else None
        
        out, attn = graph_attention(
            q_fp16, k_fp16, v_fp16,
            edge_mask_bool=edge_mask,
            time_bias=time_bias_fp16,
            impl="sdpa", 
            training=training
        )
        
        # Convert back to original precision
        return out.float(), attn
        
    def _forward_standard(self, q, k, v, edge_mask, time_bias, training) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard forward pass"""
        
        return graph_attention(
            q, k, v,
            edge_mask_bool=edge_mask,
            time_bias=time_bias,
            impl="sdpa",
            training=training
        )
        
    def _build_block_sparse_mask(self, edge_index: torch.Tensor, L: int, device: torch.device) -> torch.Tensor:
        """
        Build block-sparse attention mask using Context7 structured sparsity patterns
        """
        block_size = 64  # GPU-friendly block size
        num_blocks = L // block_size
        
        # Start with fully blocked mask
        mask = torch.ones(1, 1, L, L, dtype=torch.bool, device=device)
        
        # Allow block-diagonal pattern
        for i in range(num_blocks):
            start_i = i * block_size
            end_i = min((i + 1) * block_size, L)
            
            # Self-block
            mask[0, 0, start_i:end_i, start_i:end_i] = False
            
            # Next block connection (maintains DAG property)
            if i < num_blocks - 1:
                start_j = (i + 1) * block_size
                end_j = min((i + 2) * block_size, L)
                mask[0, 0, start_i:end_i, start_j:end_j] = False
                
        # Allow specific edges from edge_index
        if len(edge_index) > 0:
            u, v = edge_index[0], edge_index[1]
            mask[0, 0, u, v] = False
            
        return mask


class OptimizedEnhancedTemporalAttentionLayer(nn.Module):
    """
    Context7-optimized Enhanced Temporal Attention Layer
    """
    
    def __init__(self, config: OptimizedTGATConfig):
        super().__init__()
        
        self.config = config
        base_cfg = config.base_config
        
        self.input_dim = base_cfg.input_dim
        self.hidden_dim = base_cfg.hidden_dim
        self.num_heads = base_cfg.num_heads
        self.head_dim = base_cfg.hidden_dim // base_cfg.num_heads
        
        # Core attention components
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Optimized attention module
        self.attention = OptimizedGraphAttention(config)
        
        # Temporal encoding with optimization
        if config.enable_fused_ops:
            self.temporal_encoding = self._create_fused_temporal_encoding()
        else:
            self.temporal_encoding = nn.Linear(2, self.hidden_dim)
            
        # Apply initialization optimizations
        self._initialize_weights()
        
        logger.info(f"Optimized TGAT Layer: {self.input_dim}D→{self.hidden_dim}D, "
                   f"{self.num_heads} heads, AMP={config.enable_amp}, "
                   f"FP16={config.use_fp16}, Flash={config.enable_flash_attention}")
        
    def forward(self, node_features: torch.Tensor, edge_features: Optional[torch.Tensor],
                temporal_data: Optional[torch.Tensor], dag: Optional[nx.DiGraph] = None,
                return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Optimized forward pass
        
        Args:
            node_features: [N, input_dim] node features  
            edge_features: [E, edge_dim] edge features (unused in attention)
            temporal_data: [N, N, 2] temporal distance matrix
            dag: NetworkX DAG for edge connectivity
            return_attn: Whether to return attention weights
            
        Returns:
            (output_features, attention_weights): Enhanced features and optional attention
        """
        N = node_features.size(0)
        device = node_features.device
        
        # Project input features (45D -> 44D)
        projected_features = self.input_projection(node_features)  # [N, hidden_dim]
        
        # Generate Q, K, V
        Q = self.query(projected_features)  # [N, hidden_dim]
        K = self.key(projected_features)   # [N, hidden_dim]
        V = self.value(projected_features)  # [N, hidden_dim]
        
        # Reshape for multi-head attention: [N, hidden_dim] -> [1, num_heads, N, head_dim]
        Q = Q.view(1, N, self.num_heads, self.head_dim).transpose(1, 2)  # [1, H, N, D]
        K = K.view(1, N, self.num_heads, self.head_dim).transpose(1, 2)  # [1, H, N, D]
        V = V.view(1, N, self.num_heads, self.head_dim).transpose(1, 2)  # [1, H, N, D]
        
        # Extract edge connectivity from DAG
        edge_index = None
        if dag is not None:
            edges = list(dag.edges())
            if edges:
                edge_index = torch.tensor(edges, device=device).T  # [2, E]
                
        # Extract temporal distance matrix
        dt_minutes = None
        if temporal_data is not None and temporal_data.size(-1) >= 1:
            dt_minutes = temporal_data[:, :, 0]  # [N, N]
            
        # Apply optimized attention
        if self.config.gradient_checkpointing and self.training:
            attended_values, attention_weights = torch.utils.checkpoint.checkpoint(
                self.attention, Q, K, V, edge_index, dt_minutes, self.training
            )
        else:
            attended_values, attention_weights = self.attention(
                Q, K, V, edge_index, dt_minutes, self.training
            )
            
        # Reshape back: [1, H, N, D] -> [N, hidden_dim]
        attended_values = attended_values.transpose(1, 2).contiguous().view(N, self.hidden_dim)
        
        # Output projection
        output_features = self.output_projection(attended_values)
        
        # Return appropriate attention weights format
        if return_attn and attention_weights is not None:
            # Average across heads: [1, H, N, N] -> [N, N]
            attention_weights = attention_weights.mean(dim=1).squeeze(0)
            
        return output_features, attention_weights if return_attn else None
        
    def _create_fused_temporal_encoding(self):
        """Create fused temporal encoding for better performance"""
        # Context7 recommendation: Fuse operations where possible
        return nn.Sequential(
            nn.Linear(2, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
    def _initialize_weights(self):
        """Apply optimized weight initialization"""
        # Xavier/Glorot initialization for attention weights
        for module in [self.query, self.key, self.value, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
            
        # Smaller initialization for projection layers
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.5)
        nn.init.constant_(self.input_projection.bias, 0)


class OptimizedTGATDiscoveryEngine(nn.Module):
    """
    Context7-optimized TGAT Discovery Engine with comprehensive performance improvements
    """
    
    def __init__(self, config: OptimizedTGATConfig):
        super().__init__()
        
        self.config = config
        base_cfg = config.base_config
        
        # Create optimized attention layers
        self.attention_layers = nn.ModuleList([
            OptimizedEnhancedTemporalAttentionLayer(config)
            for _ in range(base_cfg.num_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(base_cfg.hidden_dim)
            for _ in range(base_cfg.num_layers)
        ])
        
        # Residual dropout
        self.dropout = nn.Dropout(0.1)
        
        logger.info(f"Optimized TGAT Discovery Engine initialized with {base_cfg.num_layers} layers")
        
    def forward(self, node_features: torch.Tensor, edge_features: Optional[torch.Tensor],
                temporal_data: Optional[torch.Tensor], dag: Optional[nx.DiGraph] = None,
                return_all_attentions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through optimized TGAT layers
        
        Args:
            node_features: [N, input_dim] input node features
            edge_features: [E, edge_dim] edge features  
            temporal_data: [N, N, 2] temporal relationships
            dag: NetworkX DAG structure
            return_all_attentions: Whether to return attention from all layers
            
        Returns:
            Dictionary containing:
                - 'node_features': [N, hidden_dim] enhanced node features
                - 'attention_weights': Optional attention weights
        """
        current_features = node_features
        all_attentions = [] if return_all_attentions else None
        
        for i, (attn_layer, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Apply attention layer
            enhanced_features, attention_weights = attn_layer(
                current_features, edge_features, temporal_data, dag,
                return_attn=return_all_attentions or (i == len(self.attention_layers) - 1)
            )
            
            # Residual connection and layer norm
            current_features = layer_norm(enhanced_features + current_features)
            current_features = self.dropout(current_features)
            
            if return_all_attentions and attention_weights is not None:
                all_attentions.append(attention_weights)
                
        result = {
            'node_features': current_features,
        }
        
        if return_all_attentions:
            result['all_attentions'] = all_attentions
        elif attention_weights is not None:
            result['attention_weights'] = attention_weights
            
        return result


def create_optimized_tgat_layer(base_config: TGATConfig, 
                               enable_optimizations: bool = True) -> OptimizedEnhancedTemporalAttentionLayer:
    """
    Factory function to create optimized TGAT layer with Context7 recommendations
    
    Args:
        base_config: Base TGAT configuration
        enable_optimizations: Whether to enable Context7 optimizations
        
    Returns:
        Optimized TGAT attention layer
    """
    
    opt_config = OptimizedTGATConfig(
        base_config=base_config,
        enable_amp=enable_optimizations and torch.cuda.is_available(),
        enable_flash_attention=enable_optimizations and torch.cuda.is_available(),
        enable_time_bias_caching=enable_optimizations,
        use_fp16=enable_optimizations and torch.cuda.is_available(),
        enable_fused_ops=enable_optimizations,
        memory_efficient_attention=enable_optimizations
    )
    
    return OptimizedEnhancedTemporalAttentionLayer(opt_config)


# Backward compatibility
def create_attention_layer(input_dim: int = 45, hidden_dim: int = 44, 
                         num_heads: int = 4, cfg: Optional[TGATConfig] = None,
                         enable_optimizations: bool = False) -> nn.Module:
    """Create attention layer with optional Context7 optimizations"""
    
    if cfg is None:
        cfg = TGATConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim, 
            num_heads=num_heads
        )
        
    if enable_optimizations:
        return create_optimized_tgat_layer(cfg, enable_optimizations=True)
    else:
        # Import original layer for compatibility
        from .tgat_discovery import EnhancedTemporalAttentionLayer
        return EnhancedTemporalAttentionLayer(input_dim, hidden_dim, num_heads, cfg)