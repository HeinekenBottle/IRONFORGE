"""
Runtime-Optimized TGAT Discovery with STRICT vs COMPAT Mode Support
Integrates with RuntimeConfig system for acceleration management
"""

import logging
import warnings
from typing import Optional, Tuple, Dict, Any, Union
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from ..sdk.runtime_config import (
    RuntimeConfig, 
    AccelerationState, 
    RuntimeModeEnforcer,
    AccelStatus,
    RuntimeMode,
    initialize_runtime_system
)
from .dual_graph_config import TGATConfig
from .tgat_discovery import build_edge_mask, build_time_bias

logger = logging.getLogger(__name__)


class RuntimeAwareGraphAttention(nn.Module):
    """Graph attention with runtime mode enforcement"""
    
    def __init__(self, runtime_config: RuntimeConfig):
        super().__init__()
        self.runtime_config = runtime_config
        self._acceleration_state = None
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                edge_mask_bool: Optional[torch.Tensor] = None,
                time_bias: Optional[torch.Tensor] = None,
                dropout_p: float = 0.0,
                is_causal: bool = False,
                training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Runtime-aware graph attention forward pass
        
        Args:
            q, k, v: [B,H,L,D] query, key, value tensors
            edge_mask_bool: [B,1,L,S] boolean mask where True = block attention  
            time_bias: [B,1,L,S] additive bias to attention logits
            dropout_p: Dropout probability for attention weights
            is_causal: Apply causal masking
            training: Whether model is in training mode
            
        Returns:
            (out, attn_probs): Attention output and probabilities (None for SDPA)
        """
        
        # Detect acceleration state if not cached
        if self._acceleration_state is None:
            from ..sdk.runtime_config import AccelerationDetector
            self._acceleration_state = AccelerationDetector.detect_all()
        
        # Route to appropriate implementation based on runtime mode
        if self.runtime_config.mode == RuntimeMode.STRICT:
            return self._strict_mode_attention(
                q, k, v, edge_mask_bool, time_bias, dropout_p, is_causal, training
            )
        else:
            return self._compat_mode_attention(
                q, k, v, edge_mask_bool, time_bias, dropout_p, is_causal, training
            )
    
    def _strict_mode_attention(self, q, k, v, edge_mask_bool, time_bias, 
                              dropout_p, is_causal, training) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STRICT mode: fail fast if required acceleration unavailable"""
        
        # Check SDPA requirement
        if self.runtime_config.enable_sdpa and self._acceleration_state.sdpa != AccelStatus.USED:
            raise RuntimeError(
                "STRICT mode: SDPA (Scaled Dot Product Attention) is required but unavailable. "
                "Please upgrade to PyTorch >= 2.0 or set IRONFORGE_RUNTIME_MODE=compat"
            )
        
        # Check Flash Attention requirement
        if (self.runtime_config.enable_flash_attention and 
            self._acceleration_state.flash != AccelStatus.USED and
            torch.cuda.is_available()):
            raise RuntimeError(
                "STRICT mode: Flash Attention is required but unavailable. "
                "Please ensure PyTorch has Flash Attention support or set IRONFORGE_RUNTIME_MODE=compat"
            )
        
        # Use SDPA implementation (required in STRICT mode)
        return self._sdpa_attention(q, k, v, edge_mask_bool, time_bias, dropout_p, is_causal, training)
    
    def _compat_mode_attention(self, q, k, v, edge_mask_bool, time_bias,
                              dropout_p, is_causal, training) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """COMPAT mode: graceful fallback with degradation logging"""
        
        # Try SDPA first if available
        if self._acceleration_state.sdpa == AccelStatus.USED:
            try:
                return self._sdpa_attention(q, k, v, edge_mask_bool, time_bias, dropout_p, is_causal, training)
            except Exception as e:
                logger.warning(f"COMPAT mode: SDPA failed, falling back to manual: {e}")
        
        # Log degradation
        if self._acceleration_state.is_degraded():
            reasons = self._acceleration_state.get_degradation_reasons()
            logger.warning("COMPAT mode: Running with degraded performance")
            for reason in reasons:
                logger.warning(f"  - {reason}")
        
        # Fallback to manual implementation
        return self._manual_attention(q, k, v, edge_mask_bool, time_bias, dropout_p, is_causal, training)
    
    def _sdpa_attention(self, q, k, v, edge_mask_bool, time_bias,
                       dropout_p, is_causal, training) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """SDPA implementation"""
        
        try:
            from torch.nn.functional import scaled_dot_product_attention as sdpa
        except ImportError:
            raise RuntimeError("SDPA not available - please upgrade to PyTorch >= 2.0")
        
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
        
        # Apply SDPA
        out = sdpa(q, k, v,
                  attn_mask=attn_mask_float,
                  dropout_p=dropout_p if training else 0.0,
                  is_causal=is_causal)
        
        return out, None
    
    def _manual_attention(self, q, k, v, edge_mask_bool, time_bias,
                         dropout_p, is_causal, training) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Manual attention implementation (fallback)"""
        
        import math
        
        B, H, L, D = q.shape
        S = k.shape[-2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        
        # Apply time bias
        if time_bias is not None:
            scores = scores + time_bias
        
        # Apply edge mask
        if edge_mask_bool is not None:
            scores = scores.masked_fill(edge_mask_bool, -1e9)
        
        # Apply causal mask
        if is_causal:
            causal = torch.ones(L, S, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(causal, -1e9)
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        
        # Dropout
        if training and dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        
        # Apply to values
        out = torch.matmul(attn, v)
        
        return out, attn


class RuntimeOptimizedTGATLayer(nn.Module):
    """TGAT layer with runtime mode support and AMP integration"""
    
    def __init__(self, config: TGATConfig, runtime_config: RuntimeConfig):
        super().__init__()
        
        self.config = config
        self.runtime_config = runtime_config
        
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Core attention components
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Runtime-aware attention module
        self.attention = RuntimeAwareGraphAttention(runtime_config)
        
        # Cache acceleration state
        self._acceleration_state = None
        
        logger.info(f"Runtime-Optimized TGAT Layer: {self.input_dim}D→{self.hidden_dim}D, "
                   f"{self.num_heads} heads, mode={runtime_config.mode.value}")
    
    def forward(self, node_features: torch.Tensor, edge_features: Optional[torch.Tensor],
                temporal_data: Optional[torch.Tensor], dag: Optional[nx.DiGraph] = None,
                return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Runtime-optimized forward pass with AMP support
        
        Args:
            node_features: [N, input_dim] node features
            edge_features: [E, edge_dim] edge features (unused)
            temporal_data: [N, N, 2] temporal distance matrix
            dag: NetworkX DAG for edge connectivity
            return_attn: Whether to return attention weights
            
        Returns:
            (output_features, attention_weights): Enhanced features and optional attention
        """
        
        # Detect acceleration state if not cached
        if self._acceleration_state is None:
            from ..sdk.runtime_config import AccelerationDetector
            self._acceleration_state = AccelerationDetector.detect_all()
        
        # Determine if we should use AMP
        use_amp = (
            self.runtime_config.enable_amp and 
            self._acceleration_state.amp == AccelStatus.USED and
            torch.cuda.is_available() and
            node_features.device.type == 'cuda'
        )
        
        # Route based on AMP availability
        if use_amp:
            return self._forward_with_amp(node_features, edge_features, temporal_data, dag, return_attn)
        else:
            if self.runtime_config.mode == RuntimeMode.STRICT and self.runtime_config.enable_amp:
                # In STRICT mode, fail if AMP is required but not available
                raise RuntimeError(
                    "STRICT mode: AMP is required but unavailable. "
                    "Please ensure CUDA is available or set IRONFORGE_RUNTIME_MODE=compat"
                )
            
            # Log degradation in COMPAT mode
            if self.runtime_config.mode == RuntimeMode.COMPAT and self.runtime_config.enable_amp:
                logger.warning("COMPAT mode: AMP unavailable, using FP32")
            
            return self._forward_standard(node_features, edge_features, temporal_data, dag, return_attn)
    
    def _forward_with_amp(self, node_features, edge_features, temporal_data, dag, return_attn):
        """Forward pass with Automatic Mixed Precision"""
        
        try:
            from torch.cuda.amp import autocast
        except ImportError:
            # Fallback for older PyTorch versions
            warnings.warn("AMP requested but torch.cuda.amp unavailable, falling back to FP32")
            return self._forward_standard(node_features, edge_features, temporal_data, dag, return_attn)
        
        with autocast():
            return self._forward_core(node_features, edge_features, temporal_data, dag, return_attn)
    
    def _forward_standard(self, node_features, edge_features, temporal_data, dag, return_attn):
        """Standard FP32 forward pass"""
        return self._forward_core(node_features, edge_features, temporal_data, dag, return_attn)
    
    def _forward_core(self, node_features, edge_features, temporal_data, dag, return_attn):
        """Core forward logic (shared between AMP and standard)"""
        
        N = node_features.size(0)
        device = node_features.device
        
        # Project input features
        projected_features = self.input_projection(node_features)
        
        # Generate Q, K, V
        Q = self.query(projected_features)
        K = self.key(projected_features)
        V = self.value(projected_features)
        
        # Reshape for multi-head attention: [N, hidden_dim] -> [1, num_heads, N, head_dim]
        Q = Q.view(1, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(1, N, self.num_heads, self.head_dim).transpose(1, 2) 
        V = V.view(1, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Extract edge connectivity from DAG
        edge_index = None
        if dag is not None:
            edges = list(dag.edges())
            if edges:
                edge_index = torch.tensor(edges, device=device).T
        
        # Build edge mask if needed
        edge_mask = None
        if self.config.use_edge_mask and edge_index is not None:
            edge_mask = build_edge_mask(edge_index, N, device=device, allow_self=True)
        
        # Build temporal bias if needed
        time_bias = None
        if self.config.use_time_bias != "none" and temporal_data is not None:
            dt_minutes = temporal_data[:, :, 0]
            
            if self.config.use_time_bias == "bucket":
                buckets = [
                    (0, 5, 0.2),      # 0-5 minutes: strong positive bias
                    (5, 15, 0.1),     # 5-15 minutes: moderate positive bias
                    (15, 60, 0.0),    # 15-60 minutes: neutral
                    (60, 240, -0.1),  # 1-4 hours: slight negative bias
                ]
                time_bias = build_time_bias(dt_minutes, N, device=device, buckets=buckets)
            elif self.config.use_time_bias == "rbf":
                time_bias = build_time_bias(dt_minutes, N, device=device, scale=0.1)
        
        # Apply runtime-aware attention
        attended_values, attention_weights = self.attention(
            Q, K, V,
            edge_mask_bool=edge_mask,
            time_bias=time_bias,
            dropout_p=0.1,
            is_causal=self.config.is_causal,
            training=self.training
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


class RuntimeOptimizedTGATDiscovery(nn.Module):
    """Runtime-optimized TGAT Discovery with comprehensive mode enforcement"""
    
    def __init__(self, base_config: TGATConfig, runtime_config: Optional[RuntimeConfig] = None):
        super().__init__()
        
        # Initialize runtime system if not provided
        if runtime_config is None:
            runtime_config, _, _ = initialize_runtime_system()
        
        self.base_config = base_config
        self.runtime_config = runtime_config
        
        # Create optimized attention layers
        self.attention_layers = nn.ModuleList([
            RuntimeOptimizedTGATLayer(base_config, runtime_config)
            for _ in range(base_config.num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(base_config.hidden_dim)
            for _ in range(base_config.num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        logger.info(f"Runtime-Optimized TGAT Discovery: {base_config.num_layers} layers, "
                   f"mode={runtime_config.mode.value}")
    
    def forward(self, node_features: torch.Tensor, edge_features: Optional[torch.Tensor],
                temporal_data: Optional[torch.Tensor], dag: Optional[nx.DiGraph] = None,
                return_all_attentions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with runtime optimizations and mode enforcement
        
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


def create_runtime_optimized_tgat(base_config: TGATConfig, 
                                 runtime_config: Optional[RuntimeConfig] = None) -> RuntimeOptimizedTGATDiscovery:
    """
    Factory function to create runtime-optimized TGAT with mode enforcement
    
    Args:
        base_config: Base TGAT configuration
        runtime_config: Runtime configuration (auto-detected if None)
        
    Returns:
        Runtime-optimized TGAT discovery engine
    """
    
    if runtime_config is None:
        runtime_config, accel_state, enforcer = initialize_runtime_system()
        
        # Create audit entry for initialization
        audit_entry = enforcer.create_audit_entry(accel_state)
        enforcer.save_audit_entry(audit_entry)
    
    return RuntimeOptimizedTGATDiscovery(base_config, runtime_config)


# Backward compatibility wrapper
def create_optimized_attention_layer(input_dim: int = 45, hidden_dim: int = 44, 
                                   num_heads: int = 4, cfg: Optional[TGATConfig] = None,
                                   enable_runtime_optimization: bool = True) -> nn.Module:
    """Create attention layer with optional runtime optimization"""
    
    if cfg is None:
        cfg = TGATConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
    
    if enable_runtime_optimization:
        runtime_config, _, _ = initialize_runtime_system()
        return RuntimeOptimizedTGATLayer(cfg, runtime_config)
    else:
        # Import original layer for compatibility
        from .tgat_discovery import EnhancedTemporalAttentionLayer
        return EnhancedTemporalAttentionLayer(input_dim, hidden_dim, num_heads, cfg)