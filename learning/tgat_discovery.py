"""
IRONFORGE TGAT Discovery Engine
Self-supervised learning for pattern discovery without prediction
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Any
import json
import os
import math

class TGAT(torch.nn.Module):
    """
    Temporal Graph Attention Network for IRONFORGE Archaeological Discovery
    
    This implementation maintains full temporal pattern discovery capabilities,
    specifically designed for finding distant time-price relationships in
    financial market data. Preserves the core attention mechanisms needed
    for archaeological discovery of market structure patterns.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_of_heads: int = 4, concat: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_of_heads = num_of_heads
        self.concat = concat
        
        # Temporal encoding for archaeological time-distance relationships
        self.temporal_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, out_channels),
            torch.nn.LayerNorm(out_channels)
        )
        
        # Multi-head attention for different archaeological pattern types
        # (session structures, cascade patterns, liquidity sweeps, etc.)
        # PROPER 45D HANDLING: Project to nearest divisible dimension, then back to 45D
        self.head_dim = max(1, in_channels // num_of_heads)  # 45//4 = 11
        self.projected_dim = self.head_dim * num_of_heads    # 11*4 = 44
        
        # Project 45D → 44D for attention, then back to 45D
        self.input_projection = torch.nn.Linear(in_channels, self.projected_dim)
        self.multi_head_attention = torch.nn.MultiheadAttention(
            embed_dim=self.projected_dim,
            num_heads=num_of_heads,
            dropout=0.1,
            batch_first=True
        )
        # Project back to full 45D to preserve all archaeological information
        self.output_projection = torch.nn.Linear(self.projected_dim, in_channels)
        
        # Temporal relation scoring for distant correlations
        # UPDATED FOR SEMANTIC EVENTS: handles 45D features
        self.temporal_relation_scorer = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2 + out_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        
        # Archaeological pattern aggregation
        final_dim = out_channels * num_of_heads if concat else out_channels
        # Use in_channels for enhanced_features (after projection back to 37D) and out_channels for temporal context
        aggregation_input_dim = in_channels + out_channels
        self.pattern_aggregator = torch.nn.Sequential(
            torch.nn.Linear(aggregation_input_dim, final_dim),
            torch.nn.LayerNorm(final_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        # Learnable temporal decay for archaeological discovery
        self.temporal_decay = torch.nn.Parameter(torch.tensor(0.1))
        
        # Initialize weights for archaeological pattern recognition
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights optimized for temporal pattern discovery"""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_times: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with full archaeological discovery capabilities
        
        Maintains temporal attention mechanisms for discovering:
        - Distant session structure correlations
        - Multi-timeframe cascade patterns  
        - Liquidity sweep archaeological patterns
        - Price action memory effects
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connections [2, num_edges]
            edge_times: Temporal distances [num_edges]
            edge_attr: Rich edge features [num_edges, 17] for scale edge detection
        
        Returns:
            Enhanced embeddings with cross-timeframe patterns [num_nodes, final_dim]
        """
        num_nodes = x.shape[0]
        device = x.device
        
        # Step 1: Encode temporal relationships for archaeological discovery
        temporal_features = self._encode_temporal_relationships(edge_times, device)
        
        # Step 2: Enhanced processing with scale edge awareness
        if edge_attr is not None:
            # Detect scale edges using edge attributes (hierarchy_distance > 0)
            scale_edge_mask = self._detect_scale_edges(edge_attr)
            enhanced_features = self._apply_scale_aware_attention(
                x, temporal_features, edge_index, edge_attr, scale_edge_mask
            )
        else:
            # Fallback to temporal-only attention
            enhanced_features = self._apply_temporal_attention(x, temporal_features)
        
        # Step 3: Score relations with scale edge preference
        relation_scores = self._score_temporal_relations(x, edge_index, temporal_features)
        
        # Step 4: Aggregate patterns with cross-timeframe weighting
        archaeological_embeddings = self._aggregate_archaeological_patterns(
            x, enhanced_features, temporal_features, relation_scores, edge_index
        )
        
        return archaeological_embeddings
    
    def _encode_temporal_relationships(self, edge_times: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Encode temporal distances for archaeological pattern recognition"""
        if edge_times.numel() == 0:
            # Return default temporal encoding for sessions without explicit edges
            return torch.zeros(1, self.out_channels, device=device)
        
        # Normalize temporal distances for archaeological analysis
        # Longer distances get special encoding for distant pattern discovery
        normalized_times = edge_times.unsqueeze(-1)
        
        # Apply temporal decay for archaeological relevance weighting
        temporal_weights = torch.exp(-torch.abs(normalized_times) * self.temporal_decay)
        weighted_times = normalized_times * temporal_weights
        
        return self.temporal_encoder(weighted_times)
    
    def _apply_temporal_attention(self, x: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention with temporal archaeological awareness"""
        # Project 37D → 36D for attention computation
        x_projected = self.input_projection(x)
        
        # Prepare for attention computation
        x_batch = x_projected.unsqueeze(0)  # [1, num_nodes, projected_dim]
        
        # Apply multi-head attention for archaeological pattern discovery
        attn_output, attn_weights = self.multi_head_attention(
            query=x_batch,
            key=x_batch, 
            value=x_batch
        )
        
        # Project back to full 37D to preserve all archaeological information
        attn_output_squeezed = attn_output.squeeze(0)  # [num_nodes, projected_dim]
        return self.output_projection(attn_output_squeezed)  # [num_nodes, in_channels=37]
    
    def _score_temporal_relations(self, x: torch.Tensor, edge_index: torch.Tensor, 
                                temporal_features: torch.Tensor) -> torch.Tensor:
        """Score temporal relations for archaeological pattern importance"""
        if edge_index.shape[1] == 0:
            return torch.zeros(1, device=x.device)
        
        # Get source and target node features
        src_features = x[edge_index[0]]  # [num_edges, in_channels]
        tgt_features = x[edge_index[1]]  # [num_edges, in_channels]
        
        # Broadcast temporal features to match edge dimensions
        if temporal_features.shape[0] == 1:
            temp_features = temporal_features.expand(src_features.shape[0], -1)
        else:
            temp_features = temporal_features
        
        # Concatenate for relation scoring
        relation_input = torch.cat([src_features, tgt_features, temp_features], dim=-1)
        
        return self.temporal_relation_scorer(relation_input).squeeze(-1)
    
    def _aggregate_archaeological_patterns(self, x: torch.Tensor, enhanced_features: torch.Tensor,
                                         temporal_features: torch.Tensor, relation_scores: torch.Tensor,
                                         edge_index: torch.Tensor) -> torch.Tensor:
        """Aggregate patterns with archaeological temporal weighting"""
        num_nodes = x.shape[0]
        
        # Create temporal context for each node
        if temporal_features.shape[0] == 1:
            node_temporal_context = temporal_features.expand(num_nodes, -1)
        else:
            # Aggregate temporal features per node based on edge connections
            node_temporal_context = torch.zeros(num_nodes, temporal_features.shape[-1], device=x.device)
            
            if edge_index.shape[1] > 0 and relation_scores.numel() > 0:
                # Weight temporal features by relation scores for archaeological importance
                weighted_temporal = temporal_features * relation_scores.unsqueeze(-1)
                
                # Aggregate weighted temporal features to nodes
                for i in range(edge_index.shape[1]):
                    src_node = edge_index[0, i]
                    if i < weighted_temporal.shape[0]:
                        node_temporal_context[src_node] += weighted_temporal[i]
        
        # Combine original features with enhanced attention and temporal context
        combined_features = torch.cat([enhanced_features, node_temporal_context], dim=-1)
        
        return self.pattern_aggregator(combined_features)
    
    def _detect_scale_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Detect scale edges from edge attributes"""
        # Scale edges have hierarchy_distance > 0 (feature index 16 in RichEdgeFeature)
        # This identifies cross-timeframe connections (1m->5m, 1m->15m, etc.)
        hierarchy_distances = edge_attr[:, 16]  # hierarchy_distance feature
        scale_mask = hierarchy_distances > 0.0
        return scale_mask
    
    def _apply_scale_aware_attention(self, x: torch.Tensor, temporal_features: torch.Tensor,
                                   edge_index: torch.Tensor, edge_attr: torch.Tensor,
                                   scale_edge_mask: torch.Tensor) -> torch.Tensor:
        """Enhanced attention that prioritizes scale edges for cross-TF discovery"""
        
        # Start with base temporal attention
        base_features = self._apply_temporal_attention(x, temporal_features)
        
        if not scale_edge_mask.any() or edge_index.shape[1] == 0:
            return base_features
        
        # Create scale edge enhanced features
        scale_edges = edge_index[:, scale_edge_mask]  # Only scale edges
        scale_edge_attrs = edge_attr[scale_edge_mask]
        
        if scale_edges.shape[1] == 0:
            return base_features
        
        # Extract scale-specific information from edge attributes
        # timeframe_jump (index 5), hierarchy_distance (index 16), scale_from/to (index 12/13)
        timeframe_jumps = scale_edge_attrs[:, 5]  # Cross-TF jump distance
        hierarchy_distances = scale_edge_attrs[:, 16]  # TF hierarchy distance
        
        # Create scale-aware node updates
        num_nodes = x.shape[0]
        scale_node_updates = torch.zeros_like(base_features)
        
        # For each scale edge, propagate cross-timeframe information
        for i in range(scale_edges.shape[1]):
            src_idx = scale_edges[0, i]
            tgt_idx = scale_edges[1, i]
            
            # Weight by hierarchy distance (higher TF = more important)
            scale_weight = hierarchy_distances[i]
            tf_jump_weight = torch.sigmoid(timeframe_jumps[i])  # Normalize jump distance
            
            # Propagate target features to source (HTF info flows down)
            if src_idx < num_nodes and tgt_idx < num_nodes:
                cross_tf_info = base_features[tgt_idx] * scale_weight * tf_jump_weight * 0.3
                scale_node_updates[src_idx] += cross_tf_info
                # Also propagate in reverse (1m info flows up) but with less weight
                reverse_info = base_features[src_idx] * scale_weight * 0.1
                scale_node_updates[tgt_idx] += reverse_info
        
        # Combine base features with scale edge enhancements
        enhanced_features = base_features + scale_node_updates
        
        return enhanced_features

class IRONFORGEDiscovery:
    """
    Pure learning/logging system using TGAT
    NO PREDICTION - only pattern discovery
    """
    
    def __init__(self, node_features: int = 37, hidden_dim: int = 128, out_dim: int = 256):
        """
        UPDATED FOR TEMPORAL CYCLES: 37D features (34 relativity + 3 temporal cycles)
        Args:
            node_features: Input feature dimension (37D with temporal cycles)
            hidden_dim: Hidden layer dimension  
            out_dim: Output embedding dimension
        """
        self.model = TGAT(
            in_channels=node_features,
            out_channels=hidden_dim,
            num_of_heads=4,  # Multi-head attention for different pattern types
            concat=True
        )
        
        # Additional layers for richer representations
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 4 heads concatenated
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim * 2, out_dim)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projection.parameters()),
            lr=0.001
        )
        
        self.learning_only = True  # Enforce separation
        self.discovered_patterns = []
        
    def learn_session(self, X: torch.Tensor, edge_index: torch.Tensor, 
                     edge_times: torch.Tensor, metadata: Dict, edge_attr: Optional[torch.Tensor] = None) -> Dict:
        """
        Learn patterns from a single session using self-supervision
        
        Returns:
            Dictionary of learned embeddings and discovered patterns
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass through TGAT with edge attributes for scale edge awareness
        h = self.model(X, edge_index, edge_times, edge_attr)
        embeddings = self.projection(h)
        
        # Self-supervised losses
        losses = {}
        
        # 1. Temporal coherence: nearby events should have similar embeddings
        temporal_loss = self._temporal_coherence_loss(embeddings, edge_index, edge_times)
        losses['temporal'] = temporal_loss
        
        # 2. Event type clustering: similar events should cluster
        type_loss = self._event_type_loss(embeddings, X)
        losses['type'] = type_loss
        
        # 3. Cascade detection: learn cascade patterns
        cascade_loss = self._cascade_pattern_loss(embeddings, edge_index, metadata)
        losses['cascade'] = cascade_loss
        
        # Combined loss - ensure all components require gradients
        total_loss = temporal_loss + 0.5 * type_loss + 0.3 * cascade_loss
        
        # Ensure loss requires gradients
        if not total_loss.requires_grad:
            # Create a dummy loss that requires gradients if needed
            dummy_param_loss = sum(p.sum() * 0 for p in self.model.parameters() if p.requires_grad)
            total_loss = total_loss + dummy_param_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Extract patterns (including cross-timeframe patterns)
        patterns = self._extract_patterns(embeddings, X, metadata, edge_index, edge_attr)
        
        # TASK 5: Enhance patterns with rich semantic context
        enhanced_patterns = self._enhance_patterns_with_semantic_context(patterns, metadata, X, embeddings)
        
        return {
            'embeddings': embeddings.detach().cpu().numpy(),
            'patterns': enhanced_patterns,  # Use enhanced patterns instead of raw patterns
            'losses': {k: v.item() for k, v in losses.items()},
            'metadata': metadata
        }
    
    def _temporal_coherence_loss(self, embeddings: torch.Tensor, 
                                edge_index: torch.Tensor, 
                                edge_times: torch.Tensor) -> torch.Tensor:
        """Events close in time should have similar embeddings"""
        if edge_index.shape[1] == 0:
            # Return a tensor that requires gradients
            return torch.tensor(0.0, requires_grad=True)
            
        source_emb = embeddings[edge_index[0]]
        target_emb = embeddings[edge_index[1]]
        
        # Weight by inverse time distance
        time_weights = torch.exp(-edge_times / 10.0)  # Decay factor
        
        # Cosine similarity weighted by time
        cos_sim = F.cosine_similarity(source_emb, target_emb)
        loss = -torch.mean(cos_sim * time_weights)
        
        return loss
    
    def _event_type_loss(self, embeddings: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Similar event types should cluster together"""
        # Use simple regularization loss based on embeddings
        # This ensures we always have a gradient-enabled loss
        
        # L2 regularization on embeddings to encourage clustering
        embedding_norm = torch.mean(embeddings ** 2)
        
        # Add small variance-based clustering loss
        if embeddings.shape[0] > 1:
            embedding_var = torch.var(embeddings, dim=0)
            clustering_loss = -torch.mean(embedding_var)  # Encourage variance
            return 0.1 * embedding_norm + 0.01 * clustering_loss
        else:
            return 0.1 * embedding_norm
    
    def _cascade_pattern_loss(self, embeddings: torch.Tensor,
                             edge_index: torch.Tensor,
                             metadata: Dict) -> torch.Tensor:
        """Learn cascade sequence patterns"""
        # Simple gradient-enabled loss for cascade learning
        # Encourage temporal smoothness in embeddings
        if embeddings.shape[0] > 1:
            # Sequential difference loss (encourages smooth transitions)
            sequential_diff = embeddings[1:] - embeddings[:-1]
            smoothness_loss = torch.mean(sequential_diff ** 2)
            return 0.01 * smoothness_loss
        else:
            # Small regularization for single nodes
            return 0.001 * torch.mean(embeddings ** 2)
    
    def _extract_patterns(self, embeddings: torch.Tensor, X: torch.Tensor, 
                         metadata: Dict, edge_index: torch.Tensor, 
                         edge_attr: Optional[torch.Tensor] = None) -> List[Dict]:
        """
        SPRINT 2 ENHANCED: TGAT Pattern Extraction with 4 Edge Types & Regime Integration
        
        Extract permanent structural relationships using 37D enhanced features with Sprint 2 intelligence:
        - 4 Edge Types: temporal, scale, structural_context, discovered 
        - Regime Integration: Auto-assign regime labels to patterns
        - Enhanced Pattern Types: Structural context patterns + temporal cycles
        - Backward Compatible: Maintains all existing pattern extraction
        
        37D Features:
        - normalized_price positions (0.0-1.0 range positions)
        - pct_from_open relationships (percentage moves from session open) 
        - price_to_HTF_ratio cross-timeframe correlations
        - week_of_month cycles (1-5 for monthly pattern detection)
        - month_of_year cycles (1-12 for seasonal patterns)
        - day_of_week_cycle emphasis for weekly patterns
        
        Focuses on structural + temporal cycle patterns that survive regime changes.
        """
        patterns = []
        
        if X.shape[0] < 3:  # Need minimum nodes for pattern detection
            return patterns
        
        # SPRINT 2: Extract structural context patterns using 4th edge type
        structural_patterns = self._extract_structural_context_patterns(X, embeddings, metadata, edge_index, edge_attr)
        patterns.extend(structural_patterns)
        
        # Innovation Architect: Extract temporal cycle patterns
        cycle_patterns = self._extract_temporal_cycle_patterns(X, embeddings, metadata)
        patterns.extend(cycle_patterns)
        
        # Innovation Architect: Extract structural relativity patterns
        rel_patterns = self._extract_relativity_structural_patterns(X, embeddings, metadata)
        patterns.extend(rel_patterns)
        
        # Cross-timeframe confluence patterns using enhanced session data
        htf_patterns = self._extract_htf_confluence_patterns(X, embeddings, metadata)
        patterns.extend(htf_patterns)
        
        # Time-based structural position patterns
        time_patterns = self._extract_temporal_structural_patterns(X, embeddings, metadata)
        patterns.extend(time_patterns)
        
        # SPRINT 2: Add regime labels to all patterns
        patterns_with_regime = self._add_regime_labels_to_patterns(patterns, X, embeddings, metadata)
        
        return patterns_with_regime
    
    def _enhance_patterns_with_semantic_context(self, patterns: List[Dict], metadata: Dict, 
                                               X: torch.Tensor, embeddings: torch.Tensor) -> List[Dict]:
        """
        TASK 5: Enhance TGAT discovery output with rich semantic context
        
        Upgrades pattern output schema to include:
        - Semantic pattern naming with context
        - Session metadata (name, start/end times, anchor timeframe)
        - Linked event and phase information
        - Semantic relationships and causality
        - Enhanced confidence scoring with temporal context
        - Archaeological significance scoring
        
        Args:
            patterns: Raw patterns from _extract_patterns
            metadata: Graph metadata with session context and constant features
            X: Feature tensor (45D semantic features)
            embeddings: TGAT embeddings for confidence scoring
            
        Returns:
            Enhanced patterns with rich semantic context schema
        """
        enhanced_patterns = []
        
        # Extract session context from metadata
        graph_metadata = metadata.get('graph', {}).get('metadata', {})
        session_data = graph_metadata.get('preserved_raw', {})
        constant_features = metadata.get('constant_features', {})
        
        # Session context extraction
        session_name = self._extract_session_name(session_data)
        session_start, session_end = self._extract_session_timespan(session_data)
        anchor_timeframe = self._determine_anchor_timeframe(X, metadata)
        
        for i, pattern in enumerate(patterns):
            # Create semantic pattern ID
            pattern_id = self._generate_semantic_pattern_id(pattern, session_name, i)
            
            # Extract semantic context from features
            semantic_context = self._extract_semantic_context(pattern, X, session_data, constant_features)
            
            # Determine linked events and phase information
            linked_events = self._identify_linked_events(pattern, session_data)
            phase_info = self._extract_phase_information(pattern, X, session_data)
            
            # Calculate enhanced confidence with temporal context
            enhanced_confidence = self._calculate_enhanced_confidence(pattern, embeddings, X)
            
            # Calculate archaeological significance
            archaeological_significance = self._calculate_archaeological_significance(pattern, constant_features, X)
            
            # Build enhanced pattern schema
            enhanced_pattern = {
                # Core pattern information (preserved from original)
                'type': pattern.get('type', 'unknown_pattern'),
                'description': pattern.get('description', 'No description available'),
                
                # TASK 5: Rich semantic context additions
                'pattern_id': pattern_id,
                'session_name': session_name,
                'session_start': session_start,
                'session_end': session_end,
                'anchor_timeframe': anchor_timeframe,
                
                # Linked events and phase information
                'linked_events': linked_events,
                'phase_information': phase_info,
                
                # Semantic context with event types and relationships
                'semantic_context': semantic_context,
                
                # Enhanced confidence and temporal context
                'confidence': enhanced_confidence['confidence'],
                'temporal_context': enhanced_confidence['temporal_context'],
                'stability_score': enhanced_confidence['stability_score'],
                
                # Archaeological significance
                'archaeological_significance': archaeological_significance,
                
                # Original pattern data (preserved for backward compatibility)
                **{k: v for k, v in pattern.items() if k not in ['type', 'description']},
                
                # Metadata about semantic enhancement
                'semantic_enhancement': {
                    'version': '1.0',
                    'enhanced_features_used': constant_features.get('training_features', 45),
                    'constant_features_filtered': constant_features.get('metadata_only_count', 0),
                    'enhancement_timestamp': self._get_current_timestamp()
                }
            }
            
            enhanced_patterns.append(enhanced_pattern)
        
        return enhanced_patterns
    
    def _extract_session_name(self, session_data: Dict) -> str:
        """Extract semantic session name from session data"""
        session_meta = session_data.get('session_metadata', {})
        session_id = session_meta.get('session_id', 'unknown')
        
        # Extract meaningful parts of session ID for semantic naming
        if '_' in session_id:
            parts = session_id.split('_')
            # Try to identify session type (NY, London, Asia, etc.)
            session_type = next((part for part in parts if part in ['NY', 'London', 'Asia', 'PM', 'AM']), parts[0])
            return f"{session_type}_session"
        
        return session_id
    
    def _extract_session_timespan(self, session_data: Dict) -> tuple:
        """Extract session start and end times"""
        session_meta = session_data.get('session_metadata', {})
        session_start = session_meta.get('session_start', '09:30:00')
        session_duration = session_meta.get('session_duration', 120)
        
        # Calculate session end time
        import datetime
        try:
            start_time = datetime.datetime.strptime(session_start, '%H:%M:%S')
            end_time = start_time + datetime.timedelta(minutes=session_duration)
            session_end = end_time.strftime('%H:%M:%S')
        except:
            session_end = '11:30:00'  # Default 2-hour session
        
        return session_start, session_end
    
    def _determine_anchor_timeframe(self, X: torch.Tensor, metadata: Dict) -> str:
        """Determine the primary timeframe for this pattern discovery"""
        # Check edge types to determine dominant timeframe relationships
        edge_types = metadata.get('edge_types', [])
        
        # Count scale edges to determine multi-timeframe presence
        scale_edge_count = sum(1 for et in edge_types if et == 'scale')
        total_edges = len(edge_types)
        
        if scale_edge_count > total_edges * 0.3:  # >30% scale edges
            return 'multi_timeframe'
        else:
            return '1m'  # Default to 1-minute primary timeframe
    
    def _generate_semantic_pattern_id(self, pattern: Dict, session_name: str, index: int) -> str:
        """Generate semantic pattern ID with meaningful naming"""
        pattern_type = pattern.get('type', 'unknown')
        
        # Create semantic abbreviation
        type_abbrev = {
            'range_position_confluence': 'RPC',
            'session_open_relationship': 'SOR', 
            'high_low_distance': 'HLD',
            'temporal_structural': 'TSP',
            'htf_confluence': 'HTC',
            'scale_alignment': 'SAP',
            'cross_tf_confluence': 'CTC',
            'cascade_pattern': 'CSC',
            'weekly_cycle': 'WCY',
            'monthly_cycle': 'MCY'
        }.get(pattern_type, 'UNK')
        
        return f"{session_name}_{type_abbrev}_{index:02d}"
    
    def _extract_semantic_context(self, pattern: Dict, X: torch.Tensor, 
                                 session_data: Dict, constant_features: Dict) -> Dict:
        """Extract semantic context including event types and relationships"""
        # Analyze feature patterns to determine semantic context
        semantic_events = self._analyze_semantic_event_features(X)
        market_regime = self._determine_market_regime(session_data)
        structural_context = self._extract_structural_context(pattern, session_data)
        
        return {
            'event_types': semantic_events,
            'market_regime': market_regime,
            'structural_context': structural_context,
            'constant_features_context': {
                'filtered_count': constant_features.get('metadata_only_count', 0),
                'constant_names': constant_features.get('constant_feature_names', [])
            },
            'relationship_type': self._classify_pattern_relationships(pattern)
        }
    
    def _identify_linked_events(self, pattern: Dict, session_data: Dict) -> List[Dict]:
        """Identify events linked to this pattern"""
        linked_events = []
        
        # Extract liquidity events that may be related
        liquidity_events = session_data.get('session_liquidity_events', [])
        for event in liquidity_events:
            # Simple linkage based on pattern timing or type
            if self._is_event_linked_to_pattern(pattern, event):
                linked_events.append({
                    'event_type': event.get('type', 'liquidity_event'),
                    'timestamp': event.get('timestamp', ''),
                    'strength': self._calculate_event_linkage_strength(pattern, event)
                })
        
        return linked_events
    
    def _extract_phase_information(self, pattern: Dict, X: torch.Tensor, session_data: Dict) -> Dict:
        """Extract session phase information relevant to this pattern"""
        # Determine session phase from temporal position
        if X.shape[0] > 0:
            avg_session_position = torch.mean(X[:, 18] if X.shape[1] > 18 else torch.zeros(X.shape[0])).item()  # normalized_time
            
            if avg_session_position < 0.25:
                phase = 'session_opening'
            elif avg_session_position < 0.75:
                phase = 'session_middle'
            else:
                phase = 'session_closing'
        else:
            phase = 'unknown'
        
        return {
            'primary_phase': phase,
            'session_position': float(avg_session_position) if 'avg_session_position' in locals() else 0.5,
            'phase_significance': self._calculate_phase_significance(pattern, phase)
        }
    
    def _calculate_enhanced_confidence(self, pattern: Dict, embeddings: torch.Tensor, X: torch.Tensor) -> Dict:
        """Calculate enhanced confidence with temporal context"""
        # Base confidence from original pattern
        base_confidence = pattern.get('coherence_score', pattern.get('confidence', 0.5))
        
        # Temporal stability analysis
        temporal_stability = self._calculate_temporal_stability(embeddings, X)
        
        # Feature consistency analysis
        feature_consistency = self._calculate_feature_consistency(X)
        
        # Combined enhanced confidence
        enhanced_confidence = (base_confidence + temporal_stability + feature_consistency) / 3.0
        enhanced_confidence = max(0.0, min(1.0, enhanced_confidence))  # Clamp to [0,1]
        
        return {
            'confidence': float(enhanced_confidence),
            'temporal_context': {
                'stability_score': float(temporal_stability),
                'feature_consistency': float(feature_consistency),
                'base_confidence': float(base_confidence)
            },
            'stability_score': float(temporal_stability)
        }
    
    def _calculate_archaeological_significance(self, pattern: Dict, constant_features: Dict, X: torch.Tensor) -> Dict:
        """Calculate archaeological significance of the pattern"""
        # Pattern permanence - patterns that don't depend on constant features are more permanent
        constant_dependency = len(constant_features.get('constant_feature_names', [])) / max(1, X.shape[1])
        permanence_score = 1.0 - constant_dependency
        
        # Pattern generalizability
        node_count = pattern.get('node_count', 1)
        generalizability = min(1.0, node_count / 10.0)  # More nodes = more generalizable
        
        # Discovery epoch (how established this pattern is)
        discovery_epoch = pattern.get('discovery_epoch', 0)
        establishment_score = min(1.0, discovery_epoch / 100.0)
        
        significance = (permanence_score + generalizability + establishment_score) / 3.0
        
        return {
            'overall_significance': float(significance),
            'permanence_score': float(permanence_score),
            'generalizability_score': float(generalizability),
            'establishment_score': float(establishment_score),
            'archaeological_value': self._classify_archaeological_value(significance)
        }
    
    # Helper methods for semantic context extraction
    def _analyze_semantic_event_features(self, X: torch.Tensor) -> List[str]:
        """Analyze semantic event features (first 8 dimensions)"""
        if X.shape[1] < 8:
            return ['insufficient_features']
        
        events = []
        semantic_features = X[:, :8].mean(dim=0)  # Average semantic features across all nodes
        
        feature_names = ['fvg_redelivery', 'expansion_phase', 'consolidation', 'liq_sweep', 
                        'pd_array_interaction', 'session_phase', 'event_type', 'semantic_label']
        
        for i, avg_value in enumerate(semantic_features):
            if avg_value > 0.5:  # Threshold for presence
                events.append(feature_names[i])
        
        return events if events else ['no_dominant_events']
    
    def _determine_market_regime(self, session_data: Dict) -> str:
        """Determine market regime from session data"""
        contamination = session_data.get('contamination_analysis', {})
        if contamination:
            regime_score = contamination.get('htf_contamination', {}).get('htf_carryover_strength', 0.5)
            if regime_score > 0.7:
                return 'trending'
            elif regime_score < 0.3:
                return 'ranging'
            else:
                return 'transitional'
        return 'unknown'
    
    def _extract_structural_context(self, pattern: Dict, session_data: Dict) -> Dict:
        """Extract structural context for the pattern"""
        return {
            'liquidity_environment': session_data.get('session_liquidity_events', [])[:3],  # First 3 events for context
            'energy_state': session_data.get('energy_state', {}),
            'pattern_strength': pattern.get('pattern_strength', 0.5)
        }
    
    def _classify_pattern_relationships(self, pattern: Dict) -> str:
        """Classify the type of relationships this pattern represents"""
        pattern_type = pattern.get('type', '')
        
        if 'confluence' in pattern_type:
            return 'confluence_relationship'
        elif 'cascade' in pattern_type:
            return 'causal_relationship'
        elif 'cycle' in pattern_type:
            return 'temporal_relationship'
        elif 'structural' in pattern_type:
            return 'structural_relationship'
        else:
            return 'unknown_relationship'
    
    def _is_event_linked_to_pattern(self, pattern: Dict, event: Dict) -> bool:
        """Simple heuristic to determine if an event is linked to a pattern"""
        # This is a placeholder - could be enhanced with more sophisticated linkage analysis
        return len(event.get('timestamp', '')) > 0
    
    def _calculate_event_linkage_strength(self, pattern: Dict, event: Dict) -> float:
        """Calculate strength of linkage between pattern and event"""
        # Simple strength calculation based on pattern coherence
        return pattern.get('coherence_score', 0.5)
    
    def _calculate_phase_significance(self, pattern: Dict, phase: str) -> float:
        """Calculate significance of pattern for the given session phase"""
        # Different patterns have different significance in different phases
        phase_weights = {
            'session_opening': 0.8,
            'session_middle': 0.6,
            'session_closing': 0.9
        }
        return phase_weights.get(phase, 0.5)
    
    def _calculate_temporal_stability(self, embeddings: torch.Tensor, X: torch.Tensor) -> float:
        """Calculate temporal stability from embeddings"""
        if embeddings.shape[0] < 2:
            return 0.5
        
        # Calculate embedding variance as proxy for stability
        embedding_var = torch.var(embeddings, dim=0).mean()
        stability = 1.0 / (1.0 + embedding_var.item())  # Higher variance = lower stability
        return max(0.0, min(1.0, stability))
    
    def _calculate_feature_consistency(self, X: torch.Tensor) -> float:
        """Calculate feature consistency across nodes"""
        if X.shape[0] < 2:
            return 0.5
        
        # Calculate coefficient of variation for numerical features
        feature_stds = torch.std(X, dim=0)
        feature_means = torch.abs(torch.mean(X, dim=0)) + 1e-8  # Avoid division by zero
        cv = feature_stds / feature_means
        consistency = 1.0 - torch.mean(cv).item()  # Lower CV = higher consistency
        return max(0.0, min(1.0, consistency))
    
    def _classify_archaeological_value(self, significance: float) -> str:
        """Classify archaeological value based on significance score"""
        if significance > 0.8:
            return 'high_archaeological_value'
        elif significance > 0.6:
            return 'moderate_archaeological_value'
        elif significance > 0.4:
            return 'low_archaeological_value'
        else:
            return 'minimal_archaeological_value'
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for enhancement metadata"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _extract_relativity_structural_patterns(self, X: torch.Tensor, embeddings: torch.Tensor, 
                                              metadata: Dict) -> List[Dict]:
        """
        Innovation Architect: Extract structural patterns using price relativity features
        
        Focus on permanent relationships that survive regime changes:
        - Range position clusters (78% of range → confluence zones)
        - Session open relationships (consistent percentage moves)
        - High/Low distance patterns (structural support/resistance)
        """
        patterns = []
        
        # Extract key features (37D feature indices - TEMPORAL CYCLES)
        week_of_month = X[:, 9]      # Index 9 in 37D features (NEW)
        month_of_year = X[:, 10]     # Index 10 in 37D features (NEW)
        day_of_week_cycle = X[:, 11] # Index 11 in 37D features (NEW)
        
        normalized_price = X[:, 12]  # Index 12 in 37D features (shifted)
        pct_from_open = X[:, 13]     # Index 13 (shifted)
        pct_from_high = X[:, 14]     # Index 14 (shifted)
        pct_from_low = X[:, 15]      # Index 15 (shifted)
        time_since_open = X[:, 17]   # Index 17 (shifted)
        normalized_time = X[:, 18]   # Index 18 (shifted)
        
        # Pattern 1: Range Position Confluence (permanent structural zones)
        range_clusters = self._find_range_position_clusters(normalized_price, embeddings)
        for cluster in range_clusters:
            # Calculate time distribution in cluster
            cluster_times = time_since_open[cluster['nodes']].cpu().numpy()
            time_span_hours = (cluster_times.max() - cluster_times.min()) / 3600.0
            
            pattern = {
                'type': 'range_position_confluence',
                'description': f"{cluster['range_pct']:.1f}% of range @ {time_span_hours:.1f}h timeframe → HTF confluence",
                'range_position': float(cluster['range_pct']),
                'node_count': len(cluster['nodes']),
                'time_span_hours': float(time_span_hours),
                'coherence_score': float(cluster['coherence']),
                'session': metadata.get('session_name', 'unknown'),
                'relativity_type': 'range_position'
            }
            patterns.append(pattern)
        
        # Pattern 2: Session Open Relationship Patterns  
        open_move_patterns = self._find_session_open_patterns(pct_from_open, normalized_time, embeddings)
        for pattern_data in open_move_patterns:
            time_position = pattern_data['avg_time_position'] * 100
            pattern = {
                'type': 'session_open_relationship', 
                'description': f"{pattern_data['move_pct']:.1f}% from open @ {time_position:.0f}% session → structural move",
                'pct_from_open': float(pattern_data['move_pct']),
                'session_time_pct': float(time_position),
                'node_count': int(pattern_data['node_count']),
                'consistency_score': float(pattern_data['consistency']),
                'session': metadata.get('session_name', 'unknown'),
                'relativity_type': 'session_open'
            }
            patterns.append(pattern)
        
        # Pattern 3: High/Low Distance Structural Patterns
        extremes_patterns = self._find_high_low_distance_patterns(pct_from_high, pct_from_low, embeddings)
        for ext_pattern in extremes_patterns:
            pattern = {
                'type': 'high_low_distance_structure',
                'description': f"{ext_pattern['distance_type']} distance {ext_pattern['distance_pct']:.1f}% → structural {ext_pattern['structure_type']}",
                'distance_from_extreme': float(ext_pattern['distance_pct']),
                'extreme_type': ext_pattern['distance_type'],  # 'high' or 'low'
                'structure_type': ext_pattern['structure_type'],  # 'support' or 'resistance'
                'node_count': int(ext_pattern['node_count']),
                'structural_strength': float(ext_pattern['strength']),
                'session': metadata.get('session_name', 'unknown'),
                'relativity_type': 'high_low_distance'
            }
            patterns.append(pattern)
        
        return patterns
    
    def _extract_temporal_cycle_patterns(self, X: torch.Tensor, embeddings: torch.Tensor,
                                       metadata: Dict) -> List[Dict]:
        """
        Innovation Architect: Extract temporal cycle patterns for weekly/monthly detection
        
        Uses the new temporal cycle features:
        - week_of_month (1-5): Weekly patterns within months
        - month_of_year (1-12): Monthly/seasonal patterns
        - day_of_week_cycle (0-6): Weekly patterns across different timeframes
        """
        patterns = []
        
        # Extract temporal cycle features (37D indices)
        week_of_month = X[:, 9]      # Week of month (1-5)
        month_of_year = X[:, 10]     # Month of year (1-12)
        day_of_week_cycle = X[:, 11] # Day of week cycle (0-6)
        normalized_price = X[:, 12]  # For pattern correlation
        
        # Pattern 1: Weekly Cycle Detection (same week of month patterns)
        weekly_patterns = self._find_weekly_cycle_patterns(week_of_month, day_of_week_cycle, 
                                                          normalized_price, embeddings, metadata)
        patterns.extend(weekly_patterns)
        
        # Pattern 2: Monthly Cycle Detection (same month patterns)
        monthly_patterns = self._find_monthly_cycle_patterns(month_of_year, week_of_month,
                                                           normalized_price, embeddings, metadata)
        patterns.extend(monthly_patterns)
        
        # Pattern 3: Cross-Cycle Confluence (week + month alignment)
        confluence_patterns = self._find_cross_cycle_confluence(week_of_month, month_of_year, 
                                                              day_of_week_cycle, embeddings, metadata)
        patterns.extend(confluence_patterns)
        
        return patterns
    
    def _find_weekly_cycle_patterns(self, week_of_month: torch.Tensor, day_of_week_cycle: torch.Tensor,
                                  normalized_price: torch.Tensor, embeddings: torch.Tensor, 
                                  metadata: Dict) -> List[Dict]:
        """Find patterns that repeat at same week of month + day of week"""
        patterns = []
        
        # Key weekly cycle combinations (week + day patterns)
        key_weekly_cycles = [
            (1, 0), (1, 4),  # First week: Monday, Friday
            (2, 0), (2, 4),  # Second week: Monday, Friday  
            (3, 2), (3, 4),  # Third week: Wednesday, Friday
            (4, 0), (4, 4),  # Fourth week: Monday, Friday
            (5, 0)           # Fifth week: Monday (month end)
        ]
        
        for target_week, target_day in key_weekly_cycles:
            # Find nodes matching this weekly cycle
            cycle_mask = (week_of_month == target_week) & (day_of_week_cycle == target_day)
            cycle_indices = torch.where(cycle_mask)[0]
            
            if len(cycle_indices) >= 2:  # Need multiple occurrences for pattern
                # Calculate embedding coherence for this cycle
                cycle_embeddings = embeddings[cycle_indices]
                coherence = torch.mean(torch.mm(cycle_embeddings, cycle_embeddings.t())).item()
                
                # Calculate price consistency across cycle
                cycle_prices = normalized_price[cycle_indices]
                price_std = torch.std(cycle_prices).item()
                price_consistency = max(0, 1.0 - price_std)  # Lower std = higher consistency
                
                if coherence > 0.7 and price_consistency > 0.5:  # Strong cycle pattern
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    avg_price_pct = torch.mean(cycle_prices).item() * 100
                    
                    pattern = {
                        'type': 'weekly_cycle_pattern',
                        'description': f"Week {target_week} {day_names[target_day]} @ {avg_price_pct:.0f}% range → weekly cycle",
                        'week_of_month': int(target_week),
                        'day_of_week': int(target_day),
                        'day_name': day_names[target_day],
                        'occurrence_count': len(cycle_indices),
                        'embedding_coherence': float(coherence),
                        'price_consistency': float(price_consistency),
                        'avg_price_position': float(avg_price_pct),
                        'session': metadata.get('session_name', 'unknown'),
                        'cycle_type': 'weekly'
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _find_monthly_cycle_patterns(self, month_of_year: torch.Tensor, week_of_month: torch.Tensor,
                                   normalized_price: torch.Tensor, embeddings: torch.Tensor,
                                   metadata: Dict) -> List[Dict]:
        """Find patterns that repeat in same months"""
        patterns = []
        
        # Key monthly patterns (seasonal behavior)
        key_months = [1, 2, 3, 6, 9, 12]  # Jan, Feb, Mar, Jun, Sep, Dec (key seasonal months)
        
        for target_month in key_months:
            # Find nodes in this month
            month_mask = (month_of_year == target_month)
            month_indices = torch.where(month_mask)[0]
            
            if len(month_indices) >= 2:
                # Check for consistent week patterns within this month
                month_weeks = week_of_month[month_indices]
                month_prices = normalized_price[month_indices]
                month_embeddings = embeddings[month_indices]
                
                # Calculate monthly coherence
                month_coherence = torch.mean(torch.mm(month_embeddings, month_embeddings.t())).item()
                
                # Check if specific weeks are consistent
                dominant_week = torch.mode(month_weeks)[0].item()
                week_consistency = (month_weeks == dominant_week).float().mean().item()
                
                if month_coherence > 0.6 and week_consistency > 0.6:  # Strong monthly pattern
                    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    avg_price_pct = torch.mean(month_prices).item() * 100
                    
                    pattern = {
                        'type': 'monthly_cycle_pattern', 
                        'description': f"{month_names[target_month]} week {dominant_week} @ {avg_price_pct:.0f}% range → monthly cycle",
                        'month_of_year': int(target_month),
                        'month_name': month_names[target_month],
                        'dominant_week': int(dominant_week),
                        'occurrence_count': len(month_indices),
                        'embedding_coherence': float(month_coherence),
                        'week_consistency': float(week_consistency),
                        'avg_price_position': float(avg_price_pct),
                        'session': metadata.get('session_name', 'unknown'),
                        'cycle_type': 'monthly'
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _find_cross_cycle_confluence(self, week_of_month: torch.Tensor, month_of_year: torch.Tensor,
                                   day_of_week_cycle: torch.Tensor, embeddings: torch.Tensor,
                                   metadata: Dict) -> List[Dict]:
        """Find confluence where weekly and monthly cycles align"""
        patterns = []
        
        # Key confluence combinations (strong seasonal + weekly alignment)
        key_confluences = [
            (1, 1, 0),   # January, Week 1, Monday (New Year effect)
            (3, 3, 4),   # March, Week 3, Friday (Quarter-end)
            (6, 4, 4),   # June, Week 4, Friday (Mid-year)
            (9, 2, 0),   # September, Week 2, Monday (Fall patterns) 
            (12, 4, 4),  # December, Week 4, Friday (Year-end)
        ]
        
        for target_month, target_week, target_day in key_confluences:
            # Find nodes with this exact confluence
            confluence_mask = ((month_of_year == target_month) & 
                             (week_of_month == target_week) & 
                             (day_of_week_cycle == target_day))
            confluence_indices = torch.where(confluence_mask)[0]
            
            if len(confluence_indices) >= 1:  # Even single occurrences are significant for confluence
                # Calculate confluence strength
                confluence_embeddings = embeddings[confluence_indices]
                
                if len(confluence_indices) > 1:
                    confluence_strength = torch.mean(torch.mm(confluence_embeddings, confluence_embeddings.t())).item()
                else:
                    # Single occurrence - use embedding magnitude as strength
                    confluence_strength = torch.norm(confluence_embeddings[0]).item()
                
                if confluence_strength > 0.5:  # Significant confluence
                    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    
                    pattern = {
                        'type': 'cross_cycle_confluence',
                        'description': f"{month_names[target_month]} week {target_week} {day_names[target_day]} → seasonal confluence",
                        'month_of_year': int(target_month),
                        'month_name': month_names[target_month],
                        'week_of_month': int(target_week),
                        'day_of_week': int(target_day),
                        'day_name': day_names[target_day],
                        'occurrence_count': len(confluence_indices),
                        'confluence_strength': float(confluence_strength),
                        'session': metadata.get('session_name', 'unknown'),
                        'cycle_type': 'cross_cycle_confluence'
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _extract_htf_confluence_patterns(self, X: torch.Tensor, embeddings: torch.Tensor,
                                       edge_index: torch.Tensor, edge_attr: torch.Tensor) -> List[Dict]:
        """Extract cross-timeframe confluence using price_to_HTF_ratio"""
        patterns = []
        
        price_to_htf_ratio = X[:, 16]  # Index 16 in 37D features (shifted)
        
        # Find scale edges (cross-timeframe connections)
        scale_mask = self._detect_scale_edges(edge_attr)
        if not scale_mask.any():
            return patterns
        
        scale_edges = edge_index[:, scale_mask]
        
        # Find HTF ratio confluence across timeframes
        for i in range(scale_edges.shape[1]):
            src_idx = scale_edges[0, i].item()
            tgt_idx = scale_edges[1, i].item()
            
            if src_idx < X.shape[0] and tgt_idx < X.shape[0]:
                src_ratio = price_to_htf_ratio[src_idx].item()
                tgt_ratio = price_to_htf_ratio[tgt_idx].item()
                
                # Strong HTF confluence when ratios align
                ratio_similarity = 1.0 - min(1.0, abs(src_ratio - tgt_ratio) / 0.1)
                
                if ratio_similarity > 0.85:  # High cross-TF correlation
                    # Calculate timeframe hierarchy from edge attributes
                    hierarchy_dist = edge_attr[i, 16].item()  # hierarchy_distance
                    
                    pattern = {
                        'type': 'htf_confluence',
                        'description': f"HTF ratio {src_ratio:.3f} @ {hierarchy_dist:.0f} TF levels → cross-timeframe confluence",
                        'htf_ratio': float(src_ratio),
                        'timeframe_levels': int(hierarchy_dist),
                        'confluence_strength': float(ratio_similarity),
                        'src_node': int(src_idx),
                        'tgt_node': int(tgt_idx),
                        'relativity_type': 'htf_ratio'
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _extract_temporal_structural_patterns(self, X: torch.Tensor, embeddings: torch.Tensor,
                                            metadata: Dict) -> List[Dict]:
        """Extract time-based structural position patterns"""
        patterns = []
        
        normalized_time = X[:, 18]      # Index 18 in 37D features (shifted)
        normalized_price = X[:, 12]     # Index 12 in 37D features (shifted)
        time_since_open = X[:, 17]      # Index 17 in 37D features (shifted)
        
        # Find temporal structural positions
        time_clusters = self._find_temporal_clusters(normalized_time, normalized_price, embeddings)
        
        for cluster in time_clusters:
            time_position = cluster['time_position'] * 100
            price_position = cluster['price_position'] * 100
            hours_from_open = cluster['avg_hours_from_open']
            
            pattern = {
                'type': 'temporal_structural_position',
                'description': f"{time_position:.0f}% session @ {price_position:.0f}% range after {hours_from_open:.1f}h → structural timing",
                'session_time_pct': float(time_position),
                'range_position_pct': float(price_position), 
                'hours_from_open': float(hours_from_open),
                'node_count': int(cluster['node_count']),
                'temporal_coherence': float(cluster['coherence']),
                'session': metadata.get('session_name', 'unknown'),
                'relativity_type': 'temporal_structure'
            }
            patterns.append(pattern)
        
        return patterns
    
    def _find_range_position_clusters(self, normalized_price: torch.Tensor, 
                                    embeddings: torch.Tensor) -> List[Dict]:
        """Find clusters at specific range positions (e.g., 78% of range)"""
        clusters = []
        
        # Define key range levels for confluence detection
        key_levels = [0.2, 0.382, 0.5, 0.618, 0.78, 0.85]  # Fibonacci + key levels
        
        for level in key_levels:
            # Find nodes near this range level
            level_mask = torch.abs(normalized_price - level) < 0.05  # 5% tolerance
            level_indices = torch.where(level_mask)[0]
            
            if len(level_indices) >= 2:  # Need at least 2 nodes for cluster
                # Calculate embedding coherence for this level
                level_embeddings = embeddings[level_indices]
                coherence = torch.mean(torch.mm(level_embeddings, level_embeddings.t())).item()
                
                if coherence > 0.7:  # High coherence threshold
                    clusters.append({
                        'range_pct': level * 100,
                        'nodes': level_indices.cpu().numpy().tolist(),
                        'coherence': coherence
                    })
        
        return clusters
    
    def _find_session_open_patterns(self, pct_from_open: torch.Tensor, normalized_time: torch.Tensor,
                                  embeddings: torch.Tensor) -> List[Dict]:
        """Find consistent percentage moves from session open"""
        patterns = []
        
        # Key percentage levels from open
        move_levels = [-5.0, -2.5, -1.0, 1.0, 2.5, 5.0]  # Percentage moves
        
        for move_pct in move_levels:
            # Find nodes near this move percentage
            move_mask = torch.abs(pct_from_open - move_pct) < 0.5  # 0.5% tolerance
            move_indices = torch.where(move_mask)[0]
            
            if len(move_indices) >= 2:
                # Calculate timing consistency
                move_times = normalized_time[move_indices]
                time_std = torch.std(move_times).item()
                consistency = max(0, 1.0 - time_std)  # Lower std = higher consistency
                
                if consistency > 0.6:  # Consistent timing threshold
                    patterns.append({
                        'move_pct': move_pct,
                        'node_count': len(move_indices),
                        'avg_time_position': torch.mean(move_times).item(),
                        'consistency': consistency
                    })
        
        return patterns
    
    def _find_high_low_distance_patterns(self, pct_from_high: torch.Tensor, pct_from_low: torch.Tensor,
                                       embeddings: torch.Tensor) -> List[Dict]:
        """Find structural patterns based on distance from session high/low"""
        patterns = []
        
        # Key distance levels from extremes
        distance_levels = [5.0, 10.0, 20.0, 30.0, 50.0]  # Percentage distances
        
        # Check distances from high
        for dist_pct in distance_levels:
            high_mask = torch.abs(pct_from_high - dist_pct) < 2.0  # 2% tolerance
            high_indices = torch.where(high_mask)[0]
            
            if len(high_indices) >= 2:
                # Calculate structural strength using embeddings
                high_embeddings = embeddings[high_indices]
                strength = torch.mean(torch.norm(high_embeddings, dim=1)).item()
                
                if strength > 0.5:  # Structural significance threshold
                    patterns.append({
                        'distance_pct': dist_pct,
                        'distance_type': 'high',
                        'structure_type': 'resistance',
                        'node_count': len(high_indices),
                        'strength': strength
                    })
        
        # Check distances from low
        for dist_pct in distance_levels:
            low_mask = torch.abs(pct_from_low - dist_pct) < 2.0  # 2% tolerance  
            low_indices = torch.where(low_mask)[0]
            
            if len(low_indices) >= 2:
                # Calculate structural strength using embeddings
                low_embeddings = embeddings[low_indices]
                strength = torch.mean(torch.norm(low_embeddings, dim=1)).item()
                
                if strength > 0.5:  # Structural significance threshold
                    patterns.append({
                        'distance_pct': dist_pct,
                        'distance_type': 'low', 
                        'structure_type': 'support',
                        'node_count': len(low_indices),
                        'strength': strength
                    })
        
        return patterns
    
    def _find_temporal_clusters(self, normalized_time: torch.Tensor, normalized_price: torch.Tensor,
                              embeddings: torch.Tensor) -> List[Dict]:
        """Find clusters at specific temporal-price positions"""
        clusters = []
        
        # Define key temporal zones in session
        time_zones = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]  # Quarter segments
        
        for start_time, end_time in time_zones:
            # Find nodes in this time zone
            time_mask = (normalized_time >= start_time) & (normalized_time <= end_time)
            time_indices = torch.where(time_mask)[0]
            
            if len(time_indices) >= 3:  # Need enough nodes for cluster
                # Calculate cluster properties
                cluster_times = normalized_time[time_indices]
                cluster_prices = normalized_price[time_indices]
                cluster_embeddings = embeddings[time_indices]
                
                # Calculate temporal coherence
                coherence = 1.0 / (1.0 + torch.std(cluster_embeddings, dim=0).mean().item())
                
                if coherence > 0.6:  # Coherence threshold
                    # Convert time since open for hours calculation
                    avg_time_position = torch.mean(cluster_times).item()
                    avg_price_position = torch.mean(cluster_prices).item()
                    hours_from_open = avg_time_position * 5.0  # Assuming 5-hour sessions
                    
                    clusters.append({
                        'time_position': avg_time_position,
                        'price_position': avg_price_position,
                        'avg_hours_from_open': hours_from_open,
                        'node_count': len(time_indices),
                        'coherence': coherence
                    })
        
        return clusters
    
    def _extract_energy_clusters(self, embeddings: torch.Tensor, X: torch.Tensor) -> List[Dict]:
        """Extract high-energy node clusters"""
        emb_np = embeddings.detach().cpu().numpy()
        energy_scores = np.linalg.norm(emb_np, axis=1)
        high_energy_idx = np.where(energy_scores > np.percentile(energy_scores, 90))[0]
        
        if len(high_energy_idx) > 0:
            return [{
                'type': 'high_energy_cluster',
                'indices': high_energy_idx.tolist(),
                'times': X[high_energy_idx, 0].cpu().numpy().tolist() if X.shape[0] > 0 else [],
                'score': float(np.mean(energy_scores[high_energy_idx])),
                'timeframe': 'mixed'  # Could span multiple timeframes
            }]
        return []
    
    def _extract_scale_alignment_patterns(self, embeddings: torch.Tensor, edge_index: torch.Tensor, 
                                         edge_attr: torch.Tensor, metadata: Dict) -> List[Dict]:
        """Detect 1m->5m->15m->1h scale alignment patterns"""
        patterns = []
        
        # Identify scale edges
        scale_mask = self._detect_scale_edges(edge_attr)
        if not scale_mask.any():
            return patterns
            
        scale_edges = edge_index[:, scale_mask]
        scale_attrs = edge_attr[scale_mask]
        
        # Find scale chains (1m->5m->15m)
        hierarchy_distances = scale_attrs[:, 16]  # hierarchy_distance
        
        # Group by hierarchy levels
        hierarchy_groups = {}
        for i, h_dist in enumerate(hierarchy_distances):
            h_level = int(h_dist.item())
            if h_level not in hierarchy_groups:
                hierarchy_groups[h_level] = []
            hierarchy_groups[h_level].append(i)
        
        # Look for chains across hierarchy levels
        if len(hierarchy_groups) >= 2:
            # Find nodes that participate in multiple hierarchy levels
            multi_level_nodes = set()
            for level_edges in hierarchy_groups.values():
                for edge_idx in level_edges:
                    src_node = scale_edges[0, edge_idx].item()
                    tgt_node = scale_edges[1, edge_idx].item()
                    multi_level_nodes.update([src_node, tgt_node])
            
            if len(multi_level_nodes) >= 3:  # At least 3 nodes in scale alignment
                # Calculate alignment strength
                node_list = list(multi_level_nodes)
                node_embeddings = embeddings[node_list]
                alignment_score = torch.mean(torch.mm(node_embeddings, node_embeddings.t())).item()
                
                patterns.append({
                    'type': 'scale_alignment',
                    'nodes': node_list,
                    'hierarchy_levels': len(hierarchy_groups),
                    'alignment_score': alignment_score,
                    'timeframe': 'multi_tf'
                })
        
        return patterns
    
    def _extract_cross_tf_confluence_patterns(self, embeddings: torch.Tensor, edge_index: torch.Tensor,
                                            edge_attr: torch.Tensor, X: torch.Tensor) -> List[Dict]:
        """Detect price confluence across timeframes"""
        patterns = []
        
        scale_mask = self._detect_scale_edges(edge_attr)
        if not scale_mask.any():
            return patterns
            
        scale_edges = edge_index[:, scale_mask]
        
        # Find nodes with high embedding similarity across scale edges
        for i in range(scale_edges.shape[1]):
            src_idx = scale_edges[0, i].item()
            tgt_idx = scale_edges[1, i].item()
            
            if src_idx < embeddings.shape[0] and tgt_idx < embeddings.shape[0]:
                # Calculate cross-TF similarity
                src_emb = embeddings[src_idx]
                tgt_emb = embeddings[tgt_idx]
                tf_similarity = F.cosine_similarity(src_emb.unsqueeze(0), tgt_emb.unsqueeze(0)).item()
                
                if tf_similarity > 0.85:  # High cross-timeframe similarity
                    patterns.append({
                        'type': 'cross_tf_confluence',
                        'source_node': src_idx,
                        'target_node': tgt_idx,
                        'similarity_score': tf_similarity,
                        'timeframe': 'cross_tf'
                    })
        
        return patterns
    
    def _extract_htf_cascade_patterns(self, embeddings: torch.Tensor, edge_index: torch.Tensor,
                                    edge_attr: torch.Tensor, metadata: Dict) -> List[Dict]:
        """Detect HTF-driven cascade patterns"""
        patterns = []
        
        scale_mask = self._detect_scale_edges(edge_attr)
        if not scale_mask.any():
            return patterns
            
        scale_edges = edge_index[:, scale_mask]
        scale_attrs = edge_attr[scale_mask]
        
        # Look for high timeframe_jump values (indicating HTF influence)
        timeframe_jumps = scale_attrs[:, 5]  # timeframe_jump feature
        high_tf_jump_mask = timeframe_jumps > 2.0  # Significant TF jumps
        
        if high_tf_jump_mask.any():
            htf_edges = scale_edges[:, high_tf_jump_mask]
            
            # Find cascading patterns from HTF nodes
            htf_source_nodes = set(htf_edges[1].cpu().numpy())  # HTF nodes (targets of scale edges)
            
            if len(htf_source_nodes) >= 2:
                # Calculate cascade strength
                htf_node_list = list(htf_source_nodes)
                htf_embeddings = embeddings[htf_node_list]
                cascade_coherence = torch.std(htf_embeddings, dim=0).mean().item()
                
                patterns.append({
                    'type': 'htf_cascade',
                    'htf_nodes': htf_node_list,
                    'cascade_strength': 1.0 / (1.0 + cascade_coherence),  # Lower std = higher coherence
                    'timeframe': 'htf_driven',
                    'tf_jump_count': high_tf_jump_mask.sum().item()
                })
        
        return patterns
    
    def _extract_enhanced_cascade_patterns(self, embeddings: torch.Tensor, edge_index: torch.Tensor, 
                                         edge_attr: Optional[torch.Tensor] = None) -> List[Dict]:
        """Enhanced cascade detection using both temporal and scale information"""
        patterns = []
        
        if embeddings.shape[0] <= 3:
            return patterns
            
        # Traditional sequential cascade detection
        for i in range(embeddings.shape[0] - 3):
            seq_embs = embeddings[i:i+4]
            seq_sim = torch.mm(seq_embs, seq_embs.t())
            base_similarity = torch.mean(seq_sim).item()
            
            if base_similarity > 0.8:
                cascade_type = 'temporal_cascade'
                enhancement_factor = 1.0
                
                # Check if this cascade is enhanced by scale edges
                if edge_attr is not None and edge_index.shape[1] > 0:
                    # See if any scale edges connect to these nodes
                    cascade_nodes = set(range(i, i+4))
                    scale_mask = self._detect_scale_edges(edge_attr)
                    
                    if scale_mask.any():
                        scale_edges = edge_index[:, scale_mask]
                        # Check for scale edge connections to cascade nodes
                        scale_connections = 0
                        for edge_idx in range(scale_edges.shape[1]):
                            src = scale_edges[0, edge_idx].item()
                            tgt = scale_edges[1, edge_idx].item()
                            if src in cascade_nodes or tgt in cascade_nodes:
                                scale_connections += 1
                        
                        if scale_connections > 0:
                            cascade_type = 'scale_enhanced_cascade'
                            enhancement_factor = 1.0 + (scale_connections * 0.2)
                
                patterns.append({
                    'type': cascade_type,
                    'start_idx': i,
                    'length': 4,
                    'similarity': base_similarity * enhancement_factor,
                    'scale_connections': scale_connections if 'scale_connections' in locals() else 0,
                    'timeframe': 'enhanced'
                })
        
        return patterns
    
    def _extract_multi_scale_liquidity_patterns(self, embeddings: torch.Tensor, edge_index: torch.Tensor,
                                              edge_attr: torch.Tensor, metadata: Dict) -> List[Dict]:
        """Detect liquidity patterns spanning multiple timeframes"""
        patterns = []
        
        scale_edge_count = metadata['edge_type_counts'].get('scale', 0)
        if scale_edge_count < 3:  # Need multiple scale edges
            return patterns
            
        scale_mask = self._detect_scale_edges(edge_attr)
        scale_edges = edge_index[:, scale_mask]
        
        # Find nodes that are highly connected via scale edges (liquidity aggregation points)
        node_scale_connections = {}
        for i in range(scale_edges.shape[1]):
            src = scale_edges[0, i].item()
            tgt = scale_edges[1, i].item()
            node_scale_connections[src] = node_scale_connections.get(src, 0) + 1
            node_scale_connections[tgt] = node_scale_connections.get(tgt, 0) + 1
        
        # Find highly connected liquidity nodes
        liquidity_nodes = [node for node, connections in node_scale_connections.items() 
                          if connections >= 3]
        
        if len(liquidity_nodes) >= 2:
            # Calculate multi-scale liquidity strength
            liq_embeddings = embeddings[liquidity_nodes]
            liquidity_coherence = torch.mean(torch.mm(liq_embeddings, liq_embeddings.t())).item()
            
            patterns.append({
                'type': 'multi_scale_liquidity',
                'liquidity_nodes': liquidity_nodes,
                'coherence_score': liquidity_coherence,
                'scale_edge_count': scale_edge_count,
                'timeframe': 'multi_scale'
            })
        
        return patterns
    
    def _detect_scale_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Detect scale edges from edge attributes (helper for pattern extraction)"""
        # Scale edges have hierarchy_distance > 0 (feature index 16 in RichEdgeFeature)  
        hierarchy_distances = edge_attr[:, 16]  # hierarchy_distance feature
        scale_mask = hierarchy_distances > 0.0
        return scale_mask
    
    def train_on_historical(self, session_graphs: List[Dict], epochs: int = 50):
        """
        Train on multiple historical sessions with progress monitoring
        
        Args:
            session_graphs: List of graphs from IRONFORGEGraphBuilder
            epochs: Number of training epochs
        """
        print(f"🧠 Training TGAT: {len(session_graphs)} sessions, {epochs} epochs")
        
        # Dynamic model re-initialization based on actual feature dimensions
        if session_graphs:
            # Get actual feature dimensions from first session (after constant filtering)
            first_session = session_graphs[0]
            if len(first_session) >= 5:
                X = first_session[0]  # Feature tensor
                actual_features = X.shape[1]
                
                # Check if model needs re-initialization for correct dimensions
                if self.model.input_projection.in_features != actual_features:
                    print(f"🔧 Re-initializing TGAT: {self.model.input_projection.in_features}D → {actual_features}D (after constant filtering)")
                    
                    # Re-initialize model with correct dimensions
                    self.model = TGAT(
                        in_channels=actual_features,
                        out_channels=128,  # Hidden dim
                        num_of_heads=4,
                        concat=True
                    )
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_patterns = []
            
            # Progress monitoring for each epoch
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1}/{epochs}")
            
            for i, graph_data in enumerate(session_graphs):
                # Progress monitoring for large batches
                if len(session_graphs) > 10 and i % 10 == 0:
                    print(f"      Processing graph batch {i+1}/{len(session_graphs)}")
                # Handle both old format (4 elements) and new format (5 elements with edge_attr)
                if len(graph_data) == 5:
                    X, edge_index, edge_times, metadata, edge_attr = graph_data
                else:
                    X, edge_index, edge_times, metadata = graph_data
                    edge_attr = None
                
                result = self.learn_session(X, edge_index, edge_times, metadata, edge_attr)
                epoch_losses.append(result['losses'])
                epoch_patterns.extend(result['patterns'])
            
            # Log progress
            avg_loss = np.mean([l['temporal'] + l['type'] for l in epoch_losses])
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, "
                  f"Patterns Found: {len(epoch_patterns)}")
            
            # Archive unique patterns
            self._archive_patterns(epoch_patterns)
    
    def _archive_patterns(self, patterns: List[Dict]):
        """Store discovered patterns for later analysis"""
        # De-duplicate and store
        for pattern in patterns:
            pattern_hash = hash(str(pattern['type']) + str(pattern.get('indices', [])))
            if pattern_hash not in [hash(str(p)) for p in self.discovered_patterns]:
                self.discovered_patterns.append(pattern)
    
    def save_discoveries(self, path: str):
        """Save discovered patterns and model"""
        # Save patterns
        patterns_path = os.path.join(path, 'discovered_patterns.json')
        with open(patterns_path, 'w') as f:
            json.dump(self.discovered_patterns, f, indent=2, default=self._json_serializer)
        
        # Save model
        model_path = os.path.join(path, 'tgat_model.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'projection_state': self.projection.state_dict(),
            'patterns': self.discovered_patterns
        }, model_path)
        
        print(f"Saved {len(self.discovered_patterns)} patterns to {patterns_path}")
    
    def _json_serializer(self, obj):
        """Handle non-serializable objects for JSON export"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def _extract_structural_context_patterns(self, X: torch.Tensor, embeddings: torch.Tensor, 
                                            metadata: Dict, edge_index: torch.Tensor, 
                                            edge_attr: Optional[torch.Tensor] = None) -> List[Dict]:
        """
        SPRINT 2: Extract patterns from structural_context edges (4th edge type)
        
        Detects patterns from structural archetype relationships:
        - Causal sequences: sweep → first_fvg_after_sweep
        - Structural alignments: imbalance_zone → htf_range_midpoint
        - Boundary interactions: cascade_origin → session_boundary
        - Reinforcement patterns: liquidity_cluster → structural_support
        """
        patterns = []
        
        # Get edge type information from metadata
        edge_types = metadata.get('edge_types', [])
        edge_type_counts = metadata.get('edge_type_counts', {})
        
        # Check if structural_context edges are present
        structural_edge_count = edge_type_counts.get('structural_context', 0)
        
        if structural_edge_count == 0:
            return patterns  # No structural context edges to analyze
        
        # Analyze structural pattern density
        total_nodes = X.shape[0]
        structural_density = structural_edge_count / max(1, total_nodes)
        
        if structural_density > 0.1:  # Minimum density for meaningful patterns
            # Extract structural archetype patterns
            
            # Pattern 1: High structural connectivity (dense structural network)
            if structural_density > 0.3:
                pattern = {
                    'type': 'structural_context_dense_network',
                    'description': f'Dense structural network @ {structural_density:.1%} density',
                    'confidence': min(0.95, structural_density * 2),
                    'nodes': list(range(total_nodes)),
                    'structural_edge_count': structural_edge_count,
                    'structural_density': structural_density,
                    'session': metadata.get('session_name', 'unknown'),
                    'edge_type_source': 'structural_context'
                }
                patterns.append(pattern)
            
            # Pattern 2: Structural archetype confluence
            if len(edge_types) >= 4:  # All 4 edge types present
                confluence_score = self._calculate_structural_confluence_score(
                    edge_type_counts, structural_edge_count
                )
                
                if confluence_score > 0.6:
                    pattern = {
                        'type': 'structural_context_confluence',
                        'description': f'4-edge type confluence @ {confluence_score:.1%} strength',
                        'confidence': confluence_score,
                        'edge_type_counts': edge_type_counts,
                        'confluence_score': confluence_score,
                        'session': metadata.get('session_name', 'unknown'),
                        'edge_type_source': 'all_four_types'
                    }
                    patterns.append(pattern)
            
            # Pattern 3: Structural context dominance
            total_edges = sum(edge_type_counts.values())
            if total_edges > 0:
                structural_ratio = structural_edge_count / total_edges
                
                if structural_ratio > 0.25:  # Structural edges dominate
                    pattern = {
                        'type': 'structural_context_dominance',
                        'description': f'Structural context dominance @ {structural_ratio:.1%}',
                        'confidence': min(0.9, structural_ratio * 3),
                        'structural_ratio': structural_ratio,
                        'total_edges': total_edges,
                        'session': metadata.get('session_name', 'unknown'),
                        'edge_type_source': 'structural_context_dominant'
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _add_regime_labels_to_patterns(self, patterns: List[Dict], X: torch.Tensor, 
                                     embeddings: torch.Tensor, metadata: Dict) -> List[Dict]:
        """
        SPRINT 2: Add regime labels to all discovered patterns
        
        Integrates with regime segmentation system to auto-classify patterns
        """
        if not patterns:
            return patterns
        
        # Determine session-level regime characteristics
        regime_characteristics = self._analyze_session_regime_characteristics(X, embeddings, metadata)
        
        # Add regime information to each pattern
        enhanced_patterns = []
        
        for pattern in patterns:
            enhanced_pattern = pattern.copy()
            
            # Add regime classification
            enhanced_pattern['regime_characteristics'] = regime_characteristics
            enhanced_pattern['regime_label'] = regime_characteristics.get('regime_label', 'unclassified')
            enhanced_pattern['regime_confidence'] = regime_characteristics.get('confidence', 0.5)
            
            # Add regime-specific metadata
            if 'temporal' in pattern.get('type', ''):
                enhanced_pattern['regime_temporal_dominance'] = regime_characteristics.get('temporal_dominance', 'mixed')
            
            if 'structural' in pattern.get('type', ''):
                enhanced_pattern['regime_structural_dominance'] = regime_characteristics.get('structural_dominance', 'mixed')
            
            enhanced_patterns.append(enhanced_pattern)
        
        return enhanced_patterns
    
    def _calculate_structural_confluence_score(self, edge_type_counts: Dict[str, int], 
                                             structural_edge_count: int) -> float:
        """Calculate confluence score based on edge type distribution"""
        
        expected_edge_types = ['temporal', 'scale', 'structural_context', 'discovered']
        present_edge_types = [et for et in expected_edge_types if edge_type_counts.get(et, 0) > 0]
        
        # Base score: how many edge types are present
        type_coverage = len(present_edge_types) / len(expected_edge_types)
        
        # Balance score: how evenly distributed the edges are
        if len(present_edge_types) > 1:
            edge_counts = [edge_type_counts.get(et, 0) for et in present_edge_types]
            total_edges = sum(edge_counts)
            
            if total_edges > 0:
                # Calculate entropy-based balance score
                proportions = [count / total_edges for count in edge_counts]
                balance_score = 1.0 - (max(proportions) - min(proportions))
            else:
                balance_score = 0.0
        else:
            balance_score = 0.0
        
        # Structural emphasis: bonus for significant structural context presence
        if structural_edge_count > 0:
            total_edges = sum(edge_type_counts.values())
            structural_emphasis = structural_edge_count / max(1, total_edges)
        else:
            structural_emphasis = 0.0
        
        # Combined confluence score
        confluence = 0.5 * type_coverage + 0.3 * balance_score + 0.2 * structural_emphasis
        
        return min(1.0, confluence)
    
    def _analyze_session_regime_characteristics(self, X: torch.Tensor, embeddings: torch.Tensor, 
                                              metadata: Dict) -> Dict[str, Any]:
        """Analyze session-level characteristics for regime classification"""
        
        # Extract temporal cycle features (37D feature indices)
        if X.shape[1] >= 37:
            week_of_month = X[:, 9].unique()  # Index 9 in 37D features
            month_of_year = X[:, 10].unique()  # Index 10
            day_of_week = X[:, 11].unique()   # Index 11
        else:
            week_of_month = torch.tensor([2])  # Default values
            month_of_year = torch.tensor([8])
            day_of_week = torch.tensor([2])
        
        # Determine temporal dominance
        if len(week_of_month) == 1 and len(day_of_week) == 1:
            temporal_dominance = 'weekly'
        elif len(month_of_year) == 1:
            temporal_dominance = 'monthly'
        else:
            temporal_dominance = 'mixed'
        
        # Determine structural dominance from edge types
        edge_type_counts = metadata.get('edge_type_counts', {})
        structural_count = edge_type_counts.get('structural_context', 0)
        total_edges = sum(edge_type_counts.values())
        
        if total_edges > 0 and structural_count > total_edges * 0.4:
            structural_dominance = 'structural_context'
        elif edge_type_counts.get('temporal', 0) > total_edges * 0.4:
            structural_dominance = 'temporal'
        elif edge_type_counts.get('scale', 0) > total_edges * 0.4:
            structural_dominance = 'scale'
        else:
            structural_dominance = 'mixed'
        
        # Price range analysis
        if X.shape[1] >= 12 and X.shape[0] > 0:
            normalized_price = X[:, 11]  # normalized_price feature
            avg_price_level = torch.mean(normalized_price).item()
            
            if avg_price_level < 0.33:
                price_preference = 'low'
            elif avg_price_level > 0.67:
                price_preference = 'high'
            else:
                price_preference = 'mid'
        else:
            price_preference = 'mid'
        
        # Generate regime label
        regime_label = f"{temporal_dominance}_{structural_dominance}_{price_preference}"
        
        # Calculate confidence based on data quality
        confidence = 0.6  # Base confidence
        if X.shape[0] > 5:  # More nodes = higher confidence
            confidence += 0.2
        if total_edges > 10:  # More edges = higher confidence
            confidence += 0.2
        
        confidence = min(1.0, confidence)
        
        return {
            'regime_label': regime_label,
            'temporal_dominance': temporal_dominance,
            'structural_dominance': structural_dominance,
            'price_range_preference': price_preference,
            'confidence': confidence,
            'week_of_month': week_of_month[0].item() if len(week_of_month) > 0 else 2,
            'month_of_year': month_of_year[0].item() if len(month_of_year) > 0 else 8,
            'day_of_week': day_of_week[0].item() if len(day_of_week) > 0 else 2
        }

    # COMPATIBILITY WRAPPER REMOVED - Use learn_session() directly with proper tensor pipeline

    def freeze_for_prediction(self):
        """
        Freeze model for use in prediction system
        THIS IS THE ONLY BRIDGE TO PREDICTION
        """
        self.model.eval()
        self.projection.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = False
        
        print("Model frozen for prediction use")
        return self
    
    def __call__(self, X: torch.Tensor, edge_index: torch.Tensor, edge_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for compatibility with validation script
        """
        # Create dummy edge times if not provided
        if edge_times is None:
            edge_times = torch.zeros(edge_index.shape[1])
        
        # Forward pass through TGAT model
        h = self.model(X, edge_index, edge_times)
        
        # Project to final embedding dimension
        embeddings = self.projection(h)
        
        return embeddings
    
    def _extract_temporal_structural_patterns(self, X, embeddings, session_data):
        """Extract temporal structural patterns from TGAT embeddings"""
        patterns = []
        
        # Extract basic temporal structural patterns from normalized features
        if X.shape[0] >= 3:
            # Look for normalized_price patterns in the movements
            price_movements = session_data.get('price_movements', [])
            
            for i, movement in enumerate(price_movements[:10]):  # Limit for performance
                if 'normalized_price' in movement and 'time_since_session_open' in movement:
                    pattern = {
                        'type': 'temporal_structural',
                        'description': f"Price position {movement['normalized_price']:.1%} at {movement['time_since_session_open']} minutes",
                        'confidence': 0.7,
                        'time_span_hours': movement['time_since_session_open'] / 3600.0,
                        'session': session_data.get('session_metadata', {}).get('session_type', 'unknown'),
                        'structural_position': movement['normalized_price'],
                        'temporal_position': movement['time_since_session_open']
                    }
                    patterns.append(pattern)
        
        return patterns[:5]  # Limit to top 5 patterns
    
    def _extract_htf_confluence_patterns(self, X, embeddings, session_data):
        """Extract HTF confluence patterns from TGAT embeddings"""
        patterns = []
        
        # Extract HTF confluence based on enhanced features
        htf_carryover = session_data.get('contamination_analysis', {}).get('htf_contamination', {}).get('htf_carryover_strength', 0.0)
        energy_density = session_data.get('energy_state', {}).get('energy_density', 0.0)
        
        if htf_carryover > 0.7 and energy_density > 0.8:  # High confluence conditions
            pattern = {
                'type': 'htf_confluence',
                'description': f"HTF confluence {htf_carryover:.2f} strength with {energy_density:.2f} energy density",
                'confidence': min(htf_carryover, energy_density),
                'time_span_hours': 4.0,  # Typical HTF timeframe
                'session': session_data.get('session_metadata', {}).get('session_type', 'unknown'),
                'htf_strength': htf_carryover,
                'energy_level': energy_density
            }
            patterns.append(pattern)
            
        return patterns
    
    def _extract_scale_alignment_patterns(self, embeddings, edge_index, session_data):
        """Extract scale alignment patterns from TGAT embeddings"""
        patterns = []
        
        # Extract scale alignment based on session liquidity events
        liquidity_events = session_data.get('session_liquidity_events', [])
        
        if len(liquidity_events) >= 10:  # Rich liquidity environment
            cross_session_events = [e for e in liquidity_events if e.get('liquidity_type') == 'cross_session']
            
            if len(cross_session_events) >= 3:
                pattern = {
                    'type': 'scale_alignment',
                    'description': f"Multi-scale alignment with {len(cross_session_events)} cross-session events",
                    'confidence': min(0.9, len(cross_session_events) / 5.0),
                    'time_span_hours': 2.0,
                    'session': session_data.get('session_metadata', {}).get('session_type', 'unknown'),
                    'event_count': len(liquidity_events),
                    'cross_session_count': len(cross_session_events)
                }
                patterns.append(pattern)
        
        return patterns
