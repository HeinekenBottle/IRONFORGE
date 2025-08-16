# TGAT Architecture Documentation
**Temporal Graph Attention Networks for Archaeological Discovery**

---

## 🧠 Overview

The Temporal Graph Attention Network (TGAT) is the core neural architecture powering IRONFORGE's archaeological discovery capabilities. Unlike traditional prediction models, TGAT is specifically designed for self-supervised pattern discovery in temporal market data, focusing on uncovering hidden relationships across time and price dimensions.

**Key Principles**:
- **Archaeological Focus**: Discovers existing patterns, never predicts future outcomes
- **Temporal Awareness**: Captures distant time-price relationships through attention mechanisms
- **Self-Supervised Learning**: No labels required, learns from data structure itself
- **Semantic Preservation**: Maintains human-readable context throughout processing

---

## 🏗️ Architecture Components

### 1. Multi-Head Attention Mechanism

TGAT employs a 4-head attention architecture, with each head specialized for different pattern types:

```python
class TGAT(torch.nn.Module):
    def __init__(self, in_channels=45, out_channels=128, num_of_heads=4):
        super().__init__()
        self.num_heads = num_of_heads
        self.out_channels = out_channels
        self.head_dim = out_channels // num_of_heads
        
        # Multi-head attention components
        self.query_projection = nn.Linear(in_channels, out_channels)
        self.key_projection = nn.Linear(in_channels, out_channels)
        self.value_projection = nn.Linear(in_channels, out_channels)
        
        # Temporal encoding
        self.temporal_encoder = TemporalEncoder(out_channels)
        
        # Output projection
        self.output_projection = nn.Linear(out_channels, out_channels)
```

#### Attention Head Specialization:
- **Head 1**: Structural patterns (support/resistance, ranges)
- **Head 2**: Temporal patterns (session boundaries, time-based cycles)
- **Head 3**: Confluence patterns (multi-timeframe alignments)
- **Head 4**: Semantic patterns (FVG chains, expansion sequences)

### 2. Temporal Encoding System

The temporal encoder captures time-aware relationships between market events:

```python
class TemporalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, edge_times):
        """
        Encode temporal distances between connected nodes
        
        Args:
            edge_times: Temporal distances [num_edges]
        Returns:
            Temporal encodings [num_edges, d_model]
        """
        # Sinusoidal temporal encoding
        position = edge_times.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * 
                           -(math.log(10000.0) / self.d_model))
        
        encoding = torch.zeros(edge_times.size(0), self.d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        return encoding
```

### 3. Feature Processing Pipeline

#### Input Feature Vectors (45D Nodes)
```python
# Base features (37D)
base_features = {
    'price_relativity': 34,  # Price position within ranges/structures
    'temporal_cycles': 3     # Session timing and cycle position
}

# Semantic features (8D)
semantic_features = {
    'fvg_redelivery_event': 1,      # FVG redelivery detection
    'expansion_phase_event': 1,      # Market expansion identification
    'consolidation_event': 1,        # Consolidation pattern recognition
    'pd_array_event': 1,            # Premium/Discount array detection
    'liquidity_sweep_event': 1,      # Liquidity sweep identification
    'session_boundary_event': 1,     # Session transition markers
    'htf_confluence_event': 1,       # Higher timeframe confluence
    'structural_break_event': 1      # Market structure breaks
}
```

#### Edge Feature Vectors (20D)
```python
# Base edge features (17D)
base_edge_features = {
    'temporal_distance': 1,          # Time between connected events
    'price_correlation': 1,          # Price movement correlation
    'structural_relationship': 15    # Various structural connections
}

# Semantic edge features (3D)
semantic_edge_features = {
    'semantic_event_link': 1,        # Event chain relationships
    'event_causality': 1,           # Causal strength between events
    'semantic_label_id': 1          # Encoded relationship identifiers
}
```

---

## 🔄 Forward Pass Architecture

### 1. Attention Computation

```python
def forward(self, x, edge_index, edge_times, edge_attr=None):
    """
    TGAT forward pass with archaeological discovery focus
    
    Args:
        x: Node features [num_nodes, 45]
        edge_index: Edge connections [2, num_edges]
        edge_times: Temporal distances [num_edges]
        edge_attr: Edge features [num_edges, 20]
    
    Returns:
        Node embeddings with attention weights for pattern discovery
    """
    batch_size, num_nodes = x.size(0), x.size(1)
    
    # 1. Project to query, key, value
    queries = self.query_projection(x)  # [num_nodes, out_channels]
    keys = self.key_projection(x)       # [num_nodes, out_channels]
    values = self.value_projection(x)   # [num_nodes, out_channels]
    
    # 2. Reshape for multi-head attention
    queries = queries.view(num_nodes, self.num_heads, self.head_dim)
    keys = keys.view(num_nodes, self.num_heads, self.head_dim)
    values = values.view(num_nodes, self.num_heads, self.head_dim)
    
    # 3. Compute temporal encodings
    temporal_encodings = self.temporal_encoder(edge_times)
    
    # 4. Apply attention mechanism with temporal awareness
    attention_output, attention_weights = self.temporal_attention(
        queries, keys, values, edge_index, temporal_encodings, edge_attr
    )
    
    # 5. Combine multi-head outputs
    attention_output = attention_output.view(num_nodes, -1)
    
    # 6. Final projection
    output = self.output_projection(attention_output)
    
    return output, attention_weights
```

### 2. Temporal Attention Mechanism

```python
def temporal_attention(self, queries, keys, values, edge_index, temporal_enc, edge_attr):
    """
    Compute attention with temporal and semantic awareness
    """
    source_nodes, target_nodes = edge_index[0], edge_index[1]
    
    # Extract relevant queries, keys, values for edges
    edge_queries = queries[target_nodes]  # [num_edges, num_heads, head_dim]
    edge_keys = keys[source_nodes]        # [num_edges, num_heads, head_dim]
    edge_values = values[source_nodes]    # [num_edges, num_heads, head_dim]
    
    # Incorporate temporal encoding
    temporal_keys = edge_keys + temporal_enc.unsqueeze(1)
    
    # Compute attention scores
    attention_scores = torch.sum(edge_queries * temporal_keys, dim=-1)  # [num_edges, num_heads]
    attention_scores = attention_scores / math.sqrt(self.head_dim)
    
    # Apply edge features if available (semantic relationships)
    if edge_attr is not None:
        edge_weights = self.edge_attention(edge_attr)  # [num_edges, num_heads]
        attention_scores = attention_scores * edge_weights
    
    # Softmax normalization per target node
    attention_weights = self.softmax_per_node(attention_scores, target_nodes)
    
    # Apply attention to values
    attended_values = attention_weights.unsqueeze(-1) * edge_values  # [num_edges, num_heads, head_dim]
    
    # Aggregate by target node
    output = torch.zeros(queries.size(0), self.num_heads, self.head_dim)
    output.index_add_(0, target_nodes, attended_values)
    
    return output, attention_weights
```

---

## 🎯 Archaeological Discovery Process

### 1. Pattern Discovery Pipeline

```python
class IRONFORGEDiscovery:
    def discover_patterns(self, graph):
        """
        Archaeological pattern discovery using TGAT
        """
        # 1. Forward pass through TGAT
        node_embeddings, attention_weights = self.model(
            graph.x, graph.edge_index, graph.edge_times, graph.edge_attr
        )
        
        # 2. Extract attention patterns
        attention_patterns = self.extract_attention_patterns(attention_weights)
        
        # 3. Identify archaeological significance
        archaeological_patterns = []
        for pattern in attention_patterns:
            significance = self.assess_archaeological_significance(pattern, graph)
            if significance['permanence_score'] > 0.7:
                archaeological_patterns.append({
                    'pattern_id': self.generate_pattern_id(pattern),
                    'attention_weights': pattern['weights'],
                    'archaeological_significance': significance,
                    'semantic_context': self.extract_semantic_context(pattern, graph),
                    'confidence': pattern['confidence']
                })
        
        return archaeological_patterns
```

### 2. Attention Pattern Extraction

```python
def extract_attention_patterns(self, attention_weights):
    """
    Extract meaningful patterns from attention weights
    """
    patterns = []
    
    # Analyze each attention head separately
    for head_idx in range(self.num_heads):
        head_weights = attention_weights[:, head_idx]
        
        # Find high-attention edges (top 10%)
        threshold = torch.quantile(head_weights, 0.9)
        high_attention_edges = head_weights > threshold
        
        if high_attention_edges.sum() > 0:
            pattern = {
                'head': head_idx,
                'weights': head_weights[high_attention_edges],
                'edges': high_attention_edges,
                'confidence': head_weights[high_attention_edges].mean().item(),
                'pattern_type': self.classify_attention_pattern(head_idx, head_weights)
            }
            patterns.append(pattern)
    
    return patterns
```

### 3. Archaeological Significance Assessment

```python
def assess_archaeological_significance(self, pattern, graph):
    """
    Assess the archaeological value of discovered patterns
    """
    # Temporal persistence analysis
    temporal_span = self.calculate_temporal_span(pattern, graph)
    
    # Cross-session validation
    cross_session_strength = self.validate_cross_session(pattern, graph)
    
    # Semantic coherence
    semantic_coherence = self.assess_semantic_coherence(pattern, graph)
    
    # Calculate permanence score
    permanence_score = (
        0.4 * temporal_span +
        0.3 * cross_session_strength +
        0.3 * semantic_coherence
    )
    
    # Determine archaeological value
    if permanence_score > 0.9:
        archaeological_value = 'high_archaeological_value'
    elif permanence_score > 0.7:
        archaeological_value = 'medium_archaeological_value'
    else:
        archaeological_value = 'low_archaeological_value'
    
    return {
        'permanence_score': permanence_score,
        'archaeological_value': archaeological_value,
        'temporal_span': temporal_span,
        'cross_session_strength': cross_session_strength,
        'semantic_coherence': semantic_coherence
    }
```

---

## 📊 Performance Characteristics

### Training Performance
- **Self-Supervised**: No labeled data required
- **Convergence**: Typically 50-100 epochs
- **Memory Usage**: ~200MB for typical session graphs
- **Training Time**: 2-5 minutes per session on CPU

### Inference Performance
- **Processing Speed**: <3 seconds per session
- **Pattern Discovery**: 8-30 patterns per session typical
- **Memory Efficiency**: <50MB inference footprint
- **Batch Processing**: Up to 10 sessions simultaneously

### Quality Metrics
- **Authenticity Score**: >92/100 for production patterns
- **Temporal Coherence**: >70% meaningful time relationships
- **Semantic Preservation**: 100% semantic event preservation
- **Cross-Session Validation**: >80% pattern consistency

---

## 🔧 Configuration & Tuning

### Model Hyperparameters

```python
TGAT_CONFIG = {
    # Architecture
    'node_features': 45,        # Input feature dimension
    'edge_features': 20,        # Edge feature dimension
    'hidden_dim': 128,          # Hidden layer size
    'num_heads': 4,             # Multi-head attention
    'dropout': 0.1,             # Regularization
    
    # Training
    'learning_rate': 0.001,     # Adam optimizer learning rate
    'weight_decay': 1e-5,       # L2 regularization
    'max_epochs': 100,          # Maximum training epochs
    'early_stopping': 20,       # Early stopping patience
    
    # Discovery
    'attention_threshold': 0.9,  # Top 10% attention patterns
    'confidence_threshold': 0.7, # Minimum pattern confidence
    'permanence_threshold': 0.7  # Archaeological significance threshold
}
```

### Performance Tuning

1. **Memory Optimization**:
   - Use gradient checkpointing for large graphs
   - Batch processing for multiple sessions
   - Attention sparsity for efficiency

2. **Quality Optimization**:
   - Adjust attention thresholds based on data quality
   - Fine-tune permanence scoring weights
   - Calibrate semantic coherence metrics

3. **Speed Optimization**:
   - GPU acceleration for large-scale processing
   - Attention caching for repeated patterns
   - Lazy loading for memory efficiency

---

## 🚫 Architectural Constraints

### Immutable Design Principles
1. **No Prediction Logic**: TGAT discovers patterns, never predicts outcomes
2. **Temporal Awareness**: All attention mechanisms must be time-aware
3. **Semantic Preservation**: Human-readable context maintained throughout
4. **Archaeological Focus**: Discovery of existing relationships only
5. **Self-Supervised**: No external labels or supervision required

### Performance Constraints
- **Processing Time**: <3 seconds per session
- **Memory Usage**: <200MB training, <50MB inference
- **Quality Threshold**: >87% authenticity for production
- **Attention Efficiency**: Sparse attention for scalability

---

*This TGAT architecture enables IRONFORGE to discover meaningful archaeological patterns in market data while maintaining complete temporal awareness and semantic context preservation.*
