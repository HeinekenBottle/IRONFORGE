# IRONFORGE API Reference
**Complete API Documentation for Archaeological Discovery System**

---

## 🏗️ Core Components

### Container System

#### `initialize_ironforge_lazy_loading()`
Initialize the IRONFORGE system with lazy loading for optimal performance.

```python
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading

container = initialize_ironforge_lazy_loading()
```

**Returns**: `IRONFORGEContainer` - Dependency injection container  
**Performance**: <2s initialization time  
**Memory**: ~50MB base footprint

#### `IRONFORGEContainer`
Main dependency injection container for system components.

**Methods**:
- `get_enhanced_graph_builder()` → `EnhancedGraphBuilder`
- `get_tgat_discovery()` → `IRONFORGEDiscovery`
- `get_pattern_graduation()` → `PatternGraduation`
- `get_broad_spectrum_archaeology()` → `BroadSpectrumArchaeology`

---

## 🧠 Learning Components

### EnhancedGraphBuilder

#### `class EnhancedGraphBuilder`
Transforms JSON sessions into 45D/20D TGAT-compatible graphs with semantic features.

```python
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

builder = EnhancedGraphBuilder()
graph = builder.enhance_session(session_data)
```

#### Methods

##### `enhance_session(session_data: dict) -> torch_geometric.data.Data`
Convert JSON session to enhanced graph with semantic features.

**Parameters**:
- `session_data` (dict): Level 1 JSON session data

**Returns**: PyTorch Geometric graph with:
- **Node features**: 45D (37D base + 8D semantic)
- **Edge features**: 20D (17D base + 3D semantic)
- **Semantic events**: FVG redelivery, expansion phases, consolidation
- **Session anchoring**: Timing and market regime preservation

**Performance**: <1s per session  
**Memory**: ~10MB per graph

##### `extract_semantic_features(session_data: dict) -> dict`
Extract semantic market events from session data.

**Returns**:
```python
{
    'fvg_redelivery_events': [...],
    'expansion_phase_events': [...],
    'consolidation_events': [...],
    'pd_array_events': [...],
    'liquidity_sweep_events': [...],
    'session_boundary_events': [...],
    'htf_confluence_events': [...],
    'structural_break_events': [...]
}
```

### IRONFORGEDiscovery

#### `class IRONFORGEDiscovery`
TGAT-based archaeological pattern discovery engine.

```python
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

discovery = IRONFORGEDiscovery(
    node_features=45,
    hidden_dim=128,
    out_dim=256
)
```

#### Methods

##### `discover_patterns(graph: torch_geometric.data.Data) -> List[dict]`
Discover archaeological patterns using TGAT temporal attention.

**Parameters**:
- `graph`: Enhanced graph from EnhancedGraphBuilder

**Returns**: List of discovered patterns:
```python
[
    {
        'pattern_id': 'NY_session_RPC_00',
        'session_name': 'NY_session',
        'confidence': 0.87,
        'archaeological_significance': {
            'archaeological_value': 'high_archaeological_value',
            'permanence_score': 0.933
        },
        'semantic_context': {
            'market_regime': 'transitional',
            'event_types': ['fvg_redelivery'],
            'relationship_type': 'confluence_relationship'
        },
        'attention_weights': [...],
        'description': 'Multi-timeframe confluence pattern'
    }
]
```

**Performance**: <3s per session, 8-30 patterns typical  
**Quality**: >87% authenticity threshold

##### `train_model(graphs: List[torch_geometric.data.Data]) -> None`
Train TGAT model on multiple enhanced graphs.

**Parameters**:
- `graphs`: List of enhanced graphs for training

**Training Features**:
- Self-supervised learning (no labels required)
- 4-head multi-attention architecture
- Temporal encoding for distant correlations
- Archaeological focus (no prediction logic)

---

## 🔍 Analysis Components

### BroadSpectrumArchaeology

#### `class BroadSpectrumArchaeology`
Comprehensive multi-timeframe archaeological pattern discovery.

```python
from ironforge.analysis.broad_spectrum_archaeology import BroadSpectrumArchaeology

archaeology = BroadSpectrumArchaeology()
results = archaeology.discover_all_sessions()
```

#### Methods

##### `discover_all_sessions() -> dict`
Discover patterns across all available sessions.

**Returns**:
```python
{
    'total_sessions': 57,
    'patterns_discovered': 2847,
    'session_results': {...},
    'cross_session_links': [...],
    'quality_metrics': {
        'authenticity_score': 92.3,
        'duplication_rate': 0.18,
        'temporal_coherence': 0.74
    }
}
```

##### `analyze_session_patterns(session_type: str) -> List[dict]`
Analyze patterns for specific session type.

**Parameters**:
- `session_type`: 'NY_AM', 'NY_PM', 'LONDON', 'ASIA', etc.

**Returns**: Filtered patterns for session type with analysis

### PatternIntelligence

#### `class PatternIntelligenceEngine`
Advanced pattern analysis and market regime identification.

```python
from ironforge.analysis.pattern_intelligence import PatternIntelligenceEngine

intel = PatternIntelligenceEngine(discovery_sdk)
trends = intel.analyze_pattern_trends()
```

#### Methods

##### `analyze_pattern_trends(days_lookback: int = 30) -> dict`
Analyze pattern trends over time with statistical significance.

**Returns**:
```python
{
    'temporal_structural': {
        'trend_strength': 0.73,
        'significance': 0.02,  # p-value
        'description': 'Strengthening trend',
        'avg_confidence_trend': 0.045
    }
}
```

##### `identify_market_regimes(min_sessions: int = 3) -> List[dict]`
Identify market regimes using clustering analysis.

**Returns**: List of market regimes with characteristics

##### `find_similar_patterns(reference_pattern: dict, similarity_threshold: float = 0.8) -> List[dict]`
Find historically similar patterns.

---

## 📊 Daily Workflows

### Daily Discovery Workflows

#### `morning_prep(days_back: int = 7) -> dict`
Comprehensive morning market preparation analysis.

```python
from ironforge.analysis.daily_discovery_workflows import morning_prep

analysis = morning_prep(days_back=7)
```

**Returns**:
```python
{
    'strength_score': 0.73,
    'confidence_level': 'High',
    'current_regime': 'Strong temporal_structural regime',
    'dominant_patterns': ['temporal_structural', 'htf_confluence'],
    'cross_session_signals': [...],
    'trading_insights': [...],
    'session_focus': 'NY_PM'
}
```

#### `hunt_patterns(session_type: str) -> dict`
Real-time pattern discovery for specific session type.

**Parameters**:
- `session_type`: Target session type for pattern hunting

**Returns**:
```python
{
    'patterns_found': [...],
    'strength_indicators': {
        'avg_confidence': 0.78,
        'pattern_count': 12,
        'quality_score': 0.85
    },
    'immediate_insights': [...],
    'next_session_expectations': [...]
}
```

---

## ✅ Synthesis Components

### PatternGraduation

#### `class PatternGraduation`
Validate discovered patterns against quality thresholds.

```python
from ironforge.synthesis.pattern_graduation import PatternGraduation

graduation = PatternGraduation()
validated = graduation.validate_patterns(discovered_patterns)
```

#### Methods

##### `validate_patterns(patterns: List[dict]) -> List[dict]`
Validate patterns against 87% baseline threshold.

**Validation Criteria**:
- Authenticity score >87/100
- Duplication rate <25%
- Temporal coherence >70%
- Confidence threshold >0.7

**Returns**: Only patterns meeting production quality standards

##### `calculate_authenticity_score(pattern: dict) -> float`
Calculate pattern authenticity score (0-100).

**Factors**:
- Temporal consistency
- Semantic coherence
- Cross-session validation
- Historical precedent

---

## 🛠️ Utility Functions

### Performance Monitoring

#### `monitor_system_performance() -> dict`
Monitor IRONFORGE system performance metrics.

**Returns**:
```python
{
    'initialization_time': 1.8,
    'memory_usage_mb': 87.3,
    'cache_hit_rate': 0.84,
    'average_processing_time': 2.7,
    'patterns_per_second': 8.2
}
```

### Data Validation

#### `validate_session_data(session_data: dict) -> bool`
Validate Level 1 JSON session data format.

**Validation Checks**:
- Required fields present
- Price movement data quality
- Timestamp consistency
- Enhanced features availability

---

## 🔧 Configuration

### System Configuration

```python
IRONFORGE_CONFIG = {
    # Processing settings
    'max_sessions_per_batch': 10,
    'discovery_timeout_seconds': 300,
    'enable_caching': True,
    
    # Quality thresholds
    'pattern_confidence_threshold': 0.7,
    'authenticity_threshold': 87.0,
    'max_duplication_rate': 0.25,
    
    # Performance settings
    'lazy_loading': True,
    'max_memory_mb': 1000,
    'enable_monitoring': True
}
```

### TGAT Configuration

```python
TGAT_CONFIG = {
    'node_features': 45,      # 37D base + 8D semantic
    'edge_features': 20,      # 17D base + 3D semantic
    'hidden_dim': 128,
    'num_heads': 4,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'max_epochs': 100
}
```

---

## 📈 Performance Specifications

### Processing Performance
- **Single Session**: <3 seconds
- **Full Discovery**: <180 seconds (57 sessions)
- **Initialization**: <2 seconds
- **Memory Usage**: <100MB total

### Quality Metrics
- **Authenticity**: >90/100 for production patterns
- **Duplication Rate**: <25%
- **Temporal Coherence**: >70%
- **Pattern Confidence**: >0.7 threshold

### Scalability
- **Batch Processing**: Up to 10 sessions simultaneously
- **Cache Efficiency**: >80% hit rate
- **Memory Optimization**: Automatic cleanup
- **Resource Management**: Lazy loading architecture

---

*This API reference provides complete documentation for all IRONFORGE components. For usage examples and workflows, see the [User Guide](USER_GUIDE.md) and [Getting Started](GETTING_STARTED.md) guides.*
