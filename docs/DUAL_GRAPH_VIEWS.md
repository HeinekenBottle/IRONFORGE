# Dual Graph Views 📊

## Overview

IRONFORGE Dual Graph Views is a comprehensive system for analyzing financial market data through multiple graph representations. It combines **temporal graphs** (undirected, pattern-focused) with **DAG graphs** (directed acyclic, causality-aware) to provide unprecedented insight into market structure and event relationships.

## Core Components

### 1. **Temporal Graphs** 🕸️
- **Undirected** temporal graphs for pattern discovery
- Rich 45D node features with ICT (Inner Circle Trader) semantic preservation
- Traditional graph neural network compatibility
- Optimized for pattern matching and similarity analysis

### 2. **DAG Graphs** 🎯
- **Directed Acyclic Graphs** enforcing temporal causality
- Guaranteed acyclicity through (timestamp, seq_idx) ordering
- ICT causality patterns: FVG formation→redelivery, sweep→reversal
- Specialized for causal inference and temporal flow analysis

### 3. **M1 Integration** ⚡
- **Sparse M1 event layer** with 6 micro-timeframe event types
- **Cross-scale edges** linking M1 events to M5 bars
- **53D enhanced features** (45D base + 8D M1-derived)
- Multi-scale coherence analysis and causality strength

### 4. **Enhanced TGAT** 🧠
- **Masked attention** with DAG-based causal constraints
- **Temporal bias networks** for sophisticated time relationships
- **PyTorch flex_attention** integration (fail-fast)
- Support for both 45D and 53D (M1-enhanced) features

### 5. **Statistical Motif Mining** 🔍
- **Dual null models**: time-jitter (±60-120m) & session permutation
- **Rigorous validation**: lift ratios, confidence intervals, p-values
- **Pattern classification**: PROMOTE, PARK, DISCARD
- **Performance optimized** with ZSTD Parquet compression

## Configuration Presets

### Minimal (`--preset minimal`)
```bash
# Basic testing configuration
ironforge build-graph --preset minimal --with-dag --dry-run
```
- DAG: k=2, basic acyclicity
- M1: Disabled  
- TGAT: Standard attention
- Motifs: 100 null iterations

### Standard (`--preset standard`) ⭐
```bash
# Production-ready configuration
ironforge build-graph --preset standard --with-dag --with-m1
```
- DAG: k=4, full causality weights
- M1: Enabled with 6 event types
- TGAT: Standard temporal attention
- Motifs: 1000 null iterations

### Enhanced (`--preset enhanced`) 🚀
```bash
# All features enabled
ironforge build-graph --preset enhanced --with-dag --with-m1 --enhanced-tgat
```
- DAG: k=6, enhanced connectivity
- M1: Full integration with cross-scale edges
- TGAT: **Masked attention + temporal bias**
- Motifs: 2000 null iterations, 8 concurrent sessions

### Research (`--preset research`) 🔬
```bash
# Maximum discovery configuration
ironforge build-graph --preset research --enable-motifs --save-config
```
- DAG: k=8, maximum causality detection
- M1: Lower thresholds for discovery
- TGAT: 3-4 attention layers, enhanced architecture
- Motifs: 5000 null iterations, 500 max patterns

## Usage Examples

### Basic DAG Construction
```bash
# Simple DAG with temporal graph
ironforge build-graph \
    --source-glob "data/enhanced/*.json" \
    --output-dir "data/graphs" \
    --with-dag \
    --format parquet
```

### M1-Enhanced Multi-Scale Analysis
```bash
# Full dual views with M1 integration
ironforge build-graph \
    --preset standard \
    --with-dag \
    --with-m1 \
    --source-glob "data/enhanced/*.json" \
    --output-dir "data/dual_graphs" \
    --max-sessions 50
```

### Research-Grade Discovery Pipeline
```bash
# Maximum feature discovery
ironforge build-graph \
    --preset research \
    --enhanced-tgat \
    --enable-motifs \
    --config-overrides '{"motifs.significance_threshold": 0.01}' \
    --save-config \
    --source-glob "data/enhanced/*.json" \
    --output-dir "data/research_graphs"
```

### Configuration File Usage
```bash
# Create custom configuration file
cat > dual_graph_config.json << 'EOF'
{
  "dag": {
    "k_successors": 5,
    "dt_max_minutes": 90,
    "causality_weights": {
      "fvg_to_fvg": 0.95,
      "sweep_to_reversal": 0.85
    }
  },
  "m1": {
    "enabled": true,
    "confidence_threshold": 0.5,
    "time_window_minutes": 7
  },
  "tgat": {
    "enhanced": true,
    "num_heads": 6,
    "num_layers": 3
  }
}
EOF

# Use configuration file
ironforge build-graph \
    --config dual_graph_config.json \
    --with-dag \
    --source-glob "data/enhanced/*.json"
```

## Output Structure

```
data/graphs/
├── dual_graph_config.json          # Final configuration used
├── build_manifest.jsonl            # Build summary
├── motifs/                          # Motif mining results
│   └── motifs_summary.json
└── enhanced_NQ_2025-01-15_AM/       # Per-session results
    ├── metadata.json                # Session metadata
    ├── temporal_graph/             # Undirected temporal graph (Parquet)
    │   ├── nodes.parquet           # Node features and embeddings
    │   ├── edges.parquet           # Edge connectivity and features
    │   └── metadata.json           # Graph metadata
    ├── dag_graph/                  # Directed acyclic graph (Parquet)
    │   ├── nodes.parquet           # Node features with causal ordering
    │   ├── edges.parquet           # Edge features with causal strengths
    │   └── metadata.json           # DAG metadata and topology
    └── edges_dag.parquet           # Optimized edge storage
```

## Advanced Features

### Custom Causality Weights
```python
from ironforge.learning.dual_graph_config import DualGraphViewsConfig

config = DualGraphViewsConfig()
config.dag.causality_weights = {
    'fvg_to_fvg': 0.95,           # Strong FVG causality
    'sweep_to_reversal': 0.90,     # Liquidity sweep patterns  
    'expansion_to_retrace': 0.75,  # Market phase transitions
    'premium_discount': 0.70,      # PD array interactions
    'generic_temporal': 0.50       # Default temporal
}
```

### M1 Event Type Configuration
```python
config.m1.event_types = [
    'micro_fvg_fill',    # FVG redelivery at M1 level
    'micro_sweep',       # Liquidity sweeps (stop runs)
    'micro_impulse',     # Price impulses/spikes  
    'vwap_touch',        # VWAP interaction points
    'imbalance_burst',   # Order flow imbalances
    'wick_extreme'       # Extreme price wicks
]
config.m1.confidence_threshold = 0.6   # Minimum confidence
config.m1.time_window_minutes = 5      # Analysis window
```

### Enhanced TGAT Configuration
```python
config.tgat.enhanced = True
config.tgat.use_flex_attention = True      # Requires modern PyTorch
config.tgat.causal_masking = True          # DAG-based masking
config.tgat.temporal_bias_enabled = True   # Sophisticated temporal bias
config.tgat.num_heads = 8                  # Multi-head attention
config.tgat.num_layers = 4                 # Deep attention stack
```

### Motif Mining Tuning
```python
config.motifs.min_nodes = 3               # Minimum pattern size
config.motifs.max_nodes = 6               # Maximum pattern size
config.motifs.null_iterations = 2000      # Statistical validation
config.motifs.promote_lift_threshold = 2.5 # PROMOTE classification
config.motifs.significance_threshold = 0.01 # P-value cutoff
```

## Performance Optimization

### Storage Configuration
```python
config.storage.compression = 'zstd'       # High compression, fast decompression
config.storage.row_group_size = 10000     # Parquet optimization
config.storage.partition_by_session = True # Efficient querying
```

### Processing Configuration  
```python
config.max_concurrent_sessions = 8        # Parallel processing
config.memory_limit_gb = 16.0             # Memory management
config.enable_performance_monitoring = True # Track metrics
```

## Architecture Insights

### DAG Acyclicity Guarantees
- **Temporal ordering**: (timestamp, seq_idx) ensures no cycles
- **Validation**: NetworkX `is_directed_acyclic_graph()` verification  
- **Topological sort**: Guaranteed valid ordering exists
- **Causality constraints**: Forward edges only (no time travel)

### M1-M5 Cross-Scale Relationships
1. **CONTAINED_IN**: M1 event occurs within M5 bar timeframe
2. **PRECEDES**: M1 event temporally precedes another M1 event  
3. **INFLUENCES**: M1 event affects subsequent M5 bar characteristics

### Statistical Null Models
1. **Time-jitter nulls**: Randomize timestamps ±60-120 minutes
2. **Session permutation nulls**: Shuffle events within sessions
3. **Significance testing**: Compare real vs null distributions
4. **Classification**: PROMOTE (lift≥2.0, p<0.01), PARK (significant), DISCARD

## Theory B Integration

The dual graph views system implements **Theory B** temporal non-locality:

- **Archaeological zones**: 40% of eventual session range
- **Dimensional relationships**: Events position relative to final structure
- **Temporal non-locality**: Events "know" their eventual context
- **Predictive power**: Early events contain forward-looking information

## CLI Quick Reference

```bash
# Basic usage
ironforge build-graph --help

# Common patterns
ironforge build-graph --preset standard --with-dag
ironforge build-graph --preset enhanced --with-m1 --enhanced-tgat  
ironforge build-graph --preset research --enable-motifs --save-config

# Configuration overrides
ironforge build-graph --config custom.json --preset standard
ironforge build-graph --config-overrides '{"dag.k_successors": 8}'

# Legacy compatibility
ironforge build-graph --dag-k 4 --dag-dt-max 120 --with-dag

# Output control
ironforge build-graph --format parquet --output-dir custom_output/
ironforge build-graph --dry-run --max-sessions 10
```

## Troubleshooting

### Common Issues

**Import Error: flex_attention not available**
```bash
# System uses fail-fast - update PyTorch for enhanced TGAT
pip install torch>=2.1.0  # Or appropriate version
```

**No motifs discovered**
```bash
# Lower thresholds for discovery
--config-overrides '{"motifs.min_frequency": 2, "motifs.significance_threshold": 0.1}'
```

**Memory issues with large datasets** 
```bash
# Reduce concurrent processing
--config-overrides '{"max_concurrent_sessions": 2, "memory_limit_gb": 4.0}'
```

**M1 data not found**
```bash
# Ensure M1 parquet files exist alongside session JSON
# Pattern: enhanced_NQ_2025-01-15_AM.json -> enhanced_NQ_2025-01-15_AM_M1.parquet
```

### Validation Commands
```bash
# Test DAG acyclicity
python3 tests/unit/test_dag_acyclicity.py

# Validate motif statistics  
python3 tests/unit/test_motif_mining_simple.py

# Check configuration
ironforge build-graph --preset enhanced --dry-run --save-config
```

## Integration Examples

### Python API Usage
```python
from ironforge.learning.dual_graph_config import load_config_with_overrides
from ironforge.learning.dag_graph_builder import DAGGraphBuilder

# Load configuration
config = load_config_with_overrides(
    preset='enhanced',
    overrides={'dag.k_successors': 6, 'm1.enabled': True}
)

# Initialize builder
builder = DAGGraphBuilder({
    'k_successors': config.dag.k_successors,
    'm1_integration': config.m1.enabled
})

# Build dual graphs
temporal_graph, dag_graph = builder.build_dual_view_graphs(session_data)
```

### Enhanced TGAT Usage
```python
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

# Initialize with enhanced TGAT
discovery = IRONFORGEDiscovery(
    node_dim=53,  # M1-enhanced features
    enhanced_tgat=True,
    use_flex_attention=True
)

# Run discovery with DAG constraints
results = discovery.forward(temporal_graph, dag=dag_graph, return_attn=True)
```

### Motif Analysis
```python  
from ironforge.learning.dag_motif_miner import DAGMotifMiner, MotifConfig

# Configure motif mining
motif_config = MotifConfig(
    null_iterations=2000,
    significance_threshold=0.01
)

miner = DAGMotifMiner(motif_config)
motifs = miner.mine_motifs(dags, session_names)

# Analyze results
promote_patterns = [m for m in motifs if m.classification == 'PROMOTE']
print(f"Found {len(promote_patterns)} significant patterns")
```

---

**🚀 IRONFORGE Dual Graph Views v1.0.0**  
*Comprehensive multi-scale graph analysis for financial markets*

**Key Benefits:**
- ✅ **Causal inference** through DAG constraints
- ✅ **Multi-scale integration** with M1 sparse events  
- ✅ **Statistical rigor** with dual null models
- ✅ **Production ready** with optimized storage
- ✅ **Theory B integration** for temporal non-locality
- ✅ **No fallbacks** - fail-fast design principles