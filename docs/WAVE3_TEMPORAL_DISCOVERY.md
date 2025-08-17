# Wave 3: Shard-Aware Temporal Discovery

**Status**: ✅ **Complete** - Production-ready shard-aware temporal TGAT discovery pipeline

Wave 3 introduces a sophisticated shard-aware temporal discovery system that scales IRONFORGE's pattern recognition capabilities across large, distributed datasets. The system reads Parquet shards from Wave 1, performs temporal neighbor sampling with configurable fanouts, stitches anchor nodes across shard boundaries, and invokes the existing TGAT discovery engine.

## Architecture Overview

### Core Components

```
Raw Parquet Shards → Shard Loading → Graph Construction → Anchor Stitching → Neighbor Sampling → TGAT Discovery → Pattern Output
```

**Key Features:**
- **Shard-Aware Loading**: Efficiently processes Parquet files with 45D node and 20D edge features
- **Temporal Neighbor Sampling**: Configurable fanouts per TGAT layer with optional time windows
- **Anchor Stitching**: Session-based or global policies for connecting graphs across shards
- **Performance Optimized**: <5s execution, <100MB memory budget
- **CLI Interface**: `ironforge sdk discover-temporal` command for production workflows

### Data Flow

1. **Shard Loading** (`load_shards`): Discovers and loads Parquet files following Wave 1 schema
2. **Graph Construction** (`build_temporal_graph`): Converts DataFrames to PyTorch Geometric HeteroData
3. **Anchor Stitching** (`stitch_anchors`): Merges graphs with session/global policies
4. **Neighbor Sampling** (`create_neighbor_loader`): Creates batched temporal sampling with fanouts
5. **TGAT Discovery** (`_run_tgat_discovery`): Invokes existing IRONFORGEDiscovery engine
6. **Output Generation** (`run`): Saves JSON discoveries and summary statistics

## Quick Start

### Installation

Wave 3 requires the existing IRONFORGE infrastructure plus PyTorch Geometric:

```bash
# Install iron-core (required)
pip install -e iron_core/

# Install IRONFORGE with Wave 3 components
pip install -r requirements.txt

# Verify installation
python -c "from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline; print('✅ Wave 3 ready')"
```

### CLI Usage

```bash
# Basic temporal discovery
python -m ironforge.sdk.cli discover-temporal \
    --data-path /path/to/parquet/shards \
    --output-dir /path/to/results

# Advanced configuration
python -m ironforge.sdk.cli discover-temporal \
    --data-path /data/shards \
    --output-dir /results \
    --fanouts 10 5 3 \
    --batch-size 128 \
    --time-window 6 \
    --stitch-policy global
```

### Programmatic Usage

```python
from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline

# Configure pipeline
pipeline = TemporalDiscoveryPipeline(
    data_path="/path/to/shards",
    num_neighbors=[8, 4, 2],  # Fanouts per TGAT layer
    batch_size=64,
    time_window=2,  # 2-hour temporal window
    stitch_policy="session"  # Session-aware stitching
)

# Execute discovery
discoveries = pipeline.run_discovery()

# Or run with file output
pipeline.output_dir = "/results"
pipeline.run()
```

## Configuration Reference

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | Path | Required | Directory containing Parquet shards |
| `num_neighbors` | List[int] | `[10, 10, 5]` | Neighbor fanouts per TGAT layer |
| `batch_size` | int | `128` | Mini-batch size for sampling |
| `time_window` | int | `None` | Temporal window in hours (unlimited if None) |
| `stitch_policy` | str | `"session"` | Anchor stitching policy (`session` or `global`) |
| `device` | str | `"auto"` | Compute device (`cpu`, `cuda:0`, etc.) |

### CLI Arguments

```bash
python -m ironforge.sdk.cli discover-temporal [OPTIONS]

Required:
  --data-path PATH              Directory containing Parquet shards

Optional:
  --output-dir PATH             Output directory (default: discoveries)
  --fanouts INT [INT ...]       Neighbor fanouts per layer (default: 10 10 5)
  --batch-size INT              Mini-batch size (default: 128)
  --time-window INT             Temporal window in hours (default: unlimited)
  --stitch-policy {session,global}  Anchor stitching policy (default: session)
```

## Data Schema Compatibility

### Input: Wave 1 Parquet Shards

**Node Files** (`*_nodes.parquet`):
- `node_id`: uint32 - Unique node identifier
- `t`: int64 - Timestamp for temporal ordering
- `kind`: uint8 - Node type/category
- `f0`-`f44`: float64 - 45-dimensional node features

**Edge Files** (`*_edges.parquet`):
- `src`, `dst`: uint32 - Source and destination node IDs
- `etype`: uint8 - Edge type/category
- `dt`: int32 - Temporal delta between nodes
- `e0`-`e19`: float64 - 20-dimensional edge features

### Output: Discovery Results

**Discoveries File** (`temporal_discoveries_YYYYMMDD_HHMMSS.json`):
```json
[
  {
    "pattern_type": "temporal_cascade",
    "confidence": 0.92,
    "batch_id": 3,
    "temporal_features": {
      "duration_seconds": 1800,
      "peak_intensity": 0.85
    },
    "spatial_features": {
      "anchor_zones": ["40%", "618%"],
      "session_span": "NY_AM_20250815"
    },
    "pipeline_metadata": {
      "fanouts": [8, 4, 2],
      "batch_size": 64,
      "time_window": 2,
      "stitch_policy": "session"
    }
  }
]
```

**Summary File** (`discovery_summary_YYYYMMDD_HHMMSS.json`):
```json
{
  "timestamp": "20250815_143022",
  "total_patterns": 27,
  "pipeline_config": {
    "data_path": "/path/to/shards",
    "fanouts": [8, 4, 2]
  },
  "pattern_types": {
    "temporal_cascade": 12,
    "cross_session_anchor": 8
  }
}
```

## Performance Guidelines

### Recommended Configurations

**Small Datasets** (< 1K nodes per shard):
```python
TemporalDiscoveryPipeline(
    num_neighbors=[5, 3, 2],
    batch_size=32,
    time_window=1
)
```

**Medium Datasets** (1K-10K nodes per shard):
```python
TemporalDiscoveryPipeline(
    num_neighbors=[10, 5, 3],
    batch_size=64,
    time_window=2
)
```

**Large Datasets** (10K+ nodes per shard):
```python
TemporalDiscoveryPipeline(
    num_neighbors=[20, 10, 5],
    batch_size=128,
    stitch_policy="session"  # More memory efficient
)
```

### Performance Budgets

Wave 3 is designed to meet strict performance requirements:

- **Execution Time**: <5 seconds for typical workloads
- **Memory Usage**: <100MB total footprint
- **Batch Processing**: <100ms per mini-batch
- **Shard Loading**: <2 seconds for 10 moderate-sized shards

Monitor performance with:
```python
import time
start = time.time()
discoveries = pipeline.run_discovery()
print(f"Discovery completed in {time.time() - start:.2f}s")
```

## Anchor Stitching Policies

### Session Policy (`stitch_policy="session"`)

Groups shards by session ID and stitches anchors only within each session:

```
NY_AM_20250815_shard_00 ──┐
                          ├── NY_AM_session_merged
NY_AM_20250815_shard_01 ──┘

LONDON_20250815_shard_00 ──┐
                           ├── LONDON_session_merged
LONDON_20250815_shard_01 ──┘
```

**Pros**: Memory efficient, preserves session boundaries
**Cons**: May miss cross-session patterns

### Global Policy (`stitch_policy="global"`)

Stitches all anchor nodes across all shards:

```
All shards ──────────────► Global merged graph
```

**Pros**: Captures cross-session patterns, maximum discovery potential
**Cons**: Higher memory usage, may blur session-specific patterns

## Error Handling & Troubleshooting

### Common Issues

**Import Error**: `TemporalDiscoveryPipeline not available`
```bash
# Solution: Install required dependencies
pip install torch torch-geometric
pip install -e iron_core/
```

**File Not Found**: `No node Parquet files found`
```bash
# Solution: Verify data path and file naming
ls /path/to/shards/*nodes*.parquet
```

**Memory Issues**: `Discovery pipeline memory budget exceeded`
```python
# Solution: Reduce batch size and use session stitching
pipeline = TemporalDiscoveryPipeline(
    batch_size=32,  # Smaller batches
    stitch_policy="session"  # Memory efficient
)
```

**Performance Issues**: `Pipeline execution time > 5s`
```python
# Solution: Optimize fanouts and enable time windows
pipeline = TemporalDiscoveryPipeline(
    num_neighbors=[5, 3],  # Smaller fanouts
    time_window=1  # Limit temporal scope
)
```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = TemporalDiscoveryPipeline(data_path="/path")
discoveries = pipeline.run_discovery()
```

## Integration with IRONFORGE Ecosystem

### Wave 1 Integration
- **Input**: Reads Parquet shards created by Wave 1 data engine
- **Schema**: Compatible with 45D node, 20D edge feature format
- **Performance**: Leverages Wave 1's optimized storage format

### Wave 2 Integration
- **Modularity**: Uses modular architecture patterns from Wave 2
- **Testing**: Follows Wave 2's comprehensive test framework
- **Quality Gates**: Maintains Wave 2's code quality standards

### Existing TGAT Engine
- **Discovery**: Integrates seamlessly with `IRONFORGEDiscovery`
- **Container**: Uses existing `ironforge_container` for dependency injection
- **Authenticity**: Preserves TGAT's 92.3/100 authenticity scoring

## Testing

### Unit Tests
```bash
# Test core pipeline components
python -m pytest tests/learning/test_discovery_pipeline.py -v

# Test CLI functionality
python -m pytest tests/sdk/test_cli.py -v
```

### Integration Tests
```bash
# Test end-to-end workflows
python -m pytest tests/integration/test_wave3_integration.py -v
```

### Performance Tests
```bash
# Test performance budgets
python -m pytest tests/performance/test_discovery_pipeline_perf.py -v -m performance
```

## Examples

See `examples/wave3_temporal_discovery.py` for comprehensive usage examples including:
- Programmatic API usage
- CLI command examples
- Expected output formats
- Performance optimization tips

## Future Enhancements

### Planned for Wave 4
- **Distributed Processing**: Multi-node temporal discovery
- **Streaming Pipeline**: Real-time shard processing
- **Advanced Stitching**: Custom anchor policies
- **GPU Acceleration**: CUDA-optimized neighbor sampling

### Extension Points
- **Custom Neighbor Samplers**: Implement domain-specific sampling strategies
- **Advanced Stitching Policies**: Create custom anchor merging logic
- **Output Formats**: Add Parquet, HDF5, or streaming outputs
- **Monitoring Integration**: Add metrics and alerting

---

## Development Notes

**Architecture Decision**: Wave 3 preserves the existing TGAT discovery engine rather than replacing it, ensuring backward compatibility and leveraging proven pattern recognition capabilities.

**Performance Philosophy**: The <5s execution budget drives all design decisions, from lazy loading to batched processing to memory-efficient stitching.

**Testing Strategy**: Comprehensive unit, integration, and performance tests ensure production readiness with synthetic data that avoids dependencies on real market data.

**CLI Design**: Simple, composable interface follows Unix philosophy while providing powerful configuration options for advanced users.

---

*Wave 3 Status: ✅ **Production Ready***  
*Last Updated: August 17, 2025*  
*Next: Wave 4 - Distributed Temporal Discovery*