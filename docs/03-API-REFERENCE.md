# IRONFORGE API Reference
**Version**: 1.1.0  
**Last Updated**: 2025-01-15

## 🎯 Overview

Complete API reference for IRONFORGE's stable public interface. All functions are available through the centralized `ironforge.api` module.

## 📋 Table of Contents
- [Quick Reference](#quick-reference)
- [Core Engines](#core-engines)
- [Configuration](#configuration)
- [CLI Interface](#cli-interface)
- [Integration](#integration)

## 🚀 Quick Reference

### Recommended Imports
```python
from ironforge.api import (
    run_discovery, score_confluence, validate_run, build_minidash,
    LoaderCfg, Paths, RunCfg, Config, load_config, materialize_run_dir
)
```

### Core Pipeline
```python
# Discovery → Confluence → Validation → Reporting
patterns = run_discovery(
    shard_paths=["data/shards/NQ_5m/shard_2024-12-15"],
    out_dir="runs/2025-01-15/NQ_5m",
    loader_cfg=LoaderCfg(fanouts=(10, 10), batch_size=2048),
)

scores_path = score_confluence(
    pattern_paths=patterns,
    out_dir="runs/2025-01-15/NQ_5m/confluence",
    _weights=None,
    threshold=65.0,
)

run_cfg = RunCfg(paths=Paths(shards_dir="data/shards/NQ_5m", out_dir="runs/2025-01-15/NQ_5m"))
validation = validate_run(run_cfg)
```

## 🔧 Core Engines

### Discovery Engine

#### `run_discovery(shard_paths: Iterable[str], out_dir: str, loader_cfg: LoaderCfg) -> list[str]`
Run TGAT-based pattern discovery over shard directories and write outputs to a run directory.

**Parameters:**
- `shard_paths`: Iterable of shard directories containing `nodes.parquet` and `edges.parquet`
- `out_dir`: Output directory for run artifacts (embeddings/, patterns/)
- `loader_cfg`: Loader configuration (fanouts, batch size, etc.)

**Returns:**
- `list[str]`: Pattern parquet file paths for each processed shard

**Example:**
```python
from ironforge.api import run_discovery, LoaderCfg

patterns = run_discovery(
    shard_paths=["data/shards/NQ_5m/shard_2024-12-15"],
    out_dir="runs/2025-01-15/NQ_5m",
    loader_cfg=LoaderCfg(fanouts=(10, 10), batch_size=2048),
)
```

**CLI Equivalent:**
```bash
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
```

### Confluence Engine

#### `score_confluence(pattern_paths: Sequence[str], out_dir: str, _weights: Mapping[str, float] | None, threshold: float, hierarchical_config: dict | None = None) -> str`
Run BMAD‑enhanced confluence scoring over discovered pattern files.

**Parameters:**
- `pattern_paths`: List of pattern parquet paths
- `out_dir`: Output directory for confluence artifacts
- `_weights`: Optional weights mapping for scoring
- `threshold`: Score threshold (0–100)
- `hierarchical_config`: Optional dict enabling hierarchical coherence analysis

**Returns:**
- `str`: Path to `scores.parquet`

**Example:**
```python
from ironforge.api import score_confluence
scores_path = score_confluence(patterns, out_dir="runs/2025-01-15/NQ_5m/confluence", _weights=None, threshold=65.0)
```

**CLI Equivalent:**
```bash
python -m ironforge.sdk.cli score-session --config configs/dev.yml
```

### Validation Engine

#### `validate_run(config) -> dict`
Runs quality gates and validation rails.

**Parameters:**
- `config`: Configuration object with validation settings

**Returns:**
- `dict`: Validation results, quality metrics, and compliance status

**Example:**
```python
from ironforge.api import validate_run, load_config

config = load_config('configs/dev.yml')
results = validate_run(config)

# Access results
quality_metrics = results['quality_metrics']
compliance_status = results['compliance_status']
validation_summary = results['validation_summary']
```

**CLI Equivalent:**
```bash
python -m ironforge.sdk.cli validate-run --config configs/dev.yml
```

### Reporting Engine

#### `build_minidash(activity: pd.DataFrame, confluence: pd.DataFrame, motifs: list[dict], out_html, out_png, width=1200, height=700, htf_regime_data=None) -> (Path, Path)`
Generate a minimal dashboard from activity and confluence time series.

**Parameters:**
- `activity`: DataFrame with columns `ts`, `count`
- `confluence`: DataFrame with columns `ts`, `score`
- `motifs`: List of motif dicts for tabular display
- `out_html`, `out_png`: Output file paths
- `width`, `height`: Figure size in pixels
- `htf_regime_data`: Optional HTF regime context

**Returns:**
- `(Path, Path)`: Paths to `(html, png)`

**Example:**
```python
import pandas as pd
from ironforge.api import build_minidash
activity = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=10, freq="min"), "count": range(10)})
confluence_df = pd.DataFrame({"ts": activity["ts"], "score": range(0, 100, 10)})
build_minidash(activity, confluence_df, motifs=[], out_html="runs/2025-01-15/minidash.html", out_png="runs/2025-01-15/minidash.png")
```

**CLI Equivalent:**
```bash
python -m ironforge.sdk.cli report-minimal --config configs/dev.yml
```

## ⚙️ Configuration

### Configuration Classes

#### `Config`
Main configuration dataclass containing all system settings.

```python
from ironforge.api import Config

config = Config(
    workspace="runs/2025-01-15",
    data=DataCfg(
        shards_glob="data/shards/*.parquet",
        symbol="ES",
        timeframe="1m"
    ),
    scoring=ScoringCfg(
        weights=WeightsCfg(
            cluster_z=0.30,
            htf_prox=0.25,
            structure=0.20,
            cycle=0.15,
            precursor=0.10
        )
    )
)
```

#### `DataCfg`
Data configuration settings.

```python
from ironforge.api import DataCfg

data_config = DataCfg(
    shards_glob="data/shards/*.parquet",
    symbol="ES",
    timeframe="1m"
)
```

#### `ScoringCfg`
Scoring configuration settings.

```python
from ironforge.api import ScoringCfg, WeightsCfg

scoring_config = ScoringCfg(
    weights=WeightsCfg(
        cluster_z=0.30,
        htf_prox=0.25,
        structure=0.20,
        cycle=0.15,
        precursor=0.10
    )
)
```

### Configuration Functions

#### `load_config(config_path: str) -> Config`
Loads configuration from YAML file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:**
- `Config`: Loaded configuration object

**Example:**
```python
from ironforge.api import load_config

config = load_config('configs/dev.yml')
```

#### `validate_config(config: Config) -> bool`
Validates configuration object.

**Parameters:**
- `config`: Configuration object to validate

**Returns:**
- `bool`: True if configuration is valid

**Example:**
```python
from ironforge.api import validate_config, load_config

config = load_config('configs/dev.yml')
if validate_config(config):
    print("✅ Configuration is valid")
else:
    print("❌ Configuration validation failed")
```

#### `materialize_run_dir(config: Config) -> str`
Creates and returns run directory path.

**Parameters:**
- `config`: Configuration object

**Returns:**
- `str`: Path to run directory

**Example:**
```python
from ironforge.api import materialize_run_dir, load_config

config = load_config('configs/dev.yml')
run_dir = materialize_run_dir(config)
print(f"Run directory: {run_dir}")
```

## 🖥️ CLI Interface

### Core Commands

#### Discovery
```bash
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
```

#### Confluence Scoring
```bash
python -m ironforge.sdk.cli score-session --config configs/dev.yml
```

#### Validation
```bash
python -m ironforge.sdk.cli validate-run --config configs/dev.yml
```

#### Reporting
```bash
python -m ironforge.sdk.cli report-minimal --config configs/dev.yml
```

#### Status Check
```bash
python -m ironforge.sdk.cli status --runs runs
```

### CLI Options

#### Common Options
- `--config`: Configuration file path
- `--verbose`: Enable verbose logging
- `--dry-run`: Show what would be done without executing
- `--help`: Show help message

#### Discovery Options
- `--oracle-enabled`: Enable Oracle predictions
- `--max-sessions`: Maximum number of sessions to process
- `--min-confidence`: Minimum confidence threshold

#### Scoring Options
- `--weights-override`: Override scoring weights
- `--validation-strict`: Use strict validation mode

## 🔗 Integration

### Container System

#### `get_ironforge_container() -> IRONContainer`
Gets the IRONFORGE dependency injection container.

**Returns:**
- `IRONContainer`: Container with lazy-loaded components

**Example:**
```python
from ironforge.api import get_ironforge_container

container = get_ironforge_container()
graph_builder = container.get_enhanced_graph_builder()
discovery = container.get_tgat_discovery()
```

#### `initialize_ironforge_lazy_loading() -> IRONContainer`
Initializes the container with lazy loading.

**Returns:**
- `IRONContainer`: Initialized container

**Example:**
```python
from ironforge.api import initialize_ironforge_lazy_loading

container = initialize_ironforge_lazy_loading()
```

### Advanced Usage

#### Custom Discovery
```python
from ironforge.api import get_ironforge_container, load_config

# Get container and components
container = get_ironforge_container()
graph_builder = container.get_enhanced_graph_builder()
discovery = container.get_tgat_discovery()

# Load configuration
config = load_config('configs/dev.yml')

# Custom discovery workflow
graphs = graph_builder.build_graphs(config.data)
patterns = discovery.discover_patterns(graphs)
```

#### Batch Processing
```python
from ironforge.api import run_discovery, load_config
from pathlib import Path

# Process multiple configurations
config_files = Path('configs').glob('*.yml')
results = {}

for config_file in config_files:
    config = load_config(str(config_file))
    results[config_file.stem] = run_discovery(config)
```

## 🔗 Related Documentation
- [Quickstart Guide](01-QUICKSTART.md) - Getting started
- [User Guide](02-USER-GUIDE.md) - Complete usage guide
- [Architecture](04-ARCHITECTURE.md) - System design
- [Troubleshooting](06-TROUBLESHOOTING.md) - Common issues