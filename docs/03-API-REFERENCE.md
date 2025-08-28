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
    Config, load_config, materialize_run_dir
)
```

### Core Pipeline
```python
# Complete pipeline in 4 steps
config = load_config('configs/dev.yml')
discovery_results = run_discovery(config)
confluence_results = score_confluence(config)
validation_results = validate_run(config)
dashboard = build_minidash(config)
```

## 🔧 Core Engines

### Discovery Engine

#### `run_discovery(config: Config) -> Dict[str, Any]`
Runs TGAT-based pattern discovery from enhanced session graphs.

**Parameters:**
- `config`: Configuration object with data paths and discovery settings

**Returns:**
- `Dict[str, Any]`: Discovery results with patterns, embeddings, and metadata

**Example:**
```python
from ironforge.api import run_discovery, load_config

config = load_config('configs/dev.yml')
results = run_discovery(config)

# Access results
patterns = results['patterns']
embeddings = results['embeddings']
attention_weights = results['attention_weights']
```

**CLI Equivalent:**
```bash
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
```

### Confluence Engine

#### `score_confluence(config: Config) -> Dict[str, Any]`
Runs rule-based confluence scoring and validation.

**Parameters:**
- `config`: Configuration object with scoring weights and validation settings

**Returns:**
- `Dict[str, Any]`: Confluence scores, statistics, and validation results

**Example:**
```python
from ironforge.api import score_confluence, load_config

config = load_config('configs/dev.yml')
results = score_confluence(config)

# Access results
scores = results['confluence_scores']
statistics = results['statistics']
validation = results['validation_results']
```

**CLI Equivalent:**
```bash
python -m ironforge.sdk.cli score-session --config configs/dev.yml
```

### Validation Engine

#### `validate_run(config: Config) -> Dict[str, Any]`
Runs quality gates and validation rails.

**Parameters:**
- `config`: Configuration object with validation settings

**Returns:**
- `Dict[str, Any]`: Validation results, quality metrics, and compliance status

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

#### `build_minidash(config: Config) -> Dict[str, Any]`
Generates interactive HTML dashboards with PNG export.

**Parameters:**
- `config`: Configuration object with reporting settings

**Returns:**
- `Dict[str, Any]`: Dashboard generation results and file paths

**Example:**
```python
from ironforge.api import build_minidash, load_config

config = load_config('configs/dev.yml')
results = build_minidash(config)

# Access results
html_path = results['html_path']
png_path = results['png_path']
dashboard_url = results['dashboard_url']
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