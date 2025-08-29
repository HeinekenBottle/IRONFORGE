# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IRONFORGE is a sophisticated archaeological discovery engine for market pattern analysis. It combines rule-based preprocessing, TGAT (Temporal Graph Attention Networks) machine learning, and rule-based scoring to discover temporal patterns within trading sessions. The system operates on enhanced session data through a canonical pipeline that preserves pattern authenticity while enabling rapid discovery.

## Architecture

### Core Pipeline
The system follows a strict 4-stage canonical pipeline:

1. **Discovery** (`discover-temporal`) - TGAT-based pattern discovery from enhanced session graphs
2. **Confluence** (`score-session`) - Rule-based confluence scoring and validation
3. **Validation** (`validate-run`) - Quality gates and validation rails
4. **Reporting** (`report-minimal`) - Minidash dashboard generation

### Key Components

- **Enhanced Graph Builder** - Transforms JSON sessions into 45D/20D TGAT-compatible graphs
- **TGAT Discovery** - Temporal graph attention networks for pattern learning
- **Pattern Graduation** - Validates patterns against 87% authenticity threshold
- **Confluence Scoring** - Rule-based scoring with configurable weights
- **Minidash Reporting** - Interactive HTML dashboards with PNG export

### Data Contracts (Golden Invariants - Never Change)

- **Events**: Exactly 6 types (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
- **Edge Intents**: Exactly 4 types (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)  
- **Feature Dimensions**: 51D nodes (f0-f50), 20D edges (e0-e19)
- **HTF Rule**: Last-closed only (no intra-candle)
- **Session Boundaries**: No cross-session edges
- **Within-session Learning**: Preserve session isolation

## Commands

### Development Setup
```bash
# Install with dev dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Alternative setup via Makefile
make setup
```

### Core Pipeline Commands
```bash
# Run complete canonical pipeline
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
python -m ironforge.sdk.cli score-session --config configs/dev.yml
python -m ironforge.sdk.cli validate-run --config configs/dev.yml
python -m ironforge.sdk.cli report-minimal --config configs/dev.yml

# Check run status
python -m ironforge.sdk.cli status --runs runs

# Makefile shortcuts
make discover  # discover-temporal
make score     # score-session
make validate  # validate-run
make report    # report-minimal
make status    # status check
```

### Testing & Quality
```bash
# Run smoke tests (lightweight validation)
python tools/smoke_checks.py

# Full test suite
pytest -v

# Specific test categories
pytest tests/contracts -v       # Contract validation tests
pytest tests/unit -v           # Unit tests
pytest tests/performance -v    # Performance tests

# Code quality
make fmt      # black formatting
make lint     # ruff linting  
make type     # mypy type checking
make test     # pytest
make precommit # run all pre-commit hooks

# CI validation
make ci-validate
```

### Data Preparation
```bash
# Convert enhanced sessions to shards (45D features by default)
python -m ironforge.sdk.cli prep-shards

# Enable HTF context features (51D features)
python -m ironforge.sdk.cli prep-shards --htf-context

# Check shard dimensions
python -c "
import pyarrow.parquet as pq
nodes = pq.read_table('data/shards/*/nodes.parquet')
print(f'Node features: {len([c for c in nodes.column_names if c.startswith(\"f\")])}')"
```

### Oracle Training (Temporal Non-locality Predictions)
```bash
# Train Oracle model
ironforge train-oracle --symbols NQ --tf M5 --from 2025-07-20 --to 2025-08-15 --out models/oracle/v1.1.0

# Run discovery with Oracle enabled
ironforge discover-temporal --oracle-enabled
```

## Code Architecture

### Package Structure
```
ironforge/
├── api.py              # Centralized API (recommended import point)
├── sdk/               # CLI and configuration management
├── learning/          # TGAT discovery and enhanced graph building
├── confluence/        # Confluence scoring engine
├── validation/        # Quality gates and validation rails
├── reporting/         # Minidash dashboard generation
├── analysis/          # Pattern intelligence and workflows
├── synthesis/         # Pattern graduation and quality control
├── contracts/         # Data contracts and schema validation
├── temporal/          # Temporal intelligence systems
├── integration/       # Container system and lazy loading
└── utilities/         # Common utilities and helpers
```

### Import Patterns

**Recommended**: Use centralized API for stable interface
```python
from ironforge.api import run_discovery, score_confluence, validate_run, build_minidash
from ironforge.api import Config, load_config, materialize_run_dir
```

**Internal Development**: Direct module imports when extending core functionality
```python
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
from ironforge.synthesis.pattern_graduation import PatternGraduation
```

### Configuration

Configuration files use YAML format in `configs/`:
- `dev.yml` - Development configuration
- Production configs follow same schema with different paths/parameters

Key configuration sections:
- `data`: Shards location, symbol, timeframe
- `scoring.weights`: Confluence scoring weights
- `reporting`: Dashboard output settings
- `validation`: Quality validation parameters

### Performance Specifications

The system has strict performance requirements:
- **Single Session**: <3 seconds processing
- **Full Discovery**: <180 seconds (57 sessions)  
- **Initialization**: <2 seconds with lazy loading
- **Memory Usage**: <100MB total footprint
- **Quality**: >87% authenticity threshold for production patterns

### Container System

IRONFORGE uses a sophisticated dependency injection container for performance:
```python
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading

# Initialize container (do this once)
container = initialize_ironforge_lazy_loading()

# Get components
graph_builder = container.get_enhanced_graph_builder()
discovery = container.get_tgat_discovery()
```

### Data Flow

1. **Enhanced Sessions** (JSON) → **Enhanced Graph Builder** → **45D/20D Graphs**
2. **Graphs** → **TGAT Discovery** → **Archaeological Patterns**  
3. **Patterns** → **Pattern Graduation** → **Validated Patterns** (>87% authenticity)
4. **Validated Patterns** → **Confluence Scoring** → **Scored Patterns**
5. **Scored Patterns** → **Minidash Reporting** → **Interactive Dashboard**

### Output Structure

Each run creates organized outputs in `runs/YYYY-MM-DD/`:
```
runs/2025-01-15/
├── embeddings/           # TGAT model outputs and attention weights
├── patterns/            # Discovered temporal patterns
├── confluence/          # Confluence scores and statistics  
├── motifs/              # Pattern motif analysis
├── aux/                 # Read-only context (trajectories, chains)
├── minidash.html        # Interactive dashboard
└── minidash.png         # Static dashboard export
```

### Quality Gates

The system enforces strict quality controls:
- **Authenticity Score**: >87/100 for production
- **Duplication Rate**: <25% 
- **Temporal Coherence**: >70%
- **Pattern Confidence**: >0.7 threshold
- **Contract Validation**: Automatic schema validation

### Development Practices

- **Feature Dimensions**: Never change 51D nodes / 20D edges without major version
- **Session Isolation**: Never add cross-session edges (violates archaeological principles)
- **HTF Rule**: Only last-closed HTF data (f45-f50), never intra-candle
- **Event Taxonomy**: Exactly 6 event types, never add/remove without architecture review
- **Lazy Loading**: Use container system for performance-critical components
- **Config-Driven**: All adapters configurable, off by default
- **Quality-First**: All patterns must pass graduation thresholds

### Testing Philosophy

- **Smoke Tests**: Quick validation via `tools/smoke_checks.py`
- **Contract Tests**: Validate data schemas and golden invariants
- **Performance Tests**: Ensure <3s session processing, <180s full discovery
- **Integration Tests**: End-to-end pipeline validation
- **Golden Tests**: Reference outputs for regression detection

### Debugging

- **Attention Analysis**: Check `embeddings/attention_topk.parquet` for model explanability
- **Scale Detection**: Dashboard shows 0-1, 0-100, or threshold-normalized badges
- **Health Gates**: System health indicators in confluence stats
- **Verbose Logging**: Use `--verbose` flag for detailed processing logs

This system represents a sophisticated balance of machine learning and rule-based archaeology, designed for production-grade pattern discovery with strict quality guarantees.