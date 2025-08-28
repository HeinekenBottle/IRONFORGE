# IRONFORGE Quickstart Guide
**Version**: 1.1.0  
**Last Updated**: 2025-01-15

## 🎯 Overview

Get IRONFORGE running in 5 minutes. This guide covers installation, first discovery run, and basic usage.

## 📋 Table of Contents
- [Installation](#installation)
- [First Discovery Run](#first-discovery-run)
- [Configuration](#configuration)
- [Next Steps](#next-steps)

## 🚀 Installation

### Prerequisites
- Python 3.9+
- Git

### Install IRONFORGE
```bash
# Clone repository
git clone <repository-url>
cd IRONFORGE

# Install with dev dependencies
pip install -e .[dev]

# Install pre-commit hooks (optional)
pre-commit install
```

### Verify Installation
```bash
# Check installation
python -m ironforge.sdk.cli --help

# Run smoke tests
python tools/smoke_checks.py
```

## 🔍 First Discovery Run

### Run Complete Pipeline
```bash
# Discover → Score → Validate → Report
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
python -m ironforge.sdk.cli score-session     --config configs/dev.yml
python -m ironforge.sdk.cli validate-run      --config configs/dev.yml
python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml

# Open dashboard
open runs/$(date +%F)/minidash.html
```

### Check Results
```bash
# View run status
python -m ironforge.sdk.cli status --runs runs

# Check artifacts
ls -la runs/$(date +%F)/
```

## ⚙️ Configuration

### Default Configuration
- **Symbol**: ES (E-mini S&P 500)
- **Timeframe**: M5 (5-minute)
- **Data**: `data/shards/*.parquet`
- **Output**: `runs/{date}/`

### Custom Configuration
Create `configs/run.local.yaml`:
```yaml
data:
  symbol: "NQ"           # Change symbol
  timeframe: "1m"        # Change timeframe
  shards_glob: "/path/to/your/shards/*.parquet"

scoring:
  weights:
    cluster_z: 0.40      # Adjust scoring weights
    htf_prox: 0.30
```

## 🎯 Next Steps

### Learn More
- **[User Guide](02-USER-GUIDE.md)** - Complete daily workflows
- **[API Reference](03-API-REFERENCE.md)** - Programmatic usage
- **[Architecture](04-ARCHITECTURE.md)** - System design

### Common Commands
```bash
# Makefile shortcuts
make discover  # discover-temporal
make score     # score-session
make validate  # validate-run
make report    # report-minimal
make status    # status check

# Development
make fmt       # black formatting
make lint      # ruff linting
make test      # pytest
```

### Troubleshooting
- **[Troubleshooting Guide](06-TROUBLESHOOTING.md)** - Common issues
- **Smoke Tests**: `python tools/smoke_checks.py`
- **Verbose Logging**: Add `--verbose` flag to commands

## 🔗 Related Documentation
- [User Guide](02-USER-GUIDE.md) - Complete usage guide
- [API Reference](03-API-REFERENCE.md) - Programmatic interface
- [Architecture](04-ARCHITECTURE.md) - System design
- [Troubleshooting](06-TROUBLESHOOTING.md) - Common issues