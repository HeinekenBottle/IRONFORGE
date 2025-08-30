# IRONFORGE Deployment Guide
**Version**: 1.1.0  
**Last Updated**: 2025-01-15

## 🎯 Overview

Complete guide for deploying IRONFORGE in production environments, including configuration, monitoring, and maintenance.

## 📋 Table of Contents
- [Production Setup](#production-setup)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Maintenance](#maintenance)
- [Troubleshooting](#troubleshooting)

## 🚀 Production Setup

### Prerequisites
- Python 3.10+
- 8GB+ RAM
- 100GB+ storage
- Network access to data sources

### Installation
```bash
# Clone repository
git clone <repository-url>
cd IRONFORGE

# Install production dependencies
pip install -e .[prod]

# Verify installation
python -m ironforge.sdk.cli --help
```

### System Requirements
- **CPU**: 4+ cores recommended
- **Memory**: 8GB minimum, 16GB recommended
- **Storage**: 100GB+ for data and results
- **Network**: Stable connection for data access

## ⚙️ Configuration

### Production Configuration
```yaml
# configs/production.yml
workspace: /data/ironforge/runs/{date}
data:
  shards_glob: "/data/shards/*.parquet"
  symbol: "ES"
  timeframe: "1m"
  
scoring:
  weights:
    cluster_z: 0.30
    htf_prox: 0.25
    structure: 0.20
    cycle: 0.15
    precursor: 0.10

validation:
  authenticity_threshold: 0.87
  confidence_level: 0.95
  significance_threshold: 0.01

reporting:
  minidash:
    out_html: "dashboard.html"
    out_png: "dashboard.png"
    width: 1400
    height: 800

logging:
  level: INFO
  file: "/var/log/ironforge/ironforge.log"
  max_size: "100MB"
  backup_count: 5
```

### Environment Variables
```bash
# Linux/macOS
export IRONFORGE_ENV=production
export IRONFORGE_CONFIG=/etc/ironforge/production.yml
export IRONFORGE_LOG_LEVEL=INFO
export IRONFORGE_DATA_PATH=/data/ironforge
```
```powershell
# Windows PowerShell
$env:IRONFORGE_ENV="production"
$env:IRONFORGE_CONFIG="C:\\ironforge\\production.yml"
$env:IRONFORGE_LOG_LEVEL="INFO"
$env:IRONFORGE_DATA_PATH="C:\\ironforge"
```

## 📊 Monitoring

### Key Metrics
- **Processing Time**: <180 seconds for full discovery
- **Memory Usage**: <100MB total footprint
- **Quality Scores**: >87% authenticity threshold
- **Error Rate**: <1% processing failures

## 🔧 Maintenance

### Daily Maintenance
```bash
# Run daily discovery
python -m ironforge.sdk.cli discover-temporal --config configs/production.yml

# Generate minimal report
python -m ironforge.sdk.cli report-minimal --config configs/production.yml
```

### Weekly Maintenance
```bash
# Validation sweep
python -m ironforge.sdk.cli validate-run --config configs/production.yml
```

### Monthly Maintenance
```bash
# Archive old run artifacts (example procedure)
find /data/ironforge/runs -type d -mtime +90 -print
```

## 🚨 Troubleshooting

See [06-TROUBLESHOOTING.md](06-TROUBLESHOOTING.md) for detailed guidance.

## 🔗 Related Documentation
- [Quickstart Guide](01-QUICKSTART.md) - Getting started
- [User Guide](02-USER-GUIDE.md) - Complete usage guide
- [API Reference](03-API-REFERENCE.md) - Programmatic interface
- [Architecture](04-ARCHITECTURE.md) - System design
- [Troubleshooting](06-TROUBLESHOOTING.md) - Common issues