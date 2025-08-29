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
- Python 3.9+
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
# Set environment variables
export IRONFORGE_ENV=production
export IRONFORGE_CONFIG=/etc/ironforge/production.yml
export IRONFORGE_LOG_LEVEL=INFO
export IRONFORGE_DATA_PATH=/data/ironforge
```

## 📊 Monitoring

### Health Checks
```bash
# System health check
python -m ironforge.sdk.cli health-check --config configs/production.yml

# Performance monitoring
python -m ironforge.sdk.cli monitor --config configs/production.yml
```

### Key Metrics
- **Processing Time**: <180 seconds for full discovery
- **Memory Usage**: <100MB total footprint
- **Quality Scores**: >87% authenticity threshold
- **Error Rate**: <1% processing failures

### Logging
```python
import logging
from ironforge.api import load_config

# Configure logging
config = load_config('configs/production.yml')
logging.basicConfig(
    level=config.logging.level,
    filename=config.logging.file,
    maxBytes=config.logging.max_size,
    backupCount=config.logging.backup_count
)
```

## 🔧 Maintenance

### Daily Maintenance
```bash
# Run daily discovery
python -m ironforge.sdk.cli discover-temporal --config configs/production.yml

# Check system health
python -m ironforge.sdk.cli health-check --config configs/production.yml

# Clean old runs (optional)
python -m ironforge.sdk.cli cleanup --runs /data/ironforge/runs --days 30
```

### Weekly Maintenance
```bash
# Full system validation
python -m ironforge.sdk.cli validate-system --config configs/production.yml

# Performance analysis
python -m ironforge.sdk.cli analyze-performance --config configs/production.yml

# Update documentation
python -m ironforge.sdk.cli update-docs --config configs/production.yml
```

### Monthly Maintenance
```bash
# Archive old data
python -m ironforge.sdk.cli archive --source /data/ironforge/runs --target /archive/ironforge

# System optimization
python -m ironforge.sdk.cli optimize --config configs/production.yml

# Security audit
python -m ironforge.sdk.cli security-audit --config configs/production.yml
```

## 🚨 Troubleshooting

### Common Issues

#### Performance Issues
```bash
# Check system resources
python -m ironforge.sdk.cli system-info

# Analyze performance bottlenecks
python -m ironforge.sdk.cli analyze-performance --config configs/production.yml

# Optimize configuration
python -m ironforge.sdk.cli optimize-config --config configs/production.yml
```

#### Memory Issues
```bash
# Check memory usage
python -m ironforge.sdk.cli memory-check

# Reduce memory footprint
python -m ironforge.sdk.cli reduce-memory --config configs/production.yml
```

#### Data Issues
```bash
# Validate data integrity
python -m ironforge.sdk.cli validate-data --config configs/production.yml

# Check data format
python -m ironforge.sdk.cli check-data-format --config configs/production.yml
```

### Error Recovery
```bash
# Recover from failed runs
python -m ironforge.sdk.cli recover --run-id <run-id>

# Reset system state
python -m ironforge.sdk.cli reset --config configs/production.yml

# Emergency shutdown
python -m ironforge.sdk.cli emergency-stop
```

## 🔗 Related Documentation
- [Quickstart Guide](01-QUICKSTART.md) - Getting started
- [User Guide](02-USER-GUIDE.md) - Complete usage guide
- [API Reference](03-API-REFERENCE.md) - Programmatic interface
- [Architecture](04-ARCHITECTURE.md) - System design
- [Troubleshooting](06-TROUBLESHOOTING.md) - Common issues