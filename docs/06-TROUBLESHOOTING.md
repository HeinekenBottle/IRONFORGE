# IRONFORGE Troubleshooting Guide
**Version**: 1.1.0  
**Last Updated**: 2025-01-15

## 🎯 Overview

Common issues, solutions, and debugging techniques for IRONFORGE.

## 📋 Table of Contents
- [Common Issues](#common-issues)
- [Performance Issues](#performance-issues)
- [Configuration Issues](#configuration-issues)
- [Data Issues](#data-issues)
- [Debugging Techniques](#debugging-techniques)

## 🚨 Common Issues

### Installation Issues

#### Python Version Compatibility
```bash
# Check Python version
python --version

# Should be 3.10 or higher
# If not, upgrade Python or use pyenv
```

#### Dependency Conflicts
```bash
# Create clean virtual environment
python -m venv ironforge-env
source ironforge-env/bin/activate  # Linux/Mac
# or
ironforge-env\Scripts\activate  # Windows

# Install dependencies
pip install -e .[dev]
```

#### Import Errors
```bash
# Check installation
pip list | grep ironforge

# Reinstall if needed
pip uninstall ironforge
pip install -e .[dev]
```

### Configuration Issues

#### Missing Configuration Files
```bash
# Check if config exists
ls -la configs/

# Create default config if missing
cp configs/dev.yml configs/local.yml
```

#### Invalid Configuration
```bash
# Validate configuration
python -c "from ironforge.api import load_config, validate_config; config = load_config('configs/dev.yml'); print('Valid' if validate_config(config) else 'Invalid')"

# Check configuration syntax
# Validate config in Python
python -c "from ironforge.api import load_config, validate_config; cfg=load_config('configs/dev.yml'); validate_config(cfg); print('✅ Valid')"
```

### Data Issues

#### Missing Data Files
```bash
# Check data directory
ls -la data/shards/

# Verify shard files exist
python -c "import glob; print(glob.glob('data/shards/*.parquet'))"
```

#### Data Format Issues
```bash
# Check data format
python -c "import pyarrow.parquet as pq; table = pq.read_table('data/shards/shard_0.parquet'); print(table.schema)"
```

## ⚡ Performance Issues

### Slow Processing

#### Check System Resources
```bash
# Monitor CPU and memory
top -p $(pgrep -f ironforge)

# Check disk I/O
iostat -x 1
```

#### Optimize Configuration
```yaml
# configs/optimized.yml
data:
  shards_glob: "data/shards/*.parquet"
  batch_size: 1000  # Increase batch size

scoring:
  weights:
    cluster_z: 0.30
    htf_prox: 0.25
    structure: 0.20
    cycle: 0.15
    precursor: 0.10

performance:
  max_workers: 4  # Adjust based on CPU cores
  memory_limit: "8GB"
```

#### Use Lazy Loading
```python
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading

# Initialize container once
container = initialize_ironforge_lazy_loading()

# Get components as needed
graph_builder = container.get_enhanced_graph_builder()
discovery = container.get_tgat_discovery()
```

### Memory Issues

#### High Memory Usage
```bash
# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Monitor memory during processing
python -m ironforge.sdk.cli monitor-memory --config configs/dev.yml
```

#### Memory Optimization
```python
# Use smaller batch sizes
config.data.batch_size = 500

# Process sessions individually
for session in sessions:
    result = process_session(session)
    del result  # Explicit cleanup
```

### Timeout Issues

#### Processing Timeouts
```bash
# Check processing time
time python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml

# If running via shell wrapper, use shell-level timeouts (e.g., GNU timeout)
timeout 300s python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml || echo "timed out"
```

## 🔧 Configuration Issues

### Invalid Paths
```bash
# Check path existence
ls -la data/shards/
ls -la runs/

# Create missing directories
mkdir -p data/shards
mkdir -p runs
```

### Permission Issues
```bash
# Check permissions
ls -la data/
ls -la runs/

# Fix permissions
chmod -R 755 data/
chmod -R 755 runs/
```

### Environment Variables
```bash
# Check environment variables
env | grep IRONFORGE

# Set required variables
export IRONFORGE_ENV=dev
export IRONFORGE_CONFIG=configs/dev.yml
```

## 📊 Data Issues

### Corrupted Data
```bash
# Check data integrity
python -c "import pyarrow.parquet as pq; table = pq.read_table('data/shards/shard_0.parquet'); print(f'Rows: {len(table)}, Columns: {len(table.column_names)}')"

# Validate data schema
python -m ironforge.sdk.cli validate-data --config configs/dev.yml
```

### Missing Columns
```bash
# Check required columns
python -c "import pyarrow.parquet as pq; table = pq.read_table('data/shards/shard_0.parquet'); print([col for col in table.column_names if col.startswith('f')])"

# Should have f0-f50 for nodes, e0-e19 for edges
```

### Data Format Mismatch
```bash
# Check data format
python -c "import pyarrow.parquet as pq; table = pq.read_table('data/shards/shard_0.parquet'); print(table.schema)"

# Expected format: Parquet with specific schema
```

## 🔍 Debugging Techniques

### Verbose Logging
```bash
# Enable verbose logging
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml --verbose

# Check log files
tail -f logs/ironforge.log
```

### Debug Mode
```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml --debug
```

### Step-by-Step Debugging
```python
# Debug individual components
from ironforge.api import load_config
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading

config = load_config('configs/dev.yml')
container = initialize_ironforge_lazy_loading()

# Test each component
graph_builder = container.get_enhanced_graph_builder()
print("✅ Graph builder loaded")

discovery = container.get_tgat_discovery()
print("✅ Discovery engine loaded")

# Test data loading
graphs = graph_builder.build_graphs(config.data)
print(f"✅ Built {len(graphs)} graphs")
```

### Performance Profiling
```python
# Profile performance
import cProfile
import pstats

def profile_discovery():
    from ironforge.api import run_discovery, load_config
    config = load_config('configs/dev.yml')
    return run_discovery(config)

# Run profiler
cProfile.run('profile_discovery()', 'profile_output.prof')

# Analyze results
stats = pstats.Stats('profile_output.prof')
stats.sort_stats('cumulative').print_stats(10)
```

## 🆘 Getting Help

### Diagnostic Information
```bash
# Collect diagnostic information
python -m ironforge.sdk.cli diagnose --config configs/dev.yml

# System information
python -m ironforge.sdk.cli system-info
```

### Log Analysis
```bash
# Check recent errors
grep -i error logs/ironforge.log | tail -20

# Check warnings
grep -i warning logs/ironforge.log | tail -20

# Check performance issues
grep -i "slow\|timeout\|memory" logs/ironforge.log | tail -20
```

### Community Support
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check [User Guide](02-USER-GUIDE.md) and [API Reference](03-API-REFERENCE.md)
- **Examples**: Review example configurations and workflows

## 🔗 Related Documentation
- [Quickstart Guide](01-QUICKSTART.md) - Getting started
- [User Guide](02-USER-GUIDE.md) - Complete usage guide
- [API Reference](03-API-REFERENCE.md) - Programmatic interface
- [Architecture](04-ARCHITECTURE.md) - System design
- [Deployment](05-DEPLOYMENT.md) - Production deployment