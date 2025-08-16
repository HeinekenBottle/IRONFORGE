# IRONFORGE Refactoring Migration Guide

## Overview

This guide helps you migrate from the old flat file structure to the new organized package structure introduced in the comprehensive refactoring.

## Breaking Changes Summary

### 1. Import Path Changes

**All imports now use the `ironforge.` package prefix:**

```python
# OLD (will fail)
from learning.enhanced_graph_builder import EnhancedGraphBuilder
from analysis.broad_spectrum_archaeology import BroadSpectrumArchaeology
from integration.ironforge_container import get_ironforge_container

# NEW (correct)
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.analysis.broad_spectrum_archaeology import BroadSpectrumArchaeology
from ironforge.integration.ironforge_container import get_ironforge_container
```

### 2. File Location Changes

| Old Location | New Location |
|--------------|--------------|
| `test_*.py` (root) | `tests/integration/test_*.py` |
| `run_*.py` (root) | `scripts/analysis/run_*.py` |
| `analyze_*.py` (root) | `scripts/analysis/analyze_*.py` |
| `*_processor.py` (root) | `scripts/data_processing/*_processor.py` |
| `performance_monitor.py` (root) | `ironforge/utilities/performance_monitor.py` |
| `*.json` data (root) | `data/discoveries/*.json` |
| `*.md` docs (root) | `docs/*.md` |

### 3. Data Directory Structure

```
# OLD
enhanced_sessions/
enhanced_sessions_with_relativity/
adapted_enhanced_sessions/
discoveries/

# NEW
data/
├── raw/           # Level 1 raw market data
├── enhanced/      # Enhanced/processed sessions
├── adapted/       # Adapted sessions with relativity
└── discoveries/   # Pattern discoveries
```

## Migration Steps

### Step 1: Update Import Statements

Use this regex pattern to find and replace imports:

```bash
# Find old imports
grep -r "from learning\." your_files/
grep -r "from analysis\." your_files/
grep -r "from synthesis\." your_files/
grep -r "from integration\." your_files/

# Replace with new imports
sed -i 's/from learning\./from ironforge.learning./g' your_files/*.py
sed -i 's/from analysis\./from ironforge.analysis./g' your_files/*.py
sed -i 's/from synthesis\./from ironforge.synthesis./g' your_files/*.py
sed -i 's/from integration\./from ironforge.integration./g' your_files/*.py
```

### Step 2: Update File Paths in Code

```python
# OLD
data_path = "enhanced_sessions/session_data.json"
discovery_path = "discoveries/patterns.json"

# NEW
data_path = "data/enhanced/session_data.json"
discovery_path = "data/discoveries/patterns.json"
```

### Step 3: Update Configuration

The configuration system automatically handles the new paths, but if you have hardcoded paths:

```python
# OLD
config = {
    'session_data_path': 'data/sessions/level_1',
    'discoveries_path': 'discoveries'
}

# NEW (handled automatically by config.py)
config = {
    'session_data_path': 'data/raw',
    'discoveries_path': 'data/discoveries'
}
```

## Common Issues & Solutions

### Issue 1: Import Errors

**Error**: `ModuleNotFoundError: No module named 'learning'`

**Solution**: Update import to use `ironforge.learning`

```python
# Fix this
from learning.enhanced_graph_builder import EnhancedGraphBuilder

# To this
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
```

### Issue 2: File Not Found Errors

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'enhanced_sessions/data.json'`

**Solution**: Update path to new data structure

```python
# Fix this
with open('enhanced_sessions/data.json') as f:

# To this
with open('data/enhanced/data.json') as f:
```

### Issue 3: Container Component Access

**Error**: `AttributeError: 'IRONFORGEContainer' object has no attribute 'get_performance_monitor'`

**Solution**: The performance monitor is now properly registered

```python
# This now works correctly
container = get_ironforge_container()
monitor = container.get_performance_monitor()
```

## Validation

After migration, run these commands to validate:

```bash
# Test imports
python -c "from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder; print('✅ Imports working')"

# Test structure
python test_refactored_structure.py

# Test performance
python benchmark_performance.py
```

## Rollback Procedure

If you need to rollback to the old structure:

```bash
git checkout main
# This restores the previous flat file structure
```

## Support

If you encounter issues during migration:

1. Check this migration guide for common solutions
2. Run the validation scripts to identify specific problems
3. Review the test files in `tests/integration/` for examples of correct usage
4. Check the benchmark results to ensure performance is maintained

## Benefits After Migration

- ✅ Clean package structure
- ✅ No circular imports
- ✅ Organized data flow
- ✅ Better IDE support
- ✅ Easier testing and debugging
- ✅ Production-ready architecture
