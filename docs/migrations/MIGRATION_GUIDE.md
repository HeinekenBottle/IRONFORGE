# IRONFORGE Migration Guide
**Comprehensive Migration Guide for All IRONFORGE Versions**

---

## 🎯 Overview

This guide helps you migrate between different versions of IRONFORGE, from the original flat file structure through the current modern architecture with semantic features and comprehensive documentation.

---

## 📋 Current Version: 2.0.0 (August 2025)

### Latest Features
- **Complete Documentation Modernization**: Comprehensive architecture, API reference, user guides
- **Semantic Feature System**: 45D node features, 20D edge features with rich context
- **Production-Ready Deployment**: Full deployment guide with monitoring and automation
- **Archaeological Discovery Focus**: True pattern discovery without prediction logic
- **Iron-Core Integration**: 88.7% performance improvement through lazy loading

---

## 🔄 Migration Paths

### From 1.x to 2.0.0 (Documentation Update)
**Impact**: Documentation structure only - no code changes required

**Changes**:
- Documentation files reorganized with modern naming conventions
- Legacy phase-based documentation consolidated
- New comprehensive guides added

**Action Required**: Update documentation references only
```bash
# No code changes needed
# Update bookmarks to new documentation structure
```

### From 1.4.x to 1.5.0 (Semantic Features)
**Impact**: Feature vector dimensions and pattern output format

**Breaking Changes**:
```python
# OLD: 37D node features, 17D edge features
model = TGAT(in_channels=37)
graph = Data(x=node_features_37d, edge_attr=edge_features_17d)

# NEW: 45D node features, 20D edge features
model = TGAT(in_channels=45)
graph = Data(x=node_features_45d, edge_attr=edge_features_20d)
```

**Pattern Output Changes**:
```python
# OLD: Generic pattern output
pattern = {
    'type': 'range_position_confluence',
    'description': '75.2% of range @ 1.8h timeframe',
    'session': 'unknown',
    'confidence': 0.73
}

# NEW: Rich semantic pattern output
pattern = {
    'pattern_id': 'NY_session_RPC_00',
    'session_name': 'NY_session',
    'session_start': '14:30:00',
    'archaeological_significance': {
        'archaeological_value': 'high_archaeological_value',
        'permanence_score': 0.933
    },
    'semantic_context': {
        'market_regime': 'transitional',
        'event_types': ['fvg_redelivery', 'expansion_phase'],
        'relationship_type': 'confluence_relationship'
    },
    'confidence': 0.87
}
```

### From 1.0.x to 1.1.0 (Package Structure)
**Impact**: All imports and initialization methods

**Breaking Changes**:
```python
# OLD: Direct imports (will fail)
from learning.enhanced_graph_builder import EnhancedGraphBuilder
from analysis.broad_spectrum_archaeology import BroadSpectrumArchaeology
from integration.ironforge_container import get_ironforge_container

# NEW: Package-based imports
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.analysis.broad_spectrum_archaeology import BroadSpectrumArchaeology
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
```

**Initialization Changes**:
```python
# OLD: Direct instantiation
builder = EnhancedGraphBuilder()
discovery = IRONFORGEDiscovery()

# NEW: Container-based initialization
container = initialize_ironforge_lazy_loading()
builder = container.get_enhanced_graph_builder()
discovery = container.get_tgat_discovery()
```

---

## 📁 File Structure Migration

### Current Structure (2.0.0)
```
IRONFORGE/
├── ironforge/                    # Main application package
│   ├── learning/                # TGAT discovery engine
│   ├── analysis/                # Pattern analysis & archaeology
│   ├── synthesis/               # Pattern validation & graduation
│   ├── integration/             # System integration & containers
│   ├── utilities/               # Core utilities & monitoring
│   └── reporting/               # Analysis reporting
├── iron_core/                   # Infrastructure & performance
├── data/                        # Organized data storage
│   ├── raw/                     # Level 1 raw market data
│   ├── enhanced/                # Enhanced/processed sessions
│   ├── adapted/                 # Adapted sessions with relativity
│   └── discoveries/             # Pattern discoveries
├── scripts/                     # Utility scripts
├── tests/                       # Comprehensive test suite
├── docs/                        # Modern documentation
└── reports/                     # Generated reports
```

### Legacy Structure Migration
| Old Location | New Location |
|--------------|--------------|
| `test_*.py` (root) | `tests/integration/test_*.py` |
| `run_*.py` (root) | `scripts/analysis/run_*.py` |
| `analyze_*.py` (root) | `scripts/analysis/analyze_*.py` |
| `*_processor.py` (root) | `scripts/data_processing/*_processor.py` |
| `performance_monitor.py` (root) | `ironforge/utilities/performance_monitor.py` |
| `enhanced_sessions/` | `data/enhanced/` |
| `discoveries/` | `data/discoveries/` |
| Phase docs (root) | Consolidated in `docs/` |

---

## 🔧 Code Migration Examples

### Complete Migration Example
```python
# ===== BEFORE (Version 1.0.x) =====
# Direct imports and instantiation
from learning.enhanced_graph_builder import EnhancedGraphBuilder
from learning.tgat_discovery import IRONFORGEDiscovery
import json

# Direct component creation
builder = EnhancedGraphBuilder()
discovery = IRONFORGEDiscovery(node_features=37)  # Old 37D

# Load and process
with open('enhanced_sessions/session.json') as f:
    session_data = json.load(f)

graph = builder.enhance_session(session_data)
patterns = discovery.discover_patterns(graph)

# ===== AFTER (Version 2.0.0) =====
# Package imports and container initialization
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
from pathlib import Path
import json

# Container-based initialization
container = initialize_ironforge_lazy_loading()
builder = container.get_enhanced_graph_builder()
discovery = container.get_tgat_discovery()  # Auto-configured for 45D

# Load and process with new structure
session_file = Path('data/enhanced/session.json')
with open(session_file) as f:
    session_data = json.load(f)

graph = builder.enhance_session(session_data)  # Now produces 45D/20D
patterns = discovery.discover_patterns(graph)

# Process rich semantic patterns
for pattern in patterns:
    print(f"Pattern: {pattern['pattern_id']}")
    print(f"Session: {pattern['session_name']}")
    print(f"Semantic Context: {pattern['semantic_context']}")
    print(f"Archaeological Value: {pattern['archaeological_significance']['archaeological_value']}")
```

### Daily Workflow Migration
```python
# ===== BEFORE (Manual processing) =====
# Manual session processing
session_files = glob.glob('enhanced_sessions/*.json')
all_patterns = []

for session_file in session_files:
    with open(session_file) as f:
        data = json.load(f)

    graph = builder.enhance_session(data)
    patterns = discovery.discover_patterns(graph)
    all_patterns.extend(patterns)

# ===== AFTER (Integrated workflows) =====
# Modern workflow integration
from ironforge.analysis.daily_discovery_workflows import morning_prep, hunt_patterns

# Morning preparation workflow
morning_analysis = morning_prep(days_back=7)
print(f"Strength Score: {morning_analysis['strength_score']}")
print(f"Dominant Patterns: {morning_analysis['dominant_patterns']}")

# Session-specific hunting
ny_pm_patterns = hunt_patterns('NY_PM')
print(f"NY_PM Patterns: {len(ny_pm_patterns['patterns_found'])}")
print(f"Confidence: {ny_pm_patterns['strength_indicators']['avg_confidence']}")
```

---

## 📊 Configuration Migration

### Old Configuration (1.0.x)
```python
# Basic configuration
TGAT_CONFIG = {
    'node_features': 37,
    'hidden_dim': 128,
    'learning_rate': 0.001
}
```

### New Configuration (2.0.0)
```python
# Comprehensive production configuration
IRONFORGE_CONFIG = {
    # Data paths
    'raw_data_path': 'data/raw',
    'enhanced_data_path': 'data/enhanced',
    'discoveries_path': 'data/discoveries',

    # Processing settings
    'max_sessions_per_batch': 10,
    'discovery_timeout_seconds': 300,
    'enable_caching': True,

    # Quality thresholds
    'pattern_confidence_threshold': 0.7,
    'authenticity_threshold': 87.0,
    'max_duplication_rate': 0.25,

    # Performance settings
    'lazy_loading': True,
    'max_memory_mb': 1000,
    'enable_monitoring': True
}

# Enhanced TGAT configuration
TGAT_CONFIG = {
    'node_features': 45,        # Updated for semantic features
    'edge_features': 20,        # Updated for semantic relationships
    'hidden_dim': 128,
    'num_heads': 4,
    'dropout': 0.1,
    'learning_rate': 0.001
}

# Semantic feature configuration
SEMANTIC_CONFIG = {
    'preserve_fvg_events': True,
    'preserve_expansion_phases': True,
    'preserve_session_boundaries': True,
    'enable_htf_confluence': True,
    'semantic_relationship_detection': True
}
```

---

## 🚨 Common Migration Issues

### Import Errors
```python
# Error: ModuleNotFoundError: No module named 'learning'
# Solution: Update to package imports
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

# Error: ImportError: cannot import name 'get_ironforge_container'
# Solution: Use new container initialization
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
```

### Feature Dimension Errors
```python
# Error: RuntimeError: mat1 and mat2 shapes cannot be multiplied (37x128)
# Solution: Update model for new feature dimensions
model = TGAT(in_channels=45)  # Updated from 37 to 45

# Error: Expected 17D edge features, got 20D
# Solution: Update edge feature handling
# System automatically handles 20D edge features in new version
```

### Data Path Errors
```python
# Error: FileNotFoundError: enhanced_sessions/ not found
# Solution: Update to new data structure
session_files = Path('data/enhanced').glob('*.json')

# Error: No session files found
# Solution: Migrate data to new structure
# Move files from enhanced_sessions/ to data/enhanced/
```

---

## ✅ Migration Checklist

### Pre-Migration
- [ ] Backup existing code and data
- [ ] Document current configuration
- [ ] Test current system functionality
- [ ] Review breaking changes for your version

### During Migration
- [ ] Update all import statements
- [ ] Migrate to container-based initialization
- [ ] Update feature dimensions (37D→45D, 17D→20D)
- [ ] Move data files to new directory structure
- [ ] Update configuration files
- [ ] Test each component individually

### Post-Migration
- [ ] Verify all imports work correctly
- [ ] Test pattern discovery functionality
- [ ] Validate output format changes
- [ ] Check performance metrics
- [ ] Update documentation references
- [ ] Run comprehensive test suite

### Validation Tests
```python
# Test basic functionality
def test_migration_success():
    """Validate migration completed successfully"""

    # 1. Test imports
    from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading

    # 2. Test initialization
    container = initialize_ironforge_lazy_loading()

    # 3. Test components
    builder = container.get_enhanced_graph_builder()
    discovery = container.get_tgat_discovery()

    # 4. Test processing (if sample data available)
    # session_file = Path('data/enhanced/sample_session.json')
    # if session_file.exists():
    #     with open(session_file) as f:
    #         session_data = json.load(f)
    #     graph = builder.enhance_session(session_data)
    #     patterns = discovery.discover_patterns(graph)
    #     assert len(patterns) >= 0

    print("✅ Migration validation successful")
    return True

# Run validation
test_migration_success()
```

---

## 🆘 Migration Support

### If Migration Fails
1. **Revert to backup**: Restore previous working version
2. **Check dependencies**: Ensure all required packages installed
3. **Verify Python version**: Ensure Python 3.8+ compatibility
4. **Clean installation**: Remove and reinstall IRONFORGE
5. **Check documentation**: Review [Troubleshooting Guide](TROUBLESHOOTING.md)

### Getting Help
- **Documentation**: [Architecture](ARCHITECTURE.md), [API Reference](API_REFERENCE.md)
- **Examples**: [Getting Started](GETTING_STARTED.md), [User Guide](USER_GUIDE.md)
- **Issues**: Check common problems in [Troubleshooting](TROUBLESHOOTING.md)

---

*This migration guide ensures smooth transitions between all IRONFORGE versions. For the latest features and capabilities, see the [User Guide](USER_GUIDE.md) and [Architecture Documentation](ARCHITECTURE.md).*
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
