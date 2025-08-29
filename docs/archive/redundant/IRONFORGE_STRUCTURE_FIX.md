# IRONFORGE Structure Fix Plan
**Fix the package structure to match architecture requirements**

## Problem Analysis

Based on `test_refactored_structure.py` results and `.claude/ironforge_architecture.md` requirements:

### FAILED Components:
1. **Missing Package Structure**: No `ironforge/` package directory exists
2. **Import Resolution**: `No module named 'ironforge'` - package not discoverable  
3. **Root Directory Clutter**: 26 files in root (expected: 7)
4. **Missing Required Directories**: `ironforge/learning`, `ironforge/analysis`, `ironforge/synthesis`, `ironforge/integration`, `data/raw`, `tests/unit`

### Architecture Requirements (from ironforge_architecture.md):
```
ironforge/
├── learning/           # Archaeological Discovery Engine
├── analysis/           # Pattern analysis components  
├── synthesis/          # Pattern Validation & Production Bridge
├── integration/        # System Integration Layer
└── utilities/          # Core utilities
```

## Fix Strategy: CREATE PROPER PACKAGE STRUCTURE

### Phase 1: Create Package Directory Structure
```bash
# Create main package
mkdir -p ironforge/learning
mkdir -p ironforge/analysis
mkdir -p ironforge/synthesis
mkdir -p ironforge/integration
mkdir -p ironforge/utilities
mkdir -p ironforge/reporting

# Create missing data directories
mkdir -p data/raw
mkdir -p tests/unit
```

### Phase 2: Create Package __init__.py Files
```python
# ironforge/__init__.py
"""
IRONFORGE Archaeological Discovery System
Package version and main exports
"""
__version__ = "1.0.0"

# ironforge/learning/__init__.py  
"""Learning components for archaeological discovery"""

# ironforge/analysis/__init__.py
"""Analysis components for pattern analysis"""

# ironforge/synthesis/__init__.py  
"""Synthesis components for pattern validation"""

# ironforge/integration/__init__.py
"""Integration layer for system coordination"""

# ironforge/utilities/__init__.py
"""Core utilities and helpers"""

# ironforge/reporting/__init__.py
"""Reporting and analysis output"""
```

### Phase 3: Move Files to Correct Package Locations

**Move to ironforge/learning/:**
- `enhanced_graph_builder.py` (if exists)
- `tgat_discovery.py` (if exists) 
- `simple_event_clustering.py` (if exists)
- `regime_segmentation.py` (if exists)

**Move to ironforge/analysis/:**
- `timeframe_lattice_mapper.py` (if exists)
- `enhanced_session_adapter.py` (if exists)
- `broad_spectrum_archaeology.py` (if exists)

**Move to ironforge/synthesis/:**
- `pattern_graduation.py` (if exists)
- `production_graduation.py` (if exists)

**Move to ironforge/integration/:**
- `ironforge_container.py` (if exists)

**Move to ironforge/utilities/:**
- `performance_monitor.py` (if exists)

**Clean Root Directory - Keep Only:**
1. `orchestrator.py` (main entry point)
2. `config.py` (configuration system) 
3. `setup.py` (package installation)
4. `requirements.txt` (dependencies)
5. `README.md` (documentation)
6. `__init__.py` (root package init)
7. `CLAUDE.md` (claude code instructions)

**Move to appropriate subdirectories:**
- Scripts → `scripts/`
- Tests → `tests/integration/` or `tests/unit/`
- Utilities → `ironforge/utilities/`
- Documentation → `docs/`

### Phase 4: Create Missing Core Files

**If core files don't exist, create minimal implementations:**

```python
# ironforge/learning/enhanced_graph_builder.py
"""Enhanced Graph Builder for 45D/20D architecture"""
class EnhancedGraphBuilder:
    def __init__(self):
        pass

# ironforge/learning/tgat_discovery.py  
"""TGAT Discovery Engine"""
class IRONFORGEDiscovery:
    def __init__(self):
        pass

# ironforge/integration/ironforge_container.py
"""Container for lazy loading"""
def get_ironforge_container():
    from iron_core.performance import IRONContainer
    return IRONContainer()

def initialize_ironforge_lazy_loading():
    return get_ironforge_container()
```

### Phase 5: Update Import Statements

**NO CHANGES NEEDED** - imports already follow correct architecture:
- `from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder`
- `from ironforge.integration.ironforge_container import get_ironforge_container`
- `from ironforge.synthesis.pattern_graduation import PatternGraduation`

### Phase 6: Validation

After implementation, these must pass:
```python
# Test package discovery
import ironforge
print(f"✅ IRONFORGE v{ironforge.__version__}")

# Test subpackage imports
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.integration.ironforge_container import get_ironforge_container

# Test installation
pip install -e .  # Must succeed

# Test structure
python3 test_refactored_structure.py  # All tests must pass
```

## Critical Requirements

### MUST FOLLOW Architecture Constraints:
1. **45D/20D Feature Architecture**: Maintain dimensional requirements
2. **87% Graduation Threshold**: All validation components must enforce
3. **No Prediction Logic**: Only discovery and validation
4. **Complete Preservation**: Never lose data through pipeline
5. **Lazy Loading**: Use iron-core container patterns
6. **Configuration System**: No hardcoded paths

### MUST AVOID Anti-Patterns:
1. **No Fallback Logic**: Either works correctly or fails explicitly
2. **No Hardcoded Paths**: Use configuration system only
3. **No Quick Fixes**: Follow architecture exactly
4. **No Bandaid Solutions**: Implement proper structure

## Success Criteria

1. **Package Import**: `import ironforge` succeeds
2. **Subpackage Access**: All `ironforge.*` imports work
3. **Installation**: `pip install -e .` succeeds 
4. **Test Pass**: `test_refactored_structure.py` passes 5/5 tests
5. **Root Cleanup**: Only 7 expected files in root
6. **Architecture Compliance**: Follows `.claude/ironforge_architecture.md` exactly

## Implementation Priority

**CRITICAL**: Create package structure first, then move files, then validate
**NO FALLBACKS**: If files don't exist, create minimal implementations
**NO QUICK FIXES**: Follow architecture document exactly
**CONSOLIDATE**: Move scattered files into proper package structure