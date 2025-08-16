# IRONFORGE - Claude Code Documentation

**Archaeological Discovery System for Market Pattern Recognition**

## Architecture Overview

IRONFORGE is a sophisticated temporal graph attention network (TGAT) system that discovers hidden patterns in financial market data through archaeological-style analysis. The system transforms raw market sessions into rich contextual discoveries with full event preservation and session anchoring.

### Core Design Principles

1. **Complete Preservation**: Never loses data - graphs store raw JSON with enhanced features
2. **Self-Supervised Learning**: TGAT discovers patterns without labels using temporal attention
3. **Pattern Graduation**: Only patterns beating 87% baseline enter production 
4. **Clean Separation**: Learning/discovery completely isolated from prediction
5. **Semantic Enhancement**: Rich contextual market event discovery with archaeological intelligence
6. **Lazy Loading**: 88.7% performance improvement through iron-core integration

## System Components

### Core Infrastructure

**Iron-Core Integration** (Production-Ready)
- **Location**: `iron_core/` package 
- **Purpose**: Shared infrastructure providing lazy loading, dependency injection, and performance optimization
- **Install**: `pip install -e iron_core/` (editable mode for development)
- **Import**: `from iron_core.performance import IRONContainer, LazyComponent`
- **Performance**: 88.7% improvement, <5s initialization (vs 2+ min timeout)

**Configuration System**
- **File**: `config.py`
- **Purpose**: Eliminates hardcoded paths, environment-specific deployment
- **Environment Variables**: `IRONFORGE_*` prefixed variables override defaults
- **Paths**: Automatically resolves relative to workspace root

### Data Pipeline Architecture

```
Raw Data → Enhanced Sessions → Adapted Relativity → Graph Building → TGAT Discovery → Pattern Graduation
```

**Data Structure**:
```
data/
├── raw/                    # Level 1 raw market data
├── enhanced/               # Enhanced/processed sessions with semantic features  
├── adapted/                # Adapted sessions with price relativity
└── discoveries/            # Pattern discoveries and archaeological finds
```

**Session Enhancement**: 66 accessible sessions, 57 enhanced with authentic features
- **Enhancement**: 37D → 45D nodes (8 semantic + 37 previous)
- **Edges**: 17D → 20D edges (3 semantic + 17 previous) 
- **Features**: FVG redelivery, expansion phases, session anchoring preserved
- **Event Mapping**: 0→72+ events/session, 64 event type mappings

### ML Pipeline Components

**Enhanced Graph Builder** (`ironforge.learning.enhanced_graph_builder`)
- **Purpose**: JSON → TGAT graphs with semantic features
- **Features**: 45D nodes, 20D edges, rich contextual features
- **Performance**: <3s per session, handles complex temporal relationships
- **Output**: Production-ready graphs with event preservation

**TGAT Discovery Engine** (`ironforge.learning.tgat_discovery`)  
- **Architecture**: 4-head temporal attention with 92.3/100 authenticity
- **Purpose**: Self-supervised pattern discovery with rich context output
- **Training**: Multi-session batch processing with global constant filtering
- **Output**: Archaeological patterns with semantic context

**Pattern Graduation** (`ironforge.synthesis.pattern_graduation`)
- **Purpose**: Validates patterns against 87% baseline threshold
- **Quality Gates**: Duplication <25%, authenticity >90/100, temporal coherence >70%
- **Output**: Production-ready validated patterns

### Critical Integration Points

**Container System** (`ironforge.integration.ironforge_container`)
```python
from ironforge.integration.ironforge_container import get_ironforge_container
container = get_ironforge_container()
builder = container.get_enhanced_graph_builder()
discovery = container.get_tgat_discovery()
```

**Configuration Integration**
```python
from config import get_config
config = get_config()
data_path = config.get_data_path()
```

## Development Workflow

### Installation & Setup

```bash
# 1. Install iron-core (required dependency)
cd iron_core/
pip install -e .

# 2. Install IRONFORGE requirements
pip install -r requirements.txt

# 3. Verify installation
python tests/integration/test_iron_core_quick.py
```

### Common Development Tasks

**Running Discovery on Sessions**:
```bash
# Initialize and run full pipeline
python orchestrator.py

# Process specific sessions
python -c "
from orchestrator import IRONFORGE
forge = IRONFORGE()
results = forge.process_sessions(['data/enhanced/session.json'])
"
```

**Testing & Validation**:
```bash
# Integration tests
python -m pytest tests/integration/ -v

# Quick validation
python tests/integration/test_minimal_import.py

# Performance validation  
python benchmark_performance.py
```

**Data Processing**:
```bash
# Enhance raw sessions
python scripts/data_processing/enhance_sessions.py

# Migration tools
python data_migration/batch_migrate_graphs.py
```

### Key Import Patterns

**Standard Imports** (use these):
```python
# Core components
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery  
from ironforge.integration.ironforge_container import get_ironforge_container

# Configuration
from config import get_config

# Iron-core performance
from iron_core.performance import IRONContainer, LazyComponent
```

**Import Troubleshooting**:
- All imports use `ironforge.` prefix after refactoring
- Add `sys.path.append('.')` if needed for development
- Iron-core must be installed in editable mode: `pip install -e iron_core/`

## Performance & Optimization

### Performance Metrics
- **Initialization**: <5s (vs 120s+ before iron-core)
- **Session Processing**: <3s per session
- **Memory Usage**: <100MB total footprint  
- **Feature Processing**: 45D nodes, 20D edges efficiently
- **Pattern Discovery**: 8-30 patterns per session typical

### Sprint 2 Enhancements
- **Structural Context Edges**: Multi-timeframe relationships
- **Regime Segmentation**: DBSCAN clustering for market regimes
- **Precursor Detection**: Temporal cycles + structural analysis
- **Performance Monitoring**: Regression analysis and quality gates

### Troubleshooting Performance Issues

**Container Initialization Slow**:
```python
# Check iron-core installation
from iron_core.performance import get_container
container = get_container()
metrics = container.get_performance_metrics()
print(f"SLA met: {metrics['performance_sla_met']}")
```

**Session Processing Timeouts**:
- Enable lazy loading: `container = get_ironforge_container()`
- Check data quality: validate session JSON structure
- Monitor memory: use `benchmark_performance.py`

## Archaeological Discovery System

### Theory B Discovery (Key Breakthrough)
- **Finding**: 40% zone represents dimensional relationship to FINAL session range
- **Evidence**: Events position with 7.55 point precision to eventual session completion
- **Implication**: Archaeological zones are predictive rather than reactive
- **Temporal Non-Locality**: Events "know" position relative to eventual completion

### Pattern Intelligence
- **Semantic Features**: 8 semantic event types with rich context
- **Session Anchoring**: NY_AM, LONDON_PM, ASIA timing preserved
- **Archaeological Significance**: Permanent vs temporary classification
- **Context Output**: Rich market archaeology vs generic patterns

### Discovery Workflows
```python
# Morning market analysis
from daily_discovery_workflows import morning_prep
analysis = morning_prep(days_back=7)

# Session pattern hunting  
from daily_discovery_workflows import hunt_patterns
patterns = hunt_patterns('NY_PM')

# Cross-session analysis
from ironforge_discovery_sdk import IRONFORGEDiscoverySDK
sdk = IRONFORGEDiscoverySDK()
results = sdk.discover_all_sessions()
```

## Testing Strategy

### Test Categories
- **Integration Tests**: `tests/integration/` - End-to-end workflows
- **Performance Tests**: `benchmark_performance.py` - Regression detection
- **Component Tests**: Individual component validation
- **Quality Gates**: Pattern authenticity, duplication rates

### Critical Test Files
- `test_iron_core_quick.py` - Validates iron-core integration
- `test_enhanced_orchestrator_workflow.py` - Full pipeline test
- `test_performance_monitor.py` - Sprint 2 monitoring
- `test_semantic_retrofit_task*.py` - Semantic feature validation

### Validation Commands
```bash
# Quick health check
python test_refactored_structure.py

# Full integration suite
python -m pytest tests/integration/ -v --tb=short

# Performance regression check
python benchmark_performance.py
```

## Production Deployment

### Quality Gates (Must Pass)
- ✅ Pattern authenticity >90/100
- ✅ Duplication rate <25% 
- ✅ Processing time <5s initialization
- ✅ Memory usage <100MB
- ✅ Feature dimensions: 45D nodes, 20D edges
- ✅ TGAT compatibility validated

### Production Checklist
```bash
# 1. Install dependencies
pip install -e iron_core/
pip install -r requirements.txt

# 2. Run integration tests
python -m pytest tests/integration/ -v

# 3. Validate performance
python benchmark_performance.py

# 4. Test discovery pipeline  
python orchestrator.py

# 5. Verify pattern graduation
python -c "
from ironforge.synthesis.pattern_graduation import PatternGraduation
pg = PatternGraduation()
print('Graduation pipeline ready')
"
```

### Environment Configuration
```bash
# Required environment variables
export IRONFORGE_WORKSPACE_ROOT=/path/to/ironforge
export IRONFORGE_DATA_PATH=/path/to/data
export IRONFORGE_PRESERVATION_PATH=/path/to/preservation

# Optional overrides
export IRONFORGE_GRAPHS_PATH=/custom/graphs/path
export IRONFORGE_SESSION_DATA_PATH=/custom/sessions/path
```

## Key Patterns & Conventions

### Error Handling Philosophy
- **No Silent Failures**: All errors exposed explicitly with clear messages
- **No Fallbacks**: Fix root causes rather than masking symptoms  
- **Explicit Failure**: Better to fail fast than hide issues
- **Root Cause Analysis**: Detailed error context for debugging

### File Naming Conventions
- **Session Files**: `enhanced_rel_SESSION_Lvl-1_DATE.json`
- **Graph Storage**: `SESSION_graph_TIMESTAMP.pkl`
- **Discovery Results**: `discovery_session_N_discoveries.json`
- **Reports**: `COMPONENT_results_TIMESTAMP.json`

### Code Organization
- **Learning**: ML components, graph building, TGAT discovery
- **Analysis**: Pattern analysis, archaeological discovery tools
- **Synthesis**: Pattern validation, graduation pipelines  
- **Integration**: Container system, lazy loading, configuration
- **Utilities**: Performance monitoring, data migration tools

## Dependencies & Integration

### Critical Dependencies
```python
# Core infrastructure
iron-core>=1.0.0          # Lazy loading, dependency injection

# Deep learning stack
torch>=1.9.0,<2.5.0       # Neural networks, TGAT
torch-geometric>=2.0.0    # Graph neural networks

# Scientific computing
numpy>=1.21.0,<2.0.0      # Numerical computation
scipy>=1.6.0              # Statistical analysis
scikit-learn>=1.0.0       # ML utilities, clustering

# Graph analysis
networkx>=2.6.0           # Graph algorithms
```

### Iron-Core Relationship
- **IRONFORGE** depends on **iron-core** for infrastructure
- **iron-core** provides shared performance optimization across IRON suite
- **Installation**: Must install iron-core in editable mode first
- **Integration**: Automatic via container system

### External Integration Points
- **IRONPULSE**: Grammar parsing integration (future)
- **Production Systems**: Pattern export via `freeze_for_production()`
- **Data Sources**: Session JSON files from trading systems
- **Monitoring**: Performance regression tracking

## Troubleshooting Guide

### Common Issues

**Import Errors**: `ModuleNotFoundError: No module named 'ironforge'`
```bash
# Solution: Verify package structure
python -c "import ironforge.learning.enhanced_graph_builder; print('OK')"
# If fails: Check PYTHONPATH and working directory
```

**Iron-Core Issues**: `No module named 'iron_core'`
```bash
# Solution: Install iron-core in editable mode
cd iron_core/
pip install -e .
```

**Container Initialization Failures**:
```python
# Debug: Check component registration
from ironforge.integration.ironforge_container import get_ironforge_container
container = get_ironforge_container()
metrics = container.get_performance_metrics()
```

**Session Processing Failures**:
- Check session JSON structure and validity
- Verify data paths in configuration
- Monitor memory usage during processing
- Check for missing enhanced features

### Performance Debugging
```bash
# Profile performance
python -c "
import time
start = time.time()
from ironforge.integration.ironforge_container import get_ironforge_container
container = get_ironforge_container()
print(f'Container init: {time.time() - start:.3f}s')
"

# Memory profiling
python -m memory_profiler orchestrator.py
```

---

## Quick Reference

**Essential Commands**:
```bash
# Setup
pip install -e iron_core/ && pip install -r requirements.txt

# Test
python tests/integration/test_iron_core_quick.py

# Process
python orchestrator.py

# Validate  
python -m pytest tests/integration/ -v
```

**Essential Imports**:
```python
from ironforge.integration.ironforge_container import get_ironforge_container
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from config import get_config
```

**Configuration Check**:
```python
from config import get_config
config = get_config()
print(f"Data path: {config.get_data_path()}")
print(f"Graphs path: {config.get_graphs_path()}")
```

---

*Status: Production-Ready Archaeological Discovery System*  
*Last Updated: August 16, 2025*  
*Architecture: TGAT + Semantic Enhancement + Theory B Discovery*