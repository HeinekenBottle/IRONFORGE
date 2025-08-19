[Note] Source of truth for versions is GitHub Releases (tags). This changelog reflects historical notes and may not match package version (current tag: v0.7.0).
# IRONFORGE Changelog
**Version History and Breaking Changes**

---

## [1.0.2-rc1] - 2025-08-19 - Oracle Training CLI (Parquet-only, strict coverage)
### 🎯 Oracle Temporal Non-Locality System
- **NEW**: Complete Oracle training CLI with Parquet-native processing
- **NEW**: Strict mode with comprehensive audit validation and zero silent skips
- **NEW**: Reproducibility manifest with git SHA and environment tracking
- **NEW**: 16-column sidecar schema for oracle predictions
- **NEW**: Enhanced feature assembly adapters for Discovery/Evaluator integration

### Phase 1.1 Coverage Repair (Completed)
- **FIXED**: Discovery KeyError in EnhancedGraphBuilder feature handling
- **FIXED**: Evaluator correction to use Discovery's pooled 44D embeddings
- **ADDED**: Runtime validation for Discovery + Oracle integration
- **ENHANCED**: Feature exception logging with exact missing keys and available columns

### Phase 2 Clean & Harden
- **PURGED**: Mock/demo scripts and temporary probe directories
- **UPDATED**: .gitignore for Oracle artifacts (*.pt, *.pkl, models/**)
- **LOCKED**: 45/51/20 architecture contracts remain green
- **POLISHED**: CLI help text and M5⇆5 timeframe normalization
- **DOCUMENTED**: Complete ORACLE.md with schemas, metrics, and troubleshooting

---

## [2.0.0] - 2025-08-16 - Documentation Modernization
### 🎯 Major Documentation Overhaul
- **BREAKING**: Complete documentation restructure with modern naming conventions
- **NEW**: Comprehensive architecture documentation
- **NEW**: Production deployment guide
- **NEW**: Complete API reference
- **IMPROVED**: User guide with daily workflows
- **REMOVED**: Deprecated phase-based documentation files

### Documentation Structure Changes
- Added `ARCHITECTURE.md` - Complete system architecture
- Added `GETTING_STARTED.md` - Quick start guide
- Added `API_REFERENCE.md` - Complete API documentation
- Added `USER_GUIDE.md` - Comprehensive user workflows
- Added `DEPLOYMENT_GUIDE.md` - Production deployment
- Added `TGAT_ARCHITECTURE.md` - Neural network details
- Added `SEMANTIC_FEATURES.md` - Semantic system documentation
- Updated `MIGRATION_GUIDE.md` - Current migration paths
- Consolidated legacy documentation into organized structure

---

## [1.5.0] - 2025-08-14 - Semantic Feature Retrofit Complete
### ✅ Mission Accomplished: Semantic Feature Integration
- **BREAKTHROUGH**: Semantic feature retrofit transforms generic patterns into rich contextual discoveries
- **FEATURE**: 45D node features (37D base + 8D semantic events)
- **FEATURE**: 20D edge features (17D base + 3D semantic relationships)
- **FEATURE**: Complete semantic event preservation through pipeline
- **FEATURE**: Session anchoring with timing and characteristics

### Semantic Features Added
- FVG redelivery event detection and preservation
- Expansion phase event identification
- Consolidation pattern recognition
- PD array event detection
- Liquidity sweep event identification
- Session boundary event markers
- HTF confluence event detection
- Structural break event recognition

### Pattern Output Enhancement
- **BEFORE**: Generic numerical patterns with minimal context
- **AFTER**: Rich contextual archaeological discoveries with human-readable descriptions
- **IMPROVEMENT**: 92.3/100 authenticity score (vs 78.5/100 without semantics)
- **IMPROVEMENT**: 100% patterns have meaningful semantic context

### Performance Maintained
- Processing time: 3.153s (maintained <4.7s SLA)
- Memory usage: <100MB total system footprint
- Quality: >87% authenticity threshold maintained
- Duplication rate: <25% (vs 96.8% contaminated baseline)

---

## [1.4.0] - 2025-08-13 - Broad Spectrum Archaeology System
### 🏛️ Archaeological Discovery Engine Complete
- **FEATURE**: Multi-timeframe archaeological pattern discovery (1m to monthly)
- **FEATURE**: Session phase analysis (opening, mid-session, closing)
- **FEATURE**: Event classification system for market phenomena
- **FEATURE**: HTF confluence detection and cross-session resonance
- **FEATURE**: 560-pattern IRONFORGE historical archive integration

### Analysis Capabilities
- Comprehensive event mining across all timeframes
- Temporal clustering and structural linking
- Predictive cascade detection
- Energy release forecasting
- Interactive visualization suite

### Production Ready Features
- Scalable, efficient, and modular design
- Real-time exploration and analysis tools
- Complete multi-timeframe coverage
- Advanced pattern recognition algorithms

---

## [1.3.0] - 2025-08-12 - TGAT Discovery Engine Restoration
### 🧠 Neural Network Architecture Complete
- **FEATURE**: 4-head temporal attention architecture
- **FEATURE**: Self-supervised learning (no labels required)
- **FEATURE**: Archaeological focus (no prediction logic)
- **FEATURE**: Complete pattern discovery capabilities

### TGAT Specifications
- **Architecture**: Multi-head attention with temporal encoding
- **Input**: 37D node features, 17D edge features (pre-semantic)
- **Output**: Discovered patterns with attention weights
- **Performance**: <3s per session processing
- **Quality**: 87%+ authenticity threshold

### Discovery Capabilities
- Distant session structure correlations
- Multi-timeframe cascade patterns
- Liquidity sweep archaeological patterns
- Price action memory effects
- Temporal pattern discovery

---

## [1.2.0] - 2025-08-11 - Enhanced Graph Builder Implementation
### 📊 Graph Construction System
- **FEATURE**: JSON to PyTorch Geometric graph conversion
- **FEATURE**: Rich feature vector construction (37D nodes, 17D edges)
- **FEATURE**: Price relativity transformations
- **FEATURE**: Temporal cycle encoding
- **FEATURE**: Session context preservation

### Graph Features
- Complete data preservation from Level 1 JSON
- Temporal relationship encoding
- Price movement correlations
- Structural connection mapping
- Session timing preservation

---

## [1.1.0] - 2025-08-10 - Iron-Core Integration & Lazy Loading
### ⚡ Performance Optimization Breakthrough
- **BREAKTHROUGH**: 88.7% performance improvement through lazy loading
- **FEATURE**: Container-based dependency injection
- **FEATURE**: Iron-core mathematical infrastructure integration
- **FEATURE**: Lazy loading architecture

### Performance Improvements
- **BEFORE**: 2+ minute timeouts, frequent failures
- **AFTER**: 3.4s initialization, reliable operation
- **IMPROVEMENT**: Memory efficiency through on-demand loading
- **IMPROVEMENT**: Clean separation of concerns

### Iron-Core Integration
- Shared mathematical operations (RG optimizers, correlators)
- Performance infrastructure (lazy loading patterns)
- Validation utilities (data validation frameworks)
- Cross-system architectural patterns

---

## [1.0.0] - 2025-08-09 - Initial IRONFORGE Release
### 🏛️ Archaeological Discovery System Launch
- **FEATURE**: Basic pattern discovery system
- **FEATURE**: Level 1 JSON session processing
- **FEATURE**: Pattern validation and graduation
- **FEATURE**: Archaeological discovery focus (no predictions)

### Core Components
- Enhanced graph builder for data transformation
- Basic TGAT implementation for pattern discovery
- Pattern graduation system with quality thresholds
- Complete data preservation architecture

### Quality Standards
- 87% authenticity threshold for production patterns
- <25% duplication rate requirement
- Complete data preservation guarantee
- Archaeological discovery mission (no prediction logic)

---

## Breaking Changes Summary

### Version 2.0.0 (Documentation)
- **BREAKING**: Documentation file structure completely reorganized
- **BREAKING**: Legacy phase-based documentation removed
- **MIGRATION**: Use new documentation structure and naming conventions
- **IMPACT**: No code changes required, documentation access only

### Version 1.5.0 (Semantic Features)
- **BREAKING**: Node features expanded from 37D to 45D
- **BREAKING**: Edge features expanded from 17D to 20D
- **BREAKING**: Pattern output format changed to include semantic context
- **MIGRATION**: Update any code expecting old feature dimensions
- **MIGRATION**: Update pattern processing to handle new semantic fields

### Version 1.4.0 (Broad Spectrum)
- **BREAKING**: Analysis output format standardized
- **BREAKING**: Session processing API updated
- **MIGRATION**: Update analysis result processing code
- **MIGRATION**: Use new session analysis methods

### Version 1.1.0 (Lazy Loading)
- **BREAKING**: All imports now require `ironforge.` package prefix
- **BREAKING**: Direct component instantiation deprecated
- **BREAKING**: Container-based initialization required
- **MIGRATION**: Update all import statements to use package structure
- **MIGRATION**: Use `initialize_ironforge_lazy_loading()` for system initialization

---

## Migration Paths

### From 1.x to 2.0.0
```python
# No code changes required - documentation only
# Update documentation references to new structure
```

### From 1.4.x to 1.5.0
```python
# OLD: Expecting 37D node features
model = TGAT(in_channels=37)

# NEW: Updated for 45D node features
model = TGAT(in_channels=45)

# OLD: Basic pattern output
pattern = {'type': 'range_position', 'confidence': 0.73}

# NEW: Rich semantic pattern output
pattern = {
    'pattern_id': 'NY_session_RPC_00',
    'semantic_context': {...},
    'archaeological_significance': {...},
    'confidence': 0.87
}
```

### From 1.0.x to 1.1.0
```python
# OLD: Direct imports (will fail)
from learning.enhanced_graph_builder import EnhancedGraphBuilder

# NEW: Package-based imports
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

# OLD: Direct instantiation
builder = EnhancedGraphBuilder()

# NEW: Container-based initialization
container = initialize_ironforge_lazy_loading()
builder = container.get_enhanced_graph_builder()
```

---

## Deprecation Notices

### Deprecated in 2.0.0
- Legacy phase-based documentation files (removed)
- Scattered technical summaries (consolidated)
- Individual feature implementation docs (integrated)

### Deprecated in 1.5.0
- Generic pattern output format (replaced with semantic context)
- Basic feature vectors without semantic events (expanded)

### Deprecated in 1.1.0
- Direct component instantiation (use container)
- Flat import structure (use package imports)
- Manual dependency management (use lazy loading)

---

## Upcoming Features

### Version 2.1.0 (Planned)
- Real-time pattern monitoring dashboard
- Advanced pattern intelligence analytics
- Cross-market archaeological discovery
- Enhanced visualization suite

### Version 2.2.0 (Planned)
- GPU acceleration for large-scale processing
- Distributed processing capabilities
- Advanced caching and persistence
- Production monitoring and alerting

---

*For detailed migration instructions, see the [Migration Guide](MIGRATION_GUIDE.md). For current system architecture, see [Architecture Documentation](ARCHITECTURE.md).*
