# IRONFORGE Architecture Documentation
**Archaeological Discovery System - Definitive Architectural Guide**

*Preventing architectural drift across IRONFORGE's 2,225-file codebase*

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Architectural Constraints](#core-architectural-constraints)
3. [Component Boundaries & Responsibilities](#component-boundaries--responsibilities)
4. [Anti-Patterns & Prohibited Practices](#anti-patterns--prohibited-practices)
5. [Domain Model Definitions](#domain-model-definitions)
6. [Decision Frameworks](#decision-frameworks)
7. [IRONFORGE-Specific Terminology](#ironforge-specific-terminology)

---

## Executive Summary

IRONFORGE is a sophisticated archaeological discovery system that uncovers hidden patterns in financial market data using advanced temporal graph attention networks (TGAT) and semantic feature analysis. The system transforms raw market sessions into rich contextual archaeological discoveries with complete event preservation and session anchoring.

**Core Mission**: Archaeological discovery of market patterns (NOT prediction)  
**Architecture**: Component-based lazy loading with iron-core integration  
**Performance**: 88.7% improvement through lazy loading (3.4s vs 2+ min timeouts)  
**Features**: 45D node vectors, 20D edge vectors with semantic event preservation  

---

## Core Architectural Constraints

### 🚫 IMMUTABLE CONSTRAINTS - NEVER VIOLATE

#### 1. Discovery vs Prediction Separation
```
✅ ALLOWED: Pattern discovery, archaeological analysis, feature extraction
❌ FORBIDDEN: Price prediction, trading signals, forecasting
```
- **Discovery Purpose**: Find hidden relationships and patterns in historical data
- **No Prediction**: System must never generate trading recommendations or future price predictions
- **Clear Separation**: Discovery components (`learning/`, `synthesis/`) completely isolated from any prediction logic

#### 2. 87% Graduation Threshold
```
✅ ALLOWED: Patterns that beat 87% baseline accuracy in validation
❌ FORBIDDEN: Patterns below 87% accuracy entering production
```
- **Validation Required**: All patterns must pass `PatternGraduation` pipeline
- **Quality Gate**: `synthesis/pattern_graduation.py` enforces 87% minimum accuracy
- **Production Bridge**: Only graduated patterns become production features

#### 3. Feature Architecture Constraints
```
✅ FIXED: 45D node features (8 semantic + 37 traditional)
✅ FIXED: 20D edge features (3 semantic + 17 traditional)
❌ FORBIDDEN: Changing dimensional architecture without system-wide updates
```
- **Node Features**: `RichNodeFeature` with exactly 45 dimensions
- **Edge Features**: `RichEdgeFeature` with exactly 20 dimensions  
- **TGAT Compatibility**: Architecture designed for 45D→44D attention projection
- **Semantic Preservation**: First 8 node features reserved for semantic events

#### 4. Performance SLA
```
✅ TARGET: <5s component initialization
✅ TARGET: 88.7% performance improvement maintained
❌ FORBIDDEN: Reverting to blocking/synchronous loading
```
- **Lazy Loading**: All components use iron-core lazy loading patterns
- **Container Pattern**: Dependency injection through `IRONFORGEContainer`
- **Thread Safety**: All lazy loading must be thread-safe

#### 5. Complete Preservation
```
✅ REQUIRED: Never lose data through pipeline
✅ REQUIRED: Raw JSON preservation in graph storage
❌ FORBIDDEN: Data loss or lossy transformations
```
- **Archaeological Principle**: Preserve all historical information
- **Graph Storage**: Raw JSON stored alongside processed features
- **Session Context**: Maintain NY_AM/LONDON_PM/ASIA session metadata

---

## Component Boundaries & Responsibilities

### 📁 `/learning/` - Archaeological Discovery Engine
**Responsibility**: Self-supervised pattern discovery without prediction

**Core Files**:
- `enhanced_graph_builder.py` - JSON → 45D/20D graph conversion with semantic features
- `tgat_discovery.py` - TGAT neural network for temporal pattern discovery
- `simple_event_clustering.py` - Time-based event clustering
- `regime_segmentation.py` - Market regime detection

**Boundaries**:
- ✅ **Input**: Level 1 JSON session data
- ✅ **Output**: Discovered patterns with attention weights
- ❌ **Forbidden**: Prediction logic, trading signals, future forecasts
- ❌ **Forbidden**: Direct file I/O (use orchestrator/config)

### 📁 `/synthesis/` - Pattern Validation & Production Bridge
**Responsibility**: Validate discovered patterns and bridge to production

**Core Files**:
- `pattern_graduation.py` - 87% threshold validation pipeline
- `production_graduation.py` - Production feature export

**Boundaries**:
- ✅ **Input**: Raw discovered patterns from `/learning/`
- ✅ **Output**: Validated patterns exceeding 87% accuracy
- ❌ **Forbidden**: Pattern modification (validation only)
- ❌ **Forbidden**: Bypassing validation thresholds

### 📁 `/preservation/` - Archaeological Data Storage
**Responsibility**: Persistent storage of graphs, patterns, and models

**Structure**:
```
preservation/
├── discovered_patterns.json    # Raw pattern discoveries
├── validated_patterns.json     # Graduated patterns (87%+)
├── production_features.json    # Production-ready features
├── tgat_model.pt              # Trained TGAT model
├── embeddings/                # Node/edge embeddings
└── full_graph_store/          # Complete graph archives
```

**Boundaries**:
- ✅ **Storage**: All patterns, graphs, embeddings, models
- ✅ **Archival**: Complete historical preservation
- ❌ **Forbidden**: Pattern modification or filtering
- ❌ **Forbidden**: Data deletion (archival system)

### 📁 `/integration/` - System Integration Layer
**Responsibility**: Iron-core integration and lazy loading management

**Core Files**:
- `ironforge_container.py` - Dependency injection container
- `__init__.py` - Clean iron-core imports

**Boundaries**:
- ✅ **Container Management**: Lazy loading coordination
- ✅ **Iron-core Integration**: Performance infrastructure
- ❌ **Forbidden**: Business logic (container only)
- ❌ **Forbidden**: Direct component instantiation

### 📁 `/iron_core/` - Shared Infrastructure
**Responsibility**: Performance optimization and mathematical operations

**Key Components**:
- `performance/` - Lazy loading, container patterns
- `mathematical/` - RG optimizers, correlators, invariants
- `validation/` - Data validation utilities

**Boundaries**:
- ✅ **Infrastructure**: Performance, math, validation utilities
- ✅ **Cross-System**: Shared across IRONFORGE and IRONPULSE
- ❌ **Forbidden**: IRONFORGE-specific business logic
- ❌ **Forbidden**: Market-specific implementations

---

## Anti-Patterns & Prohibited Practices

### 🚨 CRITICAL ANTI-PATTERNS - IMMEDIATE REJECTION

#### 1. Mixing Discovery and Prediction
```python
# ❌ FORBIDDEN - Prediction in discovery system
def discover_patterns(data):
    patterns = tgat_discovery(data)
    future_prices = predict_next_candle(patterns)  # VIOLATION
    return patterns, future_prices

# ✅ CORRECT - Pure discovery
def discover_patterns(data):
    patterns = tgat_discovery(data)
    return patterns  # Discovery only
```

#### 2. Hardcoded Paths
```python
# ❌ FORBIDDEN - Hardcoded paths
def load_session(session_name):
    path = "/Users/jack/IRONFORGE/sessions/" + session_name  # VIOLATION
    return load_json(path)

# ✅ CORRECT - Configuration system
def load_session(session_name):
    config = get_config()
    path = config.get_sessions_path() / session_name
    return load_json(path)
```

#### 3. Breaking Feature Architecture
```python
# ❌ FORBIDDEN - Wrong dimensional architecture
class BadNodeFeature:
    def __init__(self):
        self.features = torch.randn(42)  # Wrong dimension

# ✅ CORRECT - Maintain 45D architecture
class RichNodeFeature:
    def __init__(self):
        self.features = torch.randn(45)  # Exactly 45D
        # First 8 are semantic events
        # Next 37 are traditional features
```

#### 4. Bypassing Pattern Graduation
```python
# ❌ FORBIDDEN - Direct production export
def export_patterns(patterns):
    production_features = convert_to_production(patterns)  # VIOLATION
    return production_features

# ✅ CORRECT - Graduation pipeline
def export_patterns(patterns):
    graduation = PatternGraduation(baseline_accuracy=0.87)
    validated = graduation.validate_patterns(patterns)
    if validated['status'] == 'GRADUATED':
        return validated['production_features']
    return None
```

#### 5. Breaking Lazy Loading
```python
# ❌ FORBIDDEN - Direct instantiation
def get_graph_builder():
    return EnhancedGraphBuilder()  # VIOLATION - blocks on load

# ✅ CORRECT - Container pattern
def get_graph_builder():
    container = get_ironforge_container()
    return container.get_enhanced_graph_builder()  # Lazy loaded
```

#### 6. Semantic Event Loss
```python
# ❌ FORBIDDEN - Losing semantic context
def process_pattern(pattern):
    generic_data = pattern['numerical_features']  # Lost context
    return generic_data

# ✅ CORRECT - Preserve semantic context
def process_pattern(pattern):
    return {
        'pattern_id': pattern['pattern_id'],
        'session_name': pattern['session_name'],
        'semantic_context': pattern['semantic_context'],
        'archaeological_significance': pattern['archaeological_significance']
    }
```

### 🔴 Bandaid Solution Patterns - Reject Immediately

#### Pattern Recognition:
- "Quick fix" without architectural consideration
- Modifying core components for edge cases
- Adding flags/switches to bypass constraints
- Mixing concerns to solve immediate problems
- Copy-pasting code instead of proper abstraction

#### Examples to Reject:
```python
# ❌ FORBIDDEN - Bandaid flag pattern
def discover_patterns(data, skip_validation=False):  # Red flag
    if skip_validation:  # VIOLATION
        return raw_patterns
    return validated_patterns

# ❌ FORBIDDEN - Mixed concern bandaid
def graph_builder_with_prediction(data):  # Red flag
    graph = build_graph(data)
    if ENABLE_PREDICTION_HACK:  # VIOLATION
        return graph, predict(graph)
    return graph
```

---

## Domain Model Definitions

### 🏛️ Core Archaeological Concepts

#### TGAT (Temporal Graph Attention Network)
**Definition**: Self-supervised neural network for discovering temporal patterns in market data  
**Purpose**: Archaeological discovery of hidden relationships across time and price  
**Architecture**: Multi-head attention with temporal encoding for 45D node features  
**Output**: Attention weights and discovered patterns (NOT predictions)

#### Session Anchoring  
**Definition**: Preservation of market session timing and characteristics  
**Sessions**: NY_AM, NY_PM, LONDON, ASIA, LUNCH, PREMARKET, MIDNIGHT  
**Context**: Session start/end times, market regime, liquidity characteristics  
**Preservation**: Maintained through entire discovery pipeline

#### Semantic Features
**Definition**: Human-readable market event preservation in feature vectors  
**Node Features (8 semantic)**:
- `fvg_redelivery_flag` - Fair Value Gap redelivery events
- `expansion_phase_flag` - Market expansion phases  
- `consolidation_flag` - Consolidation periods
- `retracement_flag` - Price retracement events
- `reversal_flag` - Market reversal points
- `liq_sweep_flag` - Liquidity sweep events
- `pd_array_interaction_flag` - Premium/Discount array interactions

**Edge Features (3 semantic)**:
- `semantic_event_link` - Relationship between semantic events
- `event_causality` - Causal strength between events
- `relationship_type` - Type of semantic relationship

#### Archaeological Intelligence
**Definition**: System for classifying pattern permanence and significance  
**Classifications**:
- **Permanent Patterns**: Structural market features that persist across sessions
- **Temporary Patterns**: Session-specific or regime-dependent features
- **Archaeological Value**: Scoring system for pattern historical significance

#### Pattern Graduation
**Definition**: Validation process ensuring patterns exceed 87% baseline accuracy  
**Process**: Historical backtesting against baseline performance  
**Threshold**: 87% minimum accuracy for production graduation  
**Output**: Graduated patterns suitable for production use

### 🔬 Technical Architecture Concepts

#### Lazy Loading Architecture
**Definition**: Component initialization deferred until first access  
**Implementation**: Iron-core container pattern with thread-safe singletons  
**Benefit**: 88.7% performance improvement (3.4s vs 2+ min)  
**Pattern**: `@LazyComponent` decorator with dependency injection

#### Rich Feature Vectors
**Definition**: 45D node and 20D edge feature vectors with semantic preservation  
**Structure**: Semantic events (8D) + traditional features (37D) for nodes  
**Processing**: TGAT attention with 45D→44D projection for multi-head processing  
**Preservation**: Raw JSON context maintained alongside processed features

#### Complete Preservation Principle
**Definition**: Never lose data through the archaeological pipeline  
**Implementation**: Raw JSON storage alongside all processed derivatives  
**Graph Storage**: Complete session context preserved in graph archives  
**Session Context**: Timing, metadata, and semantic events fully maintained

---

## Decision Frameworks

### 🎯 Multi-Component Change Framework

#### When to Use Orchestrator vs Direct Access

**Use Orchestrator When**:
- Changes involve >2 components
- Cross-boundary data flow modifications
- Performance monitoring required
- Configuration changes needed
- Session-level processing workflows

**Use Direct Component Access When**:
- Single component modifications
- Unit testing individual components
- Performance optimization within component
- Component-specific debugging

#### Feature Vector Modification Guidelines

**45D Node Feature Changes**:
1. **Semantic Events (0-7)**: Requires semantic retrofit planning
2. **Traditional Features (8-44)**: Standard feature engineering process
3. **New Features**: Must maintain 45D total, update TGAT projection
4. **Feature Order**: Never reorder - breaks trained models

**20D Edge Feature Changes**:
1. **Semantic Relationships (0-2)**: Requires relationship model updates
2. **Traditional Edges (3-19)**: Standard edge engineering process
3. **New Edges**: Must maintain 20D total, update edge processing

#### Performance Modification Decision Tree

```
Performance Issue Identified
├─ Component Initialization >5s?
│  ├─ Yes → Apply lazy loading pattern
│  └─ No → Continue
├─ Memory Usage Excessive?
│  ├─ Yes → Implement component caching
│  └─ No → Continue
├─ Cross-Component Dependencies?
│  ├─ Yes → Use container pattern
│  └─ No → Direct optimization
└─ System-Wide Impact?
   ├─ Yes → Iron-core integration
   └─ No → Component-specific optimization
```

#### Integration Change Guidelines

**Iron-Core Dependency Changes**:
1. **New Dependencies**: Add to `iron_core/` shared infrastructure
2. **Performance Components**: Use `iron_core.performance` patterns
3. **Mathematical Operations**: Leverage `iron_core.mathematical`
4. **Validation Logic**: Extend `iron_core.validation`

**Container Pattern Changes**:
1. **New Components**: Register in `IRONFORGEContainer`
2. **Lazy Loading**: Apply `@LazyComponent` decorator
3. **Dependencies**: Use dependency injection, not direct imports
4. **Thread Safety**: Ensure all container operations are thread-safe

### 🔄 Architectural Evolution Framework

#### Adding New Discovery Methods
1. **Component Placement**: Must go in `/learning/` directory
2. **Interface Compliance**: Follow `tgat_discovery.py` patterns
3. **Feature Compatibility**: Support 45D/20D architecture
4. **Graduation Path**: Integrate with `PatternGraduation` pipeline

#### New Validation Methods
1. **Component Placement**: Must go in `/synthesis/` directory
2. **Threshold Enforcement**: Maintain 87% minimum accuracy
3. **Historical Testing**: Backtest against archived patterns
4. **Production Bridge**: Export only graduated patterns

#### Storage Schema Evolution
1. **Backwards Compatibility**: Never break existing graph storage
2. **Migration Scripts**: Provide in `/data_migration/` directory
3. **Preservation Principle**: Maintain complete historical records
4. **Schema Versioning**: Version all storage format changes

---

## IRONFORGE-Specific Terminology

### 🏛️ Archaeological Discovery Terms

**Archaeological Discovery**: The process of finding hidden market patterns through self-supervised learning, analogous to archaeological excavation of historical artifacts

**Pattern Archaeology**: Systematic excavation of market structure patterns from historical session data

**Session Character**: The unique behavioral signature of market sessions (NY_AM aggressive, LONDON_PM methodical, ASIA overnight)

**Event Preservation**: Maintaining semantic context of market events (FVG redelivery, liquidity sweeps) through the discovery pipeline

**Temporal Resonance**: Discovery of patterns that repeat across similar market conditions and timeframes

**Market Regime Archaeology**: Classification of market states based on discovered structural patterns

### 🔬 Technical Architecture Terms

**Semantic Retrofit**: Enhancement process that transforms generic numerical patterns into rich contextual discoveries with human-readable semantic information

**Pattern Intelligence**: Advanced analysis layer that provides contextual understanding of discovered patterns

**Session Anchoring**: System for preserving market session timing, characteristics, and context throughout processing

**Feature Vector Architecture**: Fixed 45D node / 20D edge dimensional structure with semantic event preservation

**Lazy Archaeological Loading**: Performance optimization pattern where discovery components initialize only on first access

**Pattern Graduation Pipeline**: Validation system ensuring discovered patterns exceed 87% baseline accuracy before production use

### 📊 Pattern Classification Terms

**Permanent Patterns**: Structural market features that persist across different sessions and market regimes

**Temporary Patterns**: Session-specific or regime-dependent patterns with limited temporal scope

**Archaeological Significance**: Scoring system measuring the historical importance and persistence of discovered patterns

**Pattern Confluence**: Discovery of multiple independent patterns reinforcing the same market structure

**Cross-Session Synchronization**: Patterns that maintain consistency across different market sessions

**Liquidity Archaeology**: Discovery of hidden liquidity patterns through sweep detection and flow analysis

### 🎯 Discovery Quality Terms

**Pattern Authenticity**: Measure of how genuine a discovered pattern is versus statistical noise

**Discovery Confidence**: Attention weight and validation score indicating pattern reliability

**Baseline Beating**: Patterns that exceed the 87% accuracy threshold in historical validation

**Archaeological Validation**: Comprehensive testing of discovered patterns against historical market behavior

**Production Graduation**: Final step where validated patterns become usable production features

**Discovery Session**: Complete archaeological analysis workflow from raw JSON to graduated patterns

---

## Implementation Guidelines

### 🚀 Getting Started with IRONFORGE Architecture

#### For New Components
1. **Choose Correct Directory**: `/learning/` for discovery, `/synthesis/` for validation
2. **Follow Naming Conventions**: `enhanced_*` for upgraded components
3. **Use Configuration System**: Never hardcode paths, use `get_config()`
4. **Apply Lazy Loading**: Register in container, use `@LazyComponent`
5. **Preserve Dimensions**: Maintain 45D/20D architecture

#### For Modifications
1. **Read This Document First**: Understand constraints and anti-patterns
2. **Check Component Boundaries**: Ensure changes fit architectural responsibilities
3. **Validate Against 87% Threshold**: All patterns must graduate
4. **Maintain Performance SLA**: Keep <5s initialization times
5. **Preserve Semantic Events**: Never lose contextual information

#### For Integration
1. **Use Iron-Core Patterns**: Leverage shared infrastructure
2. **Container-Based Dependencies**: No direct imports between major components
3. **Thread-Safe Operations**: All lazy loading must be thread-safe
4. **Complete Preservation**: Maintain archaeological data integrity

---

*This document serves as the definitive architectural guide for IRONFORGE. All development decisions should reference and comply with these specifications to prevent architectural drift.*

**Last Updated**: August 15, 2025  
**Version**: 1.0  
**Status**: Architectural Foundation Complete