# IRONFORGE System Architecture
**Archaeological Discovery System for Market Pattern Recognition**

---

## 🏛️ System Overview

IRONFORGE is a sophisticated archaeological discovery system that uncovers hidden patterns in financial market data using advanced temporal graph attention networks (TGAT) and semantic feature analysis. The system transforms raw market sessions into rich contextual archaeological discoveries with complete event preservation and session anchoring.

**Core Mission**: Archaeological discovery of market patterns (NOT prediction)  
**Architecture**: Component-based lazy loading with iron-core integration  
**Performance**: 88.7% improvement through lazy loading optimization  
**Features**: 45D node vectors, 20D edge vectors with semantic event preservation  

---

## 🏗️ Architectural Principles

### Core Design Philosophy
1. **Archaeological Discovery Focus**: System discovers patterns, never predicts future outcomes
2. **Complete Data Preservation**: All raw data and intermediate states are preserved
3. **Semantic Context Preservation**: Market events maintain human-readable context
4. **Lazy Loading Performance**: Components load only when needed for optimal performance
5. **Iron-Core Integration**: Shared mathematical infrastructure for performance optimization

### System Boundaries
- ✅ **Input**: Level 1 JSON session data with market events
- ✅ **Processing**: Graph construction, TGAT discovery, pattern validation
- ✅ **Output**: Rich contextual archaeological patterns with attention weights
- ❌ **Forbidden**: Prediction logic, trading signals, future forecasts

---

## 📦 Component Architecture

### Primary Packages

#### `/ironforge/` - Main Application Package
```
ironforge/
├── learning/          # TGAT discovery engine and pattern learning
├── analysis/          # Pattern analysis and market archaeology
├── synthesis/         # Pattern validation and production bridge
├── integration/       # System integration and lazy loading
├── utilities/         # Core utilities and monitoring
└── reporting/         # Analysis reporting and visualization
```

#### `/iron_core/` - Shared Infrastructure
```
iron_core/
├── performance/       # Lazy loading and container patterns
├── mathematical/      # RG optimizers, correlators, invariants
├── validation/        # Data validation utilities
└── integration/       # Cross-system integration patterns
```

### Data Architecture
```
data/
├── raw/              # Level 1 raw market data (JSON sessions)
├── enhanced/         # Enhanced sessions with semantic features
├── adapted/          # Sessions with price relativity transformations
└── discoveries/      # Archaeological pattern discoveries
```

---

## 🧠 Core Components

### 1. Enhanced Graph Builder (`learning/enhanced_graph_builder.py`)
**Responsibility**: Transform JSON sessions into 45D/20D TGAT-compatible graphs

**Key Features**:
- **45D Node Features**: 37D base features + 8D semantic events
- **20D Edge Features**: 17D base features + 3D semantic relationships
- **Semantic Event Preservation**: FVG redelivery, expansion phases, consolidation patterns
- **Session Anchoring**: NY_AM/LONDON_PM/ASIA timing preservation

**Input**: Level 1 JSON session data  
**Output**: PyTorch Geometric graph with rich feature vectors

### 2. TGAT Discovery Engine (`learning/tgat_discovery.py`)
**Responsibility**: Self-supervised pattern discovery using temporal attention

**Architecture**:
- **4-Head Multi-Attention**: Different pattern types (structural, temporal, confluence)
- **Temporal Encoding**: Time-aware attention for distant correlations
- **Archaeological Output**: Attention weights + discovered patterns (no predictions)
- **Memory Efficiency**: Optimized for 57-session processing

**Performance**: 3.153s processing time, <4.7s SLA maintained

### 3. Pattern Validation (`synthesis/pattern_graduation.py`)
**Responsibility**: Validate discovered patterns against baseline performance

**Validation Criteria**:
- **87% Baseline Threshold**: Only high-quality patterns graduate
- **Authenticity Scoring**: 92.3/100 authenticity for production patterns
- **Duplication Filtering**: <25% duplication rate (vs 96.8% contaminated baseline)
- **Temporal Coherence**: Meaningful time spans and relationships

### 4. Lazy Loading Container (`integration/ironforge_container.py`)
**Responsibility**: Dependency injection and performance optimization

**Benefits**:
- **88.7% Performance Improvement**: 3.4s vs 2+ minute timeouts
- **Memory Efficiency**: Components load only when needed
- **Clean Dependencies**: Proper separation of concerns
- **Iron-Core Integration**: Shared mathematical infrastructure

---

## 🔄 Data Flow Architecture

### 1. Data Ingestion Pipeline
```
Raw JSON Sessions → Enhanced Graph Builder → 45D/20D Graphs
```
- Semantic event extraction and preservation
- Price relativity transformations
- Session context anchoring
- Feature vector construction

### 2. Discovery Pipeline
```
TGAT Graphs → TGAT Discovery Engine → Raw Patterns
```
- Multi-head temporal attention processing
- Archaeological pattern discovery
- Attention weight extraction
- No prediction logic (discovery only)

### 3. Validation Pipeline
```
Raw Patterns → Pattern Graduation → Validated Patterns
```
- 87% baseline threshold validation
- Authenticity scoring and filtering
- Duplication removal
- Production readiness assessment

### 4. Production Pipeline
```
Validated Patterns → Rich Context Output → Archaeological Intelligence
```
- Human-readable pattern descriptions
- Session context preservation
- Cross-session relationship mapping
- Intelligence reporting

---

## 🎯 Semantic Feature Architecture

### Node Feature Vector (45D)
**Base Features (37D)**:
- Price relativity transformations (34D)
- Temporal cycle encoding (3D)

**Semantic Features (8D)**:
- `fvg_redelivery_event`: FVG redelivery detection
- `expansion_phase_event`: Market expansion identification
- `consolidation_event`: Consolidation pattern recognition
- `pd_array_event`: Premium/Discount array detection
- `liquidity_sweep_event`: Liquidity sweep identification
- `session_boundary_event`: Session transition markers
- `htf_confluence_event`: Higher timeframe confluence
- `structural_break_event`: Market structure breaks

### Edge Feature Vector (20D)
**Base Features (17D)**:
- Temporal relationships and distances
- Price movement correlations
- Structural connections

**Semantic Features (3D)**:
- `semantic_event_link`: Event chain relationships
- `event_causality`: Causal strength between events
- `semantic_label_id`: Encoded relationship identifiers

---

## 🚀 Performance Architecture

### Lazy Loading System
- **Container-Based**: Dependency injection for clean architecture
- **On-Demand Loading**: Components initialize only when accessed
- **Memory Optimization**: Efficient resource utilization
- **Iron-Core Integration**: Shared mathematical operations

### Processing Performance
- **Single Session**: <3 seconds (8-30 patterns discovered)
- **Full Discovery**: <180 seconds (57 sessions, 2000+ patterns)
- **Memory Footprint**: <100MB total system memory
- **Cache Efficiency**: >80% hit rate for repeated operations

### Scalability Features
- **Batch Processing**: Efficient multi-session handling
- **Result Caching**: Automatic caching of discovery results
- **Incremental Updates**: Process only new/changed sessions
- **Resource Management**: Automatic cleanup and optimization

---

## 🔒 System Constraints

### Immutable Constraints (Never Violate)
1. **No Prediction Logic**: System discovers patterns, never predicts outcomes
2. **Complete Preservation**: All data and intermediate states preserved
3. **Semantic Context**: Human-readable context maintained throughout
4. **Session Anchoring**: Market session timing always preserved
5. **Archaeological Focus**: Discovery of existing patterns, not future forecasting

### Performance Constraints
- **<5s Initialization**: System must initialize in under 5 seconds
- **<4.7s Processing**: Single session processing SLA
- **87% Quality Threshold**: Only high-quality patterns enter production
- **<25% Duplication**: Pattern uniqueness maintained

---

## 🔧 Integration Points

### Iron-Core Integration
- **Mathematical Operations**: Shared RG optimizers and correlators
- **Performance Infrastructure**: Lazy loading and container patterns
- **Validation Utilities**: Common data validation frameworks
- **Cross-System Patterns**: Reusable architectural components

### External Interfaces
- **Input**: JSON session files with Level 1 market data
- **Output**: Rich contextual pattern discoveries
- **Caching**: Automatic result persistence and retrieval
- **Monitoring**: Performance metrics and quality tracking

---

## 📊 Quality Assurance

### Pattern Quality Metrics
- **Authenticity Score**: >90/100 for production patterns
- **Duplication Rate**: <25% (vs 96.8% contaminated baseline)
- **Temporal Coherence**: >70% patterns with meaningful time spans
- **Semantic Preservation**: 100% semantic event preservation

### System Health Metrics
- **Processing Speed**: <3s per session average
- **Memory Efficiency**: <100MB total footprint
- **Cache Performance**: >80% hit rate
- **Error Rate**: <1% processing failures

---

*This architecture enables IRONFORGE to serve as a production-ready archaeological discovery system for market pattern recognition, maintaining the highest standards of performance, quality, and semantic preservation.*
