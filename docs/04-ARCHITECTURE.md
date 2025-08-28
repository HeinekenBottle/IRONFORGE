# IRONFORGE Architecture
**Version**: 1.1.0  
**Last Updated**: 2025-01-15

## 🎯 Overview

IRONFORGE is a sophisticated archaeological discovery system that uncovers hidden patterns in financial market data using advanced temporal graph attention networks (TGAT) and semantic feature analysis.

## 📋 Table of Contents
- [System Overview](#system-overview)
- [Architectural Principles](#architectural-principles)
- [Core Pipeline](#core-pipeline)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Performance Specifications](#performance-specifications)

## 🏛️ System Overview

### Core Mission
**Archaeological discovery of market patterns (NOT prediction)**

### Key Characteristics
- **Architecture**: Component-based lazy loading with iron-core integration
- **Performance**: 88.7% improvement through lazy loading optimization
- **Features**: 51D node vectors, 20D edge vectors with semantic event preservation
- **Quality**: >87% authenticity threshold for production patterns

### System Boundaries
- ✅ **Input**: Level 1 JSON session data with market events
- ✅ **Processing**: Graph construction, TGAT discovery, pattern validation
- ✅ **Output**: Rich contextual archaeological patterns with attention weights
- ❌ **Forbidden**: Prediction logic, trading signals, future forecasts

## 🏗️ Architectural Principles

### Core Design Philosophy

1. **Archaeological Discovery Focus**: System discovers patterns, never predicts future outcomes
2. **Complete Data Preservation**: All raw data and intermediate states are preserved
3. **Semantic Context Preservation**: Market events maintain human-readable context
4. **Lazy Loading Performance**: Components load only when needed for optimal performance
5. **Iron-Core Integration**: Shared mathematical infrastructure for performance optimization

### Data Contracts (Golden Invariants - Never Change)

- **Events**: Exactly 6 types (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
- **Edge Intents**: Exactly 4 types (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)
- **Feature Dimensions**: 51D nodes (f0-f50), 20D edges (e0-e19)
- **HTF Rule**: Last-closed only (no intra-candle)
- **Session Boundaries**: No cross-session edges
- **Within-session Learning**: Preserve session isolation

## 🔄 Core Pipeline

### 4-Stage Canonical Pipeline

1. **Discovery** (`discover-temporal`) - TGAT-based pattern discovery from enhanced session graphs
2. **Confluence** (`score-session`) - Rule-based confluence scoring and validation
3. **Validation** (`validate-run`) - Quality gates and validation rails
4. **Reporting** (`report-minimal`) - Minidash dashboard generation

### Pipeline Flow
```
Enhanced Sessions (JSON) → Enhanced Graph Builder → 45D/20D Graphs
Graphs → TGAT Discovery → Archaeological Patterns
Patterns → Pattern Graduation → Validated Patterns (>87% authenticity)
Validated Patterns → Confluence Scoring → Scored Patterns
Scored Patterns → Minidash Reporting → Interactive Dashboard
```

## 🧩 Component Architecture

### Package Structure
```
ironforge/
├── api.py              # Centralized API (recommended import point)
├── sdk/               # CLI and configuration management
├── learning/          # TGAT discovery and enhanced graph building
├── confluence/        # Confluence scoring engine
├── validation/        # Quality gates and validation rails
├── reporting/         # Minidash dashboard generation
├── analysis/          # Pattern intelligence and workflows
├── synthesis/         # Pattern graduation and quality control
├── contracts/         # Data contracts and schema validation
├── temporal/          # Temporal intelligence systems
├── integration/       # Container system and lazy loading
└── utilities/         # Common utilities and helpers
```

### Key Components

#### Enhanced Graph Builder
- Transforms JSON sessions into 45D/20D TGAT-compatible graphs
- Maintains temporal causality through (timestamp, seq_idx) ordering
- Supports both 45D and 53D feature dimensions
- Preserves session isolation and archaeological principles

#### TGAT Discovery
- Temporal graph attention networks for pattern learning
- Unsupervised attention mechanisms
- Archaeological pattern discovery without assumptions
- Attention weight analysis for pattern explainability

#### Pattern Graduation
- Validates patterns against 87% authenticity threshold
- Quality gates and validation rails
- Production-ready pattern certification
- Statistical significance testing

#### Confluence Scoring
- Rule-based scoring with configurable weights
- Multi-factor confluence analysis
- Statistical validation requirements
- Quality threshold enforcement

#### Container System
- Sophisticated dependency injection for performance
- Lazy loading of components
- Iron-core integration for mathematical operations
- Memory-efficient component management

## 📊 Data Flow

### Input Data
- **Enhanced Sessions**: JSON format with market events and metadata
- **Session Boundaries**: Clear temporal boundaries for archaeological analysis
- **Event Taxonomy**: Exactly 6 event types with semantic context

### Processing Pipeline
1. **Graph Construction**: Transform sessions into temporal graphs
2. **Feature Engineering**: Extract 51D node and 20D edge features
3. **TGAT Processing**: Apply temporal graph attention networks
4. **Pattern Discovery**: Identify archaeological patterns
5. **Quality Validation**: Apply authenticity thresholds
6. **Confluence Scoring**: Score pattern confluence
7. **Dashboard Generation**: Create interactive reports

### Output Data
- **Patterns**: Discovered temporal patterns with authenticity scores
- **Embeddings**: TGAT model outputs and attention weights
- **Confluence Scores**: Rule-based scoring results
- **Dashboards**: Interactive HTML reports with PNG export

## ⚡ Performance Specifications

### Strict Performance Requirements
- **Single Session**: <3 seconds processing
- **Full Discovery**: <180 seconds (57 sessions)
- **Initialization**: <2 seconds with lazy loading
- **Memory Usage**: <100MB total footprint
- **Quality**: >87% authenticity threshold for production patterns

### Performance Optimizations

#### Lazy Loading
```python
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading

# Initialize container (do this once)
container = initialize_ironforge_lazy_loading()

# Get components
graph_builder = container.get_enhanced_graph_builder()
discovery = container.get_tgat_discovery()
```

#### Memory Management
- **Component Pooling**: Reuse components across sessions
- **Efficient Data Structures**: Optimized for temporal graph processing
- **Garbage Collection**: Automatic cleanup of intermediate results

#### I/O Optimization
- **Parquet Format**: ZSTD compression for fast I/O
- **Batch Processing**: Process multiple sessions together
- **Streaming**: Process large datasets without loading everything into memory

## 🔒 Quality Gates

### System Quality Controls
- **Authenticity Score**: >87/100 for production
- **Duplication Rate**: <25%
- **Temporal Coherence**: >70%
- **Pattern Confidence**: >0.7 threshold
- **Contract Validation**: Automatic schema validation

### Validation Pipeline
1. **Input Validation**: Verify data contracts and schemas
2. **Processing Validation**: Check intermediate results
3. **Output Validation**: Verify final results meet quality standards
4. **Cross-Validation**: Use multiple validation methods

## 🚀 Deployment Architecture

### Development Environment
- **Local Processing**: Single-machine processing
- **Configuration**: `configs/dev.yml`
- **Data**: Local shard files
- **Output**: Local run directories

### Production Environment
- **Scalable Processing**: Multi-machine processing capability
- **Configuration**: `configs/production.yml`
- **Data**: Distributed shard storage
- **Output**: Centralized result storage

### Monitoring and Observability
- **Performance Metrics**: Processing time, memory usage, quality scores
- **Health Checks**: System health indicators
- **Logging**: Comprehensive logging for debugging
- **Dashboards**: Real-time monitoring dashboards

## 🔗 Related Documentation
- [Quickstart Guide](01-QUICKSTART.md) - Getting started
- [User Guide](02-USER-GUIDE.md) - Complete usage guide
- [API Reference](03-API-REFERENCE.md) - Programmatic interface
- [TGAT Architecture](specialized/TGAT-ARCHITECTURE.md) - Technical deep-dive