# IRONFORGE Glossary
**Version**: 1.1.0  
**Last Updated**: 2025-01-15

## 📋 Table of Contents
- [Core Concepts](#core-concepts)
- [Technical Terms](#technical-terms)
- [Data Structures](#data-structures)
- [Performance Metrics](#performance-metrics)
- [Quality Standards](#quality-standards)

## 🎯 Core Concepts

### Archaeological Discovery
The process of uncovering hidden patterns in market data without making predictions about future outcomes. IRONFORGE focuses on discovering existing patterns rather than forecasting.

### Enhanced Sessions
Market session data that has been processed to include rich contextual information, event detection, and semantic features. Enhanced sessions contain 51D node features and 20D edge features.

### Temporal Graph Attention Networks (TGAT)
Advanced neural network architecture used by IRONFORGE to discover temporal patterns in market data. TGAT processes temporal graphs with attention mechanisms to identify significant patterns.

### Pattern Graduation
The process of validating discovered patterns against quality thresholds (87% authenticity) to ensure they meet production standards before being used in analysis.

### Confluence Scoring
Rule-based scoring system that evaluates the confluence of multiple factors to determine pattern strength and reliability.

## 🔧 Technical Terms

### Data Contracts
Golden invariants that never change in IRONFORGE:
- **Events**: Exactly 6 types (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
- **Edge Intents**: Exactly 4 types (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)
- **Feature Dimensions**: 51D nodes (f0-f50), 20D edges (e0-e19)
- **HTF Rule**: Last-closed only (no intra-candle)
- **Session Boundaries**: No cross-session edges
- **Within-session Learning**: Preserve session isolation

### Lazy Loading
Performance optimization technique where components are loaded only when needed, reducing initialization time and memory usage.

### Container System
Dependency injection system that manages component lifecycle and provides efficient access to IRONFORGE components.

### Iron-Core Integration
Integration with the iron-core shared mathematical infrastructure for performance optimization and common functionality.

### ZSTD Compression
Compression algorithm used for Parquet files to achieve 5.8× faster I/O performance.

## 📊 Data Structures

### Enhanced Graph
Temporal graph representation of market session data with:
- **Nodes**: Market events with 51D feature vectors
- **Edges**: Relationships between events with 20D feature vectors
- **Temporal Ordering**: Maintains chronological sequence of events

### Attention Weights
TGAT model outputs that indicate which parts of the temporal graph are most important for pattern discovery.

### Embeddings
High-dimensional vector representations of patterns learned by the TGAT model.

### Confluence Scores
Numerical scores indicating the strength and reliability of discovered patterns.

### Quality Metrics
Quantitative measures of pattern quality including authenticity, confidence, and statistical significance.

## ⚡ Performance Metrics

### Processing Time
- **Single Session**: <3 seconds processing
- **Full Discovery**: <180 seconds (57 sessions)
- **Initialization**: <2 seconds with lazy loading

### Memory Usage
- **Total Footprint**: <100MB
- **Peak Memory**: 1.2MB (99.98% under limit)
- **Efficiency**: 73% reduction in peak usage

### Quality Thresholds
- **Authenticity Score**: >87/100 for production
- **Duplication Rate**: <25%
- **Temporal Coherence**: >70%
- **Pattern Confidence**: >0.7 threshold

## 🛡️ Quality Standards

### Authenticity Threshold
Minimum quality score (87%) that patterns must achieve to be considered production-ready.

### Statistical Significance
Statistical validation requirements including p-values <0.01 and confidence intervals ≥95%.

### Cross-Validation
Validation method using multiple approaches to ensure pattern reliability.

### Quality Gates
Automated checks that ensure all patterns meet quality standards before production use.

### Framework Compliance
Adherence to research-agnostic methodology with configuration-driven research rather than hardcoded assumptions.

## 🔄 Workflow Terms

### Morning Preparation
Daily workflow for market analysis including pattern discovery, regime assessment, and session focus identification.

### Session Hunting
Real-time pattern discovery for specific session types (NY_PM, LONDON, ASIA).

### Cross-Session Analysis
Analysis of pattern relationships across different trading sessions and timeframes.

### End-of-Day Review
Comprehensive analysis of daily performance including pattern effectiveness and next-day preparation.

## 🧠 Research Framework

### Configuration-Driven Research
Research methodology that uses explicit configuration parameters rather than hardcoded assumptions.

### Hypothesis Parameters
Configurable parameters that define research questions and testable hypotheses.

### Agent Coordination
Multi-agent workflows using specialized agents (data-scientist, knowledge-architect, adjacent-possible-linker, scrum-master).

### Research Templates
Standardized templates for conducting systematic, professional research.

### Framework Validator
Automated system that detects violations of research framework principles.

## 📈 Market Structure Terms

### Market Regimes
Distinct market conditions characterized by specific pattern types and behaviors.

### Pattern Types
Categories of discovered patterns including temporal_structural, htf_confluence, and others.

### Session Types
Different trading sessions (NY_PM, LONDON, ASIA) with distinct characteristics.

### Event Taxonomy
Classification system for market events with exactly 6 types.

### Temporal Non-locality
Advanced concept where events position relative to final session range with temporal precision.

## 🔗 Related Documentation
- [Quickstart Guide](01-QUICKSTART.md) - Getting started
- [User Guide](02-USER-GUIDE.md) - Complete usage guide
- [API Reference](03-API-REFERENCE.md) - Programmatic interface
- [Architecture](04-ARCHITECTURE.md) - System design
- [Troubleshooting](06-TROUBLESHOOTING.md) - Common issues