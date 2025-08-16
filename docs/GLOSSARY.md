# IRONFORGE Glossary
**Archaeological Discovery Terminology and Concepts**

---

## 🏛️ Core Concepts

### Archaeological Discovery
The process of uncovering hidden patterns and relationships in historical market data without attempting to predict future outcomes. IRONFORGE focuses on discovering what exists rather than forecasting what might happen.

### Attention Weights
Numerical values produced by the TGAT neural network indicating the strength of relationships between different market events. Higher attention weights suggest stronger archaeological significance between connected events.

### Authenticity Score
A quality metric (0-100) measuring how genuine and meaningful a discovered pattern is. Patterns must achieve >87/100 authenticity to enter production use. Factors include temporal consistency, semantic coherence, and cross-session validation.

---

## 🧠 TGAT Architecture

### Temporal Graph Attention Network (TGAT)
The core neural architecture powering IRONFORGE's pattern discovery. Uses multi-head attention mechanisms to identify temporal relationships in market data while preserving complete archaeological context.

### Multi-Head Attention
A neural network architecture with 4 specialized attention heads:
- **Head 1**: Structural patterns (support/resistance, ranges)
- **Head 2**: Temporal patterns (session boundaries, cycles)  
- **Head 3**: Confluence patterns (multi-timeframe alignments)
- **Head 4**: Semantic patterns (FVG chains, event sequences)

### Temporal Encoding
Mathematical representation of time relationships between market events, allowing TGAT to understand distant temporal correlations and session-based patterns.

### Self-Supervised Learning
Training approach where the neural network learns patterns from data structure itself without requiring external labels or supervision. Enables discovery of previously unknown market relationships.

---

## 📊 Feature Vectors

### Node Features (45D)
Mathematical representation of market events with 45 dimensions:
- **Base Features (37D)**: Price relativity (34D) + temporal cycles (3D)
- **Semantic Features (8D)**: FVG events, expansion phases, consolidation, PD arrays, liquidity sweeps, session boundaries, HTF confluence, structural breaks

### Edge Features (20D)
Mathematical representation of relationships between market events with 20 dimensions:
- **Base Features (17D)**: Temporal distances, price correlations, structural connections
- **Semantic Features (3D)**: Event chain links, causal strength, relationship identifiers

### Price Relativity
Transformation of absolute price values into relative positions within market structures (ranges, trends, key levels). Enables pattern discovery across different price scales and time periods.

---

## 🎯 Semantic Events

### FVG Redelivery Event
Fair Value Gap redelivery pattern where price returns to fill a previously created gap. Represents institutional order flow completion and market inefficiency correction.

### Expansion Phase Event
Market expansion phase where price breaks out of consolidation ranges with increased momentum. Indicates energy release and directional commitment.

### Consolidation Event
Market consolidation pattern where price moves sideways within defined ranges. Represents accumulation/distribution phases and energy building.

### PD Array Event
Premium/Discount array formation in market structure. ICT concept representing institutional order blocks and liquidity zones.

### Liquidity Sweep Event
Price movement that triggers stops or liquidity above/below key levels before reversing. Represents institutional liquidity harvesting behavior.

### Session Boundary Event
Market session transitions and their characteristic behaviors. Captures the handoff between different global trading sessions (ASIA→LONDON→NY).

### HTF Confluence Event
Higher timeframe confluence where multiple timeframe structures align. Represents multi-dimensional market agreement and structural significance.

### Structural Break Event
Market structure breaks where price violates established patterns. Indicates regime changes and new directional commitments.

---

## 🔗 Relationships & Patterns

### Semantic Relationship
Meaningful connection between market events that preserves human-readable context. Examples: FVG chains, phase transitions, PD sequences, liquidity sweep cascades.

### Cross-Session Link
Pattern relationship that spans multiple trading sessions, indicating persistent market structure or institutional behavior continuation.

### Confluence Relationship
Multiple independent factors aligning to create higher-probability market conditions. Represents convergence of different analytical dimensions.

### Causal Strength
Numerical measure (0.0-1.0) of how strongly one market event influences or leads to another. Higher values indicate stronger cause-effect relationships.

---

## 📈 Quality Metrics

### Permanence Score
Measure (0.0-1.0) of how stable and persistent a pattern is over time. Higher scores indicate patterns that maintain significance across multiple sessions and market conditions.

### Duplication Rate
Percentage of discovered patterns that are essentially identical to previously found patterns. IRONFORGE maintains <25% duplication rate vs 96.8% contaminated baseline.

### Temporal Coherence
Measure of how meaningful the time relationships are within a discovered pattern. Patterns with >70% temporal coherence show genuine time-based structure.

### Archaeological Value
Classification of pattern significance:
- **High**: Cross-session validation, strong permanence, clear semantic context
- **Medium**: Session-specific patterns with good coherence
- **Low**: Weak patterns that may be noise or temporary phenomena

---

## 🏗️ System Architecture

### Lazy Loading
Performance optimization where system components initialize only when needed. Provides 88.7% performance improvement (3.4s vs 2+ minute timeouts).

### Container System
Dependency injection architecture that manages component lifecycle and dependencies. Enables clean separation of concerns and efficient resource utilization.

### Iron-Core Integration
Shared mathematical infrastructure providing optimized operations, performance patterns, and validation utilities across the IRON ecosystem.

### Enhanced Graph Builder
Component that transforms Level 1 JSON session data into PyTorch Geometric graphs with rich 45D/20D feature vectors while preserving semantic context.

---

## 📊 Data Pipeline

### Level 1 JSON
Raw market session data format containing price movements, timestamps, session information, and semantic events. Input format for IRONFORGE processing.

### Enhanced Session
Processed session data with added semantic features, price relativity transformations, and temporal cycle encoding. Intermediate format in the pipeline.

### Pattern Graduation
Quality validation process where discovered patterns must meet 87% authenticity threshold and other quality criteria before entering production use.

### Archaeological Intelligence
Final output format providing rich contextual pattern descriptions with human-readable semantic context and archaeological significance assessment.

---

## 🎨 Session Types

### NY_AM / NY_PM
New York trading sessions representing North American market activity. NY_AM (pre-market and early session), NY_PM (afternoon and close).

### LONDON
London trading session representing European market activity. Often shows different liquidity characteristics and pattern behaviors.

### ASIA
Asian trading sessions (Tokyo, Sydney, Hong Kong). Typically lower volatility with different structural characteristics.

### Session Anchoring
Preservation of market session timing and characteristics throughout the discovery pipeline. Ensures patterns maintain their temporal and contextual meaning.

---

## 🔧 Performance Terms

### Processing SLA
Service Level Agreement for processing performance. IRONFORGE maintains <3s per session processing and <5s system initialization.

### Cache Hit Rate
Percentage of requests served from cache rather than requiring full processing. IRONFORGE achieves >80% cache hit rate for repeated operations.

### Memory Footprint
Total system memory usage. IRONFORGE maintains <100MB total footprint through lazy loading and efficient resource management.

### Batch Processing
Processing multiple sessions simultaneously for efficiency. IRONFORGE supports up to 10 sessions per batch while maintaining quality standards.

---

## 🚫 Constraints & Boundaries

### Archaeological Focus Constraint
Immutable system constraint that IRONFORGE discovers existing patterns but never predicts future outcomes. Maintains clear boundary between discovery and forecasting.

### Semantic Preservation Constraint
Requirement that human-readable context must be maintained throughout the entire processing pipeline. Prevents patterns from becoming abstract mathematical relationships.

### Quality Threshold Constraint
Minimum standards for pattern production use: >87% authenticity, <25% duplication rate, >70% temporal coherence.

### Session Context Constraint
Requirement that market session timing and characteristics must be preserved. Ensures patterns maintain their temporal and market regime context.

---

## 📚 Acronyms & Abbreviations

- **TGAT**: Temporal Graph Attention Network
- **FVG**: Fair Value Gap
- **PD**: Premium/Discount
- **HTF**: Higher Time Frame
- **ICT**: Inner Circle Trader (methodology reference)
- **SLA**: Service Level Agreement
- **API**: Application Programming Interface
- **JSON**: JavaScript Object Notation
- **CPU**: Central Processing Unit
- **GPU**: Graphics Processing Unit
- **RAM**: Random Access Memory
- **SSD**: Solid State Drive

---

## 🔄 Process Terms

### Discovery Cycle
Complete process of loading session data, building enhanced graphs, running TGAT discovery, validating patterns, and producing archaeological intelligence.

### Morning Prep
Daily workflow that analyzes recent patterns, identifies dominant themes, and provides market preparation insights for the trading day.

### Pattern Hunting
Real-time discovery process focused on specific session types or market conditions to identify immediate archaeological insights.

### Intelligence Analysis
Advanced pattern analysis including trend identification, market regime classification, and cross-session relationship mapping.

---

*This glossary provides comprehensive definitions for all IRONFORGE terminology. For practical usage examples, see the [User Guide](USER_GUIDE.md) and [Getting Started](GETTING_STARTED.md) documentation.*
