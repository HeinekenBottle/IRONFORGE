# IRONFORGE User Guide
**Complete Guide to Archaeological Market Pattern Discovery**

---

## 🎯 Overview

IRONFORGE transforms raw market session data into rich archaeological discoveries using advanced temporal graph attention networks (TGAT). This guide covers all aspects of daily usage, from basic pattern discovery to advanced cross-session analysis.

### Key Capabilities
- **Systematic Processing**: All enhanced sessions with cross-session analysis
- **Pattern Intelligence**: Advanced classification, trending, and relationship mapping
- **Daily Workflows**: Morning prep, session hunting, performance tracking
- **Real-time Analysis**: Sub-5s initialization, efficient processing
- **Archaeological Focus**: Discovery of existing patterns (no predictions)

---

## 🚀 Daily Workflows

### Morning Market Preparation

Start each trading day with comprehensive archaeological analysis:

```python
from ironforge.analysis.daily_discovery_workflows import morning_prep

# Get comprehensive morning analysis
analysis = morning_prep(days_back=7)

# Results automatically include:
# - Dominant pattern types for the day
# - Cross-session continuation signals  
# - Current market regime assessment
# - Session-specific focus areas
# - Actionable archaeological insights
```

**Example Output**:
```
🌅 MORNING MARKET ANALYSIS - 2025-08-16
================================================
📊 Pattern Overview:
   Strength Score: 0.73/1.0
   Confidence Level: High
   Current Regime: Strong temporal_structural regime

🔥 Dominant Pattern Types:
   1. temporal_structural (confidence: 0.78)
   2. htf_confluence (confidence: 0.71)

🔗 Cross-Session Signals:
   • temporal_structural patterns continuing from previous session
   • Pattern confidence strengthening over recent sessions

💡 Archaeological Insights:
   • Focus on NY_PM session - high confidence patterns detected
   • Primary pattern theme: temporal_structural patterns dominating
   • Watch for structural position entries - temporal patterns active
```

### Session Pattern Hunting

Discover patterns in real-time for specific session types:

```python
from ironforge.analysis.daily_discovery_workflows import hunt_patterns

# Focus on specific session types
ny_pm_patterns = hunt_patterns('NY_PM')
london_patterns = hunt_patterns('LONDON')
asia_patterns = hunt_patterns('ASIA')

# Each returns:
# - Patterns discovered in latest session
# - Strength indicators and confidence metrics
# - Historical comparisons to similar sessions
# - Immediate archaeological insights
# - Next session expectations
```

### Cross-Session Relationship Analysis

Discover pattern relationships across different sessions and timeframes:

```python
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading

# Initialize system
container = initialize_ironforge_lazy_loading()
discovery_sdk = container.get_discovery_sdk()

# Discover all patterns first
results = discovery_sdk.discover_all_sessions()

# Find cross-session relationships
links = discovery_sdk.find_cross_session_links(min_similarity=0.7)

print(f"Found {len(links)} cross-session pattern relationships")
for link in links[:5]:
    print(f"- {link.link_type}: {link.description}")
    print(f"  Strength: {link.link_strength:.2f}")
    print(f"  Distance: {link.temporal_distance_days:.1f} days")
```

---

## 🔍 Pattern Analysis

### Understanding Pattern Output

IRONFORGE discovers rich contextual patterns with archaeological significance:

```python
# Example discovered pattern
pattern = {
    'pattern_id': 'NY_session_RPC_00',
    'session_name': 'NY_session',
    'session_start': '14:30:00',
    'anchor_timeframe': 'multi_timeframe',
    'archaeological_significance': {
        'archaeological_value': 'high_archaeological_value',
        'permanence_score': 0.933
    },
    'semantic_context': {
        'market_regime': 'transitional',
        'event_types': ['fvg_redelivery', 'expansion_phase'],
        'relationship_type': 'confluence_relationship'
    },
    'confidence': 0.87,
    'description': 'Multi-timeframe confluence with FVG redelivery in NY session'
}
```

### Pattern Quality Metrics

- **Confidence**: 0.0-1.0 (higher = more reliable pattern)
- **Authenticity**: 0-100 (>87 required for production)
- **Permanence Score**: 0.0-1.0 (pattern stability over time)
- **Archaeological Value**: low/medium/high significance rating

### Semantic Context Understanding

Each pattern includes rich semantic context:

- **Event Types**: FVG redelivery, expansion phases, consolidation, PD arrays
- **Market Regime**: Trending, ranging, transitional, breakout
- **Relationship Type**: Confluence, sequence, causality, structural
- **Session Anchoring**: Timing, characteristics, market conditions

---

## 🧠 Pattern Intelligence

### Advanced Pattern Classification

```python
from ironforge.analysis.pattern_intelligence import PatternIntelligenceEngine

# Initialize intelligence engine
intel_engine = PatternIntelligenceEngine(discovery_sdk)

# Analyze pattern trends over time
trends = intel_engine.analyze_pattern_trends(days_lookback=30)

for pattern_type, trend in trends.items():
    if trend.significance < 0.05:  # Statistically significant
        print(f"{pattern_type}: {trend.description}")
        print(f"  Trend strength: {trend.trend_strength:.2f}")
        print(f"  Confidence trend: {trend.avg_confidence_trend:.3f}")
```

### Market Regime Identification

```python
# Identify current market regimes
regimes = intel_engine.identify_market_regimes(min_sessions=3)

for regime in regimes:
    print(f"Regime: {regime.regime_name}")
    print(f"  Sessions: {len(regime.sessions)}")
    print(f"  Patterns: {', '.join(regime.characteristic_patterns)}")
    print(f"  Period: {regime.start_date} to {regime.end_date}")
    print(f"  Description: {regime.description}")
```

### Finding Similar Patterns

```python
# Find patterns similar to those in NY_PM sessions
matches = intel_engine.find_similar_patterns('NY_PM', similarity_threshold=0.8)

for match in matches[:5]:
    print(f"Pattern: {match.description}")
    print(f"Similarity: {match.similarity_score:.2f}")
    print(f"Historical context: {match.historical_context}")
```

---

## 📊 Session Analysis

### Session Performance Comparison

Compare pattern performance across different session types:

```python
# Analyze session performance
session_performance = {}

for session_type in ['NY_AM', 'NY_PM', 'LONDON', 'ASIA']:
    session_patterns = discovery_sdk.get_session_patterns(session_type)
    
    if session_patterns:
        avg_confidence = sum(p.confidence for p in session_patterns) / len(session_patterns)
        pattern_count = len(session_patterns)
        
        session_performance[session_type] = {
            'pattern_count': pattern_count,
            'avg_confidence': avg_confidence,
            'quality_score': sum(p.authenticity_score for p in session_patterns) / len(session_patterns)
        }

# Display results
for session, metrics in session_performance.items():
    print(f"{session}:")
    print(f"  Total patterns: {metrics['pattern_count']}")
    print(f"  Avg confidence: {metrics['avg_confidence']:.2f}")
    print(f"  Quality score: {metrics['quality_score']:.1f}/100")
```

### Structural Position Analysis

Analyze patterns at key structural levels:

```python
# Find all structural position patterns
structural_patterns = [
    p for p in discovery_sdk.pattern_database.values() 
    if p.pattern_type == 'temporal_structural'
]

# Analyze structural position distribution
positions = [p.structural_position for p in structural_patterns]
print(f"Structural positions range: {min(positions):.2f} - {max(positions):.2f}")
print(f"Average position: {sum(positions)/len(positions):.2f}")

# Find patterns at key levels
key_levels = [0.25, 0.5, 0.75]  # 25%, 50%, 75% levels
for level in key_levels:
    nearby_patterns = [
        p for p in structural_patterns 
        if abs(p.structural_position - level) < 0.1
    ]
    print(f"Patterns near {level*100}% level: {len(nearby_patterns)}")
```

---

## 🔧 Advanced Usage

### Custom Pattern Analysis

```python
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

# Initialize components directly
builder = EnhancedGraphBuilder()
discovery = IRONFORGEDiscovery()

# Process specific session
session_file = Path("data/enhanced/enhanced_rel_NY_PM_Lvl-1_2025_07_29.json")
with open(session_file) as f:
    session_data = json.load(f)

# Build enhanced graph
graph = builder.enhance_session(session_data)
print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

# Discover patterns
patterns = discovery.discover_patterns(graph)
print(f"Discovered {len(patterns)} patterns")

# Analyze semantic features
semantic_features = builder.extract_semantic_features(session_data)
for event_type, events in semantic_features.items():
    if events:
        print(f"{event_type}: {len(events)} events")
```

### Batch Processing

Process multiple sessions efficiently:

```python
from pathlib import Path
import json

# Get all enhanced session files
session_files = list(Path("data/enhanced").glob("*.json"))

# Process in batches
batch_size = 10
all_patterns = []

for i in range(0, len(session_files), batch_size):
    batch = session_files[i:i+batch_size]
    
    batch_patterns = []
    for session_file in batch:
        with open(session_file) as f:
            session_data = json.load(f)
        
        graph = builder.enhance_session(session_data)
        patterns = discovery.discover_patterns(graph)
        batch_patterns.extend(patterns)
    
    all_patterns.extend(batch_patterns)
    print(f"Processed batch {i//batch_size + 1}: {len(batch_patterns)} patterns")

print(f"Total patterns discovered: {len(all_patterns)}")
```

---

## 📈 Performance Optimization

### Monitoring System Performance

```python
from ironforge.utilities.performance_monitor import monitor_system_performance
import time

# Monitor performance
start_time = time.time()
container = initialize_ironforge_lazy_loading()
init_time = time.time() - start_time

print(f"Initialization time: {init_time:.2f}s")

# Get detailed performance metrics
metrics = monitor_system_performance()
print(f"Memory usage: {metrics['memory_usage_mb']:.1f} MB")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
print(f"Average processing time: {metrics['average_processing_time']:.2f}s")
```

### Caching System

IRONFORGE automatically caches results for performance:

```python
# Cache locations
cache_dirs = [
    'discovery_cache/discovery_results_*.json',
    'discovery_cache/pattern_intelligence/',
    'discovery_cache/daily_workflows/'
]

# Check cache status
from pathlib import Path
for cache_pattern in cache_dirs:
    cache_files = list(Path('.').glob(cache_pattern))
    print(f"Cache files: {len(cache_files)} in {cache_pattern}")
```

---

## 🚨 Troubleshooting

### Common Issues

1. **No Patterns Discovered**
```python
# Check session data quality
with open('data/enhanced/session.json') as f:
    data = json.load(f)
    
print(f"Price movements: {len(data.get('price_movements', []))}")
print(f"Enhanced features: {bool(data.get('enhanced_features'))}")
print(f"Semantic events: {len(data.get('semantic_events', []))}")
```

2. **Low Confidence Patterns**
```python
# Adjust confidence threshold
patterns = [p for p in all_patterns if p.confidence >= 0.6]  # Lower threshold
print(f"Patterns with confidence ≥0.6: {len(patterns)}")
```

3. **Performance Issues**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system resources
import psutil
print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024:.0f} MB")
print(f"CPU usage: {psutil.cpu_percent()}%")
```

---

## ✅ Best Practices

### Daily Usage Routine

1. **Morning Preparation** (5 minutes)
   - Run `morning_prep()` for market overview
   - Identify focus sessions and dominant patterns
   - Review cross-session continuation signals

2. **Session Hunting** (Throughout day)
   - Use `hunt_patterns()` for active sessions
   - Monitor high-confidence pattern alerts
   - Track pattern performance in real-time

3. **End-of-Day Analysis** (10 minutes)
   - Review discovered patterns and quality metrics
   - Update pattern intelligence database
   - Prepare insights for next trading day

### Quality Assurance

- Always validate pattern confidence scores (>0.7 recommended)
- Check authenticity scores for production use (>87/100)
- Monitor duplication rates (<25% target)
- Verify semantic context preservation

### Performance Guidelines

- Use lazy loading container for optimal performance
- Process sessions in batches for efficiency
- Enable caching for repeated operations
- Monitor memory usage during large batch processing

---

*This user guide provides comprehensive coverage of IRONFORGE's archaeological discovery capabilities. For technical details, see the [API Reference](API_REFERENCE.md), and for system architecture, review the [Architecture Guide](ARCHITECTURE.md).*
