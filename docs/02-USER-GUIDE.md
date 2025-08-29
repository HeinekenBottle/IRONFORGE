# IRONFORGE User Guide
**Version**: 1.1.0  
**Last Updated**: 2025-01-15

## 🎯 Overview

Complete guide to using IRONFORGE for archaeological market pattern discovery. Covers daily workflows, advanced analysis, and best practices.

## 📋 Table of Contents
- [Daily Workflows](#daily-workflows)
- [Pattern Discovery](#pattern-discovery)
- [Advanced Analysis](#advanced-analysis)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

## 🌅 Daily Workflows

### Morning Market Preparation

Start each trading day with comprehensive archaeological analysis:

```python
from ironforge.analysis.daily_discovery_workflows import morning_prep

# Get comprehensive morning analysis
analysis = morning_prep(days_back=7)

# Results include:
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

### End-of-Day Review

```python
from ironforge.analysis.daily_discovery_workflows import end_of_day_review

# Comprehensive end-of-day analysis
review = end_of_day_review()

# Includes:
# - Pattern performance summary
# - Cross-session relationship analysis
# - Next day preparation insights
# - Performance metrics and statistics
```

## 🔍 Pattern Discovery

### Basic Pattern Discovery

```python
from ironforge.api import run_discovery, Config, load_config

# Load configuration
config = load_config('configs/dev.yml')

# Run discovery
results = run_discovery(config)

# Results include:
# - Discovered patterns with authenticity scores
# - Attention weights and explanations
# - Pattern confidence metrics
# - Archaeological insights
```

### Advanced Pattern Analysis

```python
from ironforge.analysis.pattern_intelligence import analyze_market_intelligence

# Comprehensive market intelligence
intel = analyze_market_intelligence()

# Pattern trends (statistically significant)
for pattern_type, trend in intel['pattern_trends'].items():
    if trend.get('significance', 1.0) < 0.05:  # p < 0.05
        print(f"{pattern_type}: {trend['description']}")
        print(f"Trend Strength: {trend['trend_strength']:.2f}")

# Market regimes
for regime in intel['market_regimes']:
    print(f"{regime['regime_name']}: {len(regime['sessions'])} sessions")
    print(f"Characteristics: {', '.join(regime['characteristic_patterns'][:3])}")
```

### Cross-Session Analysis

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
    print(f"Session {link['session_a']} ↔ {link['session_b']}: {link['similarity']:.3f}")
```

## ⚙️ Configuration

### Basic Configuration

```yaml
# configs/dev.yml
workspace: runs/{date}
data:
  shards_glob: "data/shards/*.parquet"
  symbol: "ES"
  timeframe: "1m"
scoring:
  weights:
    cluster_z: 0.30
    htf_prox: 0.25
    structure: 0.20
    cycle: 0.15
    precursor: 0.10
```

### Advanced Configuration

```yaml
# configs/production.yml
data:
  symbol: "NQ"
  timeframe: "M5"
  shards_glob: "/data/shards/NQ_M5/*.parquet"

scoring:
  weights:
    cluster_z: 0.40
    htf_prox: 0.30
    structure: 0.20
    cycle: 0.10

validation:
  authenticity_threshold: 0.87
  confidence_level: 0.95
  significance_threshold: 0.01

reporting:
  minidash:
    out_html: "dashboard.html"
    out_png: "dashboard.png"
    width: 1400
    height: 800
```

## 🎯 Best Practices

### Performance Optimization

1. **Use Lazy Loading**: Initialize components only when needed
```python
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading

# Initialize once
container = initialize_ironforge_lazy_loading()

# Get components as needed
graph_builder = container.get_enhanced_graph_builder()
discovery = container.get_tgat_discovery()
```

2. **Batch Processing**: Process multiple sessions together
```python
# Process all sessions at once
results = discovery_sdk.discover_all_sessions()

# Instead of individual session processing
for session in sessions:
    result = discovery_sdk.discover_session(session)  # Less efficient
```

3. **Configuration Management**: Use environment-specific configs
```python
import os
from ironforge.api import load_config

# Load appropriate config based on environment
env = os.getenv('IRONFORGE_ENV', 'dev')
config = load_config(f'configs/{env}.yml')
```

### Quality Assurance

1. **Always Validate Results**: Check authenticity scores
```python
# Validate patterns before use
valid_patterns = [p for p in patterns if p['authenticity'] >= 0.87]
print(f"Valid patterns: {len(valid_patterns)}/{len(patterns)}")
```

2. **Statistical Significance**: Check p-values and confidence intervals
```python
# Ensure statistical significance
if result['significance'] < 0.01 and result['confidence'] >= 0.95:
    print("✅ Statistically significant result")
else:
    print("⚠️ Result not statistically significant")
```

3. **Cross-Validation**: Use multiple validation methods
```python
# Multiple validation approaches
validation_results = validate_run(config, methods=['statistical', 'temporal', 'cross_session'])
```

### Research Methodology

1. **Configuration-Driven Research**: Use research templates
```python
from .ironforge.research_templates.configurable_research_template import (
    create_research_configuration, ConfigurableResearchFramework
)

# Create configuration
config = create_research_configuration(
    research_question="Do events cluster at specific percentage levels?",
    hypothesis_parameters={
        "percentage_levels": [20, 30, 40, 50, 60, 70, 80],
        "time_windows": [30, 60, 120, 300]
    },
    agents=["data-scientist", "adjacent-possible-linker"]
)

# Execute research
framework = ConfigurableResearchFramework(config)
results = framework.execute_research()
```

2. **Agent Coordination**: Use appropriate agents for complex research
```python
# For statistical analysis
agents = ["data-scientist"]

# For cross-session research
agents = ["data-scientist", "knowledge-architect"]

# For creative research
agents = ["data-scientist", "adjacent-possible-linker"]

# For complex projects
agents = ["data-scientist", "knowledge-architect", "adjacent-possible-linker", "scrum-master"]
```

## 🔗 Related Documentation
- [Quickstart Guide](01-QUICKSTART.md) - Getting started
- [API Reference](03-API-REFERENCE.md) - Programmatic interface
- [Architecture](04-ARCHITECTURE.md) - System design
- [Troubleshooting](06-TROUBLESHOOTING.md) - Common issues