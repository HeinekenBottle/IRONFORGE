# IRONFORGE Pattern Discovery Guide
**Complete Guide to Archaeological Market Pattern Discovery**

---

## 🎯 Overview

This guide provides comprehensive coverage of IRONFORGE's pattern discovery capabilities, from basic session analysis to advanced cross-session archaeological intelligence. Learn how to extract meaningful market patterns while preserving complete semantic context.

**Discovery Philosophy**: IRONFORGE discovers existing patterns in historical data without attempting to predict future outcomes. This archaeological approach reveals hidden market structures and relationships.

---

## 🚀 Quick Pattern Discovery

### Immediate Results (5 Minutes)
Get instant insights from your session data:

```python
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
from ironforge.analysis.daily_discovery_workflows import morning_prep

# Initialize system
container = initialize_ironforge_lazy_loading()

# Get immediate morning analysis
analysis = morning_prep(days_back=7)

# Results include:
# - Dominant pattern types
# - Cross-session signals
# - Market regime assessment
# - Session focus recommendations
```

### Single Session Analysis
Analyze individual sessions for detailed pattern discovery:

```python
from pathlib import Path
import json

# Load specific session
session_file = Path('data/enhanced/enhanced_rel_NY_PM_Lvl-1_2025_07_29.json')
with open(session_file) as f:
    session_data = json.load(f)

# Build enhanced graph
builder = container.get_enhanced_graph_builder()
graph = builder.enhance_session(session_data)

print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
print(f"Node features: {graph.x.shape[1]}D (45D with semantics)")
print(f"Edge features: {graph.edge_attr.shape[1]}D (20D with semantics)")

# Discover patterns
discovery = container.get_tgat_discovery()
patterns = discovery.discover_patterns(graph)

print(f"Discovered {len(patterns)} archaeological patterns")
```

---

## 🔍 Pattern Types & Recognition

### 1. Structural Patterns
Patterns related to market structure and key levels:

```python
def analyze_structural_patterns(patterns):
    """Analyze structural pattern characteristics"""
    structural_patterns = [
        p for p in patterns 
        if 'structural' in p.get('pattern_type', '').lower()
    ]
    
    for pattern in structural_patterns:
        print(f"Structural Pattern: {pattern['description']}")
        print(f"  Position: {pattern.get('structural_position', 'N/A')}")
        print(f"  Strength: {pattern.get('structural_strength', 'N/A')}")
        print(f"  Key Levels: {pattern.get('key_levels', [])}")
        
        # Analyze archaeological significance
        arch_sig = pattern.get('archaeological_significance', {})
        print(f"  Archaeological Value: {arch_sig.get('archaeological_value', 'unknown')}")
        print(f"  Permanence Score: {arch_sig.get('permanence_score', 0):.3f}")
```

### 2. Temporal Patterns
Time-based patterns and session relationships:

```python
def analyze_temporal_patterns(patterns):
    """Analyze temporal pattern characteristics"""
    temporal_patterns = [
        p for p in patterns 
        if 'temporal' in p.get('pattern_type', '').lower()
    ]
    
    for pattern in temporal_patterns:
        semantic_context = pattern.get('semantic_context', {})
        
        print(f"Temporal Pattern: {pattern['description']}")
        print(f"  Session: {pattern.get('session_name', 'unknown')}")
        print(f"  Session Start: {pattern.get('session_start', 'N/A')}")
        print(f"  Market Regime: {semantic_context.get('market_regime', 'unknown')}")
        print(f"  Event Types: {semantic_context.get('event_types', [])}")
```

### 3. Confluence Patterns
Multi-dimensional pattern alignments:

```python
def analyze_confluence_patterns(patterns):
    """Analyze confluence pattern characteristics"""
    confluence_patterns = [
        p for p in patterns 
        if 'confluence' in p.get('description', '').lower()
    ]
    
    for pattern in confluence_patterns:
        print(f"Confluence Pattern: {pattern['description']}")
        print(f"  Timeframe: {pattern.get('anchor_timeframe', 'unknown')}")
        
        # Analyze confluence factors
        semantic_context = pattern.get('semantic_context', {})
        relationship_type = semantic_context.get('relationship_type', 'unknown')
        print(f"  Relationship Type: {relationship_type}")
        
        # Check for multi-session validation
        arch_sig = pattern.get('archaeological_significance', {})
        cross_session = arch_sig.get('cross_session_strength', 0)
        print(f"  Cross-Session Strength: {cross_session:.3f}")
```

---

## 🧠 Semantic Event Analysis

### FVG Redelivery Patterns
Analyze Fair Value Gap redelivery events:

```python
def analyze_fvg_patterns(session_data, patterns):
    """Analyze FVG redelivery patterns"""
    # Extract FVG events from session data
    fvg_events = []
    for movement in session_data.get('price_movements', []):
        if movement.get('event_type') == 'fvg_redelivery':
            fvg_events.append(movement)
    
    print(f"FVG Events in Session: {len(fvg_events)}")
    
    # Find patterns containing FVG events
    fvg_patterns = []
    for pattern in patterns:
        semantic_context = pattern.get('semantic_context', {})
        event_types = semantic_context.get('event_types', [])
        if 'fvg_redelivery' in event_types:
            fvg_patterns.append(pattern)
    
    print(f"Patterns with FVG Context: {len(fvg_patterns)}")
    
    for pattern in fvg_patterns:
        print(f"  FVG Pattern: {pattern['description']}")
        print(f"  Confidence: {pattern.get('confidence', 0):.2f}")
        
        # Analyze FVG chain relationships
        arch_sig = pattern.get('archaeological_significance', {})
        temporal_span = arch_sig.get('temporal_span', 0)
        print(f"  Temporal Span: {temporal_span:.2f}")
```

### Expansion Phase Analysis
Analyze market expansion phase patterns:

```python
def analyze_expansion_patterns(patterns):
    """Analyze expansion phase patterns"""
    expansion_patterns = [
        p for p in patterns
        if 'expansion_phase' in p.get('semantic_context', {}).get('event_types', [])
    ]
    
    print(f"Expansion Phase Patterns: {len(expansion_patterns)}")
    
    for pattern in expansion_patterns:
        semantic_context = pattern.get('semantic_context', {})
        
        print(f"Expansion Pattern: {pattern['description']}")
        print(f"  Market Regime: {semantic_context.get('market_regime', 'unknown')}")
        
        # Analyze expansion characteristics
        if 'expansion_magnitude' in pattern:
            print(f"  Magnitude: {pattern['expansion_magnitude']}")
        if 'expansion_direction' in pattern:
            print(f"  Direction: {pattern['expansion_direction']}")
        
        # Check for preceding consolidation
        if 'preceding_consolidation' in pattern:
            print(f"  Preceded by: {pattern['preceding_consolidation']}")
```

---

## 📊 Session-Based Analysis

### Session Performance Comparison
Compare pattern discovery across different sessions:

```python
def compare_session_performance(all_patterns):
    """Compare pattern performance across sessions"""
    session_stats = {}
    
    for pattern in all_patterns:
        session_name = pattern.get('session_name', 'unknown')
        
        if session_name not in session_stats:
            session_stats[session_name] = {
                'pattern_count': 0,
                'total_confidence': 0,
                'high_quality_patterns': 0,
                'semantic_events': set()
            }
        
        stats = session_stats[session_name]
        stats['pattern_count'] += 1
        stats['total_confidence'] += pattern.get('confidence', 0)
        
        # Check quality
        arch_sig = pattern.get('archaeological_significance', {})
        if arch_sig.get('archaeological_value') == 'high_archaeological_value':
            stats['high_quality_patterns'] += 1
        
        # Collect semantic events
        semantic_context = pattern.get('semantic_context', {})
        event_types = semantic_context.get('event_types', [])
        stats['semantic_events'].update(event_types)
    
    # Display results
    print("Session Performance Analysis:")
    print("=" * 50)
    
    for session, stats in session_stats.items():
        if stats['pattern_count'] > 0:
            avg_confidence = stats['total_confidence'] / stats['pattern_count']
            quality_rate = stats['high_quality_patterns'] / stats['pattern_count']
            
            print(f"\n{session}:")
            print(f"  Total Patterns: {stats['pattern_count']}")
            print(f"  Avg Confidence: {avg_confidence:.2f}")
            print(f"  High Quality Rate: {quality_rate:.1%}")
            print(f"  Semantic Events: {len(stats['semantic_events'])}")
            print(f"  Event Types: {', '.join(list(stats['semantic_events'])[:3])}")
```

### Session Timing Analysis
Analyze patterns by session timing and characteristics:

```python
def analyze_session_timing(patterns):
    """Analyze patterns by session timing"""
    timing_analysis = {
        'session_starts': {},
        'market_regimes': {},
        'session_phases': {}
    }
    
    for pattern in patterns:
        # Session start times
        session_start = pattern.get('session_start')
        if session_start:
            hour = session_start.split(':')[0]
            timing_analysis['session_starts'][hour] = timing_analysis['session_starts'].get(hour, 0) + 1
        
        # Market regimes
        semantic_context = pattern.get('semantic_context', {})
        regime = semantic_context.get('market_regime', 'unknown')
        timing_analysis['market_regimes'][regime] = timing_analysis['market_regimes'].get(regime, 0) + 1
        
        # Session phases (if available)
        session_phase = semantic_context.get('session_phase')
        if session_phase:
            timing_analysis['session_phases'][session_phase] = timing_analysis['session_phases'].get(session_phase, 0) + 1
    
    # Display timing analysis
    print("Session Timing Analysis:")
    print("=" * 30)
    
    print("\nPattern Distribution by Hour:")
    for hour, count in sorted(timing_analysis['session_starts'].items()):
        print(f"  {hour}:00 - {count} patterns")
    
    print("\nPattern Distribution by Market Regime:")
    for regime, count in timing_analysis['market_regimes'].items():
        print(f"  {regime}: {count} patterns")
```

---

## 🔗 Cross-Session Analysis

### Pattern Relationship Discovery
Find relationships between patterns across different sessions:

```python
def discover_cross_session_relationships(patterns):
    """Discover relationships between patterns across sessions"""
    relationships = []
    
    # Group patterns by session
    session_patterns = {}
    for pattern in patterns:
        session = pattern.get('session_name', 'unknown')
        if session not in session_patterns:
            session_patterns[session] = []
        session_patterns[session].append(pattern)
    
    # Find cross-session relationships
    sessions = list(session_patterns.keys())
    for i, session1 in enumerate(sessions):
        for session2 in sessions[i+1:]:
            relationships.extend(
                find_session_pair_relationships(
                    session_patterns[session1], 
                    session_patterns[session2],
                    session1, session2
                )
            )
    
    return relationships

def find_session_pair_relationships(patterns1, patterns2, session1, session2):
    """Find relationships between two session pattern sets"""
    relationships = []
    
    for p1 in patterns1:
        for p2 in patterns2:
            similarity = calculate_pattern_similarity(p1, p2)
            
            if similarity > 0.7:  # High similarity threshold
                relationship = {
                    'session1': session1,
                    'session2': session2,
                    'pattern1_id': p1.get('pattern_id', 'unknown'),
                    'pattern2_id': p2.get('pattern_id', 'unknown'),
                    'similarity_score': similarity,
                    'relationship_type': classify_relationship_type(p1, p2),
                    'temporal_distance': calculate_temporal_distance(p1, p2)
                }
                relationships.append(relationship)
    
    return relationships
```

### Pattern Evolution Tracking
Track how patterns evolve across sessions:

```python
def track_pattern_evolution(patterns):
    """Track pattern evolution across sessions"""
    # Group patterns by type and semantic context
    pattern_families = {}
    
    for pattern in patterns:
        # Create pattern signature
        semantic_context = pattern.get('semantic_context', {})
        signature = (
            pattern.get('pattern_type', 'unknown'),
            tuple(sorted(semantic_context.get('event_types', []))),
            semantic_context.get('market_regime', 'unknown')
        )
        
        if signature not in pattern_families:
            pattern_families[signature] = []
        pattern_families[signature].append(pattern)
    
    # Analyze evolution for each family
    evolution_analysis = {}
    for signature, family_patterns in pattern_families.items():
        if len(family_patterns) >= 3:  # Need multiple instances
            evolution = analyze_family_evolution(family_patterns)
            evolution_analysis[signature] = evolution
    
    return evolution_analysis

def analyze_family_evolution(family_patterns):
    """Analyze evolution of a pattern family"""
    # Sort by session timing
    sorted_patterns = sorted(
        family_patterns, 
        key=lambda p: p.get('session_start', '00:00:00')
    )
    
    evolution = {
        'pattern_count': len(sorted_patterns),
        'confidence_trend': calculate_confidence_trend(sorted_patterns),
        'quality_trend': calculate_quality_trend(sorted_patterns),
        'consistency_score': calculate_consistency_score(sorted_patterns)
    }
    
    return evolution
```

---

## 📈 Quality Assessment

### Pattern Authenticity Analysis
Assess the authenticity and reliability of discovered patterns:

```python
def assess_pattern_authenticity(patterns):
    """Comprehensive pattern authenticity assessment"""
    authenticity_analysis = {
        'total_patterns': len(patterns),
        'high_authenticity': 0,
        'medium_authenticity': 0,
        'low_authenticity': 0,
        'authenticity_distribution': {},
        'quality_factors': {}
    }
    
    for pattern in patterns:
        arch_sig = pattern.get('archaeological_significance', {})
        authenticity_score = arch_sig.get('authenticity_score', 0)
        
        # Categorize authenticity
        if authenticity_score >= 90:
            authenticity_analysis['high_authenticity'] += 1
        elif authenticity_score >= 70:
            authenticity_analysis['medium_authenticity'] += 1
        else:
            authenticity_analysis['low_authenticity'] += 1
        
        # Track authenticity distribution
        score_range = f"{int(authenticity_score//10)*10}-{int(authenticity_score//10)*10+9}"
        authenticity_analysis['authenticity_distribution'][score_range] = \
            authenticity_analysis['authenticity_distribution'].get(score_range, 0) + 1
        
        # Analyze quality factors
        permanence_score = arch_sig.get('permanence_score', 0)
        temporal_coherence = arch_sig.get('temporal_coherence', 0)
        cross_session_strength = arch_sig.get('cross_session_strength', 0)
        
        if 'permanence_scores' not in authenticity_analysis['quality_factors']:
            authenticity_analysis['quality_factors']['permanence_scores'] = []
        if 'temporal_coherence_scores' not in authenticity_analysis['quality_factors']:
            authenticity_analysis['quality_factors']['temporal_coherence_scores'] = []
        if 'cross_session_scores' not in authenticity_analysis['quality_factors']:
            authenticity_analysis['quality_factors']['cross_session_scores'] = []
        
        authenticity_analysis['quality_factors']['permanence_scores'].append(permanence_score)
        authenticity_analysis['quality_factors']['temporal_coherence_scores'].append(temporal_coherence)
        authenticity_analysis['quality_factors']['cross_session_scores'].append(cross_session_strength)
    
    return authenticity_analysis
```

---

## 🎯 Advanced Discovery Techniques

### Multi-Timeframe Analysis
Discover patterns across multiple timeframes:

```python
def multi_timeframe_discovery(session_data):
    """Discover patterns across multiple timeframes"""
    timeframes = ['1m', '5m', '15m', '1h', '4h', 'daily']
    timeframe_patterns = {}
    
    for timeframe in timeframes:
        # Extract timeframe-specific data
        tf_data = extract_timeframe_data(session_data, timeframe)
        
        if tf_data:
            # Build graph for this timeframe
            graph = builder.enhance_session(tf_data)
            
            # Discover patterns
            patterns = discovery.discover_patterns(graph)
            
            # Tag patterns with timeframe
            for pattern in patterns:
                pattern['timeframe'] = timeframe
                pattern['timeframe_priority'] = get_timeframe_priority(timeframe)
            
            timeframe_patterns[timeframe] = patterns
    
    # Find cross-timeframe confluences
    confluences = find_timeframe_confluences(timeframe_patterns)
    
    return timeframe_patterns, confluences
```

### Semantic Event Clustering
Cluster patterns by semantic event types:

```python
def cluster_by_semantic_events(patterns):
    """Cluster patterns by semantic event types"""
    event_clusters = {}
    
    for pattern in patterns:
        semantic_context = pattern.get('semantic_context', {})
        event_types = semantic_context.get('event_types', [])
        
        # Create event signature
        event_signature = tuple(sorted(event_types))
        
        if event_signature not in event_clusters:
            event_clusters[event_signature] = {
                'patterns': [],
                'avg_confidence': 0,
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }
        
        cluster = event_clusters[event_signature]
        cluster['patterns'].append(pattern)
        
        # Update cluster statistics
        arch_sig = pattern.get('archaeological_significance', {})
        arch_value = arch_sig.get('archaeological_value', 'low_archaeological_value')
        
        if 'high' in arch_value:
            cluster['quality_distribution']['high'] += 1
        elif 'medium' in arch_value:
            cluster['quality_distribution']['medium'] += 1
        else:
            cluster['quality_distribution']['low'] += 1
    
    # Calculate cluster statistics
    for signature, cluster in event_clusters.items():
        patterns = cluster['patterns']
        if patterns:
            total_confidence = sum(p.get('confidence', 0) for p in patterns)
            cluster['avg_confidence'] = total_confidence / len(patterns)
    
    return event_clusters
```

---

*This pattern discovery guide provides comprehensive techniques for extracting meaningful archaeological patterns from market data while preserving complete semantic context. For implementation details, see the [API Reference](API_REFERENCE.md) and [User Guide](USER_GUIDE.md).*
