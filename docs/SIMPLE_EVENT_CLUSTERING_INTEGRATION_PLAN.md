# Simple Event-Time Clustering + Cross-TF Mapping Integration Plan
## IRONFORGE Archaeological Discovery System Enhancement

### Executive Summary

This document outlines the integration of Simple Event-Time Clustering and Cross-Timeframe (TF) Mapping capabilities into the existing IRONFORGE archaeological discovery system. The integration is designed to be **non-invasive** and **performance-preserving**, adding time pattern intelligence without disrupting the current 47D semantic feature processing pipeline.

### Current System Status
- **Architecture**: Enhanced graph builder → TGAT discovery → Pattern graduation  
- **Features**: 47D node features (37D + 7 semantic events + 3 session phases)
- **Performance**: 3.153s per session (meeting <5s SLA)
- **Capability**: Semantic event discovery (FVG redelivery, expansion phases, etc.)
- **Integration**: Iron-core dependency injection with lazy loading

### Enhancement Objectives

1. **Time Pattern Intelligence**: Add "when events cluster" analysis  
2. **Cross-TF Context**: Add "what HTF context" for LTF events
3. **Non-Breaking**: Preserve all existing functionality and performance
4. **Minimal Footprint**: <0.05s overhead per session
5. **Rich Output**: Enhance session_metadata with time_patterns

### Integration Architecture

#### New Component: `simple_event_clustering.py`
Location: `/Users/jack/IRONPULSE/IRONFORGE/learning/simple_event_clustering.py`

**Key Classes:**
- `EventTimeClusterer`: Time-bin based clustering of events
- `CrossTFMapper`: Maps LTF events to HTF structural context  
- `SimpleEventAnalyzer`: Orchestrates both analyses

**Design Principles:**
- **Read-only**: Analyzes existing graph data without modification
- **Time-focused**: Clusters by when events occur, not what they are
- **HTF-aware**: Maps lower timeframe events to higher timeframe structure
- **Performance-optimized**: <50ms processing time per session

#### Minimal Orchestrator Integration
Location: `/Users/jack/IRONPULSE/IRONFORGE/orchestrator.py`

**Changes Required (3 minimal modifications):**
1. **Import**: Add `from learning.simple_event_clustering import analyze_time_patterns`
2. **Method**: Add `_analyze_time_patterns()` helper method  
3. **Integration**: 3-line addition to `process_sessions()` loop

**Integration Point**: After graph building, before TGAT format conversion

### Technical Specifications

#### EventTimeClusterer
```python
class EventTimeClusterer:
    def __init__(self, time_bin_minutes: int = 5):
        self.time_bin_minutes = time_bin_minutes
    
    def cluster_events_by_time(self, events: List[Dict]) -> Dict:
        # Groups events into time bins
        # Returns: {bin_id: [events], cluster_stats: {...}}
```

**Input**: List of events with timestamps
**Output**: Time-binned clusters with density metrics
**Performance**: O(n log n) sorting + O(n) binning

#### CrossTFMapper  
```python
class CrossTFMapper:
    def map_ltf_to_htf(self, ltf_events: List[Dict], htf_data: Dict) -> List[Dict]:
        # Maps each LTF event to corresponding HTF structural context
        # Returns: Events with HTF context (phase, structure, confluence)
```

**Input**: LTF events + HTF structural data
**Output**: Events enriched with HTF context  
**Performance**: O(n) linear mapping

#### Integration Output
**Enhanced session_metadata structure:**
```json
{
  "time_patterns": {
    "event_clusters": [
      {
        "time_bin": "09:30-09:35",
        "event_count": 8,
        "density_score": 0.75,
        "dominant_events": ["fvg_redelivery", "expansion_phase"],
        "htf_context": {
          "15m_phase": "consolidation_break",
          "1h_structure": "uptrend_continuation",
          "daily_context": "london_open_drive"
        }
      }
    ],
    "cross_tf_mapping": {
      "ltf_to_15m": [...],
      "ltf_to_1h": [...],
      "structural_alignments": [...]
    },
    "clustering_stats": {
      "total_bins": 108,
      "active_bins": 23,
      "max_density": 0.89,
      "temporal_distribution": "front_loaded"
    }
  }
}
```

### Implementation Plan

#### Phase 1: Core Implementation (30 minutes)
1. **Create `simple_event_clustering.py`** - Implement all classes
2. **Add helper methods** - Time binning, HTF mapping utilities  
3. **Unit testing** - Validate clustering and mapping logic

#### Phase 2: Orchestrator Integration (15 minutes)
1. **Add import statement** - `from learning.simple_event_clustering import analyze_time_patterns`
2. **Create helper method** - `_analyze_time_patterns(graph, session_file)`
3. **Integrate into processing loop** - 3-line addition after graph building

#### Phase 3: Validation (15 minutes)
1. **Create test script** - Validate integration with existing sessions
2. **Performance testing** - Ensure <0.05s overhead per session
3. **Output validation** - Verify time_patterns data quality

### Code Integration Points

#### orchestrator.py Integration
**Location**: Line ~138 (after graph preservation, before TGAT conversion)

```python
# EXISTING CODE
# Preserve graph
self._preserve_graph(graph, session_file)
results['graphs_preserved'].append(session_file)

# NEW INTEGRATION (3 lines)
time_patterns = self._analyze_time_patterns(graph, session_file)
if 'time_patterns' not in metadata:
    metadata['time_patterns'] = time_patterns

# EXISTING CODE
print(f"✅ Processed {Path(session_file).name}: {metadata.get('total_nodes', 0)} nodes")
```

**New helper method**:
```python
def _analyze_time_patterns(self, graph: Dict, session_file: str) -> Dict:
    """Analyze time patterns in session events with cross-TF mapping"""
    from learning.simple_event_clustering import analyze_time_patterns
    return analyze_time_patterns(graph, session_file)
```

### Performance Impact Analysis

#### Expected Overhead
- **Time binning**: ~10ms (sorting + grouping)
- **HTF mapping**: ~20ms (context lookup)  
- **Result formatting**: ~5ms (JSON serialization)
- **Total overhead**: ~35ms per session (<0.05s target ✓)

#### Memory Impact
- **Additional data**: ~5KB per session (time_patterns metadata)
- **Processing memory**: ~1MB temporary (released after processing)
- **Total impact**: Negligible (<1% increase)

### Data Flow Integration

#### Current Flow
```
Session JSON → Enhanced Graph Builder → Graph Preservation → TGAT Format → Discovery
```

#### Enhanced Flow  
```
Session JSON → Enhanced Graph Builder → Graph Preservation → **Time Pattern Analysis** → TGAT Format → Discovery
```

**Key Points:**
- Time pattern analysis occurs **after** graph building (data available)
- Occurs **before** TGAT conversion (metadata can be included)
- **Non-blocking**: Errors in time analysis don't break main pipeline

### Quality Assurance

#### Validation Criteria
1. **Functionality**: All existing tests continue to pass
2. **Performance**: Processing time increase <5% per session
3. **Data integrity**: session_metadata preserves all existing fields
4. **Error handling**: Time pattern failures don't break main pipeline

#### Test Coverage
1. **Unit tests**: EventTimeClusterer and CrossTFMapper classes
2. **Integration tests**: Full orchestrator with time pattern analysis
3. **Performance tests**: Benchmark overhead measurement
4. **Regression tests**: Ensure existing functionality unchanged

### Risk Mitigation

#### Potential Risks
1. **Performance degradation**: Mitigated by <50ms processing limit
2. **Memory leaks**: Mitigated by explicit cleanup in analyzer
3. **Integration conflicts**: Mitigated by read-only analysis approach
4. **Data corruption**: Mitigated by separate time_patterns namespace

#### Fallback Strategy
- **Graceful degradation**: Time pattern analysis failures logged but don't break session processing
- **Feature flags**: Can disable time pattern analysis via configuration
- **Rollback capability**: Simple import removal reverts to original behavior

### Success Metrics

#### Immediate Success (Post-Integration)
- ✅ All existing IRONFORGE tests pass
- ✅ Session processing time increase <5%
- ✅ time_patterns data appears in session_metadata
- ✅ No performance regression in TGAT discovery

#### Functional Success (Post-Validation)
- ✅ Event clustering identifies temporal density patterns
- ✅ Cross-TF mapping enriches events with HTF context
- ✅ Output data enables "when + what" time pattern analysis
- ✅ Integration supports immediate tactical intelligence extraction

### Implementation Timeline

**Total Estimated Time: 60 minutes**

- **30 min**: Implement `simple_event_clustering.py` with all classes
- **15 min**: Integrate with orchestrator.py (3 minimal changes)
- **15 min**: Create validation test and verify functionality

### Conclusion

This integration adds significant "when events cluster" and "what HTF context" intelligence to IRONFORGE with minimal risk and overhead. The read-only analysis approach ensures existing functionality remains unaffected while providing immediate tactical value for time pattern discovery.

The enhancement transforms IRONFORGE from event discovery to **temporal-contextual discovery**, enabling answers to:
- "When do FVG redeliveries cluster in sessions?"
- "What HTF context surrounds expansion phase events?"  
- "How do LTF events align with HTF structural phases?"

This positions IRONFORGE as a comprehensive archaeological discovery system with both event intelligence and temporal intelligence capabilities.

---
**Document Version**: 1.0  
**Target System**: IRONFORGE Archaeological Discovery System  
**Integration Type**: Non-invasive enhancement  
**Performance Target**: <0.05s overhead per session