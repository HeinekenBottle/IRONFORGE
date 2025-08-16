# IRONFORGE Semantic Feature Retrofit Implementation Plan

## Executive Summary

This plan implements a semantic feature retrofit for the IRONFORGE archaeological discovery system to preserve semantic market events and session context that are currently being lost during the TGAT processing pipeline. The enhancement transforms generic numerical patterns into rich contextual discoveries.

## Current System Analysis

### Data Flow Architecture (Current)
```
Level 1 JSON → Enhanced Graph Builder → TGAT → Pattern Discovery
     ↓                    ↓              ↓           ↓
Rich semantic      37D feature      Generic     Numerical
events (PD,       vectors +        attention   patterns like
FVG, phases)      17D edges        weights     "23.3% dup"
```

### Feature Vector Structure (Current)
**RichNodeFeature (37D)**:
- Temporal Context: 12 features (time_minutes, daily_phase_sin/cos, etc.)
- Temporal Cycles: 3 features (week_of_month, month_of_year, day_of_week_cycle)
- Price Relativity: 7 features (normalized_price, pct_from_open, etc.)
- Price Context Legacy: 3 features (price_delta_1m/5m/15m)
- Market State: 7 features (volatility_window, fisher_regime, etc.)
- Event & Structure: 8 features (event_type_id, timeframe_source, etc.)

**RichEdgeFeature (17D)**:
- Temporal Relationships: 4 features
- Relationship Type & Semantics: 5 features
- Cross-Scale & Multi-TF Hierarchy: 4 features
- Archaeological Discovery: 4 features

### Semantic Event Loss Points (Identified)
1. **PD Array Events**: Lost after Level 1 JSON → Graph Builder conversion
2. **Session Phase Context**: Generic phase encoding (expansion_phase → session_character: int)
3. **FVG Semantic Events**: Event types reduced to event_type_id without context
4. **Liquidity Event Context**: Rich context reduced to liquidity_type: int
5. **Session Metadata**: session_name, timestamps lost in final pattern output

### Output Schema (Current - Generic)
```json
{
  "edge_id": "discovery_session_10_1_0_0",
  "edge_type": "temporal_pattern_discovered",
  "confidence_score": 0.519,
  "attention_weight": 0.277,
  "discovery_metadata": {
    "predicted_edge_features": [1986239.125, -9829691.0, ...]
  }
}
```

## Implementation Tasks

### Task 1 — Feature Vector Retrofit
**Objective**: Modify RichNodeFeature to preserve semantic market events

**Current Issue**: 
- event_type_id: int (generic encoding)
- Missing explicit PD array event preservation
- Session phase reduced to session_character: int

**Implementation**:
```python
# Add to RichNodeFeature class
EVENT_TYPES = ["fvg_redelivery", "expansion_phase", "consolidation", "liq_sweep", "pd_array_interaction"]

# New semantic features (expand from 37D to 42D)
fvg_redelivery_flag: float      # 0.0 or 1.0
expansion_phase_flag: float     # 0.0 or 1.0  
consolidation_flag: float       # 0.0 or 1.0
liq_sweep_flag: float          # 0.0 or 1.0
pd_array_strength: float       # 0.0-1.0 strength score

# Session phase preservation (one-hot encoded)
phase_open: float              # 0.0 or 1.0
phase_mid: float               # 0.0 or 1.0
phase_close: float             # 0.0 or 1.0
```

**Files to Modify**:
- `/Users/jack/IRONPULSE/IRONFORGE/learning/enhanced_graph_builder.py` (RichNodeFeature class)

**Success Criteria**:
- 42D feature vector (37D + 5 semantic events)
- Semantic events preserved from Level 1 JSON
- Session phase context maintained

### Task 2 — Edge Feature Enrichment
**Objective**: Add event-based edge labels for semantic relationships

**Current Issue**:
- relation_type: int (generic encoding)
- Missing semantic edge context
- No event-to-event relationship labeling

**Implementation**:
```python
# Add to RichEdgeFeature class
# Expand from 17D to 20D
semantic_event_link: int       # 0=none, 1=fvg_chain, 2=pd_sequence, 3=phase_transition
event_causality: float        # 0.0-1.0 causal strength
semantic_label: str           # "fvg_redelivery_link", "expansion_to_consolidation"

# Update relation_type mapping
RELATION_TYPES = {
    0: "temporal", 1: "scale", 2: "cascade", 3: "pd_array", 
    4: "discovered", 5: "confluence", 6: "echo", 7: "fvg_chain", 
    8: "phase_transition", 9: "liquidity_sweep"
}
```

**Files to Modify**:
- `/Users/jack/IRONPULSE/IRONFORGE/learning/enhanced_graph_builder.py` (RichEdgeFeature class)
- Update TGAT to handle 20D edge features

**Success Criteria**:
- 20D edge feature vector with semantic labels
- Event-based edge relationships preserved
- Semantic labels converted to vector form for TGAT

### Task 3 — Session Context Preservation
**Objective**: Extend build_rich_graph() to attach session metadata

**Current Issue**:
- Session metadata lost during graph construction
- No session_name, timestamps in final output
- Missing anchor_timeframe context

**Implementation**:
```python
# Add to build_rich_graph() method
def build_rich_graph(self, session_data: Dict[str, Any]) -> Tuple[Graph, Dict[str, Any]]:
    # ... existing graph building ...
    
    # Extract and preserve session metadata
    session_metadata = {
        "session_name": session_data.get("session_metadata", {}).get("session_type", "unknown"),
        "session_start_time": session_data.get("session_metadata", {}).get("session_start", "00:00:00"),
        "session_end_time": session_data.get("session_metadata", {}).get("session_end", "00:00:00"),
        "session_date": session_data.get("session_metadata", {}).get("session_date", "unknown"),
        "anchor_timeframe": self._determine_anchor_timeframe(session_data),
        "session_duration": session_data.get("session_metadata", {}).get("session_duration", 0)
    }
    
    return rich_graph, session_metadata
```

**Files to Modify**:
- `/Users/jack/IRONPULSE/IRONFORGE/learning/enhanced_graph_builder.py` (build_rich_graph method)
- `/Users/jack/IRONPULSE/IRONFORGE/learning/tgat_discovery.py` (to accept metadata)

**Success Criteria**:
- Session metadata flows through to pattern output
- No loss of contextual information
- Anchor timeframe determination implemented

### Task 4 — Constant Feature Filtering
**Objective**: Detect features where variance = 0 across all nodes

**Current Issue**:
- TGAT trains on constant features (open_time, close_time)
- Wastes computational resources
- May interfere with pattern discovery

**Implementation**:
```python
def analyze_feature_variance(self, node_features: List[RichNodeFeature]) -> Dict[str, Any]:
    """Detect constant features and mark as metadata-only"""
    
    feature_matrix = torch.stack([node.to_tensor() for node in node_features])
    variances = torch.var(feature_matrix, dim=0)
    
    constant_features = []
    variable_features = []
    
    for i, variance in enumerate(variances):
        if variance < 1e-6:  # Effectively constant
            constant_features.append(i)
        else:
            variable_features.append(i)
    
    return {
        "constant_feature_indices": constant_features,
        "variable_feature_indices": variable_features,
        "total_features": len(variances),
        "constant_count": len(constant_features)
    }

def filter_features_for_tgat(self, features: torch.Tensor, 
                           variance_analysis: Dict[str, Any]) -> torch.Tensor:
    """Filter out constant features for TGAT training"""
    variable_indices = variance_analysis["variable_feature_indices"]
    return features[:, variable_indices]
```

**Files to Modify**:
- `/Users/jack/IRONPULSE/IRONFORGE/learning/enhanced_graph_builder.py`
- `/Users/jack/IRONPULSE/IRONFORGE/learning/tgat_discovery.py`

**Success Criteria**:
- Automatic detection of constant features
- TGAT trains only on variable features
- Constant features preserved in metadata for output

### Task 5 — Output Schema Upgrade
**Objective**: Update TGAT discovery output to include rich context

**Current Generic Output**:
```json
{
  "edge_type": "temporal_pattern_discovered",
  "confidence_score": 0.519,
  "discovery_metadata": {
    "predicted_edge_features": [numbers...]
  }
}
```

**Target Rich Output**:
```json
{
  "pattern_id": "htf_fvg_redelivery_chain",
  "session_name": "NY_AM", 
  "session_start": "2025-08-05T12:00:00Z",
  "session_end": "2025-08-05T15:00:00Z",
  "anchor_timeframe": "Daily",
  "linked_event": "first_presented_fvg_redelivery",
  "phase": "expansion_phase",
  "confidence": 0.92,
  "semantic_context": {
    "source_event_type": "fvg_redelivery",
    "target_event_type": "pd_array_interaction", 
    "relationship_type": "causal_sequence",
    "temporal_distance": "45m",
    "price_distance": "78% range"
  }
}
```

**Implementation**:
```python
def enrich_discovery_output(self, discovery: Dict[str, Any], 
                          session_metadata: Dict[str, Any],
                          source_node: RichNodeFeature,
                          target_node: RichNodeFeature) -> Dict[str, Any]:
    """Transform generic discovery into rich semantic output"""
    
    # Generate semantic pattern ID
    pattern_id = self._generate_semantic_pattern_id(source_node, target_node)
    
    # Extract semantic context
    semantic_context = self._extract_semantic_context(source_node, target_node, discovery)
    
    # Build rich output
    enriched_discovery = {
        "pattern_id": pattern_id,
        "session_name": session_metadata["session_name"],
        "session_start": f"{session_metadata['session_date']}T{session_metadata['session_start_time']}Z",
        "session_end": f"{session_metadata['session_date']}T{session_metadata['session_end_time']}Z",
        "anchor_timeframe": session_metadata["anchor_timeframe"],
        "linked_event": self._get_primary_event_type(source_node),
        "phase": self._get_session_phase(source_node),
        "confidence": discovery["confidence_score"],
        "semantic_context": semantic_context,
        "original_discovery": discovery  # Preserve original for debugging
    }
    
    return enriched_discovery
```

**Files to Modify**:
- `/Users/jack/IRONPULSE/IRONFORGE/learning/tgat_discovery.py`
- `/Users/jack/IRONPULSE/IRONFORGE/orchestrator.py`

**Success Criteria**:
- Rich semantic context in all pattern outputs
- Session context preserved
- Human-readable pattern descriptions
- Maintains backward compatibility with existing discoveries

## Implementation Dependencies

### Task Order (Sequential)
1. **Task 1** (Feature Vector Retrofit) - Foundation for all other tasks
2. **Task 2** (Edge Feature Enrichment) - Depends on Task 1 node features  
3. **Task 3** (Session Context Preservation) - Can run parallel to Tasks 1-2
4. **Task 4** (Constant Feature Filtering) - Depends on Tasks 1-2
5. **Task 5** (Output Schema Upgrade) - Depends on all previous tasks

### Critical Dependencies
- **Iron-Core Integration**: Ensure compatibility with existing lazy loading (4.7s SLA)
- **TGAT Architecture**: Preserve 4-head attention mechanism
- **Backward Compatibility**: Existing 1,411 pattern discoveries must remain valid
- **Performance**: Maintain <5s initialization time

## Testing Approach

### Unit Testing (Per Task)
1. **Task 1**: Verify 42D feature vectors contain semantic events
2. **Task 2**: Validate 20D edge features with semantic labels
3. **Task 3**: Confirm session metadata flows through pipeline
4. **Task 4**: Test constant feature detection accuracy
5. **Task 5**: Validate rich output schema generation

### Integration Testing
1. **End-to-End Pipeline**: Level 1 JSON → Rich Semantic Output
2. **Performance Validation**: Maintain 4.7s total processing time
3. **Backward Compatibility**: Existing pattern discovery still works
4. **Semantic Accuracy**: Manual validation of 10 patterns for correctness

### Success Criteria

#### Functional Success
- ✅ Transform "23.3% duplication" → "NY_AM session, expansion_phase, fvg_redelivery_chain"
- ✅ Preserve all semantic events from Level 1 JSON
- ✅ Session context in final output
- ✅ Zero-variance features filtered from TGAT training

#### Performance Success  
- ✅ Maintain <5s total processing time
- ✅ TGAT 4-head attention preserved
- ✅ Lazy loading performance maintained (iron-core integration)

#### Quality Success
- ✅ 100% backward compatibility with existing discoveries
- ✅ Human-readable semantic pattern descriptions
- ✅ No loss of mathematical accuracy

## Risk Mitigation

### High Risk
1. **TGAT Dimension Mismatch**: 37D→42D and 17D→20D changes may break attention
   - **Mitigation**: Update input/output projections incrementally
   
2. **Performance Degradation**: Additional features may slow processing
   - **Mitigation**: Profile each task, optimize constant feature filtering

### Medium Risk  
1. **Semantic Event Extraction**: Complex parsing from Level 1 JSON
   - **Mitigation**: Robust fallbacks, comprehensive testing
   
2. **Edge Case Handling**: Missing or malformed semantic data
   - **Mitigation**: Graceful degradation to generic patterns

### Low Risk
1. **Output Schema Compatibility**: New format may confuse downstream systems
   - **Mitigation**: Include original_discovery field for compatibility

## Resource Requirements

### Development Time Estimate
- **Task 1**: 3-4 hours (Feature Vector Retrofit)
- **Task 2**: 2-3 hours (Edge Feature Enrichment)  
- **Task 3**: 2-3 hours (Session Context Preservation)
- **Task 4**: 1-2 hours (Constant Feature Filtering)
- **Task 5**: 3-4 hours (Output Schema Upgrade)
- **Testing & Integration**: 2-3 hours

**Total Estimated Time**: 13-19 hours

### Technical Requirements
- Python 3.8+ with PyTorch 1.9+
- Existing IRONFORGE infrastructure
- Iron-core compatibility maintained
- 57 enhanced session files for testing

## Expected Outcomes

### Before Implementation (Current)
```json
{
  "edge_type": "temporal_pattern_discovered",
  "confidence_score": 0.519,
  "discovery_metadata": {"predicted_edge_features": [numbers...]}
}
```

### After Implementation (Target)  
```json
{
  "pattern_id": "ny_am_expansion_fvg_redelivery_chain",
  "session_name": "NY_AM",
  "session_start": "2025-08-05T12:00:00Z", 
  "anchor_timeframe": "Daily",
  "linked_event": "first_presented_fvg_redelivery",
  "phase": "expansion_phase",
  "confidence": 0.92,
  "semantic_context": {
    "source_event_type": "fvg_redelivery",
    "temporal_distance": "45m after session open",
    "causal_relationship": "expansion_phase_trigger"
  }
}
```

This transformation achieves the core objective: **preserve semantic market events and session context** to enable meaningful archaeological discovery of market relationships rather than generic numerical patterns.