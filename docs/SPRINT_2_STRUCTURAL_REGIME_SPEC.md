# SPRINT 2: Structural & Regime Intelligence Expansion Specification

## 🎯 Mission Statement
Expand IRONFORGE from temporal cycle detection (37D) to comprehensive structural intelligence with regime segmentation and event precursor detection, maintaining the NO FALLBACKS policy and archaeological discovery mission.

---

## 🏗️ Architecture Overview

### Current Foundation (Sprint 1 Complete)
- **37D Node Features**: 12 temporal (inc. 3 cycles) + 7 relativity + 18 other features
- **3 Edge Types**: temporal, scale, discovered (via TGAT)
- **Validation System**: Strict data integrity with fail-fast behavior
- **Performance**: Clean sessions processed with 1,041 patterns discovered

### Sprint 2 Enhancements
- **4th Edge Type**: `structural_context` connecting related archetypes
- **Regime Intelligence**: Auto-clustering of patterns into market regimes  
- **Event Precursors**: Confidence-scored precursor detection using temporal cycles
- **Unified Schema**: All datasets normalized to consistent 37D structure

---

## 📋 Innovation Architect (IA) Specifications

### **Task IA-1: Structural Context Edges**
**File**: `learning/enhanced_graph_builder.py`

#### **Node Archetype Classification**
```python
def _classify_node_archetype(self, node_feature: RichNodeFeature, 
                           session_context: Dict) -> str:
    """
    Classify node into structural archetype for context edge creation
    
    Archetypes:
    - "first_fvg_after_sweep": First FVG following liquidity sweep
    - "htf_range_midpoint": Price level at HTF range midpoint  
    - "session_boundary": Open/close price levels
    - "liquidity_cluster": High liquidity concentration areas
    - "imbalance_zone": Price imbalance/inefficiency areas
    - "cascade_origin": Starting point of price cascades
    - "structural_support": Key support/resistance levels
    """
```

#### **Structural Context Edge Builder**
```python
def _build_structural_context_edges(self, graph: Dict, 
                                   node_indices: List, archetypes: List) -> None:
    """
    Build structural_context edges between related archetypes
    
    Context Relationships:
    - sweep → first_fvg_after_sweep (causal sequence)
    - imbalance_zone → htf_range_midpoint (structural alignment)
    - cascade_origin → session_boundary (boundary interaction)
    - liquidity_cluster → structural_support (reinforcement)
    """
```

### **Task IA-2: Regime Segmentation**
**File**: `learning/regime_segmentation.py` (NEW)

#### **Pattern Clustering System**
```python
class RegimeSegmentation:
    def __init__(self, clustering_method='DBSCAN'):
        self.clusterer = DBSCAN(eps=0.3, min_samples=5)  # Tunable parameters
    
    def segment_patterns(self, tgat_patterns: List[Dict]) -> Dict[str, Any]:
        """
        Auto-cluster discovered patterns into market regimes
        
        Returns:
        - regime_labels: Dict mapping pattern_id -> regime_label
        - regime_characteristics: Dict describing each regime
        - confidence_scores: Cluster stability metrics
        """
```

#### **Feature Vector Extraction**
```python
def extract_regime_features(self, pattern: Dict) -> np.ndarray:
    """
    Extract features from TGAT patterns for regime clustering
    
    Features (12D):
    - Pattern temporal characteristics (4D): duration, frequency, periodicity, temporal_spread
    - Pattern structural characteristics (4D): price_range, volatility, node_count, edge_density  
    - Pattern relativity characteristics (4D): avg_normalized_price, relativity_consistency, htf_alignment, cycle_alignment
    """
```

### **Task IA-3: Event Precursor Detection**
**File**: `learning/precursor_detection.py` (NEW)

#### **Precursor Pattern Matching**
```python
class EventPrecursorDetector:
    def detect_precursors(self, session_graph: Dict, 
                         temporal_cycles: Dict) -> Dict[str, float]:
        """
        Detect event precursor patterns using temporal cycles + structural context
        
        Criteria:
        1. Temporal Alignment: week_of_month + day_of_week_cycle matching
        2. Structural Confluence: sweep → imbalance → htf_level sequence
        3. Relativity Consistency: normalized_price and htf_ratio alignment
        
        Returns:
        - precursor_index: Dict with confidence scores per event type
        - expected_timeframe: Projected cascade timing
        - contributing_factors: Feature importance breakdown
        """
```

---

## 🔧 Technical Debt Surgeon (TDS) Specifications

### **Task TDS-1: Schema Normalization**
**File**: `data_migration/schema_normalizer.py` (NEW)

#### **Migration System**
```python
class SchemaNormalizer:
    def migrate_legacy_to_37d(self, session_data: Dict) -> Dict:
        """
        Migrate legacy 34D data to current 37D schema
        
        Process:
        1. Validate existing features (27D base + 7D relativity = 34D)
        2. Add temporal cycle features with validated defaults:
           - week_of_month: Calculate from session_date
           - month_of_year: Extract from session_date  
           - day_of_week_cycle: Match existing day_of_week
        3. Comprehensive post-migration validation
        4. NO FALLBACKS: Fail if data cannot be cleanly migrated
        """
```

### **Task TDS-2: Extended Validation**
**File**: `test_strict_validation.py` (EXTEND)

#### **New Validation Scenarios**
```python
# Test structural_context edge validation
def test_structural_context_edges():
    """Validate 4th edge type integration"""

# Test regime label validation  
def test_regime_labels():
    """Validate regime labels as graph attributes"""

# Test precursor index validation
def test_precursor_index():
    """Validate precursor index data types and ranges"""
```

### **Task TDS-3: Performance Monitoring**
**File**: `performance_monitor.py` (NEW)

#### **System Performance Tracking**
```python
class PerformanceMonitor:
    def monitor_37d_plus_structural(self, processing_metrics: Dict) -> Dict:
        """
        Monitor enhanced system performance vs baseline
        
        Metrics:
        - Processing time: 37D + 4 edge types vs current 37D + 3 edge types
        - Memory usage: Node/edge storage overhead
        - Pattern discovery quality: Regime-tagged vs non-regime patterns
        - Validation success rate: Maintain 100% accuracy standard
        """
```

### **Task TDS-4: Analyst Reporting**
**File**: `reporting/analyst_reports.py` (NEW)

#### **Visibility Layer**
```python
class AnalystReports:
    def generate_session_report(self, session_results: Dict) -> Dict:
        """
        Generate comprehensive session analysis
        
        Reports:
        1. Edge Type Distribution: temporal/scale/structural_context/discovered counts
        2. Pattern Analysis: Top contributing features per pattern
        3. Regime Analysis: Regime distribution and characteristics
        4. Precursor Analysis: Detected precursor events with confidence
        """
```

---

## 🔗 Integration Points

### **Edge Type System (4 Types)**
```python
EDGE_TYPES = {
    'temporal': 'Sequential time-based connections within timeframe',
    'scale': 'Cross-timeframe connections (1m->5m->15m)',
    'structural_context': 'Archetype-based structural relationships', # NEW
    'discovered': 'TGAT auto-discovered pattern connections'
}
```

### **Graph Metadata Enhancement**
```python
enhanced_metadata = {
    'feature_dimensions': 37,  # Maintained from Sprint 1
    'edge_types': ['temporal', 'scale', 'structural_context', 'discovered'],  # 4 types
    'regime_label': 'consolidation_breakout',  # NEW: Auto-assigned regime
    'precursor_index': {  # NEW: Event precursor confidence scores
        'cascade_probability': 0.73,
        'breakout_probability': 0.45,
        'reversal_probability': 0.12
    },
    'archetype_distribution': {  # NEW: Node archetype counts
        'first_fvg_after_sweep': 3,
        'htf_range_midpoint': 2,
        'liquidity_cluster': 5
    }
}
```

---

## 📊 Success Metrics

### **Technical Validation**
- **Feature Compatibility**: 100% backward compatibility with 37D temporal cycles
- **Edge System**: 4 edge types operational with proper classification
- **Regime Segmentation**: >80% cluster stability score
- **Precursor Detection**: Confidence scores within [0,1] range with proper calibration
- **Performance**: <15% regression vs current 37D system

### **Data Quality**
- **Schema Consistency**: 100% migration success for clean legacy data  
- **Validation Accuracy**: Maintain 100% strict validation success rate
- **Integration Tests**: All components work together without fallbacks

### **Archaeological Discovery Enhancement**
- **Pattern Quality**: Regime-tagged patterns show >20% better coherence
- **Structural Intelligence**: Archetype-based edges reveal previously hidden relationships
- **Precursor Accuracy**: Temporal cycle + structural confluence improves event prediction

---

## 🚀 Development Timeline

### **Phase 1: Foundation (Days 1-2)**
- **IA**: Implement structural archetype classification
- **TDS**: Create schema migration system
- **Coordination**: Update this specification with implementation details

### **Phase 2: Intelligence Systems (Days 3-4)**
- **IA**: Regime segmentation and precursor detection
- **TDS**: Extended validation and performance monitoring
- **Coordination**: Integration testing and refinement

### **Phase 3: Integration & Validation (Days 5-6)**
- **IA**: TGAT pipeline integration for 4 edge types
- **TDS**: Analyst reporting and comprehensive system validation
- **Coordination**: Final testing and documentation

---

## 🔄 Agent Coordination Protocol

### **Communication**
- **Shared Updates**: Both agents update this document with implementation progress
- **Integration Points**: Coordinate on edge type system, metadata schema, validation standards
- **Performance Impact**: TDS monitors IA's additions, IA considers TDS feedback

### **Quality Gates**
- **NO FALLBACKS**: Both agents maintain strict fail-fast behavior
- **Backward Compatibility**: All Sprint 1 functionality preserved
- **Archaeological Mission**: Enhanced discovery capabilities, not reduced complexity

### **Git Workflow**
- **Branch**: `sprint_2_dev` for all Sprint 2 work
- **Commits**: Clear separation of IA (features) vs TDS (quality) commits
- **Integration**: Regular coordination commits updating shared specifications

---

**Status: 🚀 READY FOR IMPLEMENTATION**  
**Next Update**: Implementation progress from both agents