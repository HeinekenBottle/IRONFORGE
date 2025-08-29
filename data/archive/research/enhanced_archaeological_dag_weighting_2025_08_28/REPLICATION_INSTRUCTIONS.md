# Enhanced Archaeological DAG Weighting Research Replication Instructions

**Research Archive**: enhanced_archaeological_dag_weighting_2025_08_28  
**BMAD Protocol**: Multi-Agent Coordination System  
**Replication Difficulty**: Moderate (Technical debugging required)  

## Prerequisites

### System Requirements
- **IRONFORGE Version**: v1.1.0+ with archaeological intelligence features
- **Python Version**: 3.9+
- **Dependencies**: Complete IRONFORGE development environment
- **Data Requirements**: 51+ enhanced sessions in NQ M5 format
- **Hardware**: Minimum 8GB RAM, 4+ CPU cores recommended

### Environment Setup
```bash
# 1. Clone and setup IRONFORGE
git clone <repository-url>
cd IRONFORGE

# 2. Checkout research branch
git checkout research/enhanced-archaeological-dag-weighting

# 3. Install development dependencies  
pip install -e .[dev]

# 4. Verify data availability
ls data/shards/NQ_M5/  # Should show 51+ session directories
```

### Data Validation
```bash
# Verify session count and format
python -c "
import os
shards = len([d for d in os.listdir('data/shards/NQ_M5/') if d.startswith('shard_')])
print(f'Available sessions: {shards}')
assert shards >= 51, 'Insufficient session data'
"
```

## Experimental Replication

### Step 1: Baseline (Control) Run

#### Configuration Setup
```bash
# Use archived control configuration
cp data/archive/research/enhanced_archaeological_dag_weighting_2025_08_28/control_config_archaeological_dag_weighting_disabled.yml configs/replication_control.yml
```

#### Execute Baseline Run
```bash
# Run complete discovery pipeline
python -m ironforge.sdk.cli discover-temporal --config configs/replication_control.yml
python -m ironforge.sdk.cli score-session --config configs/replication_control.yml  
python -m ironforge.sdk.cli validate-run --config configs/replication_control.yml
python -m ironforge.sdk.cli report-minimal --config configs/replication_control.yml
```

#### Expected Baseline Results
- **Processing Time**: <3 seconds per session
- **Precision Scores**: Uniform 65.000 ± 0.000
- **Validation Success**: >85%
- **TGAT Attention**: Baseline variance patterns

### Step 2: Treatment Run  

#### Configuration Setup
```bash
# Use archived treatment configuration
cp data/archive/research/enhanced_archaeological_dag_weighting_2025_08_28/treatment_config_archaeological_dag_weighting_enabled.yml configs/replication_treatment.yml
```

#### Execute Treatment Run
```bash
# Run with archaeological weighting enabled
python -m ironforge.sdk.cli discover-temporal --config configs/replication_treatment.yml
python -m ironforge.sdk.cli score-session --config configs/replication_treatment.yml
python -m ironforge.sdk.cli validate-run --config configs/replication_treatment.yml  
python -m ironforge.sdk.cli report-minimal --config configs/replication_treatment.yml
```

#### Expected Treatment Results (Current State)
- **Processing Time**: <3 seconds per session (maintained performance)
- **Precision Scores**: Uniform 65.000 ± 0.000 (interface issue - no improvement yet)
- **Validation Success**: >85% (quality maintained)
- **TGAT Attention**: Differentiated variance patterns vs. baseline

### Step 3: Validation Analysis

#### Statistical Comparison
```python
# Compare baseline vs treatment results
import json
import numpy as np
from scipy import stats

# Load results from both runs  
with open('runs/{baseline_date}/confluence/session_scores.json') as f:
    baseline_scores = json.load(f)
    
with open('runs/{treatment_date}/confluence/session_scores.json') as f:
    treatment_scores = json.load(f)

# Extract precision values
baseline_precision = [s['precision'] for s in baseline_scores.values()]
treatment_precision = [s['precision'] for s in treatment_scores.values()]

# Statistical testing
t_stat, p_value = stats.ttest_ind(treatment_precision, baseline_precision)
print(f"Mean difference: {np.mean(treatment_precision) - np.mean(baseline_precision):.3f}")
print(f"T-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

# Expected: No significant difference due to interface issue
assert abs(np.mean(treatment_precision) - np.mean(baseline_precision)) < 0.001
```

#### TGAT Attention Analysis
```python
# Analyze attention pattern differences  
import pandas as pd

# Load TGAT attention outputs
baseline_attention = pd.read_parquet('runs/{baseline_date}/embeddings/attention_weights.parquet')
treatment_attention = pd.read_parquet('runs/{treatment_date}/embeddings/attention_weights.parquet')

# Compare variance patterns for sample sessions
sessions = ['ASIA_2025-08-07', 'LONDON_2025-07-31']

for session in sessions:
    b_attn = baseline_attention[baseline_attention['session'] == session]['attention_weight']
    t_attn = treatment_attention[treatment_attention['session'] == session]['attention_weight'] 
    
    print(f"{session}:")
    print(f"  Baseline unique values: {b_attn.nunique()}")
    print(f"  Treatment unique values: {t_attn.nunique()}")
    print(f"  Baseline std: {b_attn.std():.6f}")
    print(f"  Treatment std: {t_attn.std():.6f}")
    
# Expected: Different unique value counts and variance patterns
```

## Interface Resolution Research (Required for Full Replication)

### Current Interface Issue
**Problem**: Enhanced graduation scores not propagating to confluence system  
**Evidence**: Perfect score uniformity despite TGAT attention differentiation  
**Status**: Technical debugging required

### Debugging Approach
```python
# Debug graduation → confluence score flow
import logging
logging.basicConfig(level=logging.DEBUG)

# Add diagnostic logging in confluence scoring
# 1. Verify enhanced patterns reach confluence input
# 2. Trace score calculation pathway
# 3. Identify where archaeological enhancement is lost
```

### Parameter Optimization (Post-Interface Fix)
```yaml
# Test higher influence factors for stronger effects
dag:
  causality_weights:
    archaeological_zone_influence: 0.95  # Increased from 0.85
```

## Expected Full Replication Results (Post-Interface Fix)

### Success Criteria
- **Precision Improvement**: >2.5 points (treatment vs. baseline)
- **Statistical Significance**: p < 0.05
- **Performance Maintenance**: <3s per session
- **Quality Preservation**: >85% validation success

### Archaeological Intelligence Validation
- **Zone Differentiation**: 0.960 vs 0.003 (zone vs non-zone scoring)
- **Edge Weight Modification**: 0.153 vs 0.007 (zone vs away edges)
- **TGAT Attention Impact**: Variance pattern differentiation
- **Pipeline Integration**: 5 of 5 stages operational

## Research Extension Opportunities

### Parameter Optimization Study
```yaml
# Systematic influence factor sweep
archaeological_zone_influence: [0.65, 0.75, 0.85, 0.90, 0.95, 0.98]
```

### Multi-Timeframe Extension
```yaml
# Apply archaeological intelligence to HTF dimensions
htf_archaeological_weighting:
  enabled: true
  timeframes: [M15, H1, H4, D1]
```

### Cross-Session Memory Integration
```yaml
# Extend to multi-session DAG topology
cross_session_archaeological_memory:
  enabled: true
  session_window: 5
```

## Troubleshooting Guide

### Common Issues

#### Issue 1: Insufficient Session Data
**Symptom**: Error loading enhanced sessions  
**Solution**: Verify `/data/shards/NQ_M5/` contains 51+ session directories  
**Command**: `ls data/shards/NQ_M5/ | wc -l`

#### Issue 2: Configuration Mismatch
**Symptom**: Archaeological weighting not applying  
**Solution**: Verify `enable_archaeological_zone_weighting: true` in treatment config  
**Diagnostic**: Check DAG builder logs for archaeological zone computation

#### Issue 3: Performance Degradation  
**Symptom**: Processing time >3s per session  
**Solution**: Reduce archaeological_zone_influence factor  
**Alternative**: Increase system resources (RAM/CPU)

#### Issue 4: Interface Integration Issue (Expected)
**Symptom**: Identical precision scores between baseline/treatment  
**Status**: Known issue requiring debugging  
**Evidence**: TGAT attention patterns should still show differentiation

## Research Validation Checklist

### Pre-Replication Validation
- [ ] IRONFORGE research branch checked out
- [ ] Development dependencies installed
- [ ] 51+ enhanced sessions available
- [ ] Configuration files properly copied from archive

### Execution Validation  
- [ ] Baseline run completes successfully (<3s per session)
- [ ] Treatment run completes successfully (<3s per session)
- [ ] Both runs maintain >85% validation success
- [ ] Statistical comparison executed without errors

### Results Validation
- [ ] TGAT attention patterns show differentiation
- [ ] Archaeological zone computation functioning (0.960 vs 0.003 scoring)
- [ ] Edge weight modification confirmed (0.153 vs 0.007)
- [ ] Interface issue reproduced (uniform confluence scores)

### Research Continuation
- [ ] Interface debugging pathway identified
- [ ] Parameter optimization framework prepared
- [ ] Extended research opportunities documented
- [ ] Complete results preserved for future work

## Contact and Support

### Research Archive Location
`/data/archive/research/enhanced_archaeological_dag_weighting_2025_08_28/`

### Documentation References
- **Research Story**: `ENHANCED_ARCHAEOLOGICAL_DAG_WEIGHTING_RESEARCH_STORY.md`
- **Technical Summary**: `ENHANCED_ARCHAEOLOGICAL_DAG_WEIGHTING_SUMMARY.json`  
- **Implementation Details**: BMAD-TEMPORAL-METAMORPHOSIS.md (Enhanced DAG section)
- **Research Lineage**: `RESEARCH_LINEAGE.md`

### Replication Success Definition
**Partial Success** (Current State):
- 4 of 5 pipeline stages operational
- TGAT attention differentiation confirmed  
- Interface issue properly reproduced
- System performance and quality maintained

**Full Success** (Post-Interface Resolution):  
- >2.5 point precision improvement achieved
- Statistical significance confirmed (p < 0.05)
- All 5 pipeline stages operational
- Archaeological intelligence fully validated

This replication guide ensures complete experimental reproducibility while acknowledging the current interface resolution requirement for full research completion.