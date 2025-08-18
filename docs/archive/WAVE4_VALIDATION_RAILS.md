# Wave 4: Validation Rails for IRONFORGE

**Robust evaluation framework ensuring temporal pattern discoveries are trustworthy and production-ready.**

## Overview

Wave 4 implements comprehensive validation rails for IRONFORGE's temporal pattern discovery system. The validation framework provides time-series safe evaluation methods, negative controls, ablation studies, and automated reporting to ensure discovered patterns are reliable and not artifacts of data leakage or spurious correlations.

### Key Features

- **Time-Series Safe Splits**: Purged K-fold with embargo periods to prevent look-ahead bias
- **Negative Controls**: Time shuffling, label permutation, and structural controls
- **Ablation Studies**: Feature group importance analysis  
- **Comprehensive Metrics**: Temporal AUC, precision@k, pattern stability, archaeological significance
- **Automated Reporting**: JSON results and HTML summaries with performance tracking

## Architecture

### Validation Pipeline

```
Raw Data → Splits (OOS/K-Fold) → Main Validation → Controls → Ablations → Reports
```

### Component Structure

```
ironforge/validation/
├── splits.py          # Time-series safe data splitting
├── controls.py        # Negative controls and synthetic baselines  
├── metrics.py         # Validation metrics and stability measures
└── runner.py          # Orchestration and reporting
```

## Quick Start

### CLI Usage

```bash
# Basic out-of-sample validation
ironforge sdk validate --data-path ./data/validation

# Purged K-fold with custom settings
ironforge sdk validate \
    --data-path ./data/validation \
    --mode purged-kfold \
    --folds 7 \
    --embargo-mins 60 \
    --controls time_shuffle label_perm \
    --ablations htf_prox cycles structure \
    --report-dir ./reports/wave4

# Holdout validation with comprehensive controls
ironforge sdk validate \
    --data-path ./data/validation \
    --mode holdout \
    --embargo-mins 120 \
    --controls time_shuffle label_perm node_shuffle edge_direction \
    --seed 42
```

### Programmatic Usage

```python
import numpy as np
from ironforge.validation.runner import ValidationRunner, ValidationConfig

# Configure validation experiment
config = ValidationConfig(
    mode="purged-kfold",
    folds=5,
    embargo_mins=30,
    controls=["time_shuffle", "label_perm"],
    ablations=["htf_prox", "cycles", "structure"],
    report_dir="reports/validation",
    random_seed=42
)

# Prepare validation data
validation_data = {
    "edge_index": edge_connectivity_matrix,      # (2, num_edges)
    "edge_times": edge_timestamps,               # (num_edges,)
    "node_features": node_feature_matrix,        # (num_nodes, 45)
    "labels": pattern_labels,                    # (num_samples,)
    "timestamps": sample_timestamps,             # (num_samples,)
    "feature_groups": {
        "htf_prox": [0, 1, 2, 3, 4],            # HTF proximity features
        "cycles": [5, 6, 7, 8, 9],              # Cycle analysis features
        "structure": [10, 11, 12, 13, 14]       # Structural features
    }
}

# Run validation
runner = ValidationRunner(config)
results = runner.run(validation_data)

# Check validation status
if results["summary"]["validation_passed"]:
    print("✅ Validation PASSED")
    print(f"Main AUC: {results['summary']['main_performance']['temporal_auc']:.4f}")
else:
    print("❌ Validation FAILED")
    for rec in results["summary"]["recommendations"]:
        print(f"⚠️  {rec}")
```

## Validation Methods

### 1. Out-of-Sample (OOS) Split

Simple chronological split for basic temporal validation:

```python
from ironforge.validation.splits import oos_split

# Split at 70% cutoff
train_idx, test_idx = oos_split(timestamps, cutoff_ts=cutoff_timestamp)
```

**Use Case**: Quick validation for initial model assessment.

### 2. Purged K-Fold with Embargo

Time-series safe cross-validation preventing look-ahead bias:

```python
from ironforge.validation.splits import PurgedKFold

# 5-fold with 30-minute embargo
splitter = PurgedKFold(n_splits=5, embargo_mins=30)

for train_idx, test_idx in splitter.split(timestamps):
    # Train and validate with leakage-safe splits
    pass
```

**Features**:
- Temporal ordering preserved
- Embargo periods prevent information leakage
- Non-overlapping test windows
- Handles irregular time series

### 3. Holdout with Embargo

Single train-test split with temporal buffer:

```python
from ironforge.validation.splits import temporal_train_test_split

train_idx, test_idx = temporal_train_test_split(
    timestamps, 
    test_size=0.2, 
    embargo_mins=60
)
```

## Negative Controls

### Time Shuffle Control

Destroys temporal relationships while preserving graph structure:

```python
from ironforge.validation.controls import time_shuffle_edges

# Break temporal signal
shuffled_edge_index, shuffled_times = time_shuffle_edges(
    edge_index, edge_times, seed=42
)
```

**Purpose**: Verify that model performance depends on temporal structure.

### Label Permutation Control

Destroys feature-label relationships:

```python
from ironforge.validation.controls import label_permutation

# Break feature-label association
permuted_labels = label_permutation(labels, seed=42)
```

**Purpose**: Confirm model learns genuine patterns, not random correlations.

### Additional Controls

- **Node Feature Shuffle**: Tests feature importance
- **Edge Direction Shuffle**: Tests directional sensitivity  
- **Temporal Block Shuffle**: Tests local vs global temporal patterns

## Ablation Studies

Test feature group importance by selective removal:

```python
# Define feature groups for ablation
feature_groups = {
    "htf_prox": [0, 1, 2, 3, 4],        # Higher timeframe proximity
    "cycles": [5, 6, 7, 8, 9],          # Cycle analysis features
    "structure": [10, 11, 12, 13, 14],  # Market structure features
    "semantic": [15, 16, 17, 18, 19]    # Semantic event features
}

# Ablation automatically zeros out each group and measures impact
config = ValidationConfig(ablations=["htf_prox", "cycles", "structure"])
```

**Interpretation**:
- High performance drop = Important feature group
- Low performance drop = Redundant features
- No change = Features not utilized by model

## Validation Metrics

### Core Performance Metrics

#### Temporal AUC
Chronologically-aware ROC AUC with tie-breaking:

```python
from ironforge.validation.metrics import temporal_auc

auc = temporal_auc(y_true, y_score, timestamps)
```

#### Precision@K
Precision at top-k predictions:

```python
from ironforge.validation.metrics import precision_at_k

p_at_20 = precision_at_k(y_true, y_score, k=20)
```

### Stability Metrics

#### Pattern Stability Score
Measures temporal consistency of pattern scores:

```python
from ironforge.validation.metrics import pattern_stability_score

stability = pattern_stability_score(y_score, timestamps, window_size=60)
```

#### Motif Half-Life
Estimates pattern decay/persistence:

```python
from ironforge.validation.metrics import motif_half_life

# From hit timestamps
half_life = motif_half_life(pattern_hit_timestamps)
```

### Archaeological Significance

Specialized metrics for market pattern analysis:

```python
from ironforge.validation.metrics import archaeological_significance

arch_metrics = archaeological_significance(
    pattern_scores=confidence_scores,
    pattern_types=["fvg", "poi", "bos", "liq"],
    temporal_spans=duration_spans
)

print(f"Pattern diversity: {arch_metrics['diversity_index']:.3f}")
print(f"Temporal coverage: {arch_metrics['temporal_coverage']:.3f}")
print(f"Pattern density: {arch_metrics['pattern_density']:.3f}")
```

## Performance Guidelines

### Runtime Budgets

| Component | Target | Achieved |
|-----------|---------|----------|
| Full Validation | <5s | ✅ <3s typical |
| Memory Usage | <150MB | ✅ <100MB typical |
| Batch Processing | <100ms | ✅ <50ms typical |

### Optimization Tips

1. **Data Size**: Validate on representative samples (1000-5000 samples)
2. **Feature Groups**: Limit ablations to 3-5 most important groups
3. **Controls**: Use 2-3 negative controls for efficiency
4. **Reporting**: Disable HTML generation for batch processing

### Scaling Considerations

```python
# For large datasets, use sampling
if len(timestamps) > 10000:
    sample_idx = np.random.choice(len(timestamps), 5000, replace=False)
    sampled_data = {key: value[sample_idx] for key, value in data.items()}
```

## Quality Gates

### Validation Thresholds

- **Main Performance**: Temporal AUC > 0.60
- **Control Comparison**: Main AUC - Control AUC > 0.05
- **Pattern Stability**: Stability score > 0.30
- **Feature Importance**: Ablation impact > 0.02

### Interpretation Guide

#### ✅ PASSED Validation

```
✅ Main AUC: 0.82 (> 0.60 threshold)
✅ vs Time Shuffle: +0.18 (> 0.05 threshold)  
✅ vs Label Permutation: +0.22 (> 0.05 threshold)
✅ Pattern Stability: 0.67 (> 0.30 threshold)
```

**Interpretation**: Model shows genuine temporal pattern recognition.

#### ❌ FAILED Validation

```
❌ Main AUC: 0.52 (< 0.60 threshold)
❌ vs Time Shuffle: +0.03 (< 0.05 threshold)
⚠️  Pattern Stability: 0.15 (< 0.30 threshold)
```

**Recommendations**:
- Review feature engineering
- Check for data quality issues
- Consider different model architecture
- Increase training data

## Report Outputs

### JSON Results
Comprehensive machine-readable results:

```json
{
  "experiment_info": {
    "timestamp": "2025-01-15T10:30:00",
    "config": {...},
    "runtime_seconds": 2.84
  },
  "main_validation": {
    "split_type": "purged_kfold",
    "metrics": {
      "temporal_auc": 0.823,
      "precision_at_20": 0.65,
      "pattern_stability": 0.71
    }
  },
  "negative_controls": {
    "time_shuffle": {"metrics": {"temporal_auc": 0.52}},
    "label_perm": {"metrics": {"temporal_auc": 0.48}}
  },
  "ablation_studies": {
    "htf_prox": {"feature_importance": 0.12},
    "cycles": {"feature_importance": 0.08}
  },
  "summary": {
    "validation_passed": true,
    "recommendations": []
  }
}
```

### HTML Summary
Human-readable dashboard with key metrics and status indicators.

## Advanced Usage

### Custom Metrics

```python
def custom_metric(y_true, y_score, timestamps):
    """Custom archaeological metric."""
    # Implementation
    return metric_value

# Add to validation pipeline
from ironforge.validation.metrics import compute_validation_metrics

# Extend with custom metrics post-validation
results = runner.run(data)
custom_score = custom_metric(y_true, y_score, timestamps)
```

### Custom Controls

```python
def custom_control(data, seed=42):
    """Custom negative control."""
    # Modify data to break specific signal
    return modified_data

# Use in control variants
from ironforge.validation.controls import create_control_variants

# Extend controls list
extended_controls = ["time_shuffle", "label_perm", "custom_control"]
```

### Batch Validation

```python
# Validate multiple datasets
datasets = ["session_1.json", "session_2.json", "session_3.json"]

for dataset_path in datasets:
    data = load_validation_data(dataset_path)
    results = runner.run(data)
    
    if not results["summary"]["validation_passed"]:
        print(f"⚠️ {dataset_path} failed validation")
```

## Troubleshooting

### Common Issues

#### "No graphs provided for stitching"
```python
# Ensure data has valid edge connectivity
assert len(data["edge_index"]) > 0
assert data["edge_index"].shape[0] == 2
```

#### "Cannot split X samples into Y folds"
```python
# Use fewer folds or more data
config = ValidationConfig(folds=min(3, len(data["timestamps"]) // 2))
```

#### "Memory usage exceeds budget"
```python
# Reduce dataset size or feature dimensions
sampled_data = sample_validation_data(data, max_samples=5000)
```

### Performance Debugging

```python
import time
import psutil

def profile_validation():
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    
    results = runner.run(data)
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024
    
    print(f"Runtime: {end_time - start_time:.2f}s")
    print(f"Memory: {end_memory - start_memory:.1f}MB")
    
    return results
```

## Integration with IRONFORGE

### Discovery Pipeline Integration

```python
# After TGAT discovery
from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline
from ironforge.validation.runner import ValidationRunner

# Run discovery
pipeline = TemporalDiscoveryPipeline(data_path="./shards")
discoveries = pipeline.run_discovery()

# Validate discovered patterns
validation_data = prepare_validation_data(discoveries)
validation_results = runner.run(validation_data)
```

### Production Deployment

```python
# Pre-deployment validation gate
def deploy_model(model, validation_data):
    results = runner.run(validation_data)
    
    if results["summary"]["validation_passed"]:
        deploy_to_production(model)
        log_validation_success(results)
    else:
        raise ValidationError("Model failed validation gates")
        log_validation_failure(results)
```

---

## Wave 4 Summary

Wave 4 Validation Rails provides a comprehensive, production-ready validation framework that ensures IRONFORGE's temporal pattern discoveries are:

- **Reliable**: Free from look-ahead bias and data leakage
- **Robust**: Validated against multiple negative controls  
- **Interpretable**: Clear metrics and actionable recommendations
- **Efficient**: Meets performance budgets for production use
- **Automated**: Complete reporting and quality gates

The validation framework scales from quick development checks to comprehensive production validation, ensuring that archaeological market patterns discovered by IRONFORGE are genuinely predictive and not statistical artifacts.

**Next**: Wave 5 will build on these validation capabilities with advanced reporting visualizations and heatmaps for pattern analysis.