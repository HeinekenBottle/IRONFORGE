# Oracle Temporal Non-locality Training System

The Oracle system enables IRONFORGE to predict session ranges from early temporal events using calibrated neural networks. This document covers the complete Oracle training pipeline, usage, and integration.

## Overview

The Oracle system addresses a fundamental limitation in temporal market analysis: predicting session extremes from limited early data. Through **temporal non-locality**, Oracle can forecast where a session will complete its range based on just the first 20% of events.

### Key Capabilities

- **Cold-start Mode**: Uncalibrated predictions using Xavier weights (0.5 confidence, NaN ranges)
- **Calibrated Mode**: Trained predictions with quantitative accuracy metrics
- **Temporal Non-locality**: Early events contain forward-looking information about session completion
- **Production Ready**: Complete CLI pipeline with validation, metrics, and version control

## Architecture

### Core Components

1. **TGAT Discovery Engine**: 45D/51D node features → 44D embeddings via temporal attention
2. **Range Head**: 44D → 32D → 2D neural network (center, half_range) prediction
3. **Training Pipeline**: Data normalization → embedding computation → supervised learning
4. **Calibration System**: Weights + scaler persistence with validation

### Training Flow

```
Raw Sessions → Audit → Normalize → Build Embeddings → Train → Evaluate → Deploy
```

### Files Generated

- `weights.pt` - Trained range_head parameters
- `scaler.parquet` - Target normalization (StandardScaler in Parquet format) 
- `training_manifest.json` - Complete training metadata and validation
- `metrics.json` - Comprehensive evaluation results

## Command Line Interface

### Basic Usage

Train Oracle on NQ 5-minute data:

```bash
python -m ironforge.sdk.cli train-oracle \
  --symbols NQ \
  --tf 5 \
  --from 2025-05-01 \
  --to 2025-08-15 \
  --early-pct 0.20 \
  --out models/oracle/v1.0.2
```

### Full Parameter Set

```bash
python -m ironforge.sdk.cli train-oracle \
  --symbols NQ,ES \
  --tf 5 \
  --from 2025-05-01 \
  --to 2025-08-15 \
  --early-pct 0.20 \
  --htf-context \
  --out models/oracle/v1.0.2 \
  --rebuild \
  --data-dir data/enhanced \
  --max-sessions 1000
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--symbols` | Yes | - | Comma-separated symbols (NQ,ES) |
| `--tf` | Yes | - | Timeframe (5 for M5) |
| `--from` | Yes | - | Start date (YYYY-MM-DD) |
| `--to` | Yes | - | End date (YYYY-MM-DD) |
| `--early-pct` | No | 0.20 | Early batch percentage (0-0.5) |
| `--htf-context` | No | False | Enable 51D features vs 45D |
| `--out` | Yes | - | Output model directory |
| `--rebuild` | No | False | Force rebuild embeddings |
| `--data-dir` | No | data/enhanced | Session data directory |
| `--max-sessions` | No | None | Limit training sessions |

## Pipeline Stages

### Stage 1: Session Audit

**Purpose**: Validate data quality and schema consistency

**Script**: `scripts/audit_sessions.py`

**Output**: 
- Schema validation report
- Data quality assessment  
- Missing field identification
- Coverage statistics

**Usage**:
```bash
python scripts/audit_sessions.py --data-dir data/enhanced --symbol NQ --tf M5 --output audit.json
```

### Stage 2: Data Normalization  

**Purpose**: Convert diverse session formats to canonical training rows

**Script**: `scripts/normalize_sessions.py`

**Canonical Schema**:
```
symbol, tf, session_date, start_ts, end_ts,
final_high, final_low, center, half_range, 
htf_mode, n_events
```

**Features**:
- Timezone-aware timestamp parsing
- OHLC computation from events when missing
- Data quality scoring (excellent/good/fair/poor)
- Filtering by quality threshold

**Usage**:
```bash
python scripts/normalize_sessions.py \
  --data-dir data/enhanced \
  --symbol NQ \
  --tf M5 \
  --quality good \
  --output normalized_sessions.parquet
```

### Stage 3: Training Data Builder

**Purpose**: Recompute early embeddings using TGAT forward pass

**Script**: `oracle/data_builder.py`

**Process**:
1. Load normalized sessions
2. Build graphs via `EnhancedGraphBuilder` 
3. Extract early subgraph (20% of events)
4. Run TGAT forward pass with attention
5. Attention-weighted pooling → 44D embedding
6. Pair with ground truth ranges

**Output Schema**:
```
symbol, tf, session_date, htf_mode, early_pct,
pooled_00, pooled_01, ..., pooled_43,  # 44D embedding
target_center, target_half_range
```

**Usage**:
```bash  
python oracle/data_builder.py \
  --sessions normalized_sessions.parquet \
  --output training_pairs.parquet \
  --early-pct 0.20 \
  --enhanced-dir data/enhanced
```

### Stage 4: Model Training

**Purpose**: Train range_head (44→32→2) with frozen TGAT

**Script**: `oracle/trainer.py`

**Training Strategy**:
- Freeze all TGAT attention layers  
- Train only `range_head` parameters
- StandardScaler normalization on targets
- Early stopping with validation monitoring
- Gradient clipping for stability

**Hyperparameters**:
- Learning rate: 0.001
- Batch size: 32  
- Max epochs: 100
- Patience: 15 epochs
- Weight decay: 1e-5

**Usage**:
```bash
python oracle/trainer.py \
  --training-data training_pairs.parquet \
  --model-dir models/oracle/v1.0.2 \
  --epochs 100 \
  --lr 0.001 \
  --batch-size 32
```

### Stage 5: Model Evaluation

**Purpose**: Comprehensive performance assessment

**Script**: `oracle/eval.py`

**Metrics**:
- Center prediction: MAE, RMSE, MAPE
- Range prediction: MAE, RMSE, MAPE  
- Direction accuracy
- Percentile error analysis
- Breakdown by symbol/timeframe/confidence

**Quality Assessment**:
- Excellent: >80% range accuracy
- Good: >70% range accuracy  
- Fair: >60% range accuracy
- Poor: <60% range accuracy

**Usage**:
```bash
python oracle/eval.py \
  --model-dir models/oracle/v1.0.2 \
  --training-data training_pairs.parquet
```

## Integration with Discovery

### Loading Calibrated Weights

```python
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

discovery = IRONFORGEDiscovery()
success = discovery.load_calibrated_oracle_weights("models/oracle/v1.0.2")

if success:
    print("✅ Oracle calibrated mode enabled")
else:
    print("⚠️  Oracle running in cold-start mode") 
```

### Configuration File

Create `config.yaml`:
```yaml
oracle:
  enabled: true
  early_pct: 0.20
  weights_dir: models/oracle/v1.0.2
```

### Discovery with Oracle

```bash
python -m ironforge.sdk.cli discover-temporal \
  --symbol NQ --tf 5 \
  --out runs/$(date +%F)/NQ_5m \
  --config config.yaml
```

**Output**: `oracle_predictions.parquet` (16 columns, Schema v0)

## Schema v0 Compliance

Oracle predictions follow the exact 16-column schema:

```
run_dir, session_date, pct_seen, n_events,
pred_low, pred_high, center, half_range, confidence,  
pattern_id, start_ts, end_ts,
early_expansion_cnt, early_retracement_cnt, early_reversal_cnt, first_seq
```

## File Structure

```
models/oracle/v1.0.2/
├── weights.pt                 # Trained range_head parameters
├── scaler.parquet            # StandardScaler for target normalization  
├── training_manifest.json    # Complete training metadata
├── metrics.json              # Evaluation results
├── audit_report.json         # Data quality report
├── normalized_sessions.parquet  # Canonical session format
├── training_pairs.parquet    # 44D embeddings + targets
└── training_pairs_metadata.json  # Training data summary
```

## Training Manifest

Complete reproducibility metadata:

```json
{
  "version": "v1.0.2",
  "timestamp": "2025-08-19T12:00:00Z",
  "git_sha": "abc123...",
  "model_architecture": {
    "input_dim": 44,
    "hidden_dim": 32, 
    "output_dim": 2,
    "layers": ["Linear(44->32)", "ReLU", "Linear(32->2)"]
  },
  "training_config": {
    "symbols": ["NQ", "ES"],
    "timeframes": ["M5"],
    "date_range": {"start": "2025-05-01", "end": "2025-08-15"},
    "early_pct": 0.20,
    "training_samples": 1247,
    "hyperparameters": {...}
  },
  "model_validation": {
    "embedding_dim": 44,
    "expected_input_shape": [44],
    "expected_output_shape": [2]
  }
}
```

## Performance Metrics

Example evaluation results:

```json
{
  "metrics": {
    "overall": {
      "center_mae": 4.2,
      "center_rmse": 6.8,
      "center_mape": 8.3,
      "range_mae": 2.1,
      "range_rmse": 3.4,
      "range_mape": 12.7,
      "range_direction_accuracy": 0.73
    },
    "by_symbol": {
      "NQ": {"range_mape": 11.2, "center_mae": 3.8},
      "ES": {"range_mape": 14.1, "center_mae": 4.6}  
    }
  }
}
```

## Version Control

### Versioning Strategy

- **v1.0.0**: Initial Oracle implementation
- **v1.0.1**: Bug fixes and optimizations  
- **v1.0.2**: Complete training CLI + validation (current)

### Model Compatibility  

Each training manifest includes:
- Git SHA for exact code version
- Dimension validation (44D input, 2D output)
- Architecture fingerprint
- Training data hash

### Release Process

1. Train model: `train-oracle --out models/oracle/v1.0.2`
2. Validate: Check metrics.json quality assessment  
3. Test: Run smoke tests on hold-out data
4. Tag: `git tag v1.0.2`
5. Archive: Save training artifacts

## Testing

### Unit Tests

```bash
python -m pytest tests/oracle/ -v
```

### Smoke Test

```bash  
# Manual smoke test
python -m ironforge.sdk.cli train-oracle \
  --symbols NQ --tf 5 \
  --from 2025-08-15 --to 2025-08-17 \
  --out models/oracle/smoke-test \
  --max-sessions 10
```

### CI/CD Pipeline

GitHub Action: `.github/workflows/oracle-train-smoke.yml`

- Runs on Oracle-related file changes
- Tests across Python 3.9, 3.10, 3.11
- Validates CLI argument parsing
- Smoke test with mock data
- Integration point verification

## Troubleshooting

### Common Issues

**"No session data found"**
- Verify `--data-dir` points to enhanced sessions
- Check file naming patterns match filters
- Run audit first to validate data availability

**"Architecture mismatch"**  
- TGAT hidden_dim must equal Oracle input_dim (44)
- Range_head output must be 2D (center, half_range)
- Check training manifest compatibility

**"Training fails immediately"**
- Insufficient data: Need >10 sessions minimum
- Poor data quality: Filter by `--quality good`
- Memory issues: Reduce `--batch-size` or `--max-sessions`

**"Poor evaluation metrics"**
- Low data quality: Improve session filtering
- Insufficient diversity: Train on multiple symbols/dates
- Overfitting: Reduce epochs or add regularization

### Performance Optimization

**Large Datasets**:
- Use `--max-sessions` to limit training size
- Batch normalization in chunks
- Monitor memory usage during embedding computation

**Training Speed**:
- Increase `--batch-size` if memory allows
- Use GPU if available (auto-detected)
- Reduce `--epochs` for faster iteration

**Prediction Accuracy**:
- Increase training data diversity (more symbols/dates)
- Tune `--early-pct` (0.15-0.25 range)
- Filter by higher data quality thresholds

## Best Practices

### Data Preparation
- Always run audit before normalization
- Filter sessions by quality (minimum "fair")
- Validate date ranges cover sufficient diversity
- Check symbol distribution in training data

### Model Training
- Use version-controlled output directories
- Save training artifacts for reproducibility
- Monitor validation loss convergence
- Compare metrics across different configurations

### Production Deployment
- Validate model dimensions before loading
- Test Oracle predictions on hold-out data
- Monitor calibrated vs cold-start performance
- Implement automated retraining workflows

### Continuous Improvement
- Retrain periodically with fresh data
- A/B test different early_pct values
- Experiment with multi-symbol training
- Track prediction accuracy over time

## Theory: Temporal Non-locality

The Oracle system demonstrates that early session events contain **forward-looking information** about eventual session completion. This temporal non-locality effect enables:

### Archaeological Zone Prediction

Early events position themselves relative to the **final** session range, not the range created so far:

- **40% Zone Event** at 14:35:00 → Session Low at 14:53:00 (18 minutes future)
- **Distance Accuracy**: 7.55 points to final 40% vs 30.80 points to current 40%
- **Temporal Non-locality**: Events "know" their eventual structural position

### Practical Implications

1. **Early Range Prediction**: 20% of events predict 80% completion accuracy
2. **Structural Destiny**: Archaeological zones represent relationships to final structure  
3. **Forward-looking Markets**: Session events contain information about future extremes

This theoretical foundation underpins the Oracle's predictive capability and explains why early temporal embeddings can forecast session completion ranges.

---

*Oracle Training System v1.0.2 - IRONFORGE Temporal Non-locality Pipeline*