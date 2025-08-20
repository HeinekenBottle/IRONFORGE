# Session Fingerprinting

Optional real-time session classification system for early pattern recognition in IRONFORGE archaeological discovery.

## What

Session Fingerprinting provides **opt-in** real-time classification of trading sessions at early completion stages (~30%). The system:

- Extracts 30-dimensional feature vectors from partial session events
- Assigns sessions to pre-trained archetypes using k-means clustering
- Generates confidence scores and predicted session characteristics
- Outputs structured sidecars without affecting canonical discovery pipeline

## Why

**Early Insight**: Predict eventual session patterns from 25-30% of session events, enabling proactive analysis strategies.

**Non-intrusive**: Completely optional system (OFF by default) that operates alongside existing discovery pipeline without schema changes.

**Structured Output**: Provides standardized JSON sidecars for downstream tools and dashboards.

## How

### 1. Training Phase (Offline)

Build clustering model from historical sessions:

```bash
# Extract fingerprints from enhanced sessions
python ironforge/learning/session_fingerprinting.py

# Build clustering library with k-means
python ironforge/learning/session_clustering.py
```

**Artifacts Generated:**
- `models/session_fingerprints/v1.0.2/kmeans_model.pkl`
- `models/session_fingerprints/v1.0.2/scaler.pkl`
- `models/session_fingerprints/v1.0.2/cluster_stats.json`
- `models/session_fingerprints/v1.0.2/metadata.json`

### 2. Classification Phase (Online)

Enable during discovery/report pipeline:

```python
from ironforge.learning.online_session_classifier import create_online_classifier

# Create classifier (OFF by default)
classifier = create_online_classifier(
    enabled=True,  # Opt-in activation
    model_path=Path("models/session_fingerprints/v1.0.2"),
    completion_threshold=30.0,  # 30% session completion
    distance_metric="euclidean",  # Distance calculation method
    confidence_method="softmax"  # Confidence scoring method
)

# Classify partial session at 30% completion
prediction = classifier.classify_partial_session(
    session_data, 
    session_id="NY_AM_2025-07-29",
    target_completion_pct=30.0
)

# Write sidecar to run directory
if prediction:
    sidecar_path = classifier.write_sidecar(prediction, run_dir)
```

## Configuration

### Flag Activation

**Default State**: `enabled=False` (no impact on existing workflows)

**Activation Methods:**
- Configuration parameter: `enabled=True`
- CLI flag: `--session-fingerprinting` (when implemented)
- Environment variable: `IRONFORGE_SESSION_FINGERPRINTING=1`

### Model Artifacts

**Location**: `models/session_fingerprints/v{version}/`

**Required Files:**
- `kmeans_model.pkl` - Trained k-means clustering model
- `scaler.pkl` - Feature normalization scaler (StandardScaler/RobustScaler)
- `cluster_stats.json` - Archetype statistics and characteristics
- `metadata.json` - Model metadata and feature definitions

**Version Management**: Models versioned independently from main codebase (v1.0.2, v1.0.3, etc.)

### Completion Thresholds

**Default**: 30% session completion
**Range**: 25-50% (configurable)
**Minimum Events**: 5 events required for classification

### Distance Metrics

**Supported**: 
- `"euclidean"` (recommended, fastest)
- `"cosine"` (angular similarity)
- `"mahalanobis"` (statistical distance)

### Confidence Methods

**Supported**:
- `"inverse_distance"` (intuitive 0-1 scale)
- `"softmax"` (probabilistic distribution)

## Inputs

### Session Data Structure

```json
{
  "events": [
    {
      "timestamp": "2025-07-29T14:35:00",
      "event_type": "expansion",
      "price": 23162.25,
      "intensity": 0.8,
      "htf_regime": 1
    }
  ],
  "session_metadata": {
    "session_type": "NY_AM",
    "date": "2025-07-29",
    "start_time": "14:30:00",
    "timezone": "UTC"
  }
}
```

### Feature Vector (30D)

1. **Semantic Phase Rates** (6D): Per-event-type rates per 100 events
2. **HTF Regime Distribution** (3D): Proportion of events in regimes {0,1,2}
3. **Range/Tempo Features** (8D): Session volatility and movement characteristics
4. **Timing Features** (8D): Time-based patterns and distributions
5. **Event Distribution** (5D): Event density and clustering patterns

## Outputs

### Sidecar JSON Schema

**File**: `{run_directory}/session_fingerprint.json`

```json
{
  "session_id": "NY_AM_2025-07-29",
  "date": "2025-08-19",
  "pct_seen": 28.8,
  "archetype_id": 1,
  "confidence": 0.321,
  "predicted_stats": {
    "volatility_class": "medium",
    "range_p50": -0.36,
    "dominant_htf_regime": 1,
    "top_phases": [
      ["reversal_signal", -0.44],
      ["liquidity_sweep", -0.61]
    ],
    "session_type_probabilities": {
      "london": 0.2,
      "premarket": 0.24,
      "ny_am": 0.12
    }
  },
  "artifact_path": "models/session_fingerprints/v1.0.2",
  "notes": ["Early classification at 28.8% completion"],
  "classification_metadata": {
    "distance_to_centroid": 19.31,
    "distance_metric": "euclidean",
    "confidence_method": "softmax",
    "model_version": "1.0.2",
    "timestamp": "2025-08-19T21:20:45.123456"
  }
}
```

### Predicted Characteristics

**Volatility Classification**: Low/medium/high based on historical archetype behavior
**Range Predictions**: P50 normalized range estimates
**HTF Regime Affinity**: Dominant regime patterns for archetype
**Phase Sequences**: Top semantic event patterns with occurrence rates
**Session Type Probabilities**: Distribution over session types (asia, london, ny_am, etc.)

## Failure Modes

### Hard-Fail Scenarios

**Missing Model Artifacts**
```
FileNotFoundError: Model path does not exist: models/session_fingerprints/v1.0.2
Please ensure the offline library has been built (Stage 2) before enabling online classification.
```

**Corrupted Models**
```
RuntimeError: Failed to load clustering artifacts: corrupted pickle file
Please verify the offline library build completed successfully.
```

**Dimension Mismatch**
```
ValueError: Feature dimension mismatch: expected 30D, got 17D
Please rebuild offline library with matching feature extraction configuration.
```

### Graceful Degradation

**Insufficient Events**: Returns `None` when session has <5 events for classification

**Non-finite Features**: NaN imputation applied automatically, logged as warning

**Covariance Estimation Failure**: Falls back to Euclidean distance for Mahalanobis method

**Timestamp Parsing Errors**: Uses event order if timestamps unavailable

### Error Recovery

**Check Model Artifacts**:
```bash
ls -la models/session_fingerprints/v1.0.2/
# Should show: cluster_stats.json, kmeans_model.pkl, metadata.json, scaler.pkl
```

**Rebuild Offline Library**:
```bash
python ironforge/learning/session_clustering.py
```

**Verify Feature Compatibility**:
```python
from ironforge.learning.session_fingerprinting import SessionFingerprintExtractor
extractor = SessionFingerprintExtractor()
print(f"Feature dimensions: {len(extractor._define_feature_names())}")  # Should be 30
```

## Integration Patterns

### Discovery Pipeline Integration

```python
# At 30% session completion checkpoint
if classifier and classifier.config.enabled:
    prediction = classifier.classify_partial_session(session_data, session_id)
    if prediction:
        classifier.write_sidecar(prediction, run_dir)
        logger.info(f"Session classified: archetype {prediction.archetype_id}, confidence {prediction.confidence:.3f}")
```

### Minidash Integration

```python
# Optional dashboard row insertion
if prediction and classifier.config.write_minidash_row:
    classifier.write_minidash_row(prediction, minidash_path)
```

### A/B Testing Framework

The system includes comprehensive A/B testing for optimization:
- **Distance Metrics**: Euclidean vs Cosine vs Mahalanobis
- **Completion Thresholds**: 25% vs 30% timing
- **Confidence Methods**: Inverse distance vs Softmax
- **Scaler Types**: StandardScaler vs RobustScaler

## Schema Compatibility

**No Breaking Changes**: Session fingerprinting operates as sidecar system with zero impact on canonical IRONFORGE schemas.

**Contracts Preserved**: All existing contracts tests (45/51/20 dimensions) pass unchanged.

**Backward Compatibility**: System can be enabled/disabled without affecting existing workflows.

## Performance Characteristics

**Artifact Loading**: ~1-2 seconds startup cost (one-time)
**Classification**: <10ms per session at 30% completion
**Memory Usage**: ~15MB for loaded clustering models
**Scaling**: Linear with number of concurrent sessions

## Release Assets Policy

Session fingerprinting model artifacts are **not included** in release distributions due to size and optional nature.

**Production Deployment**:
1. Build offline library from historical data
2. Version and store artifacts separately
3. Configure deployment to reference artifact location
4. Enable classification via configuration flag

**Development Setup**:
```bash
# Build local model artifacts
python ironforge/learning/session_clustering.py
ls models/session_fingerprints/  # Verify artifacts present
```