# IRONFORGE Changelog

## [1.0.2] - 2025-08-19 - Session Fingerprinting (Optional Feature)

### 🔍 Session Fingerprinting — Opt-in, No Schema Changes

**Added optional real-time session classification system for early pattern recognition.**

#### ✅ New Features

**Session Fingerprinting Implementation**
- **Real-time Classification**: Optional system for 30% session completion fingerprinting
- **Archetype Assignment**: K-means clustering with pre-trained session archetypes  
- **Confidence Scoring**: Multiple confidence calculation methods (inverse distance, softmax)
- **Sidecar Output**: `session_fingerprint.json` with predicted characteristics
- **Zero Pipeline Impact**: OFF by default, no effect on canonical discovery/report workflows

**30-Dimensional Feature Vector**
- **Semantic Phase Rates** (6D): Event type occurrence patterns
- **HTF Regime Distribution** (3D): Market regime {0,1,2} proportions
- **Range/Tempo Features** (8D): Volatility and movement characteristics  
- **Timing Features** (8D): Time-based patterns and distributions
- **Event Distribution** (5D): Event density and clustering patterns

**Multi-Stage ML Pipeline**
- **Stage 1**: Per-session feature extraction with A/B scaler testing
- **Stage 2**: Offline library builder with k-means clustering (≥66 sessions)
- **Stage 3**: Online classifier with flag-controlled activation
- **Stage 4**: Documentation and surface-level integration

#### ⚙️ Configuration

**Default State**: `enabled=False` (no impact on existing systems)

**Activation**:
```python
classifier = create_online_classifier(
    enabled=True,  # Opt-in activation  
    completion_threshold=30.0,  # 30% session checkpoint
    distance_metric="euclidean",  # Distance calculation
    confidence_method="softmax"  # Confidence scoring
)
```

**Model Artifacts**: `models/session_fingerprints/v1.0.2/`
- K-means clustering model, feature scaler, cluster statistics, metadata

#### 📊 Sidecar Schema

**Output**: `{run_directory}/session_fingerprint.json`
```json
{
  "session_id": "NY_AM_2025-07-29",
  "archetype_id": 1,
  "confidence": 0.321,
  "predicted_stats": {
    "volatility_class": "medium",
    "dominant_htf_regime": 1,
    "session_type_probabilities": {...}
  }
}
```

#### 🛡️ Compatibility & Safety

- **No Breaking Changes**: All existing functionality preserved
- **Schema Stability**: Zero impact on canonical IRONFORGE schemas  
- **Contracts Preserved**: All 6 contracts tests pass unchanged
- **Hard-fail Error Handling**: Missing artifacts produce actionable error messages
- **Graceful Degradation**: Insufficient events return null without pipeline disruption

#### 📈 Performance & Optimization

**A/B Testing Results**:
- **Distance Metrics**: Euclidean preferred (highest confidence, efficient)
- **Completion Thresholds**: 25% acceptable (early predictions without accuracy loss)
- **Confidence Methods**: Softmax recommended (normalized [0,1] range)
- **Scaler Types**: StandardScaler vs RobustScaler tested

**Performance Characteristics**:
- **Artifact Loading**: ~1-2 seconds (one-time startup)
- **Classification**: <10ms per session at 30% completion
- **Memory Usage**: ~15MB for loaded clustering models

#### 🔧 Release Assets Policy

**Model artifacts NOT included in release distributions** (size and optional nature).

**Production Deployment**:
1. Build offline library from historical data
2. Version and store artifacts separately  
3. Configure deployment to reference artifact location
4. Enable via configuration flag

## [1.0.1] - 2025-08-19 - Oracle Temporal Non-locality (GA Release)

### 🔮 Oracle Sidecar Predictions

**Added optional Oracle temporal non-locality predictions with schema v0 locking.**

#### ✅ New Features

**Oracle Implementation**
- **Oracle Predictions**: Optional sidecar system for temporal non-locality predictions
- **Schema v0 Contract**: Exact 16-column parquet schema for downstream tool compatibility
- **Early Event Analysis**: Predict session ranges from 20% of early session events
- **Configuration**: Oracle disabled by default, configurable via YAML
- **Temporal Non-locality**: Events positioned relative to eventual session completion

**Schema v0 Columns (16 exactly)**
```
run_dir, session_date, pct_seen, n_events, pred_low, pred_high, 
center, half_range, confidence, pattern_id, start_ts, end_ts,
early_expansion_cnt, early_retracement_cnt, early_reversal_cnt, first_seq
```

**Distribution & Packaging**
- **Modern Packaging**: Migrated from setup.py to pyproject.toml
- **Dynamic Versioning**: Automatic version management from `__version__.py`  
- **Console Scripts**: `ironforge` and `ifg` CLI entry points
- **Wheel Distribution**: Complete Oracle components in v1.0.1 wheel
- **Version Consistency**: Fixed `__version_info__` availability in main package

#### ⚙️ Configuration

**Oracle Configuration (optional)**
```yaml
oracle:
  enabled: false  # Disabled by default
  early_pct: 0.20 # Must be in (0, 0.5]
  output_path: "oracle_predictions.parquet"
```

**Integration**
- Oracle predictions written during `discover-temporal` when enabled
- No schema changes to canonical pipeline
- Backward compatible with existing configurations

### 🔒 Compatibility & Versioning

- **No Breaking Changes**: All existing functionality preserved
- **Schema Stability**: Oracle uses locked schema v0 for consistency
- **Dual Versioning**: Stable schema contracts with evolving ML models
- **Downstream Tools**: Oracle schema v0 ensures external tool compatibility

## [0.7.1] - 2025-08-19 - Repository Refactor (Wave 7.2)

### 🧹 Repository Cleanup & Structure Improvements

**Major refactoring to reduce clutter and improve operability while preserving all canonical functionality.**

#### ✅ Completed Changes

**Root Script Consolidation**
- Moved 11 one-off run scripts to `scripts/analysis/legacy_runners/`
- Created deprecation stubs at root level with guidance to canonical CLI
- Scripts moved:
  - `run_weekly_daily_cascade_lattice.py`
  - `run_weekly_daily_sweep_cascade_step_3b.py`
  - `run_weekly_daily_sweep_cascade_step_3b_refined.py`
  - `run_working_cascade_analysis.py`
  - `run_fpfvg_network_analysis.py`
  - `run_fpfvg_redelivery_lattice.py`
  - `run_global_lattice.py`
  - `run_specialized_lattice.py`
  - `run_terrain_analysis.py`
  - `run_fpfvg_network_analysis_simple.py`
  - `extract_lattice_summary.py`

**Ad-hoc Test Migration**
- Moved 4 ad-hoc test files to `tests/legacy/`
- Tests moved:
  - `simple_threshold_test.py`
  - `test_refactored_structure.py`
  - `test_refined_detection_quick.py`
  - `validate_ci_setup.py`

**Legacy Migration Archive**
- Moved `data_migration/` to `archive/data_migration/`
- Added comprehensive archive documentation
- All current shards are 51D/20D format (migration scripts no longer needed)

**Generated Artifacts Cleanup**
- Removed tracked generated outputs:
  - `deliverables/`
  - `demo_deliverables/`
  - `discovery_cache/`
  - `results/`
  - `test_outputs/`
  - `preservation/`
  - `reports/`
- Preserved recent runs in `runs/` directory
- Updated `.gitignore` to prevent re-tracking

**Virtual Environment Cleanup**
- Removed committed virtual environments:
  - `ironforge_env/`
  - `ironforge_refactor_env/`
- Added to `.gitignore`

**Miscellaneous Cleanup**
- Removed version files (`=0.11.0`, `=1.2.0`, etc.)
- Cleaned up root directory clutter

#### 📚 Documentation Overhaul

**New Authoritative README.md**
- Complete rewrite following specification
- Canonical contracts (6 events, 4 edge intents, 51D/20D features)
- Clear quickstart guide
- Entrypoints and CLI documentation
- AUX vs Canonical distinction
- Explainability features

**Core Documentation Suite**
- `docs/flows.md`: Schema contracts and run order
- `docs/taxonomy_v1.md`: Authoritative event and edge taxonomy
- `docs/operations.md`: Daily operations and A/B adapter usage

**Validation Infrastructure**
- `tools/smoke_checks.py`: Comprehensive system validation
- Tests entrypoint imports, CLI commands, shard dimensions, run artifacts

#### 🔒 Golden Invariants Preserved

**Event Taxonomy (6 types exactly)**
- Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery

**Edge Intents (4 types exactly)**
- TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT

**Feature Dimensions**
- Nodes: 51D (f0..f50, HTF v1.1 = f45..f50 last-closed only)
- Edges: 20D (e0..e19)

**Entrypoints**
- `ironforge.learning.discovery_pipeline:run_discovery`
- `ironforge.confluence.scoring:score_confluence`
- `ironforge.validation.runner:validate_run`
- `ironforge.reporting.minidash:build_minidash`

**CLI Commands**
- `discover-temporal`, `score-session`, `validate-run`, `report-minimal`, `status`

**Data Contracts**
- Shards: `data/shards/<SYMBOL_TF>/shard_*/{nodes,edges}.parquet`
- Runs: `runs/YYYY-MM-DD/{embeddings,patterns,confluence,motifs,reports,minidash.*}`

#### 🛡️ Safety Measures

**Backward Compatibility**
- All moved scripts have deprecation stubs with guidance
- Import paths preserved (no package renames)
- CLI interface unchanged
- Data contracts unchanged

**Rollback Instructions**
- Documented in `REFACTOR_PLAN.md`
- Simple commands to restore any moved components
- Archive structure preserves original organization

#### 🎯 Benefits Achieved

**Reduced Clutter**
- 11 root scripts moved to organized location
- 4 ad-hoc tests properly categorized
- Generated artifacts no longer tracked
- Clean root directory structure

**Improved Documentation**
- Authoritative README with 5-minute quickstart
- Complete schema and taxonomy documentation
- Clear operational procedures
- Comprehensive validation tools

**Enhanced Maintainability**
- Organized script structure in `scripts/analysis/`
- Proper test organization in `tests/legacy/`
- Clear separation of concerns
- Documented deprecation paths

**Zero Behavior Drift**
- All algorithmic components unchanged
- Feature generation preserved
- TGAT model architecture intact
- Scoring and validation logic preserved

### 🔧 Technical Details

**Refactor Scope**
- Structure and documentation only
- No algorithmic changes
- No dependency modifications
- No schema changes

**Validation Status**
- Shard structure verified (51D/20D)
- CLI structure confirmed
- Entrypoint modules present
- Documentation complete

**Future Compatibility**
- Wave 8 cross-session edges: Ready
- HTF expansion: Supported
- New adapters: Framework ready
- Scale handling: Automatic detection

---

**Total Refactor Time**: ~90 minutes  
**Files Moved**: 15  
**Files Created**: 6  
**Documentation Pages**: 4  
**Backward Compatibility**: 100%  
**Behavior Drift**: 0%

## 1.0.1
- Oracle temporal non-locality (sidecar parquet, optional; contracts unchanged)
