# IRONFORGE Repository Refactor Plan (v0.7.1 / Wave 7.2)

## Executive Summary

This plan outlines the safe refactoring of the IRONFORGE repository to reduce clutter and improve operability while preserving all canonical functionality. The refactor follows a phased approach with safety rails to ensure zero behavior drift.

## Current State Analysis

### Root-Level Clutter Identified

**One-off Run Scripts (11 files)**
- `run_weekly_daily_cascade_lattice.py` - Weekly→Daily liquidity sweep cascade analysis
- `run_weekly_daily_sweep_cascade_step_3b.py` - Macro driver analysis (Step 3B)
- `run_weekly_daily_sweep_cascade_step_3b_refined.py` - Refined Step 3B with lowered thresholds
- `run_working_cascade_analysis.py` - Streamlined cascade analysis
- `run_fpfvg_network_analysis.py` - FPFVG network analysis (Step 3A)
- `run_fpfvg_redelivery_lattice.py` - Theory B testing for FVG formations
- `run_global_lattice.py` - Global lattice build (Step 1)
- `run_specialized_lattice.py` - Specialized lattice execution
- `run_terrain_analysis.py` - Terrain analysis execution
- `run_fpfvg_network_analysis_simple.py` - Simplified FPFVG analysis
- `extract_lattice_summary.py` - Lattice summary extraction

**Ad-hoc Test Files (4 files)**
- `simple_threshold_test.py` - Direct threshold testing for sweep detection
- `test_refactored_structure.py` - Directory structure validation
- `test_refined_detection_quick.py` - Quick refined detection testing
- `validate_ci_setup.py` - Local CI validation

**Generated Artifacts & Tracked Outputs**
- `deliverables/` - Generated analysis results and reports
- `demo_deliverables/` - Demo outputs
- `discovery_cache/` - Cached discovery results
- `results/` - Session analysis results
- `reports/` - Generated reports (keep recent runs/)
- `test_outputs/` - Test result artifacts
- `preservation/` - Preservation snapshots

**Bundled Virtual Environments**
- `ironforge_env/` - Committed virtual environment
- `ironforge_refactor_env/` - Refactor environment

**Legacy Migration Code**
- `data_migration/` - 34D→37D schema migration scripts

**Empty/Redundant Directories**
- `analysis/__pycache__/` - Python cache files
- Various `__pycache__` directories

## Proposed Changes by Phase

### Phase A: Inventory & Plan (COMPLETED)

✅ **Status**: Analysis complete, plan documented

### Phase B: Apply Minimal Moves with Facades

#### B1: Root Script Consolidation

**Action**: Move run scripts to `scripts/analysis/` with deprecation stubs

**Risk**: LOW - Scripts are standalone, no imports depend on them

**Implementation**:
```bash
# Create scripts/analysis/legacy_runners/
mkdir -p scripts/analysis/legacy_runners

# Move scripts
mv run_*.py scripts/analysis/legacy_runners/

# Create deprecation stubs at root
for script in run_*.py; do
  cat > "$script" << 'EOF'
#!/usr/bin/env python3
import sys, subprocess
print(f"⚠️  DEPRECATED: {sys.argv[0]} moved to scripts/analysis/legacy_runners/")
print("Use: python -m ironforge.sdk.cli discover-temporal|score-session|validate-run|report-minimal")
sys.exit(subprocess.call([sys.executable, "-m", "ironforge.sdk.cli", "discover-temporal"]))
EOF
done
```

**Compatibility Layer**: Deprecation stubs maintain backward compatibility

#### B2: Ad-hoc Test Migration

**Action**: Move to `tests/` and convert to pytest style

**Risk**: LOW - These are test files, not production dependencies

**Implementation**:
```bash
# Move to tests/legacy/
mkdir -p tests/legacy
mv simple_threshold_test.py tests/legacy/
mv test_refactored_structure.py tests/legacy/
mv test_refined_detection_quick.py tests/legacy/
mv validate_ci_setup.py tests/legacy/

# Convert to pytest format (manual review required)
```

**Compatibility Layer**: None needed - these are development tools

#### B3: Legacy Migration Archive

**Action**: Move `data_migration/` to `archive/data_migration/`

**Risk**: MEDIUM - May be referenced in some workflows

**Implementation**:
```bash
mkdir -p archive
mv data_migration archive/
echo "# Data Migration Archive

This directory contains legacy schema migration scripts for 34D→37D transitions.

## Status: DEPRECATED
All current shards are 51D/20D format. These scripts are preserved for historical reference.

## Rollback Instructions
If needed, restore with: \`mv archive/data_migration ./\`
" > archive/data_migration/README.md
```

**Compatibility Layer**: Add import facade if any code references these modules

#### B4: Generated Artifacts Cleanup

**Action**: Remove tracked artifacts, add .gitignore rules

**Risk**: LOW - These are generated outputs

**Implementation**:
```bash
# Remove tracked artifacts (preserve recent runs/)
rm -rf deliverables/ demo_deliverables/ discovery_cache/ results/ test_outputs/ preservation/

# Keep recent runs (last few days)
find runs/ -type d -name "20*" | head -n -3 | xargs rm -rf

# Add .gitignore rules
cat >> .gitignore << 'EOF'
# Generated artifacts
deliverables/
demo_deliverables/
discovery_cache/
results/
test_outputs/
preservation/

# Virtual environments
ironforge_env/
ironforge_refactor_env/
*_env/

# Python cache
__pycache__/
*.pyc
*.pyo
EOF
```

**Compatibility Layer**: None needed - these are artifacts

#### B5: Virtual Environment Removal

**Action**: Remove committed virtual environments

**Risk**: LOW - These should not be tracked

**Implementation**:
```bash
rm -rf ironforge_env/ ironforge_refactor_env/
```

**Compatibility Layer**: None needed

### Phase C: Documentation & Smoke Tests

#### C1: Update README.md

**Action**: Replace with authoritative, concise README

**Risk**: LOW - Documentation update

**Implementation**: Create new README.md following the specified outline

#### C2: Create Documentation Suite

**Action**: Create/update core documentation files

**Files to Create/Update**:
- `docs/flows.md` - Schema contracts and run order
- `docs/taxonomy_v1.md` - Authoritative event/edge taxonomy
- `docs/operations.md` - Daily operations guide

#### C3: Smoke Test Implementation

**Action**: Create `tools/smoke_checks.py`

**Implementation**: Comprehensive validation script covering:
- Import validation for all entrypoints
- CLI command availability
- Schema dimension validation (51/20)
- Run artifact presence
- Minidash rendering

## Risk Assessment

### HIGH RISK (Requires Careful Handling)
- None identified - all changes are structural/cleanup

### MEDIUM RISK (Requires Facades)
- `data_migration/` move - May have hidden dependencies
- Root script removal - External scripts may reference them

### LOW RISK (Safe to Proceed)
- Generated artifact removal
- Virtual environment cleanup
- Ad-hoc test migration
- Documentation updates

## Golden Invariants Verification

✅ **Event Taxonomy**: 6 events preserved (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
✅ **Edge Intents**: 4 intents preserved (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)
✅ **Feature Dimensions**: 51D nodes (f0..f50), 20D edges (e0..e19)
✅ **Entrypoints**: All import paths preserved with facades
✅ **CLI Commands**: All 5 commands maintained
✅ **Data Contracts**: Shard and run directory structures unchanged

## Success Criteria Checklist

- [ ] All imports & CLI working unchanged
- [ ] Schema dims validated on sample shards (51/20)
- [ ] Run artifacts present and minidash renders
- [ ] Root clutter reduced with proper stubs/archives
- [ ] New README.md provides 5-minute quickstart
- [ ] Smoke tests pass
- [ ] Zero behavior drift confirmed

## Implementation Timeline

1. **Phase B1-B2**: Script and test migration (30 minutes)
2. **Phase B3-B5**: Archive and cleanup (15 minutes)
3. **Phase C1-C3**: Documentation and validation (45 minutes)

**Total Estimated Time**: 90 minutes

## Rollback Plan

Each phase includes specific rollback instructions. Critical rollback commands:
```bash
# Restore data migration
mv archive/data_migration ./

# Restore root scripts
mv scripts/analysis/legacy_runners/run_*.py ./

# Restore tests
mv tests/legacy/*.py ./
```
