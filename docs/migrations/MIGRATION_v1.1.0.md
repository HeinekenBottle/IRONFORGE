# Migration Guide: IRONFORGE v1.0.x → v1.1.0

## Overview

IRONFORGE v1.1.0 introduces a comprehensive refactor that consolidates scripts, unifies dependency management, and aligns with modern GitHub practices. This guide helps you migrate from v1.0.x to v1.1.0.

## Breaking Changes

### 1. Removed Scripts

The following deprecated `run_*.py` scripts have been removed from the root directory:

```bash
# REMOVED (v1.0.x)
./run_fpfvg_network_analysis.py
./run_fpfvg_network_analysis_simple.py
./run_fpfvg_redelivery_lattice.py
./run_global_lattice.py
./run_specialized_lattice.py
./run_terrain_analysis.py
./run_weekly_daily_cascade_lattice.py
./run_weekly_daily_sweep_cascade_step_3b.py
./run_weekly_daily_sweep_cascade_step_3b_refined.py
./run_working_cascade_analysis.py
```

**Migration**: Use the new unified runner:
```bash
# NEW (v1.1.0)
python scripts/unified_runner.py <workflow> [options]
```

### 2. Dependency Management

**REMOVED**:
- `requirements.txt`
- `requirements-dev.txt`
- `setup.py`

**NEW**: All dependencies are now managed through `pyproject.toml`

```bash
# OLD (v1.0.x)
pip install -r requirements.txt
pip install -r requirements-dev.txt

# NEW (v1.1.0)
pip install -e .[dev]
```

### 3. Removed Files

- `README_OLD.md` → Use main `README.md`
- `config.py` → Use configuration files in `configs/`
- `scripts/bump_version.py` → Use `scripts/bump_version_modern.py`

## New Features

### Unified Script Runner

Replace individual script calls with the unified runner:

```bash
# Discovery workflow
python scripts/unified_runner.py discovery --config configs/dev.yml

# Confluence scoring
python scripts/unified_runner.py confluence --config configs/dev.yml

# Full pipeline
python scripts/unified_runner.py pipeline --config configs/dev.yml

# Oracle training
python scripts/unified_runner.py oracle --train --symbols NQ --tf M5

# Analysis workflows
python scripts/unified_runner.py analysis --analysis-type comprehensive
```

### Updated Dependencies

Major version updates:
- **numpy**: 1.20.0 → 1.24.0+
- **pandas**: 1.3.0 → 2.2.0+
- **torch**: 1.9.0 → 2.0.0+
- **scikit-learn**: 1.0.0 → 1.3.0+
- **networkx**: 2.5 → 3.0+

New dependencies:
- **torch-geometric**: 2.4.0+ (enhanced graph processing)
- **pyarrow**: 14.0.0+ (improved data handling)
- **orjson**: 3.9.0+ (faster JSON processing)

## Migration Steps

### 1. Update Installation

```bash
# Remove old virtual environment (recommended)
rm -rf venv/

# Create new environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with new dependency management
pip install -e .[dev]
```

### 2. Update Scripts

Replace any calls to removed `run_*.py` scripts:

```bash
# OLD
python run_fpfvg_network_analysis.py

# NEW
python scripts/unified_runner.py discovery --config configs/dev.yml
```

### 3. Update CI/CD

Update your CI/CD pipelines:

```yaml
# OLD
- pip install -r requirements.txt
- pip install -r requirements-dev.txt

# NEW
- pip install -e .[dev]
```

### 4. Update Documentation References

- Update any documentation that references removed scripts
- Use new version numbers (v1.1.0)
- Reference unified runner for workflow examples

## Compatibility

### What Still Works

- All CLI commands (`ironforge`, `ifg`)
- Core API and module imports
- Configuration files in `configs/`
- Existing data formats and schemas
- All analysis scripts in `scripts/analysis/`

### What Changed

- Installation method (now uses pyproject.toml)
- Script execution (now uses unified runner)
- Version numbers (updated to v1.1.0)
- Dependency versions (updated to latest)

## Testing Migration

Verify your migration:

```bash
# Test imports
python -m pytest tests/test_imports_smoke.py -v

# Test unified runner
python scripts/unified_runner.py --help

# Test CLI
ironforge --version  # Should show v1.1.0

# Run smoke tests
python tools/smoke_checks.py
```

## Rollback Plan

If you need to rollback:

```bash
# Switch back to previous version
git checkout main  # or your previous branch
pip install -e .[dev]
```

## Support

For migration issues:
1. Check this migration guide
2. Review the [CHANGELOG](../CHANGELOG.md)
3. Check existing issues in the repository
4. Create a new issue with migration details

## Benefits of v1.1.0

- **Simplified installation**: Single command dependency management
- **Cleaner repository**: Removed 15+ redundant/deprecated files
- **Modern dependencies**: Latest versions with security updates
- **Better organization**: Consolidated scripts and documentation
- **GitHub alignment**: Follows modern repository best practices
- **Enhanced functionality**: New dependencies enable better performance
