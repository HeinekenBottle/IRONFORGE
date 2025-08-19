# Data Migration Archive

## ⚠️ STATUS: DEPRECATED

This directory contains legacy schema migration scripts for 34D→37D transitions.

**All current shards are now 51D/20D format.** These scripts are preserved for historical reference only.

## Historical Context

These scripts were used during the transition from:
- **Old Schema**: 34D feature vectors
- **Intermediate Schema**: 37D feature vectors  
- **Current Schema**: 51D nodes (f0..f50), 20D edges (e0..e19)

## Contents

- `batch_migrate_graphs.py` - Batch processing for large datasets
- `schema_normalizer.py` - Schema normalization utilities
- `test_schema_migration.py` - Migration validation tests
- `README.md` - Original documentation (preserved)

## Rollback Instructions

If these scripts are needed for any reason:

```bash
# Restore to root directory
mv archive/data_migration ./

# Verify functionality
python data_migration/test_schema_migration.py
```

## Current Migration Path

For current schema operations, use:
```bash
# Convert enhanced sessions to current 51D/20D shards
python -m ironforge.sdk.cli prep-shards --htf-context
```

## Archive Date
Archived during IRONFORGE v0.7.1 refactor (Wave 7.2)
