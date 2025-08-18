# IRONFORGE Codebase Analysis Report

Generated on: 2025-08-17 21:51:24 UTC
Analysis Duration: 8.08 seconds
Files Analyzed: 162

## Project Overview

**IRONFORGE** - Multi-Engine Archaeological Discovery System

### Key Metrics
- **Total Files**: 162
- **Lines of Code**: 50,977
- **Functions**: 286
- **Classes**: 192
- **Average Complexity**: 5.8

### Key Technologies
- TGAT
- PyTorch
- NetworkX
- NumPy
- iron-core

## Engine Architecture

IRONFORGE follows a multi-engine architecture pattern with clear separation of concerns:

### Analysis Engine

**Description**: Pattern analysis and session adaptation components

**Metrics**:
- Files: 62
- Lines of Code: 24,778
- Average Complexity: 42.8

**Key Components**:
- **extract_lattice_summary** (161 LOC, Complexity: 19)
- **temporal_correlator** (482 LOC, Complexity: 57)
- **__init__** (14 LOC, Complexity: 0)
- **__init__** (28 LOC, Complexity: 0)
- **broad_spectrum_archaeology** (47 LOC, Complexity: 3)
- ... and 57 more components

**Key Classes**:
- **CorrelationResult**: Result from temporal correlation analysis...
- **TemporalCorrelationEngine**: Engine for correlating predictions with validation data across sequences...
- **SequencePatternAnalyzer**: Analyzer for detecting patterns in cascade sequences...
- ... and 45 more classes

---

### Learning Engine

**Description**: Machine learning, TGAT discovery, and graph building

**Metrics**:
- Files: 10
- Lines of Code: 3,059
- Average Complexity: 35.9

**Key Components**:
- **contracts** (11 LOC, Complexity: 0)
- **__init__** (1 LOC, Complexity: 0)
- **igraph_builder** (15 LOC, Complexity: 1)
- **pyg_converters** (11 LOC, Complexity: 1)
- **__init__** (35 LOC, Complexity: 0)
- ... and 5 more components

**Key Classes**:
- **DiscoveryResult**: 
- **TemporalDiscoveryPipeline**: Shard‑aware pipeline for temporal TGAT discovery.

Parameters
----------
data_path : str or Path
   ...
- **RichNodeFeature**: 45D node feature vector with semantic preservation...
- ... and 5 more classes

---

### Synthesis Engine

**Description**: Pattern validation and production graduation

**Metrics**:
- Files: 5
- Lines of Code: 1,534
- Average Complexity: 25.4

**Key Components**:
- **__init__** (21 LOC, Complexity: 0)
- **pattern_graduation** (210 LOC, Complexity: 15)
- **production_graduation** (271 LOC, Complexity: 32)
- **metrics** (352 LOC, Complexity: 30)
- **runner** (680 LOC, Complexity: 50)

**Key Classes**:
- **PatternGraduation**: Validation system ensuring discovered patterns exceed 87% baseline accuracy...
- **ProductionGraduation**: Production feature export for graduated patterns
Converts validated archaeological discoveries into ...
- **ValidationConfig**: Configuration for validation experiments.

Parameters
----------
mode : str
    Validation mode: "oo...
- ... and 1 more classes

---

### Integration Engine

**Description**: System integration, configuration, and dependency injection

**Metrics**:
- Files: 25
- Lines of Code: 7,959
- Average Complexity: 29.4

**Key Components**:
- **config** (236 LOC, Complexity: 38)
- **__init__** (34 LOC, Complexity: 0)
- **__init__** (9 LOC, Complexity: 0)
- **__init__** (21 LOC, Complexity: 0)
- **adaptive_rg_optimizer** (812 LOC, Complexity: 78)
- ... and 20 more components

**Key Classes**:
- **IRONFORGEConfig**: Configuration manager for IRONFORGE system.

Eliminates hardcoded paths and provides environment-spe...
- **AdaptiveRGParameters**: Optimized RG parameters for current market regime...
- **ThresholdOptimizationResult**: Result from information-theoretic threshold optimization...
- ... and 85 more classes

---

### Validation Engine

**Description**: Testing, validation, and quality assurance

**Metrics**:
- Files: 9
- Lines of Code: 3,520
- Average Complexity: 34.4

**Key Components**:
- **validation_framework** (1403 LOC, Complexity: 153)
- **__init__** (8 LOC, Complexity: 0)
- **cards** (63 LOC, Complexity: 1)
- **__init__** (72 LOC, Complexity: 0)
- **controls** (302 LOC, Complexity: 20)
- ... and 4 more components

**Key Classes**:
- **ValidationLevel**: Validation thoroughness levels...
- **TestResult**: Test result status...
- **ValidationResult**: Result of a validation test...
- ... and 13 more classes

---

### Reporting Engine

**Description**: Report generation and data visualization

**Metrics**:
- Files: 5
- Lines of Code: 183
- Average Complexity: 3.8

**Key Components**:
- **__init__** (22 LOC, Complexity: 0)
- **confluence** (58 LOC, Complexity: 8)
- **heatmap** (61 LOC, Complexity: 6)
- **html** (26 LOC, Complexity: 3)
- **writer** (16 LOC, Complexity: 2)

**Key Classes**:
- **ConfluenceStripSpec**: 
- **TimelineHeatmapSpec**: 

---

### Utilities Engine

**Description**: Utility functions, scripts, and support tools

**Metrics**:
- Files: 39
- Lines of Code: 9,772
- Average Complexity: 28.9

**Key Components**:
- **batch_migrate_graphs** (506 LOC, Complexity: 58)
- **schema_normalizer** (383 LOC, Complexity: 47)
- **__init__** (6 LOC, Complexity: 0)
- **prepare_motifs_input** (91 LOC, Complexity: 16)
- **__init__** (2 LOC, Complexity: 0)
- ... and 34 more components

**Key Classes**:
- **BatchGraphMigrator**: Batch migration system for IRONFORGE graph files

Technical Debt Surgeon: Comprehensive batch proces...
- **SchemaNormalizer**: Technical Debt Surgeon implementation for schema normalization
Migrates 34D legacy data to 37D tempo...
- **PerformanceMonitor**: Monitor performance of IRONFORGE components
Track timing, memory usage, and component initialization...
- ... and 22 more classes

---

### Data Engine

**Description**: Data storage and preservation

**Metrics**:
- Files: 3
- Lines of Code: 54
- Average Complexity: 2.3

**Key Components**:
- **__init__** (1 LOC, Complexity: 0)
- **parquet_writer** (37 LOC, Complexity: 7)
- **schemas** (16 LOC, Complexity: 0)

**Key Classes**:
- No public classes found

---

## Dependency Analysis

### Cross-Engine Flows
- **analysis -> learning**: 18 dependencies
- **analysis -> integration**: 15 dependencies
- **validation -> integration**: 10 dependencies
- **utilities -> learning**: 6 dependencies
- **integration -> synthesis**: 2 dependencies
- **validation -> utilities**: 2 dependencies
- **integration -> learning**: 2 dependencies
- **synthesis -> integration**: 2 dependencies
- **analysis -> validation**: 2 dependencies
- **learning -> utilities**: 1 dependencies
- **learning -> integration**: 1 dependencies
- **validation -> learning**: 1 dependencies
- **validation -> synthesis**: 1 dependencies
- **analysis -> utilities**: 1 dependencies
- **utilities -> integration**: 1 dependencies

### Circular Dependencies
✅ No circular dependencies detected.

### Hub Modules (High Centrality)
- **typing** (Centrality: 324, Type: provider_hub)
- **datetime** (Centrality: 96, Type: provider_hub)
- **json** (Centrality: 82, Type: provider_hub)
- **pathlib** (Centrality: 81, Type: provider_hub)
- **logging** (Centrality: 69, Type: provider_hub)

## Complexity Analysis

### Complexity Hotspots
- **main** in run_weekly_daily_cascade_lattice.py (Complexity: 43)
- **main** in run_fpfvg_network_analysis.py (Complexity: 38)
- **analyze_actual_events_by_subpattern** in scripts/analysis/analyze_concrete_patterns.py (Complexity: 29)
- **main** in run_specialized_lattice.py (Complexity: 25)
- **demonstrate_structural_analysis** in scripts/analysis/run_archaeology_demonstration.py (Complexity: 25)

### Summary
- **Files with High Complexity**: 39
- **Total Hotspot Functions**: 39

## Architecture Health Assessment

❌ **Engine Balance**: Significant imbalance - some engines are much larger than others
✅ **Dependency Health**: No circular dependencies detected
❌ **Complexity Health**: 20 high-complexity functions require refactoring

## Recommendations

🔄 **Analysis Engine**: Consider splitting into sub-engines (currently 62 files)
🧹 **Analysis Engine**: High average complexity (42.8) - consider refactoring
🧹 **Learning Engine**: High average complexity (35.9) - consider refactoring
🧹 **Synthesis Engine**: High average complexity (25.4) - consider refactoring
🔄 **Integration Engine**: Consider splitting into sub-engines (currently 25 files)
🧹 **Integration Engine**: High average complexity (29.4) - consider refactoring
🧹 **Validation Engine**: High average complexity (34.4) - consider refactoring
🔄 **Utilities Engine**: Consider splitting into sub-engines (currently 39 files)
🧹 **Utilities Engine**: High average complexity (28.9) - consider refactoring
⚡ **Dependency**: high_coupling in analysis -> integration
⚡ **Dependency**: high_coupling in analysis -> learning
🔧 **Refactor**: main in run_weekly_daily_cascade_lattice.py (complexity: 43)
🔧 **Refactor**: main in run_fpfvg_network_analysis.py (complexity: 38)
🔧 **Refactor**: analyze_actual_events_by_subpattern in scripts/analysis/analyze_concrete_patterns.py (complexity: 29)

---

*Report generated by IRONFORGE Semantic Indexer v1.0.0*
