# IRONFORGE Codebase Analysis Report

Generated on: 2025-08-18 19:17:34 UTC
Analysis Duration: 5.53 seconds
Files Analyzed: 155

## Project Overview

**IRONFORGE** - Multi-Engine Archaeological Discovery System

### Key Metrics
- **Total Files**: 155
- **Lines of Code**: 42,503
- **Functions**: 263
- **Classes**: 206
- **Average Complexity**: 5.7

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
- Files: 51
- Lines of Code: 16,588
- Average Complexity: 31.8

**Key Components**:
- **extract_lattice_summary** (162 LOC, Complexity: 19)
- **temporal_correlator** (484 LOC, Complexity: 57)
- **__init__** (15 LOC, Complexity: 0)
- **broad_spectrum_archaeology** (49 LOC, Complexity: 3)
- **enhanced_session_adapter** (49 LOC, Complexity: 3)
- ... and 46 more components

**Key Classes**:
- **CorrelationResult**: Result from temporal correlation analysis...
- **TemporalCorrelationEngine**: Engine for correlating predictions with validation data across sequences...
- **SequencePatternAnalyzer**: Analyzer for detecting patterns in cascade sequences...
- ... and 32 more classes

---

### Learning Engine

**Description**: Machine learning, TGAT discovery, and graph building

**Metrics**:
- Files: 12
- Lines of Code: 3,791
- Average Complexity: 41.8

**Key Components**:
- **archaeological_discovery_clean** (375 LOC, Complexity: 44)
- **htf_regime_analysis** (352 LOC, Complexity: 31)
- **taxonomy_v1** (119 LOC, Complexity: 6)
- **json_to_parquet** (836 LOC, Complexity: 142)
- **__init__** (1 LOC, Complexity: 0)
- ... and 7 more components

**Key Classes**:
- **ArchaeologicalZone**: Represents a discovered archaeological zone...
- **ArchaeologicalDiscoverer**: Enhanced archaeological discovery using HTF context features...
- **RegimeCharacteristics**: Characteristics of a market regime...
- ... and 19 more classes

---

### Synthesis Engine

**Description**: Pattern validation and production graduation

**Metrics**:
- Files: 3
- Lines of Code: 496
- Average Complexity: 15.7

**Key Components**:
- **__init__** (15 LOC, Complexity: 0)
- **pattern_graduation** (213 LOC, Complexity: 15)
- **production_graduation** (268 LOC, Complexity: 32)

**Key Classes**:
- **PatternGraduation**: Validation system ensuring discovered patterns exceed 87% baseline accuracy...
- **ProductionGraduation**: Production feature export for graduated patterns
Converts validated archaeological discoveries into ...

---

### Integration Engine

**Description**: System integration, configuration, and dependency injection

**Metrics**:
- Files: 28
- Lines of Code: 8,779
- Average Complexity: 30.0

**Key Components**:
- **config** (236 LOC, Complexity: 38)
- **__init__** (39 LOC, Complexity: 0)
- **__init__** (9 LOC, Complexity: 0)
- **__init__** (21 LOC, Complexity: 0)
- **adaptive_rg_optimizer** (814 LOC, Complexity: 78)
- ... and 23 more components

**Key Classes**:
- **IRONFORGEConfig**: Configuration manager for IRONFORGE system.

Eliminates hardcoded paths and provides environment-spe...
- **AdaptiveRGParameters**: Optimized RG parameters for current market regime...
- **ThresholdOptimizationResult**: Result from information-theoretic threshold optimization...
- ... and 105 more classes

---

### Validation Engine

**Description**: Testing, validation, and quality assurance

**Metrics**:
- Files: 7
- Lines of Code: 1,775
- Average Complexity: 26.6

**Key Components**:
- **validation_framework** (1403 LOC, Complexity: 153)
- **__init__** (8 LOC, Complexity: 0)
- **__init__** (1 LOC, Complexity: 0)
- **oos** (16 LOC, Complexity: 1)
- **__init__** (25 LOC, Complexity: 1)
- ... and 2 more components

**Key Classes**:
- **ValidationLevel**: Validation thoroughness levels...
- **TestResult**: Test result status...
- **ValidationResult**: Result of a validation test...
- ... and 8 more classes

---

### Reporting Engine

**Description**: Report generation and data visualization

**Metrics**:
- Files: 3
- Lines of Code: 501
- Average Complexity: 15.7

**Key Components**:
- **htf_observer** (374 LOC, Complexity: 38)
- **minidash** (99 LOC, Complexity: 5)
- **writers** (28 LOC, Complexity: 4)

**Key Classes**:
- **HTFRunSummary**: Summary statistics for HTF features across a discovery run...
- **HTFObserver**: Light observability for HTF context features...

---

### Utilities Engine

**Description**: Utility functions, scripts, and support tools

**Metrics**:
- Files: 40
- Lines of Code: 10,376
- Average Complexity: 29.7

**Key Components**:
- **batch_migrate_graphs** (550 LOC, Complexity: 58)
- **schema_normalizer** (386 LOC, Complexity: 47)
- **__init__** (12 LOC, Complexity: 0)
- **performance_monitor** (69 LOC, Complexity: 6)
- **__init__** (9 LOC, Complexity: 0)
- ... and 35 more components

**Key Classes**:
- **BatchGraphMigrator**: Batch migration system for IRONFORGE graph files

Technical Debt Surgeon: Comprehensive batch proces...
- **SchemaNormalizer**: Technical Debt Surgeon implementation for schema normalization
Migrates 34D legacy data to 37D tempo...
- **PerformanceMonitor**: Monitor performance of IRONFORGE components
Track timing, memory usage, and component initialization...
- ... and 23 more classes

---

### Data Engine

**Description**: Data storage and preservation

**Metrics**:
- Files: 5
- Lines of Code: 126
- Average Complexity: 2.6

**Key Components**:
- **__init__** (1 LOC, Complexity: 0)
- **__init__** (37 LOC, Complexity: 2)
- **parquet_reader** (14 LOC, Complexity: 1)
- **parquet_writer** (58 LOC, Complexity: 10)
- **schemas** (16 LOC, Complexity: 0)

**Key Classes**:
- No public classes found

---

## Dependency Analysis

### Cross-Engine Flows
- **analysis -> learning**: 17 dependencies
- **utilities -> learning**: 5 dependencies
- **analysis -> integration**: 3 dependencies
- **integration -> synthesis**: 2 dependencies
- **integration -> learning**: 2 dependencies
- **synthesis -> integration**: 2 dependencies
- **analysis -> validation**: 2 dependencies
- **learning -> utilities**: 1 dependencies
- **validation -> utilities**: 1 dependencies
- **analysis -> utilities**: 1 dependencies
- **utilities -> integration**: 1 dependencies

### Circular Dependencies
✅ No circular dependencies detected.

### Hub Modules (High Centrality)
- **typing** (Centrality: 209, Type: provider_hub)
- **pathlib** (Centrality: 81, Type: provider_hub)
- **json** (Centrality: 70, Type: provider_hub)
- **logging** (Centrality: 54, Type: provider_hub)
- **visualizations.lattice_visualizer** (Centrality: 53, Type: consumer_hub)

## Complexity Analysis

### Complexity Hotspots
- **main** in run_weekly_daily_cascade_lattice.py (Complexity: 43)
- **main** in run_fpfvg_network_analysis.py (Complexity: 38)
- **analyze_actual_events_by_subpattern** in scripts/analysis/analyze_concrete_patterns.py (Complexity: 29)
- **main** in run_specialized_lattice.py (Complexity: 25)
- **demonstrate_structural_analysis** in scripts/analysis/run_archaeology_demonstration.py (Complexity: 25)

### Summary
- **Files with High Complexity**: 40
- **Total Hotspot Functions**: 39

## Architecture Health Assessment

❌ **Engine Balance**: Significant imbalance - some engines are much larger than others
✅ **Dependency Health**: No circular dependencies detected
❌ **Complexity Health**: 20 high-complexity functions require refactoring

## Recommendations

🔄 **Analysis Engine**: Consider splitting into sub-engines (currently 51 files)
🧹 **Analysis Engine**: High average complexity (31.8) - consider refactoring
🧹 **Learning Engine**: High average complexity (41.8) - consider refactoring
🧹 **Synthesis Engine**: High average complexity (15.7) - consider refactoring
🔄 **Integration Engine**: Consider splitting into sub-engines (currently 28 files)
🧹 **Integration Engine**: High average complexity (30.0) - consider refactoring
🧹 **Validation Engine**: High average complexity (26.6) - consider refactoring
🧹 **Reporting Engine**: High average complexity (15.7) - consider refactoring
🔄 **Utilities Engine**: Consider splitting into sub-engines (currently 40 files)
🧹 **Utilities Engine**: High average complexity (29.7) - consider refactoring
⚡ **Dependency**: high_coupling in analysis -> learning
🔧 **Refactor**: main in run_weekly_daily_cascade_lattice.py (complexity: 43)
🔧 **Refactor**: main in run_fpfvg_network_analysis.py (complexity: 38)
🔧 **Refactor**: analyze_actual_events_by_subpattern in scripts/analysis/analyze_concrete_patterns.py (complexity: 29)

---

*Report generated by IRONFORGE Semantic Indexer v1.0.0*
