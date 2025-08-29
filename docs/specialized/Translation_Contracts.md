# Translation Layer Contracts

## Overview

This document specifies contracts for translation layers in the IRONFORGE motif analysis pipeline, addressing issues identified during the recent audit. All contracts are opt-in with feature flags to ensure zero behavioral change by default.

## D2G (Data to Graph) Contract

**Purpose**: Transform enhanced session data into NetworkX DiGraph structures for motif analysis.

### Issues Identified
- Pandas import error in build-graph command
- Session data format mismatch (enhanced JSON → DAG events)
- Missing temporal relationship encoding

### Contract Specifications

```python
# Feature Flag: IRONFORGE_ENABLE_OPTIMIZED_DAG_BUILDER
class OptimizedDAGBuilder:
    def __init__(self, config: DAGBuilderConfig):
        self.enable_cache = config.enable_cache
        self.enable_lazy_loading = config.enable_lazy_loading
        self.enable_topological_optimization = config.enable_topological_optimization
    
    def build_graph(self, session_data: Dict) -> nx.DiGraph:
        """
        Build DAG with Context7-guided optimizations:
        - Use topological_generations for stratified node processing
        - Implement proper edge ordering based on timestamps
        - Cache intermediate graph states for incremental builds
        """
```

### Best Practices Applied
- **NetworkX topological_generations**: Process nodes layer-by-layer for better memory efficiency
- **Optimized DiGraph construction**: Use bulk edge additions with pre-sorted data
- **Proper random_state handling**: Ensure reproducible graph generation

## G2M (Graph to Model) Contract

**Purpose**: Convert DiGraph structures to statistical models for pattern detection.

### Issues Identified
- DAG motif mining scalability with 10K bootstrap iterations
- NetworkX memory inefficiencies with 114 graphs
- Subgraph isomorphism performance bottlenecks

### Contract Specifications

```python
# Feature Flag: IRONFORGE_ENABLE_EFFICIENT_MOTIF_MINING
class EfficientMotifMiner:
    def __init__(self, config: MotifMiningConfig):
        self.enable_parallel_isomorphism = config.enable_parallel_isomorphism
        self.enable_graph_caching = config.enable_graph_caching
        self.bootstrap_optimization = config.bootstrap_optimization
    
    def mine_motifs(self, graphs: List[nx.DiGraph]) -> List[Motif]:
        """
        Mine motifs with Context7-guided optimizations:
        - Use DiGraphMatcher with semantic feasibility checks
        - Implement incremental graph comparison
        - Optimize memory usage with sparse representations
        """
```

### Best Practices Applied
- **Parallel DiGraphMatcher**: Use VF2 algorithm optimizations for directed graphs
- **Memory-efficient storage**: Convert to sparse adjacency matrices when beneficial
- **Cached subgraph operations**: Reuse isomorphism computations across bootstrap iterations

## M2E (Model to Execution) Contract

**Purpose**: Execute statistical validation with reproducible bootstrap methods.

### Issues Identified
- Bootstrap null model reproducibility concerns
- Statistical validation threading/parallelization gaps
- Memory pressure during concurrent motif testing

### Contract Specifications

```python
# Feature Flag: IRONFORGE_ENABLE_REPRODUCIBLE_BOOTSTRAP
class ReproducibleBootstrap:
    def __init__(self, config: BootstrapConfig):
        self.random_state = config.random_state
        self.n_jobs = config.n_jobs
        self.enable_stratified_sampling = config.enable_stratified_sampling
    
    def validate_motifs(self, motifs: List[Motif], n_iterations: int) -> ValidationResults:
        """
        Validate motifs with scikit-learn guided reproducibility:
        - Use check_random_state for consistent RNG initialization
        - Implement stratified resampling when applicable
        - Ensure thread-safe random number generation
        """
```

### Best Practices Applied
- **sklearn.utils.check_random_state**: Ensure reproducible random number generation
- **Stratified bootstrap sampling**: Use sklearn.utils.resample with stratify parameter
- **Thread-safe RNG**: Separate RandomState instances per worker thread

## E2R (Execution to Results) Contract

**Purpose**: Serialize results with optimized PyArrow schemas and row group management.

### Issues Identified
- PyArrow schema inconsistencies in parquet output
- Row group optimization missing for large result sets
- CDC (Change Data Capture) not implemented for incremental updates

### Contract Specifications

```python
# Feature Flag: IRONFORGE_ENABLE_OPTIMIZED_PARQUET
class OptimizedParquetWriter:
    def __init__(self, config: ParquetConfig):
        self.enable_cdc = config.enable_cdc
        self.enable_row_group_optimization = config.enable_row_group_optimization
        self.enable_content_defined_chunking = config.enable_content_defined_chunking
    
    def write_results(self, results: pd.DataFrame, path: str) -> None:
        """
        Write results with PyArrow-guided optimizations:
        - Use content-defined chunking for optimal storage
        - Implement row group sizing based on query patterns
        - Enable CDC tracking for incremental updates
        """
```

### Best Practices Applied
- **Content-defined chunking**: Use PyArrow's CDC for optimal compression and query performance
- **Optimized row group sizing**: Balance between query performance and storage efficiency
- **Schema evolution support**: Maintain backward compatibility with field metadata

## RTP (Results to Presentation) Contract

**Purpose**: Generate consistent, validated presentation outputs.

### Issues Identified
- Markdown generation lacks structured templating
- Statistical formatting inconsistencies
- Missing validation of output contracts

### Contract Specifications

```python
# Feature Flag: IRONFORGE_ENABLE_VALIDATED_PRESENTATION
class ValidatedPresentationEngine:
    def __init__(self, config: PresentationConfig):
        self.enable_template_validation = config.enable_template_validation
        self.enable_statistical_formatting = config.enable_statistical_formatting
        self.enable_contract_validation = config.enable_contract_validation
    
    def generate_report(self, results: ValidationResults) -> ReportOutput:
        """
        Generate reports with validated formatting:
        - Use Jinja2 templates with schema validation
        - Apply consistent statistical number formatting
        - Validate output against predefined contracts
        """
```

### Best Practices Applied
- **Template schema validation**: Ensure consistent report structure
- **Statistical formatting standards**: Use scientific notation and confidence intervals consistently
- **Contract validation**: Verify all required fields are present and properly formatted

## Implementation Strategy

### Phase 1: Foundation (Safe Defaults)
1. Add feature flags with default `False` values
2. Implement basic contracts with existing behavior preservation
3. Add comprehensive unit tests for all new components

### Phase 2: Optimization (Opt-in Features)
1. Enable individual feature flags for testing
2. Benchmark performance improvements
3. Validate reproducibility across different environments

### Phase 3: Integration (Full Pipeline)
1. Enable multiple flags in combination
2. Run full regression tests on historical data
3. Document performance characteristics and trade-offs

## Configuration Management

```python
@dataclass
class TranslationConfig:
    """Master configuration for all translation layers."""
    
    # D2G Configuration
    enable_optimized_dag_builder: bool = False
    dag_builder: DAGBuilderConfig = field(default_factory=DAGBuilderConfig)
    
    # G2M Configuration  
    enable_efficient_motif_mining: bool = False
    motif_mining: MotifMiningConfig = field(default_factory=MotifMiningConfig)
    
    # M2E Configuration
    enable_reproducible_bootstrap: bool = False
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    
    # E2R Configuration
    enable_optimized_parquet: bool = False
    parquet: ParquetConfig = field(default_factory=ParquetConfig)
    
    # RTP Configuration
    enable_validated_presentation: bool = False
    presentation: PresentationConfig = field(default_factory=PresentationConfig)
```

## Testing Strategy

Each contract implementation includes:

1. **Unit tests**: Verify individual component behavior
2. **Integration tests**: Validate inter-layer communication
3. **Regression tests**: Ensure backward compatibility
4. **Performance tests**: Benchmark optimization effectiveness
5. **Reproducibility tests**: Validate deterministic behavior across runs

## Migration Path

1. **Assessment**: Use existing codebase without changes (all flags disabled)
2. **Selective adoption**: Enable individual flags based on specific needs
3. **Full migration**: Gradually enable all optimizations after validation
4. **Monitoring**: Track performance metrics and error rates during transition