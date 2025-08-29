# Archaeological Zone Detection Agent

**Production-Ready Archaeological Intelligence for IRONFORGE Pipeline**

## Overview

The Archaeological Zone Detection Agent is a sophisticated temporal pattern analysis system that integrates seamlessly with IRONFORGE's discovery pipeline. It implements archaeological intelligence principles to detect dimensional anchor points with temporal non-locality, enhancing TGAT discovery capabilities with 40% dimensional anchoring and Theory B forward positioning validation.

### Core Capabilities

- **40% Dimensional Anchoring**: Detects anchor zones using archaeological constant (40% of previous session range)
- **Temporal Non-Locality Analysis**: Theory B forward positioning with eventual completion validation
- **7.55-Point Precision**: Target precision scoring for archaeological zone accuracy
- **87% Authenticity Threshold**: Production-grade authenticity filtering for pattern graduation
- **Sub-3s Processing**: High-performance session analysis meeting IRONFORGE requirements
- **Session Isolation**: Absolute boundary respect with HTF last-closed compliance

## Quick Start

### Installation

```bash
# Agent is part of IRONFORGE agents collection
cd /Users/jack/IRONFORGE/agents/archaeological_zone_detector

# Install dependencies (handled by IRONFORGE pip install -e .[dev])
pip install -e /Users/jack/IRONFORGE[dev]
```

### Basic Usage

```python
from agents.archaeological_zone_detector.agent import (
    create_archaeological_zone_detector,
    enhance_discovery_with_archaeological_intelligence
)
import pandas as pd

# Create detector with default configuration
detector = create_archaeological_zone_detector()

# Sample session data
session_data = pd.DataFrame({
    'price': [100.0, 101.0, 102.0, 101.5, 100.5],
    'high': [100.5, 101.5, 102.5, 102.0, 101.0],
    'low': [99.5, 100.5, 101.5, 101.0, 100.0],
    'timestamp': [1, 2, 3, 4, 5],
    'session_id': ['example_session'] * 5
})

# Previous session for dimensional anchoring
previous_session = pd.DataFrame({
    'high': [105.0, 106.0], 
    'low': [95.0, 94.0],
    'session_id': ['previous_session'] * 2
})

# Detect archaeological zones
analysis = detector.detect_archaeological_zones(
    session_data, 
    previous_session
)

# Results
print(f"Session: {analysis.session_id}")
print(f"Zones detected: {len(analysis.archaeological_zones)}")
print(f"Processing time: {analysis.performance_metrics['detection_time']:.3f}s")
print(f"Authenticity: {analysis.performance_metrics.get('authenticity_score', 0):.1f}")
```

### IRONFORGE Pipeline Integration

```python
from ironforge.api import run_discovery
from agents.archaeological_zone_detector.agent import (
    enhance_discovery_with_archaeological_intelligence
)

# Standard IRONFORGE discovery
discovery_results = run_discovery(
    shard_paths=['data/shards/session_001'],
    out_dir='runs/2025-01-15',
    loader_cfg=my_loader_config
)

# Enhance with archaeological intelligence
session_data = load_session_data('data/sessions/session_001.json')
enhanced_results = enhance_discovery_with_archaeological_intelligence(
    discovery_results,
    session_data,
    archaeological_config
)

# Enhanced results include:
# - archaeological_analysis: Complete zone analysis
# - archaeological_enhancement: TGAT discovery enhancements  
# - authenticity_boost: Boost factor for pattern graduation
```

## Architecture

### Core Components

```
archaeological_zone_detector/
├── agent.py                   # Main ArchaeologicalZoneDetector class
├── ironforge_config.py        # Configuration management
├── tools.py                   # Core analysis tools
├── contracts.py               # Data contract validation
├── performance.py             # Performance monitoring
└── tests/                     # Comprehensive test suite
    ├── unit/                  # Component unit tests
    ├── integration/           # IRONFORGE integration tests
    ├── performance/           # Performance benchmarks
    ├── contracts/             # Contract validation tests
    └── end_to_end/            # Complete workflow tests
```

### Key Classes

#### ArchaeologicalZoneDetector

Main agent class providing archaeological intelligence for temporal pattern analysis.

```python
class ArchaeologicalZoneDetector:
    def detect_archaeological_zones(
        self,
        session_data: pd.DataFrame,
        previous_session_data: Optional[pd.DataFrame] = None,
        enhanced_graph: Optional[nx.Graph] = None
    ) -> ArchaeologicalAnalysis
    
    def enhance_tgat_discovery(
        self,
        graph: nx.Graph,
        archaeological_analysis: ArchaeologicalAnalysis
    ) -> Dict[str, Any]
```

#### Core Analysis Tools

- **DimensionalAnchorCalculator**: 40% dimensional anchor point calculations
- **TemporalNonLocalityValidator**: Theory B forward positioning analysis
- **TheoryBValidator**: Forward positioning validation and precision scoring
- **ZoneAnalyzer**: Comprehensive integration of all analysis components

### Archaeological Principles

#### 40% Dimensional Anchoring

```python
# Archaeological constant: anchor_zone = previous_day_range * 0.40
anchor_zone_width = previous_range * 0.40
```

The agent calculates dimensional anchor points using the archaeological discovery that 40% of the previous session's range provides optimal temporal reference points for zone detection.

#### Theory B Forward Positioning

Events position relative to eventual session completion, not intermediate states:

```python
# Forward positioning validation
def validate_theory_b_positioning(events, final_state):
    # Events should position relative to FINAL completion
    # Not intermediate positioning states
    return forward_coherence_score
```

#### Temporal Non-Locality

Information propagates through forward temporal echoes:

```python
# Temporal echo detection
temporal_echoes = detector.detect_temporal_echoes(
    session_data, anchor_zones
)
# Each echo represents forward-propagating market structure information
```

## Configuration

### Default Configuration

```python
from agents.archaeological_zone_detector.ironforge_config import ArchaeologicalConfig

config = ArchaeologicalConfig()

# Archaeological constants (DO NOT MODIFY)
assert config.dimensional_anchor.anchor_percentage == 0.40  # 40% anchoring
assert config.dimensional_anchor.precision_target == 7.55   # Precision target
assert config.authenticity.authenticity_threshold == 87.0   # Authenticity threshold

# Performance requirements
assert config.performance.max_session_processing_time <= 3.0  # <3s processing
assert config.performance.min_anchor_accuracy >= 95.0        # >95% accuracy  
assert config.performance.max_memory_usage_mb <= 100.0       # <100MB memory
```

### Configuration Presets

```python
from agents.archaeological_zone_detector.ironforge_config import ConfigurationPresets

# Production optimized (strict performance, high authenticity)
prod_config = ConfigurationPresets.production_config()

# Development with debugging (relaxed constraints, detailed logging)
dev_config = ConfigurationPresets.development_config()

# Research mode (all features enabled, flexible constraints)
research_config = ConfigurationPresets.research_config()

# Minimal mode (basic features only, maximum performance)
minimal_config = ConfigurationPresets.minimal_config()
```

### Custom Configuration

```python
config = ArchaeologicalConfig()

# Modify non-archaeological settings
config.performance.max_session_processing_time = 2.0  # Stricter timing
config.debugging.debug_mode_enabled = True            # Enable debugging
config.temporal_nonlocality.temporal_echo_detection = False  # Disable echoes

# Archaeological constants remain immutable
# config.dimensional_anchor.anchor_percentage = 0.35  # ❌ Would break compatibility
```

## Performance Requirements

### Production Requirements

| Metric | Requirement | Validation |
|--------|-------------|------------|
| Session Processing Time | <3.0 seconds | `performance_metrics['detection_time']` |
| Zone Detection Time | <1.0 seconds | Measured in `time_operation()` context |
| Anchor Accuracy | >95% | `performance_metrics['accuracy']` |
| Memory Usage | <100MB | `performance_metrics['memory_usage_mb']` |
| Authenticity Threshold | >87% | `zone.authenticity_score` |
| Precision Target | ~7.55 points | `zone.precision_score` |

### Performance Monitoring

```python
from agents.archaeological_zone_detector.performance import (
    create_production_monitor,
    benchmark_zone_detection
)

# Production monitoring
monitor = create_production_monitor(config)
session_id = monitor.start_session_analysis()

with monitor.time_operation('zone_detection'):
    zones = detector.detect_archaeological_zones(session_data)

session_performance = monitor.end_session_analysis()

# Performance summary
summary = monitor.get_performance_summary()
print(f"Average processing time: {summary['historical_performance']['avg_processing_time']:.3f}s")
print(f"Sessions meeting requirements: {summary['historical_performance']['sessions_exceeding_time_limit']}")
```

### Benchmarking

```python
# Benchmark against test sessions
test_sessions = [load_session(f'test_session_{i}.json') for i in range(10)]

benchmark_results = benchmark_zone_detection(
    detector.detect_archaeological_zones,
    test_sessions,
    config
)

print(f"Average processing time: {benchmark_results['avg_processing_time']:.3f}s")
print(f"Performance score: {benchmark_results['performance_score']:.1f}%")
```

## Contract Validation

### Golden Invariants (Immutable)

```python
from agents.archaeological_zone_detector.contracts import create_strict_validator

validator = create_strict_validator(config)

# Event taxonomy validation (exactly 6 types)
EVENT_TYPES = ["Expansion", "Consolidation", "Retracement", 
               "Reversal", "Liquidity Taken", "Redelivery"]

# Edge intent validation (exactly 4 types)  
EDGE_INTENTS = ["TEMPORAL_NEXT", "MOVEMENT_TRANSITION", 
                "LIQ_LINK", "CONTEXT"]

# Feature dimensions (51D nodes, 20D edges)
validator.validate_feature_dimensions(51, 20)  # HTF mode
validator.validate_feature_dimensions(45, 20)  # Standard mode
```

### Session Isolation

```python
# Absolute session boundary enforcement
validator.validate_session_isolation(session_data, session_id)

# HTF compliance (last-closed only)
htf_validator = HTFComplianceValidator(config)
violations = htf_validator.validate_htf_compliance(session_data, session_id)
assert len(violations) == 0, "HTF compliance violations detected"
```

### Archaeological Constants

```python
# Validation of archaeological discoveries
contract_results = validator.validate_archaeological_output(
    archaeological_zones,
    session_id,
    session_data
)

assert contract_results['archaeological_constants'] == True
assert contract_results['golden_invariants'] == True
assert contract_results['session_isolation'] == True
```

## Testing

### Running Tests

```bash
# All tests
pytest agents/archaeological_zone_detector/tests/ -v

# Unit tests only
pytest agents/archaeological_zone_detector/tests/unit/ -v

# Integration tests
pytest agents/archaeological_zone_detector/tests/integration/ -v

# Performance benchmarks
pytest agents/archaeological_zone_detector/tests/performance/ --benchmark-only

# Coverage report
pytest --cov=agents.archaeological_zone_detector --cov-report=html
```

### Test Categories

#### Unit Tests (`tests/unit/`)

- **test_tools.py**: Core analysis components
- **test_config.py**: Configuration validation  
- **test_contracts.py**: Contract validation logic
- **test_performance.py**: Performance monitoring components

#### Integration Tests (`tests/integration/`)

- **test_ironforge_integration.py**: IRONFORGE pipeline integration
- **test_enhanced_graph_builder.py**: Enhanced Graph Builder compatibility
- **test_tgat_discovery.py**: TGAT discovery enhancement

#### Performance Tests (`tests/performance/`)

- **test_session_processing_performance.py**: Sub-3s processing requirements
- **test_memory_performance.py**: <100MB memory requirements
- **test_accuracy_performance.py**: >95% accuracy validation

#### Contract Tests (`tests/contracts/`)

- **test_golden_invariant_compliance.py**: Golden invariant enforcement
- **test_archaeological_constant_preservation.py**: Archaeological constant validation
- **test_session_isolation_enforcement.py**: Session boundary validation

### Test Data

```python
# Generate test sessions
from agents.archaeological_zone_detector.tests.test_data_generator import (
    generate_test_session,
    generate_enhanced_graph,
    generate_performance_baseline
)

# Create test data
test_session = generate_test_session(
    n_events=100,
    price_range=(99.0, 101.0),
    session_id='test_session_001'
)

test_graph = generate_enhanced_graph(
    n_nodes=50,
    node_feature_dim=45,  # or 51 for HTF
    edge_feature_dim=20
)
```

## Error Handling

### Common Issues

#### Performance Degradation

```python
# Monitor performance degradation
alerts = monitor.trend_analyzer.detect_performance_degradation()

for alert in alerts:
    if alert.severity == AlertSeverity.CRITICAL:
        logger.error(f"Critical performance issue: {alert.message}")
        # Take corrective action
```

#### Contract Violations

```python
from agents.archaeological_zone_detector.contracts import (
    ArchaeologicalContractViolationError,
    ValidationMode
)

try:
    # Strict validation mode (production)
    validator = create_strict_validator(config)
    results = validator.validate_archaeological_output(zones, session_id)
    
except ArchaeologicalContractViolationError as e:
    logger.error(f"Contract violation: {e}")
    # Handle violation according to severity
    
# Development mode (warnings only)
dev_validator = create_development_validator(config)  # Logs warnings but continues
```

#### Memory Issues

```python
# Memory monitoring with automatic GC
memory_usage, gc_triggered = monitor.memory_monitor.check_memory_usage()

if memory_usage > config.performance.memory_warning_threshold:
    logger.warning(f"High memory usage: {memory_usage:.1f}MB")
    
if gc_triggered:
    logger.info("Automatic garbage collection triggered")
```

### Debugging

```python
# Enable debugging mode
config.debugging.debug_mode_enabled = True
config.debugging.zone_detection_debugging = True
config.debugging.temporal_analysis_debugging = True
config.debugging.save_intermediate_results = True

detector = ArchaeologicalZoneDetector(config)

# Debug output will be saved to config.debugging.debug_output_directory
```

## Integration Examples

### Enhanced Graph Builder Integration

```python
from ironforge.integration.ironforge_container import get_ironforge_container

# Get IRONFORGE container
container = get_ironforge_container()
enhanced_graph_builder = container.get_enhanced_graph_builder()

# Create archaeological detector with container
detector = ArchaeologicalZoneDetector(config, container)

# Process session with enhanced graph
nodes_df, edges_df = read_nodes_edges('data/shards/session_001')
enhanced_graph = enhanced_graph_builder.build_graph(nodes_df, edges_df)

# Detect zones with graph integration
analysis = detector.detect_archaeological_zones(
    session_data, 
    previous_session_data,
    enhanced_graph=enhanced_graph
)

# Enhanced features available in analysis.enhanced_features
```

### TGAT Discovery Enhancement

```python
# Standard TGAT discovery
discovery_results = tgat_discovery.forward(enhanced_graph, return_attn=True)

# Enhance with archaeological intelligence
archaeological_analysis = detector.detect_archaeological_zones(session_data)
enhanced_discovery = detector.enhance_tgat_discovery(
    enhanced_graph, archaeological_analysis
)

# Enhanced results include:
enhanced_graph = enhanced_discovery['enhanced_graph']
temporal_features = enhanced_discovery['temporal_features']
authenticity_boost = enhanced_discovery['authenticity_boost']
```

### Pattern Graduation Integration

```python
from ironforge.synthesis.pattern_graduation import PatternGraduation

pattern_graduation = PatternGraduation(archaeological_config=config)

# Graduate patterns with archaeological enhancement
patterns_df = load_discovered_patterns('runs/2025-01-15/patterns/')
graduated_patterns = pattern_graduation.graduate_patterns(
    patterns_df, 
    authenticity_threshold=87.0,
    archaeological_boost=enhanced_discovery['authenticity_boost']
)
```

## Advanced Usage

### Custom Zone Calculation Methods

```python
from agents.archaeological_zone_detector.tools import (
    DimensionalAnchorCalculator,
    RangeCalculationMethod
)

calculator = DimensionalAnchorCalculator(config)

# Different range calculation methods
methods = [
    RangeCalculationMethod.HIGH_LOW,        # Session high - low
    RangeCalculationMethod.OPEN_CLOSE,      # Session close - open  
    RangeCalculationMethod.BODY_RANGE,      # Average candle body
    RangeCalculationMethod.WEIGHTED_AVERAGE # Weighted combination
]

for method in methods:
    session_range = calculator.calculate_session_range(session_data, method)
    anchor_zones = calculator.calculate_dimensional_anchors(
        session_range, current_session_data
    )
```

### Temporal Echo Analysis

```python
from agents.archaeological_zone_detector.tools import TemporalNonLocalityValidator

validator = TemporalNonLocalityValidator(config)

# Detect temporal echoes
temporal_echoes = validator.detect_temporal_echoes(session_data, anchor_zones)

for echo in temporal_echoes:
    print(f"Echo {echo.echo_id}:")
    print(f"  Propagation strength: {echo.propagation_strength:.3f}")
    print(f"  Temporal distance: {echo.temporal_distance}")
    print(f"  Forward validation: {echo.forward_validation}")
    print(f"  Coherence: {echo.coherence_score:.3f}")
```

### Theory B Validation

```python
from agents.archaeological_zone_detector.tools import TheoryBValidator

theory_b_validator = TheoryBValidator(config)

# Validate forward positioning
validation_results = theory_b_validator.validate_forward_positioning(
    session_data, anchor_zones, temporal_analysis
)

print(f"Validation rate: {validation_results['validation_rate']:.1%}")
print(f"Average precision: {validation_results['average_precision_score']:.2f}")

# Calculate precision scores
precision_scores = theory_b_validator.calculate_precision_scores(
    anchor_zones, session_data
)

for score in precision_scores:
    print(f"Zone {score['zone_id']}: {score['precision']:.2f} points")
    print(f"  Target achievement: {score['target_achievement']}")
    print(f"  Grade: {score['precision_grade']}")
```

## Troubleshooting

### Common Issues

#### 1. Processing Time Exceeds Limits

```python
# Optimize configuration for speed
config = ConfigurationPresets.minimal_config()
config.temporal_nonlocality.temporal_echo_detection = False
config.performance.detailed_timing_enabled = False

detector = ArchaeologicalZoneDetector(config)
```

#### 2. Memory Usage Too High

```python
# Enable memory optimization
config.performance.enable_lazy_loading = True
config.performance.garbage_collection_threshold = 50.0  # Lower threshold

# Monitor memory usage
with detector.performance_monitor.time_operation('zone_detection'):
    analysis = detector.detect_archaeological_zones(session_data)
    memory_usage = detector.performance_monitor.memory_monitor.get_memory_stats()
    print(f"Peak memory: {memory_usage['peak_memory_mb']:.1f}MB")
```

#### 3. Low Authenticity Scores

```python
# Debug authenticity calculation
config.debugging.authenticity_scoring_debugging = True

# Check authenticity components
for zone in analysis.archaeological_zones:
    print(f"Zone at {zone.anchor_point}:")
    print(f"  Base confidence: {zone.confidence:.3f}")
    print(f"  Theory B alignment: {zone.theory_b_alignment}")
    print(f"  Precision score: {zone.precision_score:.2f}")
    print(f"  Final authenticity: {zone.authenticity_score:.1f}%")
```

#### 4. Contract Violations

```python
# Check specific contract violations
contract_results = validator.validate_archaeological_output(zones, session_id)

if not contract_results['overall_passed']:
    print("Contract violations detected:")
    for key, passed in contract_results.items():
        if not passed:
            print(f"  - {key}: FAILED")
```

## API Reference

### Main Classes

- **ArchaeologicalZoneDetector**: Main detection agent
- **ArchaeologicalConfig**: Configuration management
- **DimensionalAnchorCalculator**: 40% anchor calculations
- **TemporalNonLocalityValidator**: Theory B validation
- **TheoryBValidator**: Forward positioning validation
- **ArchaeologicalPerformanceMonitor**: Performance monitoring
- **ArchaeologicalContractValidator**: Contract validation

### Data Structures

- **ArchaeologicalZone**: Individual zone with authenticity
- **ArchaeologicalAnalysis**: Complete session analysis
- **TemporalEcho**: Forward-propagating temporal pattern
- **TheoryBResult**: Forward positioning validation result
- **SessionPerformanceData**: Performance metrics per session

### Configuration Classes

- **DimensionalAnchorConfig**: 40% anchoring settings
- **TemporalNonLocalityConfig**: Theory B settings
- **SessionIsolationConfig**: Boundary enforcement
- **AuthenticityConfig**: Authenticity scoring
- **PerformanceConfig**: Performance requirements
- **ValidationConfig**: Contract validation

## Contributing

### Development Setup

```bash
# Clone IRONFORGE
git clone <ironforge-repo>
cd IRONFORGE

# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks  
pre-commit install

# Run tests
pytest agents/archaeological_zone_detector/tests/ -v
```

### Code Standards

- Follow IRONFORGE coding standards
- Maintain golden invariant compliance
- Preserve archaeological constants (40%, 7.55, 87%)
- Ensure session isolation
- Meet performance requirements (<3s, >95%, <100MB)
- Add comprehensive tests for new features

### Testing Requirements

- All new features must have unit tests
- Integration tests for IRONFORGE compatibility
- Performance tests for requirements compliance
- Contract tests for validation compliance
- Coverage must remain >90%

## License

Part of IRONFORGE archaeological intelligence system.

## Support

For issues and questions:
1. Check troubleshooting section
2. Review test cases for examples
3. Consult IRONFORGE documentation
4. Enable debugging mode for detailed analysis

---

**Archaeological Zone Detection Agent v1.0.0**  
*Production-Ready Temporal Pattern Analysis for IRONFORGE*