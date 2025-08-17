"""
IRONFORGE Validation Framework (Waves 1-4)
==========================================

Comprehensive validation, testing, and quality assurance framework for IRONFORGE.

Core Framework Components:
- StatisticalValidator: Statistical pattern quality validation
- RegressionTester: Architectural compliance and regression testing  
- PerformanceMonitor: Performance benchmarking and monitoring
- IntegrationTester: End-to-end integration testing

Wave 4 Validation Rails:
- Time-series safe splits (PurgedKFold, OOS, Holdout)
- Negative controls (time shuffle, label permutation, structural controls)
- Comprehensive metrics (temporal AUC, precision@k, stability, archaeological significance)
- Automated validation orchestration and reporting

This framework ensures:
- Pattern quality meets 87% baseline threshold
- Architectural compliance (45D/20D features, session independence)
- Performance SLA compliance 
- Integration health across all components
- Regression detection and prevention
- Time-series safe evaluation without look-ahead bias
- Robust validation against spurious patterns
"""

# Core validation framework
from .statistical_validator import StatisticalValidator
from .regression_tester import RegressionTester
from .performance_monitor import PerformanceMonitor
from .integration_tester import IntegrationTester

# Wave 4 validation rails
from .splits import PurgedKFold, oos_split, temporal_train_test_split
from .controls import (
    create_control_variants,
    time_shuffle_edges,
    label_permutation,
    node_feature_shuffle,
    edge_direction_shuffle,
    temporal_block_shuffle,
)
from .metrics import (
    precision_at_k,
    temporal_auc,
    motif_half_life,
    pattern_stability_score,
    archaeological_significance,
    compute_validation_metrics,
)
from .runner import ValidationConfig, ValidationRunner

__all__ = [
    # Core framework
    "StatisticalValidator", "RegressionTester", "PerformanceMonitor", "IntegrationTester",
    # Wave 4 splits
    "PurgedKFold", "oos_split", "temporal_train_test_split",
    # Wave 4 controls
    "create_control_variants", "time_shuffle_edges", "label_permutation",
    "node_feature_shuffle", "edge_direction_shuffle", "temporal_block_shuffle",
    # Wave 4 metrics
    "precision_at_k", "temporal_auc", "motif_half_life", "pattern_stability_score",
    "archaeological_significance", "compute_validation_metrics",
    # Wave 4 runner
    "ValidationConfig", "ValidationRunner",
]

__version__ = "1.0.0"
__framework__ = "IRONFORGE Validation Framework"
