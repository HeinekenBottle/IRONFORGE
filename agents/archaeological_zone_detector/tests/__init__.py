"""
Archaeological Zone Detection Agent Test Suite
==============================================

Comprehensive test suite for archaeological intelligence within IRONFORGE.
Tests validate production-grade performance, accuracy, and integration compliance.

Test Categories:
- Unit Tests: Individual component validation (tools, contracts, performance)
- Integration Tests: IRONFORGE pipeline integration and Enhanced Graph Builder compatibility
- Performance Tests: Sub-3s processing, >95% accuracy, <100MB memory requirements
- Contract Tests: Golden invariant enforcement and data contract compliance
- End-to-End Tests: Complete archaeological analysis workflows

Test Requirements:
- All tests must complete within performance requirements
- Contract validation must pass in strict mode
- Archaeological constants (40% anchoring, 7.55 precision) must be preserved
- Session isolation must be absolute (no cross-session contamination)
- HTF compliance must be enforced (last-closed only)

Usage:
    pytest tests/                           # Run all tests
    pytest tests/unit/                      # Run unit tests only
    pytest tests/integration/               # Run integration tests only  
    pytest tests/performance/               # Run performance tests only
    pytest tests/contracts/                 # Run contract tests only
    pytest tests/end_to_end/                # Run end-to-end tests only
    
    pytest -v --tb=short                    # Verbose output with short tracebacks
    pytest --benchmark-only                 # Run benchmarks only
    pytest --cov=agents.archaeological_zone_detector  # Coverage report
"""

import sys
from pathlib import Path

# Add agent module to Python path for testing
agent_path = Path(__file__).parent.parent
if str(agent_path) not in sys.path:
    sys.path.insert(0, str(agent_path))

# Test configuration constants
TEST_CONFIG = {
    "max_test_processing_time": 5.0,  # Allow extra time for test overhead
    "min_test_accuracy": 90.0,        # Slightly relaxed for test data
    "max_test_memory_mb": 150.0,      # Allow extra memory for test framework
    "test_session_count": 10,         # Number of test sessions for performance tests
    "benchmark_iterations": 100,      # Iterations for benchmark tests
    "precision_tolerance": 0.5,       # Tolerance for precision target testing
    "authenticity_tolerance": 5.0,    # Tolerance for authenticity testing
}

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
SAMPLE_SESSIONS_PATH = TEST_DATA_DIR / "sample_sessions.json"
ENHANCED_GRAPHS_PATH = TEST_DATA_DIR / "enhanced_graphs.pkl"
PERFORMANCE_BASELINES_PATH = TEST_DATA_DIR / "performance_baselines.json"

__version__ = "1.0.0"