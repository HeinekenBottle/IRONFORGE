"""
IRONFORGE Validation Framework
==============================

Comprehensive validation, testing, and quality assurance framework for IRONFORGE.

Components:
- StatisticalValidator: Statistical pattern quality validation
- RegressionTester: Architectural compliance and regression testing  
- PerformanceMonitor: Performance benchmarking and monitoring
- IntegrationTester: End-to-end integration testing

This framework ensures:
- Pattern quality meets 87% baseline threshold
- Architectural compliance (45D/20D features, session independence)
- Performance SLA compliance 
- Integration health across all components
- Regression detection and prevention
"""

from .statistical_validator import StatisticalValidator
from .regression_tester import RegressionTester  
from .performance_monitor import PerformanceMonitor
from .integration_tester import IntegrationTester

__all__ = [
    'StatisticalValidator',
    'RegressionTester', 
    'PerformanceMonitor',
    'IntegrationTester'
]

__version__ = '1.0.0'
__framework__ = 'IRONFORGE Validation Framework'