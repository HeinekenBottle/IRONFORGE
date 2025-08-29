"""
BMAD Performance Monitoring Module

Provides comprehensive performance monitoring and optimization for BMAD
temporal metamorphosis detection with sub-3-second processing targets.
"""

from .performance_tracker import (
    BMadPerformanceTracker,
    PerformanceMetrics,
    get_performance_tracker,
    initialize_performance_monitoring,
    shutdown_performance_monitoring
)

__all__ = [
    'BMadPerformanceTracker',
    'PerformanceMetrics', 
    'get_performance_tracker',
    'initialize_performance_monitoring',
    'shutdown_performance_monitoring'
]