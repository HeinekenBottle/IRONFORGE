"""
Core utilities and helpers
"""

import contextlib

# Core utilities
with contextlib.suppress(ImportError):
    from .performance_monitor import PerformanceMonitor

__all__ = ["PerformanceMonitor"]
