"""
Synthesis components for pattern validation
"""

import contextlib

# Core synthesis components
with contextlib.suppress(ImportError):
    from .pattern_graduation import PatternGraduation

with contextlib.suppress(ImportError):
    from .production_graduation import ProductionGraduation

__all__ = ["PatternGraduation", "ProductionGraduation"]
