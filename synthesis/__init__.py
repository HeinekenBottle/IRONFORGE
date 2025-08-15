"""
IRONFORGE Synthesis Module
==========================
Pattern graduation and production synthesis components.

Components:
- PatternGraduation: Archaeological pattern validation and graduation
- ProductionGraduation: Production-ready pattern synthesis
"""

from .pattern_graduation import PatternGraduation
from .production_graduation import ProductionGraduation

__all__ = [
    'PatternGraduation',
    'ProductionGraduation'
]