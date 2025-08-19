"""
IRONFORGE Reporting Module
=========================

Reporting and visualization components for IRONFORGE archaeological discovery.
Generates minidash reports, confluence visualizations, and analysis summaries.
"""

from .minidash import build_minidash
from .writers import write_report

__all__ = [
    'build_minidash',
    'write_report'
]

# Module metadata
__version__ = "0.7.1"
__author__ = "IRONFORGE Team"
__description__ = "Archaeological discovery reporting and visualization"
