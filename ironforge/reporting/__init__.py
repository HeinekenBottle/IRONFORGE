"""
IRONFORGE Reporting Module
=========================

Reporting and visualization components for IRONFORGE archaeological discovery.
Generates minidash reports, confluence visualizations, and analysis summaries.
"""

from ..__version__ import __version__
from .minidash import build_minidash
from .writers import write_report

__all__ = ["build_minidash", "write_report"]

# Module metadata
__author__ = "IRONFORGE Team"
__description__ = "Archaeological discovery reporting and visualization"
