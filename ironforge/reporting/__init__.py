"""
IRONFORGE Wave 5 Reporting System
==================================
Visualization and reporting components for temporal pattern discovery.

Provides:
- Timeline heatmap generation for session density visualization
- Confluence strip generation for pattern scoring visualization
- PNG export capabilities
- Integration with discovery pipeline outputs
"""

from .confluence import ConfluenceStripSpec, build_confluence_strip
from .heatmap import TimelineHeatmapSpec, build_session_heatmap

__all__ = [
    "TimelineHeatmapSpec",
    "build_session_heatmap",
    "ConfluenceStripSpec",
    "build_confluence_strip",
]
