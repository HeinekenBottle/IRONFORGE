"""
Learning components for archaeological discovery
"""

import contextlib

# Core learning components
with contextlib.suppress(ImportError):
    from .enhanced_graph_builder import EnhancedGraphBuilder

with contextlib.suppress(ImportError):
    from .tgat_discovery import IRONFORGEDiscovery

with contextlib.suppress(ImportError):
    from .simple_event_clustering import SimpleEventClustering

with contextlib.suppress(ImportError):
    from .regime_segmentation import RegimeSegmentation

__all__ = [
    "EnhancedGraphBuilder",
    "IRONFORGEDiscovery",
    "SimpleEventClustering",
    "RegimeSegmentation",
]
