"""
Learning components for archaeological discovery
"""

# Core learning components
try:
    from .enhanced_graph_builder import EnhancedGraphBuilder
except ImportError:
    # Component not yet available
    pass

try:
    from .tgat_discovery import IRONFORGEDiscovery
except ImportError:
    # Component not yet available
    pass

try:
    from .simple_event_clustering import SimpleEventClustering
except ImportError:
    # Component not yet available
    pass

try:
    from .regime_segmentation import RegimeSegmentation
except ImportError:
    # Component not yet available
    pass

__all__ = [
    "EnhancedGraphBuilder",
    "IRONFORGEDiscovery",
    "SimpleEventClustering",
    "RegimeSegmentation",
]
