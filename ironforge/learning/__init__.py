"""
IRONFORGE Learning Module
========================
Archaeological discovery and pattern learning components.

Core Components:
- EnhancedGraphBuilder: 37D rich feature graph construction
- IRONFORGEDiscovery: TGAT-based archaeological discovery
- RegimeSegmentation: DBSCAN clustering for pattern analysis
- EventPrecursorDetector: Temporal cycles + structural context detection
"""

from .enhanced_graph_builder import EnhancedGraphBuilder
from .tgat_discovery import IRONFORGEDiscovery, TGAT
from .regime_segmentation import RegimeSegmentation
from .precursor_detection import EventPrecursorDetector
from .graph_builder import IRONFORGEGraphBuilder

__all__ = [
    'EnhancedGraphBuilder',
    'IRONFORGEDiscovery', 
    'TGAT',
    'RegimeSegmentation',
    'EventPrecursorDetector',
    'IRONFORGEGraphBuilder'
]