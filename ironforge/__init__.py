"""
IRONFORGE - Archaeological Discovery System
==========================================

IRONFORGE is a sophisticated archaeological discovery system that uncovers hidden patterns 
in financial market data using advanced temporal graph attention networks (TGAT) and 
semantic feature analysis.

Core Mission: Archaeological discovery of market patterns (NOT prediction)
Architecture: Component-based lazy loading with iron-core integration
Performance: 88.7% improvement through lazy loading

Key Components:
- learning: TGAT discovery engine and pattern learning
- analysis: Pattern analysis and market archaeology  
- synthesis: Pattern validation and production bridge
- integration: Iron-core integration and lazy loading
- core: Core business logic and orchestration

Usage:
    from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
    from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
    from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
    
    # Initialize lazy loading container
    container = initialize_ironforge_lazy_loading()
    
    # Get components through container
    graph_builder = container.get_enhanced_graph_builder()
    discovery_engine = container.get_tgat_discovery()
"""

from iron_core import __version__ as iron_core_version

__version__ = "1.0.0"
__author__ = "IRON Ecosystem"
__description__ = "Archaeological discovery system for market pattern analysis"

# Core exports for easy access
__all__ = [
    '__version__',
    '__author__', 
    '__description__'
]

# Lazy loading note: Components are accessed through the container system
# to maintain the 88.7% performance improvement from lazy loading
