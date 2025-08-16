"""
IRONFORGE Container for lazy loading and dependency injection
"""

from iron_core.performance import IRONContainer
import logging

logger = logging.getLogger(__name__)

class IRONFORGEContainer:
    """Container for lazy loading IRONFORGE components
    
    NOTE: Components are created fresh for each session to ensure complete session independence.
    No state is shared between sessions.
    """
    
    def __init__(self):
        self.iron_container = IRONContainer()
        # REMOVED: self._components = {} to ensure session independence
        logger.info("IRONFORGE Container initialized")
    
    def get_enhanced_graph_builder(self):
        """Get enhanced graph builder - creates fresh instance for session independence"""
        try:
            from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
            instance = EnhancedGraphBuilder()
            logger.debug("Enhanced Graph Builder created (fresh instance)")
            return instance
        except ImportError as e:
            logger.warning(f"Enhanced Graph Builder not available: {e}")
            # Return minimal stub
            return object()
    
    def get_tgat_discovery(self):
        """Get TGAT discovery engine - creates fresh instance for session independence"""
        try:
            from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
            instance = IRONFORGEDiscovery()
            logger.debug("TGAT Discovery created (fresh instance)")
            return instance
        except ImportError as e:
            logger.warning(f"TGAT Discovery not available: {e}")
            # Return minimal stub
            return object()
    
    def get_pattern_graduation(self):
        """Get pattern graduation system - creates fresh instance for session independence"""
        try:
            from ironforge.synthesis.pattern_graduation import PatternGraduation
            instance = PatternGraduation()
            logger.debug("Pattern Graduation created (fresh instance)")
            return instance
        except ImportError as e:
            logger.warning(f"Pattern Graduation not available: {e}")
            # Return minimal stub
            return object()

# Global container instance
_ironforge_container = None

def get_ironforge_container():
    """Get the global IRONFORGE container instance"""
    global _ironforge_container
    if _ironforge_container is None:
        _ironforge_container = IRONFORGEContainer()
    return _ironforge_container

def initialize_ironforge_lazy_loading():
    """Initialize IRONFORGE lazy loading system"""
    container = get_ironforge_container()
    logger.info("IRONFORGE lazy loading initialized")
    return container