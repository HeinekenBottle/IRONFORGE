"""
IRONFORGE Container for lazy loading and dependency injection
"""

from iron_core.performance import IRONContainer
import logging

logger = logging.getLogger(__name__)

class IRONFORGEContainer:
    """Container for lazy loading IRONFORGE components"""
    
    def __init__(self):
        self.iron_container = IRONContainer()
        self._components = {}
        logger.info("IRONFORGE Container initialized")
    
    def get_enhanced_graph_builder(self):
        """Get enhanced graph builder with lazy loading"""
        if 'enhanced_graph_builder' not in self._components:
            try:
                from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
                self._components['enhanced_graph_builder'] = EnhancedGraphBuilder()
                logger.info("Enhanced Graph Builder loaded")
            except ImportError as e:
                logger.warning(f"Enhanced Graph Builder not available: {e}")
                # Return minimal stub
                self._components['enhanced_graph_builder'] = object()
        return self._components['enhanced_graph_builder']
    
    def get_tgat_discovery(self):
        """Get TGAT discovery engine with lazy loading"""
        if 'tgat_discovery' not in self._components:
            try:
                from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
                self._components['tgat_discovery'] = IRONFORGEDiscovery()
                logger.info("TGAT Discovery loaded")
            except ImportError as e:
                logger.warning(f"TGAT Discovery not available: {e}")
                # Return minimal stub
                self._components['tgat_discovery'] = object()
        return self._components['tgat_discovery']
    
    def get_pattern_graduation(self):
        """Get pattern graduation system with lazy loading"""
        if 'pattern_graduation' not in self._components:
            try:
                from ironforge.synthesis.pattern_graduation import PatternGraduation
                self._components['pattern_graduation'] = PatternGraduation()
                logger.info("Pattern Graduation loaded")
            except ImportError as e:
                logger.warning(f"Pattern Graduation not available: {e}")
                # Return minimal stub
                self._components['pattern_graduation'] = object()
        return self._components['pattern_graduation']

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