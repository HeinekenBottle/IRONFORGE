"""
Integration layer for system coordination
"""

# Core integration components
from .ironforge_container import (
    IRONFORGEContainer,
    get_ironforge_container,
    initialize_ironforge_lazy_loading,
)

__all__ = ["get_ironforge_container", "initialize_ironforge_lazy_loading", "IRONFORGEContainer"]
