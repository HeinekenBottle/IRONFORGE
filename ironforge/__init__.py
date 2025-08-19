"""
IRONFORGE Archaeological Discovery System
Package version and main exports
"""

from .__version__ import __version__, __version_info__

# Core exports for easy access
from .integration.ironforge_container import (
    get_ironforge_container,
    initialize_ironforge_lazy_loading,
)

__all__ = ["__version__", "__version_info__", "get_ironforge_container", "initialize_ironforge_lazy_loading"]
