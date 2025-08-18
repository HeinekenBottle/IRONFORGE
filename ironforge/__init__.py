"""
IRONFORGE Archaeological Discovery System
Package version and main exports
"""

__version__ = "1.0.0"

# Core exports for easy access
from .integration.ironforge_container import (
    get_ironforge_container,
    initialize_ironforge_lazy_loading,
)

__all__ = ["__version__", "get_ironforge_container", "initialize_ironforge_lazy_loading"]
