"""
IRONFORGE Integration Layer
==========================

Lazy loading and dependency injection for IRONFORGE components
to resolve Sprint 2 timeout issues and achieve performance improvements.
"""

import sys
import os

# Ensure iron_core package can be found
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .ironforge_container import (
    IRONFORGEContainer,
    get_ironforge_container, 
    initialize_ironforge_lazy_loading
)

__all__ = [
    'IRONFORGEContainer',
    'get_ironforge_container',
    'initialize_ironforge_lazy_loading'
]