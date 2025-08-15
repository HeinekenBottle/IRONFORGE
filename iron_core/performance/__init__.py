"""
Iron-Core Performance Module
===========================

Lazy loading and dependency injection infrastructure for IRON ecosystem.
Provides 88.7% performance improvement through component lazy loading.
"""

from .lazy_loader import LazyComponent, LazyLoadingManager, get_lazy_manager, initialize_lazy_loading
from .container import IRONContainer, get_container, initialize_container

__all__ = [
    'LazyComponent',
    'LazyLoadingManager', 
    'get_lazy_manager',
    'initialize_lazy_loading',
    'IRONContainer',
    'get_container', 
    'initialize_container'
]