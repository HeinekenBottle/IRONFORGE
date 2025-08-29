"""
IRONFORGE Discovery Module
TGAT Archaeological Memory with Cross-Session Persistence

This module implements TGAT-based discovery workflows with archaeological memory
that persists and evolves across sessions for continuous pattern learning.
"""

from .tgat_memory_workflows import (
    TGATMemoryWorkflow,
    ArchaeologicalMemoryState,
    EnhancedDiscovery,
    TGATInput,
    TGATDiscoveryActivity,
    ArchaeologicalMemoryManager
)

__all__ = [
    "TGATMemoryWorkflow",
    "ArchaeologicalMemoryState",
    "EnhancedDiscovery",
    "TGATInput",
    "TGATDiscoveryActivity", 
    "ArchaeologicalMemoryManager"
]