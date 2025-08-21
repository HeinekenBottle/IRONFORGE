"""
Enhanced Graph Builder for 45D/20D architecture
Archaeological discovery graph construction with semantic features

BACKWARD COMPATIBILITY MODULE:
This module now imports from the refactored graph/ submodules while maintaining
all original import paths and functionality for existing code.
"""

import logging
from typing import Any

import networkx as nx
import torch

# Import from new modular structure
from .graph import (
    EnhancedGraphBuilder,
    RichNodeFeature,
    RichEdgeFeature,
    NodeFeatureProcessor,
    EdgeFeatureProcessor,
    FeatureUtils
)

logger = logging.getLogger(__name__)

# Re-export classes for backward compatibility
__all__ = [
    'EnhancedGraphBuilder',
    'RichNodeFeature',
    'RichEdgeFeature'
]
# All classes are now imported from the modular structure above
# This file serves as a backward compatibility layer

# Note: The original implementation has been refactored into:
# - ironforge/learning/graph/node_features.py (RichNodeFeature + semantic processing)
# - ironforge/learning/graph/edge_features.py (RichEdgeFeature + temporal relationships)
# - ironforge/learning/graph/graph_builder.py (EnhancedGraphBuilder core logic)
# - ironforge/learning/graph/feature_utils.py (Shared utilities and calculations)
#
# All original functionality is preserved through the imports above.
