"""
IRONFORGE Semantic Codebase Indexer
===================================

Deep semantic analysis tool for IRONFORGE project architecture.
Generates AI-friendly reports of the multi-engine system.
"""

from .analyzer import CodeAnalyzer
from .dependency_mapper import DependencyMapper
from .engine_classifier import EngineClassifier
from .indexer import IRONFORGEIndexer
from .report_generator import ReportGenerator

__version__ = "1.0.0"
__all__ = [
    "IRONFORGEIndexer",
    "CodeAnalyzer", 
    "EngineClassifier",
    "DependencyMapper",
    "ReportGenerator"
]