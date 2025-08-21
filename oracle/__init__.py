#!/usr/bin/env python3
"""
Oracle Training System - Refactored Modular Architecture

Enhanced Oracle system with improved organization, shared utilities, and better separation of concerns.
Maintains full backward compatibility while providing cleaner module structure.

The Oracle system is organized into focused submodules:
- core/: Shared constants, exceptions, and utilities
- data/: Data processing, auditing, and session management
- models/: Data models and structures
- evaluation/: Model evaluation and performance analysis

Legacy modules (trainer.py, data_builder.py, etc.) remain in the root for backward compatibility.
"""

# Import core components for easy access
from .core import (
    # Constants and configuration
    ORACLE_VERSION,
    ORACLE_MODEL_DIR,
    DEFAULT_EARLY_PCT,
    MIN_NODES_THRESHOLD,
    MIN_EDGES_THRESHOLD,
    ERROR_CODES,
    SESSION_TYPES,
    QUALITY_LEVELS,
    ValidationRules,
    OracleStatus,

    # Exceptions
    OracleError,
    OracleConfigError,
    OracleDataError,
    OracleModelError,
    OracleTrainingError,
    OracleValidationError,
    AuditError,
    SessionMappingError,

    # Utilities
    get_error_description,
    normalize_timeframe,
    validate_oracle_config
)

# Import data models
from .models import (
    SessionMetadata,
    TrainingPair,
    AuditResult,
    TrainingManifest,
    OraclePrediction,
    TrainingDataset
)

# Import data processing components
from .data import (
    OracleAuditor,
    SessionMapper
)

# Import evaluation components
from .evaluation import (
    OracleEvaluator
)

# Backward compatibility imports - keep existing modules accessible
# These will be gradually migrated to the new structure
try:
    from .trainer import OracleTrainer
except ImportError:
    OracleTrainer = None

try:
    from .data_builder import OracleDataBuilder
except ImportError:
    OracleDataBuilder = None

try:
    from .pairs_builder import OraclePairsBuilder
except ImportError:
    OraclePairsBuilder = None

try:
    from .data_normalizer import OracleDataNormalizer
except ImportError:
    OracleDataNormalizer = None

# Main exports for public API
__all__ = [
    # Version and metadata
    'ORACLE_VERSION',
    '__version__',

    # Core constants and utilities
    'ORACLE_MODEL_DIR',
    'DEFAULT_EARLY_PCT',
    'MIN_NODES_THRESHOLD',
    'MIN_EDGES_THRESHOLD',
    'ERROR_CODES',
    'SESSION_TYPES',
    'QUALITY_LEVELS',
    'ValidationRules',
    'OracleStatus',
    'get_error_description',
    'normalize_timeframe',
    'validate_oracle_config',

    # Exceptions
    'OracleError',
    'OracleConfigError',
    'OracleDataError',
    'OracleModelError',
    'OracleTrainingError',
    'OracleValidationError',
    'AuditError',
    'SessionMappingError',

    # Data models
    'SessionMetadata',
    'TrainingPair',
    'AuditResult',
    'TrainingManifest',
    'OraclePrediction',
    'TrainingDataset',

    # Core components
    'OracleAuditor',
    'SessionMapper',
    'OracleEvaluator',

    # Legacy components (backward compatibility)
    'OracleTrainer',
    'OracleDataBuilder',
    'OraclePairsBuilder',
    'OracleDataNormalizer'
]

# Version information
__version__ = ORACLE_VERSION
__version_info__ = tuple(map(int, ORACLE_VERSION.split('.')))
__description__ = "Oracle Training System - Modular architecture for TGAT range head training"

# Module metadata
ORACLE_INFO = {
    "name": "oracle",
    "version": __version__,
    "description": __description__,
    "architecture": "modular",
    "components": {
        "core": "Shared constants, exceptions, and utilities",
        "data": "Data processing, auditing, and session management",
        "models": "Data models and structures",
        "evaluation": "Model evaluation and performance analysis"
    },
    "features": [
        "Centralized constants and error codes",
        "Comprehensive exception hierarchy",
        "Session discovery and validation",
        "TGAT integration for embeddings",
        "Range head training pipeline",
        "Model evaluation and metrics",
        "Backward compatibility with legacy modules"
    ],
    "dependencies": [
        "ironforge.learning.tgat_discovery",
        "pandas >= 1.3.0",
        "numpy >= 1.21.0",
        "torch >= 1.9.0",
        "scikit-learn >= 1.0.0"
    ]
}

def get_oracle_info() -> dict:
    """Get comprehensive Oracle system information"""
    return ORACLE_INFO.copy()

def list_available_components() -> list:
    """List all available Oracle components"""
    components = []

    # Core components (always available)
    components.extend([
        "OracleAuditor - Session discovery and validation",
        "SessionMapper - Session ID mapping and path resolution",
        "OracleEvaluator - Model evaluation and metrics",
        "SessionMetadata - Session data model",
        "TrainingPair - Training data model",
        "TrainingDataset - Dataset container"
    ])

    # Legacy components (if available)
    if OracleTrainer is not None:
        components.append("OracleTrainer - Range head training pipeline")
    if OracleDataBuilder is not None:
        components.append("OracleDataBuilder - Training data preparation")
    if OraclePairsBuilder is not None:
        components.append("OraclePairsBuilder - Training pairs generation")
    if OracleDataNormalizer is not None:
        components.append("OracleDataNormalizer - Data normalization")

    return components

def create_oracle_auditor(data_dir: str = "data/shards", verbose: bool = False) -> OracleAuditor:
    """
    Create and initialize an OracleAuditor instance

    Args:
        data_dir: Directory containing shard data
        verbose: Enable verbose logging

    Returns:
        Initialized OracleAuditor instance
    """
    return OracleAuditor(data_dir=data_dir, verbose=verbose)

def create_session_mapper(base_shard_dir: str = "data/shards") -> SessionMapper:
    """
    Create and initialize a SessionMapper instance

    Args:
        base_shard_dir: Base directory for shard data

    Returns:
        Initialized SessionMapper instance
    """
    return SessionMapper(base_shard_dir=base_shard_dir)