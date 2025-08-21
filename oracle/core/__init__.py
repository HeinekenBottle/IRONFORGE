#!/usr/bin/env python3
"""
Oracle Core Module
Shared constants, exceptions, and utilities for the Oracle training system
"""

# Import all constants for easy access
from .constants import (
    # Version and paths
    ORACLE_VERSION,
    ORACLE_MODEL_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_ENHANCED_DIR,
    DEFAULT_OUTPUT_DIR,
    
    # Training constants
    DEFAULT_EARLY_PCT,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    
    # Thresholds
    MIN_NODES_THRESHOLD,
    MIN_EDGES_THRESHOLD,
    MIN_TRAINING_SESSIONS,
    DEFAULT_MIN_SESSIONS,
    
    # Error codes and session types
    ERROR_CODES,
    SESSION_TYPES,
    QUALITY_LEVELS,
    
    # File patterns
    ORACLE_MODEL_FILES,
    PARQUET_EXT,
    JSON_EXT,
    PICKLE_EXT,
    PYTORCH_EXT,
    
    # Schema definitions
    ORACLE_PREDICTION_SCHEMA_COLUMNS,
    TRAINING_PAIRS_SCHEMA,
    EMBEDDING_COLUMNS,
    TARGET_COLUMNS,
    
    # TGAT integration
    TGAT_EMBEDDING_DIM,
    ORACLE_OUTPUT_DIM,
    RANGE_HEAD_HIDDEN_DIM,
    
    # Validation and utilities
    ValidationRules,
    ReturnCodes,
    OracleStatus,
    DEFAULT_CONFIG,
    
    # Utility functions
    get_error_description,
    normalize_timeframe,
    validate_oracle_config
)

# Import all exceptions
from .exceptions import (
    # Base exceptions
    OracleError,
    OracleConfigError,
    OracleDataError,
    OracleModelError,
    OracleTrainingError,
    OracleValidationError,
    
    # Specific exceptions
    AuditError,
    SessionMappingError,
    DataBuilderError,
    PairsBuilderError,
    NormalizationError,
    EvaluationError,
    TGATIntegrationError,
    GraphConstructionError,
    EmbeddingComputationError,
    
    # Exception factory functions
    create_audit_error,
    create_session_mapping_error,
    create_data_builder_error,
    create_training_error,
    create_model_error,
    create_validation_error,
    create_tgat_integration_error,
    create_embedding_error,
    create_graph_construction_error,
    
    # Error utilities
    add_error_context,
    get_error_summary,
    format_error_for_logging,
    
    # Decorators
    handle_oracle_errors,
    validate_oracle_operation
)

__all__ = [
    # Constants
    'ORACLE_VERSION',
    'ORACLE_MODEL_DIR',
    'DEFAULT_DATA_DIR',
    'DEFAULT_ENHANCED_DIR',
    'DEFAULT_OUTPUT_DIR',
    'DEFAULT_EARLY_PCT',
    'DEFAULT_EPOCHS',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_PATIENCE',
    'DEFAULT_WEIGHT_DECAY',
    'MIN_NODES_THRESHOLD',
    'MIN_EDGES_THRESHOLD',
    'MIN_TRAINING_SESSIONS',
    'DEFAULT_MIN_SESSIONS',
    'ERROR_CODES',
    'SESSION_TYPES',
    'QUALITY_LEVELS',
    'ORACLE_MODEL_FILES',
    'PARQUET_EXT',
    'JSON_EXT',
    'PICKLE_EXT',
    'PYTORCH_EXT',
    'ORACLE_PREDICTION_SCHEMA_COLUMNS',
    'TRAINING_PAIRS_SCHEMA',
    'EMBEDDING_COLUMNS',
    'TARGET_COLUMNS',
    'TGAT_EMBEDDING_DIM',
    'ORACLE_OUTPUT_DIM',
    'RANGE_HEAD_HIDDEN_DIM',
    'ValidationRules',
    'ReturnCodes',
    'OracleStatus',
    'DEFAULT_CONFIG',
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
    'DataBuilderError',
    'PairsBuilderError',
    'NormalizationError',
    'EvaluationError',
    'TGATIntegrationError',
    'GraphConstructionError',
    'EmbeddingComputationError',
    'create_audit_error',
    'create_session_mapping_error',
    'create_data_builder_error',
    'create_training_error',
    'create_model_error',
    'create_validation_error',
    'create_tgat_integration_error',
    'create_embedding_error',
    'create_graph_construction_error',
    'add_error_context',
    'get_error_summary',
    'format_error_for_logging',
    'handle_oracle_errors',
    'validate_oracle_operation'
]

# Module metadata
__version__ = ORACLE_VERSION
__description__ = "Oracle Core - Shared constants, exceptions, and utilities"

def get_oracle_info() -> dict:
    """Get Oracle system information"""
    return {
        'version': ORACLE_VERSION,
        'description': __description__,
        'components': {
            'constants': 'Centralized configuration and constants',
            'exceptions': 'Oracle-specific exception hierarchy',
            'models': 'Data models and structures'
        },
        'features': [
            'Centralized error codes and constants',
            'Comprehensive exception hierarchy',
            'Validation utilities',
            'Configuration management',
            'TGAT integration constants'
        ]
    }
