#!/usr/bin/env python3
"""
Oracle System Constants
Centralized constants, error codes, and configuration values for the Oracle training system
"""

from typing import Dict, Set, Pattern
import re

# Oracle Version Information
ORACLE_VERSION = "1.0.2"
ORACLE_MODEL_DIR = "models/oracle/v1.0.2"

# Training Constants
DEFAULT_EARLY_PCT = 0.20
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 15
DEFAULT_WEIGHT_DECAY = 1e-5

# Data Quality Thresholds
MIN_NODES_THRESHOLD = 10
MIN_EDGES_THRESHOLD = 5
MIN_TRAINING_SESSIONS = 10
DEFAULT_MIN_SESSIONS = 57

# Data Quality Levels
QUALITY_LEVELS = {
    "excellent": 4,
    "good": 3, 
    "fair": 2,
    "poor": 1
}

# Oracle Error Codes (centralized from audit.py)
ERROR_CODES = {
    'SUCCESS': 'Session fully processable',
    'SHARD_NOT_FOUND': 'Shard directory does not exist',
    'META_MISSING': 'meta.json file not found',
    'META_INVALID': 'meta.json file corrupted or invalid format',
    'NODES_MISSING': 'nodes.parquet file not found',
    'NODES_CORRUPTED': 'nodes.parquet file unreadable or corrupted',
    'EDGES_MISSING': 'edges.parquet file not found',
    'EDGES_CORRUPTED': 'edges.parquet file unreadable or corrupted',
    'INSUFFICIENT_NODES': 'Too few nodes for meaningful graph construction',
    'INSUFFICIENT_EDGES': 'Too few edges for graph connectivity',
    'DATE_OUT_OF_RANGE': 'Session date outside specified range',
    'TF_MISMATCH': 'Timeframe mismatch between request and shard metadata',
    'SYMBOL_MISMATCH': 'Symbol mismatch between request and shard metadata',
    'TRAINING_DATA_INSUFFICIENT': 'Insufficient training data for model calibration',
    'MODEL_LOAD_FAILED': 'Failed to load trained model weights',
    'SCALER_LOAD_FAILED': 'Failed to load data scaler',
    'EMBEDDING_COMPUTATION_FAILED': 'Failed to compute TGAT embeddings',
    'GRAPH_CONSTRUCTION_FAILED': 'Failed to construct graph from session data'
}

# Session Types (from session_mapping.py)
SESSION_TYPES: Set[str] = {
    'MIDNIGHT', 'PREMARKET', 'LONDON', 'NY', 'NY_AM', 'NY_PM',
    'LUNCH', 'ASIA', 'PREASIA', 'NYAM', 'NYPM'
}

# Timeframe Normalization Patterns (from session_mapping.py)
TF_PATTERNS: Dict[str, callable] = {
    r'^\d+$': lambda x: f'M{x}',      # '5' -> 'M5'
    r'^M\d+$': lambda x: x,           # 'M5' -> 'M5'
    r'^m\d+$': lambda x: x.upper()    # 'm5' -> 'M5'
}

# Compiled regex patterns for performance
COMPILED_TF_PATTERNS: Dict[Pattern, callable] = {
    re.compile(pattern): func for pattern, func in TF_PATTERNS.items()
}

# File Extensions and Patterns
PARQUET_EXT = ".parquet"
JSON_EXT = ".json"
PICKLE_EXT = ".pkl"
PYTORCH_EXT = ".pt"

# Required Files for Oracle Model
ORACLE_MODEL_FILES = {
    'weights': f'weights{PYTORCH_EXT}',
    'scaler': f'scaler{PICKLE_EXT}',
    'manifest': f'training_manifest{JSON_EXT}',
    'metrics': f'metrics{JSON_EXT}'
}

# Default Paths
DEFAULT_DATA_DIR = "data/shards"
DEFAULT_ENHANCED_DIR = "data/enhanced"
DEFAULT_OUTPUT_DIR = "oracle_output"

# TGAT Integration Constants
TGAT_EMBEDDING_DIM = 44
ORACLE_OUTPUT_DIM = 2  # center, half_range
RANGE_HEAD_HIDDEN_DIM = 32

# Training Validation Thresholds
MIN_VALIDATION_SPLIT = 0.1
MAX_VALIDATION_SPLIT = 0.3
DEFAULT_VALIDATION_SPLIT = 0.2

# Performance Thresholds
MAX_TRAINING_TIME_HOURS = 24
MAX_MEMORY_USAGE_GB = 8
MIN_MODEL_ACCURACY = 0.5

# Logging Configuration
LOG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Schema Validation
ORACLE_PREDICTION_SCHEMA_COLUMNS = [
    'run_dir', 'session_date', 'pct_seen', 'n_events',
    'pred_low', 'pred_high', 'center', 'half_range', 'confidence',
    'pattern_id', 'start_ts', 'end_ts',
    'early_expansion_cnt', 'early_retracement_cnt', 'early_reversal_cnt', 'first_seq'
]

# Training Data Schema
TRAINING_PAIRS_SCHEMA_PREFIX = [
    'symbol', 'tf', 'session_date', 'htf_mode', 'early_pct'
]

# Embedding columns (pooled_00 through pooled_43)
EMBEDDING_COLUMNS = [f'pooled_{i:02d}' for i in range(TGAT_EMBEDDING_DIM)]

# Target columns
TARGET_COLUMNS = ['target_center', 'target_half_range']

# Complete training pairs schema
TRAINING_PAIRS_SCHEMA = TRAINING_PAIRS_SCHEMA_PREFIX + EMBEDDING_COLUMNS + TARGET_COLUMNS

# Validation Rules
class ValidationRules:
    """Validation rules for Oracle system components"""
    
    @staticmethod
    def validate_early_pct(early_pct: float) -> bool:
        """Validate early percentage is in valid range"""
        return 0.0 < early_pct <= 1.0
    
    @staticmethod
    def validate_session_type(session_type: str) -> bool:
        """Validate session type is recognized"""
        return session_type.upper() in SESSION_TYPES
    
    @staticmethod
    def validate_quality_level(quality: str) -> bool:
        """Validate data quality level"""
        return quality.lower() in QUALITY_LEVELS
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate timeframe format"""
        for pattern in COMPILED_TF_PATTERNS:
            if pattern.match(timeframe):
                return True
        return False
    
    @staticmethod
    def validate_training_data_schema(df_columns: list) -> bool:
        """Validate training data has required schema"""
        required_columns = set(TRAINING_PAIRS_SCHEMA)
        actual_columns = set(df_columns)
        return required_columns.issubset(actual_columns)
    
    @staticmethod
    def validate_prediction_schema(df_columns: list) -> bool:
        """Validate prediction output has required schema"""
        required_columns = set(ORACLE_PREDICTION_SCHEMA_COLUMNS)
        actual_columns = set(df_columns)
        return required_columns.issubset(actual_columns)

# Success/Failure Return Codes
class ReturnCodes:
    """Standard return codes for Oracle operations"""
    SUCCESS = 0
    GENERAL_ERROR = 1
    CONFIG_ERROR = 2
    DATA_ERROR = 3
    MODEL_ERROR = 4
    VALIDATION_ERROR = 5
    INSUFFICIENT_DATA = 6

# Oracle System Status
class OracleStatus:
    """Oracle system operational status"""
    COLD_START = "cold_start"      # Uncalibrated mode
    CALIBRATED = "calibrated"      # Trained mode
    TRAINING = "training"          # Currently training
    ERROR = "error"                # Error state
    DISABLED = "disabled"          # Disabled state

# Default Configuration
DEFAULT_CONFIG = {
    'version': ORACLE_VERSION,
    'early_pct': DEFAULT_EARLY_PCT,
    'min_sessions': DEFAULT_MIN_SESSIONS,
    'quality_threshold': 'fair',
    'min_events': MIN_NODES_THRESHOLD,
    'training': {
        'epochs': DEFAULT_EPOCHS,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'batch_size': DEFAULT_BATCH_SIZE,
        'patience': DEFAULT_PATIENCE,
        'weight_decay': DEFAULT_WEIGHT_DECAY,
        'validation_split': DEFAULT_VALIDATION_SPLIT
    },
    'paths': {
        'data_dir': DEFAULT_DATA_DIR,
        'enhanced_dir': DEFAULT_ENHANCED_DIR,
        'model_dir': ORACLE_MODEL_DIR,
        'output_dir': DEFAULT_OUTPUT_DIR
    }
}

def get_error_description(error_code: str) -> str:
    """Get human-readable description for error code"""
    return ERROR_CODES.get(error_code, f"Unknown error code: {error_code}")

def normalize_timeframe(timeframe: str) -> str:
    """Normalize timeframe to standard format using compiled patterns"""
    for pattern, func in COMPILED_TF_PATTERNS.items():
        if pattern.match(timeframe):
            return func(timeframe)
    raise ValueError(f"Invalid timeframe format: {timeframe}")

def validate_oracle_config(config: dict) -> tuple[bool, list[str]]:
    """
    Validate Oracle configuration dictionary
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required keys
    required_keys = ['version', 'early_pct', 'min_sessions']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config key: {key}")
    
    # Validate early_pct
    if 'early_pct' in config:
        if not ValidationRules.validate_early_pct(config['early_pct']):
            errors.append(f"Invalid early_pct: {config['early_pct']}")
    
    # Validate min_sessions
    if 'min_sessions' in config:
        if not isinstance(config['min_sessions'], int) or config['min_sessions'] < 1:
            errors.append(f"Invalid min_sessions: {config['min_sessions']}")
    
    return len(errors) == 0, errors
