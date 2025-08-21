"""Shared constants and configuration values for IRONFORGE."""

# Architecture Constants
NODE_FEATURE_DIM_STANDARD = 45  # Standard node feature dimensions
NODE_FEATURE_DIM_HTF = 51       # HTF (High Timeframe) enhanced dimensions  
EDGE_FEATURE_DIM = 20           # Edge feature dimensions
TGAT_EMBEDDING_DIM = 44         # TGAT output embedding dimensions
ORACLE_OUTPUT_DIM = 2           # Oracle range head output (center, half_range)

# Event Types (canonical taxonomy)
EVENT_TYPES = [
    "Expansion",        # Market range extension
    "Consolidation",    # Range compression
    "Retracement",      # Partial reversal within trend
    "Reversal",         # Full directional change
    "Liquidity Taken",  # Order flow absorption
    "Redelivery"        # Return to prior levels
]

# Edge Intent Types (canonical taxonomy)
EDGE_INTENTS = [
    "TEMPORAL_NEXT",        # Sequential time progression
    "MOVEMENT_TRANSITION",  # Price movement relationships
    "LIQ_LINK",            # Liquidity flow connections
    "CONTEXT"              # Contextual relationships
]

# Training Constants
DEFAULT_EARLY_PCT = 0.20           # Default early batch percentage for Oracle
MIN_TRAINING_SESSIONS = 10         # Minimum sessions for training
DEFAULT_MIN_SESSIONS = 57          # Default minimum sessions for Oracle audit
DEFAULT_EPOCHS = 100               # Default training epochs
DEFAULT_LEARNING_RATE = 0.001      # Default learning rate
DEFAULT_BATCH_SIZE = 32            # Default batch size
DEFAULT_PATIENCE = 15              # Default early stopping patience

# Oracle Sidecar Schema
ORACLE_SIDECAR_COLUMNS = [
    "run_dir", "session_date", "pct_seen", "n_events", 
    "pred_low", "pred_high", "pred_center", "pred_half_range",
    "confidence", "pattern_id", "start_ts", "end_ts",
    "early_expansion_cnt", "early_retracement_cnt", 
    "early_reversal_cnt", "first_seq"
]
ORACLE_SIDECAR_COLUMN_COUNT = 16

# File Extensions and Patterns
PARQUET_EXT = ".parquet"
JSON_EXT = ".json"
SHARD_PREFIX = "shard_"
ENHANCED_SESSION_PATTERN = "enhanced_*_Lvl-1_*.json"

# Default Paths
DEFAULT_DATA_DIR = "data/shards"
DEFAULT_CONFIG_PATH = "configs/dev.yml"
DEFAULT_RUNS_DIR = "runs"

# Validation Thresholds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MIN_QUALITY_THRESHOLD = "fair"
MIN_EVENTS_PER_SESSION = 10

# Timeframe Constants
TIMEFRAME_PATTERNS = {
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440
}

# CLI Return Codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2