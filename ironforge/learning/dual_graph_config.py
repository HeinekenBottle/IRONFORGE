"""
Comprehensive Configuration for Dual Graph Views System
Centralizes all configuration options for DAG construction, M1 integration, 
TGAT enhancement, and motif mining
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)


@dataclass
class DAGConfig:
    """Configuration for DAG construction and acyclicity guarantees"""
    
    # Core DAG parameters
    k_successors: int = 4                # Number of forward connections per node
    dt_min_minutes: int = 1              # Minimum time delta (minutes)
    dt_max_minutes: int = 120            # Maximum time delta (minutes)
    predicate: str = 'WINDOW_KNN'        # Connection strategy
    enabled: bool = True                 # DAG construction enabled
    
    # Causality weights for different event type pairs
    causality_weights: Dict[str, float] = field(default_factory=lambda: {
        'fvg_to_fvg': 0.9,              # FVG formation → redelivery
        'sweep_to_reversal': 0.8,        # Liquidity sweep → reversal
        'expansion_to_retrace': 0.7,     # Market phase transitions
        'premium_discount': 0.6,         # PD array interactions
        'generic_temporal': 0.4          # Default temporal causality
    })
    
    # Acyclicity validation
    strict_acyclicity: bool = True       # Fail if DAG is not acyclic
    validate_topological_sort: bool = True  # Verify topological ordering exists


@dataclass
class M1Config:
    """Configuration for M1 integration and cross-scale features"""
    
    # M1 integration control
    enabled: bool = True                 # Enable M1 event detection and integration
    time_window_minutes: int = 5         # Time window for M1 event analysis around M5 events
    
    # M1 event detection thresholds
    confidence_threshold: float = 0.6    # Minimum confidence for M1 events
    volume_threshold: float = 50.0       # Minimum volume for significant events
    price_change_threshold: float = 0.1  # Minimum price change % for micro events
    
    # Cross-scale edge types and weights
    cross_scale_edges: Dict[str, float] = field(default_factory=lambda: {
        'CONTAINED_IN': 1.0,             # M1 event within M5 bar
        'PRECEDES': 0.8,                 # M1 event sequence
        'INFLUENCES': 0.9                # M1 event affects subsequent M5 bar
    })
    
    # M1-derived feature configuration
    feature_aggregation_method: str = 'weighted_mean'  # How to aggregate M1 features
    max_m1_events_per_window: int = 20  # Limit M1 events per analysis window
    
    # Event types to detect
    event_types: List[str] = field(default_factory=lambda: [
        'micro_fvg_fill', 'micro_sweep', 'micro_impulse', 
        'vwap_touch', 'imbalance_burst', 'wick_extreme'
    ])


@dataclass
class TGATConfig:
    """Configuration for TGAT enhancement with masked attention"""
    
    # Enhanced TGAT control
    enhanced: bool = False               # Use enhanced TGAT with DAG masking
    attention_impl: str = "sdpa"         # "sdpa" | "manual" - attention implementation
    use_edge_mask: bool = True           # Enable edge masking for graph structure
    use_time_bias: str = "bucket"        # "none" | "bucket" | "rbf" - temporal bias type
    is_causal: bool = False              # Set true for strict sequential order
    
    # Attention architecture
    input_dim: int = 45                  # Base input dimension (53 for M1-enhanced)
    hidden_dim: int = 44                 # Hidden dimension for attention
    num_heads: int = 4                   # Number of attention heads
    num_layers: int = 2                  # Number of attention layers
    
    # Enhanced attention features
    temporal_bias_enabled: bool = True   # Use sophisticated temporal bias
    dag_positional_encoding: bool = True # Add DAG-aware positional encoding
    max_sequence_length: int = 1000      # Maximum nodes for positional encoding
    
    # Temporal bias network architecture
    temporal_bias_hidden_dim: int = 22   # Hidden dim for temporal bias network (hidden_dim // 2)
    temporal_encoding_dim: int = 11      # Temporal encoding dimension (hidden_dim // 4)
    
    # Attention masking
    causal_masking: bool = True          # Apply DAG-based causal masking
    self_attention: bool = True          # Allow self-attention in masked version


@dataclass
class MotifConfig:
    """Configuration for DAG motif mining with statistical validation"""
    
    # Motif discovery parameters
    min_nodes: int = 3                   # Minimum nodes in motif
    max_nodes: int = 5                   # Maximum nodes in motif  
    min_frequency: int = 3               # Minimum occurrences to consider
    max_motifs: int = 100                # Maximum motifs to discover
    
    # Statistical validation
    null_iterations: int = 1000          # Iterations for null model generation
    significance_threshold: float = 0.05  # P-value threshold for significance
    lift_threshold: float = 1.5          # Minimum lift ratio for PROMOTE
    confidence_level: float = 0.95       # Confidence level for intervals
    
    # Null model configuration
    time_jitter_range_minutes: int = 120 # ±120 minutes for time jitter nulls
    session_permutation_enabled: bool = True  # Enable session permutation nulls
    
    # Classification thresholds
    promote_lift_threshold: float = 2.0  # Lift ratio for PROMOTE classification
    promote_p_threshold: float = 0.01    # P-value for PROMOTE classification
    park_p_threshold: float = 0.05       # P-value for PARK classification
    
    # Performance optimization
    max_candidate_motifs: int = 10000    # Limit candidate motifs for performance
    parallel_null_generation: bool = True  # Parallelize null model generation


@dataclass
class StorageConfig:
    """Configuration for Parquet storage and serialization"""
    
    # Parquet optimization (based on Context7 recommendations)
    compression: str = 'zstd'            # High compression ratio, fast decompression
    row_group_size: int = 10000          # Optimize for read performance
    engine: str = 'pyarrow'              # Use PyArrow for best performance
    
    # File organization
    partition_by_session: bool = True    # Partition files by session
    include_metadata: bool = True        # Include graph metadata in files
    
    # Feature storage
    feature_precision: str = 'float32'   # Precision for feature storage
    compress_node_features: bool = True  # Compress node feature vectors
    compress_edge_features: bool = True  # Compress edge feature vectors
    
    # Output paths
    dag_output_dir: str = "dual_graphs/dags"      # DAG storage directory
    motifs_output_dir: str = "dual_graphs/motifs" # Motif results directory
    logs_output_dir: str = "dual_graphs/logs"     # Processing logs directory


@dataclass
class DualGraphViewsConfig:
    """Master configuration for the complete Dual Graph Views system"""
    
    # Component configurations
    dag: DAGConfig = field(default_factory=DAGConfig)
    m1: M1Config = field(default_factory=M1Config)
    tgat: TGATConfig = field(default_factory=TGATConfig)
    motifs: MotifConfig = field(default_factory=MotifConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # System-wide settings
    system_name: str = "IRONFORGE Dual Graph Views"
    version: str = "1.0.0"
    
    # Logging and monitoring
    log_level: str = "INFO"              # Logging level
    enable_performance_monitoring: bool = True  # Track performance metrics
    
    # Validation and safety
    validate_inputs: bool = True         # Validate input data structure
    fail_on_warnings: bool = False       # Convert warnings to errors
    
    # Processing control
    max_concurrent_sessions: int = 4     # Maximum sessions to process concurrently
    memory_limit_gb: float = 8.0         # Memory limit for processing
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        
        # Validate configuration consistency
        self._validate_config()
        
        # Adjust M1 input dimension for TGAT if M1 is enabled
        if self.m1.enabled:
            self.tgat.input_dim = 53  # 45 + 8 M1-derived features
        else:
            self.tgat.input_dim = 45  # Standard features only
        
        # Ensure temporal bias hidden dimensions are consistent
        self.tgat.temporal_bias_hidden_dim = self.tgat.hidden_dim // 2
        self.tgat.temporal_encoding_dim = self.tgat.hidden_dim // 4
        
        logger.info(f"Dual Graph Views configured: DAG={self.dag.enabled}, "
                   f"M1={self.m1.enabled}, Enhanced TGAT={self.tgat.enhanced}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        
        # DAG validation
        if self.dag.dt_min_minutes >= self.dag.dt_max_minutes:
            raise ValueError("DAG dt_min_minutes must be < dt_max_minutes")
        
        if self.dag.k_successors < 1:
            raise ValueError("DAG k_successors must be >= 1")
        
        # M1 validation
        if self.m1.confidence_threshold < 0 or self.m1.confidence_threshold > 1:
            raise ValueError("M1 confidence_threshold must be between 0 and 1")
        
        # TGAT validation
        if self.tgat.hidden_dim % self.tgat.num_heads != 0:
            raise ValueError("TGAT hidden_dim must be divisible by num_heads")
        
        # Motif validation
        if self.motifs.min_nodes > self.motifs.max_nodes:
            raise ValueError("Motifs min_nodes must be <= max_nodes")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DualGraphViewsConfig':
        """Create configuration from dictionary"""
        
        # Extract component configurations
        components = {}
        
        for component_name in ['dag', 'm1', 'tgat', 'motifs', 'storage']:
            if component_name in config_dict:
                component_class = {
                    'dag': DAGConfig,
                    'm1': M1Config, 
                    'tgat': TGATConfig,
                    'motifs': MotifConfig,
                    'storage': StorageConfig
                }[component_name]
                
                components[component_name] = component_class(**config_dict[component_name])
        
        # Extract system-wide settings
        system_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['dag', 'm1', 'tgat', 'motifs', 'storage']}
        
        return cls(**components, **system_settings)
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'DualGraphViewsConfig':
        """Load configuration from JSON file"""
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        
        result = {}
        
        # Component configurations
        for component_name in ['dag', 'm1', 'tgat', 'motifs', 'storage']:
            component = getattr(self, component_name)
            result[component_name] = {
                field.name: getattr(component, field.name)
                for field in component.__dataclass_fields__.values()
            }
        
        # System-wide settings
        system_fields = [f for f in self.__dataclass_fields__.keys() 
                        if f not in ['dag', 'm1', 'tgat', 'motifs', 'storage']]
        
        for field_name in system_fields:
            result[field_name] = getattr(self, field_name)
        
        return result
    
    def save_to_file(self, config_path: Path):
        """Save configuration to JSON file"""
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def create_preset_configs(self) -> Dict[str, 'DualGraphViewsConfig']:
        """Create common preset configurations"""
        
        presets = {}
        
        # Minimal configuration for testing
        minimal = DualGraphViewsConfig()
        minimal.dag.k_successors = 2
        minimal.m1.enabled = False
        minimal.tgat.enhanced = False
        minimal.motifs.null_iterations = 100
        presets['minimal'] = minimal
        
        # Standard production configuration
        standard = DualGraphViewsConfig()
        standard.dag.k_successors = 4
        standard.m1.enabled = True
        standard.tgat.enhanced = False  # Standard TGAT
        standard.motifs.null_iterations = 1000
        presets['standard'] = standard
        
        # Enhanced configuration with all features
        enhanced = DualGraphViewsConfig()
        enhanced.dag.k_successors = 6
        enhanced.m1.enabled = True
        enhanced.tgat.enhanced = True   # Enhanced TGAT with masking
        enhanced.tgat.attention_impl = "sdpa"
        enhanced.tgat.use_edge_mask = True
        enhanced.tgat.use_time_bias = "bucket"
        enhanced.motifs.null_iterations = 2000
        enhanced.max_concurrent_sessions = 8
        presets['enhanced'] = enhanced
        
        # Research configuration for maximum discovery
        research = DualGraphViewsConfig()
        research.dag.k_successors = 8
        research.m1.enabled = True
        research.m1.confidence_threshold = 0.4  # Lower threshold
        research.tgat.enhanced = True
        research.tgat.num_layers = 3
        research.motifs.min_frequency = 2  # Lower frequency requirement
        research.motifs.null_iterations = 5000
        research.motifs.max_motifs = 500
        presets['research'] = research
        
        return presets


def load_config_with_overrides(
    base_config_path: Optional[Path] = None,
    preset: str = 'standard', 
    overrides: Optional[Dict[str, Any]] = None
) -> DualGraphViewsConfig:
    """
    Load configuration with flexible override system
    
    Args:
        base_config_path: Path to base configuration file (optional)
        preset: Preset configuration name to start with
        overrides: Dictionary of configuration overrides
        
    Returns:
        Configured DualGraphViewsConfig instance
    """
    
    # Start with preset
    base_config = DualGraphViewsConfig()
    presets = base_config.create_preset_configs()
    
    if preset in presets:
        config = presets[preset]
        logger.info(f"Using '{preset}' preset configuration")
    else:
        config = DualGraphViewsConfig()
        logger.warning(f"Unknown preset '{preset}', using default configuration")
    
    # Load from file if provided
    if base_config_path and base_config_path.exists():
        file_config = DualGraphViewsConfig.from_file(base_config_path)
        # Merge configurations (file overrides preset)
        config_dict = config.to_dict()
        file_dict = file_config.to_dict()
        
        # Deep merge
        for key, value in file_dict.items():
            if isinstance(value, dict) and key in config_dict:
                config_dict[key].update(value)
            else:
                config_dict[key] = value
        
        config = DualGraphViewsConfig.from_dict(config_dict)
        logger.info(f"Configuration loaded from {base_config_path}")
    
    # Apply overrides
    if overrides:
        config_dict = config.to_dict()
        
        # Apply nested overrides
        for key, value in overrides.items():
            if '.' in key:  # Nested key like 'dag.k_successors'
                component, field = key.split('.', 1)
                if component in config_dict:
                    config_dict[component][field] = value
            else:
                config_dict[key] = value
        
        config = DualGraphViewsConfig.from_dict(config_dict)
        logger.info(f"Applied {len(overrides)} configuration overrides")
    
    return config


# Example usage and factory functions
def create_development_config() -> DualGraphViewsConfig:
    """Create configuration optimized for development and testing"""
    return load_config_with_overrides(
        preset='minimal',
        overrides={
            'log_level': 'DEBUG',
            'motifs.null_iterations': 50,
            'storage.row_group_size': 1000,
            'fail_on_warnings': True
        }
    )


def create_production_config() -> DualGraphViewsConfig:
    """Create configuration optimized for production use"""
    return load_config_with_overrides(
        preset='standard',
        overrides={
            'log_level': 'INFO',
            'enable_performance_monitoring': True,
            'storage.compression': 'zstd',
            'max_concurrent_sessions': 6
        }
    )


def create_research_config() -> DualGraphViewsConfig:
    """Create configuration optimized for research and discovery"""
    return load_config_with_overrides(
        preset='research',
        overrides={
            'log_level': 'DEBUG',
            'tgat.num_layers': 4,
            'motifs.significance_threshold': 0.1,
            'memory_limit_gb': 16.0
        }
    )