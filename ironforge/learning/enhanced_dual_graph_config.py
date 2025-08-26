"""
Enhanced Dual Graph Configuration with Context7 Optimizations
Extended configuration system supporting performance optimizations

Key additions:
1. Context7 optimization toggles
2. Performance tuning parameters  
3. Backward compatibility
4. Configuration validation
5. Preset management
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json

from .dual_graph_config import (
    DAGConfig, M1Config, TGATConfig, MotifConfig, StorageConfig, 
    DualGraphViewsConfig, load_config_with_overrides
)

logger = logging.getLogger(__name__)


@dataclass
class Context7OptimizationConfig:
    """Context7 performance optimization configuration"""
    
    # Global optimization control
    enable_all_optimizations: bool = False
    enable_performance_monitoring: bool = True
    
    # TGAT optimizations (Context7 PyTorch recommendations)
    tgat_enable_amp: bool = False                    # Automatic Mixed Precision
    tgat_enable_flash_attention: bool = False        # Flash attention via SDPA
    tgat_enable_block_sparse_mask: bool = False      # Block-sparse attention patterns
    tgat_enable_time_bias_caching: bool = True       # Cache time bias computations
    tgat_use_fp16: bool = False                      # FP16 precision
    tgat_enable_fused_ops: bool = False              # Fused operations
    tgat_gradient_checkpointing: bool = False        # Memory-efficient training
    
    # DAG optimization (Context7 NetworkX recommendations)  
    dag_enable_vectorized_ops: bool = True           # Vectorized edge operations
    dag_enable_topological_generations: bool = True  # Use topological_generations
    dag_enable_sparse_adjacency: bool = True         # Sparse adjacency matrices
    dag_enable_batch_edge_creation: bool = True      # Batch edge operations
    dag_parallel_validation: bool = True             # Parallel DAG validation
    dag_cache_adjacency_matrix: bool = True          # Cache adjacency matrices
    
    # Parquet optimizations (Context7 PyArrow recommendations)
    parquet_enable_zstd_optimization: bool = True    # ZSTD compression tuning
    parquet_enable_content_chunking: bool = True     # Content-defined chunking
    parquet_optimize_row_groups: bool = True         # Optimal row group sizes
    parquet_enable_dtype_optimization: bool = True   # Data type optimization
    parquet_enable_parallel_io: bool = True          # Parallel I/O operations
    parquet_use_memory_map: bool = True              # Memory-mapped reading
    
    # Advanced performance tuning
    max_memory_usage_gb: float = 8.0                # Memory limit
    cpu_thread_count: Optional[int] = None          # CPU thread count (None = auto)
    gpu_memory_fraction: float = 0.8                # GPU memory fraction
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        
        # Auto-enable GPU optimizations if CUDA available
        try:
            import torch
            if torch.cuda.is_available():
                if self.enable_all_optimizations:
                    self.tgat_enable_amp = True
                    self.tgat_enable_flash_attention = True
                    self.tgat_use_fp16 = True
                    logger.info("Auto-enabled GPU optimizations (CUDA available)")
        except ImportError:
            logger.warning("PyTorch not available, GPU optimizations disabled")
            
        # Set CPU thread count if not specified
        if self.cpu_thread_count is None:
            import os
            self.cpu_thread_count = os.cpu_count()
            
        # Validate memory settings
        if self.max_memory_usage_gb <= 0:
            raise ValueError("max_memory_usage_gb must be positive")
            
        if not 0 < self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be between 0 and 1")


@dataclass 
class EnhancedTGATConfig(TGATConfig):
    """Enhanced TGAT configuration with Context7 optimizations"""
    
    # Context7 optimization flags
    c7_optimizations: Context7OptimizationConfig = field(default_factory=Context7OptimizationConfig)
    
    # Advanced attention parameters  
    sparse_attention_density: float = 0.1           # Density for sparse attention
    block_size: int = 64                           # Block size for block-sparse attention
    time_bias_cache_size: int = 128               # LRU cache size for time bias
    
    # Memory optimization
    enable_gradient_checkpointing: bool = False    # Gradient checkpointing
    memory_efficient_attention: bool = True        # Memory-efficient attention
    
    # Performance monitoring
    profile_attention_performance: bool = False    # Profile attention layers
    log_memory_usage: bool = False                # Log memory usage
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of enabled optimizations"""
        
        summary = {
            'amp_enabled': self.c7_optimizations.tgat_enable_amp,
            'flash_attention': self.c7_optimizations.tgat_enable_flash_attention,
            'fp16_precision': self.c7_optimizations.tgat_use_fp16,
            'time_bias_caching': self.c7_optimizations.tgat_enable_time_bias_caching,
            'block_sparse_mask': self.c7_optimizations.tgat_enable_block_sparse_mask,
            'fused_operations': self.c7_optimizations.tgat_enable_fused_ops,
            'gradient_checkpointing': self.enable_gradient_checkpointing,
            'memory_efficient': self.memory_efficient_attention
        }
        
        enabled_count = sum(1 for v in summary.values() if v)
        summary['total_optimizations_enabled'] = enabled_count
        
        return summary


@dataclass
class EnhancedDAGConfig(DAGConfig):
    """Enhanced DAG configuration with Context7 optimizations"""
    
    # Context7 optimization flags
    c7_optimizations: Context7OptimizationConfig = field(default_factory=Context7OptimizationConfig)
    
    # Vectorization parameters
    edge_batch_size: int = 1000                    # Batch size for edge operations
    parallel_workers: int = 4                      # Number of parallel workers
    
    # Caching parameters
    adjacency_cache_size: int = 100               # Adjacency matrix cache size
    topological_cache_enabled: bool = True        # Cache topological information
    
    # Advanced DAG construction
    use_csr_adjacency: bool = True                # Use CSR sparse format
    enable_dag_analytics: bool = True             # Enable detailed analytics
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of enabled DAG optimizations"""
        
        summary = {
            'vectorized_operations': self.c7_optimizations.dag_enable_vectorized_ops,
            'topological_generations': self.c7_optimizations.dag_enable_topological_generations,
            'sparse_adjacency': self.c7_optimizations.dag_enable_sparse_adjacency,
            'batch_edge_creation': self.c7_optimizations.dag_enable_batch_edge_creation,
            'parallel_validation': self.c7_optimizations.dag_parallel_validation,
            'adjacency_caching': self.c7_optimizations.dag_cache_adjacency_matrix,
            'csr_format': self.use_csr_adjacency,
            'dag_analytics': self.enable_dag_analytics
        }
        
        enabled_count = sum(1 for v in summary.values() if v)
        summary['total_optimizations_enabled'] = enabled_count
        
        return summary


@dataclass
class EnhancedStorageConfig(StorageConfig):
    """Enhanced storage configuration with Context7 Parquet optimizations"""
    
    # Context7 optimization flags
    c7_optimizations: Context7OptimizationConfig = field(default_factory=Context7OptimizationConfig)
    
    # ZSTD optimization (Context7 recommendation)
    zstd_compression_level: int = 3               # Optimal ZSTD level
    
    # Content-defined chunking parameters
    content_chunking_min_size: int = 256 * 1024  # 256 KiB
    content_chunking_max_size: int = 1024 * 1024 # 1 MiB
    
    # Row group optimization
    optimal_row_group_size: int = 10000           # Context7 recommended size
    
    # Data type optimization
    dtype_optimization_enabled: bool = True       # Enable dtype optimization
    dictionary_encoding_columns: List[str] = field(default_factory=lambda: [
        'event_type', 'session_id'
    ])
    
    # I/O performance
    max_open_files: int = 900                     # Context7 recommendation
    use_memory_mapping: bool = True               # Memory-mapped I/O
    parallel_io_workers: int = 4                  # Parallel I/O workers
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of enabled storage optimizations"""
        
        summary = {
            'zstd_optimization': self.c7_optimizations.parquet_enable_zstd_optimization,
            'content_chunking': self.c7_optimizations.parquet_enable_content_chunking,
            'row_group_optimization': self.c7_optimizations.parquet_optimize_row_groups,
            'dtype_optimization': self.c7_optimizations.parquet_enable_dtype_optimization,
            'parallel_io': self.c7_optimizations.parquet_enable_parallel_io,
            'memory_mapping': self.c7_optimizations.parquet_use_memory_map,
            'compression': self.compression,
            'compression_level': self.zstd_compression_level,
            'dictionary_encoding': len(self.dictionary_encoding_columns) > 0
        }
        
        enabled_count = sum(1 for v in summary.values() if v)
        summary['total_optimizations_enabled'] = enabled_count
        
        return summary


@dataclass
class EnhancedDualGraphViewsConfig(DualGraphViewsConfig):
    """Enhanced Dual Graph Views configuration with Context7 optimizations"""
    
    # Enhanced component configurations
    dag: EnhancedDAGConfig = field(default_factory=EnhancedDAGConfig)
    m1: M1Config = field(default_factory=M1Config)  # Keep original for now
    tgat: EnhancedTGATConfig = field(default_factory=EnhancedTGATConfig)
    motifs: MotifConfig = field(default_factory=MotifConfig)  # Keep original
    storage: EnhancedStorageConfig = field(default_factory=EnhancedStorageConfig)
    
    # Global Context7 optimizations
    c7_optimizations: Context7OptimizationConfig = field(default_factory=Context7OptimizationConfig)
    
    # System performance monitoring
    enable_detailed_profiling: bool = False       # Enable detailed performance profiling
    performance_log_interval: int = 100           # Log performance every N operations
    
    def __post_init__(self):
        """Enhanced post-initialization with optimization setup"""
        
        # Call parent post-init
        super().__post_init__()
        
        # Propagate global optimizations to components
        self._propagate_optimizations()
        
        # Validate optimization compatibility
        self._validate_optimization_compatibility()
        
        # Setup performance monitoring
        self._setup_performance_monitoring()
        
        logger.info(f"Enhanced Dual Graph Views configured with Context7 optimizations")
        logger.info(f"Optimization summary: {self.get_optimization_summary()}")
        
    def _propagate_optimizations(self):
        """Propagate global optimization flags to component configs"""
        
        # Propagate to TGAT
        self.tgat.c7_optimizations = self.c7_optimizations
        
        # Propagate to DAG  
        self.dag.c7_optimizations = self.c7_optimizations
        
        # Propagate to Storage
        self.storage.c7_optimizations = self.c7_optimizations
        
        # Update component-specific optimizations based on global settings
        if self.c7_optimizations.enable_all_optimizations:
            self._enable_all_component_optimizations()
            
    def _enable_all_component_optimizations(self):
        """Enable all available optimizations for maximum performance"""
        
        # Enable TGAT optimizations
        self.tgat.enable_gradient_checkpointing = True
        self.tgat.memory_efficient_attention = True
        self.tgat.profile_attention_performance = self.enable_detailed_profiling
        
        # Enable DAG optimizations  
        self.dag.topological_cache_enabled = True
        self.dag.enable_dag_analytics = True
        self.dag.use_csr_adjacency = True
        
        # Enable Storage optimizations
        self.storage.dtype_optimization_enabled = True
        self.storage.use_memory_mapping = True
        
    def _validate_optimization_compatibility(self):
        """Validate that enabled optimizations are compatible"""
        
        # Check GPU optimizations
        if (self.c7_optimizations.tgat_enable_amp or 
            self.c7_optimizations.tgat_enable_flash_attention or
            self.c7_optimizations.tgat_use_fp16):
            
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("GPU optimizations enabled but CUDA not available")
            except ImportError:
                logger.error("GPU optimizations enabled but PyTorch not available")
                
        # Check memory constraints
        total_workers = (self.dag.parallel_workers + 
                        self.storage.parallel_io_workers)
        
        if total_workers > self.c7_optimizations.cpu_thread_count:
            logger.warning(f"Total workers ({total_workers}) exceeds available CPUs "
                          f"({self.c7_optimizations.cpu_thread_count})")
            
    def _setup_performance_monitoring(self):
        """Setup performance monitoring based on configuration"""
        
        if self.c7_optimizations.enable_performance_monitoring:
            # Configure PyTorch profiler if available
            try:
                import torch
                if hasattr(torch.profiler, 'ProfilerActivity'):
                    logger.info("PyTorch profiler available for performance monitoring")
            except ImportError:
                pass
                
        if self.enable_detailed_profiling:
            logger.info("Detailed profiling enabled - performance may be impacted")
            
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        
        summary = {
            'global_optimizations': {
                'all_optimizations_enabled': self.c7_optimizations.enable_all_optimizations,
                'performance_monitoring': self.c7_optimizations.enable_performance_monitoring,
                'detailed_profiling': self.enable_detailed_profiling
            },
            'tgat_optimizations': self.tgat.get_optimization_summary(),
            'dag_optimizations': self.dag.get_optimization_summary(), 
            'storage_optimizations': self.storage.get_optimization_summary(),
            'system_resources': {
                'max_memory_gb': self.c7_optimizations.max_memory_usage_gb,
                'cpu_threads': self.c7_optimizations.cpu_thread_count,
                'gpu_memory_fraction': self.c7_optimizations.gpu_memory_fraction
            }
        }
        
        # Calculate total optimizations enabled
        total_enabled = (
            summary['tgat_optimizations']['total_optimizations_enabled'] +
            summary['dag_optimizations']['total_optimizations_enabled'] +
            summary['storage_optimizations']['total_optimizations_enabled']
        )
        
        summary['total_optimizations_enabled'] = total_enabled
        
        return summary
        
    def create_enhanced_preset_configs(self) -> Dict[str, 'EnhancedDualGraphViewsConfig']:
        """Create enhanced preset configurations with Context7 optimizations"""
        
        presets = {}
        
        # Development configuration - minimal optimizations for debugging
        dev_c7_opts = Context7OptimizationConfig(
            enable_all_optimizations=False,
            tgat_enable_time_bias_caching=True,  # Only cache optimization
            dag_enable_topological_generations=True,
            parquet_enable_zstd_optimization=True
        )
        
        dev_config = EnhancedDualGraphViewsConfig(
            c7_optimizations=dev_c7_opts,
            enable_detailed_profiling=True
        )
        dev_config.dag.k_successors = 2
        dev_config.motifs.null_iterations = 50
        presets['development'] = dev_config
        
        # Standard production configuration - balanced optimizations
        std_c7_opts = Context7OptimizationConfig(
            enable_all_optimizations=False,
            tgat_enable_time_bias_caching=True,
            tgat_enable_fused_ops=True,
            dag_enable_vectorized_ops=True,
            dag_enable_topological_generations=True,
            dag_enable_sparse_adjacency=True,
            parquet_enable_zstd_optimization=True,
            parquet_enable_content_chunking=True,
            parquet_optimize_row_groups=True
        )
        
        std_config = EnhancedDualGraphViewsConfig(
            c7_optimizations=std_c7_opts
        )
        std_config.dag.k_successors = 4
        std_config.tgat.enhanced = True
        presets['standard'] = std_config
        
        # High-performance configuration - all optimizations enabled
        perf_c7_opts = Context7OptimizationConfig(
            enable_all_optimizations=True,
            max_memory_usage_gb=16.0,
            gpu_memory_fraction=0.9
        )
        
        perf_config = EnhancedDualGraphViewsConfig(
            c7_optimizations=perf_c7_opts
        )
        perf_config.dag.k_successors = 6
        perf_config.tgat.enhanced = True
        perf_config.tgat.num_layers = 3
        perf_config.max_concurrent_sessions = 8
        presets['high_performance'] = perf_config
        
        # Research configuration - maximum discovery with optimizations
        research_c7_opts = Context7OptimizationConfig(
            enable_all_optimizations=True,
            max_memory_usage_gb=32.0,
            enable_performance_monitoring=True
        )
        
        research_config = EnhancedDualGraphViewsConfig(
            c7_optimizations=research_c7_opts,
            enable_detailed_profiling=True
        )
        research_config.dag.k_successors = 8
        research_config.tgat.enhanced = True
        research_config.tgat.num_layers = 4
        research_config.motifs.null_iterations = 5000
        research_config.motifs.max_motifs = 500
        presets['research'] = research_config
        
        return presets
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedDualGraphViewsConfig':
        """Create enhanced configuration from dictionary with Context7 support"""
        
        # Extract Context7 optimizations if present
        c7_opts = config_dict.pop('c7_optimizations', {})
        c7_config = Context7OptimizationConfig(**c7_opts)
        
        # Extract enhanced component configurations
        components = {}
        
        # Enhanced TGAT config
        if 'tgat' in config_dict:
            tgat_dict = config_dict.pop('tgat')
            tgat_c7_opts = tgat_dict.pop('c7_optimizations', {})
            if tgat_c7_opts:
                c7_config = Context7OptimizationConfig(**{**c7_config.__dict__, **tgat_c7_opts})
            components['tgat'] = EnhancedTGATConfig(**tgat_dict, c7_optimizations=c7_config)
            
        # Enhanced DAG config  
        if 'dag' in config_dict:
            dag_dict = config_dict.pop('dag')
            dag_c7_opts = dag_dict.pop('c7_optimizations', {})
            if dag_c7_opts:
                c7_config = Context7OptimizationConfig(**{**c7_config.__dict__, **dag_c7_opts})
            components['dag'] = EnhancedDAGConfig(**dag_dict, c7_optimizations=c7_config)
            
        # Enhanced Storage config
        if 'storage' in config_dict:
            storage_dict = config_dict.pop('storage')
            storage_c7_opts = storage_dict.pop('c7_optimizations', {})
            if storage_c7_opts:
                c7_config = Context7OptimizationConfig(**{**c7_config.__dict__, **storage_c7_opts})
            components['storage'] = EnhancedStorageConfig(**storage_dict, c7_optimizations=c7_config)
            
        # Standard configs for M1 and Motifs
        if 'm1' in config_dict:
            components['m1'] = M1Config(**config_dict.pop('m1'))
        if 'motifs' in config_dict:
            components['motifs'] = MotifConfig(**config_dict.pop('motifs'))
            
        # Remaining system settings
        return cls(**components, c7_optimizations=c7_config, **config_dict)


def load_enhanced_config_with_overrides(
    base_config_path: Optional[Path] = None,
    preset: str = 'standard',
    c7_optimizations: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> EnhancedDualGraphViewsConfig:
    """
    Load enhanced configuration with Context7 optimizations and flexible overrides
    
    Args:
        base_config_path: Path to base configuration file
        preset: Preset configuration name ('development', 'standard', 'high_performance', 'research')
        c7_optimizations: Context7 optimization overrides
        overrides: General configuration overrides
        
    Returns:
        Enhanced configuration with Context7 optimizations
    """
    
    # Start with enhanced preset
    base_config = EnhancedDualGraphViewsConfig()
    presets = base_config.create_enhanced_preset_configs()
    
    if preset in presets:
        config = presets[preset]
        logger.info(f"Using enhanced '{preset}' preset with Context7 optimizations")
    else:
        config = EnhancedDualGraphViewsConfig()
        logger.warning(f"Unknown preset '{preset}', using default enhanced configuration")
        
    # Apply Context7 optimization overrides
    if c7_optimizations:
        current_opts = config.c7_optimizations.__dict__
        updated_opts = Context7OptimizationConfig(**{**current_opts, **c7_optimizations})
        config.c7_optimizations = updated_opts
        logger.info(f"Applied {len(c7_optimizations)} Context7 optimization overrides")
        
    # Apply general overrides
    if overrides:
        config_dict = config.to_dict()
        
        for key, value in overrides.items():
            if '.' in key:  # Nested key
                component, field = key.split('.', 1)
                if component in config_dict:
                    if isinstance(config_dict[component], dict):
                        config_dict[component][field] = value
            else:
                config_dict[key] = value
                
        config = EnhancedDualGraphViewsConfig.from_dict(config_dict)
        logger.info(f"Applied {len(overrides)} general configuration overrides")
        
    return config


# Factory functions for common configurations
def create_development_config() -> EnhancedDualGraphViewsConfig:
    """Create development configuration with minimal Context7 optimizations"""
    return load_enhanced_config_with_overrides(
        preset='development',
        c7_optimizations={
            'enable_performance_monitoring': True
        }
    )


def create_production_config() -> EnhancedDualGraphViewsConfig:
    """Create production configuration with balanced Context7 optimizations"""  
    return load_enhanced_config_with_overrides(
        preset='standard',
        c7_optimizations={
            'enable_performance_monitoring': True,
            'max_memory_usage_gb': 16.0
        }
    )


def create_high_performance_config() -> EnhancedDualGraphViewsConfig:
    """Create high-performance configuration with all Context7 optimizations"""
    return load_enhanced_config_with_overrides(
        preset='high_performance',
        c7_optimizations={
            'enable_all_optimizations': True,
            'max_memory_usage_gb': 32.0,
            'gpu_memory_fraction': 0.9
        }
    )


def create_research_config() -> EnhancedDualGraphViewsConfig:
    """Create research configuration optimized for discovery with Context7 enhancements"""
    return load_enhanced_config_with_overrides(
        preset='research', 
        c7_optimizations={
            'enable_all_optimizations': True,
            'enable_performance_monitoring': True,
            'max_memory_usage_gb': 64.0
        },
        overrides={
            'enable_detailed_profiling': True,
            'tgat.num_layers': 4,
            'motifs.significance_threshold': 0.01
        }
    )