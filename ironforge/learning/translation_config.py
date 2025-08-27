"""
Translation Layer Configuration
Opt-in feature flags for Context7-guided optimizations
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DAGBuilderConfig:
    """Configuration for optimized DAG building (D2G layer)"""
    enable_cache: bool = False
    enable_lazy_loading: bool = False
    enable_topological_optimization: bool = False
    cache_size: int = 1000
    lazy_loading_threshold: int = 100
    

@dataclass
class MotifMiningConfig:
    """Configuration for efficient motif mining (G2M layer)"""
    enable_parallel_isomorphism: bool = False
    enable_graph_caching: bool = False
    enable_sparse_representation: bool = False
    max_cache_size: int = 500
    parallel_workers: Optional[int] = None
    

@dataclass 
class BootstrapConfig:
    """Configuration for reproducible bootstrap (M2E layer)"""
    enable_stratified_sampling: bool = False
    enable_thread_safe_rng: bool = False
    enable_sklearn_utils: bool = False
    random_state: Optional[int] = None
    n_jobs: int = 1
    

@dataclass
class ParquetConfig:
    """Configuration for optimized Parquet I/O (E2R layer)"""
    enable_cdc: bool = False
    enable_row_group_optimization: bool = False
    enable_content_defined_chunking: bool = False
    target_row_group_size: int = 50000
    cdc_min_chunk_size: int = 256 * 1024  # 256 KiB
    cdc_max_chunk_size: int = 1024 * 1024  # 1 MiB
    

@dataclass
class PresentationConfig:
    """Configuration for validated presentation (RTP layer)"""
    enable_template_validation: bool = False
    enable_statistical_formatting: bool = False
    enable_contract_validation: bool = False
    decimal_precision: int = 4
    confidence_level: float = 0.95


@dataclass
class TranslationConfig:
    """Master configuration for all translation layers"""
    
    # Feature flags - all disabled by default for zero behavioral change
    enable_optimized_dag_builder: bool = False
    enable_efficient_motif_mining: bool = False
    enable_reproducible_bootstrap: bool = False
    enable_optimized_parquet: bool = False
    enable_validated_presentation: bool = False
    
    # Layer configurations
    dag_builder: DAGBuilderConfig = field(default_factory=DAGBuilderConfig)
    motif_mining: MotifMiningConfig = field(default_factory=MotifMiningConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    parquet: ParquetConfig = field(default_factory=ParquetConfig)
    presentation: PresentationConfig = field(default_factory=PresentationConfig)
    
    @classmethod
    def from_environment(cls) -> "TranslationConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # Check environment flags
        config.enable_optimized_dag_builder = _env_bool("IRONFORGE_ENABLE_OPTIMIZED_DAG_BUILDER")
        config.enable_efficient_motif_mining = _env_bool("IRONFORGE_ENABLE_EFFICIENT_MOTIF_MINING")  
        config.enable_reproducible_bootstrap = _env_bool("IRONFORGE_ENABLE_REPRODUCIBLE_BOOTSTRAP")
        config.enable_optimized_parquet = _env_bool("IRONFORGE_ENABLE_OPTIMIZED_PARQUET")
        config.enable_validated_presentation = _env_bool("IRONFORGE_ENABLE_VALIDATED_PRESENTATION")
        
        return config
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "flags": {
                "enable_optimized_dag_builder": self.enable_optimized_dag_builder,
                "enable_efficient_motif_mining": self.enable_efficient_motif_mining,
                "enable_reproducible_bootstrap": self.enable_reproducible_bootstrap,
                "enable_optimized_parquet": self.enable_optimized_parquet,
                "enable_validated_presentation": self.enable_validated_presentation,
            },
            "dag_builder": self.dag_builder.__dict__,
            "motif_mining": self.motif_mining.__dict__,
            "bootstrap": self.bootstrap.__dict__,
            "parquet": self.parquet.__dict__,
            "presentation": self.presentation.__dict__,
        }


def _env_bool(key: str, default: bool = False) -> bool:
    """Parse boolean from environment variable"""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


# Global configuration instance
_global_config: Optional[TranslationConfig] = None


def get_global_config() -> TranslationConfig:
    """Get or create global translation configuration"""
    global _global_config
    if _global_config is None:
        _global_config = TranslationConfig.from_environment()
    return _global_config


def set_global_config(config: TranslationConfig) -> None:
    """Set global translation configuration"""
    global _global_config
    _global_config = config