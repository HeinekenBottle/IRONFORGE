"""
Archaeological Zone Detection Configuration for IRONFORGE
=========================================================

Configuration module for archaeological intelligence integration with IRONFORGE pipeline.
Defines detection parameters, performance thresholds, and integration settings.

Archaeological Principles Configuration:
- 40% dimensional anchor detection from previous session range
- 7.55-point precision target for temporal non-locality
- 87% authenticity threshold for pattern graduation
- Theory B forward positioning compliance
- Session isolation and HTF last-closed compliance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class DimensionalAnchorConfig:
    """Configuration for 40% dimensional anchor calculations"""
    # Core archaeological constant - DO NOT MODIFY
    anchor_percentage: float = 0.40
    
    # Precision targets
    precision_target: float = 7.55  # Archaeological precision requirement
    precision_tolerance: float = 0.5  # Acceptable deviation
    
    # Zone detection parameters
    min_zone_width: float = 5.0  # Minimum zone width in points
    max_zone_width: float = 100.0  # Maximum zone width in points
    zone_confidence_threshold: float = 0.7  # Minimum confidence for zone validity
    
    # Previous session range calculation
    range_calculation_method: str = "high_low"  # "high_low", "open_close", "body_range"
    range_smoothing_factor: float = 0.1  # Exponential smoothing for range calculation
    
    # Multi-timeframe support
    daily_scaling_factor: float = 2.46  # Daily timeframe scaling (67.4% more accurate)
    session_scaling_factor: float = 1.0  # Session-level baseline
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (0.0 < self.anchor_percentage < 1.0):
            raise ValueError("anchor_percentage must be between 0 and 1")
        
        if self.precision_target <= 0:
            raise ValueError("precision_target must be positive")
        
        if self.min_zone_width >= self.max_zone_width:
            raise ValueError("min_zone_width must be less than max_zone_width")


@dataclass
class TemporalNonLocalityConfig:
    """Configuration for temporal non-locality analysis"""
    # Theory B forward positioning
    enable_theory_b_validation: bool = True
    forward_positioning_window: int = 50  # Events to look ahead for validation
    temporal_echo_detection: bool = True
    
    # Non-locality detection parameters
    causality_threshold: float = 0.8  # Minimum causality strength
    temporal_coherence_threshold: float = 0.7  # Minimum temporal coherence
    echo_propagation_decay: float = 0.9  # Decay factor for echo propagation
    
    # Forward validation settings
    completion_validation_enabled: bool = True  # Validate against eventual completion
    intermediate_state_filtering: bool = True  # Filter intermediate positioning states
    final_state_weight: float = 2.0  # Weight factor for final state positioning
    
    # Temporal window configuration
    short_term_window: int = 10  # Short-term temporal analysis window
    medium_term_window: int = 25  # Medium-term temporal analysis window
    long_term_window: int = 50  # Long-term temporal analysis window
    
    # Non-locality scoring
    locality_penalty: float = 0.3  # Penalty for local-only patterns
    nonlocality_bonus: float = 0.5  # Bonus for confirmed non-local patterns


@dataclass
class SessionIsolationConfig:
    """Configuration for session boundary enforcement"""
    # Golden invariant enforcement
    strict_session_boundaries: bool = True  # Absolute session isolation
    cross_session_edge_detection: bool = True  # Detect and reject cross-session edges
    session_contamination_threshold: float = 0.0  # Zero tolerance for contamination
    
    # HTF compliance
    htf_last_closed_only: bool = True  # Enforce last-closed HTF data only
    intra_candle_data_rejection: bool = True  # Reject intra-candle HTF data
    htf_feature_range: Tuple[int, int] = (45, 50)  # HTF features f45-f50
    
    # Session independence validation
    validate_session_independence: bool = True
    independence_check_features: List[str] = field(default_factory=lambda: [
        "session_start", "session_end", "cross_references"
    ])
    
    # Within-session learning preservation
    preserve_session_learning: bool = True
    learning_context_isolation: bool = True


@dataclass
class AuthenticityConfig:
    """Configuration for archaeological authenticity scoring"""
    # Core authenticity threshold (Golden Invariant)
    authenticity_threshold: float = 87.0  # Minimum authenticity for production
    
    # Scoring components and weights
    base_confidence_weight: float = 40.0  # Weight for base zone confidence
    temporal_coherence_weight: float = 25.0  # Weight for temporal coherence
    theory_b_alignment_weight: float = 15.0  # Weight for Theory B compliance
    precision_bonus_weight: float = 10.0  # Weight for precision bonus
    archaeological_context_weight: float = 10.0  # Weight for archaeological context
    
    # Bonus thresholds
    precision_bonus_threshold: float = 7.0  # Minimum precision for bonus
    temporal_coherence_threshold: float = 0.7  # Minimum coherence for bonus
    theory_b_compliance_required: bool = True  # Require Theory B compliance
    
    # Quality gates
    minimum_zones_for_analysis: int = 1  # Minimum zones required for valid analysis
    maximum_zones_per_session: int = 50  # Maximum zones to prevent overflow
    
    # Pattern graduation integration
    pattern_graduation_enabled: bool = True
    graduation_authenticity_boost: float = 5.0  # Boost for graduated patterns


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and optimization"""
    # Core performance requirements (Golden Invariants)
    max_session_processing_time: float = 3.0  # Maximum seconds per session
    max_detection_time: float = 1.0  # Maximum seconds for zone detection
    min_anchor_accuracy: float = 95.0  # Minimum anchor point accuracy percentage
    
    # Memory constraints
    max_memory_usage_mb: float = 100.0  # Maximum memory usage in MB
    memory_monitoring_enabled: bool = True
    garbage_collection_threshold: float = 80.0  # GC trigger threshold (MB)
    
    # Optimization settings
    enable_lazy_loading: bool = True  # Enable lazy loading for components
    cache_archaeological_results: bool = False  # Disable caching for session independence
    batch_processing_enabled: bool = False  # Disable batching for session isolation
    
    # Performance monitoring
    detailed_timing_enabled: bool = True
    performance_logging_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # Quality gates
    accuracy_monitoring_enabled: bool = True
    performance_degradation_threshold: float = 2.0  # Multiplier for time limits
    
    # Alerting thresholds
    slow_session_threshold: float = 2.5  # Warn if session takes > 2.5s
    memory_warning_threshold: float = 85.0  # Warn at 85MB usage
    accuracy_warning_threshold: float = 92.0  # Warn if accuracy < 92%


@dataclass
class IronforgeIntegrationConfig:
    """Configuration for IRONFORGE pipeline integration"""
    # Discovery stage integration
    enhance_tgat_discovery: bool = True  # Enable TGAT discovery enhancement
    discovery_feature_extension: bool = True  # Extend discovery with archaeological features
    
    # Enhanced Graph Builder integration
    graph_builder_integration: bool = True
    node_feature_extension_dims: int = 5  # Additional archaeological node features
    edge_feature_preservation: bool = True  # Preserve original edge features
    
    # Pattern graduation integration
    pattern_graduation_integration: bool = True
    authenticity_boost_enabled: bool = True
    graduation_threshold_override: Optional[float] = None  # Override default thresholds
    
    # Confluence scoring integration
    confluence_archaeological_weight: float = 0.15  # Weight in confluence scoring
    archaeological_scoring_enabled: bool = True
    
    # Container system integration
    lazy_loading_integration: bool = True
    component_independence_enforced: bool = True  # Enforce session independence
    container_performance_monitoring: bool = True
    
    # Feature dimension compliance
    node_feature_dim_standard: int = 45  # Standard node features
    node_feature_dim_htf: int = 51  # HTF-enhanced node features
    edge_feature_dim: int = 20  # Edge feature dimensions
    archaeological_feature_dim: int = 8  # Additional archaeological features
    
    # Pipeline stage configuration
    discovery_stage_enabled: bool = True
    confluence_stage_contribution: bool = True
    validation_stage_participation: bool = True
    reporting_stage_visualization: bool = True


@dataclass
class ValidationConfig:
    """Configuration for contract and quality validation"""
    # Contract validation
    golden_invariant_enforcement: bool = True
    event_taxonomy_validation: bool = True
    feature_dimension_validation: bool = True
    edge_intent_validation: bool = True
    
    # Event type validation
    expected_event_types: List[str] = field(default_factory=lambda: [
        "Expansion", "Consolidation", "Retracement", 
        "Reversal", "Liquidity Taken", "Redelivery"
    ])
    
    # Edge intent validation
    expected_edge_intents: List[str] = field(default_factory=lambda: [
        "TEMPORAL_NEXT", "MOVEMENT_TRANSITION", "LIQ_LINK", "CONTEXT"
    ])
    
    # Feature dimension validation
    validate_node_dimensions: bool = True
    validate_edge_dimensions: bool = True
    allow_htf_extension: bool = True
    
    # Quality validation
    quality_gate_enforcement: bool = True
    minimum_quality_threshold: str = "fair"
    authenticity_gate_enabled: bool = True
    
    # Session validation
    session_isolation_validation: bool = True
    cross_session_contamination_check: bool = True
    htf_compliance_validation: bool = True
    
    # Error handling
    validation_failure_mode: str = "strict"  # "strict", "warn", "ignore"
    contract_violation_logging: bool = True


@dataclass
class DebuggingConfig:
    """Configuration for debugging and development support"""
    # Debug mode settings
    debug_mode_enabled: bool = False
    verbose_logging_enabled: bool = False
    
    # Archaeological analysis debugging
    zone_detection_debugging: bool = False
    temporal_analysis_debugging: bool = False
    authenticity_scoring_debugging: bool = False
    
    # Performance debugging
    performance_profiling_enabled: bool = False
    memory_profiling_enabled: bool = False
    timing_breakdown_enabled: bool = False
    
    # Integration debugging
    ironforge_integration_debugging: bool = False
    container_debugging: bool = False
    
    # Output debugging
    debug_output_directory: Optional[str] = None
    save_intermediate_results: bool = False
    debug_visualization_enabled: bool = False


@dataclass
class ArchaeologicalConfig:
    """
    Complete archaeological zone detection configuration
    
    Master configuration class that combines all archaeological intelligence
    settings for seamless IRONFORGE integration.
    """
    # Core configuration components
    dimensional_anchor: DimensionalAnchorConfig = field(default_factory=DimensionalAnchorConfig)
    temporal_nonlocality: TemporalNonLocalityConfig = field(default_factory=TemporalNonLocalityConfig)
    session_isolation: SessionIsolationConfig = field(default_factory=SessionIsolationConfig)
    authenticity: AuthenticityConfig = field(default_factory=AuthenticityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ironforge_integration: IronforgeIntegrationConfig = field(default_factory=IronforgeIntegrationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    debugging: DebuggingConfig = field(default_factory=DebuggingConfig)
    
    # Global settings
    agent_version: str = "1.0.0"
    config_version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate complete configuration"""
        self._validate_configuration()
        logger.info(f"Archaeological configuration loaded (v{self.config_version})")
    
    def _validate_configuration(self) -> None:
        """Validate configuration consistency"""
        # Validate performance requirements against authenticity thresholds
        if (self.authenticity.authenticity_threshold > 90.0 and 
            self.performance.max_session_processing_time < 2.0):
            logger.warning(
                "High authenticity threshold with low processing time may cause conflicts"
            )
        
        # Validate feature dimension consistency
        if (self.ironforge_integration.node_feature_dim_standard != 45 or
            self.ironforge_integration.node_feature_dim_htf != 51 or
            self.ironforge_integration.edge_feature_dim != 20):
            raise ValueError("Feature dimensions must match IRONFORGE golden invariants")
        
        # Validate archaeological constants
        if self.dimensional_anchor.anchor_percentage != 0.40:
            logger.warning("Modifying archaeological anchor percentage may break compatibility")
        
        # Validate session isolation settings
        if not self.session_isolation.strict_session_boundaries:
            raise ValueError("Session isolation is required for archaeological validity")
        
        # Validate HTF compliance
        if (not self.session_isolation.htf_last_closed_only or 
            not self.session_isolation.intra_candle_data_rejection):
            raise ValueError("HTF compliance is required for archaeological validity")
    
    def get_discovery_config(self) -> Dict[str, Any]:
        """Get configuration subset for discovery stage integration"""
        return {
            "dimensional_anchor": self.dimensional_anchor,
            "temporal_nonlocality": self.temporal_nonlocality,
            "performance": self.performance,
            "validation": self.validation
        }
    
    def get_confluence_config(self) -> Dict[str, Any]:
        """Get configuration subset for confluence scoring integration"""
        return {
            "authenticity": self.authenticity,
            "ironforge_integration": self.ironforge_integration,
            "performance": self.performance
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get configuration subset for validation stage"""
        return {
            "validation": self.validation,
            "session_isolation": self.session_isolation,
            "performance": self.performance
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            "dimensional_anchor": self.dimensional_anchor.__dict__,
            "temporal_nonlocality": self.temporal_nonlocality.__dict__,
            "session_isolation": self.session_isolation.__dict__,
            "authenticity": self.authenticity.__dict__,
            "performance": self.performance.__dict__,
            "ironforge_integration": self.ironforge_integration.__dict__,
            "validation": self.validation.__dict__,
            "debugging": self.debugging.__dict__,
            "agent_version": self.agent_version,
            "config_version": self.config_version
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ArchaeologicalConfig':
        """Create configuration from dictionary"""
        # Extract component configurations
        dimensional_anchor = DimensionalAnchorConfig(**config_dict.get("dimensional_anchor", {}))
        temporal_nonlocality = TemporalNonLocalityConfig(**config_dict.get("temporal_nonlocality", {}))
        session_isolation = SessionIsolationConfig(**config_dict.get("session_isolation", {}))
        authenticity = AuthenticityConfig(**config_dict.get("authenticity", {}))
        performance = PerformanceConfig(**config_dict.get("performance", {}))
        ironforge_integration = IronforgeIntegrationConfig(**config_dict.get("ironforge_integration", {}))
        validation = ValidationConfig(**config_dict.get("validation", {}))
        debugging = DebuggingConfig(**config_dict.get("debugging", {}))
        
        return cls(
            dimensional_anchor=dimensional_anchor,
            temporal_nonlocality=temporal_nonlocality,
            session_isolation=session_isolation,
            authenticity=authenticity,
            performance=performance,
            ironforge_integration=ironforge_integration,
            validation=validation,
            debugging=debugging,
            agent_version=config_dict.get("agent_version", "1.0.0"),
            config_version=config_dict.get("config_version", "1.0.0")
        )


# Configuration presets for different use cases
class ConfigurationPresets:
    """Pre-configured settings for common archaeological analysis scenarios"""
    
    @staticmethod
    def production_config() -> ArchaeologicalConfig:
        """Production configuration with maximum performance and reliability"""
        config = ArchaeologicalConfig()
        
        # Optimize for production performance
        config.performance.max_session_processing_time = 2.5
        config.performance.max_detection_time = 0.8
        config.performance.enable_lazy_loading = True
        config.performance.detailed_timing_enabled = False
        
        # Strict authenticity requirements
        config.authenticity.authenticity_threshold = 90.0
        config.authenticity.theory_b_compliance_required = True
        
        # Disable debugging
        config.debugging.debug_mode_enabled = False
        config.debugging.verbose_logging_enabled = False
        
        return config
    
    @staticmethod
    def development_config() -> ArchaeologicalConfig:
        """Development configuration with debugging and validation enabled"""
        config = ArchaeologicalConfig()
        
        # Enable debugging features
        config.debugging.debug_mode_enabled = True
        config.debugging.verbose_logging_enabled = True
        config.debugging.zone_detection_debugging = True
        config.debugging.performance_profiling_enabled = True
        
        # Relaxed performance for debugging
        config.performance.max_session_processing_time = 5.0
        config.performance.detailed_timing_enabled = True
        
        # Lower authenticity threshold for experimentation
        config.authenticity.authenticity_threshold = 80.0
        
        return config
    
    @staticmethod
    def research_config() -> ArchaeologicalConfig:
        """Research configuration for archaeological discovery validation"""
        config = ArchaeologicalConfig()
        
        # Enable all analysis features
        config.temporal_nonlocality.enable_theory_b_validation = True
        config.temporal_nonlocality.temporal_echo_detection = True
        config.temporal_nonlocality.completion_validation_enabled = True
        
        # Enhanced debugging for research
        config.debugging.zone_detection_debugging = True
        config.debugging.temporal_analysis_debugging = True
        config.debugging.authenticity_scoring_debugging = True
        config.debugging.save_intermediate_results = True
        
        # Flexible performance constraints for research
        config.performance.max_session_processing_time = 10.0
        
        return config
    
    @staticmethod
    def minimal_config() -> ArchaeologicalConfig:
        """Minimal configuration for basic zone detection"""
        config = ArchaeologicalConfig()
        
        # Minimal features enabled
        config.temporal_nonlocality.temporal_echo_detection = False
        config.temporal_nonlocality.completion_validation_enabled = False
        config.ironforge_integration.discovery_feature_extension = False
        
        # Basic authenticity requirements
        config.authenticity.authenticity_threshold = 85.0
        config.authenticity.pattern_graduation_enabled = False
        
        # Maximum performance optimization
        config.performance.max_session_processing_time = 1.5
        config.performance.max_detection_time = 0.5
        
        return config


# Configuration validation utilities
def validate_archaeological_config(config: ArchaeologicalConfig) -> List[str]:
    """
    Validate archaeological configuration against IRONFORGE requirements
    
    Returns list of validation warnings/errors
    """
    warnings = []
    
    # Validate core archaeological constants
    if config.dimensional_anchor.anchor_percentage != 0.40:
        warnings.append("Archaeological anchor percentage should be 0.40 for compatibility")
    
    if config.dimensional_anchor.precision_target != 7.55:
        warnings.append("Precision target should be 7.55 for archaeological compliance")
    
    # Validate authenticity threshold
    if config.authenticity.authenticity_threshold < 87.0:
        warnings.append("Authenticity threshold below 87% may not meet production requirements")
    
    # Validate performance requirements
    if config.performance.max_session_processing_time > 3.0:
        warnings.append("Session processing time exceeds IRONFORGE 3s requirement")
    
    # Validate feature dimensions
    if config.ironforge_integration.node_feature_dim_standard != 45:
        warnings.append("Standard node feature dimension must be 45 for IRONFORGE compatibility")
    
    if config.ironforge_integration.node_feature_dim_htf != 51:
        warnings.append("HTF node feature dimension must be 51 for IRONFORGE compatibility")
    
    if config.ironforge_integration.edge_feature_dim != 20:
        warnings.append("Edge feature dimension must be 20 for IRONFORGE compatibility")
    
    # Validate session isolation
    if not config.session_isolation.strict_session_boundaries:
        warnings.append("Session isolation must be enabled for archaeological validity")
    
    # Validate HTF compliance
    if not config.session_isolation.htf_last_closed_only:
        warnings.append("HTF last-closed enforcement required for temporal validity")
    
    return warnings


# Default configuration factory
def create_default_archaeological_config() -> ArchaeologicalConfig:
    """Create default archaeological configuration for IRONFORGE integration"""
    return ArchaeologicalConfig()


# Export configuration loading utilities
__all__ = [
    "ArchaeologicalConfig",
    "DimensionalAnchorConfig",
    "TemporalNonLocalityConfig", 
    "SessionIsolationConfig",
    "AuthenticityConfig",
    "PerformanceConfig",
    "IronforgeIntegrationConfig",
    "ValidationConfig",
    "DebuggingConfig",
    "ConfigurationPresets",
    "validate_archaeological_config",
    "create_default_archaeological_config"
]