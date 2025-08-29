"""
IRONFORGE Confluence Scoring Configuration
==========================================

Config-driven weight validation and enhanced archaeological DAG weighting.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConfluenceWeights:
    """Confluence scoring weights configuration."""
    
    # Standard confluence weights
    temporal_coherence: float = 0.25
    pattern_strength: float = 0.30
    archaeological_significance: float = 0.20
    session_context: float = 0.15
    discovery_confidence: float = 0.10
    
    # Enhanced archaeological DAG weighting (feature flag controlled)
    dag_topology_weight: float = 0.0
    dag_centrality_weight: float = 0.0
    dag_flow_weight: float = 0.0
    dag_clustering_weight: float = 0.0
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (
            self.temporal_coherence +
            self.pattern_strength +
            self.archaeological_significance +
            self.session_context +
            self.discovery_confidence +
            self.dag_topology_weight +
            self.dag_centrality_weight +
            self.dag_flow_weight +
            self.dag_clustering_weight
        )
        
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Confluence weights must sum to 1.0, got {total:.6f}")


@dataclass
class DAGWeightingConfig:
    """Enhanced archaeological DAG weighting configuration."""
    
    # Feature flag - disabled by default for backward compatibility
    enable_archaeological_zone_weighting: bool = False
    
    # DAG analysis parameters
    topology_analysis: bool = True
    centrality_analysis: bool = True
    flow_analysis: bool = True
    clustering_analysis: bool = True
    
    # DAG weighting parameters
    zone_influence_radius: float = 3.0
    temporal_decay_factor: float = 0.8
    centrality_boost_factor: float = 1.2
    flow_amplification: float = 1.1
    
    # Archaeological zone parameters
    zone_detection_threshold: float = 0.7
    zone_significance_weight: float = 0.3
    inter_zone_penalty: float = 0.1
    
    def validate(self) -> bool:
        """Validate DAG weighting configuration."""
        if self.zone_influence_radius <= 0:
            raise ValueError("zone_influence_radius must be positive")
        
        if not 0 < self.temporal_decay_factor <= 1:
            raise ValueError("temporal_decay_factor must be in (0, 1]")
        
        if self.centrality_boost_factor < 1:
            raise ValueError("centrality_boost_factor must be >= 1")
        
        if not 0 < self.zone_detection_threshold <= 1:
            raise ValueError("zone_detection_threshold must be in (0, 1]")
        
        return True


@dataclass
class ConfluenceConfig:
    """Complete confluence scoring configuration."""
    
    weights: ConfluenceWeights = field(default_factory=ConfluenceWeights)
    dag_weighting: DAGWeightingConfig = field(default_factory=DAGWeightingConfig)
    
    # Scoring parameters
    threshold: float = 65.0
    quality_gate: float = 87.0
    authenticity_threshold: float = 0.87
    
    # Output configuration
    save_detailed_scores: bool = True
    save_statistics: bool = True
    export_dag_analysis: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        self.weights.__post_init__()  # Validate weights
        self.dag_weighting.validate()  # Validate DAG config
        
        if self.threshold < 0 or self.threshold > 100:
            raise ValueError("threshold must be in [0, 100]")
        
        if self.quality_gate < 0 or self.quality_gate > 100:
            raise ValueError("quality_gate must be in [0, 100]")
        
        if not 0 <= self.authenticity_threshold <= 1:
            raise ValueError("authenticity_threshold must be in [0, 1]")


def create_confluence_config(
    weights: Optional[Dict[str, float]] = None,
    dag_features: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ConfluenceConfig:
    """
    Create confluence configuration with optional overrides.
    
    Args:
        weights: Weight overrides
        dag_features: DAG weighting feature configuration
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured ConfluenceConfig instance
    """
    # Create base weights
    weight_config = ConfluenceWeights()
    
    # Apply weight overrides
    if weights:
        for key, value in weights.items():
            if hasattr(weight_config, key):
                setattr(weight_config, key, float(value))
            else:
                logger.warning(f"Unknown weight key: {key}")
    
    # Create DAG weighting config
    dag_config = DAGWeightingConfig()
    
    # Apply DAG feature overrides
    if dag_features:
        for key, value in dag_features.items():
            if hasattr(dag_config, key):
                setattr(dag_config, key, value)
            else:
                logger.warning(f"Unknown DAG feature key: {key}")
    
    # Create main config
    config = ConfluenceConfig(
        weights=weight_config,
        dag_weighting=dag_config,
        **kwargs
    )
    
    return config


def validate_weights(weights: Dict[str, float]) -> bool:
    """
    Validate confluence scoring weights.
    
    Args:
        weights: Weight dictionary to validate
        
    Returns:
        True if weights are valid
        
    Raises:
        ValueError: If weights are invalid
    """
    if not weights:
        raise ValueError("Weights dictionary cannot be empty")
    
    # Check for negative weights
    negative_weights = {k: v for k, v in weights.items() if v < 0}
    if negative_weights:
        raise ValueError(f"Negative weights not allowed: {negative_weights}")
    
    # Check weight sum
    total = sum(weights.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
    
    # Check for known weight keys
    known_keys = {
        'temporal_coherence', 'pattern_strength', 'archaeological_significance',
        'session_context', 'discovery_confidence', 'dag_topology_weight',
        'dag_centrality_weight', 'dag_flow_weight', 'dag_clustering_weight'
    }
    
    unknown_keys = set(weights.keys()) - known_keys
    if unknown_keys:
        logger.warning(f"Unknown weight keys (will be ignored): {unknown_keys}")
    
    return True


def get_default_weights(enable_dag_weighting: bool = False) -> Dict[str, float]:
    """
    Get default confluence weights.
    
    Args:
        enable_dag_weighting: Whether to enable DAG weighting features
        
    Returns:
        Default weights dictionary
    """
    if enable_dag_weighting:
        # Redistribute weights to include DAG features
        return {
            'temporal_coherence': 0.20,
            'pattern_strength': 0.25,
            'archaeological_significance': 0.15,
            'session_context': 0.10,
            'discovery_confidence': 0.10,
            'dag_topology_weight': 0.08,
            'dag_centrality_weight': 0.07,
            'dag_flow_weight': 0.03,
            'dag_clustering_weight': 0.02,
        }
    else:
        # Standard weights (backward compatible)
        return {
            'temporal_coherence': 0.25,
            'pattern_strength': 0.30,
            'archaeological_significance': 0.20,
            'session_context': 0.15,
            'discovery_confidence': 0.10,
        }


def load_confluence_config_from_dict(config_dict: Dict[str, Any]) -> ConfluenceConfig:
    """
    Load confluence configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        ConfluenceConfig instance
    """
    # Extract weights
    weights_dict = config_dict.get('weights', {})
    
    # Extract DAG features
    dag_dict = config_dict.get('dag', {})
    dag_features = dag_dict.get('features', {})
    
    # Extract other parameters
    other_params = {
        k: v for k, v in config_dict.items() 
        if k not in ['weights', 'dag']
    }
    
    return create_confluence_config(
        weights=weights_dict,
        dag_features=dag_features,
        **other_params
    )


def export_confluence_config(config: ConfluenceConfig) -> Dict[str, Any]:
    """
    Export confluence configuration to dictionary.
    
    Args:
        config: ConfluenceConfig to export
        
    Returns:
        Configuration dictionary
    """
    return {
        'weights': {
            'temporal_coherence': config.weights.temporal_coherence,
            'pattern_strength': config.weights.pattern_strength,
            'archaeological_significance': config.weights.archaeological_significance,
            'session_context': config.weights.session_context,
            'discovery_confidence': config.weights.discovery_confidence,
            'dag_topology_weight': config.weights.dag_topology_weight,
            'dag_centrality_weight': config.weights.dag_centrality_weight,
            'dag_flow_weight': config.weights.dag_flow_weight,
            'dag_clustering_weight': config.weights.dag_clustering_weight,
        },
        'dag': {
            'features': {
                'enable_archaeological_zone_weighting': config.dag_weighting.enable_archaeological_zone_weighting,
                'topology_analysis': config.dag_weighting.topology_analysis,
                'centrality_analysis': config.dag_weighting.centrality_analysis,
                'flow_analysis': config.dag_weighting.flow_analysis,
                'clustering_analysis': config.dag_weighting.clustering_analysis,
                'zone_influence_radius': config.dag_weighting.zone_influence_radius,
                'temporal_decay_factor': config.dag_weighting.temporal_decay_factor,
                'centrality_boost_factor': config.dag_weighting.centrality_boost_factor,
                'flow_amplification': config.dag_weighting.flow_amplification,
                'zone_detection_threshold': config.dag_weighting.zone_detection_threshold,
                'zone_significance_weight': config.dag_weighting.zone_significance_weight,
                'inter_zone_penalty': config.dag_weighting.inter_zone_penalty,
            }
        },
        'threshold': config.threshold,
        'quality_gate': config.quality_gate,
        'authenticity_threshold': config.authenticity_threshold,
        'save_detailed_scores': config.save_detailed_scores,
        'save_statistics': config.save_statistics,
        'export_dag_analysis': config.export_dag_analysis,
    }
