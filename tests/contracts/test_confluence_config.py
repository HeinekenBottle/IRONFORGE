"""
Test Confluence Configuration
=============================

Tests for confluence scoring configuration and archaeological DAG weighting feature flag.
"""

import pytest
from ironforge.confluence.config import (
    ConfluenceWeights,
    DAGWeightingConfig,
    ConfluenceConfig,
    create_confluence_config,
    validate_weights,
    get_default_weights,
    load_confluence_config_from_dict,
    export_confluence_config,
)


class TestConfluenceWeights:
    """Test confluence weights validation."""
    
    def test_valid_default_weights(self):
        """Test default weights are valid."""
        weights = ConfluenceWeights()
        
        # Should not raise exception
        assert weights.temporal_coherence == 0.25
        assert weights.pattern_strength == 0.30
        assert weights.archaeological_significance == 0.20
        assert weights.session_context == 0.15
        assert weights.discovery_confidence == 0.10
    
    def test_weights_sum_validation(self):
        """Test weights sum to 1.0 validation."""
        # Valid weights
        weights = ConfluenceWeights(
            temporal_coherence=0.2,
            pattern_strength=0.3,
            archaeological_significance=0.2,
            session_context=0.2,
            discovery_confidence=0.1
        )
        # Should not raise exception
        
        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ConfluenceWeights(
                temporal_coherence=0.5,
                pattern_strength=0.3,
                archaeological_significance=0.2,
                session_context=0.2,
                discovery_confidence=0.1
            )
    
    def test_dag_weights_included(self):
        """Test DAG weights are included in sum validation."""
        # Valid with DAG weights
        weights = ConfluenceWeights(
            temporal_coherence=0.15,
            pattern_strength=0.25,
            archaeological_significance=0.15,
            session_context=0.10,
            discovery_confidence=0.10,
            dag_topology_weight=0.10,
            dag_centrality_weight=0.10,
            dag_flow_weight=0.03,
            dag_clustering_weight=0.02
        )
        # Should not raise exception


class TestDAGWeightingConfig:
    """Test DAG weighting configuration."""
    
    def test_default_dag_config(self):
        """Test default DAG configuration."""
        config = DAGWeightingConfig()
        
        assert config.enable_archaeological_zone_weighting is False  # Disabled by default
        assert config.topology_analysis is True
        assert config.centrality_analysis is True
        assert config.flow_analysis is True
        assert config.clustering_analysis is True
    
    def test_dag_config_validation(self):
        """Test DAG configuration validation."""
        # Valid config
        config = DAGWeightingConfig(
            zone_influence_radius=3.0,
            temporal_decay_factor=0.8,
            centrality_boost_factor=1.2,
            zone_detection_threshold=0.7
        )
        assert config.validate()
        
        # Invalid zone_influence_radius
        with pytest.raises(ValueError, match="zone_influence_radius must be positive"):
            DAGWeightingConfig(zone_influence_radius=0.0).validate()
        
        # Invalid temporal_decay_factor
        with pytest.raises(ValueError, match="temporal_decay_factor must be in"):
            DAGWeightingConfig(temporal_decay_factor=1.5).validate()
        
        # Invalid centrality_boost_factor
        with pytest.raises(ValueError, match="centrality_boost_factor must be"):
            DAGWeightingConfig(centrality_boost_factor=0.5).validate()
        
        # Invalid zone_detection_threshold
        with pytest.raises(ValueError, match="zone_detection_threshold must be in"):
            DAGWeightingConfig(zone_detection_threshold=1.5).validate()


class TestConfluenceConfig:
    """Test complete confluence configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ConfluenceConfig()
        
        assert config.threshold == 65.0
        assert config.quality_gate == 87.0
        assert config.authenticity_threshold == 0.87
        assert config.dag_weighting.enable_archaeological_zone_weighting is False
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ConfluenceConfig(
            threshold=70.0,
            quality_gate=85.0,
            authenticity_threshold=0.8
        )
        # Should not raise exception
        
        # Invalid threshold
        with pytest.raises(ValueError, match="threshold must be in"):
            ConfluenceConfig(threshold=150.0)
        
        # Invalid quality_gate
        with pytest.raises(ValueError, match="quality_gate must be in"):
            ConfluenceConfig(quality_gate=-10.0)
        
        # Invalid authenticity_threshold
        with pytest.raises(ValueError, match="authenticity_threshold must be in"):
            ConfluenceConfig(authenticity_threshold=1.5)


class TestConfigurationFactory:
    """Test configuration factory functions."""
    
    def test_create_confluence_config(self):
        """Test confluence config creation."""
        config = create_confluence_config(
            weights={'temporal_coherence': 0.3, 'pattern_strength': 0.7},
            dag_features={'enable_archaeological_zone_weighting': True},
            threshold=75.0
        )
        
        assert config.weights.temporal_coherence == 0.3
        assert config.weights.pattern_strength == 0.7
        assert config.dag_weighting.enable_archaeological_zone_weighting is True
        assert config.threshold == 75.0
    
    def test_validate_weights_function(self):
        """Test weights validation function."""
        # Valid weights
        valid_weights = {
            'temporal_coherence': 0.25,
            'pattern_strength': 0.30,
            'archaeological_significance': 0.20,
            'session_context': 0.15,
            'discovery_confidence': 0.10
        }
        assert validate_weights(valid_weights)
        
        # Invalid weights (negative)
        with pytest.raises(ValueError, match="Negative weights not allowed"):
            validate_weights({'temporal_coherence': -0.1, 'pattern_strength': 1.1})
        
        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_weights({'temporal_coherence': 0.5, 'pattern_strength': 0.3})
        
        # Empty weights
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_weights({})
    
    def test_get_default_weights(self):
        """Test default weights generation."""
        # Standard weights
        standard_weights = get_default_weights(enable_dag_weighting=False)
        assert len(standard_weights) == 5
        assert abs(sum(standard_weights.values()) - 1.0) < 0.001
        
        # DAG-enabled weights
        dag_weights = get_default_weights(enable_dag_weighting=True)
        assert len(dag_weights) == 9  # 5 standard + 4 DAG weights
        assert abs(sum(dag_weights.values()) - 1.0) < 0.001
        assert 'dag_topology_weight' in dag_weights


class TestConfigSerialization:
    """Test configuration serialization/deserialization."""
    
    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            'weights': {
                'temporal_coherence': 0.2,
                'pattern_strength': 0.3,
                'archaeological_significance': 0.2,
                'session_context': 0.2,
                'discovery_confidence': 0.1
            },
            'dag': {
                'features': {
                    'enable_archaeological_zone_weighting': True,
                    'zone_influence_radius': 4.0
                }
            },
            'threshold': 70.0,
            'quality_gate': 85.0
        }
        
        config = load_confluence_config_from_dict(config_dict)
        
        assert config.weights.temporal_coherence == 0.2
        assert config.dag_weighting.enable_archaeological_zone_weighting is True
        assert config.dag_weighting.zone_influence_radius == 4.0
        assert config.threshold == 70.0
        assert config.quality_gate == 85.0
    
    def test_export_to_dict(self):
        """Test exporting configuration to dictionary."""
        config = ConfluenceConfig(
            threshold=75.0,
            quality_gate=90.0
        )
        config.dag_weighting.enable_archaeological_zone_weighting = True
        
        exported = export_confluence_config(config)
        
        assert exported['threshold'] == 75.0
        assert exported['quality_gate'] == 90.0
        assert exported['dag']['features']['enable_archaeological_zone_weighting'] is True
        assert 'weights' in exported
        assert 'dag' in exported
    
    def test_roundtrip_serialization(self):
        """Test roundtrip serialization preserves configuration."""
        original_config = create_confluence_config(
            weights={'temporal_coherence': 0.3, 'pattern_strength': 0.7},
            dag_features={'enable_archaeological_zone_weighting': True, 'zone_influence_radius': 5.0},
            threshold=80.0
        )
        
        # Export to dict
        exported = export_confluence_config(original_config)
        
        # Import from dict
        imported_config = load_confluence_config_from_dict(exported)
        
        # Should be equivalent
        assert imported_config.weights.temporal_coherence == original_config.weights.temporal_coherence
        assert imported_config.weights.pattern_strength == original_config.weights.pattern_strength
        assert imported_config.dag_weighting.enable_archaeological_zone_weighting == original_config.dag_weighting.enable_archaeological_zone_weighting
        assert imported_config.dag_weighting.zone_influence_radius == original_config.dag_weighting.zone_influence_radius
        assert imported_config.threshold == original_config.threshold


class TestFeatureFlagBehavior:
    """Test archaeological DAG weighting feature flag behavior."""
    
    def test_feature_flag_disabled_by_default(self):
        """Test feature flag is disabled by default."""
        config = ConfluenceConfig()
        assert config.dag_weighting.enable_archaeological_zone_weighting is False
        
        # Default weights should not include DAG weights
        default_weights = get_default_weights(enable_dag_weighting=False)
        dag_weight_keys = ['dag_topology_weight', 'dag_centrality_weight', 'dag_flow_weight', 'dag_clustering_weight']
        for key in dag_weight_keys:
            assert key not in default_weights
    
    def test_feature_flag_enabled_behavior(self):
        """Test behavior when feature flag is enabled."""
        config = create_confluence_config(
            dag_features={'enable_archaeological_zone_weighting': True}
        )
        
        assert config.dag_weighting.enable_archaeological_zone_weighting is True
        
        # DAG-enabled weights should include DAG weights
        dag_weights = get_default_weights(enable_dag_weighting=True)
        dag_weight_keys = ['dag_topology_weight', 'dag_centrality_weight', 'dag_flow_weight', 'dag_clustering_weight']
        for key in dag_weight_keys:
            assert key in dag_weights
            assert dag_weights[key] > 0
    
    def test_backward_compatibility(self):
        """Test backward compatibility when feature flag is disabled."""
        # Old-style configuration should still work
        old_config_dict = {
            'weights': {
                'temporal_coherence': 0.25,
                'pattern_strength': 0.30,
                'archaeological_significance': 0.20,
                'session_context': 0.15,
                'discovery_confidence': 0.10
            },
            'threshold': 65.0
        }
        
        config = load_confluence_config_from_dict(old_config_dict)
        
        # Should work without DAG features
        assert config.dag_weighting.enable_archaeological_zone_weighting is False
        assert config.weights.temporal_coherence == 0.25
        assert config.threshold == 65.0
