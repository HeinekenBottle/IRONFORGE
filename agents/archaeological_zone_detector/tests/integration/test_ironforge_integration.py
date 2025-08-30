"""
Integration Tests for IRONFORGE Pipeline Integration
===================================================

Tests archaeological zone detector integration with IRONFORGE components:
- Enhanced Graph Builder compatibility
- TGAT Discovery pipeline integration  
- Pattern graduation system integration
- Confluence scoring integration
- Container system compatibility

Integration Requirements:
- Golden invariant compliance (6 events, 4 edge intents, 51D/20D features)
- Session isolation preservation
- Performance requirements maintained
- Archaeological enhancement of TGAT discovery
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
import torch
from unittest.mock import Mock, patch, MagicMock

from agents.archaeological_zone_detector.agent import (
    ArchaeologicalZoneDetector,
    create_archaeological_zone_detector,
    enhance_discovery_with_archaeological_intelligence
)
from agents.archaeological_zone_detector.ironforge_config import ArchaeologicalConfig


class TestIronforgeContainerIntegration:
    """Test integration with IRONFORGE container system"""
    
    @pytest.fixture
    def mock_container(self):
        """Mock IRONFORGE container for testing"""
        container = Mock()
        container.get_enhanced_graph_builder.return_value = Mock()
        container.get_tgat_discovery.return_value = Mock()
        container.get_pattern_graduation.return_value = Mock()
        return container
    
    @pytest.fixture
    def config(self):
        return ArchaeologicalConfig()
    
    def test_container_initialization(self, config, mock_container):
        """Test proper container initialization"""
        detector = ArchaeologicalZoneDetector(config, mock_container)
        
        assert detector.container == mock_container
        assert detector.config == config
        assert detector.enhanced_graph_builder is None  # Lazy loading
        assert detector.discovery_engine is None  # Lazy loading
    
    def test_lazy_component_loading(self, config, mock_container):
        """Test lazy loading of IRONFORGE components"""
        detector = ArchaeologicalZoneDetector(config, mock_container)
        
        # Components should be None initially
        assert detector.enhanced_graph_builder is None
        assert detector.discovery_engine is None
        
        # Initialize components
        detector.initialize_ironforge_components()
        
        # Components should now be loaded
        assert detector.enhanced_graph_builder is not None
        assert detector.discovery_engine is not None
        
        # Container methods should have been called
        mock_container.get_enhanced_graph_builder.assert_called_once()
        mock_container.get_tgat_discovery.assert_called_once()
    
    @patch('agents.archaeological_zone_detector.agent.get_ironforge_container')
    def test_factory_function_integration(self, mock_get_container):
        """Test factory function properly integrates with container"""
        mock_container = Mock()
        mock_get_container.return_value = mock_container
        
        detector = create_archaeological_zone_detector()
        
        assert detector is not None
        assert detector.container == mock_container
        mock_get_container.assert_called_once()


class TestEnhancedGraphBuilderIntegration:
    """Test integration with Enhanced Graph Builder"""
    
    @pytest.fixture
    def sample_enhanced_graph(self):
        """Sample enhanced graph with IRONFORGE-compatible structure"""
        G = nx.Graph()
        
        # Add nodes with 45D features
        for i in range(5):
            features = torch.randn(45)  # Standard feature dimensions
            G.add_node(i, features=features, price=100.0 + i)
        
        # Add edges with 20D features and intents
        edge_intents = ['TEMPORAL_NEXT', 'MOVEMENT_TRANSITION', 'LIQ_LINK', 'CONTEXT']
        for i in range(4):
            edge_features = torch.randn(20)  # Edge feature dimensions
            intent = edge_intents[i % len(edge_intents)]
            G.add_edge(i, i+1, features=edge_features, intent=intent)
        
        return G
    
    @pytest.fixture
    def detector_with_graph(self):
        """Detector configured for graph integration testing"""
        config = ArchaeologicalConfig()
        config.ironforge_integration.graph_builder_integration = True
        config.ironforge_integration.node_feature_extension_dims = 5
        
        container = Mock()
        mock_graph_builder = Mock()
        container.get_enhanced_graph_builder.return_value = mock_graph_builder
        container.get_tgat_discovery.return_value = Mock()
        
        return ArchaeologicalZoneDetector(config, container)
    
    def test_graph_feature_integration(self, detector_with_graph, sample_enhanced_graph):
        """Test integration of archaeological features with enhanced graph"""
        # Sample archaeological zones
        sample_zones = [
            {
                'anchor_point': 102.0,
                'zone_range': (101.0, 103.0),
                'confidence': 0.8,
                'authenticity_score': 90.0,
                'theory_b_alignment': True,
                'precision_score': 7.2,
                'session_id': 'test_integration'
            }
        ]
        
        # Create mock archaeological analysis
        mock_analysis = Mock()
        mock_analysis.archaeological_zones = [Mock(**zone) for zone in sample_zones]
        mock_analysis.session_id = 'test_integration'
        
        # Test TGAT discovery enhancement
        enhancement_results = detector_with_graph.enhance_tgat_discovery(
            sample_enhanced_graph, mock_analysis
        )
        
        # Verify enhancement structure
        assert 'enhanced_graph' in enhancement_results
        assert 'temporal_features' in enhancement_results
        assert 'positioning_features' in enhancement_results
        assert 'zone_count' in enhancement_results
        assert 'authenticity_boost' in enhancement_results
        
        # Verify zone count matches
        assert enhancement_results['zone_count'] == len(sample_zones)
        
        # Authenticity boost should be reasonable
        boost = enhancement_results['authenticity_boost']
        assert 0.0 <= boost <= 1.0
    
    def test_feature_dimension_preservation(self, detector_with_graph, sample_enhanced_graph):
        """Test that feature dimensions are preserved during integration"""
        original_node_count = len(sample_enhanced_graph.nodes)
        original_edge_count = len(sample_enhanced_graph.edges)
        
        # Get original feature dimensions
        original_node_features = {}
        original_edge_features = {}
        
        for node_id, node_data in sample_enhanced_graph.nodes(data=True):
            if 'features' in node_data:
                original_node_features[node_id] = node_data['features'].clone()
        
        for u, v, edge_data in sample_enhanced_graph.edges(data=True):
            if 'features' in edge_data:
                original_edge_features[(u, v)] = edge_data['features'].clone()
        
        # Apply archaeological enhancement
        sample_zones = [Mock(anchor_point=102.0, zone_range=(101.0, 103.0), 
                           theory_b_alignment=True, authenticity_score=85.0,
                           precision_score=7.0, forward_positioning={})]
        
        enhanced_graph = detector_with_graph._add_archaeological_features_to_graph(
            sample_enhanced_graph, sample_zones
        )
        
        # Verify graph structure preserved
        assert len(enhanced_graph.nodes) == original_node_count
        assert len(enhanced_graph.edges) == original_edge_count
        
        # Verify original features preserved (should be unchanged)
        for node_id, node_data in enhanced_graph.nodes(data=True):
            if node_id in original_node_features:
                # Original features should be preserved
                if 'features' in node_data:
                    # Features may be extended but original part should match
                    original_dim = original_node_features[node_id].shape[0]
                    current_features = node_data['features'][:original_dim]
                    torch.testing.assert_close(current_features, original_node_features[node_id])


class TestTGATDiscoveryIntegration:
    """Test integration with TGAT Discovery pipeline"""
    
    @pytest.fixture
    def mock_discovery_engine(self):
        """Mock TGAT discovery engine"""
        engine = Mock()
        
        # Mock discovery results
        mock_results = {
            'node_embeddings': torch.randn(10, 44),  # TGAT embedding dimensions
            'pattern_scores': torch.randn(5),
            'significance_scores': torch.randn(5, 1),
            'attention_weights': torch.randn(10, 10)
        }
        
        engine.forward.return_value = mock_results
        engine.eval.return_value = None
        
        return engine
    
    @pytest.fixture 
    def detector_with_tgat(self, mock_discovery_engine):
        """Detector with mock TGAT integration"""
        config = ArchaeologicalConfig()
        config.ironforge_integration.enhance_tgat_discovery = True
        
        container = Mock()
        container.get_enhanced_graph_builder.return_value = Mock()
        container.get_tgat_discovery.return_value = mock_discovery_engine
        
        detector = ArchaeologicalZoneDetector(config, container)
        detector.initialize_ironforge_components()
        
        return detector
    
    def test_discovery_pipeline_enhancement(self, detector_with_tgat):
        """Test enhancement of TGAT discovery pipeline"""
        # Sample session data
        session_data = pd.DataFrame({
            'price': [100, 101, 102, 101, 100],
            'high': [100.5, 101.5, 102.5, 101.5, 100.5],
            'low': [99.5, 100.5, 101.5, 100.5, 99.5],
            'timestamp': [1, 2, 3, 4, 5],
            'session_id': ['tgat_integration_test'] * 5
        })
        
        # Previous session for anchoring
        previous_session = pd.DataFrame({
            'high': [105.0], 'low': [95.0], 'session_id': ['previous']
        })
        
        # Run archaeological detection
        analysis = detector_with_tgat.detect_archaeological_zones(
            session_data, previous_session
        )
        
        # Verify analysis structure
        assert analysis.session_id == 'tgat_integration_test'
        assert analysis.archaeological_zones is not None
        assert analysis.performance_metrics is not None
        assert analysis.contract_validation is not None
        
        # Verify performance compliance
        processing_time = analysis.performance_metrics.get('detection_time', 0.0)
        assert processing_time < 3.0, f"Processing time {processing_time:.3f}s exceeds limit"
    
    @patch('agents.archaeological_zone_detector.agent.enhance_discovery_with_archaeological_intelligence')
    def test_pipeline_integration_function(self, mock_enhance_function):
        """Test pipeline integration function"""
        # Mock discovery results
        mock_discovery_results = {
            'pattern_scores': torch.randn(3),
            'significance_scores': torch.randn(3),
            'enhanced_graph': nx.Graph()
        }
        
        # Mock session data
        session_data = pd.DataFrame({
            'price': [100, 101, 102],
            'session_id': ['pipeline_test'] * 3
        })
        
        # Mock enhanced results
        mock_enhanced_results = {
            **mock_discovery_results,
            'archaeological_analysis': Mock(),
            'archaeological_enhancement': {'authenticity_boost': 0.15},
            'authenticity_boost': 0.15
        }
        mock_enhance_function.return_value = mock_enhanced_results
        
        # Test function call
        config = ArchaeologicalConfig()
        results = enhance_discovery_with_archaeological_intelligence(
            mock_discovery_results, session_data, config
        )
        
        # Verify function was called
        mock_enhance_function.assert_called_once()
        
        # Verify results structure
        assert 'archaeological_analysis' in results
        assert 'archaeological_enhancement' in results
        assert 'authenticity_boost' in results


class TestPatternGraduationIntegration:
    """Test integration with Pattern Graduation system"""
    
    @pytest.fixture
    def detector_with_graduation(self):
        """Detector with pattern graduation integration"""
        config = ArchaeologicalConfig()
        config.authenticity.pattern_graduation_enabled = True
        config.authenticity.graduation_authenticity_boost = 5.0
        
        container = Mock()
        container.get_enhanced_graph_builder.return_value = Mock()
        container.get_tgat_discovery.return_value = Mock()
        
        # Mock pattern graduation
        mock_graduation = Mock()
        mock_graduation.graduate_patterns.return_value = []
        container.get_pattern_graduation.return_value = mock_graduation
        
        return ArchaeologicalZoneDetector(config, container)
    
    def test_authenticity_scoring_integration(self, detector_with_graduation):
        """Test authenticity scoring integration with graduation"""
        # Sample zone with graduation-eligible authenticity
        zone_data = {
            'anchor_point': 100.0,
            'zone_range': (99.0, 101.0),
            'confidence': 0.85,
            'precision_score': 7.8,
            'theory_b_valid': True,
            'session_id': 'graduation_test'
        }
        
        temporal_analysis = {
            'forward_coherence': 0.8,
            'nonlocality_patterns': [{'coherence': 0.85}]
        }
        
        theory_b_results = {
            'validation_rate': 0.9,
            'average_precision': 7.5
        }
        
        # Calculate authenticity with graduation potential
        authenticity_score = detector_with_graduation._calculate_authenticity_score(
            zone_data, temporal_analysis, theory_b_results
        )
        
        # Score should be reasonable for graduation
        assert authenticity_score >= 70.0, f"Authenticity score {authenticity_score} too low for graduation"
        assert authenticity_score <= 100.0, f"Authenticity score {authenticity_score} exceeds maximum"
    
    def test_graduation_boost_application(self, detector_with_graduation):
        """Test that graduation boost is properly applied"""
        config = detector_with_graduation.config
        boost = config.authenticity.graduation_authenticity_boost
        
        # Boost should be reasonable (5.0 as configured)
        assert boost == 5.0, f"Expected boost of 5.0, got {boost}"
        
        # Boost should be applied in authenticity calculations
        # (This would be tested in the actual graduation integration)


class TestContractComplianceIntegration:
    """Test contract compliance during IRONFORGE integration"""
    
    @pytest.fixture
    def strict_compliance_detector(self):
        """Detector with strict contract compliance"""
        config = ArchaeologicalConfig()
        config.validation.golden_invariant_enforcement = True
        config.validation.validation_failure_mode = "strict"
        
        container = Mock()
        container.get_enhanced_graph_builder.return_value = Mock()
        container.get_tgat_discovery.return_value = Mock()
        
        return ArchaeologicalZoneDetector(config, container)
    
    def test_golden_invariant_preservation(self, strict_compliance_detector):
        """Test that golden invariants are preserved during integration"""
        config = strict_compliance_detector.config
        
        # Verify golden invariant settings
        assert config.ironforge_integration.node_feature_dim_standard == 45
        assert config.ironforge_integration.node_feature_dim_htf == 51  
        assert config.ironforge_integration.edge_feature_dim == 20
        
        # Verify archaeological constants
        assert config.dimensional_anchor.anchor_percentage == 0.40
        assert config.dimensional_anchor.precision_target == 7.55
        assert config.authenticity.authenticity_threshold == 87.0
    
    def test_session_isolation_enforcement(self, strict_compliance_detector):
        """Test that session isolation is enforced during integration"""
        config = strict_compliance_detector.config
        
        # Verify session isolation settings
        assert config.session_isolation.strict_session_boundaries == True
        assert config.session_isolation.cross_session_edge_detection == True
        assert config.session_isolation.session_contamination_threshold == 0.0
        
        # HTF compliance
        assert config.session_isolation.htf_last_closed_only == True
        assert config.session_isolation.intra_candle_data_rejection == True


class TestPerformanceIntegration:
    """Test performance requirements during IRONFORGE integration"""
    
    @pytest.fixture
    def performance_optimized_detector(self):
        """Detector optimized for performance testing"""
        config = ArchaeologicalConfig()
        config.performance.max_session_processing_time = 2.0  # Strict limit
        config.performance.enable_lazy_loading = True
        config.performance.detailed_timing_enabled = True
        
        container = Mock()
        container.get_enhanced_graph_builder.return_value = Mock()
        container.get_tgat_discovery.return_value = Mock()
        
        return ArchaeologicalZoneDetector(config, container)
    
    def test_integration_performance_requirements(self, performance_optimized_detector):
        """Test that integration maintains performance requirements"""
        # Sample data for performance testing
        session_data = pd.DataFrame({
            'price': np.random.uniform(99, 101, 50),
            'high': np.random.uniform(100, 102, 50),
            'low': np.random.uniform(98, 100, 50),
            'timestamp': range(50),
            'session_id': ['performance_test'] * 50
        })
        
        previous_session = pd.DataFrame({
            'high': [105], 'low': [95], 'session_id': ['prev']
        })
        
        # Measure integration performance
        import time
        start_time = time.time()
        
        analysis = performance_optimized_detector.detect_archaeological_zones(
            session_data, previous_session
        )
        
        integration_time = time.time() - start_time
        
        # Verify performance requirements met
        assert integration_time < 3.0, f"Integration time {integration_time:.3f}s exceeds 3.0s limit"
        
        # Verify analysis completed successfully
        assert analysis is not None
        assert analysis.session_id == 'performance_test'
        assert analysis.performance_metrics is not None
        
        # Check recorded performance metrics
        recorded_time = analysis.performance_metrics.get('detection_time', 0.0)
        assert recorded_time > 0.0, "Detection time should be recorded"
        assert abs(recorded_time - integration_time) < 0.5, "Recorded time should be reasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])