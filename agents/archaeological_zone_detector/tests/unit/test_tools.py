"""
Unit Tests for Archaeological Zone Analysis Tools
=================================================

Tests core archaeological analysis components:
- DimensionalAnchorCalculator: 40% anchor point calculations
- TemporalNonLocalityValidator: Theory B forward positioning
- TheoryBValidator: Forward positioning validation
- ZoneAnalyzer: Comprehensive analysis integration

Test Requirements:
- Archaeological constants must be preserved (40% anchoring)
- Precision target validation (7.55 points)
- Session isolation enforcement
- Performance requirements (<1s zone detection)
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch

from agents.archaeological_zone_detector.tools import (
    DimensionalAnchorCalculator,
    TemporalNonLocalityValidator,
    TheoryBValidator,
    ZoneAnalyzer,
    RangeCalculationMethod,
    ArchaeologicalZoneData,
    TemporalEcho,
    TheoryBResult
)
from agents.archaeological_zone_detector.ironforge_config import (
    ArchaeologicalConfig,
    ConfigurationPresets
)


class TestDimensionalAnchorCalculator:
    """Test dimensional anchor calculation with 40% archaeological constant"""
    
    @pytest.fixture
    def config(self):
        return ArchaeologicalConfig()
    
    @pytest.fixture
    def calculator(self, config):
        return DimensionalAnchorCalculator(config)
    
    @pytest.fixture
    def sample_session_data(self):
        return pd.DataFrame({
            'high': [100.5, 101.2, 102.0, 101.8, 100.9],
            'low': [99.1, 99.8, 100.2, 100.5, 99.7],
            'close': [100.2, 101.0, 101.5, 101.2, 100.3],
            'open': [99.8, 100.2, 101.0, 101.5, 101.2],
            'session_id': ['test_session'] * 5,
            'timestamp': [1, 2, 3, 4, 5]
        })
    
    def test_archaeological_constant_preserved(self, calculator, sample_session_data):
        """Test that 40% archaeological constant is preserved in calculations"""
        previous_range = 10.0  # Test range
        
        anchor_zones = calculator.calculate_dimensional_anchors(
            previous_range, sample_session_data
        )
        
        # Verify 40% constant is applied
        for zone in anchor_zones:
            expected_width = previous_range * 0.40
            actual_width = zone['zone_width']
            
            # Allow small tolerance for floating point arithmetic
            assert abs(actual_width - expected_width) / expected_width < 0.1, \
                f"Zone width {actual_width} deviates significantly from 40% constant {expected_width}"
    
    def test_range_calculation_methods(self, calculator, sample_session_data):
        """Test different range calculation methods produce reasonable results"""
        methods = [
            RangeCalculationMethod.HIGH_LOW,
            RangeCalculationMethod.OPEN_CLOSE,
            RangeCalculationMethod.BODY_RANGE,
            RangeCalculationMethod.WEIGHTED_AVERAGE
        ]
        
        results = {}
        for method in methods:
            range_value = calculator.calculate_session_range(sample_session_data, method)
            results[method] = range_value
            
            # All methods should produce positive ranges
            assert range_value >= 0, f"Range calculation {method} produced negative value: {range_value}"
        
        # High-low should typically be the largest range
        assert results[RangeCalculationMethod.HIGH_LOW] >= results[RangeCalculationMethod.OPEN_CLOSE]
    
    def test_anchor_zone_structure(self, calculator, sample_session_data):
        """Test that anchor zones have correct structure and properties"""
        previous_range = 5.0
        anchor_zones = calculator.calculate_dimensional_anchors(previous_range, sample_session_data)
        
        for i, zone in enumerate(anchor_zones):
            # Required properties
            required_keys = [
                'anchor_point', 'zone_range', 'confidence', 'precision_score',
                'session_id', 'zone_width', 'previous_range'
            ]
            
            for key in required_keys:
                assert key in zone, f"Zone {i} missing required property: {key}"
            
            # Zone range structure
            zone_range = zone['zone_range']
            assert isinstance(zone_range, tuple), f"Zone range should be tuple, got {type(zone_range)}"
            assert len(zone_range) == 2, f"Zone range should have 2 elements, got {len(zone_range)}"
            assert zone_range[0] < zone_range[1], f"Zone range order incorrect: {zone_range}"
            
            # Confidence bounds
            confidence = zone['confidence']
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of bounds [0, 1]"
            
            # Precision score bounds
            precision = zone['precision_score']
            assert 0.0 <= precision <= 10.0, f"Precision score {precision} out of bounds [0, 10]"
    
    def test_performance_requirements(self, calculator, sample_session_data):
        """Test that anchor calculation meets performance requirements"""
        previous_range = 8.0
        
        start_time = time.time()
        anchor_zones = calculator.calculate_dimensional_anchors(previous_range, sample_session_data)
        calculation_time = time.time() - start_time
        
        # Should complete well under 1 second
        assert calculation_time < 0.1, f"Anchor calculation took {calculation_time:.3f}s, should be <0.1s"
        
        # Should produce reasonable number of zones
        assert 1 <= len(anchor_zones) <= 10, f"Unexpected zone count: {len(anchor_zones)}"
    
    def test_edge_cases(self, calculator):
        """Test edge cases and error handling"""
        # Empty data
        empty_data = pd.DataFrame()
        zones = calculator.calculate_dimensional_anchors(5.0, empty_data)
        assert len(zones) == 0, "Empty data should produce no zones"
        
        # Zero previous range
        sample_data = pd.DataFrame({'high': [100], 'low': [99], 'session_id': ['test']})
        zones = calculator.calculate_dimensional_anchors(0.0, sample_data)
        assert len(zones) == 0, "Zero previous range should produce no zones"
        
        # Invalid session data (no price columns)
        invalid_data = pd.DataFrame({'other_col': [1, 2, 3]})
        zones = calculator.calculate_dimensional_anchors(5.0, invalid_data)
        assert len(zones) == 0, "Data without price columns should produce no zones"


class TestTemporalNonLocalityValidator:
    """Test temporal non-locality analysis and Theory B validation"""
    
    @pytest.fixture
    def config(self):
        return ArchaeologicalConfig()
    
    @pytest.fixture
    def validator(self, config):
        return TemporalNonLocalityValidator(config)
    
    @pytest.fixture
    def sample_session_with_events(self):
        np.random.seed(42)  # Reproducible test data
        
        timestamps = range(1, 51)  # 50 time points
        prices = 100 + np.cumsum(np.random.randn(50) * 0.1)  # Random walk
        event_types = np.random.choice(['Expansion', 'Consolidation', 'Retracement'], 50)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'event_type': event_types,
            'volume': np.random.uniform(1000, 5000, 50),
            'session_id': ['test_session'] * 50
        })
    
    @pytest.fixture
    def sample_anchor_zones(self):
        return [
            {
                'anchor_point': 100.0,
                'zone_range': (98.0, 102.0),
                'confidence': 0.8,
                'session_id': 'test_session',
                'zone_width': 4.0
            },
            {
                'anchor_point': 105.0,
                'zone_range': (103.0, 107.0),
                'confidence': 0.7,
                'session_id': 'test_session',
                'zone_width': 4.0
            }
        ]
    
    def test_nonlocality_analysis_structure(self, validator, sample_session_with_events, sample_anchor_zones):
        """Test that non-locality analysis produces correct structure"""
        analysis = validator.analyze_nonlocality(sample_session_with_events, sample_anchor_zones)
        
        # Required analysis components
        required_keys = [
            'event_sequence_length', 'causality_patterns', 'temporal_coherence',
            'forward_positioning', 'nonlocality_score'
        ]
        
        for key in required_keys:
            assert key in analysis, f"Analysis missing required component: {key}"
        
        # Validation bounds
        assert isinstance(analysis['event_sequence_length'], int)
        assert analysis['event_sequence_length'] > 0
        assert 0.0 <= analysis['nonlocality_score'] <= 1.0
        
        # Causality patterns structure
        causality_patterns = analysis['causality_patterns']
        assert isinstance(causality_patterns, list)
        
        for pattern in causality_patterns:
            assert 'causality_strength' in pattern
            assert 0.0 <= pattern['causality_strength'] <= 1.0
    
    def test_temporal_echo_detection(self, validator, sample_session_with_events, sample_anchor_zones):
        """Test temporal echo detection functionality"""
        temporal_echoes = validator.detect_temporal_echoes(sample_session_with_events, sample_anchor_zones)
        
        assert isinstance(temporal_echoes, list)
        
        for echo in temporal_echoes:
            assert isinstance(echo, TemporalEcho)
            assert echo.echo_id is not None
            assert 0.0 <= echo.propagation_strength <= 1.0
            assert 0.0 <= echo.coherence_score <= 1.0
            assert echo.temporal_distance > 0
            assert isinstance(echo.forward_validation, bool)
    
    def test_theory_b_compliance(self, validator, sample_session_with_events, sample_anchor_zones):
        """Test Theory B forward positioning compliance"""
        analysis = validator.analyze_nonlocality(sample_session_with_events, sample_anchor_zones)
        
        # Forward positioning should be analyzed
        forward_positioning = analysis['forward_positioning']
        assert 'enabled' in forward_positioning
        
        if forward_positioning['enabled']:
            assert 'patterns' in forward_positioning
            assert 'average_positioning_strength' in forward_positioning
            
            # Positioning strength should be bounded
            avg_strength = forward_positioning['average_positioning_strength']
            assert 0.0 <= avg_strength <= 1.0
    
    def test_performance_requirements(self, validator, sample_session_with_events, sample_anchor_zones):
        """Test that temporal analysis meets performance requirements"""
        start_time = time.time()
        analysis = validator.analyze_nonlocality(sample_session_with_events, sample_anchor_zones)
        analysis_time = time.time() - start_time
        
        # Should complete quickly for temporal analysis
        assert analysis_time < 0.5, f"Temporal analysis took {analysis_time:.3f}s, should be <0.5s"
        
        # Should produce meaningful results
        assert analysis['nonlocality_score'] > 0.0, "Non-locality analysis should produce positive score"


class TestTheoryBValidator:
    """Test Theory B forward positioning validation"""
    
    @pytest.fixture
    def config(self):
        return ArchaeologicalConfig()
    
    @pytest.fixture
    def validator(self, config):
        return TheoryBValidator(config)
    
    @pytest.fixture
    def sample_session_with_completion(self):
        """Session data with clear completion pattern"""
        prices = [99.0, 100.0, 101.0, 102.0, 101.5, 100.5, 99.5, 100.0]
        return pd.DataFrame({
            'timestamp': range(len(prices)),
            'price': prices,
            'high': [p + 0.2 for p in prices],
            'low': [p - 0.2 for p in prices],
            'close': prices,
            'session_id': ['test_session'] * len(prices)
        })
    
    @pytest.fixture
    def sample_anchor_zones_for_theory_b(self):
        return [
            {
                'anchor_point': 100.0,
                'zone_range': (99.0, 101.0),
                'confidence': 0.8,
                'session_id': 'test_session',
                'zone_width': 2.0,
                'previous_range': 5.0
            }
        ]
    
    def test_precision_score_calculation(self, validator, sample_anchor_zones_for_theory_b, sample_session_with_completion):
        """Test precision score calculation targeting 7.55"""
        precision_scores = validator.calculate_precision_scores(
            sample_anchor_zones_for_theory_b,
            sample_session_with_completion
        )
        
        assert len(precision_scores) == len(sample_anchor_zones_for_theory_b)
        
        for score_data in precision_scores:
            assert 'precision' in score_data
            assert 'precision_target' in score_data
            assert 'precision_accuracy' in score_data
            assert 'target_achievement' in score_data
            
            # Precision should be positive and bounded
            precision = score_data['precision']
            assert 0.0 <= precision <= 10.0, f"Precision {precision} out of bounds [0, 10]"
            
            # Target should be 7.55 (archaeological constant)
            target = score_data['precision_target']
            assert target == 7.55, f"Precision target should be 7.55, got {target}"
    
    def test_forward_positioning_validation(self, validator, sample_session_with_completion, sample_anchor_zones_for_theory_b):
        """Test forward positioning validation logic"""
        temporal_analysis = {'forward_positioning': {'patterns': []}}
        
        results = validator.validate_forward_positioning(
            sample_session_with_completion,
            sample_anchor_zones_for_theory_b,
            temporal_analysis
        )
        
        assert 'enabled' in results
        assert 'validation_results' in results
        assert 'validation_rate' in results
        
        # Validation rate should be bounded
        rate = results['validation_rate']
        assert 0.0 <= rate <= 1.0, f"Validation rate {rate} out of bounds [0, 1]"
        
        # Each validation result should be proper TheoryBResult
        for result in results['validation_results']:
            assert isinstance(result, TheoryBResult)
            assert 0.0 <= result.completion_accuracy <= 1.0
            assert 0.0 <= result.precision_score <= 10.0
            assert 0.0 <= result.temporal_coherence <= 1.0


class TestZoneAnalyzer:
    """Test comprehensive zone analysis integration"""
    
    @pytest.fixture
    def config(self):
        return ConfigurationPresets.development_config()
    
    @pytest.fixture
    def analyzer(self, config):
        return ZoneAnalyzer(config)
    
    @pytest.fixture
    def comprehensive_session_data(self):
        """Comprehensive session data for full analysis testing"""
        np.random.seed(123)  # Reproducible
        
        n_points = 100
        base_price = 1000.0
        price_walk = np.cumsum(np.random.randn(n_points) * 0.5)
        prices = base_price + price_walk
        
        return pd.DataFrame({
            'timestamp': range(n_points),
            'price': prices,
            'high': prices + np.random.uniform(0.1, 0.5, n_points),
            'low': prices - np.random.uniform(0.1, 0.5, n_points),
            'close': prices + np.random.uniform(-0.2, 0.2, n_points),
            'open': prices + np.random.uniform(-0.3, 0.3, n_points),
            'volume': np.random.uniform(1000, 10000, n_points),
            'event_type': np.random.choice(['Expansion', 'Consolidation', 'Retracement'], n_points),
            'session_id': ['comprehensive_test'] * n_points
        })
    
    @pytest.fixture
    def previous_session_data(self):
        """Previous session for dimensional anchoring"""
        return pd.DataFrame({
            'high': [1005.0, 1008.0, 1010.0],
            'low': [995.0, 998.0, 1000.0],
            'close': [1000.0, 1005.0, 1008.0],
            'session_id': ['previous_session'] * 3
        })
    
    def test_comprehensive_analysis_structure(self, analyzer, comprehensive_session_data, previous_session_data):
        """Test that comprehensive analysis produces all required components"""
        analysis = analyzer.analyze_session_zones(comprehensive_session_data, previous_session_data)
        
        # Top-level structure
        required_sections = [
            'session_analysis', 'dimensional_anchors', 'temporal_analysis',
            'theory_b_validation', 'quality_metrics'
        ]
        
        for section in required_sections:
            assert section in analysis, f"Analysis missing required section: {section}"
        
        # Session analysis
        session_info = analysis['session_analysis']
        assert 'session_id' in session_info
        assert 'processing_time' in session_info
        assert 'previous_range' in session_info
        
        # Dimensional anchors
        anchors = analysis['dimensional_anchors']
        assert 'anchor_zones' in anchors
        assert 'zone_count' in anchors
        assert anchors['anchor_percentage'] == 0.40  # Archaeological constant
        
        # Quality metrics
        quality = analysis['quality_metrics']
        assert 'processing_performance' in quality
        assert 'authenticity_ready' in quality
        assert 'theory_b_compliance' in quality
    
    def test_performance_compliance(self, analyzer, comprehensive_session_data, previous_session_data):
        """Test that comprehensive analysis meets performance requirements"""
        start_time = time.time()
        analysis = analyzer.analyze_session_zones(comprehensive_session_data, previous_session_data)
        total_time = time.time() - start_time
        
        # Should meet performance requirements
        assert total_time < 3.0, f"Comprehensive analysis took {total_time:.3f}s, should be <3.0s"
        
        # Processing time should be recorded accurately
        recorded_time = analysis['session_analysis']['processing_time']
        assert abs(recorded_time - total_time) < 0.1, "Recorded processing time should match actual"
        
        # Quality metrics should indicate performance compliance
        quality = analysis['quality_metrics']
        assert quality['processing_performance'] == True, "Processing performance should be compliant"
    
    def test_archaeological_constants_preserved(self, analyzer, comprehensive_session_data, previous_session_data):
        """Test that archaeological constants are preserved throughout analysis"""
        analysis = analyzer.analyze_session_zones(comprehensive_session_data, previous_session_data)
        
        # 40% anchoring constant
        anchors = analysis['dimensional_anchors']
        assert anchors['anchor_percentage'] == 0.40
        
        # Verify 40% is applied to actual zones
        anchor_zones = anchors['anchor_zones']
        previous_range = analysis['session_analysis']['previous_range']
        
        if anchor_zones and previous_range > 0:
            for zone in anchor_zones:
                expected_width = previous_range * 0.40
                actual_width = zone.get('zone_width', 0.0)
                if actual_width > 0:
                    # Allow reasonable tolerance for filtering and adjustments
                    deviation = abs(actual_width - expected_width) / expected_width
                    assert deviation < 0.3, f"Zone width deviates too much from 40% constant"
    
    def test_error_handling(self, analyzer):
        """Test error handling in comprehensive analysis"""
        # Empty data
        empty_data = pd.DataFrame()
        analysis = analyzer.analyze_session_zones(empty_data)
        
        assert 'error' in analysis['session_analysis'] or len(analysis['dimensional_anchors']['anchor_zones']) == 0
        
        # Malformed data
        bad_data = pd.DataFrame({'bad_column': [1, 2, 3]})
        analysis = analyzer.analyze_session_zones(bad_data)
        
        # Should handle gracefully without crashing
        assert analysis is not None
        assert isinstance(analysis, dict)


# Integration tests within tools module
class TestToolsIntegration:
    """Test integration between different tool components"""
    
    @pytest.fixture
    def full_config(self):
        return ArchaeologicalConfig()
    
    def test_component_compatibility(self, full_config):
        """Test that all tool components work together"""
        # Initialize all components
        calculator = DimensionalAnchorCalculator(full_config)
        temporal_validator = TemporalNonLocalityValidator(full_config)
        theory_b_validator = TheoryBValidator(full_config)
        analyzer = ZoneAnalyzer(full_config)
        
        # All should use same config
        assert calculator.config == temporal_validator.config == theory_b_validator.config == analyzer.config
        
        # All should be properly initialized
        assert calculator is not None
        assert temporal_validator is not None  
        assert theory_b_validator is not None
        assert analyzer is not None
    
    def test_data_flow_compatibility(self, full_config):
        """Test that data flows correctly between components"""
        calculator = DimensionalAnchorCalculator(full_config)
        temporal_validator = TemporalNonLocalityValidator(full_config)
        theory_b_validator = TheoryBValidator(full_config)
        
        # Create test data
        session_data = pd.DataFrame({
            'price': [100, 101, 102, 101, 100],
            'high': [100.5, 101.5, 102.5, 101.5, 100.5],
            'low': [99.5, 100.5, 101.5, 100.5, 99.5],
            'timestamp': [1, 2, 3, 4, 5],
            'session_id': ['integration_test'] * 5
        })
        
        # Flow data through components
        anchor_zones = calculator.calculate_dimensional_anchors(5.0, session_data)
        temporal_analysis = temporal_validator.analyze_nonlocality(session_data, anchor_zones)
        theory_b_results = theory_b_validator.validate_forward_positioning(
            session_data, anchor_zones, temporal_analysis
        )
        
        # Verify data compatibility
        assert len(anchor_zones) >= 0  # May be empty but should not error
        assert 'nonlocality_score' in temporal_analysis
        assert 'validation_rate' in theory_b_results
        
        # Data types should be consistent
        for zone in anchor_zones:
            assert isinstance(zone, dict)
            assert 'anchor_point' in zone
            assert 'zone_range' in zone


# Performance benchmarking tests
class TestToolsPerformance:
    """Performance benchmarks for archaeological tools"""
    
    @pytest.fixture
    def benchmark_session(self):
        """Large session for performance testing"""
        n_points = 1000
        np.random.seed(456)
        
        return pd.DataFrame({
            'price': 1000 + np.cumsum(np.random.randn(n_points) * 0.1),
            'high': 1000 + np.cumsum(np.random.randn(n_points) * 0.1) + 0.5,
            'low': 1000 + np.cumsum(np.random.randn(n_points) * 0.1) - 0.5,
            'timestamp': range(n_points),
            'session_id': ['benchmark'] * n_points
        })
    
    def test_anchor_calculation_performance(self, benchmark_session):
        """Benchmark anchor calculation performance"""
        config = ArchaeologicalConfig()
        calculator = DimensionalAnchorCalculator(config)
        
        # Multiple runs for average timing
        times = []
        for _ in range(10):
            start = time.time()
            zones = calculator.calculate_dimensional_anchors(20.0, benchmark_session)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        assert avg_time < 0.1, f"Average anchor calculation time {avg_time:.3f}s exceeds 0.1s limit"
    
    def test_temporal_analysis_performance(self, benchmark_session):
        """Benchmark temporal analysis performance"""
        config = ArchaeologicalConfig()
        validator = TemporalNonLocalityValidator(config)
        
        # Create anchor zones for analysis
        anchor_zones = [
            {'anchor_point': 1000, 'zone_range': (998, 1002), 'confidence': 0.8, 'session_id': 'benchmark'}
        ]
        
        start = time.time()
        analysis = validator.analyze_nonlocality(benchmark_session, anchor_zones)
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"Temporal analysis time {elapsed:.3f}s exceeds 1.0s limit"
        assert analysis['nonlocality_score'] >= 0.0  # Should produce valid results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])