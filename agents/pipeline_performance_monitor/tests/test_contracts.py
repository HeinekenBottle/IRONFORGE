"""
Test Suite for IRONFORGE Performance Contract Validation

Tests the contract validation system to ensure strict adherence to
performance requirements and golden invariants.
"""

import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts import (
    PerformanceContractValidator,
    ContractViolation,
    ContractValidationResult,
    ContractSeverity,
    ContractType
)
from ironforge_config import IRONFORGEPerformanceConfig


class TestContractViolation:
    """Test suite for ContractViolation data class."""
    
    def test_violation_creation(self):
        """Test creating contract violation."""
        violation = ContractViolation(
            contract_name="session_processing_time",
            contract_type=ContractType.TIMING,
            severity=ContractSeverity.CRITICAL,
            current_value="4.2s",
            expected_value="<3.0s",
            violation_percentage=40.0,
            description="Session processing exceeded 3s limit",
            blocking_production=True
        )
        
        assert violation.contract_name == "session_processing_time"
        assert violation.contract_type == ContractType.TIMING
        assert violation.severity == ContractSeverity.CRITICAL
        assert violation.blocking_production is True
        assert isinstance(violation.timestamp, datetime)
    
    def test_violation_to_dict(self):
        """Test converting violation to dictionary."""
        violation = ContractViolation(
            contract_name="memory_limit",
            contract_type=ContractType.MEMORY,
            severity=ContractSeverity.WARNING,
            current_value="85MB",
            expected_value="<100MB",
            violation_percentage=15.0,
            description="Memory usage approaching limit"
        )
        
        violation_dict = violation.to_dict()
        
        assert violation_dict["contract_name"] == "memory_limit"
        assert violation_dict["contract_type"] == "memory"
        assert violation_dict["severity"] == "warning"
        assert "timestamp" in violation_dict


class TestContractValidationResult:
    """Test suite for ContractValidationResult."""
    
    def test_successful_validation_result(self):
        """Test successful validation result."""
        result = ContractValidationResult(
            passed=True,
            total_contracts=10,
            compliance_score=1.0,
            production_ready=True
        )
        
        assert result.passed is True
        assert result.total_contracts == 10
        assert result.compliance_score == 1.0
        assert result.production_ready is True
        assert len(result.violations) == 0
        assert len(result.warnings) == 0
    
    def test_failed_validation_result(self):
        """Test failed validation result with violations."""
        violation = ContractViolation(
            contract_name="test_contract",
            contract_type=ContractType.TIMING,
            severity=ContractSeverity.CRITICAL,
            current_value="5.0",
            expected_value="<3.0",
            violation_percentage=66.7,
            description="Test violation",
            blocking_production=True
        )
        
        result = ContractValidationResult(
            passed=False,
            total_contracts=5,
            violations=[violation],
            compliance_score=0.8,
            production_ready=False
        )
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.compliance_score == 0.8
        assert result.production_ready is False
    
    def test_get_summary(self):
        """Test getting validation result summary."""
        result = ContractValidationResult(
            passed=True,
            total_contracts=8,
            compliance_score=0.95,
            production_ready=True
        )
        
        summary = result.get_summary()
        
        assert summary["passed"] is True
        assert summary["total_contracts"] == 8
        assert summary["compliance_score"] == 0.95
        assert summary["production_ready"] is True
        assert summary["violations_count"] == 0
        assert summary["critical_violations"] == 0


class TestPerformanceContractValidator:
    """Test suite for PerformanceContractValidator."""
    
    @pytest.fixture
    def config(self):
        """Fixture providing test configuration."""
        return IRONFORGEPerformanceConfig()
    
    @pytest.fixture
    def validator(self, config):
        """Fixture providing test validator instance."""
        return PerformanceContractValidator(config)
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.config is not None
        assert len(validator.timing_contracts) > 0
        assert len(validator.memory_contracts) > 0
        assert len(validator.quality_contracts) > 0
        assert len(validator.system_contracts) > 0
        assert len(validator.invariant_contracts) > 0
        assert validator.strict_mode is True
    
    def test_timing_contract_definitions(self, validator):
        """Test timing contract definitions."""
        timing_contracts = validator.timing_contracts
        
        # Check essential timing contracts exist
        assert "session_processing_time" in timing_contracts
        assert "full_discovery_time" in timing_contracts
        assert "initialization_time" in timing_contracts
        
        # Check contract structure
        session_contract = timing_contracts["session_processing_time"]
        assert session_contract["threshold"] == 3.0  # 3 seconds
        assert session_contract["operator"] == "less_than"
        assert session_contract["severity"] == ContractSeverity.CRITICAL
        assert session_contract["blocking"] is True
    
    def test_memory_contract_definitions(self, validator):
        """Test memory contract definitions."""
        memory_contracts = validator.memory_contracts
        
        assert "total_memory_limit" in memory_contracts
        limit_contract = memory_contracts["total_memory_limit"]
        assert limit_contract["threshold"] == 100.0  # 100MB
        assert limit_contract["operator"] == "less_than"
        assert limit_contract["severity"] == ContractSeverity.BLOCKING
    
    def test_quality_contract_definitions(self, validator):
        """Test quality contract definitions."""
        quality_contracts = validator.quality_contracts
        
        assert "pattern_authenticity" in quality_contracts
        auth_contract = quality_contracts["pattern_authenticity"]
        assert auth_contract["threshold"] == 0.87  # 87%
        assert auth_contract["operator"] == "greater_than"
        assert auth_contract["severity"] == ContractSeverity.BLOCKING
    
    def test_invariant_contract_definitions(self, validator):
        """Test golden invariant contract definitions."""
        invariant_contracts = validator.invariant_contracts
        
        # Check golden invariants
        assert "event_types_count" in invariant_contracts
        assert "edge_intent_types_count" in invariant_contracts
        assert "node_feature_dimensions" in invariant_contracts
        assert "edge_feature_dimensions" in invariant_contracts
        
        # Verify exact values for invariants
        assert invariant_contracts["event_types_count"]["threshold"] == 6
        assert invariant_contracts["edge_intent_types_count"]["threshold"] == 4
        assert invariant_contracts["node_feature_dimensions"]["threshold"] == 51
        assert invariant_contracts["edge_feature_dimensions"]["threshold"] == 20
        
        # All invariants should be blocking
        for contract in invariant_contracts.values():
            assert contract["blocking"] is True
            assert contract["severity"] == ContractSeverity.BLOCKING
    
    def test_successful_pipeline_validation(self, validator):
        """Test successful pipeline validation."""
        # Mock perfect pipeline results
        pipeline_results = {
            'overall_metrics': {
                'total_processing_time': 165.0,  # Under 180s limit
                'peak_memory_mb': 85.0,          # Under 100MB limit
                'time_compliance': True,
                'memory_compliance': True
            },
            'quality_metrics': {
                'average_authenticity': 0.92,    # Above 87% threshold
                'quality_gate_compliance': True
            },
            'stages': {
                'discovery': {'processing_time': 55.0},      # Under 60s
                'confluence': {'processing_time': 25.0},     # Under 30s
                'validation': {'processing_time': 12.0},     # Under 15s
                'reporting': {'processing_time': 70.0}       # Under 75s
            },
            'discovered_event_types': ['Expansion', 'Consolidation', 'Retracement', 'Reversal', 'LiquidityTaken', 'Redelivery'],
            'discovered_edge_intents': ['TEMPORAL_NEXT', 'MOVEMENT_TRANSITION', 'LIQ_LINK', 'CONTEXT'],
            'node_feature_dimensions': 51,
            'edge_feature_dimensions': 20,
            'errors': [],
            'total_operations': 100
        }
        
        result = validator.validate_pipeline_run(pipeline_results)
        
        assert result.passed is True
        assert result.production_ready is True
        assert len(result.violations) == 0
        assert result.compliance_score == 1.0
    
    def test_timing_violation_detection(self, validator):
        """Test detection of timing contract violations."""
        pipeline_results = {
            'overall_metrics': {
                'total_processing_time': 200.0,  # Exceeds 180s limit
                'peak_memory_mb': 85.0,
                'time_compliance': False,
                'memory_compliance': True
            },
            'quality_metrics': {
                'average_authenticity': 0.92,
                'quality_gate_compliance': True
            },
            'stages': {
                'discovery': {'processing_time': 55.0},
                'confluence': {'processing_time': 25.0},
                'validation': {'processing_time': 12.0},
                'reporting': {'processing_time': 70.0}
            },
            'discovered_event_types': ['Expansion', 'Consolidation', 'Retracement', 'Reversal', 'LiquidityTaken', 'Redelivery'],
            'discovered_edge_intents': ['TEMPORAL_NEXT', 'MOVEMENT_TRANSITION', 'LIQ_LINK', 'CONTEXT'],
            'node_feature_dimensions': 51,
            'edge_feature_dimensions': 20,
            'errors': [],
            'total_operations': 100
        }
        
        result = validator.validate_pipeline_run(pipeline_results)
        
        assert result.passed is False
        assert result.production_ready is False  # Should block production
        assert len(result.violations) > 0
        
        # Find the timing violation
        timing_violations = [v for v in result.violations if v.contract_type == ContractType.TIMING]
        assert len(timing_violations) > 0
        
        violation = timing_violations[0]
        assert violation.contract_name == "full_discovery_time"
        assert violation.severity == ContractSeverity.BLOCKING
        assert violation.blocking_production is True
    
    def test_memory_violation_detection(self, validator):
        """Test detection of memory contract violations."""
        pipeline_results = {
            'overall_metrics': {
                'total_processing_time': 165.0,
                'peak_memory_mb': 120.0,  # Exceeds 100MB limit
                'time_compliance': True,
                'memory_compliance': False
            },
            'quality_metrics': {
                'average_authenticity': 0.92,
                'quality_gate_compliance': True
            },
            'stages': {},
            'discovered_event_types': ['Expansion', 'Consolidation', 'Retracement', 'Reversal', 'LiquidityTaken', 'Redelivery'],
            'discovered_edge_intents': ['TEMPORAL_NEXT', 'MOVEMENT_TRANSITION', 'LIQ_LINK', 'CONTEXT'],
            'node_feature_dimensions': 51,
            'edge_feature_dimensions': 20,
            'errors': [],
            'total_operations': 100
        }
        
        result = validator.validate_pipeline_run(pipeline_results)
        
        assert result.passed is False
        assert result.production_ready is False
        
        memory_violations = [v for v in result.violations if v.contract_type == ContractType.MEMORY]
        assert len(memory_violations) > 0
        
        violation = memory_violations[0]
        assert violation.contract_name == "total_memory_limit"
        assert violation.severity == ContractSeverity.BLOCKING
    
    def test_quality_violation_detection(self, validator):
        """Test detection of quality contract violations."""
        pipeline_results = {
            'overall_metrics': {
                'total_processing_time': 165.0,
                'peak_memory_mb': 85.0,
                'time_compliance': True,
                'memory_compliance': True
            },
            'quality_metrics': {
                'average_authenticity': 0.82,  # Below 87% threshold
                'quality_gate_compliance': True
            },
            'stages': {},
            'discovered_event_types': ['Expansion', 'Consolidation', 'Retracement', 'Reversal', 'LiquidityTaken', 'Redelivery'],
            'discovered_edge_intents': ['TEMPORAL_NEXT', 'MOVEMENT_TRANSITION', 'LIQ_LINK', 'CONTEXT'],
            'node_feature_dimensions': 51,
            'edge_feature_dimensions': 20,
            'errors': [],
            'total_operations': 100
        }
        
        result = validator.validate_pipeline_run(pipeline_results)
        
        assert result.passed is False
        assert result.production_ready is False
        
        quality_violations = [v for v in result.violations if v.contract_type == ContractType.QUALITY]
        assert len(quality_violations) > 0
        
        violation = quality_violations[0]
        assert violation.contract_name == "pattern_authenticity"
        assert violation.severity == ContractSeverity.BLOCKING
    
    def test_invariant_violation_detection(self, validator):
        """Test detection of golden invariant violations."""
        pipeline_results = {
            'overall_metrics': {
                'total_processing_time': 165.0,
                'peak_memory_mb': 85.0,
                'time_compliance': True,
                'memory_compliance': True
            },
            'quality_metrics': {
                'average_authenticity': 0.92,
                'quality_gate_compliance': True
            },
            'stages': {},
            'discovered_event_types': ['Expansion', 'Consolidation', 'Retracement'],  # Only 3 types instead of 6
            'discovered_edge_intents': ['TEMPORAL_NEXT', 'MOVEMENT_TRANSITION'],      # Only 2 types instead of 4
            'node_feature_dimensions': 45,  # Wrong dimension
            'edge_feature_dimensions': 20,  # Correct dimension
            'errors': [],
            'total_operations': 100
        }
        
        result = validator.validate_pipeline_run(pipeline_results)
        
        assert result.passed is False
        assert result.production_ready is False
        
        invariant_violations = [v for v in result.violations if v.contract_type == ContractType.INVARIANT]
        assert len(invariant_violations) >= 3  # Should detect event types, edge intents, and node dimensions
        
        # Check specific violations
        violation_names = [v.contract_name for v in invariant_violations]
        assert "event_types_count" in violation_names
        assert "edge_intent_types_count" in violation_names
        assert "node_feature_dimensions" in violation_names
    
    def test_session_performance_validation(self, validator):
        """Test single session performance validation."""
        # Test successful session
        good_session_metrics = {
            'processing_time': 2.5,  # Under 3s threshold
            'memory_usage_mb': 65.0  # Reasonable memory usage
        }
        
        result = validator.validate_session_performance(good_session_metrics)
        
        assert result.passed is True
        assert result.production_ready is True
        assert len(result.violations) == 0
        
        # Test failed session
        bad_session_metrics = {
            'processing_time': 4.5,  # Exceeds 3s threshold
            'memory_usage_mb': 90.0  # High memory usage
        }
        
        result = validator.validate_session_performance(bad_session_metrics)
        
        assert result.passed is False
        assert len(result.violations) > 0
        
        violation = result.violations[0]
        assert violation.contract_name == "session_processing_time"
        assert violation.severity == ContractSeverity.CRITICAL
    
    def test_compliance_summary_generation(self, validator):
        """Test compliance summary generation."""
        # Add some validation history
        for i in range(3):
            mock_result = ContractValidationResult(
                passed=i > 0,  # First fails, others pass
                total_contracts=10,
                compliance_score=0.8 if i == 0 else 1.0,
                production_ready=i > 0
            )
            validator.validation_history.append(mock_result)
        
        summary = validator.get_compliance_summary()
        
        assert "latest_validation" in summary
        assert "compliance_trend" in summary
        assert "contract_categories" in summary
        assert "production_readiness" in summary
        
        # Check latest validation
        latest = summary["latest_validation"]
        assert latest["passed"] is True
        assert latest["production_ready"] is True
        
        # Check trend analysis
        trend = summary["compliance_trend"]
        assert trend["direction"] in ["improving", "stable", "declining"]
        assert "recent_scores" in trend
    
    def test_validation_history_management(self, validator):
        """Test validation history management."""
        # Add many validation results
        for i in range(150):  # More than the 100 limit
            mock_result = ContractValidationResult(
                passed=True,
                total_contracts=10,
                compliance_score=1.0,
                production_ready=True
            )
            validator.validation_history.append(mock_result)
        
        # Should be trimmed to keep only recent results
        assert len(validator.validation_history) <= 100
    
    def test_error_handling_in_validation(self, validator):
        """Test error handling during contract validation."""
        # Test with malformed pipeline results
        malformed_results = {
            'overall_metrics': None,  # Invalid structure
            'quality_metrics': "invalid",  # Wrong type
        }
        
        # Should not crash, but may produce warnings/errors
        result = validator.validate_pipeline_run(malformed_results)
        
        # Should still return a result object
        assert isinstance(result, ContractValidationResult)
        assert isinstance(result.passed, bool)


@pytest.mark.integration
class TestContractIntegration:
    """Integration tests for contract validation system."""
    
    def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow."""
        config = IRONFORGEPerformanceConfig()
        validator = PerformanceContractValidator(config)
        
        # Simulate a complete pipeline run with mixed results
        pipeline_results = {
            'overall_metrics': {
                'total_processing_time': 175.0,  # Good timing
                'peak_memory_mb': 95.0,          # Good memory
                'time_compliance': True,
                'memory_compliance': True
            },
            'quality_metrics': {
                'average_authenticity': 0.89,    # Good quality
                'quality_gate_compliance': True
            },
            'stages': {
                'discovery': {'processing_time': 58.0},
                'confluence': {'processing_time': 28.0},
                'validation': {'processing_time': 14.0},
                'reporting': {'processing_time': 72.0}
            },
            'discovered_event_types': ['Expansion', 'Consolidation', 'Retracement', 'Reversal', 'LiquidityTaken', 'Redelivery'],
            'discovered_edge_intents': ['TEMPORAL_NEXT', 'MOVEMENT_TRANSITION', 'LIQ_LINK', 'CONTEXT'],
            'node_feature_dimensions': 51,
            'edge_feature_dimensions': 20,
            'errors': [],
            'total_operations': 100
        }
        
        # Validate
        result = validator.validate_pipeline_run(pipeline_results)
        
        # Should pass all contracts
        assert result.passed is True
        assert result.production_ready is True
        assert result.compliance_score >= 0.95
        
        # Generate compliance summary
        summary = validator.get_compliance_summary()
        assert summary["latest_validation"]["passed"] is True
        
        # Validate the validation was recorded in history
        assert len(validator.validation_history) == 1
        assert validator.validation_history[0].passed is True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])