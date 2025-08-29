"""
Integration tests for PlanningBackedAgent system and document-mediated communication.

These tests validate that all IRONFORGE agents properly implement the
document-mediated communication protocol using planning documents.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import os

# Import all agents that should be PlanningBackedAgent instances
from agents.session_boundary_guardian.agent import SessionBoundaryGuardian
from agents.authenticity_validator.agent import AuthenticityValidator
from agents.tgat_attention_analyzer.agent import TGATAttentionAnalyzer
from agents.archaeological_zone_detector.agent import ArchaeologicalZoneDetector
from agents.pattern_intelligence_analyst.agent import PatternIntelligenceAnalyst
from agents.htf_cascade_predictor.agent import HTFCascadePredictor
from agents.confluence_intelligence.agent import ConfluenceIntelligence
from agents.pipeline_performance_monitor.agent import PipelinePerformanceMonitor
from agents.motif_pattern_miner.agent import MotifPatternMiner
from agents.minidash_enhancer.agent import MinidashEnhancer
from agents.pipeline_orchestrator.agent import PipelineOrchestratorAgent
from agents.contract_compliance_enforcer.agent import ContractComplianceEnforcer

from agents.base.planning_backed_agent import PlanningBackedAgent


class TestPlanningBackedAgentSystem:
    """Test suite for the complete PlanningBackedAgent ecosystem."""

    @pytest.fixture
    def agent_classes(self):
        """Return all agent classes that should inherit from PlanningBackedAgent."""
        return [
            SessionBoundaryGuardian,
            AuthenticityValidator, 
            TGATAttentionAnalyzer,
            ArchaeologicalZoneDetector,
            PatternIntelligenceAnalyst,
            HTFCascadePredictor,
            ConfluenceIntelligence,
            PipelinePerformanceMonitor,
            MotifPatternMiner,
            MinidashEnhancer,
            PipelineOrchestratorAgent,
            ContractComplianceEnforcer
        ]

    @pytest.fixture
    def sample_validation_data(self):
        """Sample data for testing agent execution."""
        return {
            "patterns": [
                {"id": 1, "type": "expansion", "authenticity": {"score": 92.5, "passed": True}},
                {"id": 2, "type": "consolidation", "authenticity": {"score": 88.1, "passed": True}},
                {"id": 3, "type": "retracement", "authenticity": {"score": 85.2, "passed": False}}
            ],
            "graph_data": {"nodes": 150, "edges": 230, "sessions": ["session_1"]},
            "tgat_output": {"attention_weights": {"layer_1": [0.8, 0.2]}, "patterns": []},
            "pipeline_stage": "discovery",
            "analysis_data": {"session_count": 1, "violations": []}
        }

    def test_all_agents_inherit_planning_backed_agent(self, agent_classes):
        """Verify all agents inherit from PlanningBackedAgent."""
        for agent_class in agent_classes:
            assert issubclass(agent_class, PlanningBackedAgent), (
                f"{agent_class.__name__} must inherit from PlanningBackedAgent"
            )

    def test_all_agents_have_execute_primary_function(self, agent_classes):
        """Verify all agents implement execute_primary_function method."""
        for agent_class in agent_classes:
            # Create instance to check method exists
            agent = agent_class()
            assert hasattr(agent, 'execute_primary_function'), (
                f"{agent_class.__name__} must implement execute_primary_function"
            )
            assert callable(getattr(agent, 'execute_primary_function')), (
                f"{agent_class.__name__}.execute_primary_function must be callable"
            )

    def test_all_agents_have_planning_context_methods(self, agent_classes):
        """Verify all agents have planning context accessor methods."""
        required_methods = [
            'get_requirements_from_planning',
            'get_behavior_from_planning', 
            'get_tools_from_planning',
            'get_dependencies_from_planning'
        ]
        
        for agent_class in agent_classes:
            agent = agent_class()
            for method_name in required_methods:
                assert hasattr(agent, method_name), (
                    f"{agent_class.__name__} must have {method_name} method"
                )
                assert callable(getattr(agent, method_name)), (
                    f"{agent_class.__name__}.{method_name} must be callable"
                )

    @pytest.mark.asyncio
    async def test_session_boundary_guardian_execution(self, sample_validation_data):
        """Test SessionBoundaryGuardian execute_primary_function."""
        agent = SessionBoundaryGuardian()
        
        result = await agent.execute_primary_function(sample_validation_data)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "message" in result
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)

    @pytest.mark.asyncio
    async def test_authenticity_validator_execution(self, sample_validation_data):
        """Test AuthenticityValidator execute_primary_function."""
        agent = AuthenticityValidator()
        
        result = await agent.execute_primary_function(sample_validation_data)
        
        assert isinstance(result, dict)
        assert "validation_completed" in result
        assert "patterns_validated" in result
        assert "authenticity_report" in result
        assert "threshold_status" in result
        assert isinstance(result.get("recommendations", []), list)

    @pytest.mark.asyncio
    async def test_tgat_attention_analyzer_execution(self, sample_validation_data):
        """Test TGATAttentionAnalyzer execute_primary_function."""
        agent = TGATAttentionAnalyzer()
        
        result = await agent.execute_primary_function(sample_validation_data)
        
        assert isinstance(result, dict)
        assert "analysis_completed" in result
        assert "attention_analysis" in result
        assert "pattern_authenticity" in result
        assert "authenticity_summary" in result

    @pytest.mark.asyncio
    async def test_planning_context_loading(self):
        """Test that agents can load planning context from existing planning documents."""
        # Test with agents that have existing planning documents
        agents_with_planning = [
            ("pipeline_orchestrator", PipelineOrchestratorAgent),
            ("contract_compliance_enforcer", ContractComplianceEnforcer)
        ]
        
        for agent_name, agent_class in agents_with_planning:
            agent = agent_class()
            
            # Test context loading methods
            behavior = await agent.get_behavior_from_planning()
            dependencies = await agent.get_dependencies_from_planning()
            tools = await agent.get_tools_from_planning()
            
            # Should not be empty strings/lists for agents with planning docs
            assert isinstance(behavior, str)
            assert isinstance(dependencies, dict)
            assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_agent_configuration_from_planning(self):
        """Test that agents extract configuration from planning documents."""
        agent = SessionBoundaryGuardian()
        
        # Get dependencies which should include configuration
        dependencies = await agent.get_dependencies_from_planning()
        
        # Dependencies should be a dictionary
        assert isinstance(dependencies, dict)
        
        # Agent should be able to use this for configuration
        # (This tests the integration even if planning docs don't exist yet)
        result = await agent.execute_primary_function({"graph_data": {"test": True}})
        assert "status" in result

    def test_agent_factory_method(self, agent_classes):
        """Test from_planning_documents factory method works for all agents."""
        for agent_class in agent_classes:
            try:
                agent = agent_class.from_planning_documents()
                assert isinstance(agent, agent_class)
                assert isinstance(agent, PlanningBackedAgent)
            except Exception as e:
                pytest.fail(f"{agent_class.__name__}.from_planning_documents() failed: {e}")

    @pytest.mark.asyncio
    async def test_error_handling_in_execute_primary_function(self):
        """Test that agents handle errors gracefully in execute_primary_function."""
        agent = AuthenticityValidator()
        
        # Test with invalid data
        invalid_data = {"invalid": "data"}
        result = await agent.execute_primary_function(invalid_data)
        
        assert isinstance(result, dict)
        assert "status" in result
        # Should handle error gracefully without crashing
        assert result.get("status") in ["ERROR", "PASSED"]

    def test_agent_naming_convention(self, agent_classes):
        """Test that agents follow proper naming conventions."""
        for agent_class in agent_classes:
            agent = agent_class()
            
            # Should have agent_name attribute
            assert hasattr(agent, 'agent_name')
            assert isinstance(agent.agent_name, str)
            assert len(agent.agent_name) > 0
            
            # Agent name should be snake_case
            assert '_' in agent.agent_name or agent.agent_name.islower()

    @pytest.mark.asyncio 
    async def test_multi_agent_coordination_pattern(self, sample_validation_data):
        """Test that multiple agents can work together using planning context."""
        # Create multiple agents
        boundary_guardian = SessionBoundaryGuardian()
        authenticity_validator = AuthenticityValidator()
        
        # Test sequential execution pattern
        boundary_result = await boundary_guardian.execute_primary_function(sample_validation_data)
        
        # Use boundary result in authenticity validation
        auth_data = {**sample_validation_data, "boundary_validation": boundary_result}
        auth_result = await authenticity_validator.execute_primary_function(auth_data)
        
        assert isinstance(boundary_result, dict)
        assert isinstance(auth_result, dict)
        assert "status" in boundary_result
        assert "status" in auth_result

    def test_planning_context_initialization(self, agent_classes):
        """Test that agents initialize planning context properly."""
        for agent_class in agent_classes:
            agent = agent_class()
            
            # Should have planning_context attribute (may be None if no docs exist)
            assert hasattr(agent, 'planning_context')
            
            # Should have agent_name
            assert hasattr(agent, 'agent_name')
            assert agent.agent_name is not None


class TestDocumentMediatedCommunication:
    """Test document-mediated communication patterns."""

    @pytest.mark.asyncio
    async def test_planning_document_workflow(self):
        """Test the complete planning document workflow."""
        # This simulates the workflow described in the requirements:
        # Phase 0 → Phase 1 (Planner) → Phase 2A-C (Parallel) → Phase 3 (Implementation)
        
        agent = SessionBoundaryGuardian()
        
        # Test that agent can load its planning context
        requirements = await agent.get_requirements_from_planning()
        behavior = await agent.get_behavior_from_planning()
        tools = await agent.get_tools_from_planning()
        dependencies = await agent.get_dependencies_from_planning()
        
        # All should return appropriate types
        assert isinstance(requirements, str)
        assert isinstance(behavior, str)
        assert isinstance(tools, list)
        assert isinstance(dependencies, dict)

    def test_ironforge_integration_compliance(self):
        """Test that the agent system complies with IRONFORGE requirements."""
        # Test key IRONFORGE agents exist
        key_agents = [
            SessionBoundaryGuardian,
            AuthenticityValidator,
            TGATAttentionAnalyzer,
            ArchaeologicalZoneDetector
        ]
        
        for agent_class in key_agents:
            agent = agent_class()
            assert isinstance(agent, PlanningBackedAgent)
            
            # Should have proper IRONFORGE naming
            assert hasattr(agent, 'agent_name')
            
            # Should be able to execute primary function
            assert hasattr(agent, 'execute_primary_function')


# Performance and Integration Tests
class TestSystemPerformance:
    """Test system performance requirements."""

    @pytest.mark.asyncio
    async def test_agent_execution_performance(self, sample_validation_data):
        """Test that agents meet performance requirements."""
        import time
        
        agent = AuthenticityValidator()
        
        start_time = time.time()
        result = await agent.execute_primary_function(sample_validation_data)
        execution_time = time.time() - start_time
        
        # Should complete quickly (under 1 second for test data)
        assert execution_time < 1.0, f"Agent execution took {execution_time}s, should be < 1s"
        assert isinstance(result, dict)

    def test_memory_usage_reasonable(self):
        """Test that agents don't consume excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple agents
        agents = [
            SessionBoundaryGuardian(),
            AuthenticityValidator(),
            TGATAttentionAnalyzer()
        ]
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not increase memory by more than 50MB for test agents
        assert memory_increase < 50, f"Memory increased by {memory_increase}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])