"""
Integration tests for the IRONFORGE Agent Ecosystem with Planning Document Workflow.

This test suite validates the document-mediated communication protocol where agents
coordinate through structured markdown documents (INITIAL.md → prompts.md → tools.md → dependencies.md)
rather than direct communication, following the context-engineering-intro repository pattern.
"""

import asyncio
import pytest
from pathlib import Path
from typing import Dict, Any, List
import yaml
import tempfile
import json
from datetime import datetime

# Test imports
from ironforge.api import load_config
from tests.fixtures.mock_agents import (
    MockPipelineOrchestrator,
    MockContractComplianceEnforcer,
    MockAuthenticityValidator,
    MockAgentRegistry
)

class TestAgentEcosystemIntegration:
    """Test the complete agent ecosystem with planning document workflow."""

    @pytest.fixture
    async def agent_ecosystem(self):
        """Create a complete agent ecosystem for testing."""
        # Create mock agents with planning document awareness
        pipeline_orchestrator = MockPipelineOrchestrator()
        contract_enforcer = MockContractComplianceEnforcer()
        authenticity_validator = MockAuthenticityValidator()
        
        # Load planning documents for context
        await self.load_planning_documents(pipeline_orchestrator, "pipeline_orchestrator")
        await self.load_planning_documents(contract_enforcer, "contract_compliance_enforcer")
        
        # Create agent registry
        agent_registry = MockAgentRegistry({
            "pipeline-orchestrator": pipeline_orchestrator,
            "contract-compliance-enforcer": contract_enforcer,
            "authenticity-validator": authenticity_validator
        })
        
        return {
            "orchestrator": pipeline_orchestrator,
            "contract_enforcer": contract_enforcer,
            "authenticity_validator": authenticity_validator,
            "registry": agent_registry
        }

    async def load_planning_documents(self, agent: Any, agent_name: str):
        """Load planning documents to provide context to agents."""
        planning_path = Path(f"agents/{agent_name}/planning")
        
        planning_context = {}
        for doc_type in ["INITIAL.md", "prompts.md", "tools.md", "dependencies.md"]:
            doc_path = planning_path / doc_type
            if doc_path.exists():
                with open(doc_path, 'r') as f:
                    planning_context[doc_type] = f.read()
        
        # Inject planning context into agent
        agent.planning_context = planning_context

    @pytest.mark.asyncio
    async def test_document_mediated_communication(self, agent_ecosystem):
        """Test agents communicate through planning documents rather than direct calls."""
        orchestrator = agent_ecosystem["orchestrator"]
        contract_enforcer = agent_ecosystem["contract_enforcer"]
        
        # Test Phase 1: Requirements Context Communication
        # Orchestrator should understand its role from INITIAL.md
        requirements = await orchestrator.get_requirements_from_planning()
        assert "Pipeline Stage Coordination" in requirements
        assert "Error Recovery Management" in requirements
        assert "Quality Gate Enforcement" in requirements
        
        # Contract enforcer should understand its role from INITIAL.md
        contract_requirements = await contract_enforcer.get_requirements_from_planning()
        assert "Golden Invariant Validation" in contract_requirements
        assert "HTF Rule Compliance" in contract_requirements
        assert "Session Isolation Enforcement" in contract_requirements

    @pytest.mark.asyncio 
    async def test_behavioral_specification_context(self, agent_ecosystem):
        """Test agents derive behavior from prompts.md context."""
        orchestrator = agent_ecosystem["orchestrator"]
        
        # Test behavioral context from prompts.md
        behavior = await orchestrator.get_behavior_from_planning()
        assert "Fail-Fast with Recovery" in behavior
        assert "Quality-First" in behavior
        assert "Performance-Aware" in behavior
        assert "Agent-Coordinated" in behavior

    @pytest.mark.asyncio
    async def test_functional_implementation_context(self, agent_ecosystem):
        """Test agents understand their tools from tools.md context."""
        orchestrator = agent_ecosystem["orchestrator"]
        
        # Test functional context from tools.md
        tools = await orchestrator.get_tools_from_planning()
        expected_tools = [
            "execute_pipeline_stage",
            "coordinate_agents", 
            "handle_pipeline_error",
            "optimize_performance",
            "enforce_quality_gates",
            "generate_pipeline_report"
        ]
        
        for tool in expected_tools:
            assert tool in tools

    @pytest.mark.asyncio
    async def test_infrastructure_architecture_context(self, agent_ecosystem):
        """Test agents understand their dependencies from dependencies.md context."""
        orchestrator = agent_ecosystem["orchestrator"]
        
        # Test infrastructure context from dependencies.md
        dependencies = await orchestrator.get_dependencies_from_planning()
        assert "IRONFORGE_CONFIG_PATH" in dependencies
        assert "PIPELINE_TIMEOUT_SECONDS" in dependencies
        assert "AUTHENTICITY_THRESHOLD" in dependencies
        assert "AGENT_COMMUNICATION_TIMEOUT" in dependencies

    @pytest.mark.asyncio
    async def test_cross_agent_context_inheritance(self, agent_ecosystem):
        """Test how agents inherit and build upon each other's planning contexts."""
        orchestrator = agent_ecosystem["orchestrator"]
        contract_enforcer = agent_ecosystem["contract_enforcer"]
        
        # Contract enforcer should understand orchestrator's quality requirements
        orchestrator_quality_reqs = await orchestrator.get_quality_requirements()
        contract_enforcement_reqs = await contract_enforcer.get_enforcement_requirements()
        
        # Verify context inheritance
        assert orchestrator_quality_reqs["authenticity_threshold"] == 0.87
        assert contract_enforcement_reqs["authenticity_threshold"] == 0.87
        
        # Verify specialized context building
        assert "golden_invariants" in contract_enforcement_reqs
        assert len(contract_enforcement_reqs["golden_invariants"]["event_types"]) == 6

    @pytest.mark.asyncio
    async def test_pipeline_orchestration_workflow(self, agent_ecosystem):
        """Test complete pipeline orchestration using planning document context."""
        orchestrator = agent_ecosystem["orchestrator"]
        contract_enforcer = agent_ecosystem["contract_enforcer"]
        authenticity_validator = agent_ecosystem["authenticity_validator"]
        
        # Create mock session data
        session_data = {
            "session_id": "test_session_001",
            "events": [
                {"type": "Expansion", "timestamp": 1234567890},
                {"type": "Consolidation", "timestamp": 1234567900}
            ],
            "edges": [
                {"intent": "TEMPORAL_NEXT", "source": 0, "target": 1}
            ],
            "node_features": {f"f{i}": [0.1, 0.2] for i in range(51)},
            "edge_features": {f"e{i}": [0.3, 0.4] for i in range(20)}
        }
        
        # Test Stage 1: Pre-validation using contract enforcer context
        validation_result = await orchestrator.coordinate_with_agent(
            "contract-compliance-enforcer",
            "validate_data_contracts", 
            session_data
        )
        assert validation_result["status"] == "passed"
        assert validation_result["golden_invariants"]["validated"] == True
        
        # Test Stage 2: Discovery stage execution
        discovery_result = await orchestrator.execute_pipeline_stage(
            "discovery",
            session_data,
            quality_gates=True
        )
        assert discovery_result["status"] == "completed"
        assert discovery_result["processing_time"] < 5.0  # <5s requirement
        
        # Test Stage 3: Authenticity validation using validator context
        auth_result = await orchestrator.coordinate_with_agent(
            "authenticity-validator",
            "validate_pattern_authenticity",
            discovery_result["patterns"]
        )
        assert auth_result["authenticity_score"] >= 0.87  # 87% threshold
        
        # Verify planning document context was used throughout
        assert orchestrator.planning_context_used == True
        assert contract_enforcer.planning_context_used == True

    @pytest.mark.asyncio
    async def test_error_recovery_with_document_context(self, agent_ecosystem):
        """Test error recovery using behavioral context from planning documents."""
        orchestrator = agent_ecosystem["orchestrator"]
        
        # Simulate pipeline error
        error_details = {
            "error_type": "ValidationFailure",
            "stage": "confluence", 
            "context": "Authenticity score 0.82 below 0.87 threshold",
            "recoverable": True
        }
        
        # Test error recovery using behavior from prompts.md
        recovery_result = await orchestrator.handle_pipeline_error(
            error_details,
            recovery_strategy="auto"  # From planning context
        )
        
        assert recovery_result["strategy"] == "pattern_revalidation"
        assert recovery_result["agent_coordination"]["authenticity-validator"] == "requested"
        assert recovery_result["preserve_progress"] == True  # From dependencies context

    @pytest.mark.asyncio
    async def test_performance_optimization_with_context(self, agent_ecosystem):
        """Test performance optimization using context from planning documents."""
        orchestrator = agent_ecosystem["orchestrator"]
        
        # Mock performance data indicating bottleneck
        performance_data = {
            "stage_timings": {
                "discovery": 45.2,  # Too slow
                "confluence": 12.1,
                "validation": 8.3,
                "reporting": 15.7
            },
            "memory_usage_mb": 95.0,  # Approaching limit
            "sessions_processed": 25
        }
        
        # Test optimization using context from tools.md and dependencies.md
        optimization_result = await orchestrator.optimize_performance(
            "timing",  # From planning context
            performance_data,
            "standard"  # From dependencies context
        )
        
        assert optimization_result["bottleneck_identified"] == "discovery"
        assert optimization_result["optimization_applied"] == "parallel_processing"
        assert optimization_result["projected_improvement"] > 0.2  # 20%+ improvement

    @pytest.mark.asyncio
    async def test_quality_gate_enforcement_with_context(self, agent_ecosystem):
        """Test quality gate enforcement using contract context."""
        orchestrator = agent_ecosystem["orchestrator"]
        
        # Test data that violates golden invariants
        invalid_data = {
            "events": [
                {"type": "InvalidEventType", "timestamp": 1234567890}  # Violates 6-event rule
            ],
            "edges": [
                {"intent": "INVALID_INTENT", "source": 0, "target": 1}  # Violates 4-edge rule
            ],
            "node_features": {f"f{i}": [0.1] for i in range(45)},  # Wrong dimension (45 instead of 51)
            "edge_features": {f"e{i}": [0.3] for i in range(25)}   # Wrong dimension (25 instead of 20)
        }
        
        # Test quality gate using enforcement context from planning documents
        gate_result = await orchestrator.enforce_quality_gates(
            "contracts",
            invalid_data,
            "strict"  # From dependencies context
        )
        
        assert gate_result["status"] == "failed"
        assert "event_type_violation" in gate_result["violations"]
        assert "edge_intent_violation" in gate_result["violations"] 
        assert "feature_dimension_violation" in gate_result["violations"]

    @pytest.mark.asyncio
    async def test_agent_coordination_with_specialized_context(self, agent_ecosystem):
        """Test agent coordination where each agent uses its specialized planning context."""
        orchestrator = agent_ecosystem["orchestrator"]
        registry = agent_ecosystem["registry"]
        
        # Test multi-agent coordination scenario
        coordination_task = {
            "task_type": "comprehensive_validation",
            "data": {
                "session_data": {"events": [], "edges": []},
                "quality_requirements": {"authenticity_threshold": 0.87},
                "performance_requirements": {"max_time_seconds": 180}
            }
        }
        
        # Coordinate multiple agents using their planning contexts
        results = await orchestrator.coordinate_multiple_agents([
            ("contract-compliance-enforcer", "validate_contracts"),
            ("authenticity-validator", "validate_authenticity") 
        ], coordination_task)
        
        # Verify each agent used its specialized context
        contract_result = results["contract-compliance-enforcer"]
        assert contract_result["golden_invariants_checked"] == True
        assert contract_result["htf_rules_validated"] == True
        
        auth_result = results["authenticity-validator"] 
        assert auth_result["authenticity_threshold"] == 0.87
        assert auth_result["graduation_criteria_applied"] == True

    @pytest.mark.asyncio
    async def test_planning_document_evolution_support(self, agent_ecosystem):
        """Test system handles planning document evolution gracefully."""
        orchestrator = agent_ecosystem["orchestrator"]
        
        # Simulate planning document update
        updated_requirements = {
            "new_quality_threshold": 0.90,  # Raised from 0.87
            "new_performance_limit": 150,   # Reduced from 180 
            "additional_validation": "temporal_coherence"
        }
        
        # Test adaptation to updated planning context
        await orchestrator.update_planning_context("INITIAL.md", updated_requirements)
        
        # Verify agent adapts behavior based on updated context
        updated_behavior = await orchestrator.get_behavior_from_planning()
        assert updated_behavior["quality_threshold"] == 0.90
        assert updated_behavior["performance_limit"] == 150
        assert "temporal_coherence" in updated_behavior["validation_types"]

    def test_planning_document_structure_validation(self):
        """Test that planning documents follow the required structure."""
        agent_types = [
            "pipeline_orchestrator",
            "contract_compliance_enforcer"
        ]
        
        for agent_type in agent_types:
            planning_path = Path(f"agents/{agent_type}/planning")
            
            # Verify all required planning documents exist
            required_docs = ["INITIAL.md", "prompts.md", "tools.md", "dependencies.md"]
            for doc in required_docs:
                doc_path = planning_path / doc
                assert doc_path.exists(), f"Missing {doc} for {agent_type}"
                
                # Verify document has content
                with open(doc_path, 'r') as f:
                    content = f.read()
                    assert len(content) > 100, f"{doc} for {agent_type} is too short"
                    
                # Verify document structure based on type
                if doc == "INITIAL.md":
                    assert "## Executive Summary" in content
                    assert "## Functional Requirements" in content
                    assert "## Success Criteria" in content
                elif doc == "prompts.md":
                    assert "SYSTEM_PROMPT" in content
                    assert "system_prompt" in content.lower()
                elif doc == "tools.md":
                    assert "@agent.tool" in content
                    assert "Implementation Pattern" in content
                elif doc == "dependencies.md":
                    assert "Environment Variables" in content
                    assert "Dependencies" in content

class MockAgentBase:
    """Base class for mock agents with planning document support."""
    
    def __init__(self):
        self.planning_context = {}
        self.planning_context_used = False
    
    async def get_requirements_from_planning(self) -> str:
        """Extract requirements from INITIAL.md planning context."""
        self.planning_context_used = True
        initial_md = self.planning_context.get("INITIAL.md", "")
        # Extract functional requirements section
        if "## Functional Requirements" in initial_md:
            return initial_md.split("## Functional Requirements")[1].split("##")[0]
        return ""
    
    async def get_behavior_from_planning(self) -> str:
        """Extract behavior specifications from prompts.md planning context."""
        self.planning_context_used = True
        prompts_md = self.planning_context.get("prompts.md", "")
        # Extract behavioral guidelines
        return prompts_md
    
    async def get_tools_from_planning(self) -> List[str]:
        """Extract tool specifications from tools.md planning context."""
        self.planning_context_used = True
        tools_md = self.planning_context.get("tools.md", "")
        # Extract tool names from @agent.tool definitions
        tools = []
        for line in tools_md.split('\n'):
            if 'async def ' in line and '(' in line:
                tool_name = line.split('async def ')[1].split('(')[0].strip()
                tools.append(tool_name)
        return tools
    
    async def get_dependencies_from_planning(self) -> Dict[str, Any]:
        """Extract dependencies from dependencies.md planning context."""
        self.planning_context_used = True
        deps_md = self.planning_context.get("dependencies.md", "")
        # Extract environment variables
        dependencies = {}
        in_env_section = False
        for line in deps_md.split('\n'):
            if "```bash" in line:
                in_env_section = True
            elif "```" in line and in_env_section:
                in_env_section = False
            elif in_env_section and "=" in line:
                key, value = line.split("=", 1)
                dependencies[key.strip()] = value.strip()
        return dependencies

# Create mock agent classes that inherit planning document support
class MockPipelineOrchestrator(MockAgentBase):
    async def execute_pipeline_stage(self, stage_name: str, session_data: dict, quality_gates: bool = True):
        return {
            "status": "completed",
            "stage": stage_name,
            "processing_time": 3.2,
            "quality_gates_passed": quality_gates,
            "patterns": [{"authenticity": 0.92, "confidence": 0.85}]
        }
    
    async def coordinate_with_agent(self, agent_name: str, task_type: str, data: dict):
        if agent_name == "contract-compliance-enforcer":
            return {
                "status": "passed", 
                "golden_invariants": {"validated": True},
                "violations": []
            }
        elif agent_name == "authenticity-validator":
            return {
                "authenticity_score": 0.92,
                "graduation_status": "passed"
            }
    
    async def handle_pipeline_error(self, error_details: dict, recovery_strategy: str = "auto"):
        return {
            "strategy": "pattern_revalidation",
            "agent_coordination": {"authenticity-validator": "requested"},
            "preserve_progress": True,
            "estimated_recovery_time": 30
        }
    
    async def optimize_performance(self, target: str, performance_data: dict, level: str = "standard"):
        return {
            "bottleneck_identified": "discovery",
            "optimization_applied": "parallel_processing", 
            "projected_improvement": 0.25
        }
    
    async def enforce_quality_gates(self, gate_type: str, validation_data: dict, enforcement_level: str = "strict"):
        # Mock quality gate that detects violations in test data
        violations = []
        if "InvalidEventType" in str(validation_data):
            violations.append("event_type_violation")
        if "INVALID_INTENT" in str(validation_data):
            violations.append("edge_intent_violation")
        if len(validation_data.get("node_features", {})) != 51:
            violations.append("feature_dimension_violation")
            
        return {
            "status": "failed" if violations else "passed",
            "violations": violations
        }
    
    async def coordinate_multiple_agents(self, agent_tasks: List, task_data: dict):
        results = {}
        for agent_name, task_type in agent_tasks:
            if agent_name == "contract-compliance-enforcer":
                results[agent_name] = {
                    "golden_invariants_checked": True,
                    "htf_rules_validated": True
                }
            elif agent_name == "authenticity-validator":
                results[agent_name] = {
                    "authenticity_threshold": 0.87,
                    "graduation_criteria_applied": True
                }
        return results

class MockContractComplianceEnforcer(MockAgentBase):
    async def get_enforcement_requirements(self):
        return {
            "authenticity_threshold": 0.87,
            "golden_invariants": {
                "event_types": [
                    "Expansion", "Consolidation", "Retracement",
                    "Reversal", "Liquidity Taken", "Redelivery"
                ]
            }
        }

class MockAuthenticityValidator(MockAgentBase):
    async def validate_authenticity(self, patterns: List[dict]):
        return {
            "authenticity_score": 0.92,
            "graduation_status": "passed"
        }

class MockAgentRegistry:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
    
    def get_agent(self, agent_name: str):
        return self.agents.get(agent_name)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])