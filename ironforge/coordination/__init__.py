"""
IRONFORGE Coordination Module  
BMAD Multi-Agent Coordination via Temporal Workflows

This module implements BMAD (Behavioral Multi-Agent Decision) coordination
patterns for systematic analysis and consensus-driven research execution.
"""

from .bmad_workflows import (
    BMadCoordinationWorkflow,
    AgentConsensusInput,
    CoordinationResults,
    PreStructureAnalysisActivity,
    TargetTrackingActivity,
    StatisticalPredictionActivity
)

__all__ = [
    "BMadCoordinationWorkflow",
    "AgentConsensusInput",
    "CoordinationResults", 
    "PreStructureAnalysisActivity",
    "TargetTrackingActivity",
    "StatisticalPredictionActivity"
]