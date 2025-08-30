from __future__ import annotations

from agents.pipeline_orchestrator.agent import PipelineOrchestrator


def test_pipeline_orchestration_flow() -> None:
    p = PipelineOrchestrator()
    result = p.coordinate_pipeline_stages({})
    assert result.get("report", {}).get("path")
