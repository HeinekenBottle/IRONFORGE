from __future__ import annotations
from typing import Any, Dict


class DiscoveryStageManager:
    def run(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"patterns": [{"confidence": 0.9}]}


class ConfluenceStageManager:
    def run(self, discovered: Dict[str, Any]) -> Dict[str, Any]:
        return {**discovered, "scored": True}


class ValidationStageManager:
    def run(self, scored: Dict[str, Any]) -> Dict[str, Any]:
        return {**scored, "validated": True}


class ReportingStageManager:
    def run(self, validated: Dict[str, Any]) -> Dict[str, Any]:
        return {**validated, "report": {"path": "runs/latest/minidash.html"}}
