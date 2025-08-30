from __future__ import annotations
from typing import Any, Dict


class PerformanceContractChecker:
    def __init__(self, requirements: Dict[str, Any]) -> None:
        self.req = requirements

    def check(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if "session_processing" in metrics:
            results["session_processing"] = metrics["session_processing"] <= float(
                self.req["session_processing"]
            )
        if "full_discovery" in metrics:
            results["full_discovery"] = metrics["full_discovery"] <= float(
                self.req["full_discovery"]
            )
        if "memory_usage" in metrics:
            results["memory_usage"] = metrics["memory_usage"] <= float(
                self.req["memory_usage"]
            )
        return results
