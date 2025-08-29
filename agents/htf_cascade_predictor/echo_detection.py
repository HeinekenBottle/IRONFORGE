from __future__ import annotations
from typing import Any, Dict


class EchoDetector:
    def detect_forward_propagating_patterns(self, temporal_data: Dict[str, Any]) -> Dict[str, Any]:
        echo_count = len(temporal_data.get("echoes", []))
        return {"echo_count": echo_count, "temporal_non_locality": echo_count > 0}
