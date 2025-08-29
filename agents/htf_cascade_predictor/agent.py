"""
HTF Cascade Predictor Agent for IRONFORGE
========================================

Purpose: HTF temporal cascade analysis with sub-second precision.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .cascade_analyzer import TemporalCascadeAnalyzer
from .echo_detection import EchoDetector
from .temporal_tools import TemporalTools


class HTFCascadePredictor:
    def __init__(self) -> None:
        self.analyzer = TemporalCascadeAnalyzer()
        self.echo = EchoDetector()
        self.tools = TemporalTools()

    def analyze_htf_cascades(self, multi_timeframe_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.analyzer.process_f45_f50_features(multi_timeframe_data)

    def predict_cascade_timing(self, htf_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.tools.predict_timing(htf_patterns)

    def detect_temporal_echoes(self, cascade_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.echo.detect_forward_propagating_patterns(cascade_data)
