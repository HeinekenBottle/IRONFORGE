"""
Minidash Enhancer Agent for IRONFORGE
====================================

Purpose: Create enhanced dashboards with archaeological visualizations.
"""
from __future__ import annotations

from typing import Any, Dict

from .dashboard_generator import DashboardGenerator
from .visualizations import ArchaeologicalVisualizer
from .export_tools import ExportTools


class MinidashEnhancer:
    def __init__(self) -> None:
        self.generator = DashboardGenerator()
        self.visualizer = ArchaeologicalVisualizer()
        self.exporter = ExportTools()

    def generate_enhanced_dashboard(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        viz = self.create_archaeological_visualizations(run_data.get("zones", []))
        html = self.generator.create_html_dashboard(viz)
        return {"html": html, "visualizations": viz}

    def create_archaeological_visualizations(self, zone_data: Any) -> Dict[str, Any]:
        return {
            "heatmaps": self.visualizer.create_zone_heatmaps(zone_data),
            "temporal": self.visualizer.generate_temporal_pattern_charts(zone_data),
            "attention": self.visualizer.build_attention_weight_visualizations(zone_data),
        }

    def export_interactive_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        return self.exporter.export_png_dashboard(dashboard_config.get("html", ""))
