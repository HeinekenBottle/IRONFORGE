"""
Minidash Enhancer Agent for IRONFORGE
====================================

Purpose: Create enhanced dashboards with archaeological visualizations.
"""
from __future__ import annotations

from typing import Any, Dict

from ..base import PlanningBackedAgent
from .dashboard_generator import DashboardGenerator
from .visualizations import ArchaeologicalVisualizer
from .export_tools import ExportTools


class MinidashEnhancer(PlanningBackedAgent):
    def __init__(self, agent_name: str = "minidash_enhancer") -> None:
        super().__init__(agent_name=agent_name)
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

    async def execute_primary_function(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute primary minidash enhancement using planning context.
        
        Args:
            dashboard_data: Dict containing run data and visualization configuration
            
        Returns:
            Dict containing enhanced dashboard and export results
        """
        results = {
            "enhanced_dashboard": {},
            "archaeological_visualizations": {},
            "export_results": {},
            "dashboard_metrics": {},
            "recommendations": []
        }
        
        try:
            # Get behavior and dependencies from planning context
            behavior = await self.get_behavior_from_planning()
            dependencies = await self.get_dependencies_from_planning()
            
            # Extract configuration from planning context
            enable_archaeological_viz = dependencies.get("ENABLE_ARCHAEOLOGICAL_VISUALIZATIONS", "true").lower() == "true"
            enable_interactive_export = dependencies.get("ENABLE_INTERACTIVE_EXPORT", "true").lower() == "true"
            include_attention_weights = dependencies.get("INCLUDE_ATTENTION_WEIGHT_VIZ", "true").lower() == "true"
            dashboard_quality = dependencies.get("DASHBOARD_QUALITY", "high")
            
            # Extract dashboard data
            run_data = dashboard_data.get("run_data", {})
            visualization_config = dashboard_data.get("visualization_config", {})
            export_config = dashboard_data.get("export_config", {})
            
            if not run_data:
                results["status"] = "WARNING"
                results["message"] = "No run data provided for dashboard enhancement"
                results["recommendations"].append("Ensure IRONFORGE run data is available before dashboard generation")
                return results
            
            # Generate enhanced dashboard
            dashboard_result = self.generate_enhanced_dashboard(run_data)
            results["enhanced_dashboard"] = dashboard_result
            
            # Extract visualization components
            if "visualizations" in dashboard_result:
                results["archaeological_visualizations"] = dashboard_result["visualizations"]
                
                # Analyze visualization quality
                viz_metrics = {
                    "heatmap_count": len(dashboard_result["visualizations"].get("heatmaps", [])),
                    "temporal_charts": len(dashboard_result["visualizations"].get("temporal", [])),
                    "attention_viz_available": "attention" in dashboard_result["visualizations"]
                }
                results["dashboard_metrics"] = viz_metrics
            
            # Create archaeological visualizations if enabled
            if enable_archaeological_viz:
                zone_data = run_data.get("archaeological_zones", run_data.get("zones", []))
                if zone_data:
                    arch_visualizations = self.create_archaeological_visualizations(zone_data)
                    results["archaeological_visualizations"].update(arch_visualizations)
                else:
                    results["recommendations"].append("Archaeological visualization enabled but no zone data available")
            
            # Export interactive dashboard if enabled
            if enable_interactive_export and "html" in dashboard_result:
                export_config_full = {
                    "html": dashboard_result["html"],
                    "quality": dashboard_quality,
                    **export_config
                }
                
                export_path = self.export_interactive_dashboard(export_config_full)
                results["export_results"] = {
                    "export_path": export_path,
                    "export_successful": bool(export_path),
                    "export_format": "PNG"
                }
            
            # Generate recommendations based on behavior
            if behavior.get("PROVIDE_DASHBOARD_RECOMMENDATIONS", True):
                recommendations = []
                
                # Dashboard quality recommendations
                if "dashboard_metrics" in results:
                    metrics = results["dashboard_metrics"]
                    
                    if metrics.get("heatmap_count", 0) == 0:
                        recommendations.append("No heatmaps generated. Consider adding archaeological zone heatmap visualizations")
                    
                    if metrics.get("temporal_charts", 0) == 0:
                        recommendations.append("No temporal charts generated. Consider adding temporal pattern visualizations")
                    
                    if include_attention_weights and not metrics.get("attention_viz_available", False):
                        recommendations.append("Attention weight visualization requested but not available. Check TGAT attention data")
                
                # Archaeological visualization recommendations
                if enable_archaeological_viz:
                    arch_viz = results["archaeological_visualizations"]
                    if not arch_viz.get("heatmaps") and not arch_viz.get("temporal"):
                        recommendations.append("Archaeological visualizations enabled but none generated. Check archaeological zone data quality")
                
                # Export recommendations
                if enable_interactive_export:
                    export_results = results.get("export_results", {})
                    if not export_results.get("export_successful", False):
                        recommendations.append("Interactive dashboard export failed. Check export configuration and system resources")
                    elif dashboard_quality == "high" and export_results.get("export_format") != "PNG":
                        recommendations.append("High quality dashboard requested but PNG export may be limited. Consider vector format export")
                
                # Data completeness recommendations
                if "patterns" in run_data:
                    pattern_count = len(run_data["patterns"]) if isinstance(run_data["patterns"], list) else 0
                    if pattern_count == 0:
                        recommendations.append("No patterns in run data. Dashboard will have limited pattern visualizations")
                    elif pattern_count > 50:
                        recommendations.append(f"Large number of patterns ({pattern_count}). Consider pattern filtering for cleaner visualizations")
                
                results["recommendations"] = recommendations
            
            # Success metrics
            viz_count = len(results["archaeological_visualizations"])
            has_export = results.get("export_results", {}).get("export_successful", False)
            dashboard_generated = bool(results["enhanced_dashboard"].get("html"))
            
            results["status"] = "SUCCESS"
            results["message"] = f"Generated enhanced dashboard with {viz_count} visualizations" + (" and export" if has_export else "")
            
        except Exception as e:
            results["status"] = "ERROR"
            results["message"] = f"Minidash enhancement failed: {str(e)}"
            results["recommendations"].append("Check run data format and dashboard generator configuration")
        
        return results
