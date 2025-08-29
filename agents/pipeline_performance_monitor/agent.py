"""
IRONFORGE Pipeline Performance Monitor Agent

A production-ready performance monitoring agent that ensures IRONFORGE meets its
strict performance requirements across all pipeline stages:
- Discovery → Confluence → Validation → Reporting

Performance Contracts:
- Single Session Processing: <3 seconds
- Full Discovery (57 sessions): <180 seconds  
- Memory Footprint: <100MB total usage
- Authenticity Threshold: >87% for production patterns
- Initialization: <2 seconds with lazy loading
- Monitoring Overhead: Sub-millisecond impact

The agent provides real-time monitoring, bottleneck detection, optimization
recommendations, and automated performance contract validation.
"""

import asyncio
import logging
import time
import threading
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import psutil
import gc

# IRONFORGE imports
from ironforge.api import (
    run_discovery, 
    score_confluence, 
    validate_run, 
    build_minidash,
    get_ironforge_container,
    initialize_ironforge_lazy_loading
)
from ironforge.sdk.app_config import Config, load_config
from ironforge.utils.performance_monitor import PerformanceMonitor
from ironforge.monitoring.performance_tracker import get_performance_tracker

from ..base import PlanningBackedAgent
from .ironforge_config import IRONFORGEPerformanceConfig, StageThresholds
from .tools import PerformanceAnalysisTools, OptimizationRecommender
from .contracts import PerformanceContractValidator
from .performance import SelfPerformanceMonitor
from .dashboard import PerformanceDashboard

logger = logging.getLogger(__name__)


@dataclass
class PipelineStageMetrics:
    """Performance metrics for a specific pipeline stage."""
    
    stage_name: str
    processing_times: List[float] = field(default_factory=list)
    memory_snapshots: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    error_count: int = 0
    bottlenecks_detected: List[str] = field(default_factory=list)
    
    def add_processing_time(self, time_seconds: float, threshold: float):
        """Add processing time and check against threshold."""
        self.processing_times.append(time_seconds)
        if time_seconds > threshold:
            logger.warning(f"⚠️  {self.stage_name} exceeded threshold: {time_seconds:.2f}s > {threshold:.2f}s")
    
    def get_average_time(self) -> float:
        """Get average processing time for this stage."""
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
    
    def get_compliance_rate(self, threshold: float) -> float:
        """Get percentage of executions that met the time threshold."""
        if not self.processing_times:
            return 1.0
        compliant = len([t for t in self.processing_times if t <= threshold])
        return compliant / len(self.processing_times)


@dataclass
class PipelineHealthStatus:
    """Overall pipeline health assessment."""
    
    status: str  # GREEN, YELLOW, RED
    overall_score: float  # 0.0 to 1.0
    stage_scores: Dict[str, float] = field(default_factory=dict)
    active_issues: List[str] = field(default_factory=list)
    trending: str = "STABLE"  # IMPROVING, STABLE, DEGRADING
    last_assessment: datetime = field(default_factory=datetime.now)


class PipelinePerformanceMonitorAgent(PlanningBackedAgent):
    """
    Elite IRONFORGE Pipeline Performance Monitor Agent
    
    Provides comprehensive real-time performance monitoring across all
    pipeline stages with microsecond precision tracking, bottleneck detection,
    and automated optimization recommendations.
    """
    
    def __init__(self, config: Optional[IRONFORGEPerformanceConfig] = None, agent_name: str = "pipeline_performance_monitor"):
        super().__init__(agent_name=agent_name)
        self.config = config or IRONFORGEPerformanceConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Core monitoring components
        self.stage_metrics: Dict[str, PipelineStageMetrics] = {
            "discovery": PipelineStageMetrics("Discovery"),
            "confluence": PipelineStageMetrics("Confluence"), 
            "validation": PipelineStageMetrics("Validation"),
            "reporting": PipelineStageMetrics("Reporting")
        }
        
        # Integration with existing IRONFORGE monitoring
        self.ironforge_monitor = PerformanceMonitor()
        self.bmad_tracker = get_performance_tracker()
        
        # Analysis and optimization tools
        self.analysis_tools = PerformanceAnalysisTools(self.config)
        self.optimizer = OptimizationRecommender(self.config)
        self.contract_validator = PerformanceContractValidator(self.config)
        self.self_monitor = SelfPerformanceMonitor()
        
        # Real-time dashboard
        self.dashboard = PerformanceDashboard(self.config)
        
        # Pipeline health tracking
        self.health_history: List[PipelineHealthStatus] = []
        self.current_health = PipelineHealthStatus(status="GREEN", overall_score=1.0)
        
        # Alert system
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Container system monitoring
        self.container_initialized = False
        self.container_init_time: Optional[float] = None
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        self.logger.info("🎯 IRONFORGE Pipeline Performance Monitor Agent initialized")
        self.logger.info(f"   Performance contracts loaded: {len(self.config.stage_thresholds)} stages")
    
    async def initialize(self) -> bool:
        """
        Initialize the performance monitoring agent with sub-2-second target.
        
        Returns:
            bool: True if initialization completed within threshold
        """
        start_time = time.time()
        
        try:
            # Initialize container system with lazy loading
            with self.self_monitor.track_operation("container_initialization"):
                container = initialize_ironforge_lazy_loading()
                self.container_initialized = True
                self.container_init_time = time.time() - start_time
            
            # Start background monitoring
            self.start_monitoring()
            
            # Initialize dashboard
            await self.dashboard.initialize()
            
            # Validate initialization time
            init_time = time.time() - start_time
            threshold = self.config.stage_thresholds.initialization_seconds
            
            if init_time > threshold:
                self.logger.warning(f"⚠️  Initialization exceeded {threshold}s target: {init_time:.2f}s")
                return False
            
            self.logger.info(f"✅ Agent initialized in {init_time:.3f}s (target: <{threshold}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Agent initialization failed: {e}")
            return False
    
    def start_monitoring(self):
        """Start continuous background monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("📊 Background performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring and generate final report."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Generate final performance report
        self.log_comprehensive_report()
        self.logger.info("📊 Background performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop for system health."""
        while self.monitoring_active:
            try:
                # Update system health assessment
                self._assess_pipeline_health()
                
                # Check for performance regression
                self._detect_performance_regression()
                
                # Update dashboard
                asyncio.run(self.dashboard.update_metrics(self._get_dashboard_data()))
                
                time.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)  # Back off on errors
    
    @contextmanager
    def monitor_pipeline_stage(self, stage_name: str, session_count: int = 1):
        """
        Monitor a complete pipeline stage with comprehensive metrics.
        
        Args:
            stage_name: Name of pipeline stage (discovery, confluence, validation, reporting)
            session_count: Number of sessions being processed
        """
        if stage_name not in self.stage_metrics:
            raise ValueError(f"Unknown stage: {stage_name}. Valid: {list(self.stage_metrics.keys())}")
        
        stage_metrics = self.stage_metrics[stage_name]
        threshold = getattr(self.config.stage_thresholds, f"{stage_name.lower()}_seconds")
        
        start_time = time.perf_counter()  # Microsecond precision
        start_memory = self._get_current_memory()
        
        # Track with existing IRONFORGE monitors
        with self.ironforge_monitor.monitor_component(f"pipeline_{stage_name}"):
            with self.self_monitor.track_operation(f"stage_{stage_name}"):
                try:
                    yield stage_metrics
                    
                finally:
                    # Calculate metrics
                    processing_time = time.perf_counter() - start_time
                    end_memory = self._get_current_memory()
                    memory_delta = end_memory - start_memory
                    
                    # Record stage metrics
                    stage_metrics.add_processing_time(processing_time, threshold)
                    stage_metrics.memory_snapshots.append(end_memory)
                    
                    # Per-session performance calculation
                    per_session_time = processing_time / session_count if session_count > 0 else processing_time
                    
                    # Log performance with status indicator
                    status = "✅" if processing_time <= threshold else "⚠️ "
                    self.logger.info(
                        f"{status} {stage_name.title()} Stage: {processing_time:.3f}s "
                        f"({per_session_time:.3f}s/session) | Memory: {end_memory:.1f}MB"
                    )
                    
                    # Check for alerts
                    if processing_time > threshold:
                        self._trigger_alert("stage_timeout", {
                            "stage": stage_name,
                            "processing_time": processing_time,
                            "threshold": threshold,
                            "sessions": session_count
                        })
                    
                    # Memory alert
                    if end_memory > self.config.memory_limit_mb:
                        self._trigger_alert("memory_exceeded", {
                            "stage": stage_name,
                            "memory_mb": end_memory,
                            "limit_mb": self.config.memory_limit_mb
                        })
    
    async def monitor_full_pipeline_run(self, config_path: str) -> Dict[str, Any]:
        """
        Monitor a complete IRONFORGE pipeline run with comprehensive tracking.
        
        Args:
            config_path: Path to IRONFORGE configuration file
            
        Returns:
            Dict containing comprehensive performance analysis
        """
        start_time = time.perf_counter()
        config = load_config(config_path)
        
        self.logger.info("🚀 Starting monitored full pipeline run")
        self.logger.info(f"   Target: <{self.config.stage_thresholds.full_discovery_seconds}s total")
        
        pipeline_results = {
            "stages": {},
            "overall_metrics": {},
            "quality_metrics": {},
            "contract_compliance": {},
            "optimization_opportunities": []
        }
        
        try:
            # Stage 1: Discovery
            with self.monitor_pipeline_stage("discovery") as discovery_metrics:
                discovery_result = await self._run_discovery_stage(config)
                pipeline_results["stages"]["discovery"] = {
                    "success": discovery_result.get("success", False),
                    "patterns_discovered": discovery_result.get("pattern_count", 0),
                    "authenticity_scores": discovery_result.get("authenticity_scores", [])
                }
            
            # Stage 2: Confluence Scoring
            with self.monitor_pipeline_stage("confluence") as confluence_metrics:
                confluence_result = await self._run_confluence_stage(config)
                pipeline_results["stages"]["confluence"] = {
                    "success": confluence_result.get("success", False),
                    "patterns_scored": confluence_result.get("pattern_count", 0),
                    "average_score": confluence_result.get("average_score", 0.0)
                }
            
            # Stage 3: Validation
            with self.monitor_pipeline_stage("validation") as validation_metrics:
                validation_result = await self._run_validation_stage(config)
                pipeline_results["stages"]["validation"] = {
                    "success": validation_result.get("success", False),
                    "contracts_validated": validation_result.get("contracts_checked", 0),
                    "quality_gates_passed": validation_result.get("gates_passed", True)
                }
            
            # Stage 4: Reporting
            with self.monitor_pipeline_stage("reporting") as reporting_metrics:
                reporting_result = await self._run_reporting_stage(config)
                pipeline_results["stages"]["reporting"] = {
                    "success": reporting_result.get("success", False),
                    "dashboard_generated": reporting_result.get("dashboard_path") is not None,
                    "export_completed": reporting_result.get("export_completed", False)
                }
            
            # Calculate overall metrics
            total_time = time.perf_counter() - start_time
            pipeline_results["overall_metrics"] = {
                "total_processing_time": total_time,
                "target_time": self.config.stage_thresholds.full_discovery_seconds,
                "time_compliance": total_time <= self.config.stage_thresholds.full_discovery_seconds,
                "peak_memory_mb": max([max(m.memory_snapshots) if m.memory_snapshots else 0 
                                     for m in self.stage_metrics.values()]),
                "memory_compliance": all(max(m.memory_snapshots) <= self.config.memory_limit_mb 
                                       if m.memory_snapshots else True 
                                       for m in self.stage_metrics.values())
            }
            
            # Quality metrics analysis
            all_authenticity_scores = []
            for stage_result in pipeline_results["stages"].values():
                if "authenticity_scores" in stage_result:
                    all_authenticity_scores.extend(stage_result["authenticity_scores"])
            
            pipeline_results["quality_metrics"] = {
                "average_authenticity": sum(all_authenticity_scores) / len(all_authenticity_scores) 
                                      if all_authenticity_scores else 0.0,
                "authenticity_compliance": all(score >= self.config.authenticity_threshold 
                                             for score in all_authenticity_scores),
                "quality_gate_compliance": pipeline_results["stages"]["validation"]["quality_gates_passed"]
            }
            
            # Contract compliance validation
            pipeline_results["contract_compliance"] = self.contract_validator.validate_pipeline_run(
                pipeline_results
            )
            
            # Generate optimization recommendations
            pipeline_results["optimization_opportunities"] = self.optimizer.analyze_pipeline_performance(
                self.stage_metrics, pipeline_results
            )
            
            # Log comprehensive results
            self._log_pipeline_results(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline run failed: {e}")
            pipeline_results["error"] = str(e)
            return pipeline_results
    
    async def _run_discovery_stage(self, config: Config) -> Dict[str, Any]:
        """Run discovery stage with performance tracking."""
        try:
            result = run_discovery(config)
            
            # Extract performance metrics
            return {
                "success": True,
                "pattern_count": result.get("patterns_discovered", 0),
                "authenticity_scores": result.get("authenticity_scores", []),
                "processing_details": result
            }
        except Exception as e:
            self.logger.error(f"Discovery stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_confluence_stage(self, config: Config) -> Dict[str, Any]:
        """Run confluence scoring stage with performance tracking."""
        try:
            result = score_confluence(config)
            
            return {
                "success": True,
                "pattern_count": result.get("patterns_scored", 0),
                "average_score": result.get("average_confluence_score", 0.0),
                "processing_details": result
            }
        except Exception as e:
            self.logger.error(f"Confluence stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_validation_stage(self, config: Config) -> Dict[str, Any]:
        """Run validation stage with performance tracking."""
        try:
            result = validate_run(config)
            
            return {
                "success": True,
                "contracts_checked": result.get("contracts_validated", 0),
                "gates_passed": result.get("all_gates_passed", True),
                "processing_details": result
            }
        except Exception as e:
            self.logger.error(f"Validation stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_reporting_stage(self, config: Config) -> Dict[str, Any]:
        """Run reporting stage with performance tracking."""
        try:
            result = build_minidash(config)
            
            return {
                "success": True,
                "dashboard_path": result.get("dashboard_path"),
                "export_completed": result.get("png_export_success", False),
                "processing_details": result
            }
        except Exception as e:
            self.logger.error(f"Reporting stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _assess_pipeline_health(self):
        """Assess overall pipeline health status."""
        stage_scores = {}
        active_issues = []
        
        for stage_name, metrics in self.stage_metrics.items():
            threshold = getattr(self.config.stage_thresholds, f"{stage_name.lower()}_seconds")
            compliance_rate = metrics.get_compliance_rate(threshold)
            
            # Stage scoring (0.0 to 1.0)
            if compliance_rate >= 0.95:
                stage_scores[stage_name] = 1.0
            elif compliance_rate >= 0.80:
                stage_scores[stage_name] = 0.7
                active_issues.append(f"{stage_name} stage below 95% compliance ({compliance_rate:.1%})")
            else:
                stage_scores[stage_name] = 0.3
                active_issues.append(f"{stage_name} stage critical compliance ({compliance_rate:.1%})")
        
        # Overall health assessment
        overall_score = sum(stage_scores.values()) / len(stage_scores) if stage_scores else 1.0
        
        if overall_score >= 0.9:
            status = "GREEN"
        elif overall_score >= 0.7:
            status = "YELLOW"
        else:
            status = "RED"
        
        # Trend analysis
        if len(self.health_history) >= 3:
            recent_scores = [h.overall_score for h in self.health_history[-3:]]
            if recent_scores[-1] > recent_scores[0] + 0.1:
                trending = "IMPROVING"
            elif recent_scores[-1] < recent_scores[0] - 0.1:
                trending = "DEGRADING"
            else:
                trending = "STABLE"
        else:
            trending = "STABLE"
        
        self.current_health = PipelineHealthStatus(
            status=status,
            overall_score=overall_score,
            stage_scores=stage_scores,
            active_issues=active_issues,
            trending=trending
        )
        
        self.health_history.append(self.current_health)
        
        # Keep only recent history
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-50:]
    
    def _detect_performance_regression(self):
        """Detect performance regression across pipeline stages."""
        if len(self.health_history) < 5:
            return
        
        recent_scores = [h.overall_score for h in self.health_history[-5:]]
        
        # Check for significant degradation
        if recent_scores[-1] < recent_scores[0] - 0.2:
            self._trigger_alert("performance_regression", {
                "current_score": recent_scores[-1],
                "baseline_score": recent_scores[0],
                "degradation": recent_scores[0] - recent_scores[-1]
            })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger performance alert with callback system."""
        alert_data = {
            "type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "health_status": self.current_health.status
        }
        
        self.logger.warning(f"🚨 Performance Alert [{alert_type}]: {data}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add performance alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for external reporting."""
        return {
            "pipeline_health": {
                "status": self.current_health.status,
                "overall_score": self.current_health.overall_score,
                "trending": self.current_health.trending,
                "active_issues": self.current_health.active_issues
            },
            "stage_performance": {
                stage_name: {
                    "average_time": metrics.get_average_time(),
                    "compliance_rate": metrics.get_compliance_rate(
                        getattr(self.config.stage_thresholds, f"{stage_name.lower()}_seconds")
                    ),
                    "error_count": metrics.error_count,
                    "total_executions": len(metrics.processing_times)
                }
                for stage_name, metrics in self.stage_metrics.items()
            },
            "system_performance": {
                "container_initialized": self.container_initialized,
                "container_init_time": self.container_init_time,
                "monitoring_active": self.monitoring_active,
                "current_memory_mb": self._get_current_memory()
            },
            "contract_compliance": self.contract_validator.get_compliance_summary(),
            "optimization_recommendations": self.optimizer.get_current_recommendations()
        }
    
    def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display."""
        return {
            "timestamp": datetime.now().isoformat(),
            "health": self.current_health,
            "stage_metrics": self.stage_metrics,
            "system_metrics": {
                "memory_mb": self._get_current_memory(),
                "container_status": "initialized" if self.container_initialized else "pending"
            },
            "performance_summary": self.get_performance_summary()
        }
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _log_pipeline_results(self, results: Dict[str, Any]):
        """Log comprehensive pipeline performance results."""
        self.logger.info("📊 PIPELINE PERFORMANCE RESULTS")
        self.logger.info("=" * 60)
        
        # Overall metrics
        overall = results["overall_metrics"]
        self.logger.info(f"Total Processing Time: {overall['total_processing_time']:.2f}s")
        self.logger.info(f"Target Time: {overall['target_time']}s")
        self.logger.info(f"Time Compliance: {'✅' if overall['time_compliance'] else '❌'}")
        self.logger.info(f"Peak Memory: {overall['peak_memory_mb']:.1f}MB")
        self.logger.info(f"Memory Compliance: {'✅' if overall['memory_compliance'] else '❌'}")
        
        # Stage breakdown
        self.logger.info("\nStage Performance:")
        for stage_name, metrics in self.stage_metrics.items():
            avg_time = metrics.get_average_time()
            threshold = getattr(self.config.stage_thresholds, f"{stage_name.lower()}_seconds")
            compliance = metrics.get_compliance_rate(threshold)
            
            self.logger.info(f"  {stage_name.title()}: {avg_time:.3f}s avg, {compliance:.1%} compliance")
        
        # Quality metrics
        quality = results["quality_metrics"]
        self.logger.info(f"\nQuality Metrics:")
        self.logger.info(f"Average Authenticity: {quality['average_authenticity']:.1%}")
        self.logger.info(f"Authenticity Compliance: {'✅' if quality['authenticity_compliance'] else '❌'}")
        
        # Optimization opportunities
        if results["optimization_opportunities"]:
            self.logger.info(f"\nOptimization Opportunities:")
            for opportunity in results["optimization_opportunities"][:3]:  # Top 3
                self.logger.info(f"  • {opportunity}")
    
    def log_comprehensive_report(self):
        """Log comprehensive performance monitoring report."""
        summary = self.get_performance_summary()
        
        self.logger.info("📊 COMPREHENSIVE PERFORMANCE REPORT")
        self.logger.info("=" * 70)
        
        # Pipeline health
        health = summary["pipeline_health"]
        self.logger.info(f"Pipeline Health: {health['status']} ({health['overall_score']:.1%})")
        self.logger.info(f"Health Trend: {health['trending']}")
        
        if health["active_issues"]:
            self.logger.info(f"Active Issues: {len(health['active_issues'])}")
            for issue in health["active_issues"]:
                self.logger.info(f"  • {issue}")
        
        # Stage performance summary
        self.logger.info("\nStage Performance Summary:")
        for stage_name, perf in summary["stage_performance"].items():
            self.logger.info(
                f"  {stage_name.title()}: {perf['average_time']:.3f}s avg, "
                f"{perf['compliance_rate']:.1%} compliance, "
                f"{perf['total_executions']} executions"
            )
        
        # System performance
        system = summary["system_performance"]
        self.logger.info(f"\nSystem Performance:")
        self.logger.info(f"Container Status: {system['container_status']}")
        if system["container_init_time"]:
            self.logger.info(f"Container Init Time: {system['container_init_time']:.3f}s")
        self.logger.info(f"Current Memory: {system['current_memory_mb']:.1f}MB")
        
        self.logger.info("\n" + "=" * 70)

    async def execute_primary_function(self, monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute primary pipeline performance monitoring using planning context.
        
        Args:
            monitoring_config: Dict containing pipeline configuration and monitoring parameters
            
        Returns:
            Dict containing performance monitoring results and recommendations
        """
        results = {
            "pipeline_health": {},
            "stage_performance": {},
            "system_metrics": {},
            "optimization_opportunities": [],
            "alerts_triggered": [],
            "recommendations": []
        }
        
        try:
            # Get behavior and dependencies from planning context
            behavior = await self.get_behavior_from_planning()
            dependencies = await self.get_dependencies_from_planning()
            
            # Extract configuration from planning context
            enable_background_monitoring = dependencies.get("ENABLE_BACKGROUND_MONITORING", "true").lower() == "true"
            enable_alerts = dependencies.get("ENABLE_PERFORMANCE_ALERTS", "true").lower() == "true"
            monitor_full_pipeline = dependencies.get("MONITOR_FULL_PIPELINE_RUN", "false").lower() == "true"
            
            # Extract monitoring parameters
            config_path = monitoring_config.get("config_path")
            session_count = monitoring_config.get("session_count", 1)
            target_stages = monitoring_config.get("target_stages", ["discovery", "confluence", "validation", "reporting"])
            
            # Initialize monitoring if not already active
            if not self.monitoring_active and enable_background_monitoring:
                await self.initialize()
            
            # Monitor full pipeline run if requested
            if monitor_full_pipeline and config_path:
                pipeline_results = await self.monitor_full_pipeline_run(config_path)
                results["full_pipeline_results"] = pipeline_results
                
                # Extract key metrics from pipeline results
                if "overall_metrics" in pipeline_results:
                    overall = pipeline_results["overall_metrics"]
                    results["pipeline_health"]["total_processing_time"] = overall.get("total_processing_time", 0.0)
                    results["pipeline_health"]["time_compliance"] = overall.get("time_compliance", False)
                    results["pipeline_health"]["memory_compliance"] = overall.get("memory_compliance", False)
                
                # Extract stage performance
                if "stages" in pipeline_results:
                    for stage_name, stage_result in pipeline_results["stages"].items():
                        if stage_name in self.stage_metrics:
                            stage_perf = self.stage_metrics[stage_name]
                            results["stage_performance"][stage_name] = {
                                "success": stage_result.get("success", False),
                                "average_time": stage_perf.get_average_time(),
                                "compliance_rate": stage_perf.get_compliance_rate(
                                    getattr(self.config.stage_thresholds, f"{stage_name.lower()}_seconds")
                                ),
                                "error_count": stage_perf.error_count
                            }
            
            # Get current performance summary
            performance_summary = self.get_performance_summary()
            results["pipeline_health"] = performance_summary["pipeline_health"]
            results["stage_performance"] = performance_summary["stage_performance"]
            results["system_metrics"] = performance_summary["system_performance"]
            
            # Generate optimization opportunities
            if behavior.get("PROVIDE_OPTIMIZATION_RECOMMENDATIONS", True):
                results["optimization_opportunities"] = performance_summary["optimization_recommendations"]
            
            # Generate recommendations based on current health
            if behavior.get("PROVIDE_HEALTH_RECOMMENDATIONS", True):
                recommendations = []
                
                # Pipeline health recommendations
                health_status = results["pipeline_health"]["status"]
                if health_status == "RED":
                    recommendations.append("CRITICAL: Pipeline health is RED. Immediate attention required")
                elif health_status == "YELLOW":
                    recommendations.append("WARNING: Pipeline health is YELLOW. Consider performance tuning")
                
                # Stage performance recommendations
                for stage_name, stage_perf in results["stage_performance"].items():
                    compliance = stage_perf.get("compliance_rate", 1.0)
                    if compliance < 0.8:
                        recommendations.append(f"Stage {stage_name} compliance low ({compliance:.1%}). Review performance bottlenecks")
                    
                    if stage_perf.get("error_count", 0) > 0:
                        recommendations.append(f"Stage {stage_name} has {stage_perf['error_count']} errors. Investigate error sources")
                
                # Memory usage recommendations
                current_memory = results["system_metrics"]["current_memory_mb"]
                if current_memory > self.config.memory_limit_mb * 0.8:
                    recommendations.append(f"Memory usage high ({current_memory:.1f}MB). Consider memory optimization")
                
                # Container initialization recommendations
                if not results["system_metrics"]["container_initialized"]:
                    recommendations.append("Container system not initialized. Performance may be suboptimal")
                elif results["system_metrics"].get("container_init_time", 0) > 2.0:
                    recommendations.append("Container initialization slow. Consider lazy loading optimization")
                
                results["recommendations"] = recommendations
            
            # Handle alerts if enabled
            if enable_alerts:
                # Check for recent alerts based on current health
                active_issues = results["pipeline_health"].get("active_issues", [])
                results["alerts_triggered"] = active_issues
                
                if active_issues:
                    for issue in active_issues:
                        results["recommendations"].append(f"Alert: {issue}")
            
            # Success metrics
            total_stages_monitored = len([s for s in results["stage_performance"].values() if s.get("success", False)])
            health_score = results["pipeline_health"].get("overall_score", 0.0)
            
            results["status"] = "SUCCESS"
            results["message"] = f"Monitored {total_stages_monitored} pipeline stages with {health_score:.1%} health score"
            
        except Exception as e:
            results["status"] = "ERROR"
            results["message"] = f"Pipeline performance monitoring failed: {str(e)}"
            results["recommendations"].append("Check monitoring configuration and system resources")
        
        return results


def create_pipeline_monitor(config_path: Optional[str] = None) -> PipelinePerformanceMonitorAgent:
    """
    Factory function to create a configured pipeline performance monitor.
    
    Args:
        config_path: Optional path to performance monitoring configuration
        
    Returns:
        Configured PipelinePerformanceMonitorAgent instance
    """
    if config_path:
        config = IRONFORGEPerformanceConfig.from_file(config_path)
    else:
        config = IRONFORGEPerformanceConfig()
    
    agent = PipelinePerformanceMonitorAgent(config)
    
    # Set up default alert handlers
    def default_alert_handler(alert_type: str, data: Dict[str, Any]):
        logger.warning(f"🚨 Pipeline Alert [{alert_type}]: {data}")
    
    agent.add_alert_callback(default_alert_handler)
    
    return agent


# Global agent instance for module-level access
_global_monitor: Optional[PipelinePerformanceMonitorAgent] = None


def get_pipeline_monitor() -> PipelinePerformanceMonitorAgent:
    """Get or create global pipeline performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = create_pipeline_monitor()
    return _global_monitor


async def monitor_pipeline_run(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to monitor a full pipeline run.
    
    Args:
        config_path: Path to IRONFORGE configuration
        
    Returns:
        Comprehensive performance analysis results
    """
    monitor = get_pipeline_monitor()
    
    if not monitor.monitoring_active:
        await monitor.initialize()
    
    return await monitor.monitor_full_pipeline_run(config_path)


if __name__ == "__main__":
    # Example usage for testing
    import asyncio
    
    async def main():
        monitor = create_pipeline_monitor()
        await monitor.initialize()
        
        # Monitor a test configuration
        results = await monitor.monitor_full_pipeline_run("configs/dev.yml")
        
        print(json.dumps(results, indent=2, default=str))
        
        monitor.stop_monitoring()
    
    asyncio.run(main())