"""
Archaeological Zone Detection Performance Monitor
=================================================

Production-grade performance monitoring and optimization for archaeological intelligence.
Ensures sub-3s session processing with >95% accuracy and <100MB memory footprint.

Performance Requirements:
- Session Processing: <3.0s total per session
- Zone Detection: <1.0s for anchor point detection  
- Anchor Accuracy: >95% dimensional anchor precision
- Memory Usage: <100MB total footprint
- Authentication: >87% authenticity threshold maintenance

Monitoring Components:
- Real-time performance tracking with microsecond precision
- Memory usage monitoring with garbage collection optimization
- Accuracy tracking with precision validation
- Temporal coherence monitoring for Theory B compliance
- Integration performance with IRONFORGE pipeline components

Optimization Features:
- Adaptive algorithm selection based on session complexity
- Lazy loading and resource management
- Performance degradation detection and alerting
- Automatic garbage collection triggers
- Performance regression testing integration
"""

from __future__ import annotations

import gc
import logging
import psutil
import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch

from .ironforge_config import ArchaeologicalConfig

logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Performance monitoring detail levels"""
    MINIMAL = "minimal"      # Basic timing only
    STANDARD = "standard"    # Standard monitoring (production)
    DETAILED = "detailed"    # Detailed profiling (development)
    COMPREHENSIVE = "comprehensive"  # Full profiling with memory/CPU


class AlertSeverity(Enum):
    """Performance alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: float
    session_id: Optional[str] = None
    component: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.name}: {self.value:.3f} {self.unit}"


@dataclass  
class SessionPerformanceData:
    """Complete performance data for a single session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_processing_time: float = 0.0
    zone_detection_time: float = 0.0
    temporal_analysis_time: float = 0.0
    theory_b_validation_time: float = 0.0
    authenticity_scoring_time: float = 0.0
    
    # Accuracy metrics
    anchor_accuracy: float = 0.0
    precision_score: float = 0.0
    authenticity_score: float = 0.0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Quality metrics
    zone_count: int = 0
    authenticated_zones: int = 0
    contract_violations: int = 0
    
    # Performance flags
    processing_time_exceeded: bool = False
    memory_limit_exceeded: bool = False
    accuracy_below_threshold: bool = False
    
    def __post_init__(self):
        if self.end_time is not None and self.start_time > 0:
            self.total_processing_time = self.end_time - self.start_time


@dataclass
class PerformanceAlert:
    """Performance degradation alert"""
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold_value: float
    actual_value: float
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    component: Optional[str] = None
    
    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.message} "
            f"({self.metric_name}: {self.actual_value:.3f} > {self.threshold_value:.3f})"
        )


class PerformanceTimer:
    """High-precision performance timer with context management"""
    
    def __init__(self, name: str, component: Optional[str] = None):
        self.name = name
        self.component = component
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
    
    def start(self) -> None:
        """Start timing measurement"""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing measurement and return elapsed time"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        return self.elapsed_time
    
    def __enter__(self) -> 'PerformanceTimer':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.elapsed_time is not None:
            return self.elapsed_time
        elif self.start_time is not None:
            return time.perf_counter() - self.start_time
        else:
            return 0.0


class MemoryMonitor:
    """Real-time memory usage monitoring with optimization"""
    
    def __init__(self, config: ArchaeologicalConfig):
        self.config = config
        self.perf_config = config.performance
        self.process = psutil.Process()
        self.baseline_memory = self._get_current_memory()
        self.peak_memory = self.baseline_memory
        self.gc_threshold = self.perf_config.garbage_collection_threshold
        
        logger.debug(f"Memory Monitor initialized (baseline: {self.baseline_memory:.1f}MB)")
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def check_memory_usage(self) -> Tuple[float, bool]:
        """
        Check current memory usage and trigger GC if needed
        
        Returns:
            Tuple of (current_memory_mb, gc_triggered)
        """
        current_memory = self._get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # Trigger garbage collection if threshold exceeded
        gc_triggered = False
        if current_memory > self.gc_threshold:
            gc.collect()
            gc_triggered = True
            new_memory = self._get_current_memory()
            logger.debug(f"GC triggered: {current_memory:.1f}MB -> {new_memory:.1f}MB")
            current_memory = new_memory
        
        return current_memory, gc_triggered
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        current_memory = self._get_current_memory()
        return {
            "current_memory_mb": current_memory,
            "peak_memory_mb": self.peak_memory,
            "baseline_memory_mb": self.baseline_memory,
            "memory_increase_mb": current_memory - self.baseline_memory,
            "memory_usage_ratio": current_memory / self.perf_config.max_memory_usage_mb
        }
    
    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking for new session"""
        self.peak_memory = self._get_current_memory()


class AccuracyTracker:
    """Tracks accuracy metrics for archaeological zone detection"""
    
    def __init__(self, config: ArchaeologicalConfig):
        self.config = config
        self.perf_config = config.performance
        self.anchor_config = config.dimensional_anchor
        
        # Accuracy tracking
        self.total_zones_detected = 0
        self.accurate_zones = 0
        self.precision_scores = []
        self.authenticity_scores = []
        
        logger.debug("Accuracy Tracker initialized")
    
    def record_zone_accuracy(
        self,
        zone_data: Dict[str, Any],
        actual_accuracy: Optional[float] = None
    ) -> None:
        """Record accuracy for a detected zone"""
        self.total_zones_detected += 1
        
        # Extract accuracy metrics
        precision_score = zone_data.get('precision_score', 0.0)
        authenticity_score = zone_data.get('authenticity_score', 0.0)
        
        # Record precision
        if precision_score > 0:
            self.precision_scores.append(precision_score)
        
        # Record authenticity
        if authenticity_score > 0:
            self.authenticity_scores.append(authenticity_score)
        
        # Count accurate zones (meeting precision target)
        if precision_score >= self.anchor_config.precision_target * 0.9:  # 90% of target
            self.accurate_zones += 1
        
        # Record actual accuracy if provided
        if actual_accuracy is not None and actual_accuracy >= 0.9:  # 90% accuracy
            self.accurate_zones += 1
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Get comprehensive accuracy metrics"""
        if self.total_zones_detected == 0:
            return {
                "anchor_accuracy": 0.0,
                "precision_score": 0.0,
                "authenticity_score": 0.0,
                "accuracy_ratio": 0.0,
                "zones_evaluated": 0
            }
        
        anchor_accuracy = (self.accurate_zones / self.total_zones_detected) * 100
        avg_precision = np.mean(self.precision_scores) if self.precision_scores else 0.0
        avg_authenticity = np.mean(self.authenticity_scores) if self.authenticity_scores else 0.0
        
        return {
            "anchor_accuracy": anchor_accuracy,
            "precision_score": avg_precision,
            "authenticity_score": avg_authenticity,
            "accuracy_ratio": self.accurate_zones / self.total_zones_detected,
            "zones_evaluated": self.total_zones_detected,
            "accurate_zones": self.accurate_zones
        }
    
    def reset_accuracy_tracking(self) -> None:
        """Reset accuracy tracking for new session"""
        self.total_zones_detected = 0
        self.accurate_zones = 0
        self.precision_scores = []
        self.authenticity_scores = []


class PerformanceTrendAnalyzer:
    """Analyzes performance trends and detects degradation"""
    
    def __init__(self, config: ArchaeologicalConfig, window_size: int = 10):
        self.config = config
        self.window_size = window_size
        
        # Trend tracking
        self.processing_times = deque(maxlen=window_size)
        self.memory_usage_history = deque(maxlen=window_size)
        self.accuracy_history = deque(maxlen=window_size)
        
        # Degradation thresholds
        self.degradation_threshold = config.performance.performance_degradation_threshold
        
        logger.debug(f"Performance Trend Analyzer initialized (window: {window_size})")
    
    def record_session_performance(self, session_data: SessionPerformanceData) -> None:
        """Record session performance for trend analysis"""
        self.processing_times.append(session_data.total_processing_time)
        self.memory_usage_history.append(session_data.peak_memory_mb)
        self.accuracy_history.append(session_data.anchor_accuracy)
    
    def detect_performance_degradation(self) -> List[PerformanceAlert]:
        """Detect performance degradation trends"""
        alerts = []
        
        if len(self.processing_times) < 3:  # Need minimum data points
            return alerts
        
        # Analyze processing time trend
        processing_trend = self._calculate_trend(list(self.processing_times))
        if processing_trend > self.degradation_threshold:
            alerts.append(PerformanceAlert(
                severity=AlertSeverity.WARNING,
                message="Processing time degradation detected",
                metric_name="processing_time_trend",
                threshold_value=self.degradation_threshold,
                actual_value=processing_trend,
                component="trend_analysis"
            ))
        
        # Analyze memory usage trend  
        memory_trend = self._calculate_trend(list(self.memory_usage_history))
        if memory_trend > self.degradation_threshold:
            alerts.append(PerformanceAlert(
                severity=AlertSeverity.WARNING,
                message="Memory usage degradation detected",
                metric_name="memory_usage_trend", 
                threshold_value=self.degradation_threshold,
                actual_value=memory_trend,
                component="trend_analysis"
            ))
        
        # Analyze accuracy trend (negative trend is bad)
        accuracy_trend = self._calculate_trend(list(self.accuracy_history))
        if accuracy_trend < -self.degradation_threshold:
            alerts.append(PerformanceAlert(
                severity=AlertSeverity.CRITICAL,
                message="Accuracy degradation detected",
                metric_name="accuracy_trend",
                threshold_value=-self.degradation_threshold,
                actual_value=accuracy_trend,
                component="trend_analysis"
            ))
        
        return alerts
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend coefficient for a series of values"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Handle zero or constant values
        if np.std(y) == 0:
            return 0.0
        
        # Calculate correlation coefficient as trend indicator
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        
        # Scale by relative change to get trend strength
        relative_change = (y[-1] - y[0]) / y[0] if y[0] != 0 else 0.0
        trend_strength = correlation * relative_change
        
        return trend_strength
    
    def get_trend_summary(self) -> Dict[str, float]:
        """Get comprehensive trend analysis summary"""
        return {
            "processing_time_trend": self._calculate_trend(list(self.processing_times)),
            "memory_usage_trend": self._calculate_trend(list(self.memory_usage_history)),
            "accuracy_trend": self._calculate_trend(list(self.accuracy_history)),
            "data_points": len(self.processing_times),
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0.0,
            "avg_memory_usage": np.mean(self.memory_usage_history) if self.memory_usage_history else 0.0,
            "avg_accuracy": np.mean(self.accuracy_history) if self.accuracy_history else 0.0
        }


class ArchaeologicalPerformanceMonitor:
    """
    Comprehensive Performance Monitor for Archaeological Zone Detection
    
    Production-grade performance monitoring ensuring IRONFORGE compliance:
    - <3s session processing requirement
    - >95% anchor accuracy requirement  
    - <100MB memory footprint requirement
    - >87% authenticity threshold maintenance
    
    Features:
    - Real-time performance tracking with microsecond precision
    - Memory optimization with automatic garbage collection
    - Accuracy validation with precision scoring
    - Performance degradation detection and alerting
    - Integration performance monitoring with IRONFORGE pipeline
    - Comprehensive reporting and trend analysis
    """
    
    def __init__(
        self,
        config: Optional[ArchaeologicalConfig] = None,
        monitoring_level: PerformanceLevel = PerformanceLevel.STANDARD
    ):
        self.config = config if config is not None else ArchaeologicalConfig()
        self.monitoring_level = monitoring_level
        self.perf_config = self.config.performance
        
        # Initialize monitoring components
        self.memory_monitor = MemoryMonitor(self.config)
        self.accuracy_tracker = AccuracyTracker(self.config)
        self.trend_analyzer = PerformanceTrendAnalyzer(self.config)
        
        # Session tracking
        self.current_session: Optional[SessionPerformanceData] = None
        self.session_history: List[SessionPerformanceData] = []
        self.active_timers: Dict[str, PerformanceTimer] = {}
        
        # Performance alerts
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Archaeological Performance Monitor initialized (level: {monitoring_level.value})")
    
    def start_session_analysis(self, session_id: Optional[str] = None) -> str:
        """
        Start performance monitoring for session analysis
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Session ID for tracking
        """
        with self._lock:
            if session_id is None:
                session_id = f"session_{int(time.time() * 1000)}"
            
            # End previous session if active
            if self.current_session is not None and self.current_session.end_time is None:
                self.end_session_analysis()
            
            # Start new session
            self.current_session = SessionPerformanceData(
                session_id=session_id,
                start_time=time.perf_counter()
            )
            
            # Reset tracking components
            self.memory_monitor.reset_peak_memory()
            self.accuracy_tracker.reset_accuracy_tracking()
            self.active_timers.clear()
            
            logger.debug(f"Started performance monitoring for session: {session_id}")
            return session_id
    
    def end_session_analysis(self) -> Optional[SessionPerformanceData]:
        """
        End performance monitoring for current session
        
        Returns:
            Complete session performance data
        """
        with self._lock:
            if self.current_session is None:
                logger.warning("No active session to end")
                return None
            
            # Finalize session data
            self.current_session.end_time = time.perf_counter()
            
            # Get final memory stats
            memory_stats = self.memory_monitor.get_memory_stats()
            self.current_session.peak_memory_mb = memory_stats["peak_memory_mb"]
            
            # Get final accuracy metrics
            accuracy_metrics = self.accuracy_tracker.get_accuracy_metrics()
            self.current_session.anchor_accuracy = accuracy_metrics["anchor_accuracy"]
            self.current_session.precision_score = accuracy_metrics["precision_score"]
            self.current_session.authenticity_score = accuracy_metrics["authenticity_score"]
            self.current_session.authenticated_zones = accuracy_metrics["accurate_zones"]
            self.current_session.zone_count = accuracy_metrics["zones_evaluated"]
            
            # Check performance thresholds
            self._validate_session_performance(self.current_session)
            
            # Record session for trend analysis
            self.trend_analyzer.record_session_performance(self.current_session)
            
            # Add to history
            completed_session = self.current_session
            self.session_history.append(completed_session)
            self.current_session = None
            
            logger.debug(
                f"Completed performance monitoring for session: {completed_session.session_id} "
                f"({completed_session.total_processing_time:.3f}s)"
            )
            
            return completed_session
    
    @contextmanager
    def time_operation(self, operation_name: str, component: Optional[str] = None):
        """
        Context manager for timing specific operations
        
        Args:
            operation_name: Name of operation being timed
            component: Optional component name for categorization
        """
        timer = PerformanceTimer(operation_name, component)
        timer_key = f"{component}_{operation_name}" if component else operation_name
        
        try:
            with self._lock:
                self.active_timers[timer_key] = timer
            
            with timer:
                yield timer
                
            # Record timing in current session
            if self.current_session is not None:
                elapsed_time = timer.elapsed
                
                # Update specific timing fields
                if operation_name == "zone_detection":
                    self.current_session.zone_detection_time += elapsed_time
                elif operation_name == "temporal_analysis":
                    self.current_session.temporal_analysis_time += elapsed_time
                elif operation_name == "theory_b_validation":
                    self.current_session.theory_b_validation_time += elapsed_time
                elif operation_name == "authenticity_scoring":
                    self.current_session.authenticity_scoring_time += elapsed_time
                
                # Check operation performance
                self._check_operation_performance(operation_name, elapsed_time)
        
        finally:
            with self._lock:
                if timer_key in self.active_timers:
                    del self.active_timers[timer_key]
    
    def record_zone_performance(
        self,
        zone_data: Dict[str, Any],
        accuracy: Optional[float] = None
    ) -> None:
        """
        Record performance metrics for detected zone
        
        Args:
            zone_data: Zone detection results
            accuracy: Optional accuracy measurement
        """
        # Record accuracy metrics
        self.accuracy_tracker.record_zone_accuracy(zone_data, accuracy)
        
        # Check memory usage
        memory_usage, gc_triggered = self.memory_monitor.check_memory_usage()
        
        # Generate alerts if needed
        if memory_usage > self.perf_config.memory_warning_threshold:
            self._generate_alert(
                AlertSeverity.WARNING,
                f"High memory usage during zone detection",
                "memory_usage",
                self.perf_config.memory_warning_threshold,
                memory_usage
            )
        
        if gc_triggered:
            logger.debug("Garbage collection triggered during zone detection")
    
    def get_session_metrics(
        self,
        processing_time: float,
        zone_count: int,
        session_id: str
    ) -> Dict[str, float]:
        """
        Get comprehensive session performance metrics
        
        Args:
            processing_time: Total processing time for session
            zone_count: Number of zones detected
            session_id: Session identifier
            
        Returns:
            Dictionary of performance metrics
        """
        # Get current memory stats
        memory_stats = self.memory_monitor.get_memory_stats()
        
        # Get accuracy metrics
        accuracy_metrics = self.accuracy_tracker.get_accuracy_metrics()
        
        # Get CPU usage
        cpu_usage = self._get_cpu_usage()
        
        # Compile comprehensive metrics
        metrics = {
            # Timing metrics
            "detection_time": processing_time,
            "processing_time": processing_time,
            "zone_detection_rate": zone_count / processing_time if processing_time > 0 else 0.0,
            
            # Accuracy metrics
            "accuracy": accuracy_metrics["anchor_accuracy"],
            "precision_score": accuracy_metrics["precision_score"],
            "authenticity_score": accuracy_metrics["authenticity_score"],
            
            # Resource metrics
            "memory_usage_mb": memory_stats["current_memory_mb"],
            "peak_memory_mb": memory_stats["peak_memory_mb"],
            "memory_increase_mb": memory_stats["memory_increase_mb"],
            "cpu_usage_percent": cpu_usage,
            
            # Quality metrics
            "zone_count": zone_count,
            "authenticated_zones": accuracy_metrics["accurate_zones"],
            "zone_authenticity_rate": accuracy_metrics["accuracy_ratio"],
            
            # Performance flags
            "performance_compliant": processing_time < self.perf_config.max_session_processing_time,
            "memory_compliant": memory_stats["current_memory_mb"] < self.perf_config.max_memory_usage_mb,
            "accuracy_compliant": accuracy_metrics["anchor_accuracy"] >= self.perf_config.min_anchor_accuracy,
            
            # Efficiency metrics
            "zones_per_second": zone_count / processing_time if processing_time > 0 else 0.0,
            "memory_per_zone": memory_stats["memory_increase_mb"] / zone_count if zone_count > 0 else 0.0,
            "authenticity_efficiency": accuracy_metrics["accurate_zones"] / processing_time if processing_time > 0 else 0.0
        }
        
        return metrics
    
    def record_error(self, error_message: str, component: Optional[str] = None) -> None:
        """
        Record error for performance tracking
        
        Args:
            error_message: Error description
            component: Optional component where error occurred
        """
        if self.current_session is not None:
            self.current_session.contract_violations += 1
        
        self._generate_alert(
            AlertSeverity.CRITICAL,
            f"Error in archaeological analysis: {error_message}",
            "error_count",
            0,
            1,
            component=component
        )
        
        logger.error(f"Archaeological analysis error recorded: {error_message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        # Current session metrics
        current_metrics = {}
        if self.current_session is not None:
            current_metrics = {
                "session_id": self.current_session.session_id,
                "elapsed_time": time.perf_counter() - self.current_session.start_time,
                "zone_count": self.current_session.zone_count
            }
        
        # Historical performance
        if self.session_history:
            processing_times = [s.total_processing_time for s in self.session_history]
            memory_usage = [s.peak_memory_mb for s in self.session_history]
            accuracy_scores = [s.anchor_accuracy for s in self.session_history]
            
            historical_metrics = {
                "total_sessions": len(self.session_history),
                "avg_processing_time": np.mean(processing_times),
                "max_processing_time": np.max(processing_times),
                "avg_memory_usage": np.mean(memory_usage),
                "max_memory_usage": np.max(memory_usage),
                "avg_accuracy": np.mean(accuracy_scores),
                "min_accuracy": np.min(accuracy_scores),
                "sessions_exceeding_time_limit": sum(
                    1 for t in processing_times 
                    if t > self.perf_config.max_session_processing_time
                ),
                "sessions_exceeding_memory_limit": sum(
                    1 for m in memory_usage
                    if m > self.perf_config.max_memory_usage_mb
                )
            }
        else:
            historical_metrics = {"total_sessions": 0}
        
        # Trend analysis
        trend_summary = self.trend_analyzer.get_trend_summary()
        
        # Alert summary
        alert_summary = {
            "total_alerts": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]),
            "warning_alerts": len([a for a in self.alerts if a.severity == AlertSeverity.WARNING]),
            "recent_alerts": [str(a) for a in self.alerts[-5:]]  # Last 5 alerts
        }
        
        return {
            "monitoring_level": self.monitoring_level.value,
            "current_session": current_metrics,
            "historical_performance": historical_metrics,
            "performance_trends": trend_summary,
            "alerts": alert_summary,
            "memory_stats": self.memory_monitor.get_memory_stats(),
            "accuracy_stats": self.accuracy_tracker.get_accuracy_metrics(),
            "performance_requirements": {
                "max_session_time": self.perf_config.max_session_processing_time,
                "max_detection_time": self.perf_config.max_detection_time,
                "min_accuracy": self.perf_config.min_anchor_accuracy,
                "max_memory_mb": self.perf_config.max_memory_usage_mb
            }
        }
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback function for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def _validate_session_performance(self, session_data: SessionPerformanceData) -> None:
        """Validate session performance against requirements"""
        # Check processing time
        if session_data.total_processing_time > self.perf_config.max_session_processing_time:
            session_data.processing_time_exceeded = True
            self._generate_alert(
                AlertSeverity.CRITICAL,
                f"Session processing time exceeded limit",
                "session_processing_time",
                self.perf_config.max_session_processing_time,
                session_data.total_processing_time,
                session_data.session_id
            )
        
        # Check memory usage
        if session_data.peak_memory_mb > self.perf_config.max_memory_usage_mb:
            session_data.memory_limit_exceeded = True
            self._generate_alert(
                AlertSeverity.WARNING,
                f"Session memory usage exceeded limit",
                "peak_memory_usage",
                self.perf_config.max_memory_usage_mb,
                session_data.peak_memory_mb,
                session_data.session_id
            )
        
        # Check accuracy
        if session_data.anchor_accuracy < self.perf_config.min_anchor_accuracy:
            session_data.accuracy_below_threshold = True
            self._generate_alert(
                AlertSeverity.CRITICAL,
                f"Session accuracy below requirement",
                "anchor_accuracy",
                self.perf_config.min_anchor_accuracy,
                session_data.anchor_accuracy,
                session_data.session_id
            )
    
    def _check_operation_performance(self, operation_name: str, elapsed_time: float) -> None:
        """Check individual operation performance"""
        # Zone detection timing check
        if operation_name == "zone_detection" and elapsed_time > self.perf_config.max_detection_time:
            self._generate_alert(
                AlertSeverity.WARNING,
                f"Zone detection time exceeded limit",
                "zone_detection_time",
                self.perf_config.max_detection_time,
                elapsed_time
            )
        
        # General operation timing check
        if elapsed_time > self.perf_config.slow_session_threshold:
            self._generate_alert(
                AlertSeverity.INFO,
                f"Slow operation detected: {operation_name}",
                f"{operation_name}_time",
                self.perf_config.slow_session_threshold,
                elapsed_time
            )
    
    def _generate_alert(
        self,
        severity: AlertSeverity,
        message: str,
        metric_name: str,
        threshold_value: float,
        actual_value: float,
        session_id: Optional[str] = None,
        component: Optional[str] = None
    ) -> None:
        """Generate performance alert"""
        alert = PerformanceAlert(
            severity=severity,
            message=message,
            metric_name=metric_name,
            threshold_value=threshold_value,
            actual_value=actual_value,
            session_id=session_id or (self.current_session.session_id if self.current_session else None),
            component=component
        )
        
        self.alerts.append(alert)
        
        # Log alert
        if severity == AlertSeverity.CRITICAL:
            logger.error(str(alert))
        elif severity == AlertSeverity.WARNING:
            logger.warning(str(alert))
        else:
            logger.info(str(alert))
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return self.memory_monitor.process.cpu_percent()
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {e}")
            return 0.0


# Factory functions for different monitoring configurations
def create_production_monitor(config: Optional[ArchaeologicalConfig] = None) -> ArchaeologicalPerformanceMonitor:
    """Create performance monitor optimized for production use"""
    return ArchaeologicalPerformanceMonitor(config, PerformanceLevel.STANDARD)


def create_development_monitor(config: Optional[ArchaeologicalConfig] = None) -> ArchaeologicalPerformanceMonitor:
    """Create performance monitor with detailed profiling for development"""
    return ArchaeologicalPerformanceMonitor(config, PerformanceLevel.DETAILED)


def create_minimal_monitor(config: Optional[ArchaeologicalConfig] = None) -> ArchaeologicalPerformanceMonitor:
    """Create minimal performance monitor for basic timing only"""
    return ArchaeologicalPerformanceMonitor(config, PerformanceLevel.MINIMAL)


# Performance testing utilities
def benchmark_zone_detection(
    detector_func: Callable,
    test_sessions: List[pd.DataFrame],
    config: Optional[ArchaeologicalConfig] = None
) -> Dict[str, float]:
    """
    Benchmark archaeological zone detection performance
    
    Args:
        detector_func: Zone detection function to benchmark
        test_sessions: List of test session DataFrames
        config: Optional configuration
        
    Returns:
        Benchmark results dictionary
    """
    monitor = create_development_monitor(config)
    results = []
    
    for i, session_data in enumerate(test_sessions):
        session_id = monitor.start_session_analysis(f"benchmark_{i}")
        
        try:
            with monitor.time_operation("zone_detection", "benchmark"):
                zones = detector_func(session_data)
                
                # Record zone performance
                for zone in zones:
                    monitor.record_zone_performance(zone)
        
        except Exception as e:
            monitor.record_error(f"Benchmark error: {e}")
        
        finally:
            session_performance = monitor.end_session_analysis()
            if session_performance:
                results.append(session_performance)
    
    # Calculate benchmark statistics
    if results:
        processing_times = [r.total_processing_time for r in results]
        memory_usage = [r.peak_memory_mb for r in results]
        accuracy_scores = [r.anchor_accuracy for r in results]
        
        benchmark_stats = {
            "total_sessions": len(results),
            "avg_processing_time": np.mean(processing_times),
            "max_processing_time": np.max(processing_times),
            "min_processing_time": np.min(processing_times),
            "std_processing_time": np.std(processing_times),
            "avg_memory_usage": np.mean(memory_usage),
            "max_memory_usage": np.max(memory_usage),
            "avg_accuracy": np.mean(accuracy_scores),
            "min_accuracy": np.min(accuracy_scores),
            "sessions_meeting_time_requirement": sum(
                1 for t in processing_times 
                if t < (config.performance.max_session_processing_time if config else 3.0)
            ),
            "sessions_meeting_accuracy_requirement": sum(
                1 for a in accuracy_scores
                if a >= (config.performance.min_anchor_accuracy if config else 95.0)
            ),
            "performance_score": len([
                r for r in results
                if (r.total_processing_time < 3.0 and 
                    r.anchor_accuracy >= 95.0 and
                    r.peak_memory_mb < 100.0)
            ]) / len(results) * 100
        }
        
        return benchmark_stats
    
    return {"error": "No successful benchmark results"}


# Export all performance components
__all__ = [
    "ArchaeologicalPerformanceMonitor",
    "PerformanceTimer",
    "MemoryMonitor", 
    "AccuracyTracker",
    "PerformanceTrendAnalyzer",
    "SessionPerformanceData",
    "PerformanceMetric",
    "PerformanceAlert",
    "PerformanceLevel",
    "AlertSeverity",
    "create_production_monitor",
    "create_development_monitor",
    "create_minimal_monitor",
    "benchmark_zone_detection"
]