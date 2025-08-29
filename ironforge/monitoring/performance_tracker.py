"""
BMAD Performance Monitoring System

This module provides comprehensive performance monitoring for BMAD temporal
metamorphosis detection with specific focus on sub-3-second session processing
targets and real-time pattern analysis metrics.

Performance Targets:
- Single Session Processing: <3 seconds
- Full Discovery (57 sessions): <180 seconds  
- Initialization: <2 seconds with lazy loading
- Memory Usage: <100MB total footprint
- Pattern Quality: >87% authenticity threshold
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
import psutil
import gc
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for BMAD operations."""
    
    # Timing metrics
    session_processing_times: List[float] = field(default_factory=list)
    confluence_scoring_times: List[float] = field(default_factory=list) 
    pattern_discovery_times: List[float] = field(default_factory=list)
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    
    # Quality metrics
    pattern_authenticity_scores: List[float] = field(default_factory=list)
    metamorphosis_detection_rates: List[float] = field(default_factory=list)
    statistical_significance_levels: List[float] = field(default_factory=list)
    
    # Throughput metrics
    patterns_processed: int = 0
    sessions_completed: int = 0
    metamorphosis_events_detected: int = 0
    
    # Error tracking
    processing_errors: List[str] = field(default_factory=list)
    timeout_events: int = 0
    
    def add_session_time(self, processing_time: float):
        """Add session processing time and check against 3-second target."""
        self.session_processing_times.append(processing_time)
        self.sessions_completed += 1
        
        if processing_time > 3.0:
            logger.warning(f"⚠️  Session processing exceeded 3s target: {processing_time:.2f}s")
    
    def add_confluence_time(self, scoring_time: float):
        """Add confluence scoring time."""
        self.confluence_scoring_times.append(scoring_time)
        
    def add_memory_sample(self, memory_mb: float):
        """Add memory usage sample and track peak."""
        self.memory_samples.append(memory_mb)
        self.current_memory_mb = memory_mb
        if memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = memory_mb
            
        if memory_mb > 100.0:
            logger.warning(f"⚠️  Memory usage exceeded 100MB target: {memory_mb:.1f}MB")
    
    def get_average_session_time(self) -> float:
        """Get average session processing time."""
        if not self.session_processing_times:
            return 0.0
        return sum(self.session_processing_times) / len(self.session_processing_times)
    
    def get_sub_3s_compliance(self) -> float:
        """Get percentage of sessions processed under 3 seconds."""
        if not self.session_processing_times:
            return 1.0
        under_3s = len([t for t in self.session_processing_times if t < 3.0])
        return under_3s / len(self.session_processing_times)
    
    def get_memory_efficiency(self) -> float:
        """Get memory efficiency score (1.0 = perfect, 0.0 = over budget)."""
        if self.peak_memory_mb <= 100.0:
            return 1.0
        return max(0.0, 2.0 - (self.peak_memory_mb / 100.0))


class BMadPerformanceTracker:
    """
    BMAD Performance Tracker for real-time monitoring and optimization.
    
    Monitors all aspects of BMAD temporal metamorphosis detection performance
    including processing times, memory usage, pattern quality, and system health.
    """
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.monitoring_active = False
        self.memory_monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance thresholds
        self.thresholds = {
            'session_processing_seconds': 3.0,
            'full_discovery_seconds': 180.0,
            'initialization_seconds': 2.0,
            'memory_limit_mb': 100.0,
            'pattern_authenticity_min': 0.87
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def start_monitoring(self):\n        \"\"\"Start continuous performance monitoring.\"\"\"\n        if self.monitoring_active:\n            return\n            \n        self.monitoring_active = True\n        self.memory_monitor_thread = threading.Thread(target=self._memory_monitor_loop)\n        self.memory_monitor_thread.daemon = True\n        self.memory_monitor_thread.start()\n        \n        self.logger.info(\"📊 BMAD Performance Monitoring started\")\n        self.logger.info(f\"   Session target: <{self.thresholds['session_processing_seconds']}s\")\n        self.logger.info(f\"   Memory limit: <{self.thresholds['memory_limit_mb']}MB\")\n    \n    def stop_monitoring(self):\n        \"\"\"Stop performance monitoring.\"\"\"\n        self.monitoring_active = False\n        if self.memory_monitor_thread:\n            self.memory_monitor_thread.join(timeout=1.0)\n        self.logger.info(\"📊 BMAD Performance Monitoring stopped\")\n    \n    def _memory_monitor_loop(self):\n        \"\"\"Background thread for continuous memory monitoring.\"\"\"\n        process = psutil.Process()\n        \n        while self.monitoring_active:\n            try:\n                memory_info = process.memory_info()\n                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB\n                self.metrics.add_memory_sample(memory_mb)\n                \n                # Check for memory alerts\n                if memory_mb > self.thresholds['memory_limit_mb']:\n                    self._trigger_alert('memory_exceeded', {\n                        'current_mb': memory_mb,\n                        'limit_mb': self.thresholds['memory_limit_mb']\n                    })\n                \n                time.sleep(1.0)  # Sample every second\n                \n            except Exception as e:\n                self.logger.error(f\"Memory monitoring error: {e}\")\n                time.sleep(5.0)  # Back off on errors\n    \n    @contextmanager\n    def track_session_processing(self, session_id: str):\n        \"\"\"Context manager for tracking session processing time.\"\"\"\n        start_time = time.time()\n        start_memory = self._get_current_memory()\n        \n        try:\n            yield\n        finally:\n            processing_time = time.time() - start_time\n            end_memory = self._get_current_memory()\n            \n            # Record metrics\n            self.metrics.add_session_time(processing_time)\n            \n            # Log performance\n            status = \"✅\" if processing_time < 3.0 else \"⚠️ \"\n            self.logger.info(f\"{status} Session {session_id}: {processing_time:.2f}s\")\n            \n            # Check for performance alerts\n            if processing_time > self.thresholds['session_processing_seconds']:\n                self._trigger_alert('session_timeout', {\n                    'session_id': session_id,\n                    'processing_time': processing_time,\n                    'target_time': self.thresholds['session_processing_seconds']\n                })\n    \n    @contextmanager\n    def track_confluence_scoring(self, pattern_count: int):\n        \"\"\"Context manager for tracking confluence scoring performance.\"\"\"\n        start_time = time.time()\n        \n        try:\n            yield\n        finally:\n            scoring_time = time.time() - start_time\n            self.metrics.add_confluence_time(scoring_time)\n            self.metrics.patterns_processed += pattern_count\n            \n            # Calculate throughput\n            throughput = pattern_count / scoring_time if scoring_time > 0 else 0\n            \n            self.logger.info(f\"🎯 Confluence scoring: {scoring_time:.2f}s for {pattern_count} patterns\")\n            self.logger.info(f\"   Throughput: {throughput:.1f} patterns/second\")\n    \n    @contextmanager\n    def track_metamorphosis_detection(self):\n        \"\"\"Context manager for tracking metamorphosis detection operations.\"\"\"\n        start_time = time.time()\n        initial_detections = self.metrics.metamorphosis_events_detected\n        \n        try:\n            yield\n        finally:\n            detection_time = time.time() - start_time\n            new_detections = self.metrics.metamorphosis_events_detected - initial_detections\n            \n            self.logger.info(f\"🔍 Metamorphosis detection: {detection_time:.2f}s, {new_detections} events\")\n    \n    def record_metamorphosis_detection(self, strength: float, significance: float):\n        \"\"\"Record a metamorphosis detection event.\"\"\"\n        self.metrics.metamorphosis_events_detected += 1\n        self.metrics.metamorphosis_detection_rates.append(strength)\n        self.metrics.statistical_significance_levels.append(significance)\n        \n        # Log significant detections\n        if strength > 0.237:  # Strong metamorphosis threshold\n            self.logger.info(f\"🚨 Strong metamorphosis detected: {strength:.1%} strength\")\n    \n    def record_pattern_quality(self, authenticity_score: float):\n        \"\"\"Record pattern authenticity score.\"\"\"\n        self.metrics.pattern_authenticity_scores.append(authenticity_score)\n        \n        # Check quality alerts\n        if authenticity_score < self.thresholds['pattern_authenticity_min']:\n            self._trigger_alert('pattern_quality_low', {\n                'authenticity_score': authenticity_score,\n                'minimum_threshold': self.thresholds['pattern_authenticity_min']\n            })\n    \n    def get_performance_summary(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive performance summary.\"\"\"\n        return {\n            'processing_performance': {\n                'average_session_time': self.metrics.get_average_session_time(),\n                'sub_3s_compliance': self.metrics.get_sub_3s_compliance(),\n                'total_sessions': self.metrics.sessions_completed,\n                'total_patterns': self.metrics.patterns_processed\n            },\n            'memory_performance': {\n                'current_memory_mb': self.metrics.current_memory_mb,\n                'peak_memory_mb': self.metrics.peak_memory_mb,\n                'memory_efficiency': self.metrics.get_memory_efficiency()\n            },\n            'quality_metrics': {\n                'average_authenticity': (\n                    sum(self.metrics.pattern_authenticity_scores) / \n                    len(self.metrics.pattern_authenticity_scores)\n                ) if self.metrics.pattern_authenticity_scores else 0.0,\n                'metamorphosis_detections': self.metrics.metamorphosis_events_detected,\n                'average_detection_strength': (\n                    sum(self.metrics.metamorphosis_detection_rates) /\n                    len(self.metrics.metamorphosis_detection_rates)\n                ) if self.metrics.metamorphosis_detection_rates else 0.0\n            },\n            'system_health': {\n                'processing_errors': len(self.metrics.processing_errors),\n                'timeout_events': self.metrics.timeout_events,\n                'monitoring_active': self.monitoring_active\n            },\n            'compliance': {\n                'session_processing_target': self.metrics.get_sub_3s_compliance() >= 0.95,\n                'memory_target': self.metrics.peak_memory_mb <= 100.0,\n                'quality_target': (\n                    sum(self.metrics.pattern_authenticity_scores) / \n                    len(self.metrics.pattern_authenticity_scores) >= 0.87\n                ) if self.metrics.pattern_authenticity_scores else False\n            }\n        }\n    \n    def log_performance_report(self):\n        \"\"\"Log detailed performance report.\"\"\"\n        summary = self.get_performance_summary()\n        \n        self.logger.info(\"📊 BMAD PERFORMANCE REPORT\")\n        self.logger.info(\"=\" * 50)\n        \n        # Processing performance\n        proc = summary['processing_performance']\n        self.logger.info(f\"Processing Performance:\")\n        self.logger.info(f\"  Average session time: {proc['average_session_time']:.2f}s\")\n        self.logger.info(f\"  Sub-3s compliance: {proc['sub_3s_compliance']:.1%}\")\n        self.logger.info(f\"  Sessions completed: {proc['total_sessions']}\")\n        self.logger.info(f\"  Patterns processed: {proc['total_patterns']}\")\n        \n        # Memory performance\n        mem = summary['memory_performance']\n        self.logger.info(f\"Memory Performance:\")\n        self.logger.info(f\"  Current memory: {mem['current_memory_mb']:.1f}MB\")\n        self.logger.info(f\"  Peak memory: {mem['peak_memory_mb']:.1f}MB\")\n        self.logger.info(f\"  Memory efficiency: {mem['memory_efficiency']:.1%}\")\n        \n        # Quality metrics\n        qual = summary['quality_metrics']\n        self.logger.info(f\"Quality Metrics:\")\n        self.logger.info(f\"  Average authenticity: {qual['average_authenticity']:.1%}\")\n        self.logger.info(f\"  Metamorphosis detections: {qual['metamorphosis_detections']}\")\n        self.logger.info(f\"  Average detection strength: {qual['average_detection_strength']:.1%}\")\n        \n        # Compliance status\n        comp = summary['compliance']\n        self.logger.info(f\"Compliance Status:\")\n        self.logger.info(f\"  Session processing: {'✅' if comp['session_processing_target'] else '❌'}\")\n        self.logger.info(f\"  Memory usage: {'✅' if comp['memory_target'] else '❌'}\")\n        self.logger.info(f\"  Pattern quality: {'✅' if comp['quality_target'] else '❌'}\")\n    \n    def _get_current_memory(self) -> float:\n        \"\"\"Get current memory usage in MB.\"\"\"\n        try:\n            process = psutil.Process()\n            return process.memory_info().rss / (1024 * 1024)\n        except:\n            return 0.0\n    \n    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):\n        \"\"\"Trigger performance alert.\"\"\"\n        for callback in self.alert_callbacks:\n            try:\n                callback(alert_type, data)\n            except Exception as e:\n                self.logger.error(f\"Alert callback failed: {e}\")\n    \n    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):\n        \"\"\"Add performance alert callback.\"\"\"\n        self.alert_callbacks.append(callback)\n    \n    def force_garbage_collection(self):\n        \"\"\"Force garbage collection to optimize memory usage.\"\"\"\n        collected = gc.collect()\n        current_memory = self._get_current_memory()\n        self.logger.debug(f\"Garbage collection: {collected} objects, {current_memory:.1f}MB memory\")\n\n\n# Global performance tracker instance\n_global_tracker: Optional[BMadPerformanceTracker] = None\n\n\ndef get_performance_tracker() -> BMadPerformanceTracker:\n    \"\"\"Get global BMAD performance tracker instance.\"\"\"\n    global _global_tracker\n    if _global_tracker is None:\n        _global_tracker = BMadPerformanceTracker()\n        _global_tracker.start_monitoring()\n    return _global_tracker\n\n\ndef initialize_performance_monitoring():\n    \"\"\"Initialize BMAD performance monitoring system.\"\"\"\n    tracker = get_performance_tracker()\n    \n    # Set up default alert handlers\n    def default_alert_handler(alert_type: str, data: Dict[str, Any]):\n        logger.warning(f\"Performance Alert [{alert_type}]: {data}\")\n    \n    tracker.add_alert_callback(default_alert_handler)\n    logger.info(\"📊 BMAD Performance Monitoring initialized\")\n\n\ndef shutdown_performance_monitoring():\n    \"\"\"Shutdown BMAD performance monitoring system.\"\"\"\n    global _global_tracker\n    if _global_tracker:\n        _global_tracker.stop_monitoring()\n        _global_tracker.log_performance_report()\n        _global_tracker = None\n    logger.info(\"📊 BMAD Performance Monitoring shutdown complete\")\n