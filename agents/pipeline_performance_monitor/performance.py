"""
IRONFORGE Pipeline Performance Monitor Self-Monitoring

Self-monitoring and optimization module for the performance monitoring agent itself.
Ensures that the monitoring system maintains sub-millisecond overhead while
providing comprehensive performance insights.

Self-Monitoring Contracts:
- Monitoring overhead: <1ms per operation (sub-millisecond impact)
- Agent initialization: <500ms (faster than main system initialization)
- Memory footprint: <10MB (minimal impact on system resources)
- Data collection efficiency: >99.9% success rate
- Real-time responsiveness: <100ms dashboard updates
"""

import time
import threading
import statistics
import gc
import sys
import traceback
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import psutil
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class SelfMonitoringMetrics:
    """Metrics for self-monitoring the performance monitoring agent."""
    
    # Timing metrics (microsecond precision)
    operation_times: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=1000)))
    overhead_measurements: deque = field(default_factory=lambda: deque(maxlen=10000))
    
    # Memory and resource metrics
    agent_memory_usage: deque = field(default_factory=lambda: deque(maxlen=1000))
    cpu_usage_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Data collection metrics
    data_collection_successes: int = 0
    data_collection_failures: int = 0
    data_collection_errors: List[str] = field(default_factory=list)
    
    # Dashboard and reporting metrics
    dashboard_update_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    alert_processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Error tracking
    exception_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    critical_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        times = list(self.operation_times.get(operation, []))
        if not times:
            return {'count': 0, 'avg': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
        return {
            'count': len(times),
            'avg': statistics.mean(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0,
            'p95': sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times),
            'p99': sorted(times)[int(len(times) * 0.99)] if len(times) >= 100 else max(times)
        }
    
    def get_overhead_stats(self) -> Dict[str, float]:
        """Get monitoring overhead statistics."""
        if not self.overhead_measurements:
            return {'avg_overhead_ms': 0.0, 'max_overhead_ms': 0.0, 'samples': 0}
        
        overheads_ms = [t * 1000 for t in self.overhead_measurements]  # Convert to milliseconds
        
        return {
            'avg_overhead_ms': statistics.mean(overheads_ms),
            'max_overhead_ms': max(overheads_ms),
            'p95_overhead_ms': sorted(overheads_ms)[int(len(overheads_ms) * 0.95)] if len(overheads_ms) >= 20 else max(overheads_ms),
            'samples': len(overheads_ms),
            'sub_millisecond_compliance': sum(1 for t in overheads_ms if t < 1.0) / len(overheads_ms)
        }
    
    def get_success_rate(self) -> float:
        """Get data collection success rate."""
        total = self.data_collection_successes + self.data_collection_failures
        if total == 0:
            return 1.0
        return self.data_collection_successes / total


class SelfPerformanceMonitor:
    """
    Self-Performance Monitor for the IRONFORGE Performance Monitoring Agent
    
    Monitors the performance monitoring system itself to ensure it operates
    with minimal overhead while maintaining comprehensive monitoring capabilities.
    """
    
    def __init__(self):
        self.metrics = SelfMonitoringMetrics()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance contracts for self-monitoring
        self.contracts = {
            'max_overhead_ms': 1.0,           # Sub-millisecond overhead requirement
            'max_agent_memory_mb': 10.0,      # Minimal memory footprint
            'min_success_rate': 0.999,        # >99.9% success rate
            'max_dashboard_update_ms': 100.0, # Real-time responsiveness
            'max_initialization_ms': 500.0    # Fast initialization
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
        
        # Performance optimization features
        self.adaptive_sampling = True
        self.sampling_rate = 1.0  # Start with full sampling
        self.overhead_budget_ms = 0.5  # Target overhead budget
        
        # Alert thresholds
        self.performance_alerts: List[Callable[[str, Dict[str, Any]], None]] = []
        
        self.logger.info("🔧 Self-Performance Monitor initialized")
    
    def start_self_monitoring(self):
        """Start self-monitoring of the performance monitoring agent."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._self_monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("📊 Self-performance monitoring started")
    
    def stop_self_monitoring(self):
        """Stop self-monitoring and generate final report."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Generate final self-monitoring report
        self._generate_final_report()
        self.logger.info("📊 Self-performance monitoring stopped")
    
    @contextmanager
    def track_operation(self, operation_name: str, measure_overhead: bool = True):
        """
        Context manager for tracking operation performance with minimal overhead.
        
        Args:
            operation_name: Name of the operation being tracked
            measure_overhead: Whether to measure monitoring overhead
        """
        # Use high-precision timing
        start_time = time.perf_counter()
        overhead_start = None
        
        try:
            if measure_overhead:
                overhead_start = time.perf_counter()
            
            yield
            
        except Exception as e:
            # Track exceptions with minimal overhead
            exception_type = type(e).__name__
            self.metrics.exception_counts[exception_type] += 1
            
            # Store critical errors for analysis
            if len(self.metrics.critical_errors) < 100:  # Limit memory usage
                self.metrics.critical_errors.append({
                    'operation': operation_name,
                    'exception_type': exception_type,
                    'message': str(e)[:200],  # Truncate long messages
                    'timestamp': datetime.now().isoformat()
                })
            
            self.metrics.data_collection_failures += 1
            raise
            
        finally:
            # Calculate operation time
            operation_time = time.perf_counter() - start_time
            
            # Use adaptive sampling to reduce overhead
            if self._should_sample():
                self.metrics.operation_times[operation_name].append(operation_time)
                
                # Measure monitoring overhead
                if measure_overhead and overhead_start:
                    overhead_time = time.perf_counter() - overhead_start - operation_time
                    if overhead_time > 0:  # Only record positive overhead
                        self.metrics.overhead_measurements.append(overhead_time)
                
                self.metrics.data_collection_successes += 1
                
                # Check for performance alerts
                self._check_performance_alerts(operation_name, operation_time)
    
    def track_dashboard_update(self, update_time: float):
        """Track dashboard update performance."""
        self.metrics.dashboard_update_times.append(update_time)
        
        if update_time > self.contracts['max_dashboard_update_ms'] / 1000:
            self._trigger_alert('dashboard_slow', {
                'update_time_ms': update_time * 1000,
                'threshold_ms': self.contracts['max_dashboard_update_ms']
            })
    
    def track_alert_processing(self, processing_time: float):
        """Track alert processing performance."""
        self.metrics.alert_processing_times.append(processing_time)
    
    def _should_sample(self) -> bool:
        """Determine if current operation should be sampled (adaptive sampling)."""
        if not self.adaptive_sampling:
            return True
        
        # Reduce sampling if overhead is too high
        overhead_stats = self.metrics.get_overhead_stats()
        if overhead_stats['avg_overhead_ms'] > self.overhead_budget_ms:
            # Dynamically adjust sampling rate
            self.sampling_rate = max(0.1, self.sampling_rate * 0.9)
        elif overhead_stats['avg_overhead_ms'] < self.overhead_budget_ms * 0.5:
            # Increase sampling if overhead is low
            self.sampling_rate = min(1.0, self.sampling_rate * 1.1)
        
        return time.time() % 1.0 < self.sampling_rate
    
    def _self_monitoring_loop(self):
        """Background loop for continuous self-monitoring."""
        while self.monitoring_active:
            try:
                with self.track_operation('self_monitoring_cycle', measure_overhead=False):
                    # Sample agent memory usage
                    memory_info = self.process.memory_info()
                    agent_memory_mb = memory_info.rss / (1024 * 1024)
                    self.metrics.agent_memory_usage.append(agent_memory_mb)
                    
                    # Sample CPU usage
                    cpu_percent = self.process.cpu_percent()
                    self.metrics.cpu_usage_samples.append(cpu_percent)
                    
                    # Check memory contract
                    if agent_memory_mb > self.contracts['max_agent_memory_mb']:
                        self._trigger_alert('agent_memory_high', {
                            'current_mb': agent_memory_mb,
                            'limit_mb': self.contracts['max_agent_memory_mb']
                        })
                    
                    # Optimize sampling rate based on performance
                    self._optimize_sampling_rate()
                    
                    # Perform garbage collection if memory is growing
                    if len(self.metrics.agent_memory_usage) >= 10:
                        recent_memory = list(self.metrics.agent_memory_usage)[-10:]
                        if recent_memory[-1] > recent_memory[0] * 1.2:  # 20% growth
                            gc.collect()
                
                time.sleep(1.0)  # Sample every second
                
            except Exception as e:
                self.logger.error(f"Self-monitoring loop error: {e}")
                time.sleep(5.0)  # Back off on errors
    
    def _optimize_sampling_rate(self):
        """Optimize sampling rate based on overhead measurements."""
        overhead_stats = self.metrics.get_overhead_stats()
        
        if overhead_stats['samples'] < 100:
            return  # Not enough data for optimization
        
        avg_overhead = overhead_stats['avg_overhead_ms']
        
        # Target: keep average overhead below budget
        if avg_overhead > self.overhead_budget_ms:
            reduction_factor = self.overhead_budget_ms / avg_overhead
            self.sampling_rate *= reduction_factor
            self.sampling_rate = max(0.01, self.sampling_rate)  # Minimum 1% sampling
            
            self.logger.debug(f"Reduced sampling rate to {self.sampling_rate:.2%} (overhead: {avg_overhead:.3f}ms)")
        
        elif avg_overhead < self.overhead_budget_ms * 0.5 and self.sampling_rate < 1.0:
            # Can afford to increase sampling
            self.sampling_rate = min(1.0, self.sampling_rate * 1.1)
            self.logger.debug(f"Increased sampling rate to {self.sampling_rate:.2%} (overhead: {avg_overhead:.3f}ms)")
    
    def _check_performance_alerts(self, operation_name: str, operation_time: float):
        """Check if operation performance triggers alerts."""
        # Convert to milliseconds for comparison
        operation_time_ms = operation_time * 1000
        
        # Check for slow operations
        operation_thresholds = {
            'stage_discovery': 100.0,    # 100ms threshold for stage monitoring
            'stage_confluence': 50.0,    # 50ms threshold
            'stage_validation': 25.0,    # 25ms threshold  
            'stage_reporting': 75.0,     # 75ms threshold
            'contract_validation': 10.0, # 10ms threshold
            'data_collection': 5.0,      # 5ms threshold
            'analysis_operation': 20.0   # 20ms threshold for analysis
        }
        
        threshold = operation_thresholds.get(operation_name, 50.0)  # Default 50ms
        
        if operation_time_ms > threshold:
            self._trigger_alert('operation_slow', {
                'operation': operation_name,
                'time_ms': operation_time_ms,
                'threshold_ms': threshold
            })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger self-monitoring performance alert."""
        alert_data = {
            'type': alert_type,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'agent': 'self_monitoring'
        }
        
        self.logger.warning(f"🔧 Self-Monitoring Alert [{alert_type}]: {data}")
        
        # Call alert callbacks with minimal overhead
        for callback in self.performance_alerts:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                self.logger.error(f"Self-monitoring alert callback failed: {e}")
    
    def add_performance_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for self-monitoring performance alerts."""
        self.performance_alerts.append(callback)
    
    def get_self_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive self-monitoring performance report."""
        overhead_stats = self.metrics.get_overhead_stats()
        
        # Memory statistics
        memory_stats = {}
        if self.metrics.agent_memory_usage:
            memory_values = list(self.metrics.agent_memory_usage)
            memory_stats = {
                'current_mb': memory_values[-1] if memory_values else 0,
                'average_mb': statistics.mean(memory_values),
                'peak_mb': max(memory_values),
                'samples': len(memory_values)
            }
        
        # CPU statistics
        cpu_stats = {}
        if self.metrics.cpu_usage_samples:
            cpu_values = list(self.metrics.cpu_usage_samples)
            cpu_stats = {
                'average_percent': statistics.mean(cpu_values),
                'peak_percent': max(cpu_values),
                'samples': len(cpu_values)
            }
        
        # Operation performance breakdown
        operation_performance = {}
        for operation, times in self.metrics.operation_times.items():
            if times:
                operation_performance[operation] = self.metrics.get_operation_stats(operation)
        
        # Dashboard performance
        dashboard_stats = {}
        if self.metrics.dashboard_update_times:
            update_times_ms = [t * 1000 for t in self.metrics.dashboard_update_times]
            dashboard_stats = {
                'average_update_ms': statistics.mean(update_times_ms),
                'max_update_ms': max(update_times_ms),
                'updates_count': len(update_times_ms),
                'real_time_compliance': sum(1 for t in update_times_ms if t <= 100) / len(update_times_ms)
            }
        
        # Contract compliance
        compliance = {
            'overhead_compliance': overhead_stats['avg_overhead_ms'] <= self.contracts['max_overhead_ms'],
            'memory_compliance': memory_stats.get('peak_mb', 0) <= self.contracts['max_agent_memory_mb'],
            'success_rate_compliance': self.metrics.get_success_rate() >= self.contracts['min_success_rate'],
            'dashboard_compliance': dashboard_stats.get('average_update_ms', 0) <= self.contracts['max_dashboard_update_ms']
        }
        
        return {
            'monitoring_overhead': overhead_stats,
            'memory_usage': memory_stats,
            'cpu_usage': cpu_stats,
            'operation_performance': operation_performance,
            'dashboard_performance': dashboard_stats,
            'data_collection': {
                'success_rate': self.metrics.get_success_rate(),
                'successes': self.metrics.data_collection_successes,
                'failures': self.metrics.data_collection_failures,
                'error_types': dict(self.metrics.exception_counts)
            },
            'contract_compliance': compliance,
            'adaptive_sampling': {
                'enabled': self.adaptive_sampling,
                'current_rate': self.sampling_rate,
                'overhead_budget_ms': self.overhead_budget_ms
            },
            'health_summary': {
                'overall_healthy': all(compliance.values()),
                'critical_errors': len(self.metrics.critical_errors),
                'monitoring_active': self.monitoring_active
            }
        }
    
    def _generate_final_report(self):
        """Generate and log final self-monitoring report."""
        report = self.get_self_monitoring_report()
        
        self.logger.info("🔧 SELF-MONITORING FINAL REPORT")
        self.logger.info("=" * 50)
        
        # Overhead performance
        overhead = report['monitoring_overhead']
        compliance_icon = "✅" if overhead['avg_overhead_ms'] <= self.contracts['max_overhead_ms'] else "❌"
        self.logger.info(f"Monitoring Overhead: {compliance_icon}")
        self.logger.info(f"  Average: {overhead['avg_overhead_ms']:.3f}ms (target: <{self.contracts['max_overhead_ms']}ms)")
        self.logger.info(f"  Peak: {overhead['max_overhead_ms']:.3f}ms")
        self.logger.info(f"  Sub-ms compliance: {overhead['sub_millisecond_compliance']:.1%}")
        
        # Memory performance
        memory = report['memory_usage']
        if memory:
            memory_compliance_icon = "✅" if memory['peak_mb'] <= self.contracts['max_agent_memory_mb'] else "❌"
            self.logger.info(f"Memory Usage: {memory_compliance_icon}")
            self.logger.info(f"  Current: {memory['current_mb']:.1f}MB")
            self.logger.info(f"  Peak: {memory['peak_mb']:.1f}MB (limit: {self.contracts['max_agent_memory_mb']}MB)")
        
        # Data collection performance
        data_collection = report['data_collection']
        success_compliance_icon = "✅" if data_collection['success_rate'] >= self.contracts['min_success_rate'] else "❌"
        self.logger.info(f"Data Collection: {success_compliance_icon}")
        self.logger.info(f"  Success rate: {data_collection['success_rate']:.1%} (target: >{self.contracts['min_success_rate']:.1%})")
        self.logger.info(f"  Operations: {data_collection['successes']} success, {data_collection['failures']} failures")
        
        # Dashboard performance
        dashboard = report['dashboard_performance']
        if dashboard:
            dashboard_compliance_icon = "✅" if dashboard['average_update_ms'] <= self.contracts['max_dashboard_update_ms'] else "❌"
            self.logger.info(f"Dashboard Updates: {dashboard_compliance_icon}")
            self.logger.info(f"  Average: {dashboard['average_update_ms']:.1f}ms (target: <{self.contracts['max_dashboard_update_ms']}ms)")
            self.logger.info(f"  Real-time compliance: {dashboard['real_time_compliance']:.1%}")
        
        # Overall health
        health = report['health_summary']
        overall_icon = "✅" if health['overall_healthy'] else "❌"
        self.logger.info(f"Overall Health: {overall_icon}")
        
        # Adaptive sampling performance
        sampling = report['adaptive_sampling']
        self.logger.info(f"Adaptive Sampling:")
        self.logger.info(f"  Current rate: {sampling['current_rate']:.1%}")
        self.logger.info(f"  Overhead budget: {sampling['overhead_budget_ms']}ms")
        
        # Top performing operations
        if report['operation_performance']:
            self.logger.info("Top Operation Performance:")
            sorted_ops = sorted(report['operation_performance'].items(), 
                              key=lambda x: x[1]['avg'], reverse=True)
            for op_name, stats in sorted_ops[:5]:
                self.logger.info(f"  {op_name}: {stats['avg']*1000:.1f}ms avg ({stats['count']} samples)")
        
        if health['overall_healthy']:
            self.logger.info("🎉 Self-monitoring performance contracts satisfied")
        else:
            self.logger.warning("⚠️  Some self-monitoring contracts not met")
        
        self.logger.info("=" * 50)
    
    def optimize_for_production(self):
        """Optimize self-monitoring settings for production deployment."""
        self.logger.info("🚀 Optimizing self-monitoring for production")
        
        # More conservative overhead budget for production
        self.overhead_budget_ms = 0.3  # Even stricter 0.3ms budget
        
        # Enable adaptive sampling with production settings
        self.adaptive_sampling = True
        self.sampling_rate = 0.5  # Start with 50% sampling
        
        # Limit memory usage of stored metrics
        for operation_times in self.metrics.operation_times.values():
            operation_times.maxlen = 500  # Reduce from 1000
        
        self.metrics.overhead_measurements.maxlen = 5000  # Reduce from 10000
        self.metrics.agent_memory_usage.maxlen = 500
        self.metrics.cpu_usage_samples.maxlen = 500
        
        # Limit error storage
        if len(self.metrics.critical_errors) > 50:
            self.metrics.critical_errors = self.metrics.critical_errors[-50:]
        
        self.logger.info("✅ Self-monitoring optimized for production deployment")
    
    def validate_self_performance(self) -> bool:
        """
        Validate that self-monitoring meets all performance contracts.
        
        Returns:
            bool: True if all contracts are met, False otherwise
        """
        report = self.get_self_monitoring_report()
        compliance = report['contract_compliance']
        
        all_compliant = all(compliance.values())
        
        if not all_compliant:
            self.logger.error("❌ Self-monitoring performance contracts violated:")
            for contract, compliant in compliance.items():
                if not compliant:
                    self.logger.error(f"  - {contract}: FAILED")
        else:
            self.logger.info("✅ All self-monitoring performance contracts satisfied")
        
        return all_compliant
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing or fresh starts)."""
        self.metrics = SelfMonitoringMetrics()
        self.sampling_rate = 1.0
        self.logger.info("🔄 Self-monitoring metrics reset")


# Global self-monitoring instance
_global_self_monitor: Optional[SelfPerformanceMonitor] = None


def get_self_monitor() -> SelfPerformanceMonitor:
    """Get or create global self-monitoring instance."""
    global _global_self_monitor
    if _global_self_monitor is None:
        _global_self_monitor = SelfPerformanceMonitor()
        _global_self_monitor.start_self_monitoring()
    return _global_self_monitor


def initialize_self_monitoring():
    """Initialize global self-monitoring system."""
    monitor = get_self_monitor()
    
    # Set up default alert handler
    def default_self_alert_handler(alert_type: str, data: Dict[str, Any]):
        logger.warning(f"Self-Monitoring Alert [{alert_type}]: {data}")
    
    monitor.add_performance_alert_callback(default_self_alert_handler)
    logger.info("🔧 Self-performance monitoring initialized")


def shutdown_self_monitoring():
    """Shutdown self-monitoring system."""
    global _global_self_monitor
    if _global_self_monitor:
        _global_self_monitor.stop_self_monitoring()
        _global_self_monitor = None
    logger.info("🔧 Self-performance monitoring shutdown complete")


@contextmanager
def track_self_operation(operation_name: str):
    """Convenience context manager for tracking operations in self-monitoring."""
    monitor = get_self_monitor()
    with monitor.track_operation(operation_name):
        yield


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def test_self_monitoring():
        """Test self-monitoring functionality."""
        print("🔧 IRONFORGE Self-Performance Monitor Test")
        print("=" * 50)
        
        monitor = SelfPerformanceMonitor()
        monitor.start_self_monitoring()
        
        # Simulate various operations
        for i in range(100):
            with monitor.track_operation('test_operation'):
                # Simulate work with varying load
                time.sleep(0.001 * (i % 10 + 1))  # 1-10ms operations
        
        # Simulate dashboard updates
        for i in range(50):
            update_time = 0.05 + 0.02 * (i % 5)  # 50-130ms updates
            monitor.track_dashboard_update(update_time)
        
        # Wait for some self-monitoring cycles
        await asyncio.sleep(3.0)
        
        # Generate report
        report = monitor.get_self_monitoring_report()
        
        print(f"Monitoring Overhead: {report['monitoring_overhead']['avg_overhead_ms']:.3f}ms")
        print(f"Memory Usage: {report['memory_usage']['current_mb']:.1f}MB")
        print(f"Success Rate: {report['data_collection']['success_rate']:.1%}")
        
        # Validate performance
        is_compliant = monitor.validate_self_performance()
        print(f"Performance Compliant: {'✅ YES' if is_compliant else '❌ NO'}")
        
        monitor.stop_self_monitoring()
        print("\n✅ Self-monitoring test completed")
    
    asyncio.run(test_self_monitoring())