"""
IRONFORGE Performance Monitoring Framework
==========================================

Performance benchmarking and monitoring system to track system performance,
detect regressions, and ensure optimal operation.
"""

import time
import sys
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime
import json
from pathlib import Path
import gc
from contextlib import contextmanager

try:
    from config import get_config
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import get_config

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance monitoring and benchmarking system for IRONFORGE.

    Monitors:
    - Execution time performance
    - Memory usage patterns
    - GPU utilization (if available)
    - Component initialization times
    - Session processing throughput
    - Pattern discovery performance
    """

    def __init__(self):
        config = get_config()
        self.monitor_output_path = Path(config.get_reports_path()) / "performance"
        self.monitor_output_path.mkdir(parents=True, exist_ok=True)

        # Performance baselines (from iron-core integration)
        self.performance_baselines = {
            "container_init_seconds": 5.0,
            "component_creation_seconds": 3.0,
            "session_processing_seconds": 10.0,
            "memory_usage_mb": 100.0,
            "pattern_discovery_seconds": 15.0,
        }

        # Performance history for trend analysis
        self.performance_history = []

        logger.info("Performance Monitor initialized with SLA tracking")

    @contextmanager
    def monitor_execution(self, operation_name: str):
        """Context manager for monitoring operation performance"""

        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else None

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else None

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            gpu_memory_delta = (
                (end_gpu_memory - start_gpu_memory)
                if (start_gpu_memory and end_gpu_memory)
                else None
            )

            performance_data = {
                "operation": operation_name,
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "memory_usage_mb": end_memory,
                "memory_delta_mb": memory_delta,
                "gpu_memory_mb": end_gpu_memory,
                "gpu_memory_delta_mb": gpu_memory_delta,
            }

            self.performance_history.append(performance_data)
            logger.debug(
                f"Performance: {operation_name} took {execution_time:.3f}s, memory: {memory_delta:+.1f}MB"
            )

    def benchmark_system_performance(self) -> Dict[str, Any]:
        """
        Run comprehensive system performance benchmarks

        Returns:
            Complete performance benchmark results
        """
        try:
            logger.info("Starting comprehensive performance benchmarking...")

            benchmark_results = {
                "benchmark_timestamp": datetime.now().isoformat(),
                "system_info": self._get_system_info(),
                "container_benchmarks": {},
                "component_benchmarks": {},
                "session_processing_benchmarks": {},
                "memory_benchmarks": {},
                "performance_summary": {},
                "sla_compliance": {},
            }

            # Benchmark container initialization
            benchmark_results["container_benchmarks"] = self._benchmark_container_performance()

            # Benchmark component creation
            benchmark_results["component_benchmarks"] = self._benchmark_component_performance()

            # Benchmark session processing
            benchmark_results["session_processing_benchmarks"] = (
                self._benchmark_session_processing()
            )

            # Benchmark memory usage
            benchmark_results["memory_benchmarks"] = self._benchmark_memory_performance()

            # Generate performance summary
            benchmark_results["performance_summary"] = self._generate_performance_summary(
                benchmark_results
            )

            # Check SLA compliance
            benchmark_results["sla_compliance"] = self._check_sla_compliance(benchmark_results)

            # Save benchmark results
            self._save_benchmark_results(benchmark_results)

            logger.info(
                f"Performance benchmarking complete: {benchmark_results['sla_compliance']['overall_sla_met']}"
            )
            return benchmark_results

        except Exception as e:
            logger.error(f"Performance benchmarking failed: {e}")
            return {
                "benchmark_timestamp": datetime.now().isoformat(),
                "status": "ERROR",
                "error": str(e),
                "performance_summary": {"overall_performance": "BENCHMARK_FAILED"},
            }

    def _benchmark_container_performance(self) -> Dict[str, Any]:
        """Benchmark container initialization performance"""

        benchmarks = {}

        # Test container initialization time
        try:
            with self.monitor_execution("container_initialization"):
                from ironforge.integration.ironforge_container import get_ironforge_container

                container = get_ironforge_container()

            latest_perf = self.performance_history[-1]
            init_time = latest_perf["execution_time_seconds"]

            benchmarks["initialization"] = {
                "test": "container_initialization_time",
                "execution_time_seconds": init_time,
                "baseline_seconds": self.performance_baselines["container_init_seconds"],
                "within_baseline": init_time
                <= self.performance_baselines["container_init_seconds"],
                "performance_ratio": init_time
                / self.performance_baselines["container_init_seconds"],
                "result": (
                    "PASS"
                    if init_time <= self.performance_baselines["container_init_seconds"]
                    else "FAIL"
                ),
            }
        except Exception as e:
            benchmarks["initialization"] = {
                "test": "container_initialization_time",
                "result": "ERROR",
                "error": str(e),
            }

        # Test multiple container access (should reuse global instance)
        try:
            start_time = time.time()
            for _ in range(10):
                from ironforge.integration.ironforge_container import get_ironforge_container

                container = get_ironforge_container()
            multi_access_time = time.time() - start_time

            benchmarks["multi_access"] = {
                "test": "container_multi_access_performance",
                "total_time_seconds": multi_access_time,
                "average_time_seconds": multi_access_time / 10,
                "expected_fast_access": multi_access_time
                < 0.1,  # Should be very fast for cached access
                "result": "PASS" if multi_access_time < 0.1 else "FAIL",
            }
        except Exception as e:
            benchmarks["multi_access"] = {
                "test": "container_multi_access_performance",
                "result": "ERROR",
                "error": str(e),
            }

        return benchmarks

    def _benchmark_component_performance(self) -> Dict[str, Any]:
        """Benchmark component creation performance"""

        benchmarks = {}

        # Test component creation times
        components_to_test = ["enhanced_graph_builder", "tgat_discovery", "pattern_graduation"]

        for component_name in components_to_test:
            try:
                with self.monitor_execution(f"{component_name}_creation"):
                    from ironforge.integration.ironforge_container import get_ironforge_container

                    container = get_ironforge_container()

                    if component_name == "enhanced_graph_builder":
                        component = container.get_enhanced_graph_builder()
                    elif component_name == "tgat_discovery":
                        component = container.get_tgat_discovery()
                    elif component_name == "pattern_graduation":
                        component = container.get_pattern_graduation()

                latest_perf = self.performance_history[-1]
                creation_time = latest_perf["execution_time_seconds"]

                benchmarks[component_name] = {
                    "test": f"{component_name}_creation_time",
                    "execution_time_seconds": creation_time,
                    "baseline_seconds": self.performance_baselines["component_creation_seconds"],
                    "within_baseline": creation_time
                    <= self.performance_baselines["component_creation_seconds"],
                    "memory_usage_mb": latest_perf["memory_usage_mb"],
                    "memory_delta_mb": latest_perf["memory_delta_mb"],
                    "result": (
                        "PASS"
                        if creation_time <= self.performance_baselines["component_creation_seconds"]
                        else "FAIL"
                    ),
                }
            except Exception as e:
                benchmarks[component_name] = {
                    "test": f"{component_name}_creation_time",
                    "result": "ERROR",
                    "error": str(e),
                }

        return benchmarks

    def _benchmark_session_processing(self) -> Dict[str, Any]:
        """Benchmark session processing performance"""

        benchmarks = {}

        # Test graph building performance
        try:
            with self.monitor_execution("graph_building"):
                from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

                builder = EnhancedGraphBuilder()

                # Create mock session with realistic data size
                mock_session = {
                    "session_name": "performance_test_session",
                    "events": [
                        {
                            "timestamp": f"14:{35+i}:00",
                            "price": 23200.0 + i * 5,
                            "event_type": ["price_movement", "rebalance", "interaction"][i % 3],
                            "significance": 0.5 + (i % 5) * 0.1,
                        }
                        for i in range(50)  # 50 events for realistic load
                    ],
                    "relativity_stats": {
                        "session_high": 23350.0,
                        "session_low": 23150.0,
                        "session_open": 23200.0,
                    },
                }

                graph = builder.build_session_graph(mock_session)

            latest_perf = self.performance_history[-1]
            processing_time = latest_perf["execution_time_seconds"]

            benchmarks["graph_building"] = {
                "test": "session_graph_building_performance",
                "execution_time_seconds": processing_time,
                "baseline_seconds": self.performance_baselines["session_processing_seconds"],
                "within_baseline": processing_time
                <= self.performance_baselines["session_processing_seconds"],
                "events_processed": 50,
                "events_per_second": 50 / processing_time if processing_time > 0 else 0,
                "memory_delta_mb": latest_perf["memory_delta_mb"],
                "result": (
                    "PASS"
                    if processing_time <= self.performance_baselines["session_processing_seconds"]
                    else "FAIL"
                ),
            }
        except Exception as e:
            benchmarks["graph_building"] = {
                "test": "session_graph_building_performance",
                "result": "ERROR",
                "error": str(e),
            }

        # Test pattern graduation performance
        try:
            with self.monitor_execution("pattern_graduation"):
                from ironforge.synthesis.pattern_graduation import PatternGraduation

                graduation = PatternGraduation()

                # Create mock discovered patterns
                mock_patterns = {
                    "session_name": "performance_test",
                    "significant_patterns": [
                        {
                            "pattern_scores": [0.85, 0.87, 0.90],
                            "attention_received": 0.8,
                            "archaeological_significance": 0.88,
                            "confidence": 0.85,
                        }
                        for _ in range(10)
                    ],
                    "session_metrics": {"total_patterns": 10, "pattern_quality": 0.87},
                }

                results = graduation.validate_patterns(mock_patterns)

            latest_perf = self.performance_history[-1]
            graduation_time = latest_perf["execution_time_seconds"]

            benchmarks["pattern_graduation"] = {
                "test": "pattern_graduation_performance",
                "execution_time_seconds": graduation_time,
                "baseline_seconds": 2.0,  # Pattern graduation should be very fast
                "within_baseline": graduation_time <= 2.0,
                "patterns_processed": 10,
                "patterns_per_second": 10 / graduation_time if graduation_time > 0 else 0,
                "result": "PASS" if graduation_time <= 2.0 else "FAIL",
            }
        except Exception as e:
            benchmarks["pattern_graduation"] = {
                "test": "pattern_graduation_performance",
                "result": "ERROR",
                "error": str(e),
            }

        return benchmarks

    def _benchmark_memory_performance(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""

        benchmarks = {}

        # Test memory usage during component lifecycle
        try:
            gc.collect()  # Clean up before measurement
            initial_memory = self._get_memory_usage()

            # Create and use components
            from ironforge.integration.ironforge_container import get_ironforge_container

            container = get_ironforge_container()
            builder = container.get_enhanced_graph_builder()
            graduation = container.get_pattern_graduation()

            # Process mock session
            mock_session = {
                "session_name": "memory_test_session",
                "events": [
                    {
                        "timestamp": "14:35:00",
                        "price": 23200.0,
                        "event_type": "price_movement",
                        "significance": 0.8,
                    }
                ],
                "relativity_stats": {
                    "session_high": 23250.0,
                    "session_low": 23150.0,
                    "session_open": 23200.0,
                },
            }

            graph = builder.build_session_graph(mock_session)
            peak_memory = self._get_memory_usage()

            # Clean up references
            del container, builder, graduation, graph, mock_session
            gc.collect()
            final_memory = self._get_memory_usage()

            memory_usage = peak_memory - initial_memory
            memory_cleanup = peak_memory - final_memory

            benchmarks["memory_lifecycle"] = {
                "test": "component_memory_lifecycle",
                "initial_memory_mb": initial_memory,
                "peak_memory_mb": peak_memory,
                "final_memory_mb": final_memory,
                "memory_usage_mb": memory_usage,
                "memory_cleanup_mb": memory_cleanup,
                "baseline_memory_mb": self.performance_baselines["memory_usage_mb"],
                "within_baseline": memory_usage <= self.performance_baselines["memory_usage_mb"],
                "cleanup_percentage": (
                    (memory_cleanup / memory_usage * 100) if memory_usage > 0 else 0
                ),
                "result": (
                    "PASS"
                    if memory_usage <= self.performance_baselines["memory_usage_mb"]
                    else "FAIL"
                ),
            }
        except Exception as e:
            benchmarks["memory_lifecycle"] = {
                "test": "component_memory_lifecycle",
                "result": "ERROR",
                "error": str(e),
            }

        # Test for memory leaks (multiple session processing)
        try:
            gc.collect()
            baseline_memory = self._get_memory_usage()

            # Process multiple sessions to test for leaks
            for i in range(5):
                from ironforge.integration.ironforge_container import get_ironforge_container

                container = get_ironforge_container()
                builder = container.get_enhanced_graph_builder()

                mock_session = {
                    "session_name": f"leak_test_session_{i}",
                    "events": [
                        {
                            "timestamp": "14:35:00",
                            "price": 23200.0 + i,
                            "event_type": "price_movement",
                            "significance": 0.8,
                        }
                    ],
                    "relativity_stats": {
                        "session_high": 23250.0,
                        "session_low": 23150.0,
                        "session_open": 23200.0,
                    },
                }

                graph = builder.build_session_graph(mock_session)
                del container, builder, graph, mock_session  # Clean up each iteration

            gc.collect()
            final_memory = self._get_memory_usage()
            memory_growth = final_memory - baseline_memory

            benchmarks["memory_leak_test"] = {
                "test": "memory_leak_detection",
                "baseline_memory_mb": baseline_memory,
                "final_memory_mb": final_memory,
                "memory_growth_mb": memory_growth,
                "sessions_processed": 5,
                "memory_growth_per_session_mb": memory_growth / 5,
                "acceptable_growth": memory_growth < 10.0,  # Less than 10MB growth acceptable
                "result": "PASS" if memory_growth < 10.0 else "FAIL",
            }
        except Exception as e:
            benchmarks["memory_leak_test"] = {
                "test": "memory_leak_detection",
                "result": "ERROR",
                "error": str(e),
            }

        return benchmarks

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def _get_gpu_memory(self) -> Optional[float]:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return None

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context"""

        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_available": torch.cuda.is_available(),
            "gpu_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }

    def _generate_performance_summary(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance summary"""

        summary = {
            "total_benchmarks": 0,
            "passed_benchmarks": 0,
            "failed_benchmarks": 0,
            "error_benchmarks": 0,
            "overall_performance_rating": "UNKNOWN",
            "performance_categories": {},
        }

        # Collect all benchmark results
        all_benchmarks = {}
        for category in [
            "container_benchmarks",
            "component_benchmarks",
            "session_processing_benchmarks",
            "memory_benchmarks",
        ]:
            if category in benchmark_results:
                all_benchmarks.update(benchmark_results[category])

        # Count results
        for benchmark_name, benchmark_data in all_benchmarks.items():
            summary["total_benchmarks"] += 1
            result = benchmark_data.get("result", "UNKNOWN")

            if result == "PASS":
                summary["passed_benchmarks"] += 1
            elif result == "FAIL":
                summary["failed_benchmarks"] += 1
            elif result == "ERROR":
                summary["error_benchmarks"] += 1

        # Calculate performance rating
        if summary["total_benchmarks"] > 0:
            pass_rate = summary["passed_benchmarks"] / summary["total_benchmarks"]

            if pass_rate >= 1.0:
                summary["overall_performance_rating"] = "EXCELLENT"
            elif pass_rate >= 0.8:
                summary["overall_performance_rating"] = "GOOD"
            elif pass_rate >= 0.6:
                summary["overall_performance_rating"] = "ACCEPTABLE"
            else:
                summary["overall_performance_rating"] = "POOR"

        summary["pass_rate"] = (
            summary["passed_benchmarks"] / summary["total_benchmarks"]
            if summary["total_benchmarks"] > 0
            else 0.0
        )

        return summary

    def _check_sla_compliance(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check SLA compliance against performance baselines"""

        sla_checks = {}

        # Container initialization SLA
        container_init = benchmark_results.get("container_benchmarks", {}).get("initialization", {})
        sla_checks["container_initialization"] = {
            "sla_met": container_init.get("within_baseline", False),
            "actual_time": container_init.get("execution_time_seconds", 0),
            "sla_baseline": self.performance_baselines["container_init_seconds"],
        }

        # Component creation SLA
        component_benchmarks = benchmark_results.get("component_benchmarks", {})
        component_sla_met = all(
            benchmark.get("within_baseline", False)
            for benchmark in component_benchmarks.values()
            if benchmark.get("result") != "ERROR"
        )
        sla_checks["component_creation"] = {
            "sla_met": component_sla_met,
            "sla_baseline": self.performance_baselines["component_creation_seconds"],
        }

        # Session processing SLA
        session_benchmarks = benchmark_results.get("session_processing_benchmarks", {})
        session_sla_met = all(
            benchmark.get("within_baseline", False)
            for benchmark in session_benchmarks.values()
            if benchmark.get("result") != "ERROR"
        )
        sla_checks["session_processing"] = {
            "sla_met": session_sla_met,
            "sla_baseline": self.performance_baselines["session_processing_seconds"],
        }

        # Memory usage SLA
        memory_benchmarks = benchmark_results.get("memory_benchmarks", {})
        memory_sla_met = all(
            benchmark.get("within_baseline", False) or benchmark.get("acceptable_growth", False)
            for benchmark in memory_benchmarks.values()
            if benchmark.get("result") != "ERROR"
        )
        sla_checks["memory_usage"] = {
            "sla_met": memory_sla_met,
            "sla_baseline": self.performance_baselines["memory_usage_mb"],
        }

        # Overall SLA compliance
        all_slas_met = all(check["sla_met"] for check in sla_checks.values())
        sla_checks["overall_sla_met"] = all_slas_met

        return sla_checks

    def _save_benchmark_results(self, benchmark_results: Dict[str, Any]):
        """Save benchmark results to file"""

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_benchmark_{timestamp}.json"
            filepath = self.monitor_output_path / filename

            with open(filepath, "w") as f:
                json.dump(benchmark_results, f, indent=2, default=str)

            logger.info(f"Performance benchmark results saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")

    def get_performance_history(
        self, operation_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get performance history with optional filtering"""

        if operation_filter:
            return [
                perf for perf in self.performance_history if operation_filter in perf["operation"]
            ]
        return self.performance_history.copy()

    def clear_performance_history(self):
        """Clear performance history"""
        self.performance_history.clear()
        logger.info("Performance history cleared")

    def get_monitor_summary(self) -> Dict[str, Any]:
        """Get summary of performance monitoring system"""

        return {
            "monitoring_framework": "IRONFORGE Performance Monitor",
            "performance_baselines": self.performance_baselines,
            "monitoring_metrics": [
                "execution_time",
                "memory_usage",
                "gpu_memory_usage",
                "component_lifecycle",
                "session_throughput",
            ],
            "benchmark_categories": [
                "container_performance",
                "component_performance",
                "session_processing_performance",
                "memory_performance",
            ],
            "output_path": str(self.monitor_output_path),
            "history_entries": len(self.performance_history),
        }
