"""
IRONFORGE Performance Analysis and Optimization Tools

Advanced toolset for analyzing pipeline performance, detecting bottlenecks,
and providing actionable optimization recommendations. These tools operate
with microsecond precision and provide deep insights into IRONFORGE's
archaeological discovery pipeline performance.

Key Capabilities:
- Bottleneck detection across all pipeline stages
- Memory usage analysis and optimization recommendations
- Container lazy loading performance analysis
- Quality vs performance trade-off analysis
- Automated optimization suggestions with impact estimates
"""

import time
import statistics
import gc
import threading
import psutil
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import logging

from .ironforge_config import IRONFORGEPerformanceConfig, StageThresholds

logger = logging.getLogger(__name__)


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks in the pipeline."""
    
    stage_name: str
    bottleneck_type: str  # 'timing', 'memory', 'quality', 'throughput'
    severity: str  # 'low', 'medium', 'high', 'critical'
    impact_score: float  # 0.0 to 1.0
    description: str
    root_cause: str
    recommendations: List[str] = field(default_factory=list)
    estimated_improvement: Optional[float] = None  # Expected performance gain (0.0 to 1.0)
    implementation_complexity: str = "medium"  # 'low', 'medium', 'high'


@dataclass
class OptimizationRecommendation:
    """Specific optimization recommendation with implementation details."""
    
    title: str
    description: str
    category: str  # 'memory', 'timing', 'quality', 'container', 'system'
    priority: str  # 'low', 'medium', 'high', 'critical'
    estimated_gain: float  # Expected performance improvement (0.0 to 1.0)
    implementation_effort: str  # 'low', 'medium', 'high'
    risk_level: str  # 'low', 'medium', 'high'
    dependencies: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)


@dataclass
class PerformanceProfile:
    """Comprehensive performance profile for analysis."""
    
    timestamp: datetime
    stage_timings: Dict[str, List[float]]
    memory_usage: List[float]
    quality_scores: List[float]
    error_counts: Dict[str, int]
    system_metrics: Dict[str, float]
    container_metrics: Dict[str, Any]


class PerformanceAnalysisTools:
    """
    Advanced performance analysis tools for IRONFORGE pipeline monitoring.
    
    Provides deep analysis capabilities for identifying bottlenecks,
    performance regressions, and optimization opportunities across
    all pipeline stages.
    """
    
    def __init__(self, config: IRONFORGEPerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Historical performance data for trend analysis
        self.performance_history: deque = deque(maxlen=1000)
        self.bottleneck_history: List[BottleneckAnalysis] = []
        
        # Analysis state
        self.last_analysis_time: Optional[datetime] = None
        self.current_analysis_thread: Optional[threading.Thread] = None
        
        # Benchmarking data
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.performance_targets = self._initialize_performance_targets()
    
    def _initialize_performance_targets(self) -> Dict[str, float]:
        """Initialize performance targets from configuration."""
        return {
            'discovery_seconds': self.config.stage_thresholds.discovery_seconds,
            'confluence_seconds': self.config.stage_thresholds.confluence_seconds,
            'validation_seconds': self.config.stage_thresholds.validation_seconds,
            'reporting_seconds': self.config.stage_thresholds.reporting_seconds,
            'session_processing_seconds': self.config.stage_thresholds.session_processing_seconds,
            'memory_limit_mb': self.config.stage_thresholds.memory_limit_mb,
            'authenticity_threshold': self.config.stage_thresholds.authenticity_threshold
        }
    
    def analyze_stage_performance(self, 
                                stage_name: str,
                                processing_times: List[float],
                                memory_snapshots: List[float],
                                quality_scores: List[float]) -> List[BottleneckAnalysis]:
        """
        Analyze performance for a specific pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            processing_times: List of processing times in seconds
            memory_snapshots: List of memory usage values in MB
            quality_scores: List of quality/authenticity scores
            
        Returns:
            List of identified bottlenecks and issues
        """
        bottlenecks = []
        
        if not processing_times:
            return bottlenecks
        
        # Timing analysis
        avg_time = statistics.mean(processing_times)
        target_time = getattr(self.config.stage_thresholds, f"{stage_name.lower()}_seconds", 60.0)
        
        if avg_time > target_time:
            severity = self._calculate_severity(avg_time / target_time)
            bottlenecks.append(BottleneckAnalysis(
                stage_name=stage_name,
                bottleneck_type='timing',
                severity=severity,
                impact_score=min(1.0, avg_time / target_time - 1.0),
                description=f"{stage_name} stage averaging {avg_time:.2f}s (target: {target_time:.2f}s)",
                root_cause=self._analyze_timing_root_cause(processing_times, target_time),
                recommendations=self._get_timing_recommendations(stage_name, avg_time, target_time),
                estimated_improvement=min(0.5, (avg_time - target_time) / avg_time),
                implementation_complexity="medium"
            ))
        
        # Memory analysis
        if memory_snapshots:
            max_memory = max(memory_snapshots)
            memory_limit = self.config.stage_thresholds.memory_limit_mb
            
            if max_memory > memory_limit:
                severity = self._calculate_severity(max_memory / memory_limit)
                bottlenecks.append(BottleneckAnalysis(
                    stage_name=stage_name,
                    bottleneck_type='memory',
                    severity=severity,
                    impact_score=min(1.0, max_memory / memory_limit - 1.0),
                    description=f"{stage_name} peak memory {max_memory:.1f}MB (limit: {memory_limit:.1f}MB)",
                    root_cause=self._analyze_memory_root_cause(memory_snapshots, memory_limit),
                    recommendations=self._get_memory_recommendations(stage_name, max_memory, memory_limit),
                    estimated_improvement=min(0.3, (max_memory - memory_limit) / max_memory),
                    implementation_complexity="low"
                ))
        
        # Quality analysis
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            quality_threshold = self.config.stage_thresholds.authenticity_threshold
            
            if avg_quality < quality_threshold:
                severity = self._calculate_severity((quality_threshold - avg_quality) / quality_threshold)
                bottlenecks.append(BottleneckAnalysis(
                    stage_name=stage_name,
                    bottleneck_type='quality',
                    severity=severity,
                    impact_score=min(1.0, (quality_threshold - avg_quality) / quality_threshold),
                    description=f"{stage_name} quality {avg_quality:.1%} (threshold: {quality_threshold:.1%})",
                    root_cause=self._analyze_quality_root_cause(quality_scores, quality_threshold),
                    recommendations=self._get_quality_recommendations(stage_name, avg_quality, quality_threshold),
                    estimated_improvement=min(0.4, (quality_threshold - avg_quality) / quality_threshold),
                    implementation_complexity="high"
                ))
        
        # Variability analysis
        if len(processing_times) > 3:
            time_stdev = statistics.stdev(processing_times)
            variability_ratio = time_stdev / avg_time if avg_time > 0 else 0
            
            if variability_ratio > 0.3:  # High variability
                bottlenecks.append(BottleneckAnalysis(
                    stage_name=stage_name,
                    bottleneck_type='throughput',
                    severity='medium',
                    impact_score=min(1.0, variability_ratio - 0.3),
                    description=f"{stage_name} high timing variability ({variability_ratio:.1%})",
                    root_cause="Inconsistent performance patterns suggest resource contention or inefficient algorithms",
                    recommendations=self._get_variability_recommendations(stage_name, variability_ratio),
                    estimated_improvement=0.2,
                    implementation_complexity="medium"
                ))
        
        return bottlenecks
    
    def detect_performance_regression(self, 
                                    current_metrics: Dict[str, List[float]],
                                    lookback_windows: int = 10) -> List[BottleneckAnalysis]:
        """
        Detect performance regression by comparing current metrics with historical data.
        
        Args:
            current_metrics: Current performance metrics by stage
            lookback_windows: Number of historical windows to compare against
            
        Returns:
            List of detected regressions
        """
        regressions = []
        
        if len(self.performance_history) < lookback_windows:
            return regressions  # Not enough history for regression analysis
        
        # Get recent historical data
        recent_history = list(self.performance_history)[-lookback_windows:-1]
        
        for stage_name, current_times in current_metrics.items():
            if not current_times:
                continue
            
            current_avg = statistics.mean(current_times)
            
            # Calculate historical average
            historical_times = []
            for profile in recent_history:
                if stage_name in profile.stage_timings:
                    historical_times.extend(profile.stage_timings[stage_name])
            
            if not historical_times:
                continue
            
            historical_avg = statistics.mean(historical_times)
            
            # Check for regression (current performance significantly worse than historical)
            regression_threshold = self.config.monitoring_settings.regression_detection_sensitivity
            if current_avg > historical_avg * (1 + regression_threshold):
                regression_severity = min(1.0, (current_avg - historical_avg) / historical_avg)
                
                regressions.append(BottleneckAnalysis(
                    stage_name=stage_name,
                    bottleneck_type='regression',
                    severity=self._calculate_severity(regression_severity + 1.0),
                    impact_score=regression_severity,
                    description=f"{stage_name} performance regression: {current_avg:.2f}s vs {historical_avg:.2f}s historical",
                    root_cause=f"Performance degraded by {((current_avg - historical_avg) / historical_avg):.1%} compared to recent history",
                    recommendations=self._get_regression_recommendations(stage_name, regression_severity),
                    estimated_improvement=min(0.6, regression_severity),
                    implementation_complexity="medium"
                ))
        
        return regressions
    
    def analyze_container_performance(self, 
                                    initialization_times: Dict[str, float],
                                    lazy_loading_metrics: Dict[str, Any]) -> List[BottleneckAnalysis]:
        """
        Analyze container system and lazy loading performance.
        
        Args:
            initialization_times: Component initialization times
            lazy_loading_metrics: Lazy loading performance data
            
        Returns:
            List of container-related bottlenecks
        """
        bottlenecks = []
        
        # Analyze initialization times
        total_init_time = sum(initialization_times.values())
        init_target = self.config.stage_thresholds.initialization_seconds
        
        if total_init_time > init_target:
            slow_components = [(name, time_taken) for name, time_taken in initialization_times.items() 
                             if time_taken > init_target * 0.2]  # Components taking >20% of target time
            
            bottlenecks.append(BottleneckAnalysis(
                stage_name='container',
                bottleneck_type='timing',
                severity=self._calculate_severity(total_init_time / init_target),
                impact_score=min(1.0, (total_init_time - init_target) / init_target),
                description=f"Container initialization {total_init_time:.3f}s (target: {init_target:.3f}s)",
                root_cause=f"Slow components: {', '.join([f'{name} ({time:.3f}s)' for name, time in slow_components])}",
                recommendations=self._get_container_recommendations(slow_components, total_init_time),
                estimated_improvement=min(0.5, (total_init_time - init_target) / total_init_time),
                implementation_complexity="low"
            ))
        
        # Analyze lazy loading efficiency
        if lazy_loading_metrics:
            cache_hit_rate = lazy_loading_metrics.get('cache_hit_rate', 0.0)
            avg_loading_time = lazy_loading_metrics.get('average_loading_time', 0.0)
            
            if cache_hit_rate < 0.8:  # Poor cache performance
                bottlenecks.append(BottleneckAnalysis(
                    stage_name='container',
                    bottleneck_type='throughput',
                    severity='medium',
                    impact_score=0.8 - cache_hit_rate,
                    description=f"Poor lazy loading cache hit rate: {cache_hit_rate:.1%}",
                    root_cause="Components being loaded repeatedly instead of cached efficiently",
                    recommendations=self._get_lazy_loading_recommendations(cache_hit_rate, avg_loading_time),
                    estimated_improvement=min(0.3, 0.8 - cache_hit_rate),
                    implementation_complexity="low"
                ))
        
        return bottlenecks
    
    def analyze_memory_patterns(self, memory_timeline: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Analyze memory usage patterns to detect leaks and optimization opportunities.
        
        Args:
            memory_timeline: List of (timestamp, memory_mb) tuples
            
        Returns:
            Dictionary containing memory analysis results
        """
        if len(memory_timeline) < 10:
            return {'status': 'insufficient_data'}
        
        timestamps, memory_values = zip(*memory_timeline)
        
        # Calculate memory growth trend
        time_diffs = [timestamps[i] - timestamps[0] for i in range(len(timestamps))]
        memory_trend = np.polyfit(time_diffs, memory_values, 1)[0]  # Linear trend slope
        
        # Detect memory spikes
        memory_mean = statistics.mean(memory_values)
        memory_std = statistics.stdev(memory_values) if len(memory_values) > 1 else 0
        spikes = [m for m in memory_values if m > memory_mean + 2 * memory_std]
        
        # Calculate memory efficiency
        baseline_memory = min(memory_values)
        peak_memory = max(memory_values)
        memory_range = peak_memory - baseline_memory
        
        analysis = {
            'status': 'analyzed',
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_range_mb': memory_range,
            'growth_trend_mb_per_second': memory_trend,
            'spike_count': len(spikes),
            'average_spike_size_mb': statistics.mean(spikes) - memory_mean if spikes else 0,
            'memory_efficiency_score': max(0.0, 1.0 - (memory_range / self.config.stage_thresholds.memory_limit_mb)),
            'potential_leak_detected': memory_trend > 0.1,  # Growing by >0.1MB per second
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        if analysis['potential_leak_detected']:
            analysis['recommendations'].append(
                "Memory leak detected: investigate object retention and garbage collection patterns"
            )
        
        if len(spikes) > len(memory_values) * 0.1:  # >10% spike rate
            analysis['recommendations'].append(
                "Frequent memory spikes detected: consider implementing memory pooling or batch processing"
            )
        
        if memory_range > self.config.stage_thresholds.memory_limit_mb * 0.5:
            analysis['recommendations'].append(
                "High memory variance detected: optimize memory allocation patterns and implement streaming processing"
            )
        
        return analysis
    
    def benchmark_against_baseline(self, current_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Benchmark current performance against established baseline.
        
        Args:
            current_metrics: Current performance metrics to benchmark
            
        Returns:
            Benchmarking results and performance comparison
        """
        if self.baseline_metrics is None:
            # Establish baseline from current metrics
            self.baseline_metrics = {
                stage: statistics.mean(times) for stage, times in current_metrics.items() if times
            }
            return {'status': 'baseline_established', 'baseline': self.baseline_metrics}
        
        comparison = {
            'status': 'compared',
            'improvements': {},
            'regressions': {},
            'overall_score': 0.0,
            'recommendations': []
        }
        
        stage_scores = []
        
        for stage_name, current_times in current_metrics.items():
            if not current_times or stage_name not in self.baseline_metrics:
                continue
            
            current_avg = statistics.mean(current_times)
            baseline_avg = self.baseline_metrics[stage_name]
            
            performance_ratio = current_avg / baseline_avg
            
            if performance_ratio < 0.95:  # Improved performance
                improvement = (1.0 - performance_ratio) * 100
                comparison['improvements'][stage_name] = {
                    'current_time': current_avg,
                    'baseline_time': baseline_avg,
                    'improvement_percent': improvement
                }
                stage_scores.append(1.0 + improvement / 100)  # Bonus for improvement
                
            elif performance_ratio > 1.05:  # Performance regression
                regression = (performance_ratio - 1.0) * 100
                comparison['regressions'][stage_name] = {
                    'current_time': current_avg,
                    'baseline_time': baseline_avg,
                    'regression_percent': regression
                }
                stage_scores.append(max(0.1, 1.0 - regression / 100))
                
            else:  # Stable performance
                stage_scores.append(1.0)
        
        # Calculate overall performance score
        comparison['overall_score'] = statistics.mean(stage_scores) if stage_scores else 1.0
        
        # Generate recommendations
        if comparison['regressions']:
            comparison['recommendations'].append(
                f"Address performance regressions in: {', '.join(comparison['regressions'].keys())}"
            )
        
        if comparison['overall_score'] < 0.8:
            comparison['recommendations'].append(
                "Overall performance significantly below baseline - conduct comprehensive optimization review"
            )
        
        return comparison
    
    def _calculate_severity(self, ratio: float) -> str:
        """Calculate severity level based on performance ratio."""
        if ratio < 1.1:
            return 'low'
        elif ratio < 1.3:
            return 'medium'
        elif ratio < 2.0:
            return 'high'
        else:
            return 'critical'
    
    def _analyze_timing_root_cause(self, processing_times: List[float], target_time: float) -> str:
        """Analyze root cause of timing bottlenecks."""
        if not processing_times:
            return "Insufficient timing data for analysis"
        
        avg_time = statistics.mean(processing_times)
        stdev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        
        if stdev / avg_time > 0.3:
            return "High variability suggests resource contention or inefficient algorithms"
        elif avg_time > target_time * 2:
            return "Severe performance degradation suggests architectural bottlenecks"
        else:
            return "Moderate performance issue likely due to suboptimal configuration or resource limits"
    
    def _analyze_memory_root_cause(self, memory_snapshots: List[float], memory_limit: float) -> str:
        """Analyze root cause of memory bottlenecks."""
        if not memory_snapshots:
            return "Insufficient memory data for analysis"
        
        max_memory = max(memory_snapshots)
        min_memory = min(memory_snapshots)
        memory_range = max_memory - min_memory
        
        if memory_range > memory_limit * 0.3:
            return "High memory variance suggests memory leaks or inefficient allocation patterns"
        else:
            return "Consistent high memory usage suggests oversized data structures or insufficient garbage collection"
    
    def _analyze_quality_root_cause(self, quality_scores: List[float], threshold: float) -> str:
        """Analyze root cause of quality bottlenecks."""
        if not quality_scores:
            return "Insufficient quality data for analysis"
        
        avg_quality = statistics.mean(quality_scores)
        min_quality = min(quality_scores)
        
        if min_quality < threshold * 0.5:
            return "Some patterns have very low quality scores - investigate pattern extraction algorithms"
        elif avg_quality < threshold * 0.9:
            return "Consistently low quality scores suggest parameter tuning needed"
        else:
            return "Quality scores near threshold - minor adjustments needed"
    
    def _get_timing_recommendations(self, stage_name: str, avg_time: float, target_time: float) -> List[str]:
        """Get recommendations for timing bottlenecks."""
        recommendations = []
        severity_ratio = avg_time / target_time
        
        if severity_ratio > 2.0:
            recommendations.extend([
                f"Critical timing issue in {stage_name} - consider architectural changes",
                "Profile code to identify computational bottlenecks",
                "Implement parallel processing where possible"
            ])
        elif severity_ratio > 1.5:
            recommendations.extend([
                f"Optimize {stage_name} algorithms and data structures",
                "Review memory allocation patterns for efficiency",
                "Consider caching frequently computed results"
            ])
        else:
            recommendations.extend([
                f"Fine-tune {stage_name} configuration parameters",
                "Optimize database queries and I/O operations",
                "Review lazy loading patterns"
            ])
        
        return recommendations
    
    def _get_memory_recommendations(self, stage_name: str, max_memory: float, memory_limit: float) -> List[str]:
        """Get recommendations for memory bottlenecks."""
        recommendations = []
        
        recommendations.extend([
            f"Implement memory optimization for {stage_name} stage",
            "Use streaming processing for large datasets",
            "Force garbage collection at appropriate intervals",
            "Review object lifetime and retention patterns"
        ])
        
        if max_memory > memory_limit * 1.5:
            recommendations.extend([
                "Consider implementing memory pooling",
                "Break processing into smaller batches",
                "Implement disk-based caching for large intermediate results"
            ])
        
        return recommendations
    
    def _get_quality_recommendations(self, stage_name: str, avg_quality: float, threshold: float) -> List[str]:
        """Get recommendations for quality bottlenecks."""
        return [
            f"Review {stage_name} quality parameters and thresholds",
            "Analyze pattern extraction algorithms for accuracy",
            "Consider adjusting feature engineering parameters",
            "Review training data quality and coverage",
            "Implement quality-aware filtering mechanisms"
        ]
    
    def _get_variability_recommendations(self, stage_name: str, variability_ratio: float) -> List[str]:
        """Get recommendations for performance variability."""
        return [
            f"Investigate {stage_name} performance variability sources",
            "Implement resource pooling to reduce contention",
            "Add performance monitoring at component level",
            "Consider implementing circuit breaker patterns",
            "Review thread safety and synchronization"
        ]
    
    def _get_regression_recommendations(self, stage_name: str, regression_severity: float) -> List[str]:
        """Get recommendations for performance regression."""
        recommendations = [
            f"Investigate recent changes affecting {stage_name} performance",
            "Compare current implementation with baseline version",
            "Review resource utilization and system health"
        ]
        
        if regression_severity > 0.3:
            recommendations.extend([
                "Consider rolling back recent changes if possible",
                "Conduct detailed performance profiling",
                "Review system dependencies and external factors"
            ])
        
        return recommendations
    
    def _get_container_recommendations(self, slow_components: List[Tuple[str, float]], total_time: float) -> List[str]:
        """Get recommendations for container performance."""
        recommendations = [
            "Optimize container initialization sequence",
            "Implement lazy loading for non-critical components",
            "Use dependency injection to reduce initialization overhead"
        ]
        
        if slow_components:
            component_names = [name for name, _ in slow_components]
            recommendations.append(f"Focus optimization on: {', '.join(component_names)}")
        
        return recommendations
    
    def _get_lazy_loading_recommendations(self, cache_hit_rate: float, avg_loading_time: float) -> List[str]:
        """Get recommendations for lazy loading optimization."""
        recommendations = [
            "Optimize component caching strategy",
            "Implement predictive loading for commonly used components",
            "Review component lifecycle management"
        ]
        
        if cache_hit_rate < 0.5:
            recommendations.append("Consider preloading frequently accessed components")
        
        if avg_loading_time > 0.1:
            recommendations.append("Optimize component loading logic and dependencies")
        
        return recommendations


class OptimizationRecommender:
    """
    Advanced optimization recommendation engine for IRONFORGE pipeline.
    
    Analyzes performance data and provides actionable optimization
    recommendations with impact estimates and implementation guidance.
    """
    
    def __init__(self, config: IRONFORGEPerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Optimization knowledge base
        self.optimization_patterns = self._initialize_optimization_patterns()
        self.current_recommendations: List[OptimizationRecommendation] = []
    
    def _initialize_optimization_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize optimization patterns and techniques."""
        return {
            'memory_optimization': {
                'pooling': {
                    'description': 'Implement object pooling for frequently created objects',
                    'impact': 0.3,
                    'complexity': 'medium',
                    'applicability': ['memory_spikes', 'frequent_allocation']
                },
                'streaming': {
                    'description': 'Use streaming processing for large datasets',
                    'impact': 0.5,
                    'complexity': 'high',
                    'applicability': ['large_datasets', 'memory_limit_exceeded']
                },
                'lazy_cleanup': {
                    'description': 'Implement lazy cleanup and garbage collection optimization',
                    'impact': 0.2,
                    'complexity': 'low',
                    'applicability': ['memory_growth', 'gc_pressure']
                }
            },
            'timing_optimization': {
                'caching': {
                    'description': 'Implement result caching for expensive operations',
                    'impact': 0.4,
                    'complexity': 'low',
                    'applicability': ['repeated_computation', 'slow_queries']
                },
                'parallelization': {
                    'description': 'Implement parallel processing for independent operations',
                    'impact': 0.6,
                    'complexity': 'high',
                    'applicability': ['cpu_bound', 'independent_tasks']
                },
                'batching': {
                    'description': 'Optimize batch sizes for better throughput',
                    'impact': 0.3,
                    'complexity': 'medium',
                    'applicability': ['small_batches', 'io_overhead']
                }
            },
            'container_optimization': {
                'preloading': {
                    'description': 'Preload critical components during initialization',
                    'impact': 0.4,
                    'complexity': 'low',
                    'applicability': ['slow_initialization', 'frequent_loading']
                },
                'dependency_optimization': {
                    'description': 'Optimize dependency resolution and injection',
                    'impact': 0.3,
                    'complexity': 'medium',
                    'applicability': ['complex_dependencies', 'initialization_overhead']
                }
            }
        }
    
    def analyze_pipeline_performance(self, 
                                   stage_metrics: Dict[str, Any],
                                   pipeline_results: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """
        Analyze complete pipeline performance and generate optimization recommendations.
        
        Args:
            stage_metrics: Performance metrics for each stage
            pipeline_results: Complete pipeline execution results
            
        Returns:
            List of prioritized optimization recommendations
        """
        recommendations = []
        
        # Analyze each stage for optimization opportunities
        for stage_name, metrics in stage_metrics.items():
            stage_recommendations = self._analyze_stage_optimization(stage_name, metrics)
            recommendations.extend(stage_recommendations)
        
        # Analyze overall pipeline optimization opportunities
        overall_recommendations = self._analyze_overall_optimization(pipeline_results)
        recommendations.extend(overall_recommendations)
        
        # Prioritize and filter recommendations
        recommendations = self._prioritize_recommendations(recommendations)
        
        # Update current recommendations
        self.current_recommendations = recommendations[:10]  # Keep top 10
        
        return recommendations
    
    def _analyze_stage_optimization(self, stage_name: str, metrics: Any) -> List[OptimizationRecommendation]:
        """Analyze optimization opportunities for a specific stage."""
        recommendations = []
        
        if not hasattr(metrics, 'processing_times') or not metrics.processing_times:
            return recommendations
        
        avg_time = statistics.mean(metrics.processing_times)
        target_time = getattr(self.config.stage_thresholds, f"{stage_name.lower()}_seconds", 60.0)
        
        # Timing optimization recommendations
        if avg_time > target_time * 1.2:
            if stage_name == 'discovery':
                recommendations.append(OptimizationRecommendation(
                    title=f"Optimize {stage_name} TGAT Processing",
                    description="Implement batch processing and attention computation optimization for TGAT discovery",
                    category='timing',
                    priority='high',
                    estimated_gain=min(0.4, (avg_time - target_time) / avg_time),
                    implementation_effort='medium',
                    risk_level='low',
                    implementation_steps=[
                        "Profile TGAT forward pass computation",
                        "Implement batch processing for multiple sessions",
                        "Optimize attention weight calculations",
                        "Add GPU acceleration if available"
                    ],
                    validation_criteria=[
                        f"Processing time reduced below {target_time}s",
                        "Pattern quality maintained >87%",
                        "Memory usage within limits"
                    ]
                ))
            
            elif stage_name == 'confluence':
                recommendations.append(OptimizationRecommendation(
                    title=f"Optimize {stage_name} Scoring Engine",
                    description="Implement parallel rule evaluation and caching for confluence scoring",
                    category='timing',
                    priority='medium',
                    estimated_gain=min(0.3, (avg_time - target_time) / avg_time),
                    implementation_effort='low',
                    risk_level='low',
                    implementation_steps=[
                        "Implement rule evaluation caching",
                        "Parallelize independent scoring calculations",
                        "Optimize weight application algorithms"
                    ],
                    validation_criteria=[
                        f"Scoring time reduced below {target_time}s",
                        "All confluence scores remain consistent"
                    ]
                ))
        
        # Memory optimization recommendations
        if hasattr(metrics, 'memory_snapshots') and metrics.memory_snapshots:
            max_memory = max(metrics.memory_snapshots)
            if max_memory > self.config.stage_thresholds.memory_limit_mb * 0.8:
                recommendations.append(OptimizationRecommendation(
                    title=f"Optimize {stage_name} Memory Usage",
                    description=f"Implement memory optimization to reduce {stage_name} memory footprint",
                    category='memory',
                    priority='medium',
                    estimated_gain=min(0.3, (max_memory - self.config.stage_thresholds.memory_limit_mb * 0.8) / max_memory),
                    implementation_effort='low',
                    risk_level='low',
                    implementation_steps=[
                        "Implement streaming processing for large datasets",
                        "Add periodic garbage collection",
                        "Optimize data structure sizes",
                        "Implement object pooling for frequently created objects"
                    ],
                    validation_criteria=[
                        f"Memory usage reduced below 80MB",
                        "No impact on processing performance",
                        "All functionality preserved"
                    ]
                ))
        
        return recommendations
    
    def _analyze_overall_optimization(self, pipeline_results: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze overall pipeline optimization opportunities."""
        recommendations = []
        
        overall_metrics = pipeline_results.get('overall_metrics', {})
        total_time = overall_metrics.get('total_processing_time', 0)
        target_time = self.config.stage_thresholds.full_discovery_seconds
        
        # Overall pipeline timing optimization
        if total_time > target_time * 1.1:
            recommendations.append(OptimizationRecommendation(
                title="Implement Pipeline Parallelization",
                description="Optimize pipeline stage execution with parallel processing where possible",
                category='timing',
                priority='high',
                estimated_gain=min(0.5, (total_time - target_time) / total_time),
                implementation_effort='high',
                risk_level='medium',
                dependencies=['container_optimization', 'threading_safety'],
                implementation_steps=[
                    "Analyze stage dependencies for parallelization opportunities",
                    "Implement async processing for independent operations",
                    "Add pipeline stage orchestration optimization",
                    "Implement result streaming between stages"
                ],
                validation_criteria=[
                    f"Total pipeline time reduced below {target_time}s",
                    "All quality gates continue to pass",
                    "No data integrity issues introduced"
                ]
            ))
        
        # Container system optimization
        if not overall_metrics.get('memory_compliance', True):
            recommendations.append(OptimizationRecommendation(
                title="Optimize Container Lazy Loading",
                description="Improve container system efficiency and lazy loading patterns",
                category='container',
                priority='medium',
                estimated_gain=0.2,
                implementation_effort='medium',
                risk_level='low',
                implementation_steps=[
                    "Analyze component loading patterns",
                    "Implement predictive loading for frequently used components",
                    "Optimize dependency injection performance",
                    "Add component pooling for reusable objects"
                ],
                validation_criteria=[
                    "Container initialization time <2s",
                    "Component loading cache hit rate >80%",
                    "Memory footprint reduced by 15%"
                ]
            ))
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Prioritize recommendations based on impact, effort, and risk."""
        def priority_score(rec: OptimizationRecommendation) -> float:
            # Priority mapping
            priority_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            effort_weights = {'low': 3, 'medium': 2, 'high': 1}
            risk_weights = {'low': 3, 'medium': 2, 'high': 1}
            
            priority_weight = priority_weights.get(rec.priority, 1)
            effort_weight = effort_weights.get(rec.implementation_effort, 1)
            risk_weight = risk_weights.get(rec.risk_level, 1)
            
            return (rec.estimated_gain * priority_weight * effort_weight * risk_weight)
        
        return sorted(recommendations, key=priority_score, reverse=True)
    
    def get_current_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations in dict format."""
        return [
            {
                'title': rec.title,
                'description': rec.description,
                'category': rec.category,
                'priority': rec.priority,
                'estimated_gain': f"{rec.estimated_gain:.1%}",
                'implementation_effort': rec.implementation_effort,
                'risk_level': rec.risk_level
            }
            for rec in self.current_recommendations
        ]
    
    def generate_optimization_plan(self, 
                                 selected_recommendations: List[str]) -> Dict[str, Any]:
        """
        Generate a comprehensive optimization plan based on selected recommendations.
        
        Args:
            selected_recommendations: List of recommendation titles to include
            
        Returns:
            Detailed optimization implementation plan
        """
        selected_recs = [rec for rec in self.current_recommendations 
                        if rec.title in selected_recommendations]
        
        if not selected_recs:
            return {'status': 'no_recommendations_selected'}
        
        # Analyze dependencies and create implementation sequence
        implementation_sequence = self._determine_implementation_sequence(selected_recs)
        
        # Estimate total impact and effort
        total_estimated_gain = min(0.8, sum(rec.estimated_gain for rec in selected_recs))
        
        effort_scores = {'low': 1, 'medium': 2, 'high': 3}
        total_effort = sum(effort_scores.get(rec.implementation_effort, 2) for rec in selected_recs)
        
        plan = {
            'status': 'plan_generated',
            'selected_optimizations': len(selected_recs),
            'total_estimated_gain': f"{total_estimated_gain:.1%}",
            'total_effort_score': total_effort,
            'estimated_implementation_time': self._estimate_implementation_time(selected_recs),
            'implementation_sequence': implementation_sequence,
            'risk_assessment': self._assess_plan_risk(selected_recs),
            'validation_plan': self._create_validation_plan(selected_recs),
            'rollback_strategy': self._create_rollback_strategy(selected_recs)
        }
        
        return plan
    
    def _determine_implementation_sequence(self, recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Determine optimal implementation sequence based on dependencies."""
        # Simple dependency-aware sorting (in practice, would use topological sort)
        sequence = []
        
        # First, implement low-risk, low-effort optimizations
        low_risk_recs = [rec for rec in recommendations if rec.risk_level == 'low' and rec.implementation_effort == 'low']
        sequence.extend([{'title': rec.title, 'phase': 'quick_wins', 'estimated_days': 1} for rec in low_risk_recs])
        
        # Then medium effort optimizations
        medium_recs = [rec for rec in recommendations if rec not in low_risk_recs and rec.implementation_effort == 'medium']
        sequence.extend([{'title': rec.title, 'phase': 'incremental', 'estimated_days': 3} for rec in medium_recs])
        
        # Finally high-effort optimizations
        high_effort_recs = [rec for rec in recommendations if rec.implementation_effort == 'high']
        sequence.extend([{'title': rec.title, 'phase': 'major', 'estimated_days': 7} for rec in high_effort_recs])
        
        return sequence
    
    def _estimate_implementation_time(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Estimate total implementation time."""
        effort_days = {'low': 1, 'medium': 3, 'high': 7}
        total_days = sum(effort_days.get(rec.implementation_effort, 3) for rec in recommendations)
        
        if total_days <= 3:
            return "1-3 days"
        elif total_days <= 7:
            return "1 week"
        elif total_days <= 14:
            return "2 weeks"
        else:
            return "3+ weeks"
    
    def _assess_plan_risk(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Assess overall risk of the optimization plan."""
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        for rec in recommendations:
            risk_counts[rec.risk_level] += 1
        
        if risk_counts['high'] > 0:
            overall_risk = 'high'
        elif risk_counts['medium'] > len(recommendations) // 2:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'overall_risk': overall_risk,
            'risk_distribution': risk_counts,
            'mitigation_strategies': [
                "Implement comprehensive testing for each optimization",
                "Use feature flags for gradual rollout",
                "Maintain detailed performance baselines",
                "Prepare rollback procedures for each change"
            ]
        }
    
    def _create_validation_plan(self, recommendations: List[OptimizationRecommendation]) -> List[str]:
        """Create comprehensive validation plan for optimizations."""
        return [
            "Run complete performance regression test suite",
            "Validate all performance contracts continue to be met",
            "Verify pattern quality scores remain >87%",
            "Confirm memory usage stays within 100MB limit",
            "Test with full 57-session discovery workload",
            "Monitor system stability for 24-hour period",
            "Validate container initialization remains <2s"
        ]
    
    def _create_rollback_strategy(self, recommendations: List[OptimizationRecommendation]) -> List[str]:
        """Create rollback strategy for optimization implementations."""
        return [
            "Maintain Git branches for each optimization phase",
            "Create automated rollback scripts for configuration changes",
            "Document all modified performance thresholds",
            "Prepare containerized rollback environments",
            "Create monitoring alerts for performance regression detection",
            "Establish go/no-go criteria for each implementation phase"
        ]


if __name__ == "__main__":
    # Example usage and testing
    from .ironforge_config import IRONFORGEPerformanceConfig
    
    config = IRONFORGEPerformanceConfig()
    analyzer = PerformanceAnalysisTools(config)
    optimizer = OptimizationRecommender(config)
    
    # Example analysis
    print("🔍 IRONFORGE Performance Analysis Tools")
    print("=" * 50)
    
    # Mock performance data for testing
    mock_stage_metrics = {
        'discovery': [5.2, 4.8, 5.5, 4.9, 5.1],  # Exceeds 3s target
        'confluence': [2.1, 1.9, 2.0, 2.2, 2.0],  # Within target
        'validation': [1.2, 1.1, 1.3, 1.0, 1.2],  # Within target
        'reporting': [8.5, 7.9, 8.2, 8.0, 8.1]    # Exceeds target
    }
    
    # Analyze each stage
    for stage_name, times in mock_stage_metrics.items():
        bottlenecks = analyzer.analyze_stage_performance(
            stage_name, times, [45.2, 48.1, 46.3, 47.0, 45.8], [0.89, 0.91, 0.88, 0.90, 0.89]
        )
        
        if bottlenecks:
            print(f"\n📊 {stage_name.title()} Stage Analysis:")
            for bottleneck in bottlenecks:
                print(f"  • {bottleneck.description} ({bottleneck.severity})")
                print(f"    Estimated improvement: {bottleneck.estimated_improvement:.1%}")
    
    print("\n✅ Performance analysis tools ready for production deployment")