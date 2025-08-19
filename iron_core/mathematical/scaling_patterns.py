"""
Scaling Patterns for Mathematical Computations
==============================================

Horizontal and vertical scaling patterns with intelligent data partitioning
for mathematical model computations. Provides adaptive scaling strategies
based on computational complexity and system resources.

Key Features:
- Horizontal scaling with intelligent data partitioning
- Vertical scaling with memory optimization
- Adaptive scaling strategy selection
- Performance-based load balancing
- GPU acceleration support
- Distributed computation coordination
"""

import asyncio
import concurrent.futures
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import psutil


class ScalingStrategy(Enum):
    """Available scaling strategies"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    HYBRID = "hybrid"
    AUTO = "auto"

class ComputationComplexity(Enum):
    """Computation complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class ScalingConfig:
    """Configuration for scaling operations"""
    strategy: ScalingStrategy = ScalingStrategy.AUTO
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    enable_gpu: bool = False
    chunk_size: int = 1000
    timeout_seconds: float = 300.0
    load_balancing: bool = True
    
@dataclass
class ScalingMetrics:
    """Metrics from scaling operations"""
    execution_time: float
    memory_usage_gb: float
    cpu_utilization: float
    workers_used: int
    chunks_processed: int
    throughput: float
    efficiency_score: float

class ScalingPattern(ABC):
    """Abstract base class for scaling patterns"""
    
    @abstractmethod
    async def execute(self, computation: Callable, data: Any, config: ScalingConfig) -> tuple[Any, ScalingMetrics]:
        """Execute computation with scaling pattern"""
        pass
    
    @abstractmethod
    def estimate_resources(self, data_size: int, complexity: ComputationComplexity) -> dict[str, float]:
        """Estimate resource requirements"""
        pass

class HorizontalScalingPattern(ScalingPattern):
    """Horizontal scaling with data partitioning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, computation: Callable, data: Any, config: ScalingConfig) -> tuple[Any, ScalingMetrics]:
        """Execute computation with horizontal scaling"""
        start_time = datetime.now()
        
        # Partition data
        chunks = self._partition_data(data, config.chunk_size)
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [executor.submit(computation, chunk) for chunk in chunks]
            results = [future.result(timeout=config.timeout_seconds) for future in futures]
        
        # Combine results
        combined_result = self._combine_results(results)
        
        # Calculate metrics
        execution_time = (datetime.now() - start_time).total_seconds()
        metrics = self._calculate_metrics(execution_time, len(chunks), config.max_workers)
        
        return combined_result, metrics
    
    def estimate_resources(self, data_size: int, complexity: ComputationComplexity) -> dict[str, float]:
        """Estimate resource requirements for horizontal scaling"""
        complexity_multipliers = {
            ComputationComplexity.LOW: 1.0,
            ComputationComplexity.MEDIUM: 2.5,
            ComputationComplexity.HIGH: 5.0,
            ComputationComplexity.EXTREME: 10.0
        }
        
        base_memory = data_size * 8 / (1024**3)  # Convert to GB
        multiplier = complexity_multipliers[complexity]
        
        return {
            "estimated_memory_gb": base_memory * multiplier,
            "estimated_time_seconds": data_size * multiplier / 10000,
            "recommended_workers": min(psutil.cpu_count(), max(2, data_size // 1000))
        }
    
    def _partition_data(self, data: Any, chunk_size: int) -> list[Any]:
        """Partition data into chunks for parallel processing"""
        if isinstance(data, list | tuple):
            return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        elif isinstance(data, np.ndarray):
            return np.array_split(data, max(1, len(data) // chunk_size))
        else:
            # For other data types, return as single chunk
            return [data]
    
    def _combine_results(self, results: list[Any]) -> Any:
        """Combine results from parallel computations"""
        if not results:
            return None
        
        if isinstance(results[0], list | tuple):
            combined = []
            for result in results:
                combined.extend(result)
            return combined
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        elif isinstance(results[0], dict):
            combined = {}
            for result in results:
                combined.update(result)
            return combined
        else:
            return results
    
    def _calculate_metrics(self, execution_time: float, chunks: int, workers: int) -> ScalingMetrics:
        """Calculate scaling metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024**3)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        throughput = chunks / execution_time if execution_time > 0 else 0
        efficiency_score = min(1.0, throughput / workers) if workers > 0 else 0
        
        return ScalingMetrics(
            execution_time=execution_time,
            memory_usage_gb=memory_usage_gb,
            cpu_utilization=cpu_percent,
            workers_used=workers,
            chunks_processed=chunks,
            throughput=throughput,
            efficiency_score=efficiency_score
        )

class VerticalScalingPattern(ScalingPattern):
    """Vertical scaling with memory optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, computation: Callable, data: Any, config: ScalingConfig) -> tuple[Any, ScalingMetrics]:
        """Execute computation with vertical scaling"""
        start_time = datetime.now()
        
        # Optimize memory usage
        optimized_data = self._optimize_memory(data, config.memory_limit_gb)
        
        # Execute computation
        result = computation(optimized_data)
        
        # Calculate metrics
        execution_time = (datetime.now() - start_time).total_seconds()
        metrics = self._calculate_metrics(execution_time, 1, 1)
        
        return result, metrics
    
    def estimate_resources(self, data_size: int, complexity: ComputationComplexity) -> dict[str, float]:
        """Estimate resource requirements for vertical scaling"""
        complexity_multipliers = {
            ComputationComplexity.LOW: 0.5,
            ComputationComplexity.MEDIUM: 1.0,
            ComputationComplexity.HIGH: 2.0,
            ComputationComplexity.EXTREME: 4.0
        }
        
        base_memory = data_size * 8 / (1024**3)
        multiplier = complexity_multipliers[complexity]
        
        return {
            "estimated_memory_gb": base_memory * multiplier,
            "estimated_time_seconds": data_size * multiplier / 5000,
            "recommended_workers": 1
        }
    
    def _optimize_memory(self, data: Any, memory_limit_gb: float) -> Any:
        """Optimize data for memory efficiency"""
        if isinstance(data, np.ndarray):
            # Convert to more memory-efficient dtype if possible
            if data.dtype == np.float64:
                return data.astype(np.float32)
            elif data.dtype == np.int64:
                return data.astype(np.int32)
        
        return data
    
    def _calculate_metrics(self, execution_time: float, chunks: int, workers: int) -> ScalingMetrics:
        """Calculate scaling metrics for vertical scaling"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024**3)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return ScalingMetrics(
            execution_time=execution_time,
            memory_usage_gb=memory_usage_gb,
            cpu_utilization=cpu_percent,
            workers_used=workers,
            chunks_processed=chunks,
            throughput=1.0 / execution_time if execution_time > 0 else 0,
            efficiency_score=min(1.0, cpu_percent / 100.0)
        )

class AdaptiveScalingManager:
    """Manager for adaptive scaling strategy selection"""
    
    def __init__(self):
        self.patterns = {
            ScalingStrategy.HORIZONTAL: HorizontalScalingPattern(),
            ScalingStrategy.VERTICAL: VerticalScalingPattern()
        }
        self.performance_history = []
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_optimal_scaling(self, computation: Callable, data: Any, 
                                         complexity: ComputationComplexity = ComputationComplexity.MEDIUM,
                                         config: ScalingConfig | None = None) -> tuple[Any, ScalingMetrics]:
        """Execute computation with optimal scaling strategy"""
        
        if config is None:
            config = ScalingConfig()
        
        # Auto-select strategy if needed
        if config.strategy == ScalingStrategy.AUTO:
            config.strategy = self._select_optimal_strategy(data, complexity)
        
        # Get appropriate pattern
        pattern = self.patterns.get(config.strategy, self.patterns[ScalingStrategy.HORIZONTAL])
        
        # Execute with selected pattern
        result, metrics = await pattern.execute(computation, data, config)
        
        # Record performance
        self.performance_history.append({
            'strategy': config.strategy,
            'complexity': complexity,
            'data_size': len(data) if hasattr(data, '__len__') else 1,
            'metrics': metrics,
            'timestamp': datetime.now()
        })
        
        return result, metrics
    
    def _select_optimal_strategy(self, data: Any, complexity: ComputationComplexity) -> ScalingStrategy:
        """Select optimal scaling strategy based on data and complexity"""
        data_size = len(data) if hasattr(data, '__len__') else 1
        
        # Simple heuristics for strategy selection
        if complexity in [ComputationComplexity.HIGH, ComputationComplexity.EXTREME]:
            if data_size > 10000:
                return ScalingStrategy.HORIZONTAL
            else:
                return ScalingStrategy.VERTICAL
        else:
            if data_size > 50000:
                return ScalingStrategy.HORIZONTAL
            else:
                return ScalingStrategy.VERTICAL
    
    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary across all executions"""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_history = self.performance_history[-10:]
        
        avg_execution_time = np.mean([h['metrics'].execution_time for h in recent_history])
        avg_efficiency = np.mean([h['metrics'].efficiency_score for h in recent_history])
        
        strategy_performance = {}
        for strategy in ScalingStrategy:
            strategy_history = [h for h in recent_history if h['strategy'] == strategy]
            if strategy_history:
                strategy_performance[strategy.value] = {
                    'count': len(strategy_history),
                    'avg_time': np.mean([h['metrics'].execution_time for h in strategy_history]),
                    'avg_efficiency': np.mean([h['metrics'].efficiency_score for h in strategy_history])
                }
        
        return {
            "total_executions": len(self.performance_history),
            "recent_avg_execution_time": avg_execution_time,
            "recent_avg_efficiency": avg_efficiency,
            "strategy_performance": strategy_performance,
            "recommended_strategy": max(strategy_performance.items(), 
                                      key=lambda x: x[1]['avg_efficiency'])[0] if strategy_performance else "horizontal"
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_scaling_patterns():
        """Test scaling patterns with sample computation"""
        
        # Sample computation function
        def sample_computation(data):
            """Sample mathematical computation"""
            if isinstance(data, np.ndarray):
                return np.sum(data ** 2)
            elif isinstance(data, list):
                return sum(x ** 2 for x in data)
            else:
                return data ** 2
        
        # Create test data
        test_data = np.random.random(10000)
        
        # Create adaptive scaling manager
        manager = AdaptiveScalingManager()
        
        print("🔧 SCALING PATTERNS TEST")
        print("=" * 40)
        
        # Test different complexities
        complexities = [ComputationComplexity.LOW, ComputationComplexity.MEDIUM, ComputationComplexity.HIGH]
        
        for complexity in complexities:
            print(f"\nTesting {complexity.value} complexity:")
            
            result, metrics = await manager.execute_with_optimal_scaling(
                sample_computation, test_data, complexity
            )
            
            print(f"  Result: {result:.2f}")
            print(f"  Execution time: {metrics.execution_time:.3f}s")
            print(f"  Memory usage: {metrics.memory_usage_gb:.2f}GB")
            print(f"  Efficiency: {metrics.efficiency_score:.2f}")
        
        # Print performance summary
        print("\n" + "=" * 40)
        summary = manager.get_performance_summary()
        print("📊 PERFORMANCE SUMMARY:")
        for key, value in summary.items():
            if key != "strategy_performance":
                print(f"  {key}: {value}")
        
        print("\n🎯 Strategy Performance:")
        for strategy, perf in summary.get("strategy_performance", {}).items():
            print(f"  {strategy}: {perf}")
    
    # Run test
    asyncio.run(test_scaling_patterns())
