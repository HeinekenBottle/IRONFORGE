"""
IRONFORGE Performance Budget Gates
==================================

Performance benchmarks with specific gates:
- Single session processing: <3 seconds
- Full discovery pipeline: <180 seconds (57 sessions)
- Memory footprint: <100MB total
- Initialization time: <2 seconds with lazy loading
"""

import gc
import time
import psutil
import pytest
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from ironforge.learning.discovery_pipeline import run_discovery
from ironforge.confluence.scoring import score_confluence
from ironforge.validation.runner import validate_run
from ironforge.reporting.minidash import build_minidash
from ironforge.sdk.app_config import load_config, materialize_run_dir
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading


class PerformanceGate:
    """Performance gate with tolerance."""
    
    def __init__(self, name: str, threshold: float, tolerance: float = 0.05):
        self.name = name
        self.threshold = threshold
        self.tolerance = tolerance
        self.max_allowed = threshold * (1 + tolerance)
    
    def check(self, actual: float) -> bool:
        """Check if actual value passes the gate."""
        return actual <= self.max_allowed
    
    def __str__(self) -> str:
        return f"{self.name}: {self.threshold}s (max: {self.max_allowed:.2f}s)"


# Performance gates with 5% tolerance
PERFORMANCE_GATES = {
    'session_processing': PerformanceGate('Single Session Processing', 3.0),
    'full_pipeline': PerformanceGate('Full Discovery Pipeline', 180.0),
    'initialization': PerformanceGate('System Initialization', 2.0),
    'memory_footprint': PerformanceGate('Memory Footprint', 100.0),  # MB
}


class MemoryProfiler:
    """Memory usage profiler."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline = None
    
    def start(self):
        """Start memory profiling."""
        gc.collect()  # Force garbage collection
        self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def current_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def peak_usage(self) -> float:
        """Get peak memory usage since start in MB."""
        if self.baseline is None:
            return self.current_usage()
        return max(0, self.current_usage() - self.baseline)


@pytest.fixture
def sample_shard_paths():
    """Get sample shard paths for testing."""
    data_dir = Path("data/shards/NQ_M5")
    if not data_dir.exists():
        pytest.skip("Test data not available")
    
    shard_paths = list(data_dir.glob("shard_*"))[:5]  # Use first 5 shards for testing
    if len(shard_paths) < 5:
        pytest.skip("Insufficient test data")
    
    return [str(p) for p in shard_paths]


@pytest.fixture
def test_config():
    """Get test configuration."""
    config_path = Path("configs/dev.yml")
    if not config_path.exists():
        pytest.skip("Test configuration not available")
    
    return load_config(str(config_path))


class TestInitializationPerformance:
    """Test system initialization performance."""
    
    def test_lazy_loading_initialization(self):
        """Test lazy loading initialization meets <2s gate."""
        profiler = MemoryProfiler()
        profiler.start()
        
        start_time = time.time()
        
        # Initialize lazy loading container
        container = initialize_ironforge_lazy_loading()
        
        # Access a few components to trigger lazy loading
        container.get_enhanced_graph_builder()
        container.get_tgat_discovery()
        
        end_time = time.time()
        initialization_time = end_time - start_time
        
        gate = PERFORMANCE_GATES['initialization']
        assert gate.check(initialization_time), (
            f"Initialization time {initialization_time:.2f}s exceeds gate {gate}"
        )
        
        print(f"✅ Initialization: {initialization_time:.2f}s (gate: {gate.threshold}s)")
    
    def test_cold_start_performance(self):
        """Test cold start performance."""
        import subprocess
        import sys
        
        start_time = time.time()
        
        # Test cold start by importing in subprocess
        result = subprocess.run([
            sys.executable, "-c",
            "import ironforge; from ironforge.api import run_discovery; print('OK')"
        ], capture_output=True, text=True, timeout=10)
        
        end_time = time.time()
        cold_start_time = end_time - start_time
        
        assert result.returncode == 0, f"Cold start failed: {result.stderr}"
        
        gate = PERFORMANCE_GATES['initialization']
        assert gate.check(cold_start_time), (
            f"Cold start time {cold_start_time:.2f}s exceeds gate {gate}"
        )
        
        print(f"✅ Cold start: {cold_start_time:.2f}s (gate: {gate.threshold}s)")


class TestSessionProcessingPerformance:
    """Test single session processing performance."""
    
    def test_single_session_processing(self, sample_shard_paths, test_config):
        """Test single session processing meets <3s gate."""
        if not sample_shard_paths:
            pytest.skip("No sample shards available")
        
        profiler = MemoryProfiler()
        profiler.start()
        
        # Test single session processing
        single_shard = sample_shard_paths[0]
        run_dir = materialize_run_dir(test_config)
        
        start_time = time.time()
        
        # Process single session
        results = run_discovery([single_shard], str(run_dir), test_config.loader)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(results) > 0, "No results from single session processing"
        
        gate = PERFORMANCE_GATES['session_processing']
        assert gate.check(processing_time), (
            f"Session processing time {processing_time:.2f}s exceeds gate {gate}"
        )
        
        memory_usage = profiler.peak_usage()
        print(f"✅ Single session: {processing_time:.2f}s, {memory_usage:.1f}MB (gate: {gate.threshold}s)")
    
    def test_session_processing_memory(self, sample_shard_paths, test_config):
        """Test session processing memory usage."""
        if not sample_shard_paths:
            pytest.skip("No sample shards available")
        
        profiler = MemoryProfiler()
        profiler.start()
        
        # Process multiple sessions to test memory usage
        run_dir = materialize_run_dir(test_config)
        
        for shard_path in sample_shard_paths[:3]:  # Test 3 sessions
            run_discovery([shard_path], str(run_dir), test_config.loader)
            
            # Check memory doesn't grow excessively
            current_memory = profiler.peak_usage()
            memory_gate = PERFORMANCE_GATES['memory_footprint']
            
            assert memory_gate.check(current_memory), (
                f"Memory usage {current_memory:.1f}MB exceeds gate {memory_gate}"
            )
        
        final_memory = profiler.peak_usage()
        print(f"✅ Session memory: {final_memory:.1f}MB (gate: {memory_gate.threshold}MB)")


class TestPipelinePerformance:
    """Test full pipeline performance."""
    
    @pytest.mark.slow
    def test_full_pipeline_performance(self, sample_shard_paths, test_config):
        """Test full pipeline meets <180s gate."""
        if len(sample_shard_paths) < 5:
            pytest.skip("Insufficient shards for pipeline test")
        
        profiler = MemoryProfiler()
        profiler.start()
        
        run_dir = materialize_run_dir(test_config)
        
        start_time = time.time()
        
        # Stage 1: Discovery
        discovery_start = time.time()
        pattern_paths = run_discovery(sample_shard_paths, str(run_dir), test_config.loader)
        discovery_time = time.time() - discovery_start
        
        assert len(pattern_paths) > 0, "No patterns discovered"
        
        # Stage 2: Scoring
        scoring_start = time.time()
        confluence_dir = run_dir / "confluence"
        weights = getattr(test_config.scoring, 'weights', None)
        if weights:
            weights = {k: float(v) for k, v in vars(weights).items()}
        
        score_confluence(pattern_paths, str(confluence_dir), weights, 65.0)
        scoring_time = time.time() - scoring_start
        
        # Stage 3: Validation
        validation_start = time.time()
        validate_run(test_config)
        validation_time = time.time() - validation_start
        
        # Stage 4: Reporting
        reporting_start = time.time()
        build_minidash(test_config)
        reporting_time = time.time() - reporting_start
        
        end_time = time.time()
        total_time = end_time - start_time
        
        gate = PERFORMANCE_GATES['full_pipeline']
        assert gate.check(total_time), (
            f"Full pipeline time {total_time:.2f}s exceeds gate {gate}"
        )
        
        memory_usage = profiler.peak_usage()
        memory_gate = PERFORMANCE_GATES['memory_footprint']
        assert memory_gate.check(memory_usage), (
            f"Pipeline memory usage {memory_usage:.1f}MB exceeds gate {memory_gate}"
        )
        
        print(f"✅ Full pipeline: {total_time:.2f}s, {memory_usage:.1f}MB")
        print(f"   - Discovery: {discovery_time:.2f}s")
        print(f"   - Scoring: {scoring_time:.2f}s")
        print(f"   - Validation: {validation_time:.2f}s")
        print(f"   - Reporting: {reporting_time:.2f}s")
    
    def test_pipeline_scalability(self, sample_shard_paths, test_config):
        """Test pipeline scalability with different session counts."""
        if len(sample_shard_paths) < 3:
            pytest.skip("Insufficient shards for scalability test")
        
        results = []
        
        for session_count in [1, 3, 5]:
            if session_count > len(sample_shard_paths):
                continue
            
            profiler = MemoryProfiler()
            profiler.start()
            
            run_dir = materialize_run_dir(test_config)
            test_shards = sample_shard_paths[:session_count]
            
            start_time = time.time()
            pattern_paths = run_discovery(test_shards, str(run_dir), test_config.loader)
            end_time = time.time()
            
            processing_time = end_time - start_time
            memory_usage = profiler.peak_usage()
            
            results.append({
                'sessions': session_count,
                'time': processing_time,
                'memory': memory_usage,
                'time_per_session': processing_time / session_count,
            })
        
        # Check scalability
        for result in results:
            session_gate = PERFORMANCE_GATES['session_processing']
            assert session_gate.check(result['time_per_session']), (
                f"Time per session {result['time_per_session']:.2f}s exceeds gate {session_gate}"
            )
            
            print(f"✅ {result['sessions']} sessions: {result['time']:.2f}s total, "
                  f"{result['time_per_session']:.2f}s/session, {result['memory']:.1f}MB")


class TestMemoryLeakDetection:
    """Test for memory leaks in long-running operations."""
    
    def test_repeated_processing_memory_stability(self, sample_shard_paths, test_config):
        """Test memory stability over repeated processing."""
        if not sample_shard_paths:
            pytest.skip("No sample shards available")
        
        profiler = MemoryProfiler()
        profiler.start()
        
        memory_readings = []
        single_shard = sample_shard_paths[0]
        
        # Process same session multiple times
        for i in range(5):
            run_dir = materialize_run_dir(test_config)
            run_discovery([single_shard], str(run_dir), test_config.loader)
            
            # Force garbage collection and measure memory
            gc.collect()
            memory_usage = profiler.current_usage()
            memory_readings.append(memory_usage)
        
        # Check for memory leaks (memory should be stable)
        memory_growth = memory_readings[-1] - memory_readings[0]
        max_allowed_growth = 10.0  # MB
        
        assert memory_growth < max_allowed_growth, (
            f"Memory leak detected: {memory_growth:.1f}MB growth over 5 iterations"
        )
        
        print(f"✅ Memory stability: {memory_growth:.1f}MB growth over 5 iterations")


def test_performance_gate_summary():
    """Print performance gate summary."""
    print("\n" + "="*60)
    print("IRONFORGE Performance Gates Summary")
    print("="*60)
    
    for gate_name, gate in PERFORMANCE_GATES.items():
        print(f"  {gate}")
    
    print("="*60)
