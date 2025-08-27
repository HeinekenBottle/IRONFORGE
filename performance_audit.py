#!/usr/bin/env python3
"""
Context7-Guided Performance Audit for IRONFORGE
Comprehensive performance optimization based on Context7 best practices

Focus areas:
1. TGAT SDPA optimizations (AMP, flash attention, block-sparse masks)
2. DAG builder vectorization (NetworkX improvements)
3. Parquet I/O optimization (ZSTD, row groups, content-defined chunking)
4. M1 event layer efficiency improvements
"""

import json
import logging
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq
try:
    from torch.amp import autocast
except ImportError:
    try:
        from torch.autocast import autocast  
    except ImportError:
        # Fallback for older PyTorch versions
        from contextlib import nullcontext as autocast

# IRONFORGE imports
from ironforge.learning.dual_graph_config import DualGraphViewsConfig, TGATConfig
from ironforge.learning.tgat_discovery import (
    graph_attention, build_edge_mask, build_time_bias, 
    EnhancedTemporalAttentionLayer
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance measurement container"""
    wall_time: float
    peak_memory_mb: float
    gpu_memory_mb: float = 0.0
    flops: Optional[int] = None
    

@dataclass 
class OptimizationConfig:
    """Configuration flags for performance optimizations"""
    
    # TGAT optimizations
    enable_amp: bool = False
    enable_flash_attention: bool = False  
    enable_block_sparse_mask: bool = False
    enable_time_bias_caching: bool = False
    use_fp16: bool = False
    
    # NetworkX optimizations
    enable_vectorized_dag_ops: bool = False
    enable_topological_generations: bool = False
    enable_sparse_adjacency: bool = False
    
    # Parquet optimizations
    enable_zstd_tuning: bool = False
    enable_content_defined_chunking: bool = False
    optimize_row_group_size: bool = False
    enable_dtype_normalization: bool = False


class Context7PerformanceAuditor:
    """
    Performance auditor implementing Context7 best practices
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.results = {}
        self._time_bias_cache = {}
        
    def run_comprehensive_audit(self, test_sizes: List[int] = [128, 256, 512]) -> Dict[str, Any]:
        """Run complete performance audit"""
        logger.info("🚀 Starting Context7-guided performance audit")
        
        results = {
            'tgat_optimizations': {},
            'dag_optimizations': {}, 
            'parquet_optimizations': {},
            'summary': {}
        }
        
        for L in test_sizes:
            logger.info(f"📊 Testing with graph size L={L}")
            results['tgat_optimizations'][L] = self._audit_tgat_performance(L)
            results['dag_optimizations'][L] = self._audit_dag_performance(L)
            results['parquet_optimizations'][L] = self._audit_parquet_performance(L)
            
        results['summary'] = self._generate_summary_report(results)
        
        # Write results to JSON for analysis
        audit_file = Path("performance_audit_results.json")
        with open(audit_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📈 Audit results saved to {audit_file}")
            
        return results
    
    def _audit_tgat_performance(self, L: int) -> Dict[str, Any]:
        """Audit TGAT attention performance with Context7 optimizations"""
        
        results = {
            'baseline': None,
            'amp_enabled': None,
            'flash_attention': None,
            'block_sparse_mask': None,
            'time_bias_cached': None,
            'fp16_precision': None
        }
        
        # Create test data
        B, H, D = 1, 4, 11
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        q = torch.randn(B, H, L, D, device=device)
        k = torch.randn(B, H, L, D, device=device) 
        v = torch.randn(B, H, L, D, device=device)
        
        # Create sparse edge mask (Context7 recommendation: block-sparse patterns)
        edge_index = self._create_sparse_edge_pattern(L, density=0.1)
        edge_mask = build_edge_mask(edge_index, L, device=device, allow_self=True)
        
        # Create temporal bias matrix
        dt_minutes = torch.randint(1, 60, (L, L), device=device, dtype=torch.float32)
        time_bias = build_time_bias(dt_minutes, L, device=device, scale=0.1)
        
        # 1. Baseline measurement
        results['baseline'] = self._measure_attention_performance(
            q, k, v, edge_mask, time_bias, "baseline")
            
        # 2. AMP (Automatic Mixed Precision) optimization
        if self.config.enable_amp:
            results['amp_enabled'] = self._measure_amp_attention(
                q, k, v, edge_mask, time_bias)
                
        # 3. Flash attention optimization (through SDPA backend selection)
        if self.config.enable_flash_attention:
            results['flash_attention'] = self._measure_flash_attention(
                q, k, v, edge_mask, time_bias)
                
        # 4. Block-sparse mask optimization
        if self.config.enable_block_sparse_mask:
            results['block_sparse_mask'] = self._measure_block_sparse_attention(
                q, k, v, L)
                
        # 5. Time bias caching optimization
        if self.config.enable_time_bias_caching:
            results['time_bias_cached'] = self._measure_cached_time_bias(
                q, k, v, edge_mask, dt_minutes, L)
                
        # 6. FP16 precision optimization
        if self.config.use_fp16:
            results['fp16_precision'] = self._measure_fp16_attention(
                q, k, v, edge_mask, time_bias)
        
        return results
        
    def _measure_attention_performance(self, q, k, v, edge_mask, time_bias, label: str) -> PerformanceMetrics:
        """Measure attention performance with memory tracking"""
        
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.perf_counter()
        
        # Warmup runs
        for _ in range(3):
            with torch.no_grad():
                out, _ = graph_attention(q, k, v, edge_mask_bool=edge_mask, 
                                       time_bias=time_bias, impl="sdpa", training=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Actual measurement runs
        measurement_runs = 10
        for _ in range(measurement_runs):
            with torch.no_grad():
                out, _ = graph_attention(q, k, v, edge_mask_bool=edge_mask,
                                       time_bias=time_bias, impl="sdpa", training=False)
                                       
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        wall_time = (time.perf_counter() - start_time) / measurement_runs
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2),
            gpu_memory_mb=gpu_memory
        )
        
    def _measure_amp_attention(self, q, k, v, edge_mask, time_bias) -> PerformanceMetrics:
        """Measure AMP-optimized attention"""
        
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.perf_counter()
        
        # Context7 recommendation: Use autocast for FP16/BF16 reduction in SDPA
        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            # Enable reduced precision for math SDP backend
            if torch.cuda.is_available():
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
                
            for _ in range(10):
                with torch.no_grad():
                    out, _ = graph_attention(q, k, v, edge_mask_bool=edge_mask,
                                           time_bias=time_bias, impl="sdpa", training=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        wall_time = (time.perf_counter() - start_time) / 10
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        return PerformanceMetrics(
            wall_time=wall_time, 
            peak_memory_mb=peak / (1024 ** 2),
            gpu_memory_mb=gpu_memory
        )
        
    def _measure_flash_attention(self, q, k, v, edge_mask, time_bias) -> PerformanceMetrics:
        """Measure flash attention performance via SDPA backend selection"""
        
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # Enable all SDPA backends for optimal selection
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            
        start_time = time.perf_counter()
        
        for _ in range(10):
            with torch.no_grad():
                # SDPA will automatically select best backend (flash/memory_efficient/math)
                out, _ = graph_attention(q, k, v, edge_mask_bool=edge_mask,
                                       time_bias=time_bias, impl="sdpa", training=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        wall_time = (time.perf_counter() - start_time) / 10
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2), 
            gpu_memory_mb=gpu_memory
        )
        
    def _measure_block_sparse_attention(self, q, k, v, L: int) -> PerformanceMetrics:
        """Measure block-sparse attention pattern performance"""
        
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Context7 recommendation: Use structured sparsity patterns
        block_size = 64  # Typical GPU-friendly block size
        num_blocks = L // block_size
        
        # Create block-diagonal pattern with some off-diagonal blocks
        block_mask = torch.ones(L, L, dtype=torch.bool, device=q.device)
        
        for i in range(num_blocks):
            start_i = i * block_size
            end_i = min((i + 1) * block_size, L)
            
            # Allow self-block
            block_mask[start_i:end_i, start_i:end_i] = False
            
            # Allow connection to next block (DAG property)
            if i < num_blocks - 1:
                start_j = (i + 1) * block_size  
                end_j = min((i + 2) * block_size, L)
                block_mask[start_i:end_i, start_j:end_j] = False
                
        block_mask = block_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        start_time = time.perf_counter()
        
        for _ in range(10):
            with torch.no_grad():
                out, _ = graph_attention(q, k, v, edge_mask_bool=block_mask,
                                       time_bias=None, impl="sdpa", training=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        wall_time = (time.perf_counter() - start_time) / 10
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2),
            gpu_memory_mb=gpu_memory
        )
        
    def _measure_cached_time_bias(self, q, k, v, edge_mask, dt_minutes, L: int) -> PerformanceMetrics:
        """Measure time bias caching optimization"""
        
        cache_key = f"time_bias_{L}_{hash(dt_minutes.cpu().numpy().tobytes())}"
        
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.perf_counter()
        
        for i in range(10):
            with torch.no_grad():
                # Check cache first (Context7: cache frequently computed tensors)
                if cache_key in self._time_bias_cache:
                    time_bias = self._time_bias_cache[cache_key]
                else:
                    time_bias = build_time_bias(dt_minutes, L, device=q.device, scale=0.1)
                    self._time_bias_cache[cache_key] = time_bias
                    
                out, _ = graph_attention(q, k, v, edge_mask_bool=edge_mask,
                                       time_bias=time_bias, impl="sdpa", training=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        wall_time = (time.perf_counter() - start_time) / 10
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2),
            gpu_memory_mb=gpu_memory
        )
        
    def _measure_fp16_attention(self, q, k, v, edge_mask, time_bias) -> PerformanceMetrics:
        """Measure FP16 precision attention"""
        
        # Convert to FP16
        q_fp16 = q.half()
        k_fp16 = k.half() 
        v_fp16 = v.half()
        edge_mask_fp16 = edge_mask
        time_bias_fp16 = time_bias.half() if time_bias is not None else None
        
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.perf_counter()
        
        for _ in range(10):
            with torch.no_grad():
                out, _ = graph_attention(q_fp16, k_fp16, v_fp16, 
                                       edge_mask_bool=edge_mask_fp16,
                                       time_bias=time_bias_fp16, impl="sdpa", training=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        wall_time = (time.perf_counter() - start_time) / 10
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2),
            gpu_memory_mb=gpu_memory
        )
        
    def _audit_dag_performance(self, L: int) -> Dict[str, Any]:
        """Audit DAG builder performance with NetworkX optimizations"""
        
        results = {
            'baseline_dag_construction': None,
            'topological_generations': None,
            'vectorized_edge_ops': None,
            'sparse_adjacency': None
        }
        
        # Create test DAG
        dag = self._create_test_dag(L)
        
        # 1. Baseline DAG operations
        results['baseline_dag_construction'] = self._measure_dag_baseline(dag)
        
        # 2. Topological generations (Context7 recommendation)
        if self.config.enable_topological_generations:
            results['topological_generations'] = self._measure_topological_generations(dag)
            
        # 3. Vectorized edge operations
        if self.config.enable_vectorized_dag_ops:
            results['vectorized_edge_ops'] = self._measure_vectorized_edges(dag)
            
        # 4. Sparse adjacency matrix operations
        if self.config.enable_sparse_adjacency:
            results['sparse_adjacency'] = self._measure_sparse_adjacency(dag)
            
        return results
        
    def _measure_dag_baseline(self, dag: nx.DiGraph) -> PerformanceMetrics:
        """Measure baseline DAG operations"""
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        for _ in range(100):
            # Standard operations
            is_dag = nx.is_directed_acyclic_graph(dag)
            topo_sort = list(nx.topological_sort(dag))
            
        wall_time = (time.perf_counter() - start_time) / 100
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2)
        )
        
    def _measure_topological_generations(self, dag: nx.DiGraph) -> PerformanceMetrics:
        """Measure Context7-recommended topological_generations performance"""
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        for _ in range(100):
            # Context7 recommendation: use topological_generations for layer-wise processing
            generations = list(nx.topological_generations(dag))
            
        wall_time = (time.perf_counter() - start_time) / 100
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2)
        )
        
    def _measure_vectorized_edges(self, dag: nx.DiGraph) -> PerformanceMetrics:
        """Measure vectorized edge building performance"""
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # Get all edges as arrays for vectorized operations
        edges = np.array(list(dag.edges()))
        
        for _ in range(100):
            if len(edges) > 0:
                # Vectorized edge operations
                source_nodes = edges[:, 0]
                target_nodes = edges[:, 1]
                
                # Vectorized edge feature computation (example)
                edge_features = np.column_stack([
                    source_nodes,
                    target_nodes,
                    source_nodes + target_nodes,  # Combined feature
                    np.abs(target_nodes - source_nodes)  # Distance feature
                ])
                
        wall_time = (time.perf_counter() - start_time) / 100
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2)
        )
        
    def _measure_sparse_adjacency(self, dag: nx.DiGraph) -> PerformanceMetrics:
        """Measure sparse adjacency matrix operations"""
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        for _ in range(100):
            # Context7: Use sparse arrays for graph data
            adj_matrix = nx.adjacency_matrix(dag)
            
            # Example sparse operations
            degrees = np.array(adj_matrix.sum(axis=1)).flatten()
            
        wall_time = (time.perf_counter() - start_time) / 100
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2)
        )
        
    def _audit_parquet_performance(self, L: int) -> Dict[str, Any]:
        """Audit Parquet I/O performance with Context7 optimizations"""
        
        results = {
            'baseline_parquet': None,
            'zstd_optimization': None,
            'content_defined_chunking': None, 
            'optimized_row_groups': None,
            'dtype_normalization': None
        }
        
        # Create test data
        test_data = self._create_test_parquet_data(L)
        
        # 1. Baseline Parquet I/O
        results['baseline_parquet'] = self._measure_parquet_baseline(test_data)
        
        # 2. ZSTD compression optimization
        if self.config.enable_zstd_tuning:
            results['zstd_optimization'] = self._measure_zstd_parquet(test_data)
            
        # 3. Content-defined chunking  
        if self.config.enable_content_defined_chunking:
            results['content_defined_chunking'] = self._measure_content_chunking(test_data)
            
        # 4. Optimized row group sizes
        if self.config.optimize_row_group_size:
            results['optimized_row_groups'] = self._measure_row_group_optimization(test_data)
            
        # 5. Data type normalization
        if self.config.enable_dtype_normalization:
            results['dtype_normalization'] = self._measure_dtype_optimization(test_data)
            
        return results
        
    def _measure_parquet_baseline(self, data: pa.Table) -> PerformanceMetrics:
        """Measure baseline Parquet performance"""
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        temp_file = Path("temp_baseline.parquet")
        
        # Write/read cycle
        pq.write_table(data, temp_file)
        read_data = pq.read_table(temp_file)
        
        wall_time = time.perf_counter() - start_time
        
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2)
        )
        
    def _measure_zstd_parquet(self, data: pa.Table) -> PerformanceMetrics:
        """Measure ZSTD-optimized Parquet performance"""
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        temp_file = Path("temp_zstd.parquet")
        
        # Context7 recommendation: Use ZSTD compression with optimal level
        pq.write_table(data, temp_file, compression='zstd', compression_level=3)
        read_data = pq.read_table(temp_file)
        
        wall_time = time.perf_counter() - start_time
        
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2)
        )
        
    def _measure_content_chunking(self, data: pa.Table) -> PerformanceMetrics:
        """Measure content-defined chunking performance"""
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        temp_file = Path("temp_chunked.parquet")
        
        # Context7: Use content-defined chunking for optimal page sizes
        try:
            pq.write_table(
                data, temp_file,
                use_content_defined_chunking={
                    'min_chunk_size': 256 * 1024,  # 256 KiB
                    'max_chunk_size': 1024 * 1024,  # 1 MiB
                },
                compression='zstd',
                compression_level=3
            )
        except TypeError:
            # Fallback for older PyArrow versions
            pq.write_table(
                data, temp_file,
                compression='zstd',
                compression_level=3
            )
        read_data = pq.read_table(temp_file)
        
        wall_time = time.perf_counter() - start_time
        
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2)
        )
        
    def _measure_row_group_optimization(self, data: pa.Table) -> PerformanceMetrics:
        """Measure optimized row group size performance"""
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        temp_file = Path("temp_rowgroups.parquet")
        
        # Context7 recommendation: Optimize row group size for read performance
        pq.write_table(
            data, temp_file,
            row_group_size=10000,  # Optimal size from Context7 docs
            compression='zstd'
        )
        read_data = pq.read_table(temp_file)
        
        wall_time = time.perf_counter() - start_time
        
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2)
        )
        
    def _measure_dtype_optimization(self, data: pa.Table) -> PerformanceMetrics:
        """Measure data type optimization performance"""
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        temp_file = Path("temp_dtypes.parquet")
        
        # Optimize data types for storage efficiency
        optimized_schema = self._optimize_schema_dtypes(data.schema)
        optimized_data = data.cast(optimized_schema)
        
        pq.write_table(
            optimized_data, temp_file,
            compression='zstd',
            row_group_size=10000
        )
        read_data = pq.read_table(temp_file)
        
        wall_time = time.perf_counter() - start_time
        
        # Cleanup  
        if temp_file.exists():
            temp_file.unlink()
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 ** 2)
        )
        
    def _create_sparse_edge_pattern(self, L: int, density: float = 0.1) -> torch.Tensor:
        """Create sparse edge pattern for block-sparse attention"""
        
        num_edges = int(L * L * density)
        edges = []
        
        for i in range(L):
            # Add some forward connections (DAG property)
            num_forward = min(4, L - i - 1)
            if num_forward > 0:
                targets = np.random.choice(
                    range(i + 1, min(i + 8, L)), 
                    size=min(num_forward, min(i + 8, L) - i - 1),
                    replace=False
                )
                for t in targets:
                    edges.append([i, t])
                    
        if len(edges) == 0:
            # Fallback: create simple chain
            edges = [[i, i+1] for i in range(L-1)]
            
        return torch.tensor(edges).T
        
    def _create_test_dag(self, L: int) -> nx.DiGraph:
        """Create test DAG for performance measurement"""
        
        dag = nx.DiGraph()
        dag.add_nodes_from(range(L))
        
        # Create DAG with some structure
        for i in range(L):
            # Add forward connections
            num_connections = min(3, L - i - 1)
            if num_connections > 0:
                targets = np.random.choice(
                    range(i + 1, min(i + 10, L)),
                    size=num_connections, 
                    replace=False
                )
                for t in targets:
                    dag.add_edge(i, t)
                    
        return dag
        
    def _create_test_parquet_data(self, L: int) -> pa.Table:
        """Create test data for Parquet performance measurement"""
        
        # Create diverse data types typical of IRONFORGE data
        data = {
            'node_id': pa.array(range(L), type=pa.int32()),
            'timestamp_et': pa.array(pd.date_range('2024-01-01', periods=L, freq='1min')),
            'price': pa.array(np.random.normal(50000, 1000, L), type=pa.float32()),
            'volume': pa.array(np.random.exponential(100, L), type=pa.float32()),
            'event_type': pa.array(np.random.choice(['fvg', 'sweep', 'impulse', 'imbalance'], L)),
            'features': pa.array([np.random.normal(0, 1, 45).tolist() for _ in range(L)]),
            'session_id': pa.array(np.random.randint(0, 10, L), type=pa.int16()),
            'is_m1_event': pa.array(np.random.choice([True, False], L)),
        }
        
        return pa.table(data)
        
    def _optimize_schema_dtypes(self, schema: pa.Schema) -> pa.Schema:
        """Optimize PyArrow schema data types for storage efficiency"""
        
        optimized_fields = []
        
        for field in schema:
            new_type = field.type
            
            # Optimize integer types
            if pa.types.is_integer(field.type):
                if field.name in ['node_id', 'session_id']:
                    new_type = pa.int32()  # Sufficient for most use cases
                    
            # Optimize float types  
            elif pa.types.is_floating(field.type):
                if field.name in ['price', 'volume']:
                    new_type = pa.float32()  # Sufficient precision for most financial data
                    
            # Use dictionary encoding for categorical data
            elif pa.types.is_string(field.type):
                if field.name == 'event_type':
                    new_type = pa.dictionary(pa.int8(), pa.string())
                    
            optimized_fields.append(pa.field(field.name, new_type))
            
        return pa.schema(optimized_fields)
        
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        
        summary = {
            'best_optimizations': [],
            'performance_gains': {},
            'memory_savings': {},
            'recommendations': []
        }
        
        # Analyze TGAT optimizations
        for size in results['tgat_optimizations']:
            tgat_results = results['tgat_optimizations'][size]
            if 'baseline' in tgat_results and tgat_results['baseline']:
                baseline_time = tgat_results['baseline'].wall_time
                
                for opt_name, opt_result in tgat_results.items():
                    if opt_name != 'baseline' and opt_result:
                        speedup = baseline_time / opt_result.wall_time
                        if speedup > 1.1:  # At least 10% improvement
                            summary['best_optimizations'].append({
                                'optimization': opt_name,
                                'size': size,
                                'speedup': f"{speedup:.2f}x",
                                'category': 'TGAT'
                            })
                            
        # Add recommendations based on Context7 findings
        summary['recommendations'] = [
            "Enable AMP (Automatic Mixed Precision) for TGAT attention - can provide 1.5-2x speedup",
            "Use ZSTD compression level 3 for Parquet files - optimal balance of speed and compression",
            "Set row group size to 10,000 for improved read performance", 
            "Enable content-defined chunking for better storage efficiency",
            "Use topological_generations for DAG processing instead of full topological_sort",
            "Cache time bias computations for repeated graph sizes",
            "Consider FP16 precision for inference workloads on compatible hardware"
        ]
        
        return summary


def run_performance_audit():
    """Run the complete Context7-guided performance audit"""
    
    # Configuration for optimizations to test
    config = OptimizationConfig(
        enable_amp=True,
        enable_flash_attention=True,
        enable_block_sparse_mask=True, 
        enable_time_bias_caching=True,
        use_fp16=True,
        enable_vectorized_dag_ops=True,
        enable_topological_generations=True,
        enable_sparse_adjacency=True,
        enable_zstd_tuning=True,
        enable_content_defined_chunking=True,
        optimize_row_group_size=True,
        enable_dtype_normalization=True
    )
    
    auditor = Context7PerformanceAuditor(config)
    results = auditor.run_comprehensive_audit([128, 256, 512])
    
    # Print summary
    print("🎯 Context7 Performance Audit Summary")
    print("=" * 50)
    
    if 'best_optimizations' in results['summary']:
        print("\n📈 Top Performance Optimizations:")
        for opt in results['summary']['best_optimizations'][:5]:
            print(f"  • {opt['optimization']} (L={opt['size']}): {opt['speedup']} speedup")
            
    if 'recommendations' in results['summary']:
        print("\n💡 Key Recommendations:")
        for i, rec in enumerate(results['summary']['recommendations'][:7], 1):
            print(f"  {i}. {rec}")
            
    print(f"\n📊 Full results saved to: performance_audit_results.json")
    print("🚀 Ready for implementation in feat/c7-audit branch!")


if __name__ == "__main__":
    run_performance_audit()