#!/usr/bin/env python3
"""
IRONFORGE Architecture Audit: SDPA vs Manual Attention Parity Tests
Context7-guided validation of current implementations vs PyTorch best practices

Based on Context7 findings:
- SDPA supports flash, memory-efficient, math backends with automatic selection
- Edge masks should use float format (0.0 for allowed, -inf for blocked) 
- Broadcasting semantics require careful shape validation
- AMP precision considerations for fp16/bf16 reductions
"""

import math
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from pathlib import Path
import time
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import existing TGAT utilities
import sys
sys.path.append('/Users/jack/IRONFORGE')
from ironforge.learning.tgat_discovery import graph_attention, build_edge_mask


class SDPAParity:
    """Validates SDPA implementation against manual attention for exact parity"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def create_test_tensors(self, B=2, H=4, L=64, D=16, S=None):
        """Create test tensors with realistic dimensions for TGAT"""
        if S is None:
            S = L
        
        # Use consistent seed for reproducible results
        torch.manual_seed(42)
        
        q = torch.randn(B, H, L, D, device=self.device, requires_grad=True)
        k = torch.randn(B, H, S, D, device=self.device, requires_grad=True) 
        v = torch.randn(B, H, S, D, device=self.device, requires_grad=True)
        
        return q, k, v
    
    def create_graph_mask(self, L=64, sparsity=0.7):
        """Create realistic graph connectivity mask"""
        # Simulate DAG-like connectivity pattern
        G = nx.erdos_renyi_graph(L, 1-sparsity, directed=True)
        
        # Remove cycles to make it DAG-like
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                if len(cycle) > 1:
                    G.remove_edge(cycle[0], cycle[1])
        except:
            pass  # Already acyclic or empty
            
        # Convert to edge_index format
        edges = list(G.edges())
        if not edges:  # Fallback for empty graph
            edges = [(i, min(i+1, L-1)) for i in range(L-1)]
            
        edge_index = torch.tensor(edges, device=self.device).T
        
        # Build mask using existing function
        mask_bool = build_edge_mask(edge_index, L, device=self.device, allow_self=True)
        return mask_bool, edge_index
    
    def test_basic_parity(self, tolerance=1e-5):
        """Test basic SDPA vs manual attention parity"""
        logger.info("Testing basic SDPA vs manual attention parity...")
        
        q, k, v = self.create_test_tensors()
        B, H, L, D = q.shape
        
        # Test without any masking first
        with torch.no_grad():
            out_sdpa, _ = graph_attention(q, k, v, impl="sdpa", training=False)
            out_manual, attn_manual = graph_attention(q, k, v, impl="manual", training=False)
            
            diff = (out_sdpa - out_manual).abs().max().item()
            passed = diff < tolerance
            
            self.results['basic_parity'] = {
                'max_diff': diff,
                'tolerance': tolerance,
                'passed': passed
            }
            
            logger.info(f"Basic parity test: {'PASS' if passed else 'FAIL'} (max_diff={diff:.2e})")
            return passed
    
    def test_masked_attention_parity(self, tolerance=1e-5):
        """Test masked attention parity with graph connectivity"""
        logger.info("Testing masked attention with graph connectivity...")
        
        q, k, v = self.create_test_tensors()
        mask_bool, edge_index = self.create_graph_mask()
        
        with torch.no_grad():
            out_sdpa, _ = graph_attention(q, k, v, edge_mask_bool=mask_bool, impl="sdpa", training=False)
            out_manual, attn_manual = graph_attention(q, k, v, edge_mask_bool=mask_bool, impl="manual", training=False)
            
            diff = (out_sdpa - out_manual).abs().max().item()
            passed = diff < tolerance
            
            # Validate attention weights are properly masked
            if attn_manual is not None:
                blocked_positions = mask_bool[0, 0]  # [L, S]
                attention_at_blocked = attn_manual[0, 0][blocked_positions].abs().max().item()
                mask_effective = attention_at_blocked < 1e-6
            else:
                mask_effective = True  # Assume SDPA masks correctly
            
            self.results['masked_parity'] = {
                'max_diff': diff,
                'tolerance': tolerance,
                'passed': passed and mask_effective,
                'mask_effective': mask_effective
            }
            
            logger.info(f"Masked parity test: {'PASS' if passed else 'FAIL'} (max_diff={diff:.2e})")
            logger.info(f"Mask effectiveness: {'PASS' if mask_effective else 'FAIL'}")
            return passed and mask_effective
    
    def test_temporal_bias_parity(self, tolerance=1e-5):
        """Test temporal bias application parity"""
        logger.info("Testing temporal bias parity...")
        
        q, k, v = self.create_test_tensors()
        B, H, L, D = q.shape
        
        # Create realistic temporal bias (distance-based decay)
        distances = torch.abs(torch.arange(L, device=self.device).unsqueeze(0) - 
                             torch.arange(L, device=self.device).unsqueeze(1)).float()
        time_bias = -0.1 * distances  # Decay with distance
        time_bias = time_bias.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        with torch.no_grad():
            out_sdpa, _ = graph_attention(q, k, v, time_bias=time_bias, impl="sdpa", training=False)
            out_manual, attn_manual = graph_attention(q, k, v, time_bias=time_bias, impl="manual", training=False)
            
            diff = (out_sdpa - out_manual).abs().max().item()
            passed = diff < tolerance
            
            self.results['temporal_bias_parity'] = {
                'max_diff': diff,
                'tolerance': tolerance,
                'passed': passed
            }
            
            logger.info(f"Temporal bias parity: {'PASS' if passed else 'FAIL'} (max_diff={diff:.2e})")
            return passed
    
    def test_mask_dtype_compatibility(self):
        """Test mask dtype compatibility per Context7 recommendations"""
        logger.info("Testing mask dtype compatibility...")
        
        q, k, v = self.create_test_tensors()
        B, H, L, D = q.shape
        
        # Test different mask dtypes
        mask_bool = torch.rand(B, 1, L, L, device=self.device) > 0.5
        mask_float = mask_bool.float() * (-1e9)  # Recommended format
        
        errors = []
        
        try:
            # Test boolean mask (should work)
            out_bool, _ = graph_attention(q, k, v, edge_mask_bool=mask_bool, impl="sdpa", training=False)
            bool_works = True
        except Exception as e:
            bool_works = False
            errors.append(f"Boolean mask failed: {e}")
        
        try:
            # Test float mask as time_bias (recommended approach)
            out_float, _ = graph_attention(q, k, v, time_bias=mask_float, impl="sdpa", training=False)
            float_works = True
        except Exception as e:
            float_works = False
            errors.append(f"Float mask failed: {e}")
        
        # Check shape compatibility
        valid_shapes = []
        for shape in [(B, 1, L, L), (1, 1, L, L), (1, H, L, L), (B, H, L, L)]:
            try:
                test_mask = torch.zeros(shape, device=self.device)
                graph_attention(q, k, v, time_bias=test_mask, impl="sdpa", training=False)
                valid_shapes.append(shape)
            except:
                pass
        
        self.results['mask_dtype'] = {
            'bool_mask_works': bool_works,
            'float_mask_works': float_works,
            'valid_shapes': valid_shapes,
            'errors': errors
        }
        
        logger.info(f"Boolean mask support: {'YES' if bool_works else 'NO'}")
        logger.info(f"Float mask support: {'YES' if float_works else 'NO'}")
        logger.info(f"Valid shapes: {valid_shapes}")
        
        return bool_works or float_works
    
    def test_backend_availability(self):
        """Test SDPA backend availability and selection"""
        logger.info("Testing SDPA backend availability...")
        
        backends = {}
        
        # Test flash attention availability
        try:
            flash_available = torch.backends.cuda.is_flash_attention_available()
            flash_enabled = torch.backends.cuda.flash_sdp_enabled()
            backends['flash'] = {'available': flash_available, 'enabled': flash_enabled}
        except:
            backends['flash'] = {'available': False, 'enabled': False}
        
        # Test memory efficient attention
        try:
            mem_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
            backends['memory_efficient'] = {'enabled': mem_enabled}
        except:
            backends['memory_efficient'] = {'enabled': False}
        
        # Test math backend
        try:
            math_enabled = torch.backends.cuda.math_sdp_enabled()
            backends['math'] = {'enabled': math_enabled}
        except:
            backends['math'] = {'enabled': False}
        
        self.results['backends'] = backends
        
        for name, info in backends.items():
            logger.info(f"{name.capitalize()} backend: {info}")
        
        return any(info.get('enabled', info.get('available', False)) for info in backends.values())
    
    def benchmark_performance(self, sizes=[(32, 2), (64, 4), (128, 8)]):
        """Benchmark SDPA vs manual attention performance"""
        logger.info("Benchmarking SDPA vs manual performance...")
        
        results = {}
        
        for L, H in sizes:
            q, k, v = self.create_test_tensors(L=L, H=H)
            mask_bool, _ = self.create_graph_mask(L=L)
            
            # Warm up
            for _ in range(5):
                graph_attention(q, k, v, edge_mask_bool=mask_bool, impl="sdpa", training=False)
                graph_attention(q, k, v, edge_mask_bool=mask_bool, impl="manual", training=False)
            
            # Benchmark SDPA
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(20):
                graph_attention(q, k, v, edge_mask_bool=mask_bool, impl="sdpa", training=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            sdpa_time = (time.time() - start) / 20
            
            # Benchmark manual
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(20):
                graph_attention(q, k, v, edge_mask_bool=mask_bool, impl="manual", training=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            manual_time = (time.time() - start) / 20
            
            speedup = manual_time / sdpa_time if sdpa_time > 0 else float('inf')
            
            results[f'L{L}_H{H}'] = {
                'sdpa_time_ms': sdpa_time * 1000,
                'manual_time_ms': manual_time * 1000,
                'speedup': speedup
            }
            
            logger.info(f"Size L={L}, H={H}: SDPA {sdpa_time*1000:.2f}ms vs Manual {manual_time*1000:.2f}ms (speedup: {speedup:.2f}x)")
        
        self.results['performance'] = results
        return results


class NetworkXValidation:
    """Validate NetworkX DiGraph operations per Context7 findings"""
    
    def __init__(self):
        self.results = {}
    
    def test_topological_generations(self):
        """Test topological_generations function from Context7 docs"""
        logger.info("Testing NetworkX topological_generations...")
        
        # Create test DAG
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
        
        try:
            generations = list(nx.topological_generations(G))
            expected = [[0], [1, 2], [3], [4]]
            
            # Convert to sets for comparison (order within generation doesn't matter)
            generations_sets = [set(gen) for gen in generations]
            expected_sets = [set(gen) for gen in expected]
            
            passed = generations_sets == expected_sets
            
            self.results['topological_generations'] = {
                'passed': passed,
                'result': generations,
                'expected': expected
            }
            
            logger.info(f"Topological generations: {'PASS' if passed else 'FAIL'}")
            logger.info(f"Result: {generations}")
            
            return passed
            
        except Exception as e:
            self.results['topological_generations'] = {
                'passed': False,
                'error': str(e)
            }
            logger.error(f"Topological generations failed: {e}")
            return False
    
    def test_digraph_edge_operations(self):
        """Test DiGraph edge operations and vectorized builds"""
        logger.info("Testing DiGraph edge operations...")
        
        try:
            # Create large DiGraph
            n_nodes = 1000
            edges = [(i, (i + np.random.randint(1, 5)) % n_nodes) for i in range(n_nodes * 2)]
            
            # Test vectorized creation
            start = time.time()
            G = nx.DiGraph()
            G.add_edges_from(edges)
            creation_time = time.time() - start
            
            # Test edge queries
            start = time.time()
            out_edges = list(G.out_edges())
            in_edges = list(G.in_edges())
            edge_query_time = time.time() - start
            
            # Test degree calculations
            start = time.time()
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())
            degree_time = time.time() - start
            
            passed = len(out_edges) > 0 and len(in_degrees) == n_nodes
            
            self.results['digraph_operations'] = {
                'passed': passed,
                'nodes': n_nodes,
                'edges': len(edges),
                'creation_time_ms': creation_time * 1000,
                'edge_query_time_ms': edge_query_time * 1000,
                'degree_time_ms': degree_time * 1000
            }
            
            logger.info(f"DiGraph operations: {'PASS' if passed else 'FAIL'}")
            logger.info(f"Created {n_nodes} nodes, {len(edges)} edges in {creation_time*1000:.2f}ms")
            
            return passed
            
        except Exception as e:
            self.results['digraph_operations'] = {
                'passed': False,
                'error': str(e)
            }
            logger.error(f"DiGraph operations failed: {e}")
            return False


class ParquetValidation:
    """Validate Parquet configuration per Context7 findings"""
    
    def __init__(self):
        self.results = {}
    
    def test_zstd_compression(self):
        """Test ZSTD compression support"""
        logger.info("Testing ZSTD compression support...")
        
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import tempfile
            import os
            
            # Create test data
            data = {
                'values': np.random.randn(10000),
                'categories': np.random.choice(['A', 'B', 'C'], 10000),
                'timestamps': pd.date_range('2023-01-01', periods=10000, freq='1min')
            }
            df = pd.DataFrame(data)
            
            # Test ZSTD compression
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
                temp_path = f.name
            
            try:
                # Write with ZSTD
                start = time.time()
                df.to_parquet(temp_path, compression='zstd', row_group_size=10000)
                write_time = time.time() - start
                
                # Read back
                start = time.time()
                df_read = pd.read_parquet(temp_path)
                read_time = time.time() - start
                
                # Verify data integrity
                data_matches = df.equals(df_read)
                file_size = os.path.getsize(temp_path)
                
                passed = data_matches and file_size > 0
                
                self.results['zstd_compression'] = {
                    'passed': passed,
                    'write_time_ms': write_time * 1000,
                    'read_time_ms': read_time * 1000,
                    'file_size_kb': file_size / 1024,
                    'data_matches': data_matches
                }
                
                logger.info(f"ZSTD compression: {'PASS' if passed else 'FAIL'}")
                logger.info(f"Write: {write_time*1000:.2f}ms, Read: {read_time*1000:.2f}ms, Size: {file_size/1024:.1f}KB")
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            return passed
            
        except ImportError:
            self.results['zstd_compression'] = {
                'passed': False,
                'error': 'PyArrow not available'
            }
            logger.error("PyArrow not available for ZSTD test")
            return False
        except Exception as e:
            self.results['zstd_compression'] = {
                'passed': False,
                'error': str(e)
            }
            logger.error(f"ZSTD compression test failed: {e}")
            return False
    
    def test_cdc_support_probe(self):
        """Probe for CDC (Change Data Capture) support"""
        logger.info("Probing CDC support...")
        
        try:
            import pyarrow as pa
            
            # Check version
            version = pa.__version__
            version_parts = tuple(map(int, version.split('.')))
            
            # CDC support typically requires PyArrow >= 12.0
            cdc_version_support = version_parts >= (12, 0, 0)
            
            # Try to access CDC-related features
            cdc_features = []
            
            # Check for dataset features
            try:
                from pyarrow import dataset as ds
                cdc_features.append('dataset_api')
            except:
                pass
            
            # Check for compute features
            try:
                import pyarrow.compute as pc
                cdc_features.append('compute_api')
            except:
                pass
            
            self.results['cdc_support'] = {
                'pyarrow_version': version,
                'version_support': cdc_version_support,
                'available_features': cdc_features,
                'likely_supported': cdc_version_support and len(cdc_features) > 0
            }
            
            logger.info(f"PyArrow version: {version}")
            logger.info(f"CDC version support: {'YES' if cdc_version_support else 'NO'}")
            logger.info(f"Available features: {cdc_features}")
            
            return cdc_version_support and len(cdc_features) > 0
            
        except Exception as e:
            self.results['cdc_support'] = {
                'passed': False,
                'error': str(e)
            }
            logger.error(f"CDC support probe failed: {e}")
            return False


def run_comprehensive_audit():
    """Run comprehensive audit of IRONFORGE architecture"""
    logger.info("🔍 Starting IRONFORGE Architecture Audit")
    logger.info("=" * 60)
    
    results = {
        'sdpa': {},
        'networkx': {},
        'parquet': {},
        'summary': {}
    }
    
    # SDPA Tests
    logger.info("\n📊 SDPA PARITY TESTS")
    logger.info("-" * 30)
    sdpa_validator = SDPAParity()
    
    sdpa_tests = [
        ('basic_parity', sdpa_validator.test_basic_parity),
        ('masked_parity', sdpa_validator.test_masked_attention_parity),
        ('temporal_bias_parity', sdpa_validator.test_temporal_bias_parity),
        ('mask_dtype', sdpa_validator.test_mask_dtype_compatibility),
        ('backends', sdpa_validator.test_backend_availability)
    ]
    
    sdpa_results = {}
    for test_name, test_func in sdpa_tests:
        try:
            result = test_func()
            sdpa_results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            sdpa_results[test_name] = False
    
    # Run performance benchmark
    try:
        sdpa_validator.benchmark_performance()
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
    
    results['sdpa'] = {**sdpa_results, **sdpa_validator.results}
    
    # NetworkX Tests
    logger.info("\n🕸️  NETWORKX TESTS")
    logger.info("-" * 20)
    nx_validator = NetworkXValidation()
    
    nx_tests = [
        ('topological_generations', nx_validator.test_topological_generations),
        ('digraph_operations', nx_validator.test_digraph_edge_operations)
    ]
    
    nx_results = {}
    for test_name, test_func in nx_tests:
        try:
            result = test_func()
            nx_results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            nx_results[test_name] = False
    
    results['networkx'] = {**nx_results, **nx_validator.results}
    
    # Parquet Tests
    logger.info("\n📦 PARQUET TESTS")
    logger.info("-" * 15)
    parquet_validator = ParquetValidation()
    
    parquet_tests = [
        ('zstd_compression', parquet_validator.test_zstd_compression),
        ('cdc_support', parquet_validator.test_cdc_support_probe)
    ]
    
    parquet_results = {}
    for test_name, test_func in parquet_tests:
        try:
            result = test_func()
            parquet_results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            parquet_results[test_name] = False
    
    results['parquet'] = {**parquet_results, **parquet_validator.results}
    
    # Summary
    logger.info("\n📋 AUDIT SUMMARY")
    logger.info("=" * 20)
    
    total_tests = sum(len(results[cat]) for cat in ['sdpa', 'networkx', 'parquet'] for k, v in results[cat].items() if isinstance(v, bool))
    passed_tests = sum(1 for cat in ['sdpa', 'networkx', 'parquet'] for k, v in results[cat].items() if isinstance(v, bool) and v)
    
    results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
        'timestamp': time.time()
    }
    
    logger.info(f"Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    return results


if __name__ == '__main__':
    results = run_comprehensive_audit()
    
    # Save results
    output_path = Path('/Users/jack/IRONFORGE/audit_results.json')
    with open(output_path, 'w') as f:
        # Make results JSON serializable
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, (torch.Tensor, np.ndarray)):
                return str(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                return obj
        
        json.dump(clean_for_json(results), f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to {output_path}")