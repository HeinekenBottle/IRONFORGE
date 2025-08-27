"""
Context7-Optimized DAG Builder
Enhanced DAG construction with NetworkX optimizations

Key optimizations:
1. Topological generations for efficient layer processing
2. Vectorized edge operations  
3. Sparse adjacency matrix operations
4. Optimized DAG validation
5. Batch edge creation
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass

import numpy as np
import networkx as nx
import torch
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

from .dag_graph_builder import DAGGraphBuilder
from .enhanced_graph_builder import RichNodeFeature

logger = logging.getLogger(__name__)


@dataclass
class OptimizedDAGConfig:
    """Configuration for Context7 DAG optimizations"""
    
    # Base DAG configuration
    base_dag_config: Dict[str, Any]
    
    # Performance optimizations
    enable_vectorized_ops: bool = True
    enable_topological_generations: bool = True
    enable_sparse_adjacency: bool = True
    enable_batch_edge_creation: bool = True
    
    # Advanced optimizations
    parallel_validation: bool = True
    cache_adjacency_matrix: bool = True
    use_csr_format: bool = True
    
    # Memory optimizations
    edge_batch_size: int = 1000
    max_cache_size: int = 100


class OptimizedDAGBuilder(DAGGraphBuilder):
    """
    Context7-optimized DAG builder with advanced NetworkX optimizations
    """
    
    def __init__(self, dag_config: dict = None, opt_config: OptimizedDAGConfig = None):
        super().__init__(dag_config)
        
        self.opt_config = opt_config or OptimizedDAGConfig(
            base_dag_config=self.dag_config,
            enable_vectorized_ops=True,
            enable_topological_generations=True,
            enable_sparse_adjacency=True
        )
        
        # Caches for performance
        self._adjacency_cache = {} if opt_config and opt_config.cache_adjacency_matrix else None
        self._topo_cache = {}
        
        logger.info(f"Optimized DAG Builder initialized with vectorized ops: "
                   f"{self.opt_config.enable_vectorized_ops}, "
                   f"topological generations: {self.opt_config.enable_topological_generations}")
        
    def build_optimized_dag(self, session_data: Dict[str, Any]) -> nx.DiGraph:
        """
        Build DAG with Context7 optimizations
        
        Args:
            session_data: Session data dictionary
            
        Returns:
            Optimized NetworkX DiGraph
        """
        logger.debug("Building optimized DAG with Context7 enhancements")
        
        # Extract events and create initial graph
        events = session_data.get('events', [])
        if not events:
            logger.warning("No events found in session data")
            return nx.DiGraph()
            
        dag = nx.DiGraph()
        
        # Add nodes efficiently
        self._add_nodes_optimized(dag, events)
        
        # Add edges with optimizations
        if self.opt_config.enable_vectorized_ops:
            self._add_edges_vectorized(dag, events)
        else:
            self._add_edges_standard(dag, events)
            
        # Validate DAG with optimizations
        self._validate_dag_optimized(dag)
        
        # Cache adjacency matrix if enabled
        if self.opt_config.cache_adjacency_matrix and len(dag.nodes()) > 0:
            self._cache_adjacency_matrix(dag)
            
        return dag
        
    def _add_nodes_optimized(self, dag: nx.DiGraph, events: List[Dict[str, Any]]):
        """Add nodes to DAG with optimized batch operations"""
        
        # Context7: Batch operations for better performance
        node_data = []
        for i, event in enumerate(events):
            node_feature = RichNodeFeature()
            
            # Set event-specific features efficiently
            event_type = event.get('event_type', 'unknown')
            if event_type in ['fvg_redelivery', 'expansion_phase', 'consolidation', 
                             'retracement', 'reversal', 'liq_sweep', 'pd_array_interaction']:
                node_feature.set_semantic_event(f"{event_type}_flag", 1.0)
            
            # Set traditional features for price, volume etc.
            traditional_features = torch.zeros(37)
            if 'price_level' in event:
                traditional_features[0] = event['price_level']
            if 'volume_profile' in event:
                traditional_features[1] = event['volume_profile']
            node_feature.set_traditional_features(traditional_features)
            
            node_data.append((i, {
                'event_data': event,
                'node_features': node_feature.features.numpy(),
                'timestamp_et': event.get('timestamp_et'),
                'event_type': event.get('event_type')
            }))
            
        # Batch add nodes
        dag.add_nodes_from(node_data)
        
    def _add_edges_vectorized(self, dag: nx.DiGraph, events: List[Dict[str, Any]]):
        """Add edges using vectorized operations for better performance"""
        
        if len(events) < 2:
            return
            
        n_events = len(events)
        
        # Extract timestamps efficiently
        timestamps = np.array([
            event.get('timestamp_et').timestamp() if event.get('timestamp_et') else 0
            for event in events
        ])
        
        # Sort by timestamp to ensure DAG property (Context7: vectorized sorting)
        sorted_indices = np.argsort(timestamps)
        
        # Vectorized edge creation
        edges_to_add = []
        k_successors = self.dag_config.get('k_successors', 4)
        dt_min_seconds = self.dag_config.get('dt_min_minutes', 1) * 60
        dt_max_seconds = self.dag_config.get('dt_max_minutes', 120) * 60
        
        # Create time difference matrix efficiently
        time_diffs = timestamps[:, np.newaxis] - timestamps[np.newaxis, :]
        
        for i in range(n_events):
            current_idx = sorted_indices[i]
            current_time = timestamps[current_idx]
            
            # Find valid successors using vectorized operations
            later_mask = (timestamps > current_time)
            time_diff_mask = (
                (timestamps - current_time >= dt_min_seconds) &
                (timestamps - current_time <= dt_max_seconds)
            )
            valid_mask = later_mask & time_diff_mask
            
            if not np.any(valid_mask):
                continue
                
            valid_indices = np.where(valid_mask)[0]
            
            # Select k closest successors
            time_diffs_valid = timestamps[valid_indices] - current_time
            closest_k = np.argsort(time_diffs_valid)[:k_successors]
            
            # Add edges with causal strength features
            for successor_idx in valid_indices[closest_k]:
                edge_features = self._compute_edge_features_vectorized(
                    events[current_idx], events[successor_idx],
                    timestamps[successor_idx] - current_time
                )
                
                edges_to_add.append((current_idx, successor_idx, edge_features))
                
        # Batch add edges (Context7: reduce graph modification overhead)
        if self.opt_config.enable_batch_edge_creation and len(edges_to_add) > 10:
            self._batch_add_edges(dag, edges_to_add)
        else:
            for src, dst, attr in edges_to_add:
                dag.add_edge(src, dst, **attr)
                
    def _batch_add_edges(self, dag: nx.DiGraph, edges_to_add: List[Tuple]):
        """Add edges in batches for better performance"""
        
        batch_size = self.opt_config.edge_batch_size
        
        for i in range(0, len(edges_to_add), batch_size):
            batch = edges_to_add[i:i + batch_size]
            
            # Add batch of edges
            dag.add_edges_from([(src, dst, attr) for src, dst, attr in batch])
            
    def _compute_edge_features_vectorized(self, source_event: Dict, target_event: Dict, 
                                        dt_seconds: float) -> Dict[str, Any]:
        """Compute edge features using vectorized operations where possible"""
        
        # Vectorized feature computation
        price_diff = abs(
            target_event.get('price_level', 0) - source_event.get('price_level', 0)
        )
        
        volume_ratio = (
            target_event.get('volume_profile', 1) / 
            max(source_event.get('volume_profile', 1), 1e-8)
        )
        
        # Causal strength based on temporal proximity and event types
        causal_strength = self._compute_causal_strength_vectorized(
            source_event.get('event_type', ''),
            target_event.get('event_type', ''),
            dt_seconds
        )
        
        return {
            'dt_seconds': dt_seconds,
            'price_difference': price_diff,
            'volume_ratio': volume_ratio,
            'causal_strength': causal_strength,
            'edge_features': np.array([dt_seconds/3600, price_diff, volume_ratio, causal_strength])
        }
        
    def _compute_causal_strength_vectorized(self, source_type: str, target_type: str, 
                                          dt_seconds: float) -> float:
        """Compute causal strength using vectorized causality weights"""
        
        causality_weights = self.dag_config.get('causality_weights', {})
        
        # Create causality key
        causality_key = f"{source_type}_to_{target_type}"
        base_weight = causality_weights.get(causality_key, 
                                          causality_weights.get('generic_temporal', 0.4))
        
        # Apply temporal decay (vectorized)
        temporal_decay = np.exp(-dt_seconds / 3600)  # 1-hour half-life
        
        return base_weight * temporal_decay
        
    def _validate_dag_optimized(self, dag: nx.DiGraph):
        """Validate DAG with Context7 optimizations"""
        
        if not dag.nodes():
            return
            
        # Use NetworkX optimized validation
        if not nx.is_directed_acyclic_graph(dag):
            logger.error("Generated graph is not a valid DAG")
            raise ValueError("DAG construction failed: graph contains cycles")
            
        # Context7 recommendation: Use topological_generations for validation
        if self.opt_config.enable_topological_generations:
            try:
                generations = list(nx.topological_generations(dag))
                logger.debug(f"DAG validated with {len(generations)} topological generations")
                
                # Cache topological information
                self._cache_topological_info(dag, generations)
                
            except nx.NetworkXError as e:
                logger.error(f"Topological validation failed: {e}")
                raise
        else:
            # Standard validation
            try:
                topo_sort = list(nx.topological_sort(dag))
                logger.debug(f"DAG validated with {len(topo_sort)} nodes in topological order")
            except nx.NetworkXError as e:
                logger.error(f"Standard topological sort failed: {e}")
                raise
                
    def _cache_topological_info(self, dag: nx.DiGraph, generations: List[List]):
        """Cache topological information for later use"""
        
        dag_id = id(dag)
        self._topo_cache[dag_id] = {
            'generations': generations,
            'num_generations': len(generations),
            'max_generation_size': max(len(gen) for gen in generations) if generations else 0
        }
        
    def _cache_adjacency_matrix(self, dag: nx.DiGraph):
        """Cache adjacency matrix for optimized operations"""
        
        if self._adjacency_cache is None:
            return
            
        dag_id = id(dag)
        
        # Context7: Use sparse arrays for graph data
        # NetworkX adjacency_matrix returns CSR format by default in recent versions
        adj_matrix = nx.adjacency_matrix(dag)
            
        # Cache with size limit
        if len(self._adjacency_cache) >= self.opt_config.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._adjacency_cache))
            del self._adjacency_cache[oldest_key]
            
        self._adjacency_cache[dag_id] = {
            'adjacency_matrix': adj_matrix,
            'num_nodes': len(dag.nodes()),
            'num_edges': len(dag.edges()),
            'density': len(dag.edges()) / (len(dag.nodes()) ** 2) if len(dag.nodes()) > 1 else 0
        }
        
    def get_cached_adjacency(self, dag: nx.DiGraph) -> Optional[sp.csr_matrix]:
        """Get cached adjacency matrix if available"""
        
        if self._adjacency_cache is None:
            return None
            
        dag_id = id(dag)
        cache_entry = self._adjacency_cache.get(dag_id)
        
        if cache_entry:
            return cache_entry['adjacency_matrix']
            
        return None
        
    def get_topological_generations_optimized(self, dag: nx.DiGraph) -> Iterator[List]:
        """Get topological generations with caching optimization"""
        
        # Check cache first
        dag_id = id(dag)
        if dag_id in self._topo_cache:
            cached_info = self._topo_cache[dag_id]
            return iter(cached_info['generations'])
            
        # Context7 recommendation: Use topological_generations
        generations = list(nx.topological_generations(dag))
        
        # Cache for future use
        self._cache_topological_info(dag, generations)
        
        return iter(generations)
        
    def analyze_dag_structure_optimized(self, dag: nx.DiGraph) -> Dict[str, Any]:
        """Analyze DAG structure with optimized operations"""
        
        if not dag.nodes():
            return {'num_nodes': 0, 'num_edges': 0, 'structure': 'empty'}
            
        analysis = {
            'num_nodes': len(dag.nodes()),
            'num_edges': len(dag.edges()),
            'density': len(dag.edges()) / (len(dag.nodes()) ** 2) if len(dag.nodes()) > 1 else 0
        }
        
        # Use cached adjacency matrix if available
        adj_matrix = self.get_cached_adjacency(dag)
        
        if adj_matrix is not None:
            # Optimized analysis using sparse operations
            analysis['max_in_degree'] = int(adj_matrix.sum(axis=0).max())
            analysis['max_out_degree'] = int(adj_matrix.sum(axis=1).max())
            analysis['avg_degree'] = float(adj_matrix.sum() / len(dag.nodes()))
        else:
            # Standard analysis
            in_degrees = [d for n, d in dag.in_degree()]
            out_degrees = [d for n, d in dag.out_degree()]
            
            analysis['max_in_degree'] = max(in_degrees) if in_degrees else 0
            analysis['max_out_degree'] = max(out_degrees) if out_degrees else 0
            analysis['avg_degree'] = sum(in_degrees + out_degrees) / (2 * len(dag.nodes()))
            
        # Topological analysis
        if self.opt_config.enable_topological_generations:
            generations = list(self.get_topological_generations_optimized(dag))
            analysis['num_generations'] = len(generations)
            analysis['max_generation_size'] = max(len(gen) for gen in generations) if generations else 0
            analysis['avg_generation_size'] = sum(len(gen) for gen in generations) / len(generations) if generations else 0
        else:
            analysis['topological_length'] = len(list(nx.topological_sort(dag)))
            
        return analysis
        
    def process_dag_by_generations(self, dag: nx.DiGraph, 
                                 processor_func: callable) -> List[Any]:
        """
        Process DAG nodes by topological generations for optimal parallelization
        
        Args:
            dag: NetworkX DiGraph to process
            processor_func: Function to apply to each generation of nodes
            
        Returns:
            List of processing results per generation
        """
        
        if not dag.nodes():
            return []
            
        results = []
        
        if self.opt_config.enable_topological_generations:
            # Context7: Use topological_generations for layer-wise processing
            for generation_idx, generation in enumerate(self.get_topological_generations_optimized(dag)):
                logger.debug(f"Processing generation {generation_idx} with {len(generation)} nodes")
                
                if self.opt_config.parallel_validation and len(generation) > 10:
                    # Parallel processing within generation
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        generation_results = list(executor.map(processor_func, generation))
                else:
                    generation_results = [processor_func(node) for node in generation]
                    
                results.append(generation_results)
        else:
            # Standard sequential processing
            for node in nx.topological_sort(dag):
                result = processor_func(node) 
                results.append(result)
                
        return results
        
    def _add_edges_standard(self, dag: nx.DiGraph, events: List[Dict[str, Any]]):
        """Standard edge addition for comparison"""
        
        k_successors = self.dag_config.get('k_successors', 4)
        dt_min_minutes = self.dag_config.get('dt_min_minutes', 1)
        dt_max_minutes = self.dag_config.get('dt_max_minutes', 120)
        
        for i, source_event in enumerate(events):
            source_time = source_event.get('timestamp_et')
            if not source_time:
                continue
                
            successors = []
            
            for j, target_event in enumerate(events):
                if i >= j:  # Only forward edges (DAG property)
                    continue
                    
                target_time = target_event.get('timestamp_et')
                if not target_time:
                    continue
                    
                dt_minutes = (target_time - source_time).total_seconds() / 60
                
                if dt_min_minutes <= dt_minutes <= dt_max_minutes:
                    successors.append((j, dt_minutes))
                    
            # Sort by temporal distance and take k closest
            successors.sort(key=lambda x: x[1])
            
            for target_idx, dt_minutes in successors[:k_successors]:
                edge_features = self._compute_edge_features_vectorized(
                    source_event, events[target_idx], dt_minutes * 60
                )
                dag.add_edge(i, target_idx, **edge_features)


def create_optimized_dag_builder(dag_config: Optional[Dict[str, Any]] = None,
                               enable_all_optimizations: bool = True) -> OptimizedDAGBuilder:
    """
    Factory function to create optimized DAG builder
    
    Args:
        dag_config: Base DAG configuration
        enable_all_optimizations: Whether to enable all Context7 optimizations
        
    Returns:
        Optimized DAG builder instance
    """
    
    opt_config = OptimizedDAGConfig(
        base_dag_config=dag_config or {},
        enable_vectorized_ops=enable_all_optimizations,
        enable_topological_generations=enable_all_optimizations,
        enable_sparse_adjacency=enable_all_optimizations,
        enable_batch_edge_creation=enable_all_optimizations,
        parallel_validation=enable_all_optimizations,
        cache_adjacency_matrix=enable_all_optimizations
    )
    
    return OptimizedDAGBuilder(dag_config, opt_config)