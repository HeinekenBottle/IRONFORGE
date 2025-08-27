"""
DAG Motif Miner with Statistical Nulls
Mines frequent patterns in DAG graphs with rigorous statistical validation
Implements time-jitter and session permutation null models
"""

import logging
import random
import warnings
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Iterator

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
from networkx.algorithms import isomorphism

logger = logging.getLogger(__name__)


@dataclass
class MotifResult:
    """Results for a discovered motif pattern"""
    motif_id: str
    graph: nx.DiGraph
    frequency: int
    lift: float
    confidence_interval: Tuple[float, float]  # 95% CI
    p_value: float
    null_frequency_mean: float
    null_frequency_std: float
    classification: str  # 'PROMOTE', 'PARK', 'DISCARD'
    sessions_found: Set[str]
    instances: List[Dict[str, Any]]
    
    
@dataclass  
class MotifConfig:
    """Configuration for DAG motif mining"""
    min_nodes: int = 3
    max_nodes: int = 5
    min_frequency: int = 3
    max_motifs: int = 100
    significance_threshold: float = 0.05
    lift_threshold: float = 1.5
    confidence_level: float = 0.95
    null_iterations: int = 1000
    time_jitter_min: int = 60  # minutes
    time_jitter_max: int = 120  # minutes
    enable_time_jitter: bool = True
    enable_session_permutation: bool = True
    random_seed: Optional[int] = None


class DAGMotifMiner:
    """
    DAG Motif Miner with Statistical Validation
    
    Discovers frequent subgraph patterns in DAG graphs and validates 
    their significance using null models:
    - Time-jitter nulls: randomize timestamps within ±60-120 minutes
    - Session permutation nulls: permute events within sessions
    """
    
    def __init__(self, config: MotifConfig = None):
        self.config = config or MotifConfig()
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DAG Motif Miner initialized: {self.config}")
        
        # Pattern storage
        self.discovered_motifs: List[MotifResult] = []
        self.canonical_patterns: Dict[str, nx.DiGraph] = {}
        
    def mine_motifs(self, dags: List[nx.DiGraph], session_names: List[str] = None) -> List[MotifResult]:
        """
        Mine significant motifs from a collection of DAG graphs
        
        Args:
            dags: List of NetworkX DiGraph objects
            session_names: Optional list of session names for each DAG
            
        Returns:
            List of significant MotifResult objects
        """
        if not dags:
            self.logger.warning("No DAGs provided for motif mining")
            return []
            
        session_names = session_names or [f"session_{i}" for i in range(len(dags))]
        
        self.logger.info(f"Mining motifs from {len(dags)} DAGs")
        
        # Step 1: Extract all subgraph patterns
        pattern_instances = self._extract_patterns(dags, session_names)
        
        self.logger.info(f"Found {len(pattern_instances)} unique pattern types")
        
        # Step 2: Filter by minimum frequency
        frequent_patterns = {
            pattern_id: instances for pattern_id, instances in pattern_instances.items()
            if len(instances) >= self.config.min_frequency
        }
        
        self.logger.info(f"Filtered to {len(frequent_patterns)} frequent patterns")
        
        # Step 3: Generate null models and test significance
        significant_motifs = []
        
        for i, (pattern_id, instances) in enumerate(frequent_patterns.items()):
            if i >= self.config.max_motifs:
                self.logger.info(f"Reached max motifs limit ({self.config.max_motifs})")
                break
                
            self.logger.debug(f"Testing pattern {i+1}/{len(frequent_patterns)}: {pattern_id}")
            
            motif_result = self._test_motif_significance(
                pattern_id, instances, dags, session_names
            )
            
            if motif_result.classification != 'DISCARD':
                significant_motifs.append(motif_result)
                
        # Step 4: Sort by lift and classify
        significant_motifs.sort(key=lambda x: x.lift, reverse=True)
        
        self.discovered_motifs = significant_motifs
        self.logger.info(f"Discovered {len(significant_motifs)} significant motifs")
        
        return significant_motifs
        
    def _extract_patterns(self, dags: List[nx.DiGraph], session_names: List[str]) -> Dict[str, List[Dict]]:
        """Extract all subgraph patterns of specified sizes"""
        pattern_instances = defaultdict(list)
        
        for dag_idx, dag in enumerate(dags):
            session_name = session_names[dag_idx]
            
            if dag.number_of_nodes() < self.config.min_nodes:
                continue
                
            # Extract patterns of different sizes
            for pattern_size in range(self.config.min_nodes, self.config.max_nodes + 1):
                patterns = self._extract_patterns_of_size(dag, pattern_size, session_name)
                
                for pattern_id, instance_data in patterns.items():
                    pattern_instances[pattern_id].append({
                        'dag_idx': dag_idx,
                        'session_name': session_name,
                        'nodes': instance_data['nodes'],
                        'edges': instance_data['edges'],
                        'timestamps': instance_data.get('timestamps', []),
                        'features': instance_data.get('features', {})
                    })
                    
        return dict(pattern_instances)
        
    def _extract_patterns_of_size(self, dag: nx.DiGraph, size: int, session_name: str) -> Dict[str, Dict]:
        """Extract all connected subgraph patterns of a specific size"""
        patterns = {}
        
        # Get all connected subgraphs of the specified size
        for node_set in self._enumerate_connected_subgraphs(dag, size):
            subgraph = dag.subgraph(node_set)
            
            # Create canonical representation
            pattern_id = self._canonicalize_pattern(subgraph)
            
            # Extract instance data
            instance_data = {
                'nodes': list(node_set),
                'edges': list(subgraph.edges()),
                'timestamps': [dag.nodes[n].get('timestamp', 0) for n in node_set],
                'features': {n: dag.nodes[n].get('feature', None) for n in node_set}
            }
            
            patterns[pattern_id] = instance_data
            
        return patterns
        
    def _enumerate_connected_subgraphs(self, dag: nx.DiGraph, size: int) -> Iterator[Set[int]]:
        """Enumerate all connected subgraphs of a given size"""
        from itertools import combinations
        
        nodes = list(dag.nodes())
        
        # Generate all combinations of nodes of the specified size
        for node_combination in combinations(nodes, size):
            subgraph = dag.subgraph(node_combination)
            
            # Check if subgraph is connected (treating as undirected for connectivity)
            if nx.is_weakly_connected(subgraph):
                yield set(node_combination)
                
    def _canonicalize_pattern(self, subgraph: nx.DiGraph) -> str:
        """Create a canonical string representation of a subgraph pattern"""
        # Create a simplified graph with just the structure
        canonical_graph = nx.DiGraph()
        
        # Map nodes to canonical IDs (0, 1, 2, ...)
        node_mapping = {node: i for i, node in enumerate(sorted(subgraph.nodes()))}
        
        # Add canonical edges
        canonical_edges = []
        for src, dst in subgraph.edges():
            canonical_src = node_mapping[src]
            canonical_dst = node_mapping[dst]
            canonical_edges.append((canonical_src, canonical_dst))
            
        canonical_graph.add_edges_from(canonical_edges)
        
        # Create string representation
        edges_str = ','.join(f"{src}->{dst}" for src, dst in sorted(canonical_edges))
        pattern_id = f"P{len(canonical_graph.nodes())}[{edges_str}]"
        
        # Store the canonical pattern
        self.canonical_patterns[pattern_id] = canonical_graph
        
        return pattern_id
        
    def _test_motif_significance(
        self, 
        pattern_id: str, 
        instances: List[Dict], 
        dags: List[nx.DiGraph], 
        session_names: List[str]
    ) -> MotifResult:
        """Test statistical significance of a motif pattern"""
        
        # Observed frequency
        observed_freq = len(instances)
        sessions_found = set(inst['session_name'] for inst in instances)
        
        # Generate null frequencies
        null_frequencies = []
        
        # Time-jitter nulls
        if self.config.enable_time_jitter:
            time_jitter_nulls = self._generate_time_jitter_nulls(
                pattern_id, dags, session_names, self.config.null_iterations // 2
            )
            null_frequencies.extend(time_jitter_nulls)
            
        # Session permutation nulls  
        if self.config.enable_session_permutation:
            permutation_nulls = self._generate_session_permutation_nulls(
                pattern_id, dags, session_names, self.config.null_iterations // 2
            )
            null_frequencies.extend(permutation_nulls)
            
        if not null_frequencies:
            # Fallback to basic random nulls if no null methods enabled
            null_frequencies = [random.randint(0, observed_freq * 2) for _ in range(self.config.null_iterations)]
            
        # Calculate statistics
        null_mean = np.mean(null_frequencies)
        null_std = np.std(null_frequencies)
        
        # Calculate lift
        lift = observed_freq / max(null_mean, 0.1)  # Avoid division by zero
        
        # Calculate p-value (one-tailed test)
        if null_std > 0:
            z_score = (observed_freq - null_mean) / null_std
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            p_value = 0.0 if observed_freq > null_mean else 1.0
            
        # Calculate confidence interval for lift
        if null_std > 0:
            alpha = 1 - self.config.confidence_level
            margin = stats.norm.ppf(1 - alpha/2) * (null_std / max(null_mean, 0.1))
            lift_lower = max(0, lift - margin)
            lift_upper = lift + margin
        else:
            lift_lower = lift_upper = lift
            
        confidence_interval = (lift_lower, lift_upper)
        
        # Classify motif
        classification = self._classify_motif(lift, p_value, confidence_interval)
        
        # Create result
        result = MotifResult(
            motif_id=pattern_id,
            graph=self.canonical_patterns[pattern_id].copy(),
            frequency=observed_freq,
            lift=lift,
            confidence_interval=confidence_interval,
            p_value=p_value,
            null_frequency_mean=null_mean,
            null_frequency_std=null_std,
            classification=classification,
            sessions_found=sessions_found,
            instances=instances
        )
        
        self.logger.debug(
            f"Motif {pattern_id}: freq={observed_freq}, lift={lift:.2f}, "
            f"p={p_value:.4f}, class={classification}"
        )
        
        return result
        
    def _generate_time_jitter_nulls(
        self, 
        pattern_id: str, 
        dags: List[nx.DiGraph], 
        session_names: List[str], 
        n_iterations: int
    ) -> List[int]:
        """Generate null frequencies using time-jitter randomization"""
        null_frequencies = []
        
        for _ in range(n_iterations):
            # Create time-jittered versions of the DAGs
            jittered_dags = []
            
            for dag in dags:
                jittered_dag = self._apply_time_jitter(dag)
                jittered_dags.append(jittered_dag)
                
            # Count pattern occurrences in jittered DAGs
            pattern_instances = self._extract_patterns(jittered_dags, session_names)
            null_freq = len(pattern_instances.get(pattern_id, []))
            null_frequencies.append(null_freq)
            
        return null_frequencies
        
    def _apply_time_jitter(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Apply random time jitter to DAG timestamps"""
        jittered_dag = dag.copy()
        
        # Get all timestamps
        timestamps = [jittered_dag.nodes[n].get('timestamp', 0) for n in jittered_dag.nodes()]
        
        if not timestamps:
            return jittered_dag
            
        # Apply jitter to each node
        for node in jittered_dag.nodes():
            original_timestamp = jittered_dag.nodes[node].get('timestamp', 0)
            
            # Random jitter in seconds
            jitter_seconds = random.randint(
                self.config.time_jitter_min * 60,
                self.config.time_jitter_max * 60
            )
            
            # Randomly add or subtract jitter
            if random.random() < 0.5:
                jitter_seconds = -jitter_seconds
                
            jittered_timestamp = original_timestamp + jitter_seconds
            jittered_dag.nodes[node]['timestamp'] = max(0, jittered_timestamp)
            
        # Rebuild edges based on new timestamps (maintain DAG property)
        jittered_dag = self._rebuild_dag_from_jittered_timestamps(jittered_dag)
        
        return jittered_dag
        
    def _rebuild_dag_from_jittered_timestamps(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Rebuild DAG edges after timestamp jittering to maintain causality"""
        # Get nodes sorted by new timestamps
        nodes_with_timestamps = [
            (n, dag.nodes[n].get('timestamp', 0)) for n in dag.nodes()
        ]
        nodes_with_timestamps.sort(key=lambda x: x[1])
        
        # Create new DAG with same nodes but rebuilt edges
        new_dag = nx.DiGraph()
        
        # Add all nodes with their data
        for node, timestamp in nodes_with_timestamps:
            new_dag.add_node(node, **dag.nodes[node])
            
        # Rebuild edges following temporal order
        for i, (src_node, src_timestamp) in enumerate(nodes_with_timestamps):
            connections_made = 0
            max_connections = 4  # Same as original DAG builder
            
            for j in range(i + 1, len(nodes_with_timestamps)):
                if connections_made >= max_connections:
                    break
                    
                dst_node, dst_timestamp = nodes_with_timestamps[j]
                
                # Check if there was an edge in the original DAG (or similar nodes)
                # For simplicity, we'll create edges based on temporal proximity
                time_diff = dst_timestamp - src_timestamp
                
                if 60 <= time_diff <= 7200:  # 1 minute to 2 hours
                    # Copy edge data if it existed in original
                    if dag.has_edge(src_node, dst_node):
                        edge_data = dag.edges[src_node, dst_node].copy()
                    else:
                        # Create minimal edge data
                        edge_data = {
                            'dt_minutes': time_diff / 60.0,
                            'reason': 'JITTER_REBUILD',
                            'weight': 0.5
                        }
                        
                    new_dag.add_edge(src_node, dst_node, **edge_data)
                    connections_made += 1
                    
        return new_dag
        
    def _generate_session_permutation_nulls(
        self, 
        pattern_id: str, 
        dags: List[nx.DiGraph], 
        session_names: List[str], 
        n_iterations: int
    ) -> List[int]:
        """Generate null frequencies using session permutation"""
        null_frequencies = []
        
        for _ in range(n_iterations):
            # Create permuted versions of the DAGs
            permuted_dags = []
            
            for dag in dags:
                permuted_dag = self._apply_session_permutation(dag)
                permuted_dags.append(permuted_dag)
                
            # Count pattern occurrences in permuted DAGs
            pattern_instances = self._extract_patterns(permuted_dags, session_names)
            null_freq = len(pattern_instances.get(pattern_id, []))
            null_frequencies.append(null_freq)
            
        return null_frequencies
        
    def _apply_session_permutation(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Apply random permutation to events within a session"""
        if dag.number_of_nodes() < 2:
            return dag.copy()
            
        permuted_dag = nx.DiGraph()
        
        # Get all nodes and their data
        nodes = list(dag.nodes(data=True))
        
        # Shuffle the node data while keeping the same node IDs
        node_data_list = [data for _, data in nodes]
        random.shuffle(node_data_list)
        
        # Create permuted nodes (same IDs, shuffled data)
        for (node_id, _), new_data in zip(nodes, node_data_list):
            permuted_dag.add_node(node_id, **new_data)
            
        # Copy original edge structure (just the topology, not the node data)
        for src, dst, edge_data in dag.edges(data=True):
            permuted_dag.add_edge(src, dst, **edge_data)
            
        return permuted_dag
        
    def _classify_motif(self, lift: float, p_value: float, confidence_interval: Tuple[float, float]) -> str:
        """Classify motif as PROMOTE, PARK, or DISCARD"""
        
        # DISCARD: Not statistically significant
        if p_value > self.config.significance_threshold:
            return 'DISCARD'
            
        # DISCARD: Lift is too low
        if lift < self.config.lift_threshold:
            return 'DISCARD'
            
        # DISCARD: Confidence interval includes 1 (no effect)
        if confidence_interval[0] <= 1.0:
            return 'DISCARD'
            
        # PROMOTE: Very strong signal
        if lift >= 2.0 and p_value < 0.01 and confidence_interval[0] >= 1.5:
            return 'PROMOTE'
            
        # PARK: Significant but needs more investigation
        return 'PARK'
        
    def save_results(self, output_path: Path, format: str = 'both'):
        """Save motif mining results to disk"""
        if not self.discovered_motifs:
            self.logger.warning("No motifs to save")
            return
            
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare results DataFrame
        results_data = []
        
        for motif in self.discovered_motifs:
            result_row = {
                'motif_id': motif.motif_id,
                'frequency': motif.frequency,
                'lift': motif.lift,
                'p_value': motif.p_value,
                'lift_lower_95': motif.confidence_interval[0],
                'lift_upper_95': motif.confidence_interval[1],
                'null_mean': motif.null_frequency_mean,
                'null_std': motif.null_frequency_std,
                'classification': motif.classification,
                'n_sessions': len(motif.sessions_found),
                'sessions': ','.join(sorted(motif.sessions_found)),
                'n_nodes': motif.graph.number_of_nodes(),
                'n_edges': motif.graph.number_of_edges(),
                'pattern_structure': str(sorted(motif.graph.edges()))
            }
            results_data.append(result_row)
            
        results_df = pd.DataFrame(results_data)
        
        # Save in requested formats
        if format in ['parquet', 'both']:
            parquet_path = output_path / 'motifs.parquet'
            results_df.to_parquet(parquet_path, compression='zstd')
            self.logger.info(f"Saved motif results to {parquet_path}")
            
        if format in ['csv', 'both']:
            csv_path = output_path / 'motifs.csv'
            results_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved motif results to {csv_path}")
            
        # Save detailed patterns as pickle for further analysis
        patterns_path = output_path / 'motif_patterns.pkl'
        import pickle
        with open(patterns_path, 'wb') as f:
            pickle.dump(self.discovered_motifs, f)
            
        # Generate summary markdown
        self._generate_summary_markdown(output_path / 'motif_summary.md')
        
        self.logger.info(f"Saved {len(self.discovered_motifs)} motifs to {output_path}")
        
    def _generate_summary_markdown(self, output_path: Path):
        """Generate human-readable summary in markdown format"""
        
        promote_motifs = [m for m in self.discovered_motifs if m.classification == 'PROMOTE']
        park_motifs = [m for m in self.discovered_motifs if m.classification == 'PARK']
        
        summary = f"""# DAG Motif Mining Results

## Summary Statistics
- **Total motifs discovered**: {len(self.discovered_motifs)}
- **Promote (high confidence)**: {len(promote_motifs)}
- **Park (investigate further)**: {len(park_motifs)}
- **Configuration**: {dict(vars(self.config))}

## Top Promote Motifs
"""
        
        for i, motif in enumerate(promote_motifs[:10], 1):
            summary += f"""
### {i}. {motif.motif_id}
- **Frequency**: {motif.frequency}
- **Lift**: {motif.lift:.2f} ({motif.confidence_interval[0]:.2f} - {motif.confidence_interval[1]:.2f})
- **P-value**: {motif.p_value:.4f}
- **Sessions**: {len(motif.sessions_found)}
- **Structure**: {motif.graph.number_of_nodes()} nodes, {motif.graph.number_of_edges()} edges
- **Pattern**: {sorted(motif.graph.edges())}
"""

        summary += f"""
## Top Park Motifs
"""
        
        for i, motif in enumerate(park_motifs[:10], 1):
            summary += f"""
### {i}. {motif.motif_id}
- **Frequency**: {motif.frequency}
- **Lift**: {motif.lift:.2f} ({motif.confidence_interval[0]:.2f} - {motif.confidence_interval[1]:.2f})
- **P-value**: {motif.p_value:.4f}
- **Sessions**: {len(motif.sessions_found)}
- **Structure**: {motif.graph.number_of_nodes()} nodes, {motif.graph.number_of_edges()} edges
- **Pattern**: {sorted(motif.graph.edges())}
"""

        with open(output_path, 'w') as f:
            f.write(summary)
            
        self.logger.info(f"Generated summary: {output_path}")


def mine_dag_motifs(
    dag_files: List[Path], 
    output_dir: Path, 
    config: MotifConfig = None
) -> List[MotifResult]:
    """
    Convenience function to mine motifs from DAG pickle files
    
    Args:
        dag_files: List of paths to DAG pickle files
        output_dir: Output directory for results
        config: Mining configuration
        
    Returns:
        List of discovered MotifResult objects
    """
    import pickle
    
    config = config or MotifConfig()
    miner = DAGMotifMiner(config)
    
    # Load DAGs
    dags = []
    session_names = []
    
    for dag_file in dag_files:
        try:
            with open(dag_file, 'rb') as f:
                dag = pickle.load(f)
                dags.append(dag)
                session_names.append(dag_file.stem)
        except Exception as e:
            logger.warning(f"Failed to load DAG from {dag_file}: {e}")
            continue
            
    if not dags:
        logger.error("No DAGs could be loaded")
        return []
        
    # Mine motifs
    motifs = miner.mine_motifs(dags, session_names)
    
    # Save results
    miner.save_results(output_dir)
    
    return motifs


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Mine DAG motifs with statistical validation")
    parser.add_argument("--input-dir", required=True, help="Directory containing DAG pickle files")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--min-freq", type=int, default=3, help="Minimum motif frequency")
    parser.add_argument("--lift-threshold", type=float, default=1.5, help="Minimum lift threshold")
    parser.add_argument("--null-iterations", type=int, default=1000, help="Number of null iterations")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Find DAG files
    input_dir = Path(args.input_dir)
    dag_files = list(input_dir.glob("**/dag_graph.pkl"))
    
    if not dag_files:
        print(f"No DAG files found in {input_dir}")
        exit(1)
        
    print(f"Found {len(dag_files)} DAG files")
    
    # Configure mining
    config = MotifConfig(
        min_frequency=args.min_freq,
        lift_threshold=args.lift_threshold,
        null_iterations=args.null_iterations
    )
    
    # Mine motifs
    motifs = mine_dag_motifs(dag_files, Path(args.output_dir), config)
    
    print(f"\nDiscovered {len(motifs)} significant motifs")
    
    promote_count = len([m for m in motifs if m.classification == 'PROMOTE'])
    park_count = len([m for m in motifs if m.classification == 'PARK'])
    
    print(f"- PROMOTE: {promote_count}")
    print(f"- PARK: {park_count}")
    print(f"\nResults saved to: {args.output_dir}")