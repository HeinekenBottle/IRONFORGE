"""
Unit tests for motif statistics and null model validation
Tests statistical significance calculations and null model generation
"""

import pytest
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import copy

# Import the core motif mining functions
from ironforge.learning.dag_motif_miner import DAGMotifMiner, MotifResult


class TestMotifStatistics:
    """Test motif statistics calculations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.miner = DAGMotifMiner()
        
        # Create sample DAGs with known patterns
        self.sample_dags = self._create_sample_dags()
        self.session_names = [f"session_{i}" for i in range(len(self.sample_dags))]
    
    def _create_sample_dags(self) -> List[nx.DiGraph]:
        """Create sample DAGs with identifiable patterns"""
        dags = []
        
        for i in range(8):  # 8 sample DAGs
            dag = nx.DiGraph()
            
            # Add nodes with realistic timestamps
            base_time = 1642140000 + i * 10800  # 3 hours apart per session
            for j in range(6):
                dag.add_node(j, timestamp=base_time + j * 300)  # 5 min intervals
            
            # Create common triangle motif (appears in all DAGs)
            dag.add_edge(0, 1, dt_minutes=5)
            dag.add_edge(1, 2, dt_minutes=5) 
            dag.add_edge(0, 2, dt_minutes=10)  # Skip connection
            
            # Add sequential chain
            dag.add_edge(2, 3, dt_minutes=5)
            dag.add_edge(3, 4, dt_minutes=5)
            
            # Add variable pattern (only in some DAGs)
            if i % 3 == 0:  # Appears in 3 out of 8 DAGs
                dag.add_edge(1, 4, dt_minutes=15)  # Long skip connection
                dag.add_edge(4, 5, dt_minutes=5)
            
            dags.append(dag)
        
        return dags
    
    def test_lift_ratio_calculation(self):
        """Test lift ratio computation with various scenarios"""
        
        # Normal case
        real_count = 10
        null_counts = [2, 3, 1, 4, 2, 3, 2, 1, 3, 2]  # Mean = 2.3
        expected_lift = real_count / np.mean(null_counts)
        
        lift_ratio = self.miner._calculate_lift_ratio(real_count, null_counts)
        assert abs(lift_ratio - expected_lift) < 1e-6
        
        # Edge case: zero null counts
        zero_nulls = [0] * 10
        lift_ratio_zero = self.miner._calculate_lift_ratio(real_count, zero_nulls)
        assert lift_ratio_zero == float('inf')
        
        # Edge case: real count = 0
        zero_real = 0
        normal_nulls = [2, 3, 2, 4, 3]
        lift_zero_real = self.miner._calculate_lift_ratio(zero_real, normal_nulls)
        assert lift_zero_real == 0.0
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval computation"""
        
        # Test with known distribution
        null_counts = [10, 12, 8, 14, 9, 11, 13, 7, 15, 10]
        
        ci_lower, ci_upper = self.miner._calculate_confidence_interval(null_counts, confidence=0.95)
        
        # Basic sanity checks
        assert ci_lower <= ci_upper
        assert ci_lower >= 0  # Counts cannot be negative
        
        # Should be within the range of observed values (with some tolerance)
        min_count, max_count = min(null_counts), max(null_counts)
        assert ci_lower <= max_count
        assert ci_upper >= min_count
    
    def test_p_value_calculation(self):
        """Test p-value calculation"""
        
        # Case where real count is clearly higher than nulls
        real_count = 20
        low_null_counts = [2, 3, 1, 4, 2, 3, 1, 2, 4, 3]  # All much lower
        
        p_value_high = self.miner._calculate_p_value(real_count, low_null_counts)
        assert p_value_high < 0.05, f"P-value {p_value_high} should be significant"
        
        # Case where real count is typical
        typical_count = 3
        typical_nulls = [2, 3, 4, 3, 2, 4, 3, 2, 3, 4]
        
        p_value_typical = self.miner._calculate_p_value(typical_count, typical_nulls)
        assert p_value_typical > 0.05, f"P-value {p_value_typical} should not be significant"
        
        # Case where real count is lower than nulls
        low_count = 1
        high_nulls = [8, 9, 10, 9, 8, 10, 9, 8, 9, 10]
        
        p_value_low = self.miner._calculate_p_value(low_count, high_nulls)
        # Could be significant in the opposite direction, but >= 0
        assert 0 <= p_value_low <= 1
    
    def test_motif_classification(self):
        """Test motif classification logic"""
        
        # PROMOTE case: high lift ratio, low p-value
        promote_result = MotifResult(
            pattern_id="promote_pattern",
            real_count=15,
            null_mean=3.0,
            null_std=1.0,
            lift_ratio=5.0,  # >= 2.0
            p_value=0.005,   # < 0.01
            confidence_interval=(2.0, 4.0),
            classification="PROMOTE"
        )
        
        assert promote_result.classification == "PROMOTE"
        assert promote_result.lift_ratio >= 2.0
        assert promote_result.p_value < 0.01
        
        # PARK case: moderate significance
        park_result = MotifResult(
            pattern_id="park_pattern", 
            real_count=8,
            null_mean=4.0,
            null_std=1.5,
            lift_ratio=2.0,
            p_value=0.03,  # < 0.05 but >= 0.01
            confidence_interval=(2.5, 5.5),
            classification="PARK"
        )
        
        assert park_result.classification == "PARK"
        assert park_result.p_value < 0.05
        
        # DISCARD case: not significant
        discard_result = MotifResult(
            pattern_id="discard_pattern",
            real_count=5,
            null_mean=4.8,
            null_std=2.0, 
            lift_ratio=1.04,  # < 2.0
            p_value=0.15,     # >= 0.05
            confidence_interval=(2.0, 7.0),
            classification="DISCARD"
        )
        
        assert discard_result.classification == "DISCARD"
        assert discard_result.p_value >= 0.05
    
    def test_time_jitter_null_generation(self):
        """Test time-jitter null model generation"""
        
        # Generate jittered nulls
        jitter_nulls = self.miner._generate_time_jitter_nulls(
            self.sample_dags, self.session_names, n_nulls=5, jitter_range_minutes=120
        )
        
        assert len(jitter_nulls) == 5
        
        for null_dag, session_name in jitter_nulls:
            # Should have same number of nodes as original
            original_dag = self.sample_dags[0]  # Compare to first DAG
            assert null_dag.number_of_nodes() == original_dag.number_of_nodes()
            
            # Should still be acyclic
            assert nx.is_directed_acyclic_graph(null_dag)
            
            # Timestamps should be jittered but still ordered
            timestamps = [null_dag.nodes[n]['timestamp'] for n in null_dag.nodes()]
            sorted_timestamps = sorted(timestamps)
            assert timestamps == sorted_timestamps or len(set(timestamps)) < len(timestamps)  # Allow ties
    
    def test_session_permutation_null_generation(self):
        """Test session permutation null model generation"""
        
        # Generate permutation nulls
        perm_nulls = self.miner._generate_session_permutation_nulls(
            self.sample_dags, self.session_names, n_nulls=5
        )
        
        assert len(perm_nulls) == 5
        
        for null_dag, session_name in perm_nulls:
            # Should be a valid DAG
            assert isinstance(null_dag, nx.DiGraph)
            assert nx.is_directed_acyclic_graph(null_dag)
            
            # Should have realistic structure (not empty unless originals were empty)
            if any(dag.number_of_nodes() > 0 for dag in self.sample_dags):
                assert null_dag.number_of_nodes() > 0
    
    def test_motif_enumeration(self):
        """Test motif pattern enumeration"""
        
        # Simple test: count triangles in our sample DAGs
        triangles_per_dag = []
        
        for dag in self.sample_dags:
            triangle_count = 0
            nodes = list(dag.nodes())
            
            # Count triangles (3-node cycles)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    for k in range(j+1, len(nodes)):
                        node_set = {nodes[i], nodes[j], nodes[k]}
                        subgraph = dag.subgraph(node_set)
                        
                        # Check if it forms a triangle pattern (all connected)
                        if subgraph.number_of_edges() >= 2:  # At least 2 edges
                            triangle_count += 1
            
            triangles_per_dag.append(triangle_count)
        
        # Should find triangles in all DAGs (based on our construction)
        assert all(count > 0 for count in triangles_per_dag), f"Triangle counts: {triangles_per_dag}"
    
    def test_statistical_significance_detection(self):
        """Test that we can detect statistically significant patterns"""
        
        # Create DAGs with a very obvious pattern
        significant_dags = []
        
        for i in range(10):
            dag = nx.DiGraph()
            base_time = 1642140000 + i * 3600
            
            # Add consistent significant pattern in all DAGs
            for j in range(4):
                dag.add_node(j, timestamp=base_time + j * 300)
            
            # Always add this exact pattern
            dag.add_edge(0, 1, dt_minutes=5)
            dag.add_edge(1, 2, dt_minutes=5)
            dag.add_edge(2, 3, dt_minutes=5)
            dag.add_edge(0, 3, dt_minutes=15)  # Skip connection
            
            significant_dags.append(dag)
        
        # Mine motifs
        motifs = self.miner.discover_motifs(significant_dags, [f"sig_{i}" for i in range(10)])
        
        # Should detect at least one significant pattern
        promote_motifs = [m for m in motifs.values() if m.classification == "PROMOTE"]
        assert len(promote_motifs) > 0, "Should detect at least one PROMOTE pattern"
        
        # Check the most significant one
        best_motif = max(motifs.values(), key=lambda m: m.lift_ratio)
        assert best_motif.lift_ratio > 1.5, f"Best motif lift ratio {best_motif.lift_ratio} too low"
    
    def test_null_model_preserves_structure(self):
        """Test that null models preserve basic graph structure"""
        
        original_dag = self.sample_dags[0]
        
        # Time jitter nulls
        jitter_nulls = self.miner._generate_time_jitter_nulls(
            [original_dag], ["test"], n_nulls=3, jitter_range_minutes=60
        )
        
        for null_dag, _ in jitter_nulls:
            # Same number of nodes and edges
            assert null_dag.number_of_nodes() == original_dag.number_of_nodes()
            assert null_dag.number_of_edges() == original_dag.number_of_edges()
            
            # Preserves acyclicity
            assert nx.is_directed_acyclic_graph(null_dag)


def test_standalone_functions():
    """Test standalone utility functions"""
    
    # Test motif result creation
    result = MotifResult(
        pattern_id="test_pattern",
        real_count=10,
        null_mean=5.0,
        null_std=1.5,
        lift_ratio=2.0,
        p_value=0.02,
        confidence_interval=(3.5, 6.5),
        classification="PARK"
    )
    
    assert result.pattern_id == "test_pattern"
    assert result.real_count == 10
    assert result.classification == "PARK"
    
    print("✅ Standalone functions test passed")


if __name__ == "__main__":
    # Run tests
    test_instance = TestMotifStatistics()
    test_methods = [
        test_instance.test_lift_ratio_calculation,
        test_instance.test_confidence_interval_calculation,
        test_instance.test_p_value_calculation,
        test_instance.test_motif_classification,
        test_instance.test_time_jitter_null_generation,
        test_instance.test_session_permutation_null_generation,
        test_instance.test_motif_enumeration,
        test_instance.test_statistical_significance_detection,
        test_instance.test_null_model_preserves_structure,
        test_standalone_functions
    ]
    
    for test_method in test_methods:
        try:
            print(f"Running {test_method.__name__}...")
            test_instance.setup_method()  # Reset for each test
            test_method()
            print(f"✅ {test_method.__name__} passed")
        except Exception as e:
            print(f"❌ {test_method.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nMotif statistics tests completed!")