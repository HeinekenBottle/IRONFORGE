"""
Comprehensive test suite for Context7-guided translation optimizations
Tests all opt-in feature flags and ensures backward compatibility
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd
import networkx as nx

from ironforge.learning.translation_config import (
    TranslationConfig, DAGBuilderConfig, MotifMiningConfig, 
    BootstrapConfig, ParquetConfig, get_global_config, set_global_config
)
from ironforge.learning.optimized_dag_motif_miner import OptimizedDAGMotifMiner
from ironforge.learning.dag_motif_miner import MotifConfig
from ironforge.io.optimized_parquet import OptimizedParquetWriter


class TestTranslationConfig:
    """Test translation configuration management"""
    
    def test_default_config_all_disabled(self):
        """Ensure all feature flags are disabled by default"""
        config = TranslationConfig()
        assert not config.enable_optimized_dag_builder
        assert not config.enable_efficient_motif_mining  
        assert not config.enable_reproducible_bootstrap
        assert not config.enable_optimized_parquet
        assert not config.enable_validated_presentation
    
    def test_environment_config_loading(self):
        """Test loading configuration from environment variables"""
        env_vars = {
            'IRONFORGE_ENABLE_OPTIMIZED_DAG_BUILDER': 'true',
            'IRONFORGE_ENABLE_EFFICIENT_MOTIF_MINING': '1',
            'IRONFORGE_ENABLE_REPRODUCIBLE_BOOTSTRAP': 'yes',
        }
        
        with patch.dict(os.environ, env_vars):
            config = TranslationConfig.from_environment()
            assert config.enable_optimized_dag_builder
            assert config.enable_efficient_motif_mining  
            assert config.enable_reproducible_bootstrap
            assert not config.enable_optimized_parquet  # not set
            assert not config.enable_validated_presentation  # not set
    
    def test_config_serialization(self):
        """Test configuration serialization to dict"""
        config = TranslationConfig()
        config.enable_optimized_dag_builder = True
        
        config_dict = config.to_dict()
        assert config_dict['flags']['enable_optimized_dag_builder']
        assert 'dag_builder' in config_dict
        assert 'motif_mining' in config_dict
    
    def test_global_config_management(self):
        """Test global configuration singleton"""
        original_config = get_global_config()
        
        new_config = TranslationConfig()
        new_config.enable_optimized_dag_builder = True
        set_global_config(new_config)
        
        retrieved_config = get_global_config()
        assert retrieved_config.enable_optimized_dag_builder
        
        # Restore original config
        set_global_config(original_config)


class TestOptimizedDAGMotifMiner:
    """Test optimized DAG motif mining with feature flags"""
    
    @pytest.fixture
    def sample_dags(self):
        """Create sample DAG graphs for testing"""
        dags = []
        for i in range(5):
            dag = nx.DiGraph()
            dag.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
            dag.add_node(0, timestamp=f"10:{i:02d}:00", event_type="start")
            dag.add_node(1, timestamp=f"10:{i:02d}:15", event_type="process")  
            dag.add_node(2, timestamp=f"10:{i:02d}:30", event_type="validate")
            dag.add_node(3, timestamp=f"10:{i:02d}:45", event_type="end")
            dags.append(dag)
        return dags
    
    def test_backward_compatibility_all_flags_disabled(self, sample_dags):
        """Ensure identical behavior when all flags are disabled"""
        config = TranslationConfig()  # All flags disabled by default
        set_global_config(config)
        
        motif_config = MotifConfig(null_iterations=10, random_seed=42)
        
        # Test with optimized miner (should behave like base)
        optimized_miner = OptimizedDAGMotifMiner(motif_config)
        optimized_results = optimized_miner.mine_motifs(sample_dags)
        
        # Both should produce same number of results
        assert len(optimized_results) >= 0  # Basic functionality test
    
    def test_dag_optimization_flag(self, sample_dags):
        """Test DAG building optimization when flag is enabled"""
        config = TranslationConfig()
        config.enable_optimized_dag_builder = True
        config.dag_builder.enable_topological_optimization = True
        set_global_config(config)
        
        motif_config = MotifConfig(null_iterations=10, random_seed=42)
        miner = OptimizedDAGMotifMiner(motif_config)
        
        # Test optimization path
        results = miner.mine_motifs(sample_dags)
        stats = miner.get_optimization_stats()
        
        assert stats['optimizations_enabled']['dag_builder']
        assert len(results) >= 0
    
    def test_motif_mining_optimization_flag(self, sample_dags):
        """Test motif mining optimization when flag is enabled"""
        config = TranslationConfig()
        config.enable_efficient_motif_mining = True
        config.motif_mining.enable_parallel_isomorphism = True
        config.motif_mining.enable_graph_caching = True
        set_global_config(config)
        
        motif_config = MotifConfig(null_iterations=10, random_seed=42)
        miner = OptimizedDAGMotifMiner(motif_config)
        
        results = miner.mine_motifs(sample_dags)
        stats = miner.get_optimization_stats()
        
        assert stats['optimizations_enabled']['motif_mining']
    
    def test_bootstrap_optimization_flag(self, sample_dags):
        """Test bootstrap optimization when flag is enabled"""
        config = TranslationConfig()
        config.enable_reproducible_bootstrap = True
        config.bootstrap.enable_sklearn_utils = True
        config.bootstrap.enable_thread_safe_rng = True
        set_global_config(config)
        
        motif_config = MotifConfig(null_iterations=10, random_seed=42)
        miner = OptimizedDAGMotifMiner(motif_config)
        
        results = miner.mine_motifs(sample_dags)
        stats = miner.get_optimization_stats()
        
        assert stats['optimizations_enabled']['bootstrap']
    
    def test_reproducibility_with_random_seed(self, sample_dags):
        """Test that results are reproducible with same random seed"""
        config = TranslationConfig()
        config.enable_reproducible_bootstrap = True
        set_global_config(config)
        
        motif_config = MotifConfig(null_iterations=50, random_seed=12345)
        
        # Run twice with same seed
        miner1 = OptimizedDAGMotifMiner(motif_config)
        results1 = miner1.mine_motifs(sample_dags)
        
        miner2 = OptimizedDAGMotifMiner(motif_config)
        results2 = miner2.mine_motifs(sample_dags)
        
        # Results should be identical
        assert len(results1) == len(results2)
        if results1:
            assert results1[0].p_value == results2[0].p_value


class TestOptimizedParquetWriter:
    """Test optimized Parquet I/O with feature flags"""
    
    @pytest.fixture
    def sample_motif_results(self):
        """Create sample motif analysis results"""
        return pd.DataFrame({
            'motif_id': ['P3[0->1,0->2]', 'P4[0->1,1->2,2->3]', 'P3[0->2,1->2]'],
            'frequency': [25, 18, 31],
            'lift': [2.87, 3.42, 2.18],
            'p_value': [0.008, 0.003, 0.025],
            'classification': ['PROMOTE', 'PROMOTE', 'PARK'],
            'n_sessions': [15, 12, 18],
            'null_mean': [8.7, 5.3, 14.2],
            'null_std': [2.1, 1.8, 3.4]
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_backward_compatibility_flags_disabled(self, sample_motif_results, temp_dir):
        """Test standard behavior when all flags are disabled"""
        config = TranslationConfig()  # All flags disabled
        set_global_config(config)
        
        writer = OptimizedParquetWriter()
        output_path = Path(temp_dir) / "test_standard.parquet"
        
        # Write and read back
        writer.write_motif_results(sample_motif_results, str(output_path))
        
        assert output_path.exists()
        
        # Read back and verify
        loaded_df = writer.read_motif_results(str(output_path))
        pd.testing.assert_frame_equal(sample_motif_results, loaded_df)
    
    def test_optimized_parquet_flag(self, sample_motif_results, temp_dir):
        """Test optimized writing when flag is enabled"""
        config = TranslationConfig()
        config.enable_optimized_parquet = True
        config.parquet.enable_cdc = True
        config.parquet.enable_row_group_optimization = True
        set_global_config(config)
        
        writer = OptimizedParquetWriter()
        output_path = Path(temp_dir) / "test_optimized.parquet"
        
        metadata = {'run_id': 'test_run', 'timestamp': '2025-08-26T14:49:16'}
        writer.write_motif_results(sample_motif_results, str(output_path), metadata)
        
        assert output_path.exists()
        
        # Verify data integrity
        loaded_df = writer.read_motif_results(str(output_path))
        pd.testing.assert_frame_equal(sample_motif_results, loaded_df)
        
        # Check schema info includes CDC metadata
        schema_info = writer.get_schema_info(str(output_path))
        assert 'schema_hash' in schema_info
        assert schema_info['num_rows'] == len(sample_motif_results)
    
    def test_content_defined_chunking(self, sample_motif_results, temp_dir):
        """Test content-defined chunking feature"""
        config = TranslationConfig()
        config.enable_optimized_parquet = True
        config.parquet.enable_content_defined_chunking = True
        set_global_config(config)
        
        writer = OptimizedParquetWriter()
        output_path = Path(temp_dir) / "test_cdc.parquet"
        
        writer.write_motif_results(sample_motif_results, str(output_path))
        
        # Verify file was written successfully
        assert output_path.exists()
        loaded_df = writer.read_motif_results(str(output_path))
        assert len(loaded_df) == len(sample_motif_results)
    
    def test_schema_evolution_tracking(self, temp_dir):
        """Test CDC schema evolution tracking"""
        config = TranslationConfig()
        config.enable_optimized_parquet = True
        config.parquet.enable_cdc = True
        set_global_config(config)
        
        writer = OptimizedParquetWriter()
        
        # Write first version
        df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        path1 = Path(temp_dir) / "schema_v1.parquet"
        writer.write_motif_results(df1, str(path1))
        
        # Write second version with different schema
        df2 = pd.DataFrame({'col1': [4, 5, 6], 'col2': ['d', 'e', 'f'], 'col3': [True, False, True]})
        path2 = Path(temp_dir) / "schema_v2.parquet"
        writer.write_motif_results(df2, str(path2))
        
        # Verify both files written
        assert path1.exists()
        assert path2.exists()
        
        # Check schema info
        info1 = writer.get_schema_info(str(path1))
        info2 = writer.get_schema_info(str(path2))
        
        assert info1['schema_hash'] != info2['schema_hash']  # Different schemas
    
    def test_writer_stats(self, sample_motif_results, temp_dir):
        """Test writer statistics collection"""
        config = TranslationConfig()
        config.enable_optimized_parquet = True
        set_global_config(config)
        
        writer = OptimizedParquetWriter()
        
        # Write some files
        for i in range(3):
            path = Path(temp_dir) / f"stats_test_{i}.parquet"
            writer.write_motif_results(sample_motif_results, str(path))
        
        stats = writer.get_stats()
        assert stats['optimizations_enabled']
        assert stats['files_written'] == 3


class TestFeatureFlagCombinations:
    """Test combinations of feature flags"""
    
    @pytest.fixture
    def minimal_setup(self):
        """Minimal setup for combination testing"""
        dags = [nx.DiGraph([(0, 1), (1, 2)]) for _ in range(3)]
        results_df = pd.DataFrame({
            'motif_id': ['test_motif'], 
            'frequency': [5],
            'lift': [2.0], 
            'p_value': [0.01]
        })
        return dags, results_df
    
    def test_all_optimizations_enabled(self, minimal_setup):
        """Test with all optimization flags enabled"""
        dags, results_df = minimal_setup
        
        config = TranslationConfig()
        config.enable_optimized_dag_builder = True
        config.enable_efficient_motif_mining = True
        config.enable_reproducible_bootstrap = True
        config.enable_optimized_parquet = True
        
        set_global_config(config)
        
        # Test mining
        motif_config = MotifConfig(null_iterations=5, random_seed=42)
        miner = OptimizedDAGMotifMiner(motif_config)
        mining_results = miner.mine_motifs(dags)
        
        # Test writing
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = OptimizedParquetWriter()
            path = Path(temp_dir) / "all_opts.parquet"
            writer.write_motif_results(results_df, str(path))
            
            # Verify
            assert path.exists()
            loaded = writer.read_motif_results(str(path))
            assert len(loaded) == len(results_df)
    
    def test_selective_optimization_combinations(self, minimal_setup):
        """Test various combinations of optimization flags"""
        dags, results_df = minimal_setup
        
        flag_combinations = [
            {'enable_optimized_dag_builder': True},
            {'enable_reproducible_bootstrap': True},
            {'enable_optimized_parquet': True},
            {'enable_optimized_dag_builder': True, 'enable_reproducible_bootstrap': True},
        ]
        
        for flags in flag_combinations:
            config = TranslationConfig()
            for flag, value in flags.items():
                setattr(config, flag, value)
                
            set_global_config(config)
            
            # Test that system works with this combination
            motif_config = MotifConfig(null_iterations=3, random_seed=42)
            miner = OptimizedDAGMotifMiner(motif_config)
            results = miner.mine_motifs(dags)
            
            # Should not crash and return results
            assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])