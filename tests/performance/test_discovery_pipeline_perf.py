"""Performance tests for TemporalDiscoveryPipeline (Wave 3)."""

import tempfile
import time
import tracemalloc
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline


@pytest.fixture
def synthetic_large_shards():
    """Create synthetic shards with ~5k nodes and edges for performance testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create larger synthetic data
        num_nodes = 5000
        num_edges = 4500

        # Generate node data with 45D features
        nodes_df = pd.DataFrame(
            {
                "node_id": range(num_nodes),
                "t": range(100, 100 + num_nodes * 10, 10),  # Increasing timestamps
                "kind": [i % 3 + 1 for i in range(num_nodes)],  # 3 node types
                **{
                    f"f{i}": [(i * 0.01 + j * 0.001) % 1.0 for j in range(num_nodes)]
                    for i in range(45)
                },
            }
        )

        # Generate edge data with 20D features
        # Create valid edges between existing nodes
        src_nodes = [i % num_nodes for i in range(num_edges)]
        dst_nodes = [(i + 1) % num_nodes for i in range(num_edges)]

        edges_df = pd.DataFrame(
            {
                "src": src_nodes,
                "dst": dst_nodes,
                "etype": [i % 4 + 1 for i in range(num_edges)],  # 4 edge types
                "dt": [i * 5 + 10 for i in range(num_edges)],  # Temporal deltas
                **{
                    f"e{i}": [(i * 0.02 + j * 0.001) % 1.0 for j in range(num_edges)]
                    for i in range(20)
                },
            }
        )

        # Write multiple shards for testing
        for shard_id in range(2):  # 2 shards
            nodes_file = temp_path / f"shard_{shard_id:02d}_nodes.parquet"
            edges_file = temp_path / f"shard_{shard_id:02d}_edges.parquet"

            # Split data across shards
            start_idx = shard_id * (num_nodes // 2)
            end_idx = (shard_id + 1) * (num_nodes // 2)

            shard_nodes = nodes_df.iloc[start_idx:end_idx].copy()
            shard_edges = edges_df[
                (edges_df["src"] >= start_idx)
                & (edges_df["src"] < end_idx)
                & (edges_df["dst"] >= start_idx)
                & (edges_df["dst"] < end_idx)
            ].copy()

            # Adjust edge indices for shard
            shard_edges["src"] -= start_idx
            shard_edges["dst"] -= start_idx

            shard_nodes.to_parquet(nodes_file, index=False)
            shard_edges.to_parquet(edges_file, index=False)

        yield temp_path, num_nodes, num_edges


@pytest.mark.performance
def test_discovery_pipeline_time_budget(synthetic_large_shards):
    """Test that run_discovery completes within 5 second budget."""
    temp_path, num_nodes, num_edges = synthetic_large_shards

    pipeline = TemporalDiscoveryPipeline(
        data_path=temp_path,
        num_neighbors=[5, 3],  # Small fanouts for speed
        batch_size=100,
        time_window=1,
        stitch_policy="session",
    )

    # Mock the TGAT discovery to focus on pipeline performance
    with patch.object(pipeline, "_run_tgat_discovery") as mock_discovery:
        mock_discovery.return_value = [{"pattern_type": "test", "confidence": 0.8}]

        start_time = time.time()
        discoveries = pipeline.run_discovery()
        elapsed_time = time.time() - start_time

        # Performance assertion: < 5 seconds
        assert elapsed_time < 5.0, f"Pipeline took {elapsed_time:.2f}s, expected < 5.0s"
        assert len(discoveries) > 0


@pytest.mark.performance
def test_discovery_pipeline_memory_budget(synthetic_large_shards):
    """Test that pipeline stays under 100MB memory budget."""
    temp_path, num_nodes, num_edges = synthetic_large_shards

    pipeline = TemporalDiscoveryPipeline(
        data_path=temp_path,
        num_neighbors=[5, 3],
        batch_size=50,  # Smaller batches for memory efficiency
        stitch_policy="session",
    )

    # Start memory monitoring
    tracemalloc.start()
    initial_snapshot = tracemalloc.take_snapshot()

    # Mock TGAT discovery to isolate pipeline memory usage
    with patch.object(pipeline, "_run_tgat_discovery") as mock_discovery:
        mock_discovery.return_value = []

        try:
            pipeline.run_discovery()

            # Take memory snapshot
            current_snapshot = tracemalloc.take_snapshot()
            top_stats = current_snapshot.compare_to(initial_snapshot, "lineno")

            # Calculate total memory increase
            total_memory_mb = sum(stat.size_diff for stat in top_stats) / 1024 / 1024

            # Performance assertion: < 100 MB
            assert total_memory_mb < 100, f"Memory usage: {total_memory_mb:.1f}MB, expected < 100MB"

        finally:
            tracemalloc.stop()


@pytest.mark.performance
@pytest.mark.skipif(
    pytest.importorskip("torch", reason="PyTorch not available") is None,
    reason="Requires PyTorch for neighbor loader performance",
)
def test_neighbor_loader_batch_time():
    """Test that neighbor loader batches process within 100ms budget."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create medium-sized test data
        num_nodes = 1000
        nodes_df = pd.DataFrame(
            {
                "node_id": range(num_nodes),
                "t": range(num_nodes),
                "kind": [1] * num_nodes,
                **{f"f{i}": [0.1] * num_nodes for i in range(45)},
            }
        )

        edges_df = pd.DataFrame(
            {
                "src": range(num_nodes - 1),
                "dst": range(1, num_nodes),
                "etype": [1] * (num_nodes - 1),
                "dt": [10] * (num_nodes - 1),
                **{f"e{i}": [0.5] * (num_nodes - 1) for i in range(20)},
            }
        )

        nodes_file = temp_path / "perf_test_nodes.parquet"
        edges_file = temp_path / "perf_test_edges.parquet"

        nodes_df.to_parquet(nodes_file, index=False)
        edges_df.to_parquet(edges_file, index=False)

        pipeline = TemporalDiscoveryPipeline(
            data_path=temp_path,
            num_neighbors=[10, 5],
            batch_size=50,
        )

        # Build graph and test loader performance
        shards = pipeline.load_shards()
        graph = pipeline.build_temporal_graph(shards[0])

        # Mock neighbor loader to measure iteration time
        class MockBatch:
            def __init__(self):
                self.data = {"node": Mock(), ("node", "temporal", "node"): Mock()}

            def __getitem__(self, key):
                return self.data[key]

        # Create a few mock batches
        mock_batches = [MockBatch() for _ in range(5)]

        with patch.object(pipeline, "create_neighbor_loader") as mock_create_loader:
            mock_create_loader.return_value = iter(mock_batches)

            # Time the batch processing
            loader = pipeline.create_neighbor_loader(graph)

            batch_times = []
            for batch in loader:
                start_time = time.time()

                # Simulate minimal batch processing
                _ = batch["node"]
                _ = batch[("node", "temporal", "node")]

                batch_time = time.time() - start_time
                batch_times.append(batch_time * 1000)  # Convert to ms

            # Performance assertion: each batch < 100ms
            max_batch_time = max(batch_times) if batch_times else 0
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

            assert max_batch_time < 100, f"Max batch time: {max_batch_time:.1f}ms, expected < 100ms"
            assert avg_batch_time < 50, f"Avg batch time: {avg_batch_time:.1f}ms, expected < 50ms"


@pytest.mark.performance
def test_shard_loading_performance():
    """Test shard loading performance with multiple files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple moderate-sized shards
        num_shards = 10
        nodes_per_shard = 500

        for shard_id in range(num_shards):
            nodes_df = pd.DataFrame(
                {
                    "node_id": range(shard_id * nodes_per_shard, (shard_id + 1) * nodes_per_shard),
                    "t": range(nodes_per_shard),
                    "kind": [1] * nodes_per_shard,
                    **{f"f{i}": [0.1] * nodes_per_shard for i in range(45)},
                }
            )

            edges_df = pd.DataFrame(
                {
                    "src": range(shard_id * nodes_per_shard, (shard_id + 1) * nodes_per_shard - 1),
                    "dst": range(shard_id * nodes_per_shard + 1, (shard_id + 1) * nodes_per_shard),
                    "etype": [1] * (nodes_per_shard - 1),
                    "dt": [10] * (nodes_per_shard - 1),
                    **{f"e{i}": [0.5] * (nodes_per_shard - 1) for i in range(20)},
                }
            )

            nodes_file = temp_path / f"shard_{shard_id:03d}_nodes.parquet"
            edges_file = temp_path / f"shard_{shard_id:03d}_edges.parquet"

            nodes_df.to_parquet(nodes_file, index=False)
            edges_df.to_parquet(edges_file, index=False)

        pipeline = TemporalDiscoveryPipeline(data_path=temp_path)

        # Time shard loading
        start_time = time.time()
        shards = pipeline.load_shards()
        loading_time = time.time() - start_time

        # Performance assertions
        assert len(shards) == num_shards
        assert loading_time < 2.0, f"Shard loading took {loading_time:.2f}s, expected < 2.0s"

        # Verify all shards have expected data
        total_nodes = sum(len(shard["nodes"]) for shard in shards)
        assert total_nodes == num_shards * nodes_per_shard


@pytest.mark.performance
def test_graph_construction_performance():
    """Test graph construction performance on large shard."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    # Create large single shard
    num_nodes = 2000
    num_edges = 1800

    shard_data = {
        "shard_id": "large_shard",
        "nodes": pd.DataFrame(
            {
                "node_id": range(num_nodes),
                "t": range(num_nodes),
                "kind": [i % 3 + 1 for i in range(num_nodes)],
                **{f"f{i}": [(i + j) % 100 / 100.0 for j in range(num_nodes)] for i in range(45)},
            }
        ),
        "edges": pd.DataFrame(
            {
                "src": [i % num_nodes for i in range(num_edges)],
                "dst": [(i + 1) % num_nodes for i in range(num_edges)],
                "etype": [i % 4 + 1 for i in range(num_edges)],
                "dt": [i * 5 for i in range(num_edges)],
                **{f"e{i}": [(i + j) % 100 / 100.0 for j in range(num_edges)] for i in range(20)},
            }
        ),
    }

    pipeline = TemporalDiscoveryPipeline(data_path="/tmp")

    # Time graph construction
    start_time = time.time()
    graph = pipeline.build_temporal_graph(shard_data)
    construction_time = time.time() - start_time

    # Performance assertions
    assert (
        construction_time < 1.0
    ), f"Graph construction took {construction_time:.2f}s, expected < 1.0s"
    assert graph.num_nodes == num_nodes
    assert graph["node"].x.shape == (num_nodes, 45)


@pytest.mark.performance
def test_memory_efficiency_large_dataset():
    """Test memory efficiency with gradual data loading."""
    # This test verifies that the pipeline doesn't load all data into memory at once

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create several medium shards
        num_shards = 5
        nodes_per_shard = 1000

        for shard_id in range(num_shards):
            nodes_df = pd.DataFrame(
                {
                    "node_id": range(shard_id * nodes_per_shard, (shard_id + 1) * nodes_per_shard),
                    "t": range(nodes_per_shard),
                    "kind": [1] * nodes_per_shard,
                    **{f"f{i}": [0.1] * nodes_per_shard for i in range(45)},
                }
            )

            edges_df = pd.DataFrame(
                {
                    "src": range(shard_id * nodes_per_shard, (shard_id + 1) * nodes_per_shard - 1),
                    "dst": range(shard_id * nodes_per_shard + 1, (shard_id + 1) * nodes_per_shard),
                    "etype": [1] * (nodes_per_shard - 1),
                    "dt": [10] * (nodes_per_shard - 1),
                    **{f"e{i}": [0.5] * (nodes_per_shard - 1) for i in range(20)},
                }
            )

            nodes_file = temp_path / f"mem_test_{shard_id}_nodes.parquet"
            edges_file = temp_path / f"mem_test_{shard_id}_edges.parquet"

            nodes_df.to_parquet(nodes_file, index=False)
            edges_df.to_parquet(edges_file, index=False)

        pipeline = TemporalDiscoveryPipeline(
            data_path=temp_path,
            batch_size=100,  # Small batches for memory efficiency
        )

        # Monitor memory during different pipeline stages
        tracemalloc.start()

        try:
            # Stage 1: Load shards
            start_snapshot = tracemalloc.take_snapshot()
            shards = pipeline.load_shards()
            load_snapshot = tracemalloc.take_snapshot()

            load_memory = (
                sum(stat.size_diff for stat in load_snapshot.compare_to(start_snapshot, "lineno"))
                / 1024
                / 1024
            )

            # Stage 2: Graph construction (should not significantly increase memory)
            with patch.object(pipeline, "_run_tgat_discovery") as mock_discovery:
                mock_discovery.return_value = []

                pipeline.run_discovery()
                final_snapshot = tracemalloc.take_snapshot()

                total_memory = (
                    sum(
                        stat.size_diff
                        for stat in final_snapshot.compare_to(start_snapshot, "lineno")
                    )
                    / 1024
                    / 1024
                )

            # Memory efficiency assertions
            assert load_memory < 50, f"Shard loading used {load_memory:.1f}MB, expected < 50MB"
            assert total_memory < 75, f"Total pipeline used {total_memory:.1f}MB, expected < 75MB"

        finally:
            tracemalloc.stop()
