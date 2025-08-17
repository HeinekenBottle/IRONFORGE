"""Performance budget tests for Parquet operations."""

import os
import resource
import tempfile
import time

import numpy as np
import pandas as pd

from ironforge.data_engine.parquet_writer import write_edges, write_nodes

# Performance budgets
WRITE_BUDGET_S = 2.0  # Max 2s for write operations
READ_BUDGET_S = 1.0  # Max 1s for read operations
MEMORY_BUDGET_MB = 200  # Max 200MB memory increase


def mem_mb() -> float:
    """Get current memory usage in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def test_parquet_write_performance() -> None:
    """Test that Parquet writes meet performance budgets."""
    # Create moderately sized test data (100 nodes, 200 edges)
    nodes = pd.DataFrame(
        {
            "node_id": range(100),
            "t": np.random.randint(0, 10000, 100),
            "kind": np.random.randint(0, 5, 100),
            **{f"f{i}": np.random.random(100) for i in range(45)},
        }
    )

    edges = pd.DataFrame(
        {
            "src": np.random.randint(0, 100, 200),
            "dst": np.random.randint(0, 100, 200),
            "etype": np.random.randint(0, 10, 200),
            "dt": np.random.randint(-100, 100, 200),
            **{f"e{i}": np.random.random(200) for i in range(20)},
        }
    )

    with tempfile.TemporaryDirectory() as d:
        nodes_path = os.path.join(d, "nodes.parquet")
        edges_path = os.path.join(d, "edges.parquet")

        # Test write performance
        start = time.time()

        write_nodes(nodes, nodes_path)
        write_edges(edges, edges_path)

        elapsed = time.time() - start

        assert elapsed <= WRITE_BUDGET_S, f"Write took {elapsed:.3f}s, budget {WRITE_BUDGET_S}s"

        # Verify files were created
        assert os.path.exists(nodes_path)
        assert os.path.exists(edges_path)


def test_parquet_read_performance() -> None:
    """Test that Parquet reads meet performance budgets."""
    # Create and write test data first
    nodes = pd.DataFrame(
        {
            "node_id": range(50),
            "t": np.random.randint(0, 10000, 50),
            "kind": np.random.randint(0, 5, 50),
            **{f"f{i}": np.random.random(50) for i in range(45)},
        }
    )

    edges = pd.DataFrame(
        {
            "src": np.random.randint(0, 50, 100),
            "dst": np.random.randint(0, 50, 100),
            "etype": np.random.randint(0, 10, 100),
            "dt": np.random.randint(-100, 100, 100),
            **{f"e{i}": np.random.random(100) for i in range(20)},
        }
    )

    with tempfile.TemporaryDirectory() as d:
        nodes_path = os.path.join(d, "nodes.parquet")
        edges_path = os.path.join(d, "edges.parquet")

        write_nodes(nodes, nodes_path)
        write_edges(edges, edges_path)

        # Test read performance
        start = time.time()

        nodes_read = pd.read_parquet(nodes_path)
        edges_read = pd.read_parquet(edges_path)

        elapsed = time.time() - start

        assert elapsed <= READ_BUDGET_S, f"Read took {elapsed:.3f}s, budget {READ_BUDGET_S}s"

        # Verify data integrity
        assert len(nodes_read) == 50
        assert len(edges_read) == 100
