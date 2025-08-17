"""Test roundtrip Parquet → igraph conversions."""

import os
import tempfile

import pandas as pd

from ironforge.data_engine.parquet_writer import write_edges, write_nodes
from ironforge.graph_builder.igraph_builder import from_parquet


def test_roundtrip_counts() -> None:
    """Test that roundtrip preserves node and edge counts."""
    # Create minimal test data with all required columns
    nodes = pd.DataFrame(
        {"node_id": [0, 1], "t": [0, 1], "kind": [1, 1], **{f"f{i}": [0.1, 0.2] for i in range(45)}}
    )
    edges = pd.DataFrame(
        {"src": [0], "dst": [1], "etype": [1], "dt": [1], **{f"e{i}": [0.5] for i in range(20)}}
    )

    with tempfile.TemporaryDirectory() as d:
        nodes_path = os.path.join(d, "nodes.parquet")
        edges_path = os.path.join(d, "edges.parquet")

        write_nodes(nodes, nodes_path)
        write_edges(edges, edges_path)

        g = from_parquet(nodes, edges)
        assert g.vcount() == 2
        assert g.ecount() == 1


def test_roundtrip_attributes() -> None:
    """Test that roundtrip preserves node and edge attributes."""
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2],
            "t": [100, 200, 300],
            "kind": [1, 2, 1],
            **{f"f{i}": [0.1, 0.2, 0.3] for i in range(45)},
        }
    )
    edges = pd.DataFrame(
        {
            "src": [0, 1],
            "dst": [1, 2],
            "etype": [1, 2],
            "dt": [10, 20],
            **{f"e{i}": [0.5, 0.6] for i in range(20)},
        }
    )

    with tempfile.TemporaryDirectory() as d:
        nodes_path = os.path.join(d, "nodes.parquet")
        edges_path = os.path.join(d, "edges.parquet")

        write_nodes(nodes, nodes_path)
        write_edges(edges, edges_path)

        g = from_parquet(nodes, edges)

        # Check node attributes
        assert list(g.vs["t"]) == [100, 200, 300]
        assert list(g.vs["kind"]) == [1, 2, 1]

        # Check edge attributes
        assert list(g.es["etype"]) == [1, 2]


def test_empty_graph() -> None:
    """Test handling of empty graphs."""
    nodes = pd.DataFrame(
        {"node_id": [0], "t": [0], "kind": [1], **{f"f{i}": [0.1] for i in range(45)}}
    )
    edges = pd.DataFrame(
        {"src": [], "dst": [], "etype": [], "dt": [], **{f"e{i}": [] for i in range(20)}}
    ).astype(
        {
            "src": "uint32",
            "dst": "uint32",
            "etype": "uint8",
            "dt": "int32",
            **{f"e{i}": "float64" for i in range(20)},
        }
    )

    with tempfile.TemporaryDirectory() as d:
        nodes_path = os.path.join(d, "nodes.parquet")
        edges_path = os.path.join(d, "edges.parquet")

        write_nodes(nodes, nodes_path)
        write_edges(edges, edges_path)

        g = from_parquet(nodes, edges)
        assert g.vcount() == 1
        assert g.ecount() == 0
