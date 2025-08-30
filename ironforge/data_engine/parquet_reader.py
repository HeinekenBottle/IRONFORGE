from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _available_columns(parquet_path: Path) -> list[str]:
    """Return available column names from a Parquet file without reading data."""
    pf = pq.ParquetFile(parquet_path)
    return pf.schema.names


def read_nodes_edges(shard_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read node and edge Parquet files for a shard directory with column projection."""
    base = Path(shard_dir)
    nodes_path = base / "nodes.parquet"
    edges_path = base / "edges.parquet"

    # Determine available columns and project to required subsets to reduce I/O
    node_cols_avail = _available_columns(nodes_path)
    edge_cols_avail = _available_columns(edges_path)

    desired_node_cols = ["node_id"] + [f"f{i}" for i in range(0, 51)]  # superset (45D/51D)
    desired_edge_cols = ["src", "dst", "dt"] + [f"e{i}" for i in range(0, 20)]

    node_cols = [c for c in desired_node_cols if c in node_cols_avail]
    edge_cols = [c for c in desired_edge_cols if c in edge_cols_avail]

    nodes = pd.read_parquet(nodes_path, columns=node_cols or None)
    edges = pd.read_parquet(edges_path, columns=edge_cols or None)

    return nodes, edges
