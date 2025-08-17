from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_nodes_edges(shard_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read node and edge Parquet files for a shard directory."""
    base = Path(shard_dir)
    nodes = pd.read_parquet(base / "nodes.parquet")
    edges = pd.read_parquet(base / "edges.parquet")
    return nodes, edges
