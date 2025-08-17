"""Validated Parquet readers for nodes and edges."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Only a subset of columns are required for basic graph construction.  We
# validate these minimal sets rather than the full schemas to keep the reader
# flexible for partial datasets.
REQUIRED_NODE_COLS = ["node_id"]
REQUIRED_EDGE_COLS = ["src", "dst"]


def _validate(df: pd.DataFrame, required: list[str], kind: str) -> None:
    """Validate that ``df`` contains the required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{kind} file missing columns: {missing}")


def read_parquet_graph(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``nodes.parquet`` and ``edges.parquet`` from ``path``.

    Parameters
    ----------
    path:
        Directory containing ``nodes.parquet`` and ``edges.parquet``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        DataFrames for nodes and edges respectively.

    Raises
    ------
    FileNotFoundError
        If either ``nodes.parquet`` or ``edges.parquet`` is missing.
    ValueError
        If the required columns are not present in the respective files.
    """

    base = Path(path)
    nodes_path = base / "nodes.parquet"
    edges_path = base / "edges.parquet"

    if not nodes_path.is_file():
        raise FileNotFoundError(f"Missing nodes parquet file: {nodes_path}")
    if not edges_path.is_file():
        raise FileNotFoundError(f"Missing edges parquet file: {edges_path}")

    nodes_df = pd.read_parquet(nodes_path)
    edges_df = pd.read_parquet(edges_path)

    _validate(nodes_df, REQUIRED_NODE_COLS, "nodes")
    _validate(edges_df, REQUIRED_EDGE_COLS, "edges")

    return nodes_df, edges_df
