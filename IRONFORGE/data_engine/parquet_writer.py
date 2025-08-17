"""Validated Parquet writers for nodes and edges."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import DTYPES, EDGE_COLS, NODE_COLS


def _validate(df: pd.DataFrame, cols: list[str]) -> None:
    """Validate that DataFrame has required columns."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _cast(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Cast DataFrame columns to specified types."""
    for c, t in dtypes.items():
        if c in df.columns:
            df[c] = df[c].astype(t, copy=False)
    return df


def write_nodes(df: pd.DataFrame, path: str) -> None:
    """Write nodes DataFrame to Parquet with validation."""
    _validate(df, NODE_COLS)
    df = _cast(df, DTYPES)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")


def write_edges(df: pd.DataFrame, path: str) -> None:
    """Write edges DataFrame to Parquet with validation."""
    _validate(df, EDGE_COLS)
    df = _cast(df, DTYPES)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")
