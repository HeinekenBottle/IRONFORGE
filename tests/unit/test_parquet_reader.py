"""Tests for the Parquet reader utilities."""

import pandas as pd
import pytest
from pathlib import Path

from ironforge.data_engine.parquet_reader import read_parquet_graph


def test_missing_nodes_file(tmp_path: Path) -> None:
    """Reading should fail if ``nodes.parquet`` is absent."""
    # Create only edges file
    (tmp_path / "edges.parquet").touch()

    with pytest.raises(FileNotFoundError) as exc:
        read_parquet_graph(tmp_path)
    assert "nodes.parquet" in str(exc.value)


def test_missing_edges_file(tmp_path: Path) -> None:
    """Reading should fail if ``edges.parquet`` is absent."""
    (tmp_path / "nodes.parquet").touch()

    with pytest.raises(FileNotFoundError) as exc:
        read_parquet_graph(tmp_path)
    assert "edges.parquet" in str(exc.value)


def test_missing_node_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reader validates that nodes file has required column."""
    (tmp_path / "nodes.parquet").touch()
    (tmp_path / "edges.parquet").touch()

    def fake_read(path: str | Path, *_, **__):
        path = Path(path)
        if path.name == "nodes.parquet":
            return pd.DataFrame({"t": [0]})  # missing node_id
        return pd.DataFrame({"src": [0], "dst": [1]})

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    with pytest.raises(ValueError) as exc:
        read_parquet_graph(tmp_path)
    assert "node_id" in str(exc.value)


def test_missing_edge_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reader validates that edges file has required columns."""
    (tmp_path / "nodes.parquet").touch()
    (tmp_path / "edges.parquet").touch()

    def fake_read(path: str | Path, *_, **__):
        path = Path(path)
        if path.name == "nodes.parquet":
            return pd.DataFrame({"node_id": [0]})
        return pd.DataFrame({"src": [0]})  # missing dst

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    with pytest.raises(ValueError) as exc:
        read_parquet_graph(tmp_path)
    assert "dst" in str(exc.value)
