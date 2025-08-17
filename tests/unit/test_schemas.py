"""Test schema definitions and completeness."""

from ironforge.data_engine.schemas import DTYPES, EDGE_COLS, NODE_COLS


def test_node_edge_schemas_complete() -> None:
    """Test that node and edge schemas have required columns and types."""
    assert "node_id" in NODE_COLS
    assert "src" in EDGE_COLS
    assert {"uint32", "int64", "uint8", "int32"}.issubset(set(DTYPES.values()))


def test_node_schema_structure() -> None:
    """Test node schema has expected structure."""
    assert NODE_COLS[0] == "node_id"
    assert NODE_COLS[1] == "t"
    assert NODE_COLS[2] == "kind"
    # Should have 45 feature columns (f0-f44)
    feature_cols = [col for col in NODE_COLS if col.startswith("f")]
    assert len(feature_cols) == 45


def test_edge_schema_structure() -> None:
    """Test edge schema has expected structure."""
    assert EDGE_COLS[0] == "src"
    assert EDGE_COLS[1] == "dst"
    assert EDGE_COLS[2] == "etype"
    assert EDGE_COLS[3] == "dt"
    # Should have 20 feature columns (e0-e19)
    feature_cols = [col for col in EDGE_COLS if col.startswith("e") and col[1:].isdigit()]
    assert len(feature_cols) == 20


def test_dtypes_coverage() -> None:
    """Test that DTYPES covers core columns."""
    required_node_cols = ["node_id", "t", "kind"]
    required_edge_cols = ["src", "dst", "etype", "dt"]

    for col in required_node_cols + required_edge_cols:
        assert col in DTYPES
