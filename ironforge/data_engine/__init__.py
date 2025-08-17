"""Data engine utilities for IRONFORGE.

This module re-exports the authoritative data engine implementation
located in the legacy `IRONFORGE` package so that consumers can import
`ironforge.data_engine` regardless of case. This ensures compatibility
with existing code and tests expecting the module under the
`ironforge` namespace.
"""

from .schemas import DTYPES, EDGE_COLS, NFEATS_EDGE, NFEATS_NODE, NODE_COLS

# ``pyarrow`` is an optional dependency used by the parquet writer. Import it
# lazily so modules that only need the schema definitions do not fail when
# ``pyarrow`` is absent. Consumers requiring parquet functionality can import
# ``ironforge.data_engine.parquet_writer`` directly; it will raise if
# ``pyarrow`` is missing.
try:  # pragma: no cover - exercised indirectly
    from .parquet_writer import write_edges, write_nodes
except Exception:  # ModuleNotFoundError if pyarrow isn't installed
    # Expose stubs so attribute access fails with a helpful message
    def write_nodes(*_, **__):  # type: ignore[override]
        raise ImportError("pyarrow is required for parquet writing")

    def write_edges(*_, **__):  # type: ignore[override]
        raise ImportError("pyarrow is required for parquet writing")


__all__ = [
    "NFEATS_NODE",
    "NFEATS_EDGE",
    "NODE_COLS",
    "EDGE_COLS",
    "DTYPES",
    "write_nodes",
    "write_edges",
]
