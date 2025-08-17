"""Authoritative column definitions and data types for IRONFORGE data storage."""

NFEATS_NODE = 45
NFEATS_EDGE = 20
NODE_COLS = ["node_id", "t", "kind"] + [f"f{i}" for i in range(NFEATS_NODE)]
EDGE_COLS = ["src", "dst", "etype", "dt"] + [f"e{i}" for i in range(NFEATS_EDGE)]
DTYPES = {
    "node_id": "uint32",
    "t": "int64",
    "kind": "uint8",
    "src": "uint32",
    "dst": "uint32",
    "etype": "uint8",
    "dt": "int32",
}
