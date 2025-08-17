"""Build igraph Graph objects from Parquet DataFrames."""

import igraph as ig
import pandas as pd


def from_parquet(nodes: pd.DataFrame, edges: pd.DataFrame) -> ig.Graph:
    """Create igraph Graph from nodes and edges DataFrames."""
    g = ig.Graph(n=len(nodes), directed=True)
    g.vs["t"] = nodes["t"].to_numpy()
    g.vs["kind"] = nodes["kind"].to_numpy()
    g.add_edges(list(zip(edges["src"].to_numpy(), edges["dst"].to_numpy(), strict=False)))
    g.es["etype"] = edges["etype"].to_numpy()
    return g
