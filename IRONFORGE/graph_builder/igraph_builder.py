"""Build igraph Graph objects from Parquet DataFrames."""

import igraph as ig
import pandas as pd


def from_parquet(nodes: pd.DataFrame, edges: pd.DataFrame) -> ig.Graph:
    """Create :class:`igraph.Graph` from nodes and edges DataFrames.

    This implementation avoids ``DataFrame.iterrows`` which is known to be
    slow for large tables.  Edges are constructed using ``itertuples`` which
    yields lightweight namedtuples and allows ``igraph`` to consume the pairs
    without materialising an intermediate Python ``list``.
    """

    # Pre-allocate all vertices then assign attributes in bulk
    g = ig.Graph(n=len(nodes), directed=True)
    g.vs["t"] = nodes["t"].to_numpy()
    g.vs["kind"] = nodes["kind"].to_numpy()

    # Add edges using a generator of ``(src, dst)`` tuples to avoid Python row
    # objects and per-row overhead from ``iterrows``
    edge_pairs = ((row.src, row.dst) for row in edges[["src", "dst"]].itertuples(index=False))
    g.add_edges(edge_pairs)
    g.es["etype"] = edges["etype"].to_numpy()
    return g
