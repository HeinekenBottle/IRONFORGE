from __future__ import annotations

import networkx as nx
import pandas as pd


def from_parquet(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed NetworkX graph from node and edge dataframes."""
    g = nx.DiGraph()
    for _, row in nodes.iterrows():
        g.add_node(int(row["node_id"]), **row.to_dict())
    for _, row in edges.iterrows():
        g.add_edge(int(row["src"]), int(row["dst"]), **row.to_dict())
    return g
