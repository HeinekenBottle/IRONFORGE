"""Convert igraph objects to PyTorch Geometric format."""

import torch
from torch_geometric.data import Data


def igraph_to_pyg(g) -> Data:
    """Convert igraph Graph to PyTorch Geometric Data object."""
    ei = torch.tensor(g.get_edgelist(), dtype=torch.long).t().contiguous()
    return Data(edge_index=ei)
