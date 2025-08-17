"""Performance test for igraph builder."""

import time
import sys
import pathlib
import numpy as np
import pandas as pd

# The graph builder lives in the "IRONFORGE" namespace which isn't installed as
# a package for tests.  Add it to ``sys.path`` so the module can be imported
# directly for benchmarking.
ROOT = pathlib.Path(__file__).resolve().parents[2] / "IRONFORGE"
sys.path.append(str(ROOT))

from graph_builder.igraph_builder import from_parquet

# Allow up to 0.5s to build graph from moderately sized DataFrames
BUILD_BUDGET_S = 0.5


def test_graph_build_performance() -> None:
    """Ensure graph construction remains within the performance budget."""
    nodes = pd.DataFrame({
        "t": np.random.randint(0, 1000, 5000),
        "kind": np.random.randint(0, 5, 5000),
    })

    edges = pd.DataFrame({
        "src": np.random.randint(0, 5000, 10000),
        "dst": np.random.randint(0, 5000, 10000),
        "etype": np.random.randint(0, 10, 10000),
    })

    start = time.time()
    g = from_parquet(nodes, edges)
    elapsed = time.time() - start

    assert elapsed <= BUILD_BUDGET_S, (
        f"Graph build took {elapsed:.3f}s, budget {BUILD_BUDGET_S}s"
    )
    assert g.vcount() == len(nodes)
    assert g.ecount() == len(edges)
