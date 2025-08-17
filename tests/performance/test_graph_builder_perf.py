"""Performance regression tests for graph building."""

import time

import pytest


@pytest.mark.performance
def test_enhanced_graph_builder_time_budget():
    """Ensure enhanced graph building completes within 1s budget."""
    pytest.importorskip("networkx")
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

    builder = EnhancedGraphBuilder()
    session_data = {
        "session_name": "test",
        "events": [{"event_type": "sweep", "price": float(i), "timestamp": i} for i in range(100)],
    }

    start = time.time()
    graph = builder.build_session_graph(session_data)
    elapsed = time.time() - start

    assert elapsed < 1.0, f"Graph build took {elapsed:.2f}s, expected < 1.0s"
    assert graph.number_of_nodes() == 100
