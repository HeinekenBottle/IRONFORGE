"""Unit tests for TemporalDiscoveryPipeline (Wave 3)."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with test Parquet files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test node data (45D features)
        nodes_df = pd.DataFrame(
            {
                "node_id": [0, 1, 2, 3, 4],
                "t": [100, 200, 300, 400, 500],  # Timestamps
                "kind": [1, 2, 1, 2, 1],
                **{f"f{i}": [0.1 + i * 0.01] * 5 for i in range(45)},
            }
        )

        # Create test edge data (20D features)
        edges_df = pd.DataFrame(
            {
                "src": [0, 1, 2, 3],
                "dst": [1, 2, 3, 4],
                "etype": [1, 1, 2, 2],
                "dt": [10, 20, 30, 40],  # Temporal deltas
                **{f"e{i}": [0.5 + i * 0.02] * 4 for i in range(20)},
            }
        )

        # Write test files
        nodes_file = temp_path / "test_shard_01_nodes.parquet"
        edges_file = temp_path / "test_shard_01_edges.parquet"

        nodes_df.to_parquet(nodes_file, index=False)
        edges_df.to_parquet(edges_file, index=False)

        yield temp_path


@pytest.fixture
def pipeline(temp_data_dir):
    """Create pipeline instance with test data."""
    return TemporalDiscoveryPipeline(
        data_path=temp_data_dir,
        num_neighbors=[2, 1],
        batch_size=2,
        time_window=1,  # 1 hour window
        stitch_policy="session",
    )


def test_pipeline_initialization(temp_data_dir):
    """Test that TemporalDiscoveryPipeline initializes correctly."""
    pipeline = TemporalDiscoveryPipeline(
        data_path=temp_data_dir,
        num_neighbors=[10, 5],
        batch_size=64,
    )

    assert pipeline.data_path == temp_data_dir
    assert pipeline.num_neighbors == [10, 5]
    assert pipeline.batch_size == 64
    assert pipeline.time_window is None
    assert pipeline.stitch_policy == "session"


def test_load_shards(pipeline):
    """Test shard loading from Parquet files."""
    shards = pipeline.load_shards()

    assert len(shards) == 1
    shard = shards[0]

    assert shard["shard_id"] == "test_shard_01"
    assert len(shard["nodes"]) == 5
    assert len(shard["edges"]) == 4

    # Check node schema
    assert list(shard["nodes"]["node_id"]) == [0, 1, 2, 3, 4]
    assert "f0" in shard["nodes"].columns
    assert "f44" in shard["nodes"].columns

    # Check edge schema
    assert list(shard["edges"]["src"]) == [0, 1, 2, 3]
    assert "e0" in shard["edges"].columns
    assert "e19" in shard["edges"].columns


def test_load_shards_missing_directory():
    """Test shard loading with missing directory."""
    pipeline = TemporalDiscoveryPipeline(data_path="/nonexistent/path")

    with pytest.raises(FileNotFoundError):
        pipeline.load_shards()


def test_load_shards_no_files():
    """Test shard loading with no Parquet files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = TemporalDiscoveryPipeline(data_path=temp_dir)

        with pytest.raises(ValueError, match="No node Parquet files found"):
            pipeline.load_shards()


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="PyTorch not available") is None, reason="Requires PyTorch"
)
def test_build_temporal_graph(pipeline):
    """Test temporal graph construction."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    shards = pipeline.load_shards()
    graph = pipeline.build_temporal_graph(shards[0])

    # Check graph structure
    assert hasattr(graph, "shard_id")
    assert graph.shard_id == "test_shard_01"
    assert graph.num_nodes == 5

    # Check node features (45D)
    assert graph["node"].x.shape == (5, 45)
    assert graph["node"].node_id.shape == (5,)
    assert graph["node"].t.shape == (5,)

    # Check edge features (20D)
    assert graph["node", "temporal", "node"].edge_index.shape[0] == 2
    assert graph["node", "temporal", "node"].edge_attr.shape[1] == 20


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="PyTorch not available") is None, reason="Requires PyTorch"
)
def test_build_temporal_graph_empty_nodes():
    """Test graph construction with empty nodes."""
    pytest.importorskip("torch")

    empty_shard = {
        "shard_id": "empty",
        "nodes": pd.DataFrame(),
        "edges": pd.DataFrame(),
    }

    pipeline = TemporalDiscoveryPipeline(data_path="/tmp")

    with pytest.raises(ValueError, match="Empty nodes DataFrame"):
        pipeline.build_temporal_graph(empty_shard)


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="PyTorch not available") is None, reason="Requires PyTorch"
)
def test_create_neighbor_loader(pipeline):
    """Test neighbor loader creation."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    shards = pipeline.load_shards()
    graph = pipeline.build_temporal_graph(shards[0])

    with patch("ironforge.learning.discovery_pipeline.NeighborLoader") as mock_loader:
        mock_loader.return_value = Mock()

        loader = pipeline.create_neighbor_loader(graph)

        # Verify NeighborLoader was called with correct parameters
        mock_loader.assert_called_once()
        call_kwargs = mock_loader.call_args[1]

        assert call_kwargs["batch_size"] == 2
        assert call_kwargs["shuffle"] is True
        assert call_kwargs["directed"] is True


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="PyTorch not available") is None, reason="Requires PyTorch"
)
def test_stitch_anchors_single_graph(pipeline):
    """Test anchor stitching with single graph."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    shards = pipeline.load_shards()
    graph = pipeline.build_temporal_graph(shards[0])

    # Single graph should be returned unchanged
    stitched = pipeline.stitch_anchors([graph])
    assert stitched is graph


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="PyTorch not available") is None, reason="Requires PyTorch"
)
def test_stitch_anchors_multiple_graphs(pipeline):
    """Test anchor stitching with multiple graphs."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    shards = pipeline.load_shards()
    graph1 = pipeline.build_temporal_graph(shards[0])
    graph2 = pipeline.build_temporal_graph(shards[0])  # Duplicate for testing

    stitched = pipeline.stitch_anchors([graph1, graph2])

    # Check that graphs were merged
    assert stitched.num_nodes == graph1.num_nodes + graph2.num_nodes
    assert hasattr(stitched, "shard_id")


def test_stitch_anchors_empty_list(pipeline):
    """Test anchor stitching with empty list."""
    with pytest.raises(ValueError, match="No graphs provided"):
        pipeline.stitch_anchors([])


def test_stitch_anchors_unknown_policy():
    """Test anchor stitching with unknown policy."""
    pipeline = TemporalDiscoveryPipeline(data_path="/tmp", stitch_policy="unknown")

    with patch.object(pipeline, "build_temporal_graph") as mock_build:
        mock_graph = Mock()
        mock_build.return_value = mock_graph

        with pytest.raises(ValueError, match="Unknown stitch policy"):
            pipeline.stitch_anchors([mock_graph])


@patch("ironforge.learning.discovery_pipeline.get_ironforge_container")
def test_run_discovery(mock_container, pipeline):
    """Test full discovery pipeline execution."""
    # Mock the discovery engine
    mock_discovery = Mock()
    mock_discovery.discover_patterns.return_value = [
        {"pattern_type": "test_pattern", "confidence": 0.9}
    ]

    mock_container.return_value.get_tgat_discovery.return_value = mock_discovery

    # Mock graph construction and neighbor loader
    with (
        patch.object(pipeline, "load_shards") as mock_load,
        patch.object(pipeline, "build_temporal_graph") as mock_build,
        patch.object(pipeline, "stitch_anchors") as mock_stitch,
        patch.object(pipeline, "create_neighbor_loader") as mock_loader,
    ):

        # Setup mocks
        mock_load.return_value = [{"shard_id": "test"}]
        mock_build.return_value = Mock()
        mock_stitch.return_value = Mock()
        mock_loader.return_value = [Mock()]  # Single batch

        discoveries = pipeline.run_discovery()

        assert len(discoveries) == 1
        assert discoveries[0]["pattern_type"] == "test_pattern"
        assert "batch_id" in discoveries[0]
        assert "pipeline_metadata" in discoveries[0]


def test_run_discovery_import_error(pipeline):
    """Test discovery with missing TGAT components."""
    with patch(
        "ironforge.learning.discovery_pipeline.get_ironforge_container",
        side_effect=ImportError("Missing components"),
    ):

        with (
            patch.object(pipeline, "load_shards") as mock_load,
            patch.object(pipeline, "build_temporal_graph") as mock_build,
            patch.object(pipeline, "stitch_anchors") as mock_stitch,
            patch.object(pipeline, "create_neighbor_loader") as mock_loader,
        ):

            mock_load.return_value = [{"shard_id": "test"}]
            mock_build.return_value = Mock()
            mock_stitch.return_value = Mock()
            mock_loader.return_value = [Mock()]

            with pytest.raises(ImportError, match="Cannot import TGAT discovery components"):
                pipeline.run_discovery()


def test_run_pipeline_output(pipeline):
    """Test high-level pipeline execution and output."""
    with patch.object(pipeline, "run_discovery") as mock_discovery:
        mock_discovery.return_value = [{"pattern_type": "test", "confidence": 0.8}]

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.output_dir = temp_dir

            pipeline.run()

            # Check output files
            output_files = list(Path(temp_dir).glob("*.json"))
            assert len(output_files) >= 2  # discoveries + summary

            discoveries_files = [f for f in output_files if "temporal_discoveries" in f.name]
            summary_files = [f for f in output_files if "discovery_summary" in f.name]

            assert len(discoveries_files) == 1
            assert len(summary_files) == 1
