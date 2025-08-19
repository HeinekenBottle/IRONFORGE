"""Integration tests for Wave 3 shard-aware temporal discovery."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline
from ironforge.sdk.cli import main


@pytest.fixture
def wave3_test_data():
    """Create comprehensive test data for Wave 3 integration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple sessions with realistic data
        sessions = ["NY_AM_20250801", "LONDON_20250801", "ASIA_20250802"]

        for session_idx, session_id in enumerate(sessions):
            # Create 2 shards per session
            for shard_idx in range(2):
                shard_id = f"{session_id}_shard_{shard_idx:02d}"

                # Node data with session-specific timestamps
                base_time = session_idx * 86400 + shard_idx * 1000  # Different base times
                num_nodes = 50 + shard_idx * 25  # 50-75 nodes per shard

                nodes_df = pd.DataFrame(
                    {
                        "node_id": range(
                            session_idx * 200 + shard_idx * 100,
                            session_idx * 200 + shard_idx * 100 + num_nodes,
                        ),
                        "t": range(base_time, base_time + num_nodes * 60, 60),  # 1-minute intervals
                        "kind": [(i + session_idx) % 4 + 1 for i in range(num_nodes)],
                        **{
                            f"f{i}": [
                                (session_idx + shard_idx + i * 0.1 + j * 0.01) % 1.0
                                for j in range(num_nodes)
                            ]
                            for i in range(45)
                        },
                    }
                )

                # Edge data with temporal relationships
                num_edges = max(num_nodes - 5, 10)
                src_nodes = [nodes_df.iloc[i]["node_id"] for i in range(num_edges)]
                dst_nodes = [
                    nodes_df.iloc[min(i + 1, num_nodes - 1)]["node_id"] for i in range(num_edges)
                ]

                edges_df = pd.DataFrame(
                    {
                        "src": src_nodes,
                        "dst": dst_nodes,
                        "etype": [(i + session_idx) % 3 + 1 for i in range(num_edges)],
                        "dt": [
                            60 + i * 30 for i in range(num_edges)
                        ],  # 60-second base + increments
                        **{
                            f"e{i}": [
                                (session_idx + shard_idx + i * 0.05 + j * 0.02) % 1.0
                                for j in range(num_edges)
                            ]
                            for i in range(20)
                        },
                    }
                )

                # Write files
                nodes_file = temp_path / f"{shard_id}_nodes.parquet"
                edges_file = temp_path / f"{shard_id}_edges.parquet"

                nodes_df.to_parquet(nodes_file, index=False)
                edges_df.to_parquet(edges_file, index=False)

        yield temp_path, sessions


def test_wave3_full_pipeline_integration(wave3_test_data):
    """Test complete Wave 3 pipeline with realistic multi-session data."""
    temp_path, sessions = wave3_test_data

    pipeline = TemporalDiscoveryPipeline(
        data_path=temp_path,
        num_neighbors=[5, 3, 2],
        batch_size=32,
        time_window=2,  # 2-hour window
        stitch_policy="session",
    )

    # Mock the TGAT discovery to focus on pipeline integration
    mock_patterns = [
        {
            "pattern_type": "temporal_cascade",
            "confidence": 0.92,
            "session_context": "NY_AM_20250801",
            "temporal_span": 1800,  # 30 minutes
            "nodes_involved": 12,
        },
        {
            "pattern_type": "cross_session_anchor",
            "confidence": 0.85,
            "session_context": "LONDON_20250801",
            "temporal_span": 3600,  # 1 hour
            "nodes_involved": 8,
        },
    ]

    with patch.object(pipeline, "_run_tgat_discovery") as mock_discovery:
        mock_discovery.return_value = mock_patterns

        # Execute full pipeline
        discoveries = pipeline.run_discovery()

        # Verify pipeline execution
        assert len(discoveries) == len(mock_patterns)

        # Check that all discoveries have Wave 3 metadata
        for discovery in discoveries:
            assert "pipeline_metadata" in discovery
            metadata = discovery["pipeline_metadata"]
            assert metadata["fanouts"] == [5, 3, 2]
            assert metadata["batch_size"] == 32
            assert metadata["time_window"] == 2
            assert metadata["stitch_policy"] == "session"
            assert "batch_id" in discovery


def test_wave3_cli_end_to_end(wave3_test_data):
    """Test Wave 3 CLI from command-line arguments to output files."""
    temp_path, sessions = wave3_test_data

    with tempfile.TemporaryDirectory() as output_dir:
        output_path = Path(output_dir)

        # Mock the discovery pipeline execution
        with patch("ironforge.sdk.cli.TemporalDiscoveryPipeline") as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline

            # Simulate successful pipeline execution
            mock_pipeline.run.return_value = None

            # Execute CLI
            result = main(
                [
                    "discover-temporal",
                    "--data-path",
                    str(temp_path),
                    "--output-dir",
                    str(output_path),
                    "--fanouts",
                    "8",
                    "4",
                    "2",
                    "--batch-size",
                    "64",
                    "--time-window",
                    "3",
                    "--stitch-policy",
                    "global",
                ]
            )

            # Verify CLI execution
            assert result == 0

            # Verify pipeline was configured correctly
            mock_pipeline_class.assert_called_once_with(
                data_path=temp_path,
                num_neighbors=[8, 4, 2],
                batch_size=64,
                time_window=3,
                stitch_policy="global",
            )

            # Verify output directory was set
            assert mock_pipeline.output_dir == output_path
            mock_pipeline.run.assert_called_once()


def test_wave3_shard_loading_integration(wave3_test_data):
    """Test shard loading with realistic multi-session structure."""
    temp_path, sessions = wave3_test_data

    pipeline = TemporalDiscoveryPipeline(data_path=temp_path)

    # Load shards
    shards = pipeline.load_shards()

    # Verify we loaded the expected number of shards (3 sessions × 2 shards each)
    assert len(shards) == 6

    # Verify shard IDs follow expected pattern
    shard_ids = [shard["shard_id"] for shard in shards]
    expected_sessions = ["NY_AM_20250801", "LONDON_20250801", "ASIA_20250802"]

    for session in expected_sessions:
        session_shards = [sid for sid in shard_ids if session in sid]
        assert len(session_shards) == 2  # 2 shards per session

    # Verify data integrity
    total_nodes = sum(len(shard["nodes"]) for shard in shards)
    total_edges = sum(len(shard["edges"]) for shard in shards)

    assert total_nodes > 300  # At least 50+ nodes per shard
    assert total_edges > 200  # Reasonable edge count

    # Verify schema compliance
    for shard in shards:
        nodes_df = shard["nodes"]
        edges_df = shard["edges"]

        # Check node schema (45D features)
        assert "node_id" in nodes_df.columns
        assert "t" in nodes_df.columns
        assert "kind" in nodes_df.columns
        for i in range(45):
            assert f"f{i}" in nodes_df.columns

        # Check edge schema (20D features)
        if not edges_df.empty:
            assert "src" in edges_df.columns
            assert "dst" in edges_df.columns
            assert "etype" in edges_df.columns
            assert "dt" in edges_df.columns
            for i in range(20):
                assert f"e{i}" in edges_df.columns


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="PyTorch not available") is None,
    reason="Requires PyTorch for graph construction",
)
def test_wave3_graph_construction_integration(wave3_test_data):
    """Test graph construction and stitching with multi-session data."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    temp_path, sessions = wave3_test_data

    pipeline = TemporalDiscoveryPipeline(data_path=temp_path, stitch_policy="session")

    # Load and construct graphs
    shards = pipeline.load_shards()
    graphs = []

    for shard_data in shards:
        graph = pipeline.build_temporal_graph(shard_data)
        graphs.append(graph)

    # Verify individual graphs
    assert len(graphs) == 6

    for graph in graphs:
        # Check graph structure
        assert hasattr(graph, "shard_id")
        assert graph.num_nodes > 0
        assert graph["node"].x.shape[1] == 45  # 45D node features
        assert graph["node", "temporal", "node"].edge_attr.shape[1] == 20  # 20D edge features

    # Test anchor stitching
    stitched_graph = pipeline.stitch_anchors(graphs)

    # Verify stitched graph
    total_original_nodes = sum(g.num_nodes for g in graphs)
    assert stitched_graph.num_nodes <= total_original_nodes  # May have deduplication
    assert stitched_graph["node"].x.shape[1] == 45
    assert stitched_graph["node", "temporal", "node"].edge_attr.shape[1] == 20


def test_wave3_output_format_integration(wave3_test_data):
    """Test output file generation and format compliance."""
    temp_path, sessions = wave3_test_data

    with tempfile.TemporaryDirectory() as output_dir:
        output_path = Path(output_dir)

        pipeline = TemporalDiscoveryPipeline(data_path=temp_path)
        pipeline.output_dir = output_path

        # Mock discovery results
        mock_discoveries = [
            {
                "pattern_type": "test_pattern",
                "confidence": 0.9,
                "temporal_features": {"duration": 1800, "peak_intensity": 0.85},
                "spatial_features": {"node_count": 15, "edge_density": 0.6},
            }
        ]

        with patch.object(pipeline, "run_discovery") as mock_discovery:
            mock_discovery.return_value = mock_discoveries

            # Execute pipeline
            pipeline.run()

            # Verify output files
            output_files = list(output_path.glob("*.json"))
            assert len(output_files) >= 2  # discoveries + summary

            # Find and verify discoveries file
            discoveries_files = [f for f in output_files if "temporal_discoveries" in f.name]
            assert len(discoveries_files) == 1

            with open(discoveries_files[0]) as f:
                saved_discoveries = json.load(f)

            assert len(saved_discoveries) == 1
            assert saved_discoveries[0]["pattern_type"] == "test_pattern"

            # Find and verify summary file
            summary_files = [f for f in output_files if "discovery_summary" in f.name]
            assert len(summary_files) == 1

            with open(summary_files[0]) as f:
                summary = json.load(f)

            assert summary["total_patterns"] == 1
            assert "pipeline_config" in summary
            assert "pattern_types" in summary
            assert summary["pattern_types"]["test_pattern"] == 1


def test_wave3_error_handling_integration(wave3_test_data):
    """Test error handling and graceful degradation."""
    temp_path, sessions = wave3_test_data

    pipeline = TemporalDiscoveryPipeline(data_path=temp_path)

    # Test with missing TGAT discovery components
    with patch(
        "ironforge.learning.discovery_pipeline.get_ironforge_container",
        side_effect=ImportError("TGAT components unavailable"),
    ):

        with pytest.raises(ImportError, match="Cannot import TGAT discovery components"):
            pipeline.run_discovery()

    # Test with invalid data path
    invalid_pipeline = TemporalDiscoveryPipeline(data_path="/nonexistent/path")

    with pytest.raises(FileNotFoundError):
        invalid_pipeline.run_discovery()

    # Test graceful handling of corrupted shards
    with patch.object(
        pipeline, "build_temporal_graph", side_effect=ValueError("Corrupted shard data")
    ):

        # Should not completely fail, but log warnings
        with patch.object(pipeline, "_run_tgat_discovery") as mock_discovery:
            mock_discovery.return_value = []

            # This should raise because no valid graphs are constructed
            with pytest.raises(ValueError, match="No valid graphs constructed"):
                pipeline.run_discovery()


@pytest.mark.performance
def test_wave3_performance_integration(wave3_test_data):
    """Integration test for Wave 3 performance requirements."""
    temp_path, sessions = wave3_test_data

    pipeline = TemporalDiscoveryPipeline(
        data_path=temp_path, num_neighbors=[5, 3], batch_size=32, time_window=1
    )

    # Mock discovery to focus on pipeline performance
    with patch.object(pipeline, "_run_tgat_discovery") as mock_discovery:
        mock_discovery.return_value = [{"pattern_type": "test", "confidence": 0.8}]

        import time

        start_time = time.time()
        discoveries = pipeline.run_discovery()
        elapsed_time = time.time() - start_time

        # Performance assertions
        assert elapsed_time < 5.0, f"Integration pipeline took {elapsed_time:.2f}s, expected < 5.0s"
        assert len(discoveries) > 0
