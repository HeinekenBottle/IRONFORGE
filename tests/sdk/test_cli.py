"""Tests for IRONFORGE SDK CLI (Wave 3)."""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ironforge.sdk.cli import _parse_args, main


def test_parse_args_discover_temporal():
    """Test argument parsing for discover-temporal command."""
    args = _parse_args(
        [
            "discover-temporal",
            "--data-path",
            "/path/to/data",
            "--output-dir",
            "/path/to/output",
            "--fanouts",
            "5",
            "3",
            "2",
            "--batch-size",
            "64",
            "--time-window",
            "2",
            "--stitch-policy",
            "global",
        ]
    )

    assert args.command == "discover-temporal"
    assert args.data_path == Path("/path/to/data")
    assert args.output_dir == Path("/path/to/output")
    assert args.fanouts == [5, 3, 2]
    assert args.batch_size == 64
    assert args.time_window == 2
    assert args.stitch_policy == "global"


def test_parse_args_discover_temporal_defaults():
    """Test argument parsing with default values."""
    args = _parse_args(["discover-temporal", "--data-path", "/path/to/data"])

    assert args.command == "discover-temporal"
    assert args.data_path == Path("/path/to/data")
    assert args.output_dir == Path("discoveries")
    assert args.fanouts == [10, 10, 5]
    assert args.batch_size == 128
    assert args.time_window is None
    assert args.stitch_policy == "session"


def test_parse_args_invalid_stitch_policy():
    """Test argument parsing with invalid stitch policy."""
    with pytest.raises(SystemExit):
        _parse_args(
            ["discover-temporal", "--data-path", "/path/to/data", "--stitch-policy", "invalid"]
        )


def test_parse_args_missing_required():
    """Test argument parsing with missing required arguments."""
    with pytest.raises(SystemExit):
        _parse_args(["discover-temporal"])  # Missing --data-path


def test_parse_args_unknown_command():
    """Test argument parsing with unknown command."""
    with pytest.raises(SystemExit):
        _parse_args(["unknown-command"])


@patch("ironforge.sdk.cli.TemporalDiscoveryPipeline")
def test_main_discover_temporal(mock_pipeline_class):
    """Test main function with discover-temporal command."""
    mock_pipeline = Mock()
    mock_pipeline_class.return_value = mock_pipeline

    result = main(
        [
            "discover-temporal",
            "--data-path",
            "/test/data",
            "--fanouts",
            "5",
            "3",
            "--batch-size",
            "32",
        ]
    )

    assert result == 0
    mock_pipeline_class.assert_called_once_with(
        data_path=Path("/test/data"),
        num_neighbors=[5, 3],
        batch_size=32,
        time_window=None,
        stitch_policy="session",
    )
    mock_pipeline.run.assert_called_once()


def test_main_discover_temporal_import_error():
    """Test main function when TemporalDiscoveryPipeline is not available."""
    with patch("ironforge.sdk.cli.TemporalDiscoveryPipeline", None):
        with pytest.raises(ImportError, match="TemporalDiscoveryPipeline not available"):
            main(["discover-temporal", "--data-path", "/test/data"])


def test_main_unknown_command():
    """Test main function with unknown command."""
    with pytest.raises(NotImplementedError, match="Unknown command"):
        main(["unknown-command"])


def test_cli_integration_with_test_data():
    """Integration test: run CLI with test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create minimal test Parquet files
        import pandas as pd

        nodes_df = pd.DataFrame(
            {
                "node_id": [0, 1],
                "t": [100, 200],
                "kind": [1, 1],
                **{f"f{i}": [0.1, 0.2] for i in range(45)},
            }
        )

        edges_df = pd.DataFrame(
            {
                "src": [0],
                "dst": [1],
                "etype": [1],
                "dt": [10],
                **{f"e{i}": [0.5] for i in range(20)},
            }
        )

        nodes_file = temp_path / "test_nodes.parquet"
        edges_file = temp_path / "test_edges.parquet"

        nodes_df.to_parquet(nodes_file, index=False)
        edges_df.to_parquet(edges_file, index=False)

        output_dir = temp_path / "output"

        # Mock the discovery pipeline to avoid actual TGAT execution
        with patch("ironforge.sdk.cli.TemporalDiscoveryPipeline") as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline

            result = main(
                [
                    "discover-temporal",
                    "--data-path",
                    str(temp_path),
                    "--output-dir",
                    str(output_dir),
                    "--fanouts",
                    "2",
                    "1",
                    "--batch-size",
                    "2",
                ]
            )

            assert result == 0
            mock_pipeline.run.assert_called_once()


def test_cli_subprocess_execution():
    """Test CLI execution via subprocess (smoke test)."""
    # This is a smoke test to ensure the CLI module can be imported and executed
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from ironforge.sdk.cli import main; print('CLI import successful')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "CLI import successful" in result.stdout

    except subprocess.TimeoutExpired:
        pytest.skip("CLI import took too long (dependency issues)")
    except FileNotFoundError:
        pytest.skip("Python executable not found")


def test_cli_help_message():
    """Test CLI help message generation."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ironforge.sdk.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Should exit with 0 for help
        assert result.returncode == 0
        assert "discover-temporal" in result.stdout
        assert "IRONFORGE SDK" in result.stdout

    except subprocess.TimeoutExpired:
        pytest.skip("CLI help took too long")
    except FileNotFoundError:
        pytest.skip("Module execution not available")


@pytest.mark.parametrize(
    "fanouts,expected",
    [
        (["10"], [10]),
        (["10", "5"], [10, 5]),
        (["10", "5", "3"], [10, 5, 3]),
    ],
)
def test_fanouts_parsing(fanouts, expected):
    """Test various fanout configurations."""
    args = _parse_args(["discover-temporal", "--data-path", "/test", "--fanouts"] + fanouts)

    assert args.fanouts == expected


def test_time_window_none():
    """Test time window with None value."""
    args = _parse_args(["discover-temporal", "--data-path", "/test"])

    assert args.time_window is None


def test_time_window_integer():
    """Test time window with integer value."""
    args = _parse_args(["discover-temporal", "--data-path", "/test", "--time-window", "6"])

    assert args.time_window == 6


@patch("ironforge.sdk.cli.TemporalDiscoveryPipeline")
def test_main_with_all_options(mock_pipeline_class):
    """Test main function with all command-line options specified."""
    mock_pipeline = Mock()
    mock_pipeline_class.return_value = mock_pipeline

    result = main(
        [
            "discover-temporal",
            "--data-path",
            "/test/shards",
            "--output-dir",
            "/test/output",
            "--fanouts",
            "8",
            "4",
            "2",
            "--batch-size",
            "256",
            "--time-window",
            "3",
            "--stitch-policy",
            "global",
        ]
    )

    assert result == 0
    mock_pipeline_class.assert_called_once_with(
        data_path=Path("/test/shards"),
        num_neighbors=[8, 4, 2],
        batch_size=256,
        time_window=3,
        stitch_policy="global",
    )
    mock_pipeline.run.assert_called_once()
