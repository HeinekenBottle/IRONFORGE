"""Tests for the CLI report subcommand."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import numpy as np

    from ironforge.sdk.cli import _parse_args, main

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestCLIReport:
    """Test CLI report functionality."""

    @pytest.fixture
    def sample_json_data(self):
        """Create sample JSON data for testing."""
        return {
            "2025-08-12_NY-AM": {
                "minute_bins": [0, 1, 2, 3, 5, 10, 15, 20, 25, 30],
                "densities": [0.0, 2.0, 0.5, 1.8, 3.2, 1.1, 0.8, 2.5, 0.3, 1.0],
                "confluence": [55.0, 57.0, 62.0, 58.0, 65.0, 70.0, 75.0, 68.0, 60.0, 50.0],
                "markers": [15, 25],
            },
            "2025-08-12_LA-PM": {
                "minute_bins": [0, 2, 4, 6, 8, 12, 18, 24, 28, 35],
                "densities": [1.5, 0.8, 2.2, 1.7, 0.9, 2.8, 1.3, 3.1, 0.6, 2.0],
                "confluence": [40.0, 45.0, 50.0, 60.0, 55.0, 80.0, 85.0, 90.0, 75.0, 65.0],
                "markers": [12, 28],
            },
        }

    def test_parse_args_report_defaults(self):
        """Test default argument values for report command."""
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            args = _parse_args(["report", "--input-json", tmp.name])

            assert args.command == "report"
            assert args.input_json == Path(tmp.name)
            assert args.out_dir == Path("reports/minimal")
            assert args.sessions == []
            assert args.width == 1024
            assert args.height == 160
            assert args.strip_height == 54
            assert args.html is False

    def test_parse_args_report_custom(self):
        """Test custom argument values for report command."""
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            args = _parse_args(
                [
                    "report",
                    "--input-json",
                    tmp.name,
                    "--out-dir",
                    "custom/output",
                    "--sessions",
                    "session1",
                    "session2",
                    "--width",
                    "512",
                    "--height",
                    "80",
                    "--strip-height",
                    "32",
                    "--html",
                ]
            )

            assert args.input_json == Path(tmp.name)
            assert args.out_dir == Path("custom/output")
            assert args.sessions == ["session1", "session2"]
            assert args.width == 512
            assert args.height == 80
            assert args.strip_height == 32
            assert args.html is True

    def test_parse_args_report_missing_input_json(self):
        """Test that input-json is required."""
        with pytest.raises(SystemExit):
            _parse_args(["report"])

    @patch("ironforge.sdk.cli.json.loads")
    @patch("ironforge.sdk.cli.build_session_heatmap")
    @patch("ironforge.sdk.cli.build_confluence_strip")
    @patch("ironforge.sdk.cli.write_png")
    def test_main_report_basic_execution(
        self,
        mock_write_png,
        mock_build_confluence,
        mock_build_heatmap,
        mock_json_loads,
        sample_json_data,
    ):
        """Test basic report execution."""
        # Setup mocks
        mock_json_loads.return_value = sample_json_data
        mock_heatmap = object()  # Mock PIL Image
        mock_strip = object()  # Mock PIL Image
        mock_build_heatmap.return_value = mock_heatmap
        mock_build_confluence.return_value = mock_strip
        mock_write_png.return_value = Path("test.png")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary JSON file
            json_file = Path(temp_dir) / "test_data.json"
            json_file.write_text(json.dumps(sample_json_data))

            # Mock file existence check
            with patch.object(Path, "exists", return_value=True):
                # Run command
                result = main(["report", "--input-json", str(json_file), "--out-dir", temp_dir])

                # Check successful execution
                assert result == 0

                # Verify function calls
                assert mock_build_heatmap.call_count == 2  # Two sessions
                assert mock_build_confluence.call_count == 2
                assert mock_write_png.call_count == 4  # 2 heatmaps + 2 confluence strips

    @patch("ironforge.sdk.cli.json.loads")
    @patch("ironforge.sdk.cli.build_session_heatmap")
    @patch("ironforge.sdk.cli.build_confluence_strip")
    @patch("ironforge.sdk.cli.write_png")
    @patch("ironforge.sdk.cli.write_html")
    @patch("ironforge.sdk.cli.build_report_html")
    def test_main_report_with_html(
        self,
        mock_build_html,
        mock_write_html,
        mock_write_png,
        mock_build_confluence,
        mock_build_heatmap,
        mock_json_loads,
        sample_json_data,
    ):
        """Test report execution with HTML output."""
        # Setup mocks
        mock_json_loads.return_value = sample_json_data
        mock_heatmap = object()
        mock_strip = object()
        mock_build_heatmap.return_value = mock_heatmap
        mock_build_confluence.return_value = mock_strip
        mock_write_png.return_value = Path("test.png")
        mock_build_html.return_value = "<html>test</html>"
        mock_write_html.return_value = Path("index.html")

        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "test_data.json"
            json_file.write_text(json.dumps(sample_json_data))

            with patch.object(Path, "exists", return_value=True):
                result = main(
                    ["report", "--input-json", str(json_file), "--out-dir", temp_dir, "--html"]
                )

                assert result == 0

                # Verify HTML generation was called
                mock_build_html.assert_called_once()
                mock_write_html.assert_called_once()

                # Check HTML content contains both sessions
                html_call_args = mock_build_html.call_args[0]
                title = html_call_args[0]
                images = html_call_args[1]

                assert title == "IRONFORGE — Minimal Report"
                assert len(images) == 4  # 2 sessions × 2 images each

    @patch("ironforge.sdk.cli.json.loads")
    @patch("ironforge.sdk.cli.build_session_heatmap")
    @patch("ironforge.sdk.cli.build_confluence_strip")
    @patch("ironforge.sdk.cli.write_png")
    def test_main_report_specific_sessions(
        self,
        mock_write_png,
        mock_build_confluence,
        mock_build_heatmap,
        mock_json_loads,
        sample_json_data,
    ):
        """Test report generation for specific sessions only."""
        mock_json_loads.return_value = sample_json_data
        mock_build_heatmap.return_value = object()
        mock_build_confluence.return_value = object()
        mock_write_png.return_value = Path("test.png")

        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "test_data.json"
            json_file.write_text(json.dumps(sample_json_data))

            with patch.object(Path, "exists", return_value=True):
                result = main(
                    [
                        "report",
                        "--input-json",
                        str(json_file),
                        "--out-dir",
                        temp_dir,
                        "--sessions",
                        "2025-08-12_NY-AM",  # Only one session
                    ]
                )

                assert result == 0

                # Should only process one session
                assert mock_build_heatmap.call_count == 1
                assert mock_build_confluence.call_count == 1
                assert mock_write_png.call_count == 2  # 1 heatmap + 1 confluence

    def test_main_report_missing_input_file(self):
        """Test error handling for missing input file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_file = Path(temp_dir) / "nonexistent.json"

            with pytest.raises(SystemExit, match="Input JSON not found"):
                main(["report", "--input-json", str(nonexistent_file), "--out-dir", temp_dir])

    def test_main_report_import_error_handling(self):
        """Test error handling for missing dependencies."""
        with patch("ironforge.sdk.cli.json", side_effect=ImportError("json not available")):
            with tempfile.TemporaryDirectory() as temp_dir:
                json_file = Path(temp_dir) / "test.json"
                json_file.write_text('{"test": "data"}')

                with pytest.raises(SystemExit, match="Reporting dependencies missing"):
                    main(["report", "--input-json", str(json_file), "--out-dir", temp_dir])

    @patch("ironforge.sdk.cli.json.loads")
    @patch("ironforge.sdk.cli.build_session_heatmap")
    @patch("ironforge.sdk.cli.build_confluence_strip")
    @patch("ironforge.sdk.cli.write_png")
    def test_main_report_custom_dimensions(
        self,
        mock_write_png,
        mock_build_confluence,
        mock_build_heatmap,
        mock_json_loads,
        sample_json_data,
    ):
        """Test report generation with custom dimensions."""
        mock_json_loads.return_value = {
            "test": {"minute_bins": [0, 1], "densities": [1.0, 2.0], "confluence": [50.0, 75.0]}
        }
        mock_build_heatmap.return_value = object()
        mock_build_confluence.return_value = object()
        mock_write_png.return_value = Path("test.png")

        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "test_data.json"
            json_file.write_text('{"test": {}}')

            with patch.object(Path, "exists", return_value=True):
                result = main(
                    [
                        "report",
                        "--input-json",
                        str(json_file),
                        "--out-dir",
                        temp_dir,
                        "--width",
                        "512",
                        "--height",
                        "80",
                        "--strip-height",
                        "32",
                    ]
                )

                assert result == 0

                # Check that custom specs were used
                heatmap_call = mock_build_heatmap.call_args
                confluence_call = mock_build_confluence.call_args

                heatmap_spec = heatmap_call[0][2]  # Third argument is spec
                confluence_spec = confluence_call[0][3]  # Fourth argument is spec

                assert heatmap_spec.width == 512
                assert heatmap_spec.height == 80
                assert confluence_spec.width == 512
                assert confluence_spec.height == 32

    def test_main_report_data_without_confluence_or_markers(self):
        """Test handling of data missing optional fields."""
        minimal_data = {
            "session1": {
                "minute_bins": [0, 5, 10],
                "densities": [1.0, 2.0, 1.5],
                # No confluence or markers
            }
        }

        with patch("ironforge.sdk.cli.json.loads", return_value=minimal_data):
            with patch(
                "ironforge.sdk.cli.build_session_heatmap", return_value=object()
            ) as mock_heatmap:
                with patch(
                    "ironforge.sdk.cli.build_confluence_strip", return_value=object()
                ) as mock_confluence:
                    with patch("ironforge.sdk.cli.write_png", return_value=Path("test.png")):
                        with tempfile.TemporaryDirectory() as temp_dir:
                            json_file = Path(temp_dir) / "minimal.json"
                            json_file.write_text(json.dumps(minimal_data))

                            with patch.object(Path, "exists", return_value=True):
                                result = main(
                                    [
                                        "report",
                                        "--input-json",
                                        str(json_file),
                                        "--out-dir",
                                        temp_dir,
                                    ]
                                )

                                assert result == 0

                                # Check that defaults were used for missing fields
                                confluence_call = mock_confluence.call_args
                                confluence_scores = confluence_call[0][1]
                                markers = (
                                    confluence_call[0][2] if len(confluence_call[0]) > 2 else None
                                )

                                # Should have zeros for confluence and None for markers
                                assert len(confluence_scores) == 3  # Same length as minute_bins
                                assert markers is None


@pytest.mark.skipif(NUMPY_AVAILABLE, reason="Testing without numpy")
def test_report_command_without_numpy():
    """Test report command gracefully handles missing numpy."""
    with pytest.raises(SystemExit):
        # This should fail early due to import issues
        main(["report", "--input-json", "test.json"])
