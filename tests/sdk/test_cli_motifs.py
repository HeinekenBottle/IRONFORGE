"""Tests for CLI motifs and prepare-motifs-input commands."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ironforge.sdk.cli import _parse_args, main


class TestParseMotifsArgs:
    """Test argument parsing for motifs command."""

    def test_parse_motifs_basic(self):
        """Test basic motifs argument parsing."""
        args = _parse_args(["motifs", "--input-json", "/path/to/input.json"])

        assert args.command == "motifs"
        assert args.input_json == Path("/path/to/input.json")
        assert args.min_confluence == 65.0
        assert args.top_k == 3
        assert args.preset == "default"

    def test_parse_motifs_all_options(self):
        """Test motifs with all options."""
        args = _parse_args(
            [
                "motifs",
                "--input-json",
                "/path/to/input.json",
                "--min-confluence",
                "70.5",
                "--top-k",
                "5",
                "--preset",
                "custom",
            ]
        )

        assert args.command == "motifs"
        assert args.input_json == Path("/path/to/input.json")
        assert args.min_confluence == 70.5
        assert args.top_k == 5
        assert args.preset == "custom"

    def test_parse_motifs_missing_required(self):
        """Test motifs with missing required argument."""
        with pytest.raises(SystemExit):
            _parse_args(["motifs"])  # Missing --input-json


class TestParsePrepareArgs:
    """Test argument parsing for prepare-motifs-input command."""

    def test_parse_prepare_basic(self):
        """Test basic prepare-motifs-input argument parsing."""
        args = _parse_args(
            [
                "prepare-motifs-input",
                "--discovery-json",
                "/path/to/discovery.json",
                "--out",
                "/path/to/output.json",
            ]
        )

        assert args.command == "prepare-motifs-input"
        assert args.discovery_json == Path("/path/to/discovery.json")
        assert args.out == Path("/path/to/output.json")
        assert args.validation_json is None

    def test_parse_prepare_with_validation(self):
        """Test prepare-motifs-input with validation file."""
        args = _parse_args(
            [
                "prepare-motifs-input",
                "--discovery-json",
                "/path/to/discovery.json",
                "--validation-json",
                "/path/to/validation.json",
                "--out",
                "/path/to/output.json",
            ]
        )

        assert args.command == "prepare-motifs-input"
        assert args.discovery_json == Path("/path/to/discovery.json")
        assert args.validation_json == Path("/path/to/validation.json")
        assert args.out == Path("/path/to/output.json")

    def test_parse_prepare_missing_required(self):
        """Test prepare-motifs-input with missing required arguments."""
        with pytest.raises(SystemExit):
            _parse_args(["prepare-motifs-input"])  # Missing required args

        with pytest.raises(SystemExit):
            _parse_args(
                [
                    "prepare-motifs-input",
                    "--discovery-json",
                    "/path/to/discovery.json",
                    # Missing --out
                ]
            )


class TestMainMotifs:
    """Test main function with motifs command."""

    @patch("ironforge.motifs.scanner.run_cli_scan")
    def test_main_motifs_basic(self, mock_run_cli_scan):
        """Test main function with basic motifs command."""
        result = main(["motifs", "--input-json", "/fake/input.json"])

        assert result == 0
        mock_run_cli_scan.assert_called_once_with(
            Path("/fake/input.json"), top_k=3, min_confluence=65.0, preset="default"
        )

    @patch("ironforge.motifs.scanner.run_cli_scan")
    def test_main_motifs_all_options(self, mock_run_cli_scan):
        """Test main function with all motifs options."""
        result = main(
            [
                "motifs",
                "--input-json",
                "/fake/input.json",
                "--min-confluence",
                "80.0",
                "--top-k",
                "10",
                "--preset",
                "advanced",
            ]
        )

        assert result == 0
        mock_run_cli_scan.assert_called_once_with(
            Path("/fake/input.json"), top_k=10, min_confluence=80.0, preset="advanced"
        )

    def test_main_motifs_import_error(self):
        """Test main function when motifs scanner is not available."""
        # Test by mocking the import failure within the CLI module
        with patch.dict("sys.modules", {"ironforge.motifs.scanner": None}):
            with pytest.raises(SystemExit, match="Motif scanner unavailable"):
                main(["motifs", "--input-json", "/fake/input.json"])

    @patch("ironforge.motifs.scanner.run_cli_scan")
    def test_main_motifs_integration(self, mock_run_cli_scan):
        """Test motifs command with real file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test input file
            input_data = {
                "session_001": {
                    "events": [
                        {"type": "sweep", "minute": 10, "htf_under_mid": False},
                        {"type": "fvg_redelivery", "minute": 25, "htf_under_mid": True},
                    ],
                    "confluence": [70.0, 75.0, 80.0, 85.0, 90.0] + [75.0] * 50,
                }
            }

            input_file = temp_path / "motifs_input.json"
            input_file.write_text(json.dumps(input_data))

            result = main(
                [
                    "motifs",
                    "--input-json",
                    str(input_file),
                    "--min-confluence",
                    "70.0",
                    "--top-k",
                    "5",
                ]
            )

            assert result == 0
            mock_run_cli_scan.assert_called_once_with(
                input_file, top_k=5, min_confluence=70.0, preset="default"
            )


class TestMainPrepareMotifs:
    """Test main function with prepare-motifs-input command."""

    @patch("ironforge.scripts.prepare_motifs_input.main")
    def test_main_prepare_basic(self, mock_prep_main):
        """Test main function with basic prepare-motifs-input command."""
        mock_prep_main.return_value = 0

        result = main(
            [
                "prepare-motifs-input",
                "--discovery-json",
                "/fake/discovery.json",
                "--out",
                "/fake/output.json",
            ]
        )

        assert result == 0
        mock_prep_main.assert_called_once_with(
            ["--discovery-json", "/fake/discovery.json", "--out", "/fake/output.json"]
        )

    @patch("ironforge.scripts.prepare_motifs_input.main")
    def test_main_prepare_with_validation(self, mock_prep_main):
        """Test main function with validation file."""
        mock_prep_main.return_value = 0

        result = main(
            [
                "prepare-motifs-input",
                "--discovery-json",
                "/fake/discovery.json",
                "--validation-json",
                "/fake/validation.json",
                "--out",
                "/fake/output.json",
            ]
        )

        assert result == 0
        mock_prep_main.assert_called_once_with(
            [
                "--discovery-json",
                "/fake/discovery.json",
                "--out",
                "/fake/output.json",
                "--validation-json",
                "/fake/validation.json",
            ]
        )

    def test_main_prepare_import_error(self):
        """Test main function when prepare script is not available."""
        with patch.dict("sys.modules", {"ironforge.scripts.prepare_motifs_input": None}):
            with pytest.raises(SystemExit, match="Adapter unavailable"):
                main(
                    [
                        "prepare-motifs-input",
                        "--discovery-json",
                        "/fake/discovery.json",
                        "--out",
                        "/fake/output.json",
                    ]
                )

    @patch("ironforge.scripts.prepare_motifs_input.main")
    def test_main_prepare_error_return(self, mock_prep_main):
        """Test main function when prepare script returns error."""
        mock_prep_main.return_value = 1  # Error code

        result = main(
            [
                "prepare-motifs-input",
                "--discovery-json",
                "/fake/discovery.json",
                "--out",
                "/fake/output.json",
            ]
        )

        assert result == 1  # Should propagate error code


class TestMotifsWorkflow:
    """Test complete motifs workflow."""

    def test_end_to_end_workflow(self):
        """Test complete prepare → motifs workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Create discovery data
            discovery_data = {
                "session_001": {
                    "events": [
                        {"type": "sweep", "minute": 5, "htf_under_mid": False},
                        {"type": "fvg_redelivery", "minute": 20, "htf_under_mid": True},
                    ]
                },
                "session_002": {
                    "event_types": ["fpfvg"],
                    "event_minutes": [15],
                    "event_htf_under_mid": [False],
                },
            }

            validation_data = {
                "session_001": {"confluence": [70.0, 75.0, 80.0, 85.0, 90.0] + [75.0] * 50},
                "session_002": {"confluence": [65.0, 70.0, 75.0] + [70.0] * 50},
            }

            discovery_file = temp_path / "discovery.json"
            validation_file = temp_path / "validation.json"
            motifs_input_file = temp_path / "motifs_input.json"

            discovery_file.write_text(json.dumps(discovery_data))
            validation_file.write_text(json.dumps(validation_data))

            # Step 2: Run prepare-motifs-input
            result = main(
                [
                    "prepare-motifs-input",
                    "--discovery-json",
                    str(discovery_file),
                    "--validation-json",
                    str(validation_file),
                    "--out",
                    str(motifs_input_file),
                ]
            )

            assert result == 0
            assert motifs_input_file.exists()

            # Verify motifs input format
            motifs_data = json.loads(motifs_input_file.read_text())
            assert "session_001" in motifs_data
            assert "session_002" in motifs_data
            assert "events" in motifs_data["session_001"]
            assert "confluence" in motifs_data["session_001"]

            # Step 3: Run motifs scanner
            with patch("ironforge.motifs.scanner.run_cli_scan") as mock_scan:
                mock_scan.return_value = None  # Scanner prints output directly

                result = main(
                    [
                        "motifs",
                        "--input-json",
                        str(motifs_input_file),
                        "--min-confluence",
                        "65.0",
                        "--top-k",
                        "3",
                    ]
                )

                assert result == 0
                mock_scan.assert_called_once()

    @patch("ironforge.motifs.scanner.run_cli_scan")
    def test_motifs_with_real_data_structure(self, mock_run_cli_scan):
        """Test motifs command with realistic data structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create realistic motifs input
            motifs_input = {
                "NY_AM_2025_08_05": {
                    "events": [
                        {"type": "fpfvg", "minute": 12, "htf_under_mid": False},
                        {"type": "sweep", "minute": 35, "htf_under_mid": True},
                        {"type": "expansion", "minute": 45, "htf_under_mid": False},
                    ],
                    "confluence": [60.0] * 10 + [80.0] * 30 + [70.0] * 20,
                },
                "NY_PM_2025_08_05": {
                    "events": [
                        {"type": "sweep", "minute": 8, "htf_under_mid": False},
                        {"type": "fvg_redelivery", "minute": 22, "htf_under_mid": True},
                        {"type": "consolidation", "minute": 40, "htf_under_mid": True},
                    ],
                    "confluence": [75.0] * 15 + [85.0] * 25 + [65.0] * 20,
                },
            }

            input_file = temp_path / "realistic_input.json"
            input_file.write_text(json.dumps(motifs_input))

            result = main(
                [
                    "motifs",
                    "--input-json",
                    str(input_file),
                    "--min-confluence",
                    "70.0",
                    "--top-k",
                    "5",
                ]
            )

            assert result == 0
            mock_run_cli_scan.assert_called_once()


class TestDiscoverTemporalWithConfluence:
    """Test discover-temporal command with --with-confluence flag."""

    def test_parse_with_confluence_flag(self):
        """Test parsing --with-confluence flag."""
        args = _parse_args(["discover-temporal", "--data-path", "/test/data", "--with-confluence"])

        assert args.command == "discover-temporal"
        assert args.with_confluence is True

    def test_parse_without_confluence_flag(self):
        """Test default value for confluence flag."""
        args = _parse_args(["discover-temporal", "--data-path", "/test/data"])

        assert args.command == "discover-temporal"
        assert args.with_confluence is False

    @patch("ironforge.sdk.cli.TemporalDiscoveryPipeline")
    def test_main_discover_with_confluence(self, mock_pipeline_class):
        """Test main function passes confluence flag to pipeline."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        result = main(["discover-temporal", "--data-path", "/test/data", "--with-confluence"])

        assert result == 0
        mock_pipeline_class.assert_called_once_with(
            data_path=Path("/test/data"),
            num_neighbors=[10, 10, 5],
            batch_size=128,
            time_window=None,
            stitch_policy="session",
            with_confluence=True,
        )
        mock_pipeline.run.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_motifs_with_nonexistent_file(self):
        """Test motifs command with nonexistent input file."""
        # This would be caught by the scanner, not the CLI itself
        with patch("ironforge.motifs.scanner.run_cli_scan") as mock_scan:
            mock_scan.side_effect = FileNotFoundError("File not found")

            with pytest.raises(FileNotFoundError):
                main(["motifs", "--input-json", "/nonexistent/file.json"])

    def test_prepare_with_invalid_json(self):
        """Test prepare-motifs-input with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid JSON file
            invalid_file = temp_path / "invalid.json"
            invalid_file.write_text("{ invalid json content")

            output_file = temp_path / "output.json"

            # This should propagate the JSON decode error
            with pytest.raises(json.JSONDecodeError):  # Should raise JSON decode error directly
                main(
                    [
                        "prepare-motifs-input",
                        "--discovery-json",
                        str(invalid_file),
                        "--out",
                        str(output_file),
                    ]
                )

    @patch("ironforge.motifs.scanner.run_cli_scan")
    def test_motifs_scanner_exception(self, mock_run_cli_scan):
        """Test handling of scanner exceptions."""
        mock_run_cli_scan.side_effect = Exception("Scanner internal error")

        with pytest.raises(Exception, match="Scanner internal error"):
            main(["motifs", "--input-json", "/fake/input.json"])

    def test_invalid_confluence_threshold(self):
        """Test invalid confluence threshold values."""
        # Negative threshold
        args = _parse_args(
            ["motifs", "--input-json", "/fake/input.json", "--min-confluence", "-10.0"]
        )
        assert args.min_confluence == -10.0  # Parser allows it, scanner should handle

        # Very high threshold
        args = _parse_args(
            ["motifs", "--input-json", "/fake/input.json", "--min-confluence", "150.0"]
        )
        assert args.min_confluence == 150.0  # Parser allows it, scanner should handle

    def test_invalid_top_k_values(self):
        """Test invalid top-k values."""
        # Zero top-k
        args = _parse_args(["motifs", "--input-json", "/fake/input.json", "--top-k", "0"])
        assert args.top_k == 0

        # Negative top-k
        args = _parse_args(["motifs", "--input-json", "/fake/input.json", "--top-k", "-5"])
        assert args.top_k == -5


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_help_messages(self):
        """Test help messages for new commands."""
        import subprocess

        try:
            # Test motifs help
            result = subprocess.run(
                [sys.executable, "-m", "ironforge.sdk.cli", "motifs", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode == 0
            assert "motif card matches" in result.stdout
            assert "--input-json" in result.stdout
            assert "--min-confluence" in result.stdout

            # Test prepare-motifs-input help
            result = subprocess.run(
                [sys.executable, "-m", "ironforge.sdk.cli", "prepare-motifs-input", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode == 0
            assert "discovery/validation JSONs" in result.stdout
            assert "--discovery-json" in result.stdout
            assert "--out" in result.stdout

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI help test skipped (environment issues)")

    def test_command_discovery(self):
        """Test that new commands are discoverable in main help."""
        import subprocess

        try:
            result = subprocess.run(
                [sys.executable, "-m", "ironforge.sdk.cli", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode == 0
            assert "motifs" in result.stdout
            assert "prepare-motifs-input" in result.stdout

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI discovery test skipped (environment issues)")
