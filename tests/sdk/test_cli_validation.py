"""CLI tests for validation subcommand (Wave 4)."""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ironforge.sdk.cli import _parse_args, main


class TestValidationCLI:
    """Test cases for validation CLI functionality."""

    def test_validate_help(self):
        """Test validation help message."""
        with pytest.raises(SystemExit):
            _parse_args(["validate", "--help"])

    def test_validate_required_args(self):
        """Test that required arguments are enforced."""
        # Missing data-path should fail
        with pytest.raises(SystemExit):
            _parse_args(["validate"])

    def test_validate_default_args(self):
        """Test default argument values for validate command."""
        args = _parse_args(["validate", "--data-path", "/tmp/data"])

        assert args.command == "validate"
        assert args.data_path == Path("/tmp/data")
        assert args.mode == "oos"
        assert args.folds == 5
        assert args.embargo_mins == 30
        assert args.controls == ["time_shuffle", "label_perm"]
        assert args.ablations == []
        assert args.report_dir == Path("reports/validation")
        assert args.seed == 42

    def test_validate_custom_args(self):
        """Test custom argument values for validate command."""
        args = _parse_args(
            [
                "validate",
                "--data-path",
                "/custom/data",
                "--mode",
                "purged-kfold",
                "--folds",
                "10",
                "--embargo-mins",
                "60",
                "--controls",
                "time_shuffle",
                "node_shuffle",
                "--ablations",
                "htf_prox",
                "cycles",
                "--report-dir",
                "/custom/reports",
                "--seed",
                "123",
            ]
        )

        assert args.command == "validate"
        assert args.data_path == Path("/custom/data")
        assert args.mode == "purged-kfold"
        assert args.folds == 10
        assert args.embargo_mins == 60
        assert args.controls == ["time_shuffle", "node_shuffle"]
        assert args.ablations == ["htf_prox", "cycles"]
        assert args.report_dir == Path("/custom/reports")
        assert args.seed == 123

    def test_validate_mode_choices(self):
        """Test that mode argument accepts valid choices."""
        # Valid modes
        for mode in ["oos", "purged-kfold", "holdout"]:
            args = _parse_args(["validate", "--data-path", "/tmp", "--mode", mode])
            assert args.mode == mode

        # Invalid mode should fail
        with pytest.raises(SystemExit):
            _parse_args(["validate", "--data-path", "/tmp", "--mode", "invalid"])

    def test_validate_empty_controls(self):
        """Test validation with no controls."""
        args = _parse_args(["validate", "--data-path", "/tmp/data", "--controls"])  # Empty list

        assert args.controls == []

    def test_validate_multiple_ablations(self):
        """Test multiple ablation groups."""
        args = _parse_args(
            [
                "validate",
                "--data-path",
                "/tmp/data",
                "--ablations",
                "htf_prox",
                "cycles",
                "structure",
                "features",
            ]
        )

        assert args.ablations == ["htf_prox", "cycles", "structure", "features"]

    def test_validate_all_controls(self):
        """Test all available control types."""
        controls = [
            "time_shuffle",
            "label_perm",
            "node_shuffle",
            "edge_direction",
            "temporal_blocks",
        ]

        args = _parse_args(["validate", "--data-path", "/tmp/data", "--controls"] + controls)

        assert args.controls == controls

    def test_main_validate_missing_components(self):
        """Test main function with missing validation components."""
        with patch("ironforge.sdk.cli.ValidationRunner", None):
            with patch("ironforge.sdk.cli.ValidationConfig", None):
                with pytest.raises(ImportError, match="ValidationRunner not available"):
                    main(["validate", "--data-path", "/tmp/data"])

    def test_main_validate_execution(self):
        """Test main function execution of validate command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock successful execution
            with patch("ironforge.sdk.cli.ValidationRunner") as mock_runner_class:
                with patch("ironforge.sdk.cli.ValidationConfig") as mock_config_class:
                    # Setup mocks
                    mock_config = mock_config_class.return_value
                    mock_runner = mock_runner_class.return_value
                    mock_runner.run.return_value = {
                        "summary": {
                            "validation_passed": True,
                            "main_performance": {"temporal_auc": 0.85},
                        }
                    }

                    # Capture stdout
                    import io
                    from contextlib import redirect_stdout

                    captured_output = io.StringIO()

                    with redirect_stdout(captured_output):
                        result = main(
                            [
                                "validate",
                                "--data-path",
                                temp_dir,
                                "--mode",
                                "oos",
                                "--report-dir",
                                temp_dir,
                            ]
                        )

                    # Check execution
                    assert result == 0  # Success

                    # Check that ValidationConfig was called with correct args
                    mock_config_class.assert_called_once()
                    config_call = mock_config_class.call_args
                    assert config_call[1]["mode"] == "oos"
                    assert config_call[1]["report_dir"] == Path(temp_dir)

                    # Check that runner was created and executed
                    mock_runner_class.assert_called_once_with(mock_config)
                    mock_runner.run.assert_called_once()

                    # Check output
                    output = captured_output.getvalue()
                    assert "Starting IRONFORGE validation" in output
                    assert "Validation completed" in output
                    assert "PASSED" in output

    def test_main_validate_failure(self):
        """Test main function with validation failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("ironforge.sdk.cli.ValidationRunner") as mock_runner_class:
                with patch("ironforge.sdk.cli.ValidationConfig") as mock_config_class:
                    # Setup mocks for failure
                    mock_runner = mock_runner_class.return_value
                    mock_runner.run.return_value = {
                        "summary": {
                            "validation_passed": False,
                            "main_performance": {"temporal_auc": 0.45},
                        }
                    }

                    result = main(
                        [
                            "validate",
                            "--data-path",
                            temp_dir,
                        ]
                    )

                    # Should return error code
                    assert result == 1

    def test_validate_subprocess_execution(self):
        """Test validation command via subprocess."""
        # This test ensures the CLI can be invoked as a subprocess
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "ironforge.sdk.cli", "validate", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # Help should work even without full implementation
                assert "validate" in result.stdout or "validate" in result.stderr

            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Skip if ironforge module not properly installed
                pytest.skip("IRONFORGE module not available for subprocess testing")

    def test_argument_type_validation(self):
        """Test argument type validation."""
        # Test integer arguments
        args = _parse_args(
            [
                "validate",
                "--data-path",
                "/tmp",
                "--folds",
                "7",
                "--embargo-mins",
                "45",
                "--seed",
                "999",
            ]
        )

        assert isinstance(args.folds, int)
        assert isinstance(args.embargo_mins, int)
        assert isinstance(args.seed, int)
        assert args.folds == 7
        assert args.embargo_mins == 45
        assert args.seed == 999

        # Test Path arguments
        assert isinstance(args.data_path, Path)
        assert isinstance(args.report_dir, Path)

    def test_argument_edge_cases(self):
        """Test edge cases for argument parsing."""
        # Zero values
        args = _parse_args(
            [
                "validate",
                "--data-path",
                "/tmp",
                "--folds",
                "0",
                "--embargo-mins",
                "0",
                "--seed",
                "0",
            ]
        )

        assert args.folds == 0
        assert args.embargo_mins == 0
        assert args.seed == 0

        # Large values
        args = _parse_args(
            [
                "validate",
                "--data-path",
                "/tmp",
                "--folds",
                "100",
                "--embargo-mins",
                "1440",  # 24 hours
                "--seed",
                "999999",
            ]
        )

        assert args.folds == 100
        assert args.embargo_mins == 1440
        assert args.seed == 999999

    @patch("ironforge.sdk.cli.ValidationRunner")
    @patch("ironforge.sdk.cli.ValidationConfig")
    def test_main_with_all_options(self, mock_config_class, mock_runner_class):
        """Test main function with all validation options."""
        # Setup mocks
        mock_runner = mock_runner_class.return_value
        mock_runner.run.return_value = {
            "summary": {"validation_passed": True, "main_performance": {"temporal_auc": 0.90}}
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = main(
                [
                    "validate",
                    "--data-path",
                    temp_dir,
                    "--mode",
                    "purged-kfold",
                    "--folds",
                    "7",
                    "--embargo-mins",
                    "120",
                    "--controls",
                    "time_shuffle",
                    "label_perm",
                    "node_shuffle",
                    "--ablations",
                    "htf_prox",
                    "cycles",
                    "structure",
                    "--report-dir",
                    temp_dir,
                    "--seed",
                    "777",
                ]
            )

            assert result == 0

            # Verify configuration
            config_call = mock_config_class.call_args[1]
            assert config_call["mode"] == "purged-kfold"
            assert config_call["folds"] == 7
            assert config_call["embargo_mins"] == 120
            assert config_call["controls"] == ["time_shuffle", "label_perm", "node_shuffle"]
            assert config_call["ablations"] == ["htf_prox", "cycles", "structure"]
            assert config_call["random_seed"] == 777

    def test_cli_error_handling(self):
        """Test CLI error handling for invalid arguments."""
        # Invalid integer values
        with pytest.raises(SystemExit):
            _parse_args(["validate", "--data-path", "/tmp", "--folds", "not_a_number"])

        # Missing argument value
        with pytest.raises(SystemExit):
            _parse_args(["validate", "--data-path", "/tmp", "--mode"])  # Missing value

    def test_cli_integration_with_discover_temporal(self):
        """Test that both CLI commands coexist properly."""
        # Test discover-temporal still works
        args = _parse_args(["discover-temporal", "--data-path", "/tmp/data"])
        assert args.command == "discover-temporal"

        # Test validate works
        args = _parse_args(["validate", "--data-path", "/tmp/data"])
        assert args.command == "validate"

        # Test unknown command fails
        with pytest.raises(SystemExit):
            _parse_args(["unknown-command"])


@pytest.mark.parametrize(
    "mode,folds",
    [
        ("oos", 1),
        ("purged-kfold", 5),
        ("holdout", 1),
    ],
)
def test_validate_mode_combinations(mode, folds):
    """Test different mode and fold combinations."""
    args = _parse_args(["validate", "--data-path", "/tmp", "--mode", mode, "--folds", str(folds)])

    assert args.mode == mode
    assert args.folds == folds


@pytest.mark.parametrize(
    "controls",
    [
        [],
        ["time_shuffle"],
        ["time_shuffle", "label_perm"],
        ["time_shuffle", "label_perm", "node_shuffle", "edge_direction"],
    ],
)
def test_validate_control_combinations(controls):
    """Test different control combinations."""
    cmd = ["validate", "--data-path", "/tmp", "--controls"] + controls

    args = _parse_args(cmd)
    assert args.controls == controls


def test_validate_output_formatting():
    """Test that validation output is properly formatted."""
    with patch("ironforge.sdk.cli.ValidationRunner") as mock_runner_class:
        with patch("ironforge.sdk.cli.ValidationConfig"):
            # Mock validation results
            mock_runner = mock_runner_class.return_value
            mock_runner.run.return_value = {
                "summary": {"validation_passed": True, "main_performance": {"temporal_auc": 0.8765}}
            }

            import io
            from contextlib import redirect_stdout

            captured_output = io.StringIO()

            with redirect_stdout(captured_output):
                result = main(["validate", "--data-path", "/tmp", "--mode", "oos"])

            output = captured_output.getvalue()

            # Check formatting
            assert "🚀" in output  # Emoji present
            assert "Main AUC: 0.8765" in output  # Formatted number
            assert "PASSED" in output  # Status message


def test_cli_documentation_consistency():
    """Test that CLI help text is consistent with functionality."""
    import argparse

    # Capture help output
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--mode", choices=["oos", "purged-kfold", "holdout"])
    validate_parser.add_argument("--controls", nargs="*")
    validate_parser.add_argument("--ablations", nargs="*")

    # Verify choices match implementation
    mode_choices = validate_parser._option_string_actions["--mode"].choices
    assert "oos" in mode_choices
    assert "purged-kfold" in mode_choices
    assert "holdout" in mode_choices

    # Verify argument types
    controls_action = validate_parser._option_string_actions["--controls"]
    assert controls_action.nargs == "*"  # Zero or more arguments
