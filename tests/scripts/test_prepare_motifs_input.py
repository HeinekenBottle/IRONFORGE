"""Tests for prepare_motifs_input script."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ironforge.scripts.prepare_motifs_input import (
    _coerce_bool,
    _extract_events,
    build_motifs_input,
    main,
)


class TestCoerceBool:
    """Test _coerce_bool utility function."""

    def test_bool_input(self):
        """Test with boolean input."""
        assert _coerce_bool(True) is True
        assert _coerce_bool(False) is False

    def test_int_float_input(self):
        """Test with numeric input."""
        assert _coerce_bool(1) is True
        assert _coerce_bool(0) is False
        assert _coerce_bool(1.0) is True
        assert _coerce_bool(0.0) is False

    def test_string_input(self):
        """Test with string input."""
        assert _coerce_bool("true") is True
        assert _coerce_bool("True") is True
        assert _coerce_bool("1") is True
        assert _coerce_bool("yes") is True
        assert _coerce_bool("y") is True
        assert _coerce_bool("t") is True

        assert _coerce_bool("false") is False
        assert _coerce_bool("False") is False
        assert _coerce_bool("0") is False
        assert _coerce_bool("no") is False
        assert _coerce_bool("n") is False
        assert _coerce_bool("f") is False
        assert _coerce_bool("random") is False

    def test_other_input(self):
        """Test with other input types."""
        assert _coerce_bool(None) is False
        assert _coerce_bool([]) is False
        assert _coerce_bool({}) is False


class TestExtractEvents:
    """Test _extract_events function."""

    def test_events_list_format(self):
        """Test extraction from events list format."""
        session_payload = {
            "events": [
                {"type": "sweep", "minute": 10, "htf_under_mid": True},
                {"type": "fvg_redelivery", "minute": 25, "htf_under_mid": False},
            ]
        }

        events = _extract_events(session_payload)

        assert len(events) == 2
        assert events[0]["type"] == "sweep"
        assert events[0]["minute"] == 10
        assert events[0]["htf_under_mid"] is True

        assert events[1]["type"] == "fvg_redelivery"
        assert events[1]["minute"] == 25
        assert events[1]["htf_under_mid"] is False

    def test_zipped_arrays_format(self):
        """Test extraction from zipped arrays format."""
        session_payload = {
            "event_types": ["sweep", "expansion"],
            "event_minutes": [15, 30],
            "event_htf_under_mid": [1, 0],  # Will be coerced to bool
        }

        events = _extract_events(session_payload)

        assert len(events) == 2
        assert events[0]["type"] == "sweep"
        assert events[0]["minute"] == 15
        assert events[0]["htf_under_mid"] is True

        assert events[1]["type"] == "expansion"
        assert events[1]["minute"] == 30
        assert events[1]["htf_under_mid"] is False

    def test_missing_fields_zipped_arrays(self):
        """Test extraction with missing fields in zipped arrays."""
        session_payload = {
            "event_types": ["sweep", "expansion", "consolidation"],
            "event_minutes": [15, 30],  # Missing third minute
            "event_htf_under_mid": [True],  # Missing second and third
        }

        events = _extract_events(session_payload)

        assert len(events) == 3
        assert events[0]["minute"] == 15
        assert events[1]["minute"] == 30
        assert events[2]["minute"] == 0  # Default

        assert events[0]["htf_under_mid"] is True
        assert events[1]["htf_under_mid"] is False  # Default
        assert events[2]["htf_under_mid"] is False  # Default

    def test_empty_payload(self):
        """Test extraction from empty payload."""
        events = _extract_events({})
        assert events == []

    def test_events_with_missing_fields(self):
        """Test events with missing fields."""
        session_payload = {
            "events": [
                {"type": "sweep"},  # Missing minute and htf_under_mid
                {"minute": 25},  # Missing type and htf_under_mid
                {},  # Missing all fields
            ]
        }

        events = _extract_events(session_payload)

        assert len(events) == 3
        assert events[0]["type"] == "sweep"
        assert events[0]["minute"] == 0
        assert events[0]["htf_under_mid"] is False

        assert events[1]["type"] == ""
        assert events[1]["minute"] == 25
        assert events[1]["htf_under_mid"] is False

        assert events[2]["type"] == ""
        assert events[2]["minute"] == 0
        assert events[2]["htf_under_mid"] is False


class TestBuildMotifsInput:
    """Test build_motifs_input function."""

    def test_discovery_only(self):
        """Test with discovery JSON only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create discovery JSON
            discovery_data = {
                "session_001": {
                    "events": [{"type": "sweep", "minute": 10, "htf_under_mid": True}],
                    "confluence": [80.0, 85.0, 90.0],
                },
                "session_002": {
                    "event_types": ["expansion"],
                    "event_minutes": [20],
                    "event_htf_under_mid": [False],
                },
            }

            discovery_file = temp_path / "discovery.json"
            discovery_file.write_text(json.dumps(discovery_data))

            result = build_motifs_input(discovery_file)

            assert "session_001" in result
            assert "session_002" in result

            # Check session_001
            s1 = result["session_001"]
            assert len(s1["events"]) == 1
            assert s1["events"][0]["type"] == "sweep"
            assert s1["confluence"] == [80.0, 85.0, 90.0]

            # Check session_002
            s2 = result["session_002"]
            assert len(s2["events"]) == 1
            assert s2["events"][0]["type"] == "expansion"
            assert s2["confluence"] == []  # No confluence provided

    def test_discovery_with_validation(self):
        """Test with both discovery and validation JSONs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create discovery JSON
            discovery_data = {
                "session_001": {
                    "events": [{"type": "sweep", "minute": 10, "htf_under_mid": True}]
                    # No confluence in discovery
                },
                "session_002": {
                    "events": [{"type": "expansion", "minute": 20, "htf_under_mid": False}],
                    "confluence": [70.0, 75.0],  # Has confluence in discovery
                },
            }

            # Create validation JSON
            validation_data = {
                "session_001": {"confluence": [65.0, 70.0, 75.0]},  # Confluence from validation
                "session_003": {"confluence": [60.0, 65.0]},  # Session not in discovery
            }

            discovery_file = temp_path / "discovery.json"
            validation_file = temp_path / "validation.json"

            discovery_file.write_text(json.dumps(discovery_data))
            validation_file.write_text(json.dumps(validation_data))

            result = build_motifs_input(discovery_file, validation_file)

            # Should only have sessions from discovery
            assert set(result.keys()) == {"session_001", "session_002"}

            # session_001 should get confluence from validation
            assert result["session_001"]["confluence"] == [65.0, 70.0, 75.0]

            # session_002 should keep confluence from discovery
            assert result["session_002"]["confluence"] == [70.0, 75.0]

    def test_confluence_priority(self):
        """Test confluence priority: discovery > validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            discovery_data = {
                "session_001": {
                    "events": [],
                    "confluence": [80.0, 85.0],  # Discovery has confluence
                }
            }

            validation_data = {
                "session_001": {"confluence": [60.0, 65.0]}  # Validation also has confluence
            }

            discovery_file = temp_path / "discovery.json"
            validation_file = temp_path / "validation.json"

            discovery_file.write_text(json.dumps(discovery_data))
            validation_file.write_text(json.dumps(validation_data))

            result = build_motifs_input(discovery_file, validation_file)

            # Should use discovery confluence, not validation
            assert result["session_001"]["confluence"] == [80.0, 85.0]


class TestMain:
    """Test main CLI function."""

    def test_main_discovery_only(self):
        """Test main function with discovery file only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            discovery_data = {
                "session_001": {"events": [{"type": "sweep", "minute": 10, "htf_under_mid": True}]}
            }

            discovery_file = temp_path / "discovery.json"
            output_file = temp_path / "output.json"

            discovery_file.write_text(json.dumps(discovery_data))

            result = main(["--discovery-json", str(discovery_file), "--out", str(output_file)])

            assert result == 0
            assert output_file.exists()

            # Check output content
            output_data = json.loads(output_file.read_text())
            assert "session_001" in output_data
            assert len(output_data["session_001"]["events"]) == 1

    def test_main_with_validation(self):
        """Test main function with validation file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            discovery_data = {"session_001": {"events": []}}
            validation_data = {"session_001": {"confluence": [70.0]}}

            discovery_file = temp_path / "discovery.json"
            validation_file = temp_path / "validation.json"
            output_file = temp_path / "output.json"

            discovery_file.write_text(json.dumps(discovery_data))
            validation_file.write_text(json.dumps(validation_data))

            result = main(
                [
                    "--discovery-json",
                    str(discovery_file),
                    "--validation-json",
                    str(validation_file),
                    "--out",
                    str(output_file),
                ]
            )

            assert result == 0
            assert output_file.exists()

            output_data = json.loads(output_file.read_text())
            assert output_data["session_001"]["confluence"] == [70.0]

    def test_main_creates_output_directory(self):
        """Test that main creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            discovery_data = {"session_001": {"events": []}}
            discovery_file = temp_path / "discovery.json"
            discovery_file.write_text(json.dumps(discovery_data))

            # Output in non-existent subdirectory
            output_subdir = temp_path / "subdir" / "nested"
            output_file = output_subdir / "output.json"

            result = main(["--discovery-json", str(discovery_file), "--out", str(output_file)])

            assert result == 0
            assert output_file.exists()
            assert output_subdir.exists()

    def test_main_missing_discovery_file(self):
        """Test main function with missing discovery file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            nonexistent_file = temp_path / "nonexistent.json"
            output_file = temp_path / "output.json"

            with pytest.raises(FileNotFoundError):
                main(["--discovery-json", str(nonexistent_file), "--out", str(output_file)])

    def test_main_missing_validation_file(self):
        """Test main function with missing validation file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            discovery_data = {"session_001": {"events": []}}
            discovery_file = temp_path / "discovery.json"
            discovery_file.write_text(json.dumps(discovery_data))

            nonexistent_validation = temp_path / "nonexistent_validation.json"
            output_file = temp_path / "output.json"

            with pytest.raises(FileNotFoundError):
                main(
                    [
                        "--discovery-json",
                        str(discovery_file),
                        "--validation-json",
                        str(nonexistent_validation),
                        "--out",
                        str(output_file),
                    ]
                )

    @patch("ironforge.scripts.prepare_motifs_input.build_motifs_input")
    def test_main_argparse_integration(self, mock_build):
        """Test argparse integration."""
        mock_build.return_value = {"test": {"events": [], "confluence": []}}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_file = temp_path / "output.json"

            result = main(
                [
                    "--discovery-json",
                    "/fake/discovery.json",
                    "--validation-json",
                    "/fake/validation.json",
                    "--out",
                    str(output_file),
                ]
            )

            assert result == 0
            mock_build.assert_called_once_with(
                Path("/fake/discovery.json"), Path("/fake/validation.json")
            )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_malformed_json(self):
        """Test with malformed JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            malformed_file = temp_path / "malformed.json"
            malformed_file.write_text("{ invalid json")

            output_file = temp_path / "output.json"

            with pytest.raises(json.JSONDecodeError):
                main(["--discovery-json", str(malformed_file), "--out", str(output_file)])

    def test_empty_json_files(self):
        """Test with empty JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            discovery_file = temp_path / "empty_discovery.json"
            discovery_file.write_text("{}")

            output_file = temp_path / "output.json"

            result = main(["--discovery-json", str(discovery_file), "--out", str(output_file)])

            assert result == 0
            output_data = json.loads(output_file.read_text())
            assert output_data == {}

    def test_large_data_volumes(self):
        """Test with large data volumes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create discovery with many sessions
            discovery_data = {}
            for i in range(100):
                discovery_data[f"session_{i:03d}"] = {
                    "events": [
                        {"type": "sweep", "minute": j, "htf_under_mid": j % 2 == 0}
                        for j in range(10)
                    ],
                    "confluence": [float(60 + (i % 40)) for _ in range(60)],
                }

            discovery_file = temp_path / "large_discovery.json"
            discovery_file.write_text(json.dumps(discovery_data))

            output_file = temp_path / "output.json"

            result = main(["--discovery-json", str(discovery_file), "--out", str(output_file)])

            assert result == 0
            output_data = json.loads(output_file.read_text())
            assert len(output_data) == 100

            # Verify structure is preserved
            for session_id, session_data in output_data.items():
                assert "events" in session_data
                assert "confluence" in session_data
                assert len(session_data["events"]) == 10
                assert len(session_data["confluence"]) == 60
