import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import importlib.util

CLI_PATH = Path(__file__).resolve().parents[2] / "IRONFORGE" / "sdk" / "cli.py"
spec = importlib.util.spec_from_file_location("ironforge_sdk_cli", CLI_PATH)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_shards_dir_traversal_rejected(tmp_path, monkeypatch):
    """Ensure path traversal outside approved roots is rejected."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    # Restrict the allowed base directory to the "allowed" folder
    monkeypatch.setattr(cli, "APPROVED_SHARDS_BASE_DIRS", [allowed])

    # Construct a path that tries to escape the allowed directory
    escape_path = allowed.parent / "escape"

    with pytest.raises(ValueError):
        cli.main(["discover-temporal", "--data-path", str(escape_path)])
