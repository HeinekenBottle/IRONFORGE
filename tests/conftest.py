"""
Shared test fixtures for IRONFORGE test suite.
"""

import pytest
from pathlib import Path

from ironforge.sdk.app_config import load_config


@pytest.fixture
def test_config():
    """Get test configuration from configs/dev.yml."""
    config_path = Path("configs/dev.yml")
    if not config_path.exists():
        pytest.skip("Test configuration not available")
    
    return load_config(str(config_path))
