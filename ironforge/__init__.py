"""
IRONFORGE Archaeological Discovery System
Package version and main exports
"""

from .__version__ import __version__, __version_info__

# Import from centralized API for stable interface
from .api import (
    # Engines
    run_discovery,
    score_confluence,
    validate_run,
    build_minidash,
    # SDK helpers
    Config,
    load_config,
    materialize_run_dir,
    # Integration
    get_ironforge_container,
    initialize_ironforge_lazy_loading,
)

__all__ = [
    "__version__",
    "__version_info__",
    # Engines
    "run_discovery",
    "score_confluence",
    "validate_run",
    "build_minidash",
    # SDK helpers
    "Config",
    "load_config",
    "materialize_run_dir",
    # Integration
    "get_ironforge_container",
    "initialize_ironforge_lazy_loading",
]
