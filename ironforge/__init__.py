"""
IRONFORGE Archaeological Discovery System
Package version and main exports
"""

from .__version__ import __version__, __version_info__

# Lazy public API to avoid importing heavy dependencies on package import.
# Attributes are resolved on first access via __getattr__.

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


def __getattr__(name: str):
    # Mapping of public names to (module, attribute)
    mapping = {
        # Engines
        "run_discovery": ("ironforge.learning.discovery_pipeline", "run_discovery"),
        "score_confluence": ("ironforge.confluence.scoring", "score_confluence"),
        "validate_run": ("ironforge.validation.runner", "validate_run"),
        "build_minidash": ("ironforge.reporting.minidash", "build_minidash"),
        # SDK helpers
        "Config": ("ironforge.sdk.app_config", "Config"),
        "load_config": ("ironforge.sdk.app_config", "load_config"),
        "materialize_run_dir": ("ironforge.sdk.app_config", "materialize_run_dir"),
        # Integration
        "get_ironforge_container": ("ironforge.integration.ironforge_container", "get_ironforge_container"),
        "initialize_ironforge_lazy_loading": ("ironforge.integration.ironforge_container", "initialize_ironforge_lazy_loading"),
    }

    if name in mapping:
        mod_name, attr = mapping[name]
        module = __import__(mod_name, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(name)
