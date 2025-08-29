# IRONFORGE API v1.1.0 - Stable Public Interface
"""
IRONFORGE Central Public API

This module provides a single, stable import surface optimized for MCP (Model Context Protocol)
compatibility and clean documentation examples. All public functions are re-exported here
with stable signatures and comprehensive type hints.

Preferred imports should come from here rather than deep package paths for:
- Better MCP tool discovery and documentation
- Stable API surface across versions
- Clean, short code examples
- Predictable import patterns

Contents:
- Engines: run_discovery, score_confluence, validate_run, build_minidash
- SDK: Configuration dataclasses, load_config, materialize_run_dir, I/O utilities
- Integration: Container and lazy loading utilities

Backwards compatibility: legacy module paths remain available.
"""
from __future__ import annotations

from ironforge.confluence.scoring import score_confluence

# BMAD Integration (temporarily disabled due to syntax issues)
# from ironforge.coordination.bmad_integration import (
#     get_bmad_integration,
#     coordinate_bmad_analysis,
#     initialize_bmad_integration,
#     shutdown_bmad_integration
# )

# Optional: integration exports for advanced users
from ironforge.integration.ironforge_container import (
    get_ironforge_container,
    initialize_ironforge_lazy_loading,
)

# Core entrypoints
from ironforge.learning.discovery_pipeline import run_discovery
from ironforge.reporting.minidash import build_minidash
from ironforge.sdk.app_config import (
    Config,
    DataCfg,
    MinidashCfg,
    OracleCfg,
    OutputsCfg,
    ReportingCfg,
    ScoringCfg,
    ValidationCfg,
    WeightsCfg,
    load_config,
    materialize_run_dir,
    validate_config,
)

# SDK helpers (thin, user-facing configuration + utilities)
from ironforge.sdk.config import (
    ConfluenceCfg,
    LoaderCfg,
    Paths,
    RunCfg,
    load_cfg,
)
from ironforge.sdk.io import glob_many, write_html, write_json
from ironforge.validation.runner import validate_run

__all__ = [
    # Entrypoints
    "run_discovery",
    "score_confluence",
    "validate_run",
    "build_minidash",
    # BMAD Integration (temporarily disabled)
    # "get_bmad_integration",
    # "coordinate_bmad_analysis", 
    # "initialize_bmad_integration",
    # "shutdown_bmad_integration",
    # SDK helpers
    "LoaderCfg",
    "Paths",
    "ConfluenceCfg",
    "RunCfg",
    "load_cfg",
    "Config",
    "DataCfg",
    "OutputsCfg",
    "WeightsCfg",
    "ScoringCfg",
    "MinidashCfg",
    "ReportingCfg",
    "ValidationCfg",
    "OracleCfg",
    "load_config",
    "materialize_run_dir",
    "validate_config",
    "write_json",
    "write_html",
    "glob_many",
    # Integration
    "get_ironforge_container",
    "initialize_ironforge_lazy_loading",
]

