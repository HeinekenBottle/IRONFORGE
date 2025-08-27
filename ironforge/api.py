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

# Core entrypoints
from ironforge.learning.discovery_pipeline import run_discovery
from ironforge.confluence.scoring import score_confluence
from ironforge.validation.runner import validate_run
from ironforge.reporting.minidash import build_minidash

# SDK helpers (thin, user-facing configuration + utilities)
from ironforge.sdk.config import (
    LoaderCfg,
    Paths,
    ConfluenceCfg,
    RunCfg,
    load_cfg,
)
from ironforge.sdk.app_config import (
    Config,
    DataCfg,
    OutputsCfg,
    WeightsCfg,
    ScoringCfg,
    MinidashCfg,
    ReportingCfg,
    ValidationCfg,
    OracleCfg,
    load_config,
    materialize_run_dir,
    validate_config,
)
from ironforge.sdk.io import write_json, write_html, glob_many

# Optional: integration exports for advanced users
from ironforge.integration.ironforge_container import (
    get_ironforge_container,
    initialize_ironforge_lazy_loading,
)

__all__ = [
    # Entrypoints
    "run_discovery",
    "score_confluence",
    "validate_run",
    "build_minidash",
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

