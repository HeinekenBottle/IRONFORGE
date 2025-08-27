"""
IRONFORGE Engines Facade

Provides a clean mapping between CLI commands and engine functions for MCP discovery.
This module enables easy lookup of engine functions by command name and provides
a stable interface for programmatic access to all IRONFORGE engines.
"""
from __future__ import annotations

# Import engines from centralized API
from ..api import build_minidash, run_discovery, score_confluence, validate_run

# Command mapping for CLI and MCP tools
COMMAND_MAP = {
    "discover-temporal": run_discovery,
    "score-session": score_confluence, 
    "validate-run": validate_run,
    "report-minimal": build_minidash,
}

# Engine exports
__all__ = [
    "run_discovery",
    "score_confluence", 
    "validate_run",
    "build_minidash",
    "COMMAND_MAP",
]
