"""Unified validation API surface for IRONFORGE.

This module consolidates validation entrypoints for external consumers while we
progressively refactor internal validators. Backwards-compatible re-exports are
provided for stable functionality.
"""

from __future__ import annotations

from typing import Any, Dict

from .runner import validate_run  # Primary pipeline validation


def validate_config(config: Any) -> None:
    """Thin re-export to SDK's validate_config for discoverability.

    Note: Prefer ironforge.api.validate_config for public usage. This re-export
    enables `from ironforge.validation import validate_config` during the
    consolidation period.
    """
    from ironforge.sdk.app_config import validate_config as _validate

    _validate(config)


__all__ = [
    "validate_run",
    "validate_config",
]

