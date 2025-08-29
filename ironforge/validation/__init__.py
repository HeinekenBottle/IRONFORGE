"""
Unified validation API scaffold.

Exports current runner and provides a placeholder for future
contract/config/session validators under a consolidated namespace.
"""

from .runner import validate_run

__all__ = ["validate_run"]
