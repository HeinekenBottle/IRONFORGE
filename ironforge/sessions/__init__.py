"""Unified sessions package.

Exports session manager and adapter interfaces and session-specific validators.
"""

from __future__ import annotations

from .manager import SessionManager
from .adapter import SessionAdapter
from .isolation import ensure_session_isolation

__all__ = [
    "SessionManager",
    "SessionAdapter",
    "ensure_session_isolation",
]

