"""Session isolation helpers.

Provides minimal utilities to assert session boundaries for downstream code.
"""

from __future__ import annotations

from typing import Iterable


def ensure_session_isolation(session_ids: Iterable[str]) -> None:
    """Assert that an iterable of session IDs contain no cross-session markers.

    This is a placeholder for stricter isolation checks and can be extended to
    enforce policies like no cross-session edges, etc.
    """
    for sid in session_ids:
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError("Invalid session id in isolation check")

