"""Session adaptation façade.

Provides a stable interface over `EnhancedSessionAdapter` for feature-level
adaptation of enhanced sessions.
"""

from __future__ import annotations

from typing import Any


class SessionAdapter:
    """Wraps the analysis EnhancedSessionAdapter."""

    def __init__(self) -> None:
        from ironforge.analysis.enhanced_session_adapter import EnhancedSessionAdapter

        self._impl = EnhancedSessionAdapter()

    def adapt(self, session_data: dict[str, Any]) -> dict[str, Any]:
        return self._impl.adapt_enhanced_session(session_data)

    def stats(self) -> dict[str, Any]:
        return self._impl.get_adapter_stats()

