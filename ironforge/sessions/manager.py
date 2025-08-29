"""Core session management facade.

Lightweight abstraction that wraps existing temporal `SessionDataManager` to
offer a stable façade for future refactors.
"""

from __future__ import annotations

from typing import Any


class SessionManager:
    """Stable façade around the temporal SessionDataManager."""

    def __init__(self, shard_dir: str = "data/shards/NQ_M5", adapted_dir: str = "data/adapted"):
        from ironforge.temporal.session_manager import SessionDataManager

        self._impl = SessionDataManager(shard_dir=shard_dir, adapted_dir=adapted_dir)

    def load_all(self) -> None:
        self._impl.load_all_sessions()

    def get_session(self, session_id: str):
        return self._impl.get_session_data(session_id)

    def get_metadata(self, session_id: str) -> dict[str, Any] | None:
        return self._impl.get_session_metadata(session_id)

    def get_stats(self, session_id: str) -> dict[str, float] | None:
        return self._impl.get_session_stats(session_id)

    def list(self) -> list[str]:
        return self._impl.list_sessions()

