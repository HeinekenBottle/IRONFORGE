from __future__ import annotations

from agents.session_boundary_guardian.agent import SessionBoundaryGuardian


def test_detect_cross_session_violations() -> None:
    g = SessionBoundaryGuardian()
    graph = {"edges": [{"from_session": "A", "to_session": "A"}, {"from_session": "A", "to_session": "B"}]}
    assert g.validate_session_isolation(graph) is False
