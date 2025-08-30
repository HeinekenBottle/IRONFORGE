from __future__ import annotations

from agents.minidash_enhancer.agent import MinidashEnhancer


def test_generate_and_export() -> None:
    m = MinidashEnhancer()
    out = m.generate_enhanced_dashboard({"zones": []})
    assert "html" in out
    path = m.export_interactive_dashboard(out)
    assert path.endswith(".png")
