from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotifStep:
    """One step in a motif sequence."""

    event_type: str  # e.g., "sweep", "fvg_redelivery", "expansion", "consolidation"
    within_minutes: tuple[int, int]  # allowed delta from previous step (inclusive)
    htf_under_mid: bool | None = None  # optional structural guardrail


@dataclass(frozen=True)
class MotifCard:
    id: str
    name: str
    steps: list[MotifStep]
    window_minutes: tuple[int, int]  # overall window (from first step to last)
    min_confluence: float = 65.0
    max_confluence: float = 100.0
    description: str = ""


def default_cards() -> list[MotifCard]:
    """Three thin, testable cards derived from research notes."""
    return [
        MotifCard(
            id="c1",
            name="Sweep → FVG redelivery under HTF midpoint",
            steps=[
                MotifStep("sweep", (0, 0)),
                MotifStep("fvg_redelivery", (12, 30), htf_under_mid=True),
            ],
            window_minutes=(12, 30),
            min_confluence=65.0,
            description="FPFVG redelivery 12–30m after sweep, under HTF midline.",
        ),
        MotifCard(
            id="c2",
            name="Expansion → Consolidation → Redelivery (NY-AM)",
            steps=[
                MotifStep("expansion", (0, 0)),
                MotifStep("consolidation", (5, 40)),
                MotifStep("redelivery", (10, 40)),
            ],
            window_minutes=(15, 80),
            min_confluence=70.0,
            description="AM expansion sequence with mid-session redelivery.",
        ),
        MotifCard(
            id="c3",
            name="First-presentation FVG after Open (Wk4 bias)",
            steps=[
                MotifStep("fpfvg", (0, 0)),
            ],
            window_minutes=(10, 25),
            min_confluence=65.0,
            description="First presentation FVG soon after open; weekly cycle bias optional.",
        ),
    ]
