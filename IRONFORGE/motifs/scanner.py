from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from .cards import MotifCard, default_cards


@dataclass(frozen=True)
class MotifMatch:
    session_id: str
    card_id: str
    score: float  # simple proxy: mean confluence over matched window
    window: tuple[int, int]  # (t0, t1) minutes
    steps_at: list[int]  # minute marks for each step


def _events_by_type(events: list[dict]) -> dict[str, list[int]]:
    """Group events by type → sorted minute list."""
    buckets: dict[str, list[int]] = {}
    for e in events:
        t = e.get("type")
        m = int(e.get("minute"))
        buckets.setdefault(t, []).append(m)
    for k in buckets:
        buckets[k].sort()
    return buckets


def _find_next(mins: list[int], prev_minute: int, delta_lo: int, delta_hi: int) -> int | None:
    """Find the first minute in mins that falls within [prev+lo, prev+hi]."""
    lo = prev_minute + delta_lo
    hi = prev_minute + delta_hi
    i = bisect.bisect_left(mins, lo)
    if i < len(mins) and mins[i] <= hi:
        return mins[i]
    return None


def scan_session_for_cards(
    session_id: str,
    events: list[dict],  # [{"type": "...", "minute": int, "htf_under_mid": bool, ...}, ...]
    confluence: np.ndarray | None,  # vector aligned to minute bins (0..100)
    cards: list[MotifCard] | None = None,
    min_confluence: float = 65.0,
) -> list[MotifMatch]:
    cards = cards or default_cards()
    by_type = _events_by_type(events)
    matches: list[MotifMatch] = []

    for card in cards:
        # start at each candidate for first step
        first_type = card.steps[0].event_type
        first_minutes = by_type.get(first_type, [])
        for t0 in first_minutes:
            prev = t0
            steps_at = [t0]
            ok = True
            for step in card.steps[1:]:
                candidates = by_type.get(step.event_type, [])
                nxt = _find_next(candidates, prev, step.within_minutes[0], step.within_minutes[1])
                if nxt is None:
                    ok = False
                    break
                # optional structural guardrail check
                if step.htf_under_mid is not None:
                    # find an event record at minute==nxt and ensure flag matches
                    valid = any(
                        e.get("minute") == nxt
                        and bool(e.get("htf_under_mid")) == step.htf_under_mid
                        for e in events
                    )
                    if not valid:
                        ok = False
                        break
                steps_at.append(nxt)
                prev = nxt
            if not ok:
                continue
            t1 = steps_at[-1]
            # overall window constraint - for single events, check if the event time is within the window
            if len(steps_at) == 1:
                # For single events, check if the event occurs within the allowed time window from start
                if not (card.window_minutes[0] <= t0 <= card.window_minutes[1]):
                    continue
            else:
                # For multi-step sequences, check the total duration
                if not (card.window_minutes[0] <= (t1 - t0) <= card.window_minutes[1]):
                    continue
            # confluence threshold
            if confluence is not None and len(confluence):
                lo, hi = min(t0, t1), max(t0, t1)
                segment = confluence[lo : hi + 1]
                mean_conf = float(np.nanmean(segment)) if segment.size else 0.0
                if (
                    mean_conf < max(min_confluence, card.min_confluence)
                    or mean_conf > card.max_confluence
                ):
                    continue
                score = mean_conf
            else:
                score = float(max(min_confluence, card.min_confluence))
            matches.append(MotifMatch(session_id, card.id, score, (t0, t1), steps_at))
    # sort by score desc, shorter windows first as tiebreaker
    matches.sort(key=lambda m: (-m.score, (m.window[1] - m.window[0])))
    return matches


# --- CLI glue (used by sdk/cli.py) -------------------------------------------------
def run_cli_scan(
    input_json_path, top_k: int = 3, min_confluence: float = 65.0, preset: str = "default"
):
    import json

    data = json.loads(input_json_path.read_text())
    all_matches: list[MotifMatch] = []
    for sid, payload in data.items():
        events = payload.get("events", [])  # list of {"type":..., "minute":..., ...}
        conf = payload.get("confluence", [])
        conf_arr = np.array(conf, dtype=float) if conf else None
        all_matches.extend(
            scan_session_for_cards(sid, events, conf_arr, cards=None, min_confluence=min_confluence)
        )
    all_matches.sort(key=lambda m: -m.score)
    # print top-k in a compact JSON line format
    out = [
        dict(
            session_id=m.session_id,
            card_id=m.card_id,
            score=round(m.score, 2),
            window=m.window,
            steps_at=m.steps_at,
        )
        for m in all_matches[:top_k]
    ]
    print(json.dumps(out, indent=2))
