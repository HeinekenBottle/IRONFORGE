from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _coerce_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def _extract_events(session_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Accepts flexible shapes:
      - "events": [{"type": "...", "minute": 12, "htf_under_mid": true}, ...]
      - OR split arrays: "event_types": [], "event_minutes": [], ...
    Returns normalized event dicts.
    """
    if "events" in session_payload and isinstance(session_payload["events"], list):
        out = []
        for e in session_payload["events"]:
            out.append(
                {
                    "type": str(e.get("type", "")).strip(),
                    "minute": int(e.get("minute", 0)),
                    "htf_under_mid": _coerce_bool(e.get("htf_under_mid", False)),
                }
            )
        return out
    # fallback: zipped arrays
    types = session_payload.get("event_types", [])
    mins = session_payload.get("event_minutes", [])
    htf = session_payload.get("event_htf_under_mid", [])
    out = []
    for i, t in enumerate(types):
        m = int(mins[i]) if i < len(mins) else 0
        h = _coerce_bool(htf[i]) if i < len(htf) else False
        out.append({"type": str(t).strip(), "minute": m, "htf_under_mid": h})
    return out


def build_motifs_input(
    discovery_json: Path,
    validation_json: Path | None = None,
) -> dict[str, Any]:
    disc = _load_json(discovery_json)
    val = _load_json(validation_json) if validation_json else {}
    out: dict[str, Any] = {}
    # Expect top-level keys as session ids in discovery JSON
    for sid, payload in disc.items():
        # confluence could be already in discovery; else try validation
        conf = payload.get("confluence")
        if conf is None and sid in val:
            conf = val[sid].get("confluence")
        events = _extract_events(payload)
        out[sid] = {
            "events": events,
            "confluence": conf if conf is not None else [],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser("prepare-motifs-input")
    p.add_argument("--discovery-json", type=Path, required=True)
    p.add_argument("--validation-json", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv or sys.argv[1:])
    data = build_motifs_input(args.discovery_json, args.validation_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2))
    print(f"Wrote motifs input → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
