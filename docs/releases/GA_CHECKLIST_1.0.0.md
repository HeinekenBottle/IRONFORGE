# GA Release Checklist — IRONFORGE 1.0.0

- CI green on main (ruff/black/mypy/pytest/build/twine/smoke).
- Contracts hold: 6 events / 4 intents / edges=20 / nodes 45D(OFF)·51D(ON) / HTF=last-closed / session isolation.
- NQ 5m smoke (45D + 51D) manifests exist; validator OK.
- Merge release-please PR → tag v1.0.0; wheel/sdist smoke install prints matching `__version__`.
- Publish GA notes; open 1.0.x hotfix milestone & 1.1 “HTF default ON” issue.

Artifacts to link in notes:
- `runs/<DATE>/NQ_5m/manifest.json`
- `runs/<DATE>/NQ_5m_htf/manifest.json`
- `runs/<DATE>/report_45d/` and `runs/<DATE>/report_51d/`

