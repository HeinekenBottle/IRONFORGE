# IRONFORGE 1.0 Snapshot — 2025-08-19
## Public Surfaces
CLI: discover-temporal, score-session, validate-run, report-minimal, status, prep-shards (--htf-context)
Python: learning.discovery_pipeline:run_discovery; confluence.scoring:score_confluence; validation.runner:validate_run; reporting.minidash:build_minidash
## Invariants
Events: 6; Intents: 4; Nodes: 51D (HTF on) / 45D (off); Edges: 20; HTF: last-closed only; Session isolation
## Decisions
HTF default OFF in 1.0; flip ON in 1.1 (+ --no-htf-context); Deprecations warn in 1.0, remove in 2.0
## Status
M1, M2 complete; PR #30 (CI/tooling + RELEASING.md) open; release-please configured
## CI Gates
Ruff/Black/mypy/pytest; contracts (dims, taxonomy, intents, runs layout); build+twine; smoke install
## Next
M4: RC hardening; M5: 1.0.0 GA

