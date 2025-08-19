# IRONFORGE <VERSION> (RC)

This RC validates the 1.0 public surface and pipeline invariants ahead of GA.

## Highlights

- Public API: CLI (`discover-temporal`, `score-session`, `validate-run`, `report-minimal`, `status`, `prep-shards --htf-context`) and Python (`run_discovery`, `score_confluence`, `validate_run`, `build_minidash`).
- HTF context features toggle (OFF by default in 1.0; ON by default in 1.1 with `--no-htf-context`).
- Invariants: 6 events, 4 intents, nodes 45D/51D, edges 20D, HTF last-closed, session isolation.
- CI/tooling: pre-commit hooks, Ruff, Black, mypy, Bandit; release-please.

## Breaking Changes

- None expected; legacy imports warn in 1.0, removed in 2.0.

## Migration

- 45D base path unchanged.
- Enable 51D via `ironforge prep-shards --htf-context`.

## Quickstart

```bash
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
python -m ironforge.sdk.cli score-session     --config configs/dev.yml
python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml
python -m ironforge.sdk.cli status            --runs runs
```

## Verification (RC)

- Tests green on tag, `make precommit` clean, minidash renders.

