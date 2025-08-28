# Releasing IRONFORGE

## Versioning & Tags
- Source of truth: `ironforge/__version__.py`.
- Tags: `vMAJOR.MINOR.PATCH` (GA), `vMAJOR.MINOR.PATCH-rc.N` (RC).

## Release-please (auto)
- On merge to `main`, release-please opens/updates a Release PR (changelog + version bump).
- Merge that PR to create GitHub Release + tag; wheels/sdist are built/checked in CI.

## RC Flow
1. Branch: `release/1.0.0-rc.1`
2. Bump (optional manual): `python scripts/bump_version.py 1.0.0-rc.1 && git push && git push --tags`
3. CI: contracts + perf rails must be green.
4. Announce RC in CHANGELOG; collect issues under milestone `1.0`.

## GA Flow
- Merge final fixes to `main`.
- Let release-please PR update to `1.0.0`; merge it → tag + release.
- Verify smoke install from artifacts and run `status` + `discover-temporal` quickstart on a sample shard.

## Hotfixes
- Branch from tag (e.g., `v1.0.0`): `hotfix/1.0.1`
- Bump: `scripts/bump_version.py 1.0.1`; merge → release-please will reconcile.

## HTF Toggle Policy (1.0 → 1.1)
- 1.0 default: HTF OFF → 45D nodes. Enable via:
  - CLI: `--htf-context`
  - Config: `features.htf_context: true`
- 1.1 default flip: HTF ON → 51D nodes.
  - Escape hatch: CLI `--no-htf-context`, Config `features.htf_context: false`
- HTF features sampled at last-closed only; no schema breaks (additive fields).
- Deprecations: legacy CLI/import fallbacks warn in 1.0, removed in 2.0.

## Quality Gates (must pass)
- Contracts: taxonomy (6 events), intents (4), node dims (45/51 by toggle), edges=20, `runs/` layout & manifest.
- Lint/type/tests: Ruff/Black/mypy/pytest; packaging (`build` + `twine check`).
- Repro: seed & manifest present; deterministic unit tests.

## Publishing (optional)
- If publishing to PyPI: add a `publish.yml` workflow gated on tags `v*` with Twine credentials.

## Invariants enforced by CI
- Taxonomy v1 (6), intents (4), 51D/20D when HTF enabled, HTF last-closed, session isolation, stable `runs/YYYY-MM-DD/` layout + manifest.

## Toggle surface (user-facing)
- CLI:
  - `discover-temporal ...` → 45D (default)
  - `discover-temporal ... --htf-context` → 51D
- Config: `features.htf_context: false` (1.0 default) → flip true in 1.1; document `--no-htf-context`.

