# Repository Guidelines

## Project Structure & Module Organization
- Source: `ironforge/` (primary package), shared infra in `iron_core/`.
- Tests: `tests/` with `unit/`, `integration/`, and performance suites under `performance/`.
- Scripts & data: `scripts/`, `data/`, and example assets in `examples/` and `docs/`.
- Configs: `pyproject.toml`, `.pre-commit-config.yaml`, `requirements*.txt`, run entry points like `orchestrator.py` and `run_*.py`.

## Build, Test, and Development Commands
- `make setup`: Create a dev env, install `.[dev]`, install pre-commit.
- `make fmt`: Format code with Black.
- `make lint`: Lint with Ruff (auto-fix disabled here).
- `make type`: Type-check with mypy (focuses on `ironforge/`).
- `make test`: Run pytest (`tests/` per `pyproject.toml`).
- `make precommit`: Run all hooks locally (Black, Ruff, mypy, Bandit).

## Coding Style & Naming Conventions
- Formatting: Black, line length 100 (`pyproject.toml`).
- Linting: Ruff rules `E,F,I,UP,B,SIM,C4,ARG` (E501 ignored; Black governs wrapping).
- Types: Prefer typed functions; mypy enforces `disallow_untyped_defs = true`.
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants; tests as `test_*.py`.

## Testing Guidelines
- Framework: pytest; tests live under `tests/` (default path in config).
- Layout: unit tests in `tests/unit/`; integration in `tests/integration/` (excluded from some checks); golden/perf suites as labeled.
- Conventions: Name tests `test_<unit>_<behavior>.py::test_<case>()`.
- Run: `pytest -q` or `make test`. Add tests for all new behavior.

## Commit & Pull Request Guidelines
- Commits: Follow Conventional Commits seen in history (e.g., `feat: ...`, `fix: ...`, `style: ...`).
- Scope: Keep changes focused; include why in the body if not obvious.
- PRs: Provide description, linked issues, and test evidence (logs or artifacts). Ensure CI and pre-commit pass.
- CI/Reviews: GitHub Actions run lint/tests and Claude review on PRs; address findings before merge.

## Security & Configuration Tips
- Secrets: Do not commit credentials; prefer env vars or `configs/` templates.
- Static analysis: Bandit runs via pre-commit; fix or justify findings.
- Repo guardrails: `.githooks` verifies `ironforge/` exists; keep package structure intact.
