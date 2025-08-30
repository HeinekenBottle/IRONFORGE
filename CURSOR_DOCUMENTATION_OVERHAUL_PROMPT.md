## IRONFORGE Documentation Overhaul — Cursor Agent Execution Prompt

This prompt equips Cursor agents to systematically transform IRONFORGE’s documentation into a consistent, professional, and production‑ready suite that reflects the v1.1.0 architecture, public API, golden invariants, and modern developer workflows.

Use this prompt together with the companion commands and validations in `CURSOR_DOCUMENTATION_COMMANDS.md`.


### Objectives

- Produce complete, accurate, and readable documentation for IRONFORGE v1.1.0
- Standardize structure, terminology, and formatting across all docs
- Ensure API references, CLI commands, and examples are correct and runnable
- Preserve and clearly communicate golden invariants and performance thresholds
- Eliminate outdated references, broken links, and contradictory guidance


### Scope and Context

- Codebase modules: approximately 89–90 Python files within `ironforge/**` (public API centralized in `ironforge.api`)
- Public CLI commands: `discover-temporal`, `score-session`, `validate-run`, `report-minimal`, plus `status`, `prep-shards` where relevant
- Golden invariants (must appear consistently in docs):
  - Events: exactly 6
  - Edge intents: exactly 4
  - Node features: 45D (default) / 51D (HTF ON)
  - Edge features: 20D
  - HTF sampling: last‑closed only
  - Session isolation: strict
- Versioning: current package `ironforge.__version__ = 1.1.0`
- Architecture: v1.1.0 preserves 1.0 contracts; HTF default behavior communicated; additive HTF features f45–f50


### Primary Documentation Targets

Prioritize these top‑level docs and ensure they are authoritative:

1) `docs/README.md` — Canonical overview and entrypoints
2) `docs/01-QUICKSTART.md` — Fast path to successful first run
3) `docs/02-USER-GUIDE.md` — End‑to‑end flows for users
4) `docs/03-API-REFERENCE.md` — Top‑level Python API with `ironforge.api` imports
5) `docs/04-ARCHITECTURE.md` — v1.1.0 architecture and invariants
6) `docs/05-DEPLOYMENT.md` — Install, environment, and runtime guidance
7) `docs/06-TROUBLESHOOTING.md` — Common issues and fixes
8) `docs/07-CHANGELOG.md` — Concise, consistent change notes referencing releases
9) `docs/08-GLOSSARY.md` — Shared terminology and definitions

Then harmonize and enhance specialized guides under `docs/specialized/**` and migration/release notes under `docs/migrations/**` and `docs/releases/**`. Archive or prune redundant files under `docs/archive/**` once replacements are in place.


### Quality Bar (Success Criteria)

- All public APIs documented from the centralized `ironforge.api` surface (no deep import examples)
- CLI usage documented with accurate flags and parameter defaults
- Golden invariants explicitly stated and consistent across docs
- Code blocks runnable or clearly marked as illustrative
- No lingering `v0.*` references except inside historical archives where explicitly labeled
- No broken links; external and intra‑repo references resolve
- Clear version references to v1.0.0 and v1.1.0 where relevant; avoid ambiguous phrasing
- Troubleshooting and deployment guides cover common environments and constraints
- Doc style consistent with `docs/DOCUMENTATION_STANDARDS.md`


### Six‑Phase Execution Plan

Follow these phases in order. Detailed command snippets and checks are in `CURSOR_DOCUMENTATION_COMMANDS.md`.

1) Audit
   - Inventory all markdown files and map the current structure
   - Identify outdated references (`v0.*`), broken or stale pages, duplicated content
   - Locate golden invariant statements; note gaps and inconsistencies
   - Catalog API and CLI examples; flag deep imports that should use `ironforge.api`

2) Standardize
   - Apply consistent headings, section ordering, and code‑block styles
   - Normalize terminology (e.g., “HTF context,” “golden invariants,” “session isolation”)
   - Update import patterns to `from ironforge.api import ...`
   - Ensure all CLI references match `ironforge.sdk.cli` behaviors and defaults

3) Update
   - Replace outdated content and version references; make v1.1.0 authoritative
   - Fix examples to align with current CLI/API signatures and contracts
   - Insert explicit invariant callouts where missing (6 events, 4 intents, 51D/20D, etc.)

4) Consolidate
   - Merge overlapping/duplicated documents into single sources of truth
   - Move superseded files into `docs/archive/**` with a clear deprecation note
   - Remove “redundant” archives once modern replacements are validated

5) Enhance
   - Add quickstart flows, end‑to‑end examples, and troubleshooting guidance
   - Add HTF ON/OFF scenarios (45D/51D) with clear toggles and outcomes
   - Provide minimal reporting examples and outputs (minidash)

6) Validate
   - Run the provided greps, link checks, and counts
   - Spot‑run CLI `--help` and dry paths where feasible
   - Confirm `ironforge.api` surface and examples are consistent and correct


### Technical Accuracy Requirements

- CLI commands to be consistently documented:
  - `discover-temporal`
  - `score-session`
  - `validate-run`
  - `report-minimal`
  - Include `status`, `prep-shards` where applicable
- Public Python API imports must come from `ironforge.api`:
  - `from ironforge.api import run_discovery, score_confluence, validate_run, build_minidash`
- Contracts and invariants must appear verbatim:
  - Events: 6; Intents: 4; Nodes: 45D (default) / 51D (HTF ON); Edges: 20; HTF last‑closed; Session isolation
- Examples should reflect v1.1.0 behavior and performance guidance:
  - Typical constraints: <3s per session, <180s full discovery (indicative guidelines as present in docs)


### Documentation Structure (Recommended)

Use the following standard sections where applicable:

- Title and short summary
- Prerequisites (versions, environment)
- Quickstart or TL;DR
- Concepts and architecture
- CLI usage and examples
- Python usage and examples (`ironforge.api` only)
- Data contracts and golden invariants
- Performance notes and caveats
- Troubleshooting and FAQ
- Links to related docs


### Style and Formatting Standards

Align with `docs/DOCUMENTATION_STANDARDS.md`. Additionally:

- Headings: use `##` and `###`; avoid `#` for top‑level in long documents
- Code fences: specify language (bash, python, json, yaml)
- Use backticks for file/function/class names, and markdown links for URLs
- Avoid ambiguous terms; define acronyms on first use
- Keep examples minimal yet runnable; mark pseudo‑code as such


### Required Updates by Area

- Top‑level README: reflect v1.1.0 architecture and entrypoints; show canonical CLI pipeline
- Quickstart: a one‑screen run‑through for 45D (default) and HTF ON (51D) flows
- API reference: centralize around `ironforge.api` and show importable call signatures
- Architecture: illustrate the discovery → confluence → validation → reporting pipeline; list invariants
- Deployment: installation variants (editable install, extras), system requirements, and environment notes
- Troubleshooting: dependency issues (e.g., PyTorch), environment mismatches, file paths, missing data
- Changelog/releases: reference `docs/releases/**`; avoid duplicating detail in multiple places
- Glossary: unify terminology used across specialized docs


### Known Risk Areas and Mitigations

- Heavy dependencies (e.g., `torch`, `torch-geometric`) may not be available in minimal CI images
  - Mitigation: mark examples that require full installation; prefer CLI `--help` for structure validation
- Legacy references (`v0.*`) linger in historical notes
  - Mitigation: keep in archives with explicit “historical reference only” labels; scrub from live docs
- Deep import examples cause drift when modules move
  - Mitigation: require all examples to import from `ironforge.api`


### Deliverables Checklist

- Updated primary docs (README, Quickstart, API, Architecture, Deployment, Troubleshooting, Changelog, Glossary)
- Standardized specialized guides with cross‑links to the primary docs
- Cleaned archive with deprecated docs clearly labeled
- Verified invariants and version references across all pages
- Validation report summarizing the checks and outcomes


### How to Execute in Cursor

1) Open this file and `CURSOR_DOCUMENTATION_COMMANDS.md`
2) Run the discovery commands to build an audit report
3) Apply the six‑phase plan; commit logical batches with clear messages
4) Re‑run validations and fix any failures
5) Produce a short summary (diff highlights + validation results)


### Acceptance Criteria

- All public surfaces are documented and consistent
- Golden invariants stated and enforced in docs
- No broken links; no stale v0.* references in live docs
- `ironforge.api` import patterns used everywhere in examples
- CLI sections match implemented command surfaces
- Quickstart enables a new developer to be productive within minutes


### Appendix: Canonical Examples

Python (public API):
```python
from ironforge.api import run_discovery, score_confluence, validate_run, build_minidash
```

CLI (pipeline):
```bash
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
python -m ironforge.sdk.cli score-session     --config configs/dev.yml
python -m ironforge.sdk.cli validate-run      --config configs/dev.yml
python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml
```

Golden invariants (verbatim for docs):
- Events: 6; Intents: 4; Node features: 45D (default) / 51D (HTF ON); Edge features: 20; HTF sampling: last‑closed; Session isolation enforced.

