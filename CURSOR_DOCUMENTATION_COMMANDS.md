## IRONFORGE Documentation Overhaul — Commands and Validation

Use these commands to audit, standardize, and validate the documentation overhaul. Prefer running them from the repo root (`/workspace`). Adjust paths if your environment differs.


### 1) Discovery and Inventory

List markdown files (excluding node_modules):
```bash
find . -name "*.md" -not -path "*/node_modules/*" | sort | wc -l
find . -name "*.md" -not -path "*/node_modules/*" | sort
```

List Python modules within `ironforge/`:
```bash
find ironforge -type f -name "*.py" | sort | wc -l
find ironforge -type f -name "*.py" | sort
```


### 2) Version Reference Audit

Flag outdated references (v0.*) outside of archives:
```bash
grep -r "v0\." . --include="*.md" | grep -v node_modules | sed -n '1,200p'
```

Confirm current version in code:
```bash
sed -n '1,30p' ironforge/__version__.py | cat
```


### 3) Golden Invariant Verification

Check that docs state the invariants consistently:
```bash
grep -r -n -i "6 events\|6\s*events" docs | sed -n '1,120p'
grep -r -n -i "4 edge\|4\s*edge" docs | sed -n '1,120p'
grep -r -n -i "51D\|51\s*D" docs | sed -n '1,120p'
grep -r -n -i "20D\|20\s*D" docs | sed -n '1,120p'
```

Expected canonical phrasing for live docs:
```
Events: 6; Intents: 4; Node features: 45D (default) / 51D (HTF ON); Edge features: 20; HTF sampling: last‑closed; Session isolation enforced.
```


### 4) Public API and CLI Validation

Python public API import (requires deps installed):
```bash
python3 -c "from ironforge.api import run_discovery, score_confluence, validate_run, build_minidash; print('✅ API imports work')"
```

CLI help surfaces (non‑interactive; exit code 0 expected):
```bash
python3 -m ironforge.sdk.cli --help | head -n 50 | cat
python3 -m ironforge.sdk.cli discover-temporal --help | head -n 50 | cat
python3 -m ironforge.sdk.cli score-session --help | head -n 50 | cat
python3 -m ironforge.sdk.cli validate-run --help | head -n 50 | cat
python3 -m ironforge.sdk.cli report-minimal --help | head -n 50 | cat
```


### 5) Link and Reference Hygiene

Find potentially broken intra‑repo references (heuristic):
```bash
grep -R "(docs/\|ironforge/\|tests/)" --include "*.md" | sed -n '1,200p'
```

Spot invalid anchors (common issue):
```bash
grep -R "[#][a-z0-9_-]\+" --include "*.md" docs | sed -n '1,120p'
```


### 6) Style Consistency

Check for fenced code blocks missing language tags (heuristic):
```bash
grep -R "^```$" --include "*.md" docs | sed -n '1,80p'
```

Search for deep imports to replace with `ironforge.api`:
```bash
grep -R "from ironforge\..* import" --include "*.md" docs | grep -v "from ironforge.api" | sed -n '1,120p'
```


### 7) Optional: Dependency Installation for Checks

Install project (editable) with dev extras:
```bash
pip3 install -e .[dev]
```

If full installation is heavy in your environment, install only minimal dependencies for import checks:
```bash
pip3 install numpy pandas matplotlib pyarrow
```


### 8) Final QA Checklist

- No `v0.*` references in live (non‑archive) docs
- `ironforge.api` used in all Python examples in live docs
- CLI sections present for: discover‑temporal, score‑session, validate‑run, report‑minimal
- Golden invariants phrased consistently and present in primary docs
- Quickstart and Troubleshooting enable a successful first run
- Release/migration notes cross‑linked and accurate for v1.0 → v1.1

