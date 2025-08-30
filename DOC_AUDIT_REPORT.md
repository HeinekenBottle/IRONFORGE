## IRONFORGE Documentation Audit Report (Phase 1)

Date: 2025-08-30

### Inventory

- Markdown files: 237
- Python modules under `ironforge/`: 90

### Version References

- v0.* references found primarily in historical release notes and archive folders. No live (root) docs depend on v0.* content. Action: leave historical references in archives; ensure live docs avoid v0.*.

### Golden Invariants Coverage

- Invariants present across live docs: Events=6; Intents=4; Nodes=45D/51D; Edges=20; HTF last‑closed; Session isolation. Action: ensure invariants are restated in primary root docs and API reference.

### Public Surfaces

- CLI documentation coverage found across live docs for: `discover-temporal`, `score-session`, `validate-run`, `report-minimal` (+ `status`, `prep-shards`). Action: keep authoritative CLI examples in root docs and cross‑link specialized guides.

### Import Patterns

- Root docs generally use `from ironforge.api import ...`.
- Specialized/archived docs contain many deep import examples. Action: standardize root docs first (Phase 2). Defer specialized doc conversions to later phases; keep archives intact with historical context labels.

### Code Block Language Tags

- Unlabeled code fences detected across many files (esp. archives/specialized). Action: enforce language tags in root docs now; sweep specialized docs later.

### API Signature Drift (Docs vs Code)

- Current code (centralized in `ironforge.api`) re‑exports functions whose actual signatures differ from some documented examples:
  - `run_discovery(shard_paths: Iterable[str], out_dir: str, loader_cfg: LoaderCfg) -> list[str]`
  - `score_confluence(pattern_paths: Sequence[str], out_dir: str, _weights: Mapping[str, float] | None, threshold: float, hierarchical_config: dict | None = None) -> str`
  - `validate_run(config) -> dict`
  - `build_minidash(activity: pd.DataFrame, confluence: pd.DataFrame, motifs: list[dict], out_html, out_png, width=1200, height=700, htf_regime_data=None) -> (Path, Path)`
- Live API docs currently portray simplified `config`-only signatures for all engines. Action: update root API docs to reflect actual signatures and provide correct examples (Phase 2/3).

### Cross‑Platform Notes

- Windows PowerShell/Terminal alternatives are not consistently documented for commands like opening reports. Action: add cross‑platform notes in root docs (Phase 2).

### Recommended Next Steps (Phase 2 scope)

1) Standardize root docs (`docs/README.md`, `01–08`) to:
   - Fix API signatures and examples to match code
   - Reiterate invariants in README and API reference
   - Add Windows/macOS/Linux command variants where appropriate
   - Ensure all code fences have language tags
2) Defer specialized docs and archives to Phases 4–5 with consolidation and preservation guidelines.

