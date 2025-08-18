# Getting Started (Wave 7 — SDK/CLI)
Run the end-to-end pipeline and generate a minimal dashboard.

## Install
```bash
pip install -e .[dev]
```

## Run Pipeline
```bash
# Discover → Score → Validate → Report
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
python -m ironforge.sdk.cli score-session     --config configs/dev.yml
python -m ironforge.sdk.cli validate-run      --config configs/dev.yml
python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml

# Open dashboard
open runs/$(date +%F)/minidash.html

# Check artifacts
python -m ironforge.sdk.cli status --runs runs
```

## Configuration
- Default config: `configs/dev.yml` (symbol `NQ`, timeframe `M5`, shards at `data/shards/NQ_M5/shard_*`).
- Local override: `configs/run.local.yaml` — set `data.shards_glob` to your absolute shard path.

## Engines
- `discover-temporal`: `ironforge.learning.discovery_pipeline:run_discovery`
- `score-session`: `ironforge.confluence.scoring:score_confluence`
- `validate-run`: `ironforge.validation.runner:validate_run`

If an engine is missing, the CLI prints a clear error and exits code 2.

## Notes
- Outputs live in `runs/YYYY-MM-DD/` including `minidash.html` and `minidash.png`.
- Prefer the CLI; older module-level APIs are archived under `docs/archive/`.

