# Reporting (Wave 7 — Minimal Dashboard)

Wave 7 provides a minimal HTML/PNG dashboard (minidash) generated from run outputs.

## Quickstart
```bash
python -m ironforge.sdk.cli report-minimal --config configs/dev.yml
open runs/$(date +%F)/minidash.html
```

## Inputs
- Searches the current run dir (`runs/YYYY-MM-DD/`) for:
  - `confluence/*.parquet` — optional; falls back to synthetic strip if missing
  - `patterns/*.parquet` — optional; activity inferred from confluence timestamps
  - `motifs/*.json` — optional; falls back to a sample motif

## Outputs
- `runs/YYYY-MM-DD/minidash.html` — HTML dashboard
- `runs/YYYY-MM-DD/minidash.png` — PNG snapshot

## Dimensions
Configure output filenames and size in `configs/dev.yml` under `reporting.minidash`:
```yaml
reporting:
  minidash:
    out_html: "minidash.html"
    out_png:  "minidash.png"
    width: 1200
    height: 700
```

## Notes
- The command is idempotent and recreates files when `outputs.overwrite` is true.
- Missing inputs are handled gracefully with sensible defaults to avoid blocking workflows.

