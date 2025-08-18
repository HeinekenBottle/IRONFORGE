# IRONFORGE Context (v0.7.1)

**Purpose:** Quick snapshot so agents (and humans) can operate without reloading the entire history.

## What exists (Waves 0–7)
- W0–3: Data → shards → TGAT discovery (temporal graphs from Parquet nodes/edges).
- W4–6: Validation rails, confluence scoring (0–100), motifs, minidash.
- W7: SDK/CLI orchestration + minimal reporting.
- v0.7.1: M5 base with HTF context v1.1 (adds f45–f50 features).

## Canonical Entrypoints
- Discovery: `ironforge.learning.discovery_pipeline:run_discovery`
- Confluence: `ironforge.confluence.scoring:score_confluence`
- Validation: `ironforge.validation.runner:validate_run`

## CLI (top-level)
- `discover-temporal`, `score-session`, `validate-run`, `report-minimal`, `status`
- Example (defaults NQ/M5):
  ```bash
  python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
  python -m ironforge.sdk.cli score-session     --config configs/dev.yml
  python -m ironforge.sdk.cli validate-run      --config configs/dev.yml
  python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml
  ```

## Contracts & Defaults
- Shards in: `data/shards/<SYMBOL_TF>/shard_*/{nodes.parquet,edges.parquet}`
- Run dir out: `runs/YYYY-MM-DD/{embeddings,patterns,confluence,motifs,reports,minidash.*}`
- Defaults: `symbol=NQ`, `timeframe=M5`, `shards_glob=data/shards/NQ_M5/shard_*`, `confluence.threshold=65`
- Nodes dims: 51 (45D base + HTF v1.1); Edges dims: 20

### HTF v1.1 (context, not parallel streams)
- Rule: last closed bar only (no leakage). Timeframes used: M15, H1.
- New node features:
  - f45_sv_m15_z – Synthetic Volume (M15) z-score (lookback 30 M15 bars)
  - f46_sv_h1_z – Synthetic Volume (H1) z-score (lookback 30 H1 bars)
  - f47_barpos_m15 – position in M15 bar [0,1]
  - f48_barpos_h1 – position in H1 bar [0,1]
  - f49_dist_daily_mid – normalized distance to previous day midpoint
  - f50_htf_regime – {0=consolidation,1=transition,2=expansion}
- Edges unchanged. Evidence chain stays per-session.

## Operate Daily
```bash
ironforge prep-shards --source-glob "data/enhanced/enhanced_*_Lvl-1_*.json"
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
python -m ironforge.sdk.cli score-session     --config configs/dev.yml
python -m ironforge.sdk.cli validate-run      --config configs/dev.yml
python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml
```

## Invariants
- One symbol/TF per run dir (no mixing).
- UTC ms timestamps; past-only context; degree not inflated by HTF.
- Run dir is the audit trail.
