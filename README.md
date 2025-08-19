# IRONFORGE — Graph-Based Market Archaeology (v0.7.1, Waves 0–7.2)

**Archaeological discovery engine: rule-based → TGAT (single ML core) → rule-based**

IRONFORGE learns temporal patterns within sessions, scores confluence, and renders minidash reports. The system operates on enhanced session data through a canonical pipeline that preserves pattern authenticity while enabling rapid discovery.

## What It Is

Archaeological discovery engine that combines:
- **Rule-based preprocessing**: Enhanced session adapter with event detection
- **TGAT (single ML core)**: Temporal graph attention networks for pattern learning  
- **Rule-based scoring**: Confluence analysis and validation rails
- **Within-session learning**: No cross-session edges, preserves session boundaries

## Canonical Contracts

### Events (6 types exactly)
- **Expansion**: Market range extension
- **Consolidation**: Range compression  
- **Retracement**: Partial reversal within trend
- **Reversal**: Full directional change
- **Liquidity Taken**: Order flow absorption
- **Redelivery**: Return to prior levels

### Edge Intents (4 types exactly)  
- **TEMPORAL_NEXT**: Sequential time progression
- **MOVEMENT_TRANSITION**: Price movement relationships
- **LIQ_LINK**: Liquidity flow connections
- **CONTEXT**: Contextual relationships

### Feature Dimensions
- **Nodes**: 51D (f0..f50, HTF v1.1 = f45..f50 last-closed only)
- **Edges**: 20D (e0..e19)

### Data Contracts
- **Shards**: `data/shards/<SYMBOL_TF>/shard_*/{nodes,edges}.parquet`
- **Runs**: `runs/YYYY-MM-DD/{embeddings,patterns,confluence,motifs,reports,minidash.*}`

### Entrypoints + CLI
- **discovery** → `ironforge.learning.discovery_pipeline:run_discovery`
- **confluence** → `ironforge.confluence.scoring:score_confluence`  
- **validation** → `ironforge.validation.runner:validate_run`
- **reporting** → `ironforge.reporting.minidash:build_minidash`
- **CLI**: `discover-temporal`, `score-session`, `validate-run`, `report-minimal`, `status`

## Quickstart

```bash
# Install
pip install -e .[dev]

# Run canonical pipeline
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
python -m ironforge.sdk.cli score-session     --config configs/dev.yml
python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml
python -m ironforge.sdk.cli status            --runs runs

# Open dashboard
open runs/$(date +%F)/minidash.html
```

## Outputs (per run)

### Core Artifacts
- **embeddings/**: TGAT model outputs and attention weights
- **patterns/**: Discovered temporal patterns and sequences
- **confluence/scores.**: Confluence scoring results (0-1 or 0-100 scale)
- **confluence/stats.json**: Scale badge, health gates, summary statistics
- **minidash.html**: Interactive dashboard with confluence strips
- **minidash.png**: Static dashboard export

### AUX (read-only context)
- **aux/trajectories.parquet**: Session trajectory analysis
- **aux/phase_stats.json**: Phase transition statistics  
- **aux/chains.parquet**: Within-session chain analysis
- **motifs/candidates.csv**: Pattern motif candidates
- **motifs/cards/**: Individual motif cards and analysis

## AUX vs Canonical

**AUX is read-only context** for analysis and reporting. Key principles:
- No new labels or schema changes
- No feedback into feature generation
- Within-session analysis only
- No cross-session edges (Wave 8 later)

## Adapters (config-toggled, optional)

All adapters are **off by default** but documented:

- **confluence.phase_weighting**: HTF bucket weighting
- **confluence.chain_bonus**: Within-session chain scoring bonus
- **mt_burst_boost**: Momentum burst detection enhancement
- **mt_dt_bounds_s**: Temporal bounds for momentum analysis
- **liq_short_s**: Short-term liquidity analysis window

## Scale Handling

Automatic scale detection and normalization:
- **0-1 normalized**: Standard confluence scoring
- **0-100 normalized**: Percentage-based scoring  
- **Threshold normalized**: Adaptive threshold scaling
- **Variance watchdog**: Outlier detection and handling

## Explainability

### Attention Analysis
- **embeddings/attention_topk.parquet**: Top attention weights per edge
  - `edge_intent`: Edge type (TEMPORAL_NEXT, MOVEMENT_TRANSITION, etc.)
  - `weight`: Attention weight magnitude
  - `attn_rank`: Ranking within attention distribution
  - `true_dt_s`: Real time delta in seconds (when available)

### Dashboard Indicators
- **Δt: real seconds**: Actual time deltas available
- **Δt: rank proxy**: Using rank-based time approximation
- **Scale badge**: 0-1, 0-100, or threshold-normalized indicator
- **Health status**: System health and data quality gates

## Guardrails

**Golden Invariants** (never change):
- Event taxonomy: exactly 6 event types
- Edge intents: exactly 4 intent types  
- Feature dimensions: 51D nodes, 20D edges
- HTF rule: last-closed only (no intra-candle)
- Session boundaries: no cross-session edges
- Within-session learning: preserve session isolation

## Development

### Testing
```bash
# Run smoke tests
python tools/smoke_checks.py

# Full test suite
python -m pytest tests/ -v

# Legacy tests (moved during refactor)
python tests/legacy/simple_threshold_test.py
```

### Data Preparation
```bash
# Convert enhanced sessions to shards
python -m ironforge.sdk.cli prep-shards --htf-context

# Check shard dimensions
python -c "
import pyarrow.parquet as pq
nodes = pq.read_table('data/shards/NQ_M5/shard_*/nodes.parquet')
print(f'Node features: {len([c for c in nodes.column_names if c.startswith(\"f\")])}')"
```

## Documentation

- **docs/flows.md**: Schema contracts and run order
- **docs/taxonomy_v1.md**: Authoritative event and edge taxonomy  
- **docs/operations.md**: Daily operations and A/B adapter usage
- **REFACTOR_PLAN.md**: Repository cleanup and refactor details

---

**IRONFORGE v0.7.1** — Archaeological discovery through temporal graph attention
