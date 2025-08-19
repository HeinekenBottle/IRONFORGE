# IRONFORGE Discovery Pipeline Flow

## Core Archaeological Discovery Flow

```
Enhanced Events (E/C/R/R + Liquidity + Redelivery)
    ↓ [ironforge.converters.json_to_parquet]
Nodes(kind 0-5) / Edges(etype 0-3) / Features(51D HTF)
    ↓ [data/shards/NQ_M5/*.parquet]
Shards
    ↓ [ironforge.temporal_engine.run_discovery (TGAT)]
Patterns/Embeddings
    ↓ [ironforge.semantic_engine.score_confluence (HTF f45-f50)]
Confluence Scores 0-100 + Archaeological Motifs
    ↓ [ironforge.validation_engine.validate_run (gated)]
reports/validation.json
    ↓ [ironforge.reporting.minidash]
HTF-Enhanced Minidash (visual)
```

## Event Taxonomy (v1.0)

**Six Canonical Types → Node.kind mapping:**
- `0: EXPANSION` - Directional movement with momentum
- `1: CONSOLIDATION` - Sideways action within range boundaries  
- `2: RETRACEMENT` - Counter-trend pullback within bias
- `3: REVERSAL` - Direction change, bias invalidation
- `4: LIQUIDITY_TAKEN` - Price sweep through liquidity zones
- `5: REDELIVERY` - Return to FVG/imbalances

**Four Edge Types → Edge.etype mapping:**
- `0: TEMPORAL_NEXT` - Sequential time relationship
- `1: MOVEMENT_TRANSITION` - Price movement causality
- `2: LIQ_LINK` - Liquidity-based connection  
- `3: CONTEXT` - Archaeological relationship

## Schema Contract

**Schema contract:** TGAT consumes 51 node features (f0..f50) and 20 edge features (e0..e19). Parquet files may include extra metadata columns (e.g., ids, timestamps, event_kind). Total columns can exceed feature dims without violating the contract.

- **Total node columns**: 58 (51 features + 7 metadata)
- **Total edge columns**: 24 (20 features + 4 metadata)

## HTF Context Integration (v0.7.1)

**51D Node Features:**
- **f0-f44**: Base archaeological features (45D)
- **f45-f50**: HTF context features (6D)
  - f45: M15 Synthetic Volume z-score
  - f46: H1 Synthetic Volume z-score  
  - f47: M15 Bar Position [0,1]
  - f48: H1 Bar Position [0,1]
  - f49: Distance to Daily Midpoint
  - f50: HTF Regime (0=consolidation, 1=transition, 2=expansion)

## CLI Commands

```bash
# Complete discovery pipeline
ironforge prep-shards --htf-context          # Enhanced JSON → 51D shards
ironforge discover-temporal                   # TGAT pattern discovery  
ironforge score-session                       # Confluence scoring + motifs
ironforge validate-run                        # Quality validation (gated)
ironforge report-minimal                      # HTF-enhanced minidash
ironforge status --runs runs                  # Pipeline status
```

**All 5 CLI subcommands**: `discover-temporal`, `score-session`, `validate-run`, `report-minimal`, `status`

## Engine Architecture

- **Temporal Engine**: `ironforge.temporal_engine` → TGAT discovery
- **Semantic Engine**: `ironforge.semantic_engine` → Confluence + motifs
- **Validation Engine**: `ironforge.validation_engine` → Quality rails (disabled v0.7.x)

## Daily Summary (no code changes)

Daily run summary extractor for post-pipeline analysis:

```python
import os, glob, json, pandas as pd
def rd(run): 
    for p in ["confluence/scores.parquet","confluence/zones.parquet","confluence/scores.jsonl"]:
        p=os.path.join(run,p); 
        if os.path.exists(p): return (p.endswith(".parquet") and pd.read_parquet(p)) or pd.read_json(p,lines=True)
    return None
runs = sorted([p for p in glob.glob("runs/*") if os.path.isdir(p)])
assert runs, "No runs/* found."
run = runs[-1]
zones = rd(run)
zones_total   = 0 if zones is None else int(len(zones))
mean_conf     = None if zones is None or "confidence" not in zones else round(float(zones["confidence"].mean()),3)
theoryB_zones = 0 if zones is None else next((int(zones[c].astype(str).str.contains("theoryB",case=False,na=False).sum()) for c in ["motif","label","pattern","tag","kind"] if c in zones), 0)
edges = None
for cand in ["embeddings/graph_edges.parquet","patterns/edges.parquet","embeddings/edges.parquet"]:
    p=os.path.join(run,cand)
    if os.path.exists(p): edges = pd.read_parquet(p); break
avg_out_deg = None if edges is None else round(float(edges.groupby([c for c in ["src","source","u","from"] if c in edges.columns][0]).size().mean()),3)
meta=None
for m in ["status.json","minidash.meta.json","manifest.json"]:
    p=os.path.join(run,m)
    if os.path.exists(p): meta=json.load(open(p)); break
stage_runtime_s=None
if isinstance(meta,dict):
    for key in ["stage_runtime_s","runtime_s"]:
        if key in meta: 
            try: stage_runtime_s=float(meta[key]); break
            except: pass
    if stage_runtime_s is None:
        for k in ["durations","timings"]:
            if k in meta and isinstance(meta[k],dict):
                try: stage_runtime_s=float(sum(map(float, meta[k].values()))); break
                except: pass
print(json.dumps({
  "run_dir": run,
  "zones_total": zones_total,
  "theoryB_zones": theoryB_zones,
  "mean_confidence": mean_conf,
  "avg_out_degree": avg_out_deg,
  "stage_runtime_s": None if stage_runtime_s is None else round(stage_runtime_s,2)
}, indent=2))
```

## Current Limitations

- **Cross-session links**: Not implemented (reserved for Wave 8)
- **HTF predictive stack**: HTF used as context only, not parallel prediction
- **Full TGAT spine**: Temporal discovery partially implemented (missing components)
- **Validation rails**: Disabled in v0.7.x, will fail-fast with clear message