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

## Engine Architecture

- **Temporal Engine**: `ironforge.temporal_engine` → TGAT discovery
- **Semantic Engine**: `ironforge.semantic_engine` → Confluence + motifs
- **Validation Engine**: `ironforge.validation_engine` → Quality rails (disabled v0.7.x)

## Current Limitations

- **Cross-session links**: Not implemented (reserved for Wave 8)
- **HTF predictive stack**: HTF used as context only, not parallel prediction
- **Full TGAT spine**: Temporal discovery partially implemented (missing components)
- **Validation rails**: Disabled in v0.7.x, will fail-fast with clear message