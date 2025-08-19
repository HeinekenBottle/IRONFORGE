# IRONFORGE Data Flows and Schema Contracts

## Pipeline Overview

```
Enhanced Sessions → Graph Builder → TGAT Discovery → Confluence Scoring → Validation → Reports
```

## Schema Contract Box

### Input Schema (Enhanced Sessions)
```json
{
  "session_name": "NY_AM_2025_08_05",
  "session_type": "NY_AM",
  "semantic_events": [...],           // 6 event types
  "session_liquidity_events": [...],  // Liquidity flow events
  "structural_events": [...],         // Market structure events
  "price_movements": [...],           // Price action events
  "timeframe_data": {...}             // Multi-timeframe context
}
```

### Shard Schema (Parquet)
```
data/shards/<SYMBOL_TF>/shard_<SESSION>/
├── nodes.parquet    # 51D features (f0..f50)
└── edges.parquet    # 20D features (e0..e19)
```

**Node Features (51D)**:
- f0..f44: Base features (price, volume, time, structure)
- f45..f50: HTF v1.1 features (last-closed only)

**Edge Features (20D)**:
- e0..e19: Edge relationship features

### Run Schema (Output)
```
runs/YYYY-MM-DD/
├── embeddings/
│   ├── attention_topk.parquet      # Attention analysis
│   └── node_embeddings.parquet     # TGAT embeddings
├── patterns/
│   └── discovered_patterns.parquet # Pattern sequences
├── confluence/
│   ├── scores.parquet              # Confluence scores
│   └── stats.json                  # Scale badge, health gates
├── motifs/
│   ├── candidates.csv              # Motif candidates
│   └── cards/                      # Individual motif cards
├── aux/                            # Read-only context
│   ├── trajectories.parquet
│   ├── phase_stats.json
│   └── chains.parquet
└── minidash.html                   # Dashboard
```

## Run Order

### 1. Discovery Phase
```bash
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
```

**Input**: Enhanced session JSON files
**Output**: `embeddings/`, `patterns/`
**Function**: TGAT-based temporal pattern discovery

### 2. Scoring Phase  
```bash
python -m ironforge.sdk.cli score-session --config configs/dev.yml
```

**Input**: Discovered patterns
**Output**: `confluence/scores.parquet`, `confluence/stats.json`
**Function**: Confluence analysis and scoring

### 3. Validation Phase
```bash
python -m ironforge.sdk.cli validate-run --config configs/dev.yml
```

**Input**: Scores and patterns
**Output**: `reports/validation.json`
**Function**: Quality gates and validation rails

### 4. Reporting Phase
```bash
python -m ironforge.sdk.cli report-minimal --config configs/dev.yml
```

**Input**: All run artifacts
**Output**: `minidash.html`, `minidash.png`
**Function**: Dashboard generation with AUX panels

## Data Flow Constraints

### Golden Invariants
1. **Event Types**: Exactly 6 (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
2. **Edge Intents**: Exactly 4 (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)
3. **Feature Dimensions**: 51D nodes, 20D edges
4. **Session Boundaries**: No cross-session edges
5. **HTF Rule**: Last-closed only (f45..f50)

### AUX Constraints
- **Read-only**: No feedback into feature generation
- **Within-session**: No cross-session analysis
- **No schema drift**: Cannot change feature dimensions
- **No new labels**: Cannot create new event types

## Scale Handling

### Automatic Detection
```python
# In confluence/stats.json
{
  "scale_type": "0-1" | "0-100" | "threshold",
  "max_score": 1.0 | 100.0 | <threshold>,
  "health_status": "healthy" | "warning" | "error"
}
```

### Scale Normalization
- **0-1**: Standard normalized scores
- **0-100**: Percentage-based scores  
- **Threshold**: Adaptive threshold-based normalization

## Attention Analysis Flow

### Input: TGAT Attention Weights
```python
# From TGAT model
attention_weights = model.get_attention_weights()
```

### Processing: Top-K Selection
```python
# Select top attention weights per edge type
topk_attention = select_topk_by_edge_intent(attention_weights, k=10)
```

### Output: Explainability Data
```parquet
# embeddings/attention_topk.parquet
edge_intent     | weight | attn_rank | true_dt_s
TEMPORAL_NEXT   | 0.85   | 1         | 300.0
MOVEMENT_TRANS  | 0.72   | 2         | null
LIQ_LINK        | 0.68   | 3         | 180.0
```

## Error Handling

### Missing Data
- **Empty patterns**: Generate synthetic baseline
- **Missing confluence**: Use default scoring
- **No attention data**: Fall back to rank proxy

### Scale Issues
- **Variance watchdog**: Detect outliers
- **Auto-normalization**: Apply appropriate scaling
- **Health gates**: Flag problematic runs

## Configuration Toggles

All adapters **off by default**:

```yaml
confluence:
  phase_weighting: false      # HTF bucket weighting
  chain_bonus: false          # Within-session chain bonus
  
momentum:
  mt_burst_boost: false       # Momentum burst enhancement
  mt_dt_bounds_s: null        # Temporal bounds
  
liquidity:
  liq_short_s: null           # Short-term analysis window
```

## Validation Gates

### Data Quality
- Shard dimensions: 51D/20D verified
- Event taxonomy: 6 types present
- Edge intents: 4 types present
- Session boundaries: No cross-session edges

### Output Quality  
- Confluence scores: Valid range
- Attention weights: Normalized
- Dashboard: Renders successfully
- AUX panels: Display when data present
