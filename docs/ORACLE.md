# Oracle Temporal Non-locality

## Overview

The Oracle system predicts final session ranges from early events using temporal non-locality patterns discovered through dimensional anchoring theory. Early events contain forward-looking information about eventual session completion.

## Configuration

```yaml
# config.yaml
oracle:
  enabled: false          # Disabled by default
  early_pct: 0.20         # Use first 20% of events (0 < early_pct ≤ 0.5)
  output_path: "oracle_predictions.parquet"
```

Environment variables:
```bash
export IFG_ORACLE__ENABLED=true
export IFG_ORACLE__EARLY_PCT=0.25
```

## Usage

### CLI
```bash
# Oracle runs automatically when enabled during discovery
ironforge discover --config config.yaml

# Check output
ls runs/2025-08-19/oracle_predictions.parquet
```

### Programmatic
```python
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

# Build session graph
builder = EnhancedGraphBuilder()
graph = builder.build_session_graph(session_data)

# Run oracle predictions
discovery = IRONFORGEDiscovery()
predictions = discovery.predict_session_range(graph, early_batch_pct=0.20)

print(f"Predicted range: {predictions['pred_low']} - {predictions['pred_high']}")
print(f"Confidence: {predictions['confidence']}")
```

## Output Schema

Oracle predictions are saved as `oracle_predictions.parquet` with exact schema:

| Column | Type | Description |
|--------|------|-------------|
| `run_dir` | str | Output directory path |
| `session_date` | str | Session date (YYYY-MM-DD) |
| `pct_seen` | float | Percentage of session analyzed (0-1) |
| `n_events` | int | Number of early events used |
| `pred_low` | float | Predicted session low |
| `pred_high` | float | Predicted session high |
| `center` | float | Predicted range center |
| `half_range` | float | Predicted half-range width |
| `confidence` | float | Prediction confidence (0-1) |
| `pattern_id` | str | Linked pattern identifier |
| `start_ts` | str | Session start timestamp |
| `end_ts` | str | Session end timestamp |
| `early_expansion_cnt` | int | Expansion events in early portion |
| `early_retracement_cnt` | int | Retracement events in early portion |
| `early_reversal_cnt` | int | Reversal events in early portion |
| `first_seq` | str | Event sequence pattern (e.g., "E→R→E") |

## Theory

Based on dimensional anchoring discovery:

> "Events at 40% dimensional zones predict final session structure before completion. The 40% zone event positioned itself with 7.55 point precision to the final session range's 40% level before the session range was fully established."

This suggests **temporal non-locality**: early events contain information about future session completion through attention mechanisms in the 45D semantic feature space.

## Limitations

- **Untrained Model**: Current regression head uses random weights - requires training on historical data
- **Minimum Events**: Requires ≥3 events per session  
- **Early Percentage**: Limited to (0, 0.5] to ensure meaningful predictions
- **Cold Start**: No learned patterns yet - predictions are structural placeholders

## Integration

Oracle predictions can be joined with discovery patterns via:
- `pattern_id` - Links to discovered pattern identifiers
- `(start_ts, end_ts)` - Time window overlap matching
- `run_dir` - Common run directory for batch processing

Confluence/reporting can read the sidecar parquet to display oracle insights alongside archaeological discoveries.

## Development

Run tests:
```bash
pytest tests/oracle/test_oracle_predictions.py -v
```

Enable debug logging:
```python
import logging
logging.getLogger('ironforge.learning.tgat_discovery').setLevel(logging.DEBUG)
```