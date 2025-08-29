# IRONFORGE Operations Guide

**Daily Operations, A/B Adapters, and Watchlist Curation**

## Daily Run Workflow

### Standard Pipeline
```bash
# Morning setup
cd /path/to/ironforge
source venv/bin/activate

# Run discovery pipeline
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
python -m ironforge.sdk.cli score-session     --config configs/dev.yml
python -m ironforge.sdk.cli validate-run      --config configs/dev.yml
python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml

# Check results
python -m ironforge.sdk.cli status --runs runs
open runs/$(date +%F)/minidash.html
```

### Automated Daily Run
```bash
#!/bin/bash
# daily_ironforge.sh
set -e

IRONFORGE_DIR="/path/to/ironforge"
CONFIG="configs/dev.yml"
DATE=$(date +%F)

cd "$IRONFORGE_DIR"

echo "🏛️ IRONFORGE Daily Run - $DATE"
echo "================================"

# Run pipeline
python -m ironforge.sdk.cli discover-temporal --config "$CONFIG"
python -m ironforge.sdk.cli score-session     --config "$CONFIG"
python -m ironforge.sdk.cli validate-run      --config "$CONFIG"
python -m ironforge.sdk.cli report-minimal    --config "$CONFIG"

# Validate outputs
if [ -f "runs/$DATE/minidash.html" ]; then
    echo "✅ Daily run complete: runs/$DATE/minidash.html"
else
    echo "❌ Daily run failed: missing minidash"
    exit 1
fi
```

## A/B Adapter Configuration

### Available Adapters

All adapters are **OFF by default**. Enable selectively for testing:

#### 1. Confluence Phase Weighting
```yaml
# configs/dev.yml
confluence:
  phase_weighting: true  # Enable HTF bucket weighting
  phase_weights:
    asia: 0.8
    london: 1.2
    ny_am: 1.5
    lunch: 0.6
    ny_pm: 1.3
```

**Purpose**: Weight confluence scores by session phase
**Use Case**: Emphasize high-activity periods (London, NY)
**Risk**: May bias against off-hours patterns

#### 2. Chain Bonus Scoring
```yaml
confluence:
  chain_bonus: true      # Enable within-session chain bonus
  chain_multiplier: 1.25 # 25% bonus for chained patterns
  min_chain_length: 3    # Minimum events in chain
```

**Purpose**: Boost scores for connected event sequences
**Use Case**: Reward complex multi-step patterns
**Risk**: May over-emphasize long sequences

#### 3. Momentum Burst Enhancement
```yaml
momentum:
  mt_burst_boost: true   # Enable momentum burst detection
  burst_threshold: 2.5   # Standard deviations above mean
  burst_window_s: 300    # 5-minute detection window
```

**Purpose**: Enhance detection of momentum bursts
**Use Case**: Capture rapid price acceleration events
**Risk**: May create false positives in volatile periods

#### 4. Temporal Bounds
```yaml
momentum:
  mt_dt_bounds_s: [60, 1800]  # 1 minute to 30 minutes
```

**Purpose**: Constrain momentum analysis to specific timeframes
**Use Case**: Focus on medium-term momentum patterns
**Risk**: May miss very short or long-term patterns

#### 5. Liquidity Analysis Window
```yaml
liquidity:
  liq_short_s: 180       # 3-minute short-term window
```

**Purpose**: Define short-term liquidity analysis period
**Use Case**: Capture rapid liquidity events
**Risk**: May miss longer-term liquidity patterns

### A/B Testing Protocol

#### 1. Baseline Run
```bash
# Run with all adapters OFF (default)
python -m ironforge.sdk.cli discover-temporal --config configs/baseline.yml
mv runs/$(date +%F) runs/$(date +%F)_baseline
```

#### 2. Experimental Run
```bash
# Run with specific adapter enabled
python -m ironforge.sdk.cli discover-temporal --config configs/experiment.yml
mv runs/$(date +%F) runs/$(date +%F)_experiment
```

#### 3. Comparison Analysis
```python
# Compare results
from ironforge.analysis.ab_testing import compare_runs

baseline = "runs/2025-08-19_baseline"
experiment = "runs/2025-08-19_experiment"

comparison = compare_runs(baseline, experiment)
print(comparison.summary())
```

## Watchlist Curation

### Symbol Management
```yaml
# configs/watchlist.yml
symbols:
  primary:
    - NQ    # Nasdaq futures
    - ES    # S&P 500 futures
    - YM    # Dow futures
  
  secondary:
    - RTY   # Russell 2000 futures
    - GC    # Gold futures
    - CL    # Crude oil futures
  
  experimental:
    - BTC   # Bitcoin (if available)
    - ETH   # Ethereum (if available)
```

### Session Filtering
```yaml
sessions:
  enabled:
    - ASIA
    - LONDON  
    - NY_AM
    - LUNCH
    - NY_PM
  
  disabled:
    - MIDNIGHT    # Low activity
    - PREMARKET   # Thin liquidity
```

### Quality Gates
```yaml
quality:
  min_events_per_session: 5      # Minimum events required
  max_gap_minutes: 60            # Maximum data gap allowed
  min_volume_threshold: 1000     # Minimum volume per candle
```

## Monitoring and Alerts

### Health Checks
```bash
# Check system health
python tools/smoke_checks.py

# Validate recent runs
python -c "
import json
from pathlib import Path

runs = sorted(Path('runs').glob('20*'))[-3:]
for run in runs:
    stats_file = run / 'confluence' / 'stats.json'
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        health = stats.get('health_status', 'unknown')
        print(f'{run.name}: {health}')
"
```

### Performance Monitoring
```bash
# Check processing times
grep "Processing time" logs/*.log | tail -10

# Check memory usage
ps aux | grep python | grep ironforge

# Check disk usage
du -sh runs/ data/shards/
```

### Alert Conditions
1. **No runs for 24+ hours**: System may be down
2. **Health status: error**: Data quality issues
3. **Missing minidash**: Reporting pipeline failure
4. **Zero confluence scores**: Scoring pipeline failure
5. **Attention data missing**: TGAT model issues

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Verify entrypoints
python -c "
from ironforge.api import run_discovery, score_confluence, validate_run, build_minidash
print('✅ All entrypoints working')
"
```

#### 2. Schema Dimension Errors
```bash
# Check shard dimensions
python -c "
import pyarrow.parquet as pq
import glob

shard = glob.glob('data/shards/*/shard_*')[0]
nodes = pq.read_table(f'{shard}/nodes.parquet')
edges = pq.read_table(f'{shard}/edges.parquet')

node_features = len([c for c in nodes.column_names if c.startswith('f')])
edge_features = len([c for c in edges.column_names if c.startswith('e')])

print(f'Node features: {node_features} (expected: 51)')
print(f'Edge features: {edge_features} (expected: 20)')
"
```

#### 3. Scale Detection Issues
```bash
# Check confluence scale
python -c "
import pandas as pd
import glob

scores_files = glob.glob('runs/*/confluence/scores.parquet')
if scores_files:
    df = pd.read_parquet(scores_files[-1])
    max_score = df['confidence'].max() if 'confidence' in df.columns else 'N/A'
    print(f'Max confluence score: {max_score}')
    print(f'Scale: {\"0-1\" if max_score <= 1 else \"0-100\" if max_score <= 100 else \"threshold\"}')
"
```

## Maintenance

### Weekly Tasks
- Review run logs for errors
- Check disk space usage
- Validate recent run quality
- Update watchlist if needed

### Monthly Tasks  
- Archive old runs (keep last 30 days)
- Review adapter performance
- Update documentation
- Performance optimization review

### Quarterly Tasks
- Full system health audit
- Adapter effectiveness analysis
- Watchlist performance review
- Infrastructure capacity planning

---

**IRONFORGE Operations** — Daily workflows and system management
