# IRONFORGE Critical Bug Fix - Relativity Features Data Access

## Problem Statement
IRONFORGE is falling back to basic calculations despite enhanced sessions containing all required relativity features, causing loss of rich market intelligence and reducing pattern discovery from 420+ to ~30 patterns.

## Evidence of Bug
```
⚠️ Event at 15:17:00 lacks relativity features - using basic calculations
```

## Root Cause Diagnosed
IRONFORGE is looking for relativity features in `session_liquidity_events` but the features are stored in `price_movements` array.

## Enhanced Session Data Structure

### ✅ Relativity features ARE present in price_movements:
```json
{
  "price_movements": [
    {
      "timestamp": "15:17:00",
      "normalized_price": 0.25882352941176473,
      "pct_from_open": 0.0,
      "pct_from_high": 74.11764705882354,
      "pct_from_low": 25.882352941176475,
      "time_since_session_open": 0,
      "normalized_time": 0.0
    }
  ]
}
```

### ❌ session_liquidity_events lacks these features

## Technical Fix Required

1. Locate data access code that processes events and checks for relativity features
2. Redirect lookup from `session_liquidity_events` to `price_movements` array
3. Maintain timestamp matching to find correct relativity data
4. Preserve no-fallback policy for genuine enhanced sessions

## Files to Investigate

- `/IRONFORGE/learning/enhanced_graph_builder.py` - Contains the warning messages
- `/IRONFORGE/orchestrator.py` - Main discovery orchestration
- Any event processing pipelines

## Success Criteria

- Zero "lacks relativity features" warnings for enhanced sessions
- Rich movement types preserved: "fpfvg_formation", "expansion_start_higher"
- 420+ pattern discovery instead of 30-60 patterns
- Real market intelligence: price_momentum, liquidity_events maintained

## Test Validation
```bash
# This should run without warnings
python3 -c "
from learning.enhanced_graph_builder import EnhancedGraphBuilder
import json
with open('enhanced_sessions_with_relativity/enhanced_rel_NY_PM_Lvl-1_2025_07_29.json') as f:
    session_data = json.load(f)
builder = EnhancedGraphBuilder()
graph = builder.build_rich_graph(session_data)
print(f'Nodes: {graph[\"metadata\"][\"total_nodes\"]}')
"
```

## Agent Assignment
`iron-code-reviewer` - Find and fix the exact data access pattern causing the relativity features bug. Focus on redirecting event processing to use `price_movements` instead of `session_liquidity_events`.

## Context
Previous agent fix attempt failed - still getting warnings. Need precise technical fix to data access logic, not just code cleanup.