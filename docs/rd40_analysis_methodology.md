# RD@40 Analysis Methodology

## Overview

RD@40 (Redelivery at 40% archaeological zone) analysis examines price behavior at the 40% level of the final session range, implementing Theory B temporal non-locality principles.

## Core Concepts

### Archaeological Zones
- **40% Zone**: Events positioned at 40% of the **final** session range
- **Theory B Precision**: 7.55-point threshold for dimensional relationship validation
- **Temporal Non-locality**: Events "know" their position relative to eventual completion

### Path Classification (Outcome-Based)

**E1 (Continue)**: Neither criteria met - 46% of cases
**E2 (Mean-Revert)**: hit_mid ≤ 60 minutes - 13% of cases  
**E3 (Accelerate)**: hit_80 ≤ 90 minutes - 41% of cases

```python
def calculate_outcome_path(rd40_event, all_events, rd40_index, session_start):
    if hit_mid_time is not None and hit_mid_time <= 60:
        return {'path': 'E2', 'reason': f'hit_mid_at_{hit_mid_time:.0f}m'}
    elif hit_80_time is not None and hit_80_time <= 90:
        return {'path': 'E3', 'reason': f'hit_80_at_{hit_80_time:.0f}m'}
    else:
        return {'path': 'E1', 'reason': 'neither_criteria_met'}
```

## Statistical Framework

### Confidence Intervals
- **Wilson CI**: Robust for small sample sizes
- **Bootstrap CI**: Alternative validation method
- **Inconclusive Rule**: CI width > 30 percentage points

### Sample Size Management
- **Merge Rule**: n < 5 → "Other" bucket
- **Minimum Threshold**: 5 observations for statistical validity
- **CI Width Check**: Flag unreliable estimates

### Day-of-Week Analysis

**Tuesday Pattern**: 56% E3 acceleration bias [42-69% CI]
**Wednesday Pattern**: 59% E1 continuation bias [46-70% CI]
**Sample Sizes**: Tuesday (18), Wednesday (17) - adequate for CI

## News Integration

### Economic Calendar Integration
```python
def _get_news_bucket(self, impact: str, distance_mins: float) -> str:
    if impact == 'high' and distance_mins <= 120:
        return 'high±120m'
    elif impact == 'medium' and distance_mins <= 60:
        return 'medium±60m'  
    elif impact == 'low' and distance_mins <= 30:
        return 'low±30m'
    else:
        return 'quiet'
```

### News Proximity Effects
- **High±120m**: Counter-intuitive less acceleration (33% E3)
- **Quiet periods**: Higher acceleration tendency (46% E3)
- **Sample limitation**: News buckets often merge due to n<5 rule

## Enhanced Session Integration

### Data Enhancement Pipeline
1. Load base session files from `/data/enhanced/`
2. Add day-of-week context using session timestamp
3. Integrate economic calendar events with UTC normalization
4. Calculate news proximity buckets with impact weighting
5. Generate session overlap indicators (gap_age, session_boundary)
6. Save to `/data/day_news_enhanced/` with rich metadata

### Files Enhanced
- **Sessions**: 57 enhanced with day/news context
- **Events**: 209 RD@40 events analyzed
- **Coverage**: July 28 - August 15, 2025

## Query Integration

### Enhanced TQE Handlers
```python
# Day-specific analysis
"day split" → _analyze_rd40_day_split()

# News-specific analysis  
"news split" → _analyze_rd40_news_split()

# Combined matrix
"day news matrix" → _analyze_rd40_day_news_matrix()
```

## Validation Results

### Data Quality Assessment
- **Complete Fields**: timestamp, price_level, range_position
- **Missing Fields**: f8_level (100% null), energy_density (70.9% null)
- **Classification Source**: Outcome-based using session range dynamics

### Key Findings
1. **Realistic Distribution**: E1 46%, E3 41%, E2 13% (vs broken 100% E1)
2. **Day Effects**: Tuesday acceleration, Wednesday stalling  
3. **News Paradox**: High-impact news shows less acceleration
4. **Timing Issues**: Negative time-to-80 values suggest calculation errors

## Implementation Files

- `enhanced_statistical_framework.py`: Wilson CI + Bootstrap CI
- `economic_calendar_loader.py`: News integration system
- `rd40_fixed_classification.py`: Outcome-based path classification
- `enhanced_temporal_query_engine.py`: Query routing and analysis
- `rd40_final_scan.py`: Complete analysis with all fixes

## Next Phase Considerations

The liquidity/HTF follow-through analysis builds on this foundation by:
- Replacing time-based targets (60/80 min) with market structure
- Measuring actual liquidity sweeps and HTF level touches
- Maintaining statistical rigor with Wilson CI framework
- Preserving day/news context splits for pattern validation