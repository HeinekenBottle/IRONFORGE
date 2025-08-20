# Enhanced Session Schema Reference

## Schema Overview

Enhanced session files contain price relativity data with archaeological zone analysis, news context, and temporal positioning for IRONFORGE analysis.

## Core Event Structure

### Base Fields
```json
{
  "type": "string",                    // Event type classification
  "original_type": "string",           // Original event designation  
  "magnitude": "float",                // Event magnitude (0.0-1.0)
  "value": "float",                    // Relative value metric
  "timestamp": "HH:MM:SS",             // Session-relative time
  "price_level": "float",              // Absolute price level
  "range_position": "float"            // Position in session range (0.0-1.0)
}
```

### Enhanced Positioning
```json
{
  "duration": "int",                   // Event duration in seconds
  "htf_confluence": "float",           // Higher timeframe confluence (0.0-1.0)
  "pct_from_open": "float",            // Percentage from session open
  "pct_from_high": "float",            // Percentage from session high  
  "pct_from_low": "float",             // Percentage from session low
  "price_momentum": "float",           // Price momentum indicator
  "energy_density": "float",           // Energy density (often null - 70.9%)
  "time_since_session_open": "int",    // Minutes since session start
  "normalized_time": "float",          // Normalized time position (0.0-1.0)
  "absolute_price": "float"            // Absolute price (duplicate of price_level)
}
```

## Archaeological Zone Fields

### Zone Classification
```json
{
  "archaeological_significance": "float",      // Zone significance rating
  "event_family": "string",                   // Event family classification
  "structural_role": "string",                // Structural role in session
  "dimensional_relationship": "string",       // Theory B relationship
  "source": "string"                          // Data source identifier
}
```

### Enhancement Flags
```json
{
  "enhanced_session_data": true,              // Enhancement processing flag
  "relativity_enhanced": true                 // Relativity analysis complete
}
```

## News Context Integration

### News Proximity
```json
{
  "news_context": {
    "news_bucket": "string",                  // quiet | low±30m | medium±60m | high±120m
    "news_distance_mins": "float|null",       // Minutes to nearest economic event
    "closest_news_event": "object|null",      // Full news event details
    "news_impact_level": "string|null"        // low | medium | high
  }
}
```

### Day Context  
```json
{
  "day_context": {
    "day_of_week": "string",                  // Monday | Tuesday | ... | Sunday
    "trading_day": "YYYY-MM-DD",             // Trading day date
    "session_overlap": "string",              // gap_age | session_boundary | within_session
    "is_enhanced": true                       // Day enhancement flag
  }
}
```

## Session Metadata

### Session Information
```json
{
  "session_info": {
    "session_type": "string",                 // ASIA | LONDON | NY_AM | NY_PM | etc
    "trading_day": "YYYY-MM-DD",             // Trading day identifier
    "start_time": "HH:MM:SS",                // Session start time (ET)
    "end_time": "HH:MM:SS",                  // Session end time (ET)  
    "duration_minutes": "int",               // Total session duration
    "timezone": "ET"                         // All times in Eastern Time
  }
}
```

### Enhanced Statistics
```json
{
  "enhanced_stats": {
    "total_events": "int",                    // Total events in session
    "rd40_events": "int",                     // RD@40 archaeological events
    "news_events_mapped": "int",              // News events successfully mapped
    "enhancement_timestamp": "ISO8601",       // When enhancement was applied
    "news_source": "string"                  // Economic calendar source
  }
}
```

## File Organization

### Directory Structure
```
/data/day_news_enhanced/
├── day_news_enhanced_rel_ASIA_Lvl-1_2025_07_28.json
├── day_news_enhanced_rel_LONDON_Lvl-1_2025_07_29.json  
├── day_news_enhanced_rel_NY_AM_Lvl-1_2025_07_30.json
└── day_news_enhanced_rel_NY_PM_Lvl-1_2025_08_05.json
```

### Naming Convention
`day_news_enhanced_rel_{SESSION}_{LEVEL}_{TRADING_DAY}.json`

- **SESSION**: ASIA | MIDNIGHT | LONDON | PREMARKET | NY_AM | LUNCH | NY_PM
- **LEVEL**: Lvl-1 (relativity level indicator)  
- **TRADING_DAY**: YYYY_MM_DD format

## Data Quality Notes

### Complete Fields (100% populated)
- `timestamp`, `price_level`, `range_position`
- `news_context`, `day_context`
- `enhanced_session_data`, `relativity_enhanced`

### Partial Fields (70-90% populated)  
- `htf_confluence`, `price_momentum`
- `archaeological_significance`

### Missing Fields (often null)
- `energy_density` (70.9% null)
- `f8_level` (100% null)
- `news_distance_mins` (for quiet periods)

## Usage Examples

### RD@40 Event Filtering
```python
rd40_events = [e for e in events 
               if e.get('dimensional_relationship') == 'dimensional_destiny_40pct']
```

### News Proximity Analysis
```python
high_impact = [e for e in events 
               if e.get('news_context', {}).get('news_bucket') == 'high±120m']
```

### Day-of-Week Splitting
```python
tuesday_events = [e for e in events 
                  if e.get('day_context', {}).get('day_of_week') == 'Tuesday']
```

This schema enables comprehensive analysis of price behavior, archaeological zones, and contextual factors for IRONFORGE research.