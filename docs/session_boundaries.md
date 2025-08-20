# IRONFORGE Session Boundaries Reference

## Official Session Times (All ET)

| Session | Start | End | Duration | Notes |
|---------|-------|-----|----------|-------|
| **ASIA** | 19:00 | 23:59 | 299 min | Starts on calendar day before trading day |
| **MIDNIGHT** | 00:00 | 00:29 | 29 min | Minimal participation |
| **LONDON** | 02:00 | 04:59 | 179 min | European institutional hours |
| **PREMARKET** | 07:00 | 09:29 | 149 min | Pre-market preparation |
| **NY_AM** | 09:30 | 11:59 | 149 min | Morning institutional activity |
| **LUNCH** | 12:00 | 13:29 | 89 min | Structural break |
| **NY_PM** | 13:30 | 16:10 | 160 min | Afternoon institutional activity |

## Trading Day Definition

- **Start**: ASIA 19:00 ET (on calendar day before)
- **End**: NY_PM 16:10 ET 
- **Total Duration**: 21 hours 10 minutes
- **Session Sequence**: ASIA → MIDNIGHT → LONDON → PREMARKET → NY_AM → LUNCH → NY_PM

## Analysis Window Rules

### Same-Session Analysis
- **Scope**: From event timestamp to current session end
- **Example**: RD@40 at 14:35 ET in NY_PM → measure until 16:10 ET

### Trading Day Analysis  
- **Scope**: From session boundary to NY_PM close (16:10 ET)
- **Purpose**: Capture cross-session follow-through

### Cross-Session Chains
- **Structural Breaks**: MIDNIGHT (29min) and LUNCH (89min)
- **Chain Logic**: Track events through gaps for liquidity flow analysis

## Implementation Notes

- All times stored as `time(hour, minute)` objects in ET
- Session specs defined in `session_time_manager.py`
- ASIA cross-day logic: starts calendar day before trading day label
- Duration calculations handle cross-midnight transitions automatically

## Common Errors to Avoid

❌ **Wrong ASIA times**: 03:00-11:00 ET  
✅ **Correct ASIA times**: 19:00-23:59 ET

❌ **Wrong NY_PM times**: 13:30-17:00 ET  
✅ **Correct NY_PM times**: 13:30-16:10 ET

❌ **Wrong LONDON times**: 03:00-11:00 ET  
✅ **Correct LONDON times**: 02:00-04:59 ET

This reference prevents repeated confusion and ensures consistent temporal analysis across all IRONFORGE systems.