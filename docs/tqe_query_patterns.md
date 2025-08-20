# TQE Query Interface Patterns

## Query Routing Architecture

The Enhanced Temporal Query Engine (TQE) uses natural language pattern matching to route queries to specialized analysis handlers.

## Core Query Patterns

### RD@40 Analysis Patterns

**General RD@40 Analysis**
```
Query: "Show me RD@40 patterns"
Handler: _analyze_post_rd40_sequences()
Output: Sequence patterns, path distribution, basic statistics
```

**Day-of-Week Analysis**
```
Query: "RD@40 day split" | "day profile analysis" | "weekday patterns"
Handler: _analyze_rd40_by_day()
Output: Day-wise path distribution with Wilson CI
```

**News Proximity Analysis**  
```
Query: "RD@40 news split" | "news impact patterns" | "economic events"
Handler: _analyze_rd40_by_news()
Output: News bucket analysis with impact correlation
```

**Combined Matrix Analysis**
```
Query: "day news matrix" | "day and news interaction"
Handler: _analyze_rd40_day_news_matrix()  
Output: Cross-tabulated Day × News patterns
```

### Session Context Patterns

**Gap Age Analysis**
```
Query: "gap age split" | "session gap analysis"
Handler: _analyze_gap_age_split()
Output: Pre-session gap impact on RD@40 behavior
```

**Session Overlap Analysis**
```
Query: "overlap split" | "session boundary analysis"
Handler: _analyze_overlap_split()
Output: Session transition effects on path outcomes
```

### Archaeological Zone Patterns

**Zone Validation**
```
Query: "validate archaeological zones" | "theory b precision"
Handler: _validate_archaeological_zones()
Output: Theory B precision metrics, dimensional relationships
```

**Cross-Zone Analysis**
```
Query: "cross zone patterns" | "multi-zone analysis"
Handler: _analyze_cross_zone_patterns()
Output: Inter-zone correlation analysis
```

## Query Routing Logic

### Precedence Order
```python
# Specific patterns match first
if "day" in question.lower() and "news" in question.lower() and "matrix" in question.lower():
    return self._analyze_rd40_day_news_matrix(question)
elif "gap" in question.lower() and ("age" in question.lower() or "split" in question.lower()):
    return self._analyze_gap_age_split(question)
elif "overlap" in question.lower() and ("split" in question.lower() or "session" in question.lower()):
    return self._analyze_overlap_split(question)
elif "day" in question.lower() and ("split" in question.lower() or "profile" in question.lower()):
    return self._analyze_rd40_by_day(question)
elif "news" in question.lower() and ("split" in question.lower() or "impact" in question.lower()):
    return self._analyze_rd40_by_news(question)
# General RD@40 patterns match last
elif "rd@40" in question.lower() or "rd40" in question.lower():
    return self._analyze_post_rd40_sequences(question)
```

## Standard Output Format

### Statistical Analysis Output
```json
{
  "analysis_type": "string",
  "summary": "string",
  "sample_size": "int", 
  "path_distribution": {
    "E1": {"count": "int", "percentage": "float", "ci_lower": "float", "ci_upper": "float"},
    "E2": {"count": "int", "percentage": "float", "ci_lower": "float", "ci_upper": "float"},
    "E3": {"count": "int", "percentage": "float", "ci_lower": "float", "ci_upper": "float"}
  },
  "insights": ["string"],
  "methodology": "string"
}
```

### Matrix Analysis Output
```json
{
  "matrix_type": "Day × News",
  "rows": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
  "columns": ["quiet", "low±30m", "medium±60m", "high±120m"],
  "cells": {
    "Monday_quiet": {"E3_pct": "float", "ci": "string", "n": "int"},
    "Tuesday_high±120m": {"E3_pct": "float", "ci": "string", "n": "int"}
  },
  "significance_notes": ["string"]
}
```

## Implementation Examples

### Day Analysis Handler
```python
def _analyze_rd40_by_day(self, question: str) -> Dict[str, Any]:
    # Load enhanced session data
    enhanced_files = glob.glob('/data/day_news_enhanced/*.json')
    
    # Extract RD@40 events with day context
    rd40_events = []
    for file_path in enhanced_files:
        events = self._load_enhanced_events(file_path)
        rd40_events.extend(self._filter_rd40_events(events))
    
    # Group by day of week
    day_groups = self._group_by_day(rd40_events)
    
    # Calculate statistics with Wilson CI
    results = self._calculate_day_statistics(day_groups)
    
    return results
```

### Statistical Framework Integration
```python
def _calculate_wilson_ci(self, successes: int, trials: int) -> Tuple[float, float]:
    """Wilson confidence interval for proportions"""
    if trials == 0:
        return (0.0, 0.0)
    # Wilson CI calculation with z=1.96 for 95% confidence
    # Returns (lower_bound, upper_bound)
```

## Query Interface Best Practices

### Query Construction
- Use specific keywords for targeted analysis
- Combine terms for matrix analysis ("day news matrix")
- Include "split" for categorical breakdowns
- Reference "RD@40" or "rd40" for archaeological focus

### Output Interpretation
- **CI Width > 30pp**: Mark as inconclusive
- **Sample Size < 5**: Auto-merge to "Other" bucket  
- **Percentage Format**: Display as whole numbers with CI brackets

### Error Handling
```python
if not enhanced_files:
    return {"error": "No enhanced session files found"}
    
if total_rd40_events < 10:
    return {"warning": "Insufficient data for reliable analysis"}
```

## Extensibility Framework

### Adding New Handlers
1. Define pattern matching keywords
2. Implement analysis logic following statistical framework
3. Add to routing precedence order
4. Include Wilson CI calculations for proportions
5. Format output following standard schema

### Query Routing Extension
```python
elif "new_pattern" in question.lower():
    return self._analyze_new_pattern(question)
```

This framework ensures consistent query processing, statistical rigor, and extensible pattern analysis across all IRONFORGE research domains.