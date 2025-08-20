#!/usr/bin/env python3
"""
Economic Calendar - Surgical Loader
Clean, focused calendar loader with hard fail-safe validation
"""

import pandas as pd
from pathlib import Path

def load_calendar(path: str) -> pd.DataFrame:
    """
    Load economic calendar from CSV with hard validation
    
    Args:
        path: Path to calendar CSV file
        
    Returns:
        DataFrame with dt_et, impact, event, source columns
        
    Raises:
        RuntimeError: If calendar file is empty or missing required columns
        FileNotFoundError: If calendar file doesn't exist
    """
    # Check file exists
    calendar_path = Path(path)
    if not calendar_path.exists():
        raise FileNotFoundError(f"Calendar file not found: {path}")
    
    # Load CSV
    try:
        cal = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read calendar CSV: {e}")
    
    # Hard validation - no silent failures
    if cal.empty:
        raise RuntimeError("Calendar file is empty")
    
    required_cols = {'dt_et', 'impact', 'event', 'source'}
    missing_cols = required_cols - set(cal.columns)
    if missing_cols:
        raise RuntimeError(f"Calendar missing required columns: {missing_cols}")
    
    # Parse datetime with ET timezone
    try:
        cal["dt_et"] = pd.to_datetime(cal["dt_et"])
        # Localize to ET timezone
        import pytz
        et_tz = pytz.timezone('America/New_York')
        cal["dt_et"] = cal["dt_et"].dt.tz_localize(et_tz)
    except Exception as e:
        raise RuntimeError(f"Failed to parse dt_et column: {e}")
    
    # Normalize impact levels
    impact_map = {"high": "high", "medium": "medium", "low": "low"}
    cal["impact"] = cal["impact"].str.lower().map(impact_map)
    
    # Check for invalid impact levels
    invalid_impacts = cal["impact"].isna().sum()
    if invalid_impacts > 0:
        raise RuntimeError(f"Calendar contains {invalid_impacts} invalid impact levels")
    
    # Sort by datetime and reset index
    cal = cal.sort_values("dt_et").reset_index(drop=True)
    
    print(f"✓ Calendar loaded: {len(cal)} events from {cal['dt_et'].min()} to {cal['dt_et'].max()}")
    print(f"  Impact distribution: {dict(cal['impact'].value_counts())}")
    
    return cal


def attach_news(rd: pd.DataFrame, cal: pd.DataFrame, buckets: dict) -> pd.DataFrame:
    """
    Attach nearest news events to RD@40 events using merge_asof
    
    Args:
        rd: DataFrame with RD@40 events (must have 'dt_et' column)
        cal: Calendar DataFrame from load_calendar()
        buckets: Dict with impact level buckets {high: 120, medium: 60, low: 30}
        
    Returns:
        DataFrame with news context attached
    """
    # Validate inputs
    if rd.empty:
        raise ValueError("RD@40 DataFrame is empty")
    if cal.empty:
        raise ValueError("Calendar DataFrame is empty")
    if 'dt_et' not in rd.columns:
        raise ValueError("RD@40 DataFrame missing 'dt_et' column")
    
    # Ensure both DataFrames are sorted
    rd_sorted = rd.sort_values("dt_et").copy()
    cal_sorted = cal.sort_values("dt_et").copy()
    
    # Calculate maximum tolerance from buckets
    max_tolerance_mins = max(buckets.values())
    tolerance = pd.Timedelta(minutes=max_tolerance_mins)
    
    # Rename calendar dt_et to avoid collision
    cal_renamed = cal_sorted.rename(columns={'dt_et': 'news_dt_et'})
    
    # Perform nearest-event merge using merge_asof
    merged = pd.merge_asof(
        rd_sorted,
        cal_renamed,
        left_on="dt_et",
        right_on="news_dt_et", 
        direction="nearest",
        tolerance=tolerance
    )
    
    # Calculate signed distance in minutes
    # Positive = news event is AFTER RD@40, Negative = news event is BEFORE RD@40
    merged["news_distance_mins"] = (merged["news_dt_et"] - merged["dt_et"]).dt.total_seconds() / 60.0
    
    # Assign news buckets based on impact and distance
    def assign_bucket(row):
        if pd.isna(row["impact"]):
            return "quiet"
        
        impact = row["impact"]
        distance = abs(row["news_distance_mins"]) if not pd.isna(row["news_distance_mins"]) else float('inf')
        tolerance_mins = buckets.get(impact, 0)
        
        if distance <= tolerance_mins:
            return f"{impact}±{tolerance_mins}m"
        else:
            return "quiet"
    
    merged["news_bucket"] = merged.apply(assign_bucket, axis=1)
    
    # Rename calendar columns for clarity
    merged = merged.rename(columns={
        "news_dt_et": "news_event_time",
        "event": "news_event",
        "source": "news_source"
    })
    
    # Validation checks
    total_events = len(merged)
    quiet_events = (merged["news_bucket"] == "quiet").sum()
    news_events = total_events - quiet_events
    
    print(f"✓ News attachment complete:")
    print(f"  Total RD@40 events: {total_events}")
    print(f"  Events with news proximity: {news_events} ({news_events/total_events*100:.1f}%)")
    print(f"  Quiet periods: {quiet_events} ({quiet_events/total_events*100:.1f}%)")
    print(f"  Bucket distribution: {dict(merged['news_bucket'].value_counts())}")
    
    return merged


if __name__ == "__main__":
    # Test the loader
    try:
        cal = load_calendar("data/calendar/events.csv")
        print("Calendar loading test: ✓ PASSED")
    except Exception as e:
        print(f"Calendar loading test: ❌ FAILED - {e}")