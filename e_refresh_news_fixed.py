#!/usr/bin/env python3
"""
E Refresh — Day/News + Liquidity (Fixed News Layer)

FIXED: Real calendar integration with hard fail-safe validation
- Loads actual economic events from CSV
- Performs ±window joins with proper timezone handling  
- Hard error when calendar missing (no silent "quiet" defaults)
- Proper day-of-week extraction with ET timezone conversion
"""

from liquidity_htf_analyzer import LiquidityHTFAnalyzer
from enhanced_statistical_framework import EnhancedStatisticalAnalyzer
from economic_calendar import load_calendar, attach_news
from e_refresh_analysis import calculate_wilson_ci
import json
import yaml
from collections import defaultdict, Counter
from datetime import datetime
import pytz
import pandas as pd

def load_settings():
    """Load settings with calendar configuration"""
    try:
        with open('settings.yml', 'r') as f:
            settings = yaml.safe_load(f)
        return settings
    except Exception as e:
        raise RuntimeError(f"Failed to load settings.yml: {e}")

def format_table_with_criteria(context_data, title):
    """Format table with 👍/👎 criteria and merge n<5"""
    print(f"\n📋 {title}:")
    
    # Merge contexts with n<5 into "Other"
    merged_data = {}
    other_successes = 0
    other_total = 0
    
    for context, (successes, total) in context_data.items():
        if total >= 5:
            merged_data[context] = (successes, total)
        else:
            other_successes += successes
            other_total += total
    
    if other_total > 0:
        merged_data["Other"] = (other_successes, other_total)
    
    # Calculate stats and sort by rate
    stats = []
    for context, (successes, total) in merged_data.items():
        rate = (successes / total * 100) if total > 0 else 0
        ci_lower, ci_upper, ci_width = calculate_wilson_ci(successes, total)
        
        # Criteria flags
        conclusive = "✓" if ci_width <= 30 else "⚠"
        
        stats.append({
            'context': context,
            'n': total,
            'successes': successes,
            'rate': rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'conclusive': conclusive
        })
    
    # Sort by rate descending
    stats.sort(key=lambda x: x['rate'], reverse=True)
    
    # Print table
    print("   ┌─────────────┬─────┬────────┬─────────────┬─────────────┐")
    print("   │ Context     │  n  │ Events │ Rate (%)    │ 95% CI      │")
    print("   ├─────────────┼─────┼────────┼─────────────┼─────────────┤")
    
    for stat in stats:
        context = stat['context'][:11].ljust(11)
        n = str(stat['n']).rjust(3)
        successes = str(stat['successes']).rjust(6)
        rate = f"{stat['rate']:.1f}%".rjust(11)
        ci = f"[{stat['ci_lower']:.0f}-{stat['ci_upper']:.0f}%] {stat['conclusive']}"
        
        print(f"   │ {context} │ {n} │ {successes} │ {rate} │ {ci.ljust(11)} │")
    
    print("   └─────────────┴─────┴────────┴─────────────┴─────────────┘")
    print("   ✓ = conclusive (CI width ≤30pp), ⚠ = inconclusive")
    
    return stats

def generate_e_refresh_news_fixed():
    """Generate E refresh analysis with FIXED news integration"""
    print("🔄 E Refresh — Day/News + Liquidity (FIXED NEWS LAYER)")
    print("=" * 70)
    
    # Load settings with hard validation
    settings = load_settings()
    calendar_config = settings.get('calendar', {})
    
    if not calendar_config.get('enabled', False):
        raise RuntimeError("Calendar is disabled in settings.yml - cannot run news analysis")
    
    calendar_path = calendar_config.get('path')
    if not calendar_path:
        raise RuntimeError("No calendar path specified in settings.yml")
    
    # Load calendar with hard fail-safe
    print(f"📅 Loading calendar from: {calendar_path}")
    try:
        calendar = load_calendar(calendar_path)
    except Exception as e:
        raise RuntimeError(f"Calendar loading failed: {e}")
    
    if calendar.empty:
        raise RuntimeError("Calendar loaded but is empty - cannot run news analysis")
    
    print(f"✓ Calendar validation passed: {len(calendar)} events loaded")
    
    # Load session data
    analyzer = LiquidityHTFAnalyzer()
    enhanced_sessions = analyzer.load_enhanced_sessions()
    print(f"📁 Loaded {len(enhanced_sessions)} enhanced sessions")
    
    # Extract all RD@40 events and convert to DataFrame for news joining
    rd40_events_list = []
    
    for session in enhanced_sessions:
        session_data = session['data']
        events = session_data.get('events', [])
        trading_day = session_data.get('session_info', {}).get('trading_day', '2025-08-01')
        
        # Find RD@40 events
        rd40_events = [e for e in events 
                      if e.get('dimensional_relationship') == 'dimensional_destiny_40pct']
        
        for rd40_event in rd40_events:
            # Extract timestamp_et and convert to datetime
            timestamp_et = rd40_event.get('timestamp_et', '')
            if timestamp_et:
                try:
                    dt_str = timestamp_et.replace(' ET', '')
                    dt_naive = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                    et_tz = pytz.timezone('America/New_York')
                    dt_et = et_tz.localize(dt_naive)
                    
                    # Add to list with essential data
                    rd40_events_list.append({
                        'session_file': session['file_path'],
                        'session_data': session_data,
                        'rd40_event': rd40_event,
                        'dt_et': dt_et,
                        'trading_day': trading_day
                    })
                except Exception as e:
                    print(f"⚠️ Failed to parse timestamp for event: {timestamp_et} - {e}")
    
    if not rd40_events_list:
        raise RuntimeError("No RD@40 events found with valid timestamps")
    
    # Convert to DataFrame for news joining
    rd40_df = pd.DataFrame(rd40_events_list)
    print(f"🎯 Found {len(rd40_df)} RD@40 events for news analysis")
    
    # Attach news events using merge_asof
    buckets = calendar_config.get('buckets', {'high': 120, 'medium': 60, 'low': 30})
    print(f"📊 Attaching news with buckets: {buckets}")
    
    rd40_with_news = attach_news(rd40_df, calendar, buckets)
    
    # HARD VALIDATION - Ensure we don't have all quiet events
    news_bucket_counts = rd40_with_news['news_bucket'].value_counts()
    quiet_count = news_bucket_counts.get('quiet', 0)
    total_count = len(rd40_with_news)
    
    if quiet_count == total_count:
        raise RuntimeError("All events classified as 'quiet' - check calendar path/tolerance")
    
    print(f"✓ News validation passed: {total_count - quiet_count}/{total_count} events have news proximity")
    
    # Extract day-of-week using proper ET timezone
    rd40_with_news['day_of_week'] = rd40_with_news['dt_et'].dt.day_name()
    
    # Now perform the liquidity analysis with proper context
    day_sweeps = defaultdict(lambda: [0, 0])  # [successes, total]
    news_sweeps = defaultdict(lambda: [0, 0])
    day_news_sweeps = defaultdict(lambda: [0, 0])  # Combined key
    session_sweeps = defaultdict(lambda: [0, 0])
    minute_events = defaultdict(int)
    
    print(f"🔄 Analyzing liquidity sweeps for {len(rd40_with_news)} events...")
    
    for _, row in rd40_with_news.iterrows():
        # Extract contexts
        day_context = row['day_of_week']
        news_context = row['news_bucket']
        session_context = row['rd40_event'].get('session_context', {}).get('session_type', 'OTHER')
        
        # Analyze liquidity sweeps
        session_data = row['session_data']
        rd40_event = row['rd40_event']
        trading_day = row['trading_day']
        
        events = session_data.get('events', [])
        liquidity_levels = analyzer.calculate_liquidity_levels(session_data, trading_day)
        sweeps = analyzer.detect_liquidity_sweeps(rd40_event, events, liquidity_levels)
        
        # Check for sweeps within 90 minutes
        has_sweep = False
        for sweep in sweeps:
            if sweep.time_to_sweep_mins <= 90:
                has_sweep = True
                break
        
        # Update counters
        day_sweeps[day_context][1] += 1
        news_sweeps[news_context][1] += 1
        day_news_key = f"{day_context}×{news_context}"
        day_news_sweeps[day_news_key][1] += 1
        session_sweeps[session_context][1] += 1
        
        if has_sweep:
            day_sweeps[day_context][0] += 1
            news_sweeps[news_context][0] += 1
            day_news_sweeps[day_news_key][0] += 1
            session_sweeps[session_context][0] += 1
        
        # Track minute-of-day
        minute_key = f"{row['dt_et'].hour:02d}:{row['dt_et'].minute:02d}"
        minute_events[minute_key] += 1
    
    # Generate enhanced analysis tables
    print(f"\n📊 ENHANCED DAY/NEWS ANALYSIS RESULTS")
    print("=" * 70)
    
    day_stats = format_table_with_criteria(dict(day_sweeps), "Day of Week Analysis (FIXED)")
    news_stats = format_table_with_criteria(dict(news_sweeps), "News Proximity Analysis (REAL)")
    
    # Day×News cross-tabulation (show only if meaningful)
    print(f"\n📋 Day×News Cross-Tabulation (n≥3 shown):")
    day_news_filtered = {k: v for k, v in day_news_sweeps.items() if v[1] >= 3}
    if day_news_filtered:
        day_news_stats = format_table_with_criteria(day_news_filtered, "Day×News Cross-Analysis")
    else:
        print("   No combinations with n≥3 found")
    
    session_stats = format_table_with_criteria(dict(session_sweeps), "Session Type Analysis")
    
    # Top 5 minutes analysis
    print(f"\n⏰ Top 5 Minutes (ET) for Post-RD Events:")
    top_minutes = sorted(minute_events.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for i, (minute, count) in enumerate(top_minutes, 1):
        print(f"   {i}. {minute} ET: {count} events")
    
    # Test 14:35 ET ±3m explicitly
    target_minutes = ['14:32', '14:33', '14:34', '14:35', '14:36', '14:37', '14:38']
    target_count = sum(minute_events.get(m, 0) for m in target_minutes)
    total_events = len(rd40_with_news)
    target_pct = (target_count / total_events * 100) if total_events > 0 else 0
    
    print(f"\n🎯 14:35 ET ±3m Analysis:")
    print(f"   Events in window: {target_count}")
    print(f"   Percentage of total: {target_pct:.1f}%")
    print(f"   Window coverage: {', '.join(f'{m}:{minute_events.get(m, 0)}' for m in target_minutes)}")
    
    # One-liner conclusions
    print(f"\n🔗 One-Liner Conclusions (FIXED NEWS):")
    
    # Strongest day link
    if day_stats and day_stats[0]['n'] >= 10 and day_stats[0]['ci_width'] <= 30:
        day_conclusion = f"{day_stats[0]['context']} drives {day_stats[0]['rate']:.1f}% sweep rate"
    else:
        day_conclusion = "inconclusive"
    
    # Strongest news link  
    if news_stats and news_stats[0]['n'] >= 10 and news_stats[0]['ci_width'] <= 30:
        news_conclusion = f"{news_stats[0]['context']} shows {news_stats[0]['rate']:.1f}% follow-through"
    else:
        news_conclusion = "inconclusive"
    
    print(f"   📅 Strongest day link: {day_conclusion}")
    print(f"   📰 Strongest news link: {news_conclusion}")
    
    # Final validation summary
    print(f"\n✅ E Refresh Analysis Complete (NEWS LAYER FIXED)")
    print(f"   RD@40 events analyzed: {len(rd40_with_news)}")
    print(f"   Calendar events: {len(calendar)} loaded successfully")
    print(f"   News bucket distribution: {dict(news_bucket_counts)}")
    print(f"   Day-of-week distribution: {dict(rd40_with_news['day_of_week'].value_counts())}")
    print(f"   ✓ No more silent 'quiet' defaults!")
    print(f"   ✓ Hard fail-safe validation passed!")

if __name__ == "__main__":
    try:
        generate_e_refresh_news_fixed()
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        print(f"🔧 This is expected behavior - the system now fails fast when news integration is broken!")
        raise