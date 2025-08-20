#!/usr/bin/env python3
"""
News Impact Analysis - Understanding the News Effect on RD@40 Sweep Patterns
Analyzes how different news proximity buckets affect liquidity sweep behavior
"""

from liquidity_htf_analyzer import LiquidityHTFAnalyzer
from e_refresh_analysis import calculate_wilson_ci
from collections import defaultdict, Counter

def analyze_news_impact():
    """Comprehensive analysis of news proximity effects"""
    print("📰 NEWS IMPACT ANALYSIS ON RD@40 SWEEP PATTERNS")
    print("=" * 70)
    
    analyzer = LiquidityHTFAnalyzer()
    enhanced_sessions = analyzer.load_enhanced_sessions()
    
    # Track news bucket distribution and performance
    news_stats = defaultdict(lambda: [0, 0])  # [sweeps, total]
    news_details = defaultdict(list)  # Store individual events for analysis
    
    print(f"📁 Analyzing {len(enhanced_sessions)} enhanced sessions...")
    
    for session in enhanced_sessions:
        session_data = session['data']
        events = session_data.get('events', [])
        trading_day = session_data.get('session_info', {}).get('trading_day', '2025-08-01')
        
        # Find RD@40 events
        rd40_events = [e for e in events 
                      if e.get('dimensional_relationship') == 'dimensional_destiny_40pct']
        
        for rd40_event in rd40_events:
            # Extract news context
            news_context = rd40_event.get('news_context', {})
            news_bucket = news_context.get('news_bucket', 'quiet')
            
            # Analyze liquidity sweeps
            liquidity_levels = analyzer.calculate_liquidity_levels(session_data, trading_day)
            sweeps = analyzer.detect_liquidity_sweeps(rd40_event, events, liquidity_levels)
            
            # Check for sweeps within 90 minutes
            has_sweep = False
            sweep_times = []
            for sweep in sweeps:
                if sweep.time_to_sweep_mins <= 90:
                    has_sweep = True
                    sweep_times.append(sweep.time_to_sweep_mins)
            
            # Update statistics
            news_stats[news_bucket][1] += 1
            if has_sweep:
                news_stats[news_bucket][0] += 1
            
            # Store event details
            news_details[news_bucket].append({
                'event': rd40_event,
                'has_sweep': has_sweep,
                'sweep_times': sweep_times,
                'trading_day': trading_day,
                'news_context': news_context
            })
    
    # Analysis Results
    total_events = sum(total for sweeps, total in news_stats.values())
    print(f"🎯 Total RD@40 events analyzed: {total_events}")
    
    print(f"\n📊 NEWS BUCKET PERFORMANCE:")
    print("=" * 60)
    
    # Sort by performance rate
    sorted_news = sorted(news_stats.items(), 
                        key=lambda x: (x[1][0] / x[1][1] if x[1][1] > 0 else 0), 
                        reverse=True)
    
    for news_bucket, (sweeps, total) in sorted_news:
        rate = (sweeps / total * 100) if total > 0 else 0
        ci_lower, ci_upper, ci_width = calculate_wilson_ci(sweeps, total)
        conclusive = "✓" if ci_width <= 30 else "⚠"
        
        print(f"   {news_bucket:<15}: {rate:5.1f}% ({sweeps:2d}/{total:2d}) "
              f"[{ci_lower:.0f}-{ci_upper:.0f}%] {conclusive}")
    
    # The Big Question: Why is everything "quiet"?
    print(f"\n🤔 THE NEWS PUZZLE:")
    print("=" * 40)
    
    quiet_events = len(news_details.get('quiet', []))
    non_quiet_events = total_events - quiet_events
    
    print(f"   'quiet' periods: {quiet_events} events ({quiet_events/total_events*100:.1f}%)")
    print(f"   News proximity events: {non_quiet_events} events ({non_quiet_events/total_events*100:.1f}%)")
    
    if non_quiet_events == 0:
        print(f"\n❗ CRITICAL FINDING: ALL EVENTS CLASSIFIED AS 'QUIET'")
        print(f"   This suggests:")
        print(f"   1. No high/medium/low impact news within proximity windows")
        print(f"   2. Economic calendar integration may need adjustment")
        print(f"   3. News proximity thresholds might be too strict")
        
        # Analyze the "quiet" pattern
        quiet_data = news_details['quiet']
        
        print(f"\n📋 QUIET PERIOD ANALYSIS:")
        print(f"   Sample size: {len(quiet_data)} events")
        print(f"   Sweep success: {news_stats['quiet'][0]}/{news_stats['quiet'][1]} = {news_stats['quiet'][0]/news_stats['quiet'][1]*100:.1f}%")
        
        # Time distribution of quiet events
        hours = []
        for event_data in quiet_data:
            timestamp_et = event_data['event'].get('timestamp_et', '')
            if timestamp_et:
                try:
                    from datetime import datetime
                    dt_str = timestamp_et.replace(' ET', '')
                    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                    hours.append(dt.hour)
                except:
                    pass
        
        if hours:
            hour_dist = Counter(hours)
            print(f"   Time distribution: {dict(sorted(hour_dist.items()))}")
        
        # Day distribution 
        days = []
        for event_data in quiet_data:
            day_context = event_data['event'].get('day_context', {})
            day = day_context.get('day_of_week', 'unknown')
            days.append(day)
        
        day_dist = Counter(days)
        print(f"   Day distribution: {dict(day_dist)}")
        
    else:
        print(f"\n📈 NEWS PROXIMITY IMPACT COMPARISON:")
        for bucket in ['high±120m', 'medium±60m', 'low±30m']:
            if bucket in news_stats:
                sweeps, total = news_stats[bucket]
                rate = (sweeps / total * 100) if total > 0 else 0
                print(f"   {bucket}: {rate:.1f}% success rate (n={total})")
    
    return news_stats, news_details

def investigate_news_calendar_integration():
    """Investigate why all events are classified as 'quiet'"""
    print(f"\n🔍 INVESTIGATING NEWS CALENDAR INTEGRATION")
    print("=" * 50)
    
    # Check if economic calendar data exists and is being used
    try:
        from economic_calendar_loader import EconomicCalendarIntegrator
        integrator = EconomicCalendarIntegrator()
        
        print(f"✓ Economic calendar integrator available")
        
        # Check calendar data
        calendar_data = getattr(integrator.loader, 'calendar_data', None)
        if calendar_data:
            print(f"✓ Calendar data loaded: {len(calendar_data)} events")
            
            # Sample a few calendar events
            sample_events = calendar_data[:3] if len(calendar_data) >= 3 else calendar_data
            print(f"📋 Sample calendar events:")
            for event in sample_events:
                print(f"   {event.get('time', 'N/A')} - {event.get('event', 'N/A')} ({event.get('impact', 'N/A')})")
        else:
            print(f"❌ No calendar data found")
            
    except ImportError:
        print(f"❌ Economic calendar integrator not available")
    
    # Check proximity calculation
    print(f"\n📐 NEWS PROXIMITY THRESHOLDS:")
    print(f"   high±120m: Events within 2 hours of high impact news")
    print(f"   medium±60m: Events within 1 hour of medium impact news") 
    print(f"   low±30m: Events within 30 minutes of low impact news")
    print(f"   quiet: All other periods")

if __name__ == "__main__":
    news_stats, news_details = analyze_news_impact()
    investigate_news_calendar_integration()