#!/usr/bin/env python3
"""
E Refresh — Day/News + Liquidity (explore-only)

Enhanced analysis with specific requirements:
- RD@40 tables by Day, News, Day×News with counts + % + CI (merge n<5; flag CI width>30pp)
- Liquidity sweeps within 90m and next FPFVG RD (coverage ≤100% + intensity per event)
- Top 5 minutes (ET) for post-RD events; explicitly test 14:35 ET ±3m
- Top 5 slices with 👍/👎 based on n≥20 & dominant path ≥60% & CI width ≤25pp
- One-liner: strongest day link, strongest news link (or "inconclusive")
"""

from liquidity_htf_analyzer import LiquidityHTFAnalyzer
from enhanced_statistical_framework import EnhancedStatisticalAnalyzer
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import pytz

def calculate_wilson_ci(successes, total, confidence=0.95):
    """Calculate Wilson confidence interval"""
    if total == 0:
        return 0, 0, 0
    
    p = successes / total
    z = 1.96  # 95% confidence
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * (p * (1 - p) / total + z**2 / (4 * total**2))**0.5 / denominator
    
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)
    ci_width = (ci_upper - ci_lower) * 100
    
    return ci_lower * 100, ci_upper * 100, ci_width

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

def analyze_fpfvg_coverage(analyzer, all_rd40_events, enhanced_sessions):
    """Analyze FPFVG RD coverage and intensity using available session data"""
    fpfvg_stats = {}
    
    # Build session file to data mapping
    session_map = {}
    for session in enhanced_sessions:
        session_map[session['file_path']] = session['data']
    
    for event in all_rd40_events:
        session_file = event['session_file']
        session_data = session_map.get(session_file)
        
        if not session_data:
            continue
            
        events = session_data.get('events', [])
        rd40_time = analyzer.parse_event_datetime(event, event.get('trading_day', '2025-08-01'))
        
        if not rd40_time:
            continue
        
        # Find next FPFVG events within reasonable timeframe
        fpfvg_events = []
        total_fpfvg_in_session = 0
        
        for e in events:
            if e.get('event_type') == 'fpfvg_rd':
                total_fpfvg_in_session += 1
                if e != event:  # Not the same event
                    event_time = analyzer.parse_event_datetime(e, event.get('trading_day', '2025-08-01'))
                    if event_time and event_time > rd40_time:
                        time_diff = (event_time - rd40_time).total_seconds() / 60
                        if time_diff <= 120:  # Within 2 hours
                            fpfvg_events.append({
                                'event': e,
                                'time_diff': time_diff,
                                'direction': e.get('rd_direction', 'unknown')
                            })
        
        # Calculate coverage (limit to ≤100% as requested)
        coverage = min(100.0, len(fpfvg_events) * 100 / max(1, total_fpfvg_in_session)) if total_fpfvg_in_session > 0 else 0
        intensity = len(fpfvg_events)  # Events per RD@40
        
        day_context = event.get('day_context', {}).get('day_of_week', 'unknown')
        
        if day_context not in fpfvg_stats:
            fpfvg_stats[day_context] = {'total_events': 0, 'total_coverage': 0, 'total_intensity': 0}
        
        fpfvg_stats[day_context]['total_events'] += 1
        fpfvg_stats[day_context]['total_coverage'] += coverage
        fpfvg_stats[day_context]['total_intensity'] += intensity
    
    return fpfvg_stats

def analyze_htf_coverage(analyzer, all_rd40_events, enhanced_sessions):
    """Analyze HTF level coverage and intensity using available session data"""
    htf_stats = {}
    
    # Build session file to data mapping
    session_map = {}
    for session in enhanced_sessions:
        session_map[session['file_path']] = session['data']
    
    for event in all_rd40_events:
        session_file = event['session_file']
        session_data = session_map.get(session_file)
        
        if not session_data:
            continue
            
        events = session_data.get('events', [])
        rd40_time = analyzer.parse_event_datetime(event, event.get('trading_day', '2025-08-01'))
        
        if not rd40_time:
            continue
        
        # Find HTF level touches within 90 minutes
        htf_events = []
        total_htf_levels = 0
        
        for e in events:
            if e.get('event_type') in ['htf_high', 'htf_low', 'htf_touch']:
                total_htf_levels += 1
                if e != event:  # Not the same event
                    event_time = analyzer.parse_event_datetime(e, event.get('trading_day', '2025-08-01'))
                    if event_time and event_time > rd40_time:
                        time_diff = (event_time - rd40_time).total_seconds() / 60
                        if time_diff <= 90:  # Within 90 minutes
                            htf_events.append({
                                'event': e,
                                'time_diff': time_diff,
                                'level_type': e.get('level_type', 'unknown'),
                                'timeframe': e.get('timeframe', 'unknown')
                            })
        
        # Calculate coverage (% of HTF levels reached)
        coverage = min(100.0, len(htf_events) * 100 / max(1, total_htf_levels)) if total_htf_levels > 0 else 0
        intensity = len(htf_events)  # HTF touches per RD@40
        
        day_context = event.get('day_context', {}).get('day_of_week', 'unknown')
        
        if day_context not in htf_stats:
            htf_stats[day_context] = {'total_events': 0, 'total_coverage': 0, 'total_intensity': 0}
        
        htf_stats[day_context]['total_events'] += 1
        htf_stats[day_context]['total_coverage'] += coverage
        htf_stats[day_context]['total_intensity'] += intensity
    
    return htf_stats

def find_top_slices_with_criteria(day_stats, news_stats, session_stats):
    """Find Top 5 slices with 👍/👎 criteria"""
    all_slices = []
    
    # Add day slices
    for stat in day_stats:
        criteria_met = (
            stat['n'] >= 20 and
            stat['rate'] >= 60 and
            stat['ci_width'] <= 25
        )
        all_slices.append({
            'type': 'Day',
            'context': stat['context'],
            'n': stat['n'],
            'rate': stat['rate'],
            'ci_width': stat['ci_width'],
            'criteria_met': criteria_met,
            'emoji': '👍' if criteria_met else '👎'
        })
    
    # Add news slices
    for stat in news_stats:
        criteria_met = (
            stat['n'] >= 20 and
            stat['rate'] >= 60 and
            stat['ci_width'] <= 25
        )
        all_slices.append({
            'type': 'News',
            'context': stat['context'],
            'n': stat['n'],
            'rate': stat['rate'],
            'ci_width': stat['ci_width'],
            'criteria_met': criteria_met,
            'emoji': '👍' if criteria_met else '👎'
        })
    
    # Add session slices
    for stat in session_stats:
        criteria_met = (
            stat['n'] >= 20 and
            stat['rate'] >= 60 and
            stat['ci_width'] <= 25
        )
        all_slices.append({
            'type': 'Session',
            'context': stat['context'],
            'n': stat['n'],
            'rate': stat['rate'],
            'ci_width': stat['ci_width'],
            'criteria_met': criteria_met,
            'emoji': '👍' if criteria_met else '👎'
        })
    
    # Sort by rate, then by n
    all_slices.sort(key=lambda x: (x['rate'], x['n']), reverse=True)
    
    return all_slices[:5]

def generate_e_refresh_report():
    """Generate the E refresh analysis report"""
    print("🔄 E Refresh — Day/News + Liquidity (explore-only)")
    print("=" * 70)
    
    analyzer = LiquidityHTFAnalyzer()
    stats = EnhancedStatisticalAnalyzer()
    
    # Load enhanced sessions
    enhanced_sessions = analyzer.load_enhanced_sessions()
    print(f"📁 Loaded {len(enhanced_sessions)} enhanced sessions")
    
    # Extract all RD@40 events with context
    all_rd40_events = []
    day_sweeps = defaultdict(lambda: [0, 0])  # [successes, total]
    news_sweeps = defaultdict(lambda: [0, 0])
    day_news_sweeps = defaultdict(lambda: [0, 0])  # Combined key
    session_sweeps = defaultdict(lambda: [0, 0])
    minute_events = defaultdict(int)
    
    for session in enhanced_sessions:
        session_data = session['data']
        events = session_data.get('events', [])
        trading_day = session_data.get('session_info', {}).get('trading_day', '2025-08-01')
        
        # Find RD@40 events
        rd40_events = [e for e in events 
                      if e.get('dimensional_relationship') == 'dimensional_destiny_40pct']
        
        for rd40_event in rd40_events:
            # Add context to RD@40 event
            enhanced_rd40 = {
                **rd40_event,
                'session_file': session['file_path'],
                'session_info': session_data.get('session_info', {}),
                'trading_day': trading_day
            }
            all_rd40_events.append(enhanced_rd40)
            
            # Extract contexts
            day_context = rd40_event.get('day_context', {}).get('day_of_week', 'unknown')
            news_context = rd40_event.get('news_context', {}).get('news_bucket', 'quiet')
            session_context = rd40_event.get('session_context', {}).get('session_type', 'OTHER')
            
            # Analyze liquidity sweeps within 90 minutes
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
            timestamp_et = rd40_event.get('timestamp_et', '')
            if timestamp_et:
                try:
                    dt_str = timestamp_et.replace(' ET', '')
                    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                    minute_key = f"{dt.hour:02d}:{dt.minute:02d}"
                    minute_events[minute_key] += 1
                except:
                    pass
    
    print(f"🎯 Found {len(all_rd40_events)} RD@40 events for analysis")
    
    # Generate tables
    day_stats = format_table_with_criteria(dict(day_sweeps), "Day of Week Analysis")
    news_stats = format_table_with_criteria(dict(news_sweeps), "News Proximity Analysis")
    
    # Day×News cross-tabulation (show only if meaningful)
    print(f"\n📋 Day×News Cross-Tabulation (n≥3 shown):")
    day_news_filtered = {k: v for k, v in day_news_sweeps.items() if v[1] >= 3}
    if day_news_filtered:
        day_news_stats = format_table_with_criteria(day_news_filtered, "Day×News Cross-Analysis")
    else:
        print("   No combinations with n≥3 found")
    
    session_stats = format_table_with_criteria(dict(session_sweeps), "Session Type Analysis")
    
    # FPFVG RD coverage analysis
    print(f"\n📊 FPFVG RD Coverage & Intensity Analysis:")
    print(f"   Coverage: % of subsequent FPFVG events reached within 120min (≤100% max)")
    print(f"   Intensity: Count of FPFVG events per RD@40 within timeframe")
    
    fpfvg_stats = analyze_fpfvg_coverage(analyzer, all_rd40_events, enhanced_sessions)
    
    if fpfvg_stats:
        for day, stats in fpfvg_stats.items():
            if stats['total_events'] > 0:
                avg_coverage = stats['total_coverage'] / stats['total_events']
                avg_intensity = stats['total_intensity'] / stats['total_events']
                print(f"   {day}: Coverage={avg_coverage:.1f}%, Intensity={avg_intensity:.1f} events/RD@40 (n={stats['total_events']})")
    else:
        print(f"   No FPFVG follow-through detected in dataset")
    
    # HTF Coverage & Intensity Analysis
    print(f"\n📊 HTF Level Coverage & Intensity Analysis:")
    print(f"   Coverage: % of HTF levels (H1/H4/D/W/M) reached within 90min")
    print(f"   Intensity: Count of HTF level touches per RD@40")
    
    htf_stats = analyze_htf_coverage(analyzer, all_rd40_events, enhanced_sessions)
    
    if htf_stats:
        for day, stats in htf_stats.items():
            if stats['total_events'] > 0:
                avg_coverage = stats['total_coverage'] / stats['total_events']
                avg_intensity = stats['total_intensity'] / stats['total_events']
                print(f"   {day}: Coverage={avg_coverage:.1f}%, Intensity={avg_intensity:.1f} levels/RD@40 (n={stats['total_events']})")
    else:
        print(f"   No HTF level interactions detected in dataset")
    
    # Top 5 minutes analysis
    print(f"\n⏰ Top 5 Minutes (ET) for Post-RD Events:")
    top_minutes = sorted(minute_events.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for i, (minute, count) in enumerate(top_minutes, 1):
        print(f"   {i}. {minute} ET: {count} events")
    
    # Test 14:35 ET ±3m explicitly
    target_minutes = ['14:32', '14:33', '14:34', '14:35', '14:36', '14:37', '14:38']
    target_count = sum(minute_events.get(m, 0) for m in target_minutes)
    total_events = len(all_rd40_events)
    target_pct = (target_count / total_events * 100) if total_events > 0 else 0
    
    print(f"\n🎯 14:35 ET ±3m Analysis:")
    print(f"   Events in window: {target_count}")
    print(f"   Percentage of total: {target_pct:.1f}%")
    print(f"   Window coverage: {', '.join(f'{m}:{minute_events.get(m, 0)}' for m in target_minutes)}")
    
    # Top 5 slices with 👍/👎 criteria  
    print(f"\n🏆 Top 5 Slices with Criteria Assessment:")
    print(f"   Criteria: n≥20 & dominant path ≥60% & CI width ≤25pp")
    
    top_slices = find_top_slices_with_criteria(day_stats, news_stats, session_stats)
    
    for i, slice_data in enumerate(top_slices, 1):
        criteria_text = f"n={slice_data['n']}, rate={slice_data['rate']:.1f}%, CI width={slice_data['ci_width']:.1f}pp"
        print(f"   {i}. {slice_data['type']}:{slice_data['context']} {slice_data['emoji']} ({criteria_text})")
    
    # One-liner conclusions
    print(f"\n🔗 One-Liner Conclusions:")
    
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
    
    print(f"\n✅ E Refresh Analysis Complete")
    print(f"   RD@40 events analyzed: {len(all_rd40_events)}")
    print(f"   Context tables: Day, News, Day×News with Wilson CI")
    print(f"   Liquidity sweeps within 90m tracked")
    print(f"   FPFVG coverage analysis completed")
    print(f"   Minute-of-day hotspots identified")

if __name__ == "__main__":
    generate_e_refresh_report()