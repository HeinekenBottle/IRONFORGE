#!/usr/bin/env python3
"""
High-Performance Combination Analysis - Finding >70% Sweep Rate Combos
Explores multi-dimensional combinations to discover high-performance patterns
"""

from liquidity_htf_analyzer import LiquidityHTFAnalyzer
from enhanced_statistical_framework import EnhancedStatisticalAnalyzer
from e_refresh_analysis import calculate_wilson_ci
import json
from collections import defaultdict, Counter
from datetime import datetime
import pytz

def analyze_high_performance_combos():
    """Find all combinations that lead to >70% sweep rates"""
    print("🎯 HIGH-PERFORMANCE COMBINATION ANALYSIS")
    print("=" * 60)
    print("🔍 Searching for combinations with >70% sweep rates")
    print("📊 Minimum sample size: n≥3 for reliable statistics")
    print("=" * 60)
    
    analyzer = LiquidityHTFAnalyzer()
    
    # Load enhanced sessions
    enhanced_sessions = analyzer.load_enhanced_sessions()
    print(f"📁 Loaded {len(enhanced_sessions)} enhanced sessions")
    
    # Multi-dimensional combination tracking
    combinations = {
        'day_only': defaultdict(lambda: [0, 0]),
        'news_only': defaultdict(lambda: [0, 0]),
        'session_only': defaultdict(lambda: [0, 0]),
        'day_x_news': defaultdict(lambda: [0, 0]),
        'day_x_session': defaultdict(lambda: [0, 0]),
        'news_x_session': defaultdict(lambda: [0, 0]),
        'time_hour': defaultdict(lambda: [0, 0]),
        'day_x_hour': defaultdict(lambda: [0, 0]),
        'triple_combo': defaultdict(lambda: [0, 0])
    }
    
    # Event-level analysis for time patterns
    all_events = []
    
    for session in enhanced_sessions:
        session_data = session['data']
        events = session_data.get('events', [])
        trading_day = session_data.get('session_info', {}).get('trading_day', '2025-08-01')
        
        # Find RD@40 events
        rd40_events = [e for e in events 
                      if e.get('dimensional_relationship') == 'dimensional_destiny_40pct']
        
        for rd40_event in rd40_events:
            # Extract contexts
            day_context = rd40_event.get('day_context', {}).get('day_of_week', 'unknown')
            news_context = rd40_event.get('news_context', {}).get('news_bucket', 'quiet')
            session_context = rd40_event.get('session_context', {}).get('session_type', 'OTHER')
            
            # Extract time
            hour_context = 'unknown'
            timestamp_et = rd40_event.get('timestamp_et', '')
            if timestamp_et:
                try:
                    dt_str = timestamp_et.replace(' ET', '')
                    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                    hour_context = f"{dt.hour:02d}h"
                except:
                    pass
            
            # Analyze liquidity sweeps
            liquidity_levels = analyzer.calculate_liquidity_levels(session_data, trading_day)
            sweeps = analyzer.detect_liquidity_sweeps(rd40_event, events, liquidity_levels)
            
            # Check for sweeps within 90 minutes
            has_sweep = False
            for sweep in sweeps:
                if sweep.time_to_sweep_mins <= 90:
                    has_sweep = True
                    break
            
            # Update all combination counters
            # Single dimensions
            combinations['day_only'][day_context][1] += 1
            combinations['news_only'][news_context][1] += 1
            combinations['session_only'][session_context][1] += 1
            combinations['time_hour'][hour_context][1] += 1
            
            # Two-way combinations
            combinations['day_x_news'][f"{day_context}×{news_context}"][1] += 1
            combinations['day_x_session'][f"{day_context}×{session_context}"][1] += 1
            combinations['news_x_session'][f"{news_context}×{session_context}"][1] += 1
            combinations['day_x_hour'][f"{day_context}×{hour_context}"][1] += 1
            
            # Three-way combination
            combinations['triple_combo'][f"{day_context}×{news_context}×{session_context}"][1] += 1
            
            if has_sweep:
                combinations['day_only'][day_context][0] += 1
                combinations['news_only'][news_context][0] += 1
                combinations['session_only'][session_context][0] += 1
                combinations['time_hour'][hour_context][0] += 1
                combinations['day_x_news'][f"{day_context}×{news_context}"][0] += 1
                combinations['day_x_session'][f"{day_context}×{session_context}"][0] += 1
                combinations['news_x_session'][f"{news_context}×{session_context}"][0] += 1
                combinations['day_x_hour'][f"{day_context}×{hour_context}"][0] += 1
                combinations['triple_combo'][f"{day_context}×{news_context}×{session_context}"][0] += 1
            
            # Store event for time analysis
            all_events.append({
                'day': day_context,
                'news': news_context,
                'session': session_context,
                'hour': hour_context,
                'has_sweep': has_sweep,
                'timestamp_et': timestamp_et
            })
    
    print(f"🎯 Analyzed {len(all_events)} RD@40 events")
    
    # Find high-performance combinations (>70%, n≥3)
    high_performers = []
    
    for combo_type, combo_data in combinations.items():
        for combo_name, (successes, total) in combo_data.items():
            if total >= 3:  # Minimum sample size
                rate = (successes / total) * 100
                ci_lower, ci_upper, ci_width = calculate_wilson_ci(successes, total)
                
                if rate > 70.0:
                    high_performers.append({
                        'type': combo_type,
                        'combo': combo_name,
                        'successes': successes,
                        'total': total,
                        'rate': rate,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'ci_width': ci_width,
                        'conclusive': ci_width <= 30
                    })
    
    # Sort by rate, then by sample size
    high_performers.sort(key=lambda x: (x['rate'], x['total']), reverse=True)
    
    print(f"\n🏆 HIGH-PERFORMANCE COMBINATIONS (>70% sweep rate, n≥3)")
    print("=" * 80)
    
    if high_performers:
        for i, combo in enumerate(high_performers, 1):
            conclusive_flag = "✓" if combo['conclusive'] else "⚠"
            print(f"{i:2d}. {combo['type']:<15} | {combo['combo']:<25} | "
                  f"{combo['rate']:5.1f}% | {combo['successes']:2d}/{combo['total']:2d} | "
                  f"[{combo['ci_lower']:.0f}-{combo['ci_upper']:.0f}%] {conclusive_flag}")
        
        print(f"\n📈 PERFORMANCE INSIGHTS:")
        
        # Group by type for analysis
        by_type = defaultdict(list)
        for combo in high_performers:
            by_type[combo['type']].append(combo)
        
        for combo_type, combos in by_type.items():
            best_combo = combos[0]  # Highest rate
            print(f"   {combo_type}: Best = {best_combo['combo']} ({best_combo['rate']:.1f}%, n={best_combo['total']})")
        
        # Statistical quality assessment
        conclusive_count = sum(1 for combo in high_performers if combo['conclusive'])
        print(f"\n📊 STATISTICAL QUALITY:")
        print(f"   Total high-performers: {len(high_performers)}")
        print(f"   Conclusive (CI ≤30pp): {conclusive_count}")
        print(f"   Inconclusive (CI >30pp): {len(high_performers) - conclusive_count}")
        
    else:
        print("   No combinations found with >70% rate and n≥3")
    
    return high_performers, all_events

def detailed_time_analysis(all_events):
    """Analyze time-of-day patterns for high performers"""
    print(f"\n⏰ TIME-OF-DAY ANALYSIS FOR HIGH PERFORMERS")
    print("=" * 60)
    
    # Group by hour and calculate performance
    hour_performance = defaultdict(lambda: [0, 0])
    
    for event in all_events:
        hour = event['hour']
        if hour != 'unknown':
            hour_performance[hour][1] += 1
            if event['has_sweep']:
                hour_performance[hour][0] += 1
    
    # Find high-performing hours
    high_perf_hours = []
    for hour, (successes, total) in hour_performance.items():
        if total >= 2:  # Minimum 2 events
            rate = (successes / total) * 100
            if rate >= 70.0:
                high_perf_hours.append({
                    'hour': hour,
                    'rate': rate,
                    'successes': successes,
                    'total': total
                })
    
    high_perf_hours.sort(key=lambda x: x['rate'], reverse=True)
    
    if high_perf_hours:
        print(f"   High-performing hours (≥70%, n≥2):")
        for hour_data in high_perf_hours:
            print(f"   {hour_data['hour']}: {hour_data['rate']:.1f}% ({hour_data['successes']}/{hour_data['total']})")
    else:
        print(f"   No individual hours with ≥70% performance")
    
    return high_perf_hours

def analyze_pattern_drivers(high_performers, all_events):
    """Analyze what drives high performance in these combinations"""
    print(f"\n🔍 PATTERN DRIVER ANALYSIS")
    print("=" * 60)
    
    # Extract the highest performing combo for detailed analysis
    if high_performers:
        best_combo = high_performers[0]
        combo_type = best_combo['type']
        combo_name = best_combo['combo']
        
        print(f"🎯 Analyzing best performer: {combo_name} ({best_combo['rate']:.1f}%)")
        
        # Find events that match this combination
        matching_events = []
        
        for event in all_events:
            matches = False
            
            if combo_type == 'day_only':
                matches = event['day'] == combo_name
            elif combo_type == 'day_x_news':
                expected_day, expected_news = combo_name.split('×')
                matches = event['day'] == expected_day and event['news'] == expected_news
            elif combo_type == 'day_x_hour':
                expected_day, expected_hour = combo_name.split('×')
                matches = event['day'] == expected_day and event['hour'] == expected_hour
            # Add more combo types as needed
            
            if matches:
                matching_events.append(event)
        
        print(f"   Matching events: {len(matching_events)}")
        print(f"   Success rate: {len([e for e in matching_events if e['has_sweep']])}/{len(matching_events)}")
        
        # Time distribution for this combo
        hours = [e['hour'] for e in matching_events if e['hour'] != 'unknown']
        if hours:
            hour_dist = Counter(hours)
            print(f"   Time distribution: {dict(hour_dist)}")

if __name__ == "__main__":
    high_performers, all_events = analyze_high_performance_combos()
    
    # Create a learning opportunity for the user
    if high_performers:
        print(f"\n● **Learn by Doing**")
        print(f"**Context:** I've identified {len(high_performers)} combinations with >70% sweep rates. The analysis shows patterns across days, news contexts, sessions, and time-of-day combinations. We need to understand what makes these combinations successful.")
        print(f"**Your Task:** Choose one high-performing combination from the list above and analyze why it might be particularly effective. Consider market structure, participant behavior, and liquidity patterns.")
        print(f"**Guidance:** Look at the combination type (day, news, time), sample size, and confidence interval. Think about what market conditions or participant behaviors might drive the high success rate.")
    
    detailed_time_analysis(all_events)
    analyze_pattern_drivers(high_performers, all_events)