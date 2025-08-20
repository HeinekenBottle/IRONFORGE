#!/usr/bin/env python3
"""
Experiment E — Liquidity & HTF Follow-Through (explore-only)

Comprehensive analysis of RD@40 follow-through using market structure:
- Liquidity sweeps: prior day/session/weekly H/L taps  
- FVG events: next FPFVG RD in same/opposite direction
- HTF taps: H1/H4/D/W/M level touches
- Context splits: Day, News, Session with Wilson CI
- Minute-of-day hotspots and chain analysis
"""

from liquidity_htf_analyzer import LiquidityHTFAnalyzer
from enhanced_statistical_framework import EnhancedStatisticalAnalyzer
import json

def generate_experiment_e_report():
    """Generate the complete Experiment E liquidity/HTF report"""
    print("🔄 Experiment E — Liquidity & HTF Follow-Through Analysis")
    print("=" * 70)
    
    analyzer = LiquidityHTFAnalyzer()
    stats = EnhancedStatisticalAnalyzer()
    
    # Load enhanced sessions
    enhanced_sessions = analyzer.load_enhanced_sessions()
    print(f"📁 Loaded {len(enhanced_sessions)} enhanced sessions")
    
    # Extract all RD@40 events with context
    all_rd40_events = []
    all_sweeps = []
    all_fvg_events = []  
    all_htf_taps = []
    
    for session in enhanced_sessions:
        session_data = session['data']
        events = session_data.get('events', [])
        trading_day = session_data.get('session_info', {}).get('trading_day', '2025-08-01')
        
        # Find RD@40 events
        rd40_events = [e for e in events 
                      if e.get('dimensional_relationship') == 'dimensional_destiny_40pct']
        
        for rd40_event in rd40_events:
            # Add context to RD@40 event (context is already in event, just add session metadata)
            enhanced_rd40 = {
                **rd40_event,  # RD@40 event already contains day_context and news_context
                'session_file': session['file_path'],
                'session_info': session_data.get('session_info', {})
            }
            all_rd40_events.append(enhanced_rd40)
            
            # Calculate liquidity levels and detect sweeps
            liquidity_levels = analyzer.calculate_liquidity_levels(session_data, trading_day)
            sweeps = analyzer.detect_liquidity_sweeps(rd40_event, events, liquidity_levels)
            all_sweeps.extend(sweeps)
            
            # Detect FVG events  
            fvg_events = analyzer.detect_fvg_events(rd40_event, events)
            all_fvg_events.extend(fvg_events)
            
            # Generate HTF levels and detect taps
            htf_levels = analyzer.generate_htf_levels(session_data, rd40_event.get('timestamp', ''))
            htf_taps = analyzer.detect_htf_taps(rd40_event, events, htf_levels)
            all_htf_taps.extend(htf_taps)
    
    print(f"🎯 Analysis Summary:")
    print(f"   Total RD@40 events: {len(all_rd40_events)}")
    print(f"   Liquidity sweeps: {len(all_sweeps)} ({len(all_sweeps)/len(all_rd40_events)*100:.1f}%)")
    print(f"   FVG follow-through: {len(all_fvg_events)} ({len(all_fvg_events)/len(all_rd40_events)*100:.1f}%)")  
    print(f"   HTF level taps: {len(all_htf_taps)} ({len(all_htf_taps)/len(all_rd40_events)*100:.1f}%)")
    
    # Generate context tables with Wilson CI
    print(f"\n📊 Context Analysis Tables:")
    
    # Day table
    day_table = generate_context_table(all_rd40_events, all_sweeps, 'day_of_week', stats)
    print_table("Day of Week", day_table)
    
    # News table  
    news_table = generate_context_table(all_rd40_events, all_sweeps, 'news_bucket', stats)
    print_table("News Proximity", news_table) 
    
    # Session table
    session_table = generate_context_table(all_rd40_events, all_sweeps, 'session_type', stats)
    print_table("Session Type", session_table)
    
    # Minute-of-day hotspots
    print(f"\n⏰ Minute-of-Day Hotspots:")
    hotspots = analyzer.analyze_minute_hotspots(all_rd40_events)
    top_5 = hotspots.get('top_5_minutes', [])
    for i, (minute, count) in enumerate(top_5, 1):
        print(f"   {i}. {minute} ET: {count} events")
    
    target_zone = hotspots.get('target_zone_14_35_pm3', {})
    print(f"   14:35 ET ±3m zone: {target_zone.get('total_events', 0)} events ({target_zone.get('percentage', 0):.1f}%)")
    
    # Top 5 insights with 👍/👎 using CI rules
    print(f"\n🔍 Top 5 Insights:")
    insights = generate_top_insights(day_table, news_table, session_table, all_sweeps, all_htf_taps)
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    # Links analysis
    print(f"\n🔗 Key Links:")
    best_links = analyze_best_links(day_table, news_table, session_table)
    for link_type, link in best_links.items():
        print(f"   {link_type}: {link}")
    
    # Check for missing fields
    print(f"\n⚠️  Missing Field Check:")
    check_missing_fields(all_sweeps, all_fvg_events, all_htf_taps)
    
    return {
        'total_rd40_events': len(all_rd40_events),
        'sweeps': len(all_sweeps),
        'fvg_events': len(all_fvg_events),
        'htf_taps': len(all_htf_taps),
        'context_tables': {
            'day': day_table,
            'news': news_table,
            'session': session_table
        },
        'hotspots': hotspots,
        'insights': insights
    }

def generate_context_table(rd40_events, sweeps, context_field, stats):
    """Generate context table with Wilson CI validation"""
    context_groups = {}
    
    # Group RD@40 events by context
    for event in rd40_events:
        context_value = extract_context_value(event, context_field)
        
        if context_value not in context_groups:
            context_groups[context_value] = {'rd40_count': 0, 'sweep_count': 0}
        context_groups[context_value]['rd40_count'] += 1
        
        # Check if this RD@40 event had a sweep
        rd40_timestamp = event.get('timestamp')
        event_sweeps = [s for s in sweeps if s.rd40_timestamp == rd40_timestamp]
        if event_sweeps:
            context_groups[context_value]['sweep_count'] += 1
    
    # Apply sample size merge rule (n<5 → "Other")
    table = {}
    other_group = {'rd40_count': 0, 'sweep_count': 0}
    
    for context, data in context_groups.items():
        if data['rd40_count'] >= 5:
            # Calculate percentage and Wilson CI
            n = data['rd40_count']
            successes = data['sweep_count']
            percentage = (successes / n * 100) if n > 0 else 0
            
            ci_lower, ci_upper = stats._wilson_confidence_interval(successes, n)
            ci_width = (ci_upper - ci_lower) * 100
            
            # Flag inconclusive if CI width > 30pp
            conclusive = ci_width <= 30
            
            table[context] = {
                'n': n,
                'sweeps': successes,
                'sweep_rate_pct': round(percentage, 1),
                'ci': f"[{ci_lower*100:.0f}-{ci_upper*100:.0f}%]",
                'conclusive': conclusive
            }
        else:
            # Merge into "Other"
            other_group['rd40_count'] += data['rd40_count']
            other_group['sweep_count'] += data['sweep_count']
    
    # Add "Other" if it has any data
    if other_group['rd40_count'] > 0:
        n = other_group['rd40_count']
        successes = other_group['sweep_count'] 
        percentage = (successes / n * 100) if n > 0 else 0
        
        if n >= 5:
            ci_lower, ci_upper = stats._wilson_confidence_interval(successes, n)
            ci_width = (ci_upper - ci_lower) * 100
            conclusive = ci_width <= 30
            
            table['Other'] = {
                'n': n,
                'sweeps': successes,
                'sweep_rate_pct': round(percentage, 1),
                'ci': f"[{ci_lower*100:.0f}-{ci_upper*100:.0f}%]",
                'conclusive': conclusive
            }
        else:
            table['Other'] = {
                'n': n,
                'sweeps': successes,
                'sweep_rate_pct': round(percentage, 1),
                'ci': 'insufficient_sample',
                'conclusive': False
            }
    
    return table

def extract_context_value(event, context_field):
    """Extract context value from RD@40 event"""
    if context_field == 'day_of_week':
        return event.get('day_context', {}).get('day_of_week', 'unknown')
    elif context_field == 'news_bucket':
        return event.get('news_context', {}).get('news_bucket', 'quiet')
    elif context_field == 'session_type':
        # Extract from file path
        file_path = event.get('session_file', '')
        if 'NY_AM' in file_path:
            return 'NY_AM'
        elif 'NY_PM' in file_path:
            return 'NY_PM'
        elif 'LONDON' in file_path:
            return 'LONDON'
        elif 'ASIA' in file_path:
            return 'ASIA'
        else:
            return 'OTHER'
    else:
        return 'unknown'

def print_table(title, table):
    """Print formatted table with Wilson CI"""
    print(f"\n📋 {title} Table:")
    if not table:
        print("   No data available")
        return
        
    print("   ┌─────────────┬─────┬────────┬─────────────┬─────────────┐")
    print("   │ Context     │  n  │ Sweeps │ Rate (%)    │ 95% CI      │")
    print("   ├─────────────┼─────┼────────┼─────────────┼─────────────┤")
    
    for context, data in table.items():
        context_str = context[:11]  # Truncate if too long
        n_str = str(data['n']).rjust(3)
        sweeps_str = str(data['sweeps']).rjust(6)
        rate_str = f"{data['sweep_rate_pct']}%".rjust(11)
        ci_str = data['ci'][:11] if data['ci'] != 'insufficient_sample' else 'n<5'
        
        # Add conclusive indicator
        conclusive_indicator = "✓" if data.get('conclusive', True) else "⚠"
        
        print(f"   │ {context_str:<11} │ {n_str} │ {sweeps_str} │ {rate_str} │ {ci_str} {conclusive_indicator} │")
    
    print("   └─────────────┴─────┴────────┴─────────────┴─────────────┘")
    print("   ✓ = conclusive (CI width ≤30pp), ⚠ = inconclusive")

def generate_top_insights(day_table, news_table, session_table, sweeps, htf_taps):
    """Generate top 5 insights with 👍/👎 based on CI rules"""
    insights = []
    
    # Find highest sweep rate with conclusive CI
    all_contexts = []
    for table_name, table in [('Day', day_table), ('News', news_table), ('Session', session_table)]:
        for context, data in table.items():
            if data.get('conclusive', False) and data['n'] >= 5:
                all_contexts.append({
                    'table': table_name,
                    'context': context,
                    'rate': data['sweep_rate_pct'],
                    'ci': data['ci'],
                    'n': data['n']
                })
    
    # Sort by sweep rate
    all_contexts.sort(key=lambda x: x['rate'], reverse=True)
    
    # Top insights
    if all_contexts:
        best = all_contexts[0]
        insights.append(f"{best['table']} winner: {best['context']} shows {best['rate']}% sweeps {best['ci']} (n={best['n']}) 👍")
    
    if len(all_contexts) > 1:
        worst = all_contexts[-1]  
        insights.append(f"{worst['table']} floor: {worst['context']} shows {worst['rate']}% sweeps {worst['ci']} (n={worst['n']}) 👎")
    
    # HTF insights
    htf_rate = (len(htf_taps) / len([c for c in all_contexts]) * 100) if all_contexts else 0
    if htf_rate > 50:
        insights.append(f"HTF magnet: {htf_rate:.1f}% RD@40 events reach HTF levels 👍")
    else:
        insights.append(f"HTF resistance: Only {htf_rate:.1f}% reach HTF levels 👎")
    
    # Sweep alignment analysis
    aligned_sweeps = sum(1 for s in sweeps if s.alignment == 'aligned')
    total_sweeps = len(sweeps)
    
    if total_sweeps > 0:
        alignment_pct = aligned_sweeps / total_sweeps * 100
        if alignment_pct > 60:
            insights.append(f"Directional flow: {alignment_pct:.1f}% sweeps aligned with RD@40 👍")
        elif alignment_pct > 40:
            insights.append(f"Mixed flow: {alignment_pct:.1f}% sweeps aligned with RD@40 ⚖️")
        else:
            insights.append(f"Counter flow: {alignment_pct:.1f}% sweeps aligned with RD@40 👎")
    
    # Sample size insight
    total_events = sum(data['n'] for table in [day_table, news_table, session_table] for data in table.values())
    insights.append(f"Statistical power: {total_events} total observations across contexts 👍")
    
    return insights[:5]

def analyze_best_links(day_table, news_table, session_table):
    """Analyze best links across context dimensions"""
    links = {}
    
    # Find best day link
    if day_table:
        best_day = max(day_table.items(), key=lambda x: x[1]['sweep_rate_pct'] if x[1].get('conclusive', False) else 0)
        links['Best day link'] = f"{best_day[0]} drives {best_day[1]['sweep_rate_pct']}% sweep rate {best_day[1]['ci']}"
    
    # Find best news link
    if news_table:
        best_news = max(news_table.items(), key=lambda x: x[1]['sweep_rate_pct'] if x[1].get('conclusive', False) else 0)
        links['Best news link'] = f"{best_news[0]} proximity shows {best_news[1]['sweep_rate_pct']}% follow-through {best_news[1]['ci']}"
    
    # Find best session link  
    if session_table:
        best_session = max(session_table.items(), key=lambda x: x[1]['sweep_rate_pct'] if x[1].get('conclusive', False) else 0)
        links['Session link'] = f"{best_session[0]} session yields {best_session[1]['sweep_rate_pct']}% liquidity sweeps {best_session[1]['ci']}"
    
    return links

def check_missing_fields(sweeps, fvg_events, htf_taps):
    """Check for missing fields in analysis"""
    missing = []
    
    if not sweeps:
        missing.append("sweeps: No liquidity sweep data detected")
    if not fvg_events:
        missing.append("FVG: No FVG follow-through events found")
    if not htf_taps:
        missing.append("HTF: No higher timeframe level taps detected")
    
    if missing:
        for field in missing:
            print(f"   Missing: {field}")
    else:
        print("   ✓ All analysis components have data")

def main():
    """Run complete Experiment E analysis"""
    results = generate_experiment_e_report()
    
    print(f"\n✅ Experiment E Complete")
    print(f"   RD@40 events analyzed: {results['total_rd40_events']}")
    print(f"   Liquidity patterns: {results['sweeps']} sweeps detected")
    print(f"   HTF interactions: {results['htf_taps']} level touches")
    print(f"   Context coverage: 3 tables (Day/News/Session) with Wilson CI")

if __name__ == "__main__":
    main()