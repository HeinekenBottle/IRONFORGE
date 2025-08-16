# IRONFORGE Workflow Guide
**Daily Workflows and Best Practices for Archaeological Discovery**

---

## 🎯 Overview

This guide provides structured workflows for using IRONFORGE in daily market analysis, from morning preparation through end-of-day review. These workflows are designed to maximize the archaeological discovery value while maintaining efficient processing and high-quality results.

**Workflow Philosophy**: Systematic, repeatable processes that leverage IRONFORGE's archaeological discovery capabilities for consistent market analysis and pattern recognition.

---

## 🌅 Morning Preparation Workflow

### Daily Market Preparation (10 minutes)
Start each trading day with comprehensive archaeological analysis:

```python
from ironforge.analysis.daily_discovery_workflows import morning_prep
from ironforge.analysis.pattern_intelligence import analyze_market_intelligence
from datetime import datetime

def daily_morning_routine():
    """Complete morning preparation workflow"""
    print(f"🌅 IRONFORGE Morning Preparation - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    # 1. Morning market analysis (5 minutes)
    print("📊 Running morning market analysis...")
    morning_analysis = morning_prep(days_back=7)
    
    # Display key insights
    print(f"Strength Score: {morning_analysis['strength_score']:.2f}/1.0")
    print(f"Confidence Level: {morning_analysis['confidence_level']}")
    print(f"Current Regime: {morning_analysis['current_regime']}")
    
    # 2. Dominant pattern identification
    print("\n🔥 Dominant Pattern Types:")
    for i, pattern_type in enumerate(morning_analysis['dominant_patterns'][:3], 1):
        print(f"  {i}. {pattern_type}")
    
    # 3. Cross-session signals
    print("\n🔗 Cross-Session Signals:")
    for signal in morning_analysis['cross_session_signals'][:3]:
        print(f"  • {signal}")
    
    # 4. Session focus recommendations
    print(f"\n💡 Focus Session: {morning_analysis.get('session_focus', 'Multiple sessions')}")
    
    # 5. Trading insights
    print("\n📈 Archaeological Insights:")
    for insight in morning_analysis['trading_insights'][:3]:
        print(f"  • {insight}")
    
    return morning_analysis

# Run morning routine
morning_results = daily_morning_routine()
```

### Market Intelligence Analysis
Enhance morning preparation with advanced intelligence:

```python
def enhanced_morning_intelligence():
    """Enhanced morning analysis with market intelligence"""
    
    # Run comprehensive intelligence analysis
    intel_results = analyze_market_intelligence()
    
    print("\n🧠 Market Intelligence Analysis:")
    print("=" * 40)
    
    # Pattern trends
    if 'pattern_trends' in intel_results:
        print("📈 Pattern Trends (Statistically Significant):")
        for pattern_type, trend in intel_results['pattern_trends'].items():
            if trend.get('significance', 1.0) < 0.05:  # p < 0.05
                print(f"  {pattern_type}: {trend['description']}")
                print(f"    Trend Strength: {trend['trend_strength']:.2f}")
    
    # Market regimes
    if 'market_regimes' in intel_results:
        print("\n🏛️ Current Market Regimes:")
        for regime in intel_results['market_regimes'][:2]:
            print(f"  {regime['regime_name']}: {len(regime['sessions'])} sessions")
            print(f"    Characteristics: {', '.join(regime['characteristic_patterns'][:3])}")
    
    return intel_results

# Run enhanced intelligence
intel_results = enhanced_morning_intelligence()
```

---

## 🎯 Session Hunting Workflow

### Real-Time Pattern Hunting
Hunt for patterns during active trading sessions:

```python
from ironforge.analysis.daily_discovery_workflows import hunt_patterns

def session_hunting_workflow(target_sessions=['NY_PM', 'LONDON']):
    """Real-time session pattern hunting"""
    session_results = {}
    
    for session_type in target_sessions:
        print(f"\n🎯 Hunting patterns in {session_type} session...")
        
        # Hunt patterns for this session
        result = hunt_patterns(session_type)
        session_results[session_type] = result
        
        # Display immediate results
        patterns_found = result.get('patterns_found', [])
        strength_indicators = result.get('strength_indicators', {})
        
        print(f"  Patterns Found: {len(patterns_found)}")
        print(f"  Avg Confidence: {strength_indicators.get('avg_confidence', 0):.2f}")
        print(f"  Quality Score: {strength_indicators.get('quality_score', 0):.2f}")
        
        # Show high-confidence patterns
        high_conf_patterns = [
            p for p in patterns_found 
            if p.get('confidence', 0) >= 0.8
        ]
        
        if high_conf_patterns:
            print(f"  🚨 High Confidence Patterns ({len(high_conf_patterns)}):")
            for pattern in high_conf_patterns[:3]:
                print(f"    • {pattern.get('description', 'Pattern')}")
                print(f"      Confidence: {pattern.get('confidence', 0):.2f}")
        
        # Immediate insights
        immediate_insights = result.get('immediate_insights', [])
        if immediate_insights:
            print(f"  💡 Immediate Insights:")
            for insight in immediate_insights[:2]:
                print(f"    • {insight}")
    
    return session_results

# Run session hunting
session_results = session_hunting_workflow(['NY_PM', 'LONDON', 'ASIA'])
```

### Pattern Alert System
Set up automated alerts for high-quality patterns:

```python
def setup_pattern_alerts(alert_threshold=0.8):
    """Set up automated pattern alerts"""
    
    def check_pattern_alerts(session_type):
        """Check for alert-worthy patterns"""
        result = hunt_patterns(session_type)
        patterns = result.get('patterns_found', [])
        
        # Find high-confidence patterns
        alert_patterns = [
            p for p in patterns 
            if p.get('confidence', 0) >= alert_threshold
        ]
        
        if alert_patterns:
            print(f"🚨 PATTERN ALERT - {session_type}")
            print(f"Found {len(alert_patterns)} high-confidence patterns")
            
            for pattern in alert_patterns:
                arch_sig = pattern.get('archaeological_significance', {})
                semantic_context = pattern.get('semantic_context', {})
                
                print(f"\nPattern: {pattern.get('description', 'Unknown')}")
                print(f"Confidence: {pattern.get('confidence', 0):.2f}")
                print(f"Archaeological Value: {arch_sig.get('archaeological_value', 'unknown')}")
                print(f"Event Types: {', '.join(semantic_context.get('event_types', []))}")
                
                # Suggested actions
                if arch_sig.get('archaeological_value') == 'high_archaeological_value':
                    print("🎯 Action: High priority - investigate immediately")
                elif pattern.get('confidence', 0) >= 0.9:
                    print("⚡ Action: Very high confidence - monitor closely")
        
        return alert_patterns
    
    # Check alerts for active sessions
    active_sessions = ['NY_AM', 'NY_PM', 'LONDON']
    all_alerts = {}
    
    for session in active_sessions:
        alerts = check_pattern_alerts(session)
        if alerts:
            all_alerts[session] = alerts
    
    return all_alerts

# Set up and run alerts
alerts = setup_pattern_alerts(alert_threshold=0.75)
```

---

## 🔍 Analysis Workflow

### Cross-Session Analysis
Analyze patterns across multiple sessions for deeper insights:

```python
from ironforge.analysis.pattern_intelligence import find_similar_patterns

def cross_session_analysis_workflow():
    """Comprehensive cross-session pattern analysis"""
    print("🔗 Cross-Session Analysis Workflow")
    print("=" * 40)
    
    # 1. Find similar patterns across sessions
    session_types = ['NY_AM', 'NY_PM', 'LONDON', 'ASIA']
    cross_session_insights = {}
    
    for session_type in session_types:
        print(f"\n📊 Analyzing {session_type} patterns...")
        
        # Find similar patterns
        similar_patterns = find_similar_patterns(
            session_type, 
            similarity_threshold=0.7
        )
        
        cross_session_insights[session_type] = {
            'similar_patterns': similar_patterns,
            'pattern_count': len(similar_patterns),
            'avg_similarity': sum(p.get('similarity_score', 0) for p in similar_patterns) / len(similar_patterns) if similar_patterns else 0
        }
        
        print(f"  Similar Patterns Found: {len(similar_patterns)}")
        if similar_patterns:
            print(f"  Avg Similarity: {cross_session_insights[session_type]['avg_similarity']:.2f}")
            
            # Show top similar patterns
            top_similar = sorted(similar_patterns, key=lambda p: p.get('similarity_score', 0), reverse=True)[:3]
            for i, pattern in enumerate(top_similar, 1):
                print(f"    {i}. {pattern.get('description', 'Pattern')} (similarity: {pattern.get('similarity_score', 0):.2f})")
    
    # 2. Identify cross-session themes
    print("\n🎨 Cross-Session Themes:")
    all_patterns = []
    for insights in cross_session_insights.values():
        all_patterns.extend(insights['similar_patterns'])
    
    # Group by semantic events
    event_themes = {}
    for pattern in all_patterns:
        semantic_context = pattern.get('semantic_context', {})
        event_types = semantic_context.get('event_types', [])
        
        for event_type in event_types:
            if event_type not in event_themes:
                event_themes[event_type] = 0
            event_themes[event_type] += 1
    
    # Display top themes
    sorted_themes = sorted(event_themes.items(), key=lambda x: x[1], reverse=True)
    for theme, count in sorted_themes[:5]:
        print(f"  {theme}: {count} patterns")
    
    return cross_session_insights

# Run cross-session analysis
cross_session_results = cross_session_analysis_workflow()
```

### Pattern Evolution Tracking
Track how patterns evolve throughout the day:

```python
def pattern_evolution_workflow():
    """Track pattern evolution throughout trading day"""
    print("📈 Pattern Evolution Tracking")
    print("=" * 35)
    
    # Define session sequence
    session_sequence = ['ASIA', 'LONDON', 'NY_AM', 'NY_PM']
    evolution_data = {}
    
    for session in session_sequence:
        # Hunt patterns for this session
        result = hunt_patterns(session)
        patterns = result.get('patterns_found', [])
        
        evolution_data[session] = {
            'pattern_count': len(patterns),
            'avg_confidence': sum(p.get('confidence', 0) for p in patterns) / len(patterns) if patterns else 0,
            'dominant_events': get_dominant_events(patterns),
            'quality_distribution': get_quality_distribution(patterns)
        }
    
    # Analyze evolution
    print("Session Evolution Analysis:")
    for session in session_sequence:
        data = evolution_data[session]
        print(f"\n{session}:")
        print(f"  Patterns: {data['pattern_count']}")
        print(f"  Avg Confidence: {data['avg_confidence']:.2f}")
        print(f"  Dominant Events: {', '.join(data['dominant_events'][:3])}")
        print(f"  High Quality: {data['quality_distribution'].get('high', 0)}")
    
    # Identify evolution trends
    pattern_counts = [evolution_data[s]['pattern_count'] for s in session_sequence]
    confidence_trend = [evolution_data[s]['avg_confidence'] for s in session_sequence]
    
    print(f"\n📊 Evolution Trends:")
    print(f"  Pattern Count Trend: {analyze_trend(pattern_counts)}")
    print(f"  Confidence Trend: {analyze_trend(confidence_trend)}")
    
    return evolution_data

def get_dominant_events(patterns):
    """Get dominant event types from patterns"""
    event_counts = {}
    for pattern in patterns:
        semantic_context = pattern.get('semantic_context', {})
        event_types = semantic_context.get('event_types', [])
        for event_type in event_types:
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    return sorted(event_counts.keys(), key=lambda x: event_counts[x], reverse=True)

def get_quality_distribution(patterns):
    """Get quality distribution of patterns"""
    distribution = {'high': 0, 'medium': 0, 'low': 0}
    
    for pattern in patterns:
        arch_sig = pattern.get('archaeological_significance', {})
        arch_value = arch_sig.get('archaeological_value', 'low_archaeological_value')
        
        if 'high' in arch_value:
            distribution['high'] += 1
        elif 'medium' in arch_value:
            distribution['medium'] += 1
        else:
            distribution['low'] += 1
    
    return distribution

def analyze_trend(values):
    """Simple trend analysis"""
    if len(values) < 2:
        return "insufficient data"
    
    increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
    decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
    
    if increases > decreases:
        return "increasing"
    elif decreases > increases:
        return "decreasing"
    else:
        return "stable"

# Run evolution tracking
evolution_results = pattern_evolution_workflow()
```

---

## 📊 End-of-Day Review Workflow

### Daily Summary and Analysis
Comprehensive end-of-day review and preparation for next day:

```python
def end_of_day_review():
    """Comprehensive end-of-day review workflow"""
    print("🌙 End-of-Day Review - IRONFORGE Archaeological Summary")
    print("=" * 60)
    
    # 1. Daily pattern summary
    print("📊 Daily Pattern Summary:")
    
    # Get all session results from the day
    all_sessions = ['ASIA', 'LONDON', 'NY_AM', 'NY_PM']
    daily_summary = {
        'total_patterns': 0,
        'high_quality_patterns': 0,
        'avg_confidence': 0,
        'dominant_themes': {},
        'session_performance': {}
    }
    
    total_confidence = 0
    all_patterns = []
    
    for session in all_sessions:
        result = hunt_patterns(session)
        patterns = result.get('patterns_found', [])
        all_patterns.extend(patterns)
        
        # Session performance
        session_high_quality = sum(
            1 for p in patterns 
            if p.get('archaeological_significance', {}).get('archaeological_value') == 'high_archaeological_value'
        )
        
        session_avg_conf = sum(p.get('confidence', 0) for p in patterns) / len(patterns) if patterns else 0
        
        daily_summary['session_performance'][session] = {
            'pattern_count': len(patterns),
            'high_quality': session_high_quality,
            'avg_confidence': session_avg_conf
        }
        
        daily_summary['total_patterns'] += len(patterns)
        daily_summary['high_quality_patterns'] += session_high_quality
        total_confidence += sum(p.get('confidence', 0) for p in patterns)
    
    # Calculate overall metrics
    if daily_summary['total_patterns'] > 0:
        daily_summary['avg_confidence'] = total_confidence / daily_summary['total_patterns']
        daily_summary['quality_rate'] = daily_summary['high_quality_patterns'] / daily_summary['total_patterns']
    
    # Display summary
    print(f"  Total Patterns Discovered: {daily_summary['total_patterns']}")
    print(f"  High Quality Patterns: {daily_summary['high_quality_patterns']}")
    print(f"  Quality Rate: {daily_summary.get('quality_rate', 0):.1%}")
    print(f"  Average Confidence: {daily_summary['avg_confidence']:.2f}")
    
    # 2. Session performance comparison
    print(f"\n📈 Session Performance:")
    for session, perf in daily_summary['session_performance'].items():
        print(f"  {session}: {perf['pattern_count']} patterns, {perf['avg_confidence']:.2f} avg conf, {perf['high_quality']} high quality")
    
    # 3. Dominant themes analysis
    print(f"\n🎨 Dominant Themes Today:")
    theme_counts = {}
    for pattern in all_patterns:
        semantic_context = pattern.get('semantic_context', {})
        event_types = semantic_context.get('event_types', [])
        for event_type in event_types:
            theme_counts[event_type] = theme_counts.get(event_type, 0) + 1
    
    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    for theme, count in sorted_themes[:5]:
        print(f"  {theme}: {count} occurrences")
    
    # 4. Next day preparation
    print(f"\n🔮 Next Day Preparation:")
    
    # Identify continuing themes
    continuing_themes = [theme for theme, count in sorted_themes[:3] if count >= 3]
    if continuing_themes:
        print(f"  Watch for continuation of: {', '.join(continuing_themes)}")
    
    # Session focus recommendation
    best_session = max(daily_summary['session_performance'].items(), 
                      key=lambda x: x[1]['avg_confidence'])
    print(f"  Focus session tomorrow: {best_session[0]} (best performance today)")
    
    # Quality trends
    if daily_summary.get('quality_rate', 0) > 0.3:
        print(f"  Quality trend: Strong (maintain current approach)")
    else:
        print(f"  Quality trend: Moderate (consider adjusting thresholds)")
    
    return daily_summary

# Run end-of-day review
daily_summary = end_of_day_review()
```

---

## 🔄 Automated Workflow Integration

### Scheduled Workflow Automation
Set up automated workflows for consistent daily analysis:

```python
import schedule
import time
from datetime import datetime

def setup_automated_workflows():
    """Set up automated daily workflows"""
    
    # Morning preparation at 8:00 AM
    schedule.every().day.at("08:00").do(daily_morning_routine)
    
    # Session hunting every 2 hours during market hours
    schedule.every(2).hours.do(lambda: session_hunting_workflow(['NY_PM', 'LONDON']))
    
    # End-of-day review at 6:00 PM
    schedule.every().day.at("18:00").do(end_of_day_review)
    
    print("✅ Automated workflows scheduled:")
    print("  • Morning prep: 8:00 AM daily")
    print("  • Session hunting: Every 2 hours")
    print("  • End-of-day review: 6:00 PM daily")

def run_workflow_scheduler():
    """Run the workflow scheduler"""
    setup_automated_workflows()
    
    print("🤖 IRONFORGE Workflow Scheduler Running...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n🛑 Workflow scheduler stopped")

# Uncomment to run automated workflows
# run_workflow_scheduler()
```

---

## ✅ Workflow Best Practices

### Daily Routine Checklist
- [ ] **Morning Prep** (10 min): Run morning analysis, identify focus areas
- [ ] **Session Hunting** (ongoing): Monitor active sessions for high-confidence patterns
- [ ] **Cross-Session Analysis** (mid-day): Analyze pattern relationships and evolution
- [ ] **Pattern Alerts** (ongoing): Respond to high-quality pattern alerts
- [ ] **End-of-Day Review** (15 min): Summarize findings, prepare for next day

### Quality Assurance
- Monitor authenticity scores (target >87/100)
- Track duplication rates (keep <25%)
- Validate semantic context preservation
- Review cross-session pattern consistency

### Performance Optimization
- Use lazy loading for efficient resource utilization
- Enable caching for repeated operations
- Process sessions in appropriate batch sizes
- Monitor memory usage during large analyses

---

*These workflows provide structured approaches to maximize IRONFORGE's archaeological discovery capabilities in daily market analysis. For technical implementation details, see the [API Reference](API_REFERENCE.md) and [User Guide](USER_GUIDE.md).*
