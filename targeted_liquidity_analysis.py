#!/usr/bin/env python3
"""
Targeted Liquidity Analysis - Focus on actual liquidity events
Analyzing RD@40 patterns leading to liquidity_sweep, takeout events, and redelivery sequences
"""

import json
import glob
from datetime import datetime
from collections import defaultdict, Counter
import statistics

def load_session_data():
    """Load all adapted session data"""
    session_files = glob.glob("/Users/jack/IRONFORGE/data/adapted/adapted_*.json")
    sessions = {}
    
    for file_path in session_files:
        session_name = file_path.split('/')[-1].replace('adapted_', '').replace('.json', '')
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                sessions[session_name] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return sessions

def find_liquidity_events(events):
    """Find liquidity sweep, takeout, and related events"""
    liquidity_events = []
    
    for i, event in enumerate(events):
        event_type = event.get('type', '')
        original_type = event.get('original_type', '')
        
        # Look for explicit liquidity events
        if any(keyword in event_type.lower() for keyword in ['liquidity', 'sweep', 'takeout', 'take']):
            event['event_index'] = i
            liquidity_events.append(event)
        elif any(keyword in original_type.lower() for keyword in ['liquidity', 'sweep', 'takeout', 'take']):
            event['event_index'] = i
            liquidity_events.append(event)
        elif event.get('event_family') == 'liquidity_family':
            event['event_index'] = i
            liquidity_events.append(event)
    
    return liquidity_events

def find_rd40_events(events, session_range):
    """Find events near 40% archaeological zone (RD@40)"""
    if session_range == 0:
        return []
    
    rd40_events = []
    target_40_pct = 0.40
    tolerance = 0.025  # ±2.5% tolerance for 40% zone
    
    for i, event in enumerate(events):
        range_position = event.get('range_position', 0)
        
        # Check if event is near 40% zone
        if abs(range_position - target_40_pct) <= tolerance:
            event['event_index'] = i
            event['zone_proximity'] = abs(range_position - target_40_pct)
            rd40_events.append(event)
    
    return rd40_events

def find_redelivery_events(events):
    """Find redelivery events (FVG interactions)"""
    redelivery_events = []
    
    for i, event in enumerate(events):
        event_type = event.get('type', '')
        original_type = event.get('original_type', '')
        interaction_type = event.get('interaction_type', '')
        
        if 'redelivery' in event_type.lower() or 'redelivery' in original_type.lower():
            event['event_index'] = i
            redelivery_events.append(event)
        elif interaction_type == 'redelivery':
            event['event_index'] = i
            redelivery_events.append(event)
    
    return redelivery_events

def analyze_rd40_to_liquidity_patterns():
    """Analyze patterns from RD@40 events to liquidity being taken"""
    print("🌊 TARGETED RD@40 → LIQUIDITY ANALYSIS")
    print("=" * 60)
    
    sessions = load_session_data()
    
    total_sessions = len(sessions)
    sessions_with_rd40 = 0
    sessions_with_liquidity = 0
    sessions_with_both = 0
    
    rd40_to_liquidity_sequences = []
    timing_patterns = []
    liquidity_types = Counter()
    
    print(f"📊 Loading {total_sessions} sessions...")
    
    for session_name, session_data in sessions.items():
        events = session_data.get('events', [])
        if not events:
            continue
        
        # Calculate session range for RD@40 detection
        prices = [e.get('price_level', 0) for e in events if e.get('price_level')]
        if not prices:
            continue
        
        session_high = max(prices)
        session_low = min(prices)
        session_range = session_high - session_low
        
        # Find different event types
        rd40_events = find_rd40_events(events, session_range)
        liquidity_events = find_liquidity_events(events)
        redelivery_events = find_redelivery_events(events)
        
        if rd40_events:
            sessions_with_rd40 += 1
        if liquidity_events:
            sessions_with_liquidity += 1
        if rd40_events and liquidity_events:
            sessions_with_both += 1
        
        # Analyze sequences: RD@40 → Liquidity
        for rd40_event in rd40_events:
            rd40_time = parse_timestamp(rd40_event.get('timestamp', ''))
            rd40_index = rd40_event['event_index']
            
            # Look for liquidity events after RD@40
            for liquidity_event in liquidity_events:
                liq_time = parse_timestamp(liquidity_event.get('timestamp', ''))
                liq_index = liquidity_event['event_index']
                
                # Only consider liquidity events that happen after RD@40
                if liq_index > rd40_index and liq_time and rd40_time:
                    time_diff = calculate_time_diff(rd40_time, liq_time)
                    
                    if 0 <= time_diff <= 120:  # Within 2 hours
                        sequence = {
                            'session': session_name,
                            'rd40_event': rd40_event,
                            'liquidity_event': liquidity_event,
                            'time_diff_minutes': time_diff,
                            'event_gap': liq_index - rd40_index
                        }
                        rd40_to_liquidity_sequences.append(sequence)
                        timing_patterns.append(time_diff)
                        
                        # Count liquidity types
                        liq_type = liquidity_event.get('type', 'unknown')
                        liquidity_types[liq_type] += 1
    
    # Print analysis results
    print(f"\n📈 SESSION ANALYSIS:")
    print(f"   Total Sessions: {total_sessions}")
    print(f"   Sessions with RD@40 events: {sessions_with_rd40}")
    print(f"   Sessions with Liquidity events: {sessions_with_liquidity}")
    print(f"   Sessions with Both: {sessions_with_both}")
    
    print(f"\n🎯 RD@40 → LIQUIDITY SEQUENCES:")
    print(f"   Total sequences found: {len(rd40_to_liquidity_sequences)}")
    
    if timing_patterns:
        print(f"\n⏰ TIMING ANALYSIS:")
        print(f"   Average time RD@40 → Liquidity: {statistics.mean(timing_patterns):.1f} minutes")
        print(f"   Median time: {statistics.median(timing_patterns):.1f} minutes")
        print(f"   Min time: {min(timing_patterns):.1f} minutes")
        print(f"   Max time: {max(timing_patterns):.1f} minutes")
        
        # Timing distribution
        timing_buckets = {
            '0-15 min': sum(1 for t in timing_patterns if 0 <= t <= 15),
            '15-30 min': sum(1 for t in timing_patterns if 15 < t <= 30),
            '30-60 min': sum(1 for t in timing_patterns if 30 < t <= 60),
            '60+ min': sum(1 for t in timing_patterns if t > 60)
        }
        
        print(f"\n📊 TIMING DISTRIBUTION:")
        for bucket, count in timing_buckets.items():
            percentage = (count / len(timing_patterns)) * 100 if timing_patterns else 0
            print(f"   {bucket}: {count} events ({percentage:.1f}%)")
    
    if liquidity_types:
        print(f"\n🌊 LIQUIDITY EVENT TYPES:")
        for liq_type, count in liquidity_types.most_common(10):
            print(f"   {liq_type}: {count} events")
    
    return {
        'sequences': rd40_to_liquidity_sequences,
        'timing_patterns': timing_patterns,
        'liquidity_types': dict(liquidity_types),
        'session_stats': {
            'total_sessions': total_sessions,
            'sessions_with_rd40': sessions_with_rd40,
            'sessions_with_liquidity': sessions_with_liquidity,
            'sessions_with_both': sessions_with_both
        }
    }

def analyze_detailed_sequences():
    """Analyze detailed sequence patterns"""
    print("\n🔗 DETAILED SEQUENCE ANALYSIS")
    print("=" * 60)
    
    sessions = load_session_data()
    
    sequence_patterns = defaultdict(list)
    notable_sequences = []
    
    for session_name, session_data in sessions.items():
        events = session_data.get('events', [])
        if not events:
            continue
        
        # Find sequences with multiple event types
        for i in range(len(events) - 2):
            event1 = events[i]
            event2 = events[i + 1]
            event3 = events[i + 2]
            
            # Look for RD@40-like events followed by other patterns
            range_pos1 = event1.get('range_position', 0)
            
            if abs(range_pos1 - 0.40) <= 0.025:  # Near 40% zone
                type1 = event1.get('type', '')
                type2 = event2.get('type', '')
                type3 = event3.get('type', '')
                
                sequence_key = f"{type1} → {type2} → {type3}"
                sequence_info = {
                    'session': session_name,
                    'sequence': sequence_key,
                    'events': [event1, event2, event3],
                    'timestamps': [e.get('timestamp') for e in [event1, event2, event3]]
                }
                
                sequence_patterns[sequence_key].append(sequence_info)
                
                # Notable sequences with liquidity/takeout/redelivery
                if any(keyword in sequence_key.lower() for keyword in ['liquidity', 'takeout', 'redelivery', 'sweep']):
                    notable_sequences.append(sequence_info)
    
    print(f"📊 SEQUENCE PATTERN ANALYSIS:")
    print(f"   Unique 3-step sequences starting near RD@40: {len(sequence_patterns)}")
    print(f"   Notable sequences (with liquidity): {len(notable_sequences)}")
    
    # Show most common sequences
    print(f"\n🔝 MOST COMMON SEQUENCES:")
    sorted_sequences = sorted(sequence_patterns.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (sequence_key, occurrences) in enumerate(sorted_sequences[:10], 1):
        print(f"   {i}. {sequence_key}: {len(occurrences)} occurrences")
    
    # Show notable sequences
    if notable_sequences:
        print(f"\n⭐ NOTABLE LIQUIDITY SEQUENCES:")
        for i, seq in enumerate(notable_sequences[:5], 1):
            print(f"   {i}. Session: {seq['session']}")
            print(f"      Sequence: {seq['sequence']}")
            print(f"      Times: {' → '.join(seq['timestamps'])}")
    
    return sequence_patterns, notable_sequences

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object"""
    if not timestamp_str:
        return None
    try:
        return datetime.strptime(timestamp_str, '%H:%M:%S')
    except:
        return None

def calculate_time_diff(time1, time2):
    """Calculate time difference in minutes"""
    if not time1 or not time2:
        return None
    
    diff = time2 - time1
    return diff.total_seconds() / 60

def main():
    """Main analysis function"""
    print("🚀 TARGETED LIQUIDITY PATTERN ANALYSIS")
    print("🎯 Analyzing actual liquidity_sweep, takeout, and redelivery events")
    print("=" * 80)
    
    try:
        # Run targeted analyses
        liquidity_analysis = analyze_rd40_to_liquidity_patterns()
        sequence_patterns, notable_sequences = analyze_detailed_sequences()
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'analysis_type': 'Targeted_RD40_Liquidity_Analysis',
            'liquidity_analysis': liquidity_analysis,
            'sequence_patterns': {k: len(v) for k, v in sequence_patterns.items()},
            'notable_sequences_count': len(notable_sequences)
        }
        
        output_file = f"targeted_liquidity_analysis_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"\n✅ Targeted Liquidity Analysis Complete!")
        
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()