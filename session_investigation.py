#!/usr/bin/env python3
"""
Investigate session naming and timestamp inconsistencies
"""

from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
import pandas as pd

def investigate_session_inconsistencies():
    """Investigate the midnight session timestamp issue"""
    print("🔍 Investigating Session Naming and Timestamp Issues")
    print("=" * 60)
    
    engine = EnhancedTemporalQueryEngine()
    
    # Look at the MIDNIGHT sessions specifically
    midnight_sessions = [sid for sid in engine.sessions.keys() if sid.startswith('MIDNIGHT')]
    
    print(f"📊 Found {len(midnight_sessions)} MIDNIGHT sessions:")
    for session_id in midnight_sessions[:5]:  # Show first 5
        print(f"   {session_id}")
    
    if midnight_sessions:
        # Examine the first midnight session in detail
        first_midnight = midnight_sessions[0]
        session_data = engine.sessions[first_midnight]
        
        print(f"\n🔍 Detailed analysis of {first_midnight}:")
        print(f"   Total events: {len(session_data)}")
        print(f"   Timestamp range (ms): {session_data['t'].min()} - {session_data['t'].max()}")
        
        # Show raw timestamp values
        print(f"   First few timestamps (raw ms):")
        for i, t in enumerate(session_data['t'].head(10)):
            print(f"     {i}: {t}")
        
        # Check if timestamp is since epoch or since session start
        min_t = session_data['t'].min()
        max_t = session_data['t'].max()
        duration_ms = max_t - min_t
        duration_hours = duration_ms / (1000 * 60 * 60)
        
        print(f"\n📈 Session duration analysis:")
        print(f"   Duration (ms): {duration_ms}")
        print(f"   Duration (hours): {duration_hours:.2f}")
        print(f"   Duration (minutes): {duration_ms / (1000 * 60):.1f}")
        
        # Check if timestamps look like epoch time or relative time
        if min_t > 1000000000000:  # Epoch time in ms (after year 2001)
            print("   ✅ Timestamps appear to be epoch time (absolute)")
            import datetime
            start_time = datetime.datetime.fromtimestamp(min_t / 1000)
            end_time = datetime.datetime.fromtimestamp(max_t / 1000)
            print(f"   Session actual time range: {start_time} - {end_time}")
        else:
            print("   📝 Timestamps appear to be relative (since session start)")
        
        # Check for f8 spikes in this session
        if 'f8' in session_data.columns:
            f8_data = session_data['f8'].dropna()
            if len(f8_data) > 0:
                f8_95th = f8_data.quantile(0.95)
                spikes = session_data[session_data['f8'] > f8_95th]
                
                print(f"\n🚀 f8 Analysis for {first_midnight}:")
                print(f"   f8 95th percentile: {f8_95th:.3f}")
                print(f"   f8 spikes found: {len(spikes)}")
                
                if len(spikes) > 0:
                    for i, (_, spike) in enumerate(spikes.head(3).iterrows()):
                        print(f"   Spike {i+1}: t={spike['t']}, price={spike['price']:.2f}, f8={spike['f8']:.3f}")
    
    # Check all session types and their naming patterns
    print(f"\n📋 All Session Types Analysis:")
    session_types = {}
    session_dates = {}
    
    for session_id in engine.sessions.keys():
        parts = session_id.split('_')
        session_type = parts[0]
        session_types[session_type] = session_types.get(session_type, 0) + 1
        
        # Extract dates if present
        if len(parts) > 1:
            date_part = parts[1]
            if session_type not in session_dates:
                session_dates[session_type] = []
            session_dates[session_type].append(date_part)
    
    for session_type, count in sorted(session_types.items()):
        print(f"   {session_type}: {count} sessions")
        if session_type in session_dates:
            unique_dates = set(session_dates[session_type])
            print(f"     Dates: {sorted(list(unique_dates))[:5]}")  # Show first 5 dates
    
    # Look for patterns in session naming
    print(f"\n🔍 Session ID Pattern Analysis:")
    sample_sessions = list(engine.sessions.keys())[:10]
    for session_id in sample_sessions:
        session_data = engine.sessions[session_id]
        min_t = session_data['t'].min()
        max_t = session_data['t'].max()
        
        # Try to determine actual time if possible
        if min_t > 1000000000000:  # Epoch time
            import datetime
            actual_time = datetime.datetime.fromtimestamp(min_t / 1000)
            hour = actual_time.hour
            time_classification = "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening" if 18 <= hour < 24 else "night"
            print(f"   {session_id}: {actual_time.strftime('%Y-%m-%d %H:%M')} ({time_classification})")
        else:
            print(f"   {session_id}: relative timestamps (duration: {(max_t-min_t)/(1000*60):.1f} min)")


if __name__ == "__main__":
    investigate_session_inconsistencies()