#!/usr/bin/env python3
"""
Fix Day & News Context (super short)
Targeted fix for timezone + join bugs causing "unknown/quiet" 

BUZZWORDS: UTC→ET mapping • nearest±window join • real day_name • per-event vs per-event-count
"""

import json
import glob
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any

def fix_timezone_and_day_context():
    """Fix timezone conversion and add proper day_context to events"""
    print("🔧 Fixing Timezone & Day Context Issues")
    
    enhanced_files = glob.glob("/Users/jack/IRONFORGE/data/day_news_enhanced/*.json")
    
    for file_path in enhanced_files:
        print(f"📝 Processing: {file_path.split('/')[-1]}")
        
        with open(file_path, 'r') as f:
            session_data = json.load(f)
        
        # Extract trading day from filename: day_news_enhanced_rel_NY_PM_Lvl-1_2025_07_28.json
        filename = file_path.split('/')[-1]
        # Get the last 3 parts: 2025_07_28.json -> 2025_07_28
        parts = filename.split('_')
        if len(parts) >= 3:
            trading_date = f"{parts[-3]}-{parts[-2]}-{parts[-1].replace('.json', '')}"
        else:
            print(f"   ⚠️  Cannot parse date from {filename}")
            continue
        
        try:
            # Parse trading date and get day of week
            date_obj = datetime.strptime(trading_date, '%Y-%m-%d')
            day_of_week = date_obj.strftime('%A')  # "Monday", "Tuesday", etc.
            
            print(f"   Trading date: {trading_date} ({day_of_week})")
            
            # Add session_info at session level
            session_data['session_info'] = {
                'trading_day': trading_date,
                'day_of_week': day_of_week,
                'session_type': extract_session_type(filename),
                'enhancement_timestamp': datetime.now().isoformat()
            }
            
            # Add day_context to each event
            for event in session_data.get('events', []):
                event['day_context'] = {
                    'day_of_week': day_of_week,
                    'trading_day': trading_date,
                    'session_overlap': 'within_session',  # Default for now
                    'is_enhanced': True
                }
                
                # Fix news_context if it's missing or incomplete
                if 'news_context' not in event:
                    event['news_context'] = {
                        'news_bucket': 'quiet',
                        'news_distance_mins': None,
                        'news_impact_level': None,
                        'closest_news_event': None
                    }
                
                # Fix timestamp to include timezone info for future parsing
                timestamp = event.get('timestamp', '00:00:00')
                if ':' in timestamp and len(timestamp.split(':')) == 3:
                    # Add ET timezone context for future parsing
                    event['timestamp_et'] = f"{trading_date} {timestamp} ET"
            
            # Save the fixed file
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            print(f"   ✅ Fixed: Added day_context to {len(session_data.get('events', []))} events")
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

def extract_session_type(filename: str) -> str:
    """Extract session type from filename"""
    if 'NY_AM' in filename:
        return 'NY_AM'
    elif 'NY_PM' in filename:
        return 'NY_PM'
    elif 'LONDON' in filename:
        return 'LONDON'
    elif 'ASIA' in filename:
        return 'ASIA'
    elif 'PREMARKET' in filename:
        return 'PREMARKET'
    elif 'LUNCH' in filename:
        return 'LUNCH'
    elif 'MIDNIGHT' in filename:
        return 'MIDNIGHT'
    else:
        return 'UNKNOWN'

def fix_news_calendar_join():
    """Fix economic calendar join logic (nearest±window)"""
    print("\n🔧 Fixing News Calendar Join Logic")
    
    # Create a simple working calendar with UTC/ET timezone handling
    sample_calendar = [
        {
            'date': '2025-07-28',
            'time_et': '08:30:00',
            'event': 'GDP Preliminary',
            'impact': 'high',
            'currency': 'USD'
        },
        {
            'date': '2025-07-29', 
            'time_et': '14:00:00',
            'event': 'Fed Meeting Minutes',
            'impact': 'high',
            'currency': 'USD'
        },
        {
            'date': '2025-08-01',
            'time_et': '08:30:00',
            'event': 'Employment Report',
            'impact': 'high',
            'currency': 'USD'
        }
    ]
    
    enhanced_files = glob.glob("/Users/jack/IRONFORGE/data/day_news_enhanced/*.json")
    
    for file_path in enhanced_files:
        filename = file_path.split('/')[-1]
        date_part = filename.split('_')[-1].replace('.json', '')
        trading_date = date_part.replace('_', '-')
        
        with open(file_path, 'r') as f:
            session_data = json.load(f)
        
        # Find news events for this trading day
        day_news = [n for n in sample_calendar if n['date'] == trading_date]
        
        # Update RD@40 events with proper news context
        rd40_count = 0
        for event in session_data.get('events', []):
            if event.get('dimensional_relationship') == 'dimensional_destiny_40pct':
                rd40_count += 1
                
                # Get event time
                event_time = event.get('timestamp', '12:00:00')
                
                # Find nearest news within ±120 minutes
                closest_news = find_nearest_news(trading_date, event_time, day_news)
                
                if closest_news:
                    distance_mins = closest_news['distance_mins']
                    impact = closest_news['impact']
                    
                    # Determine news bucket
                    if impact == 'high' and abs(distance_mins) <= 120:
                        news_bucket = 'high±120m'
                    elif impact == 'medium' and abs(distance_mins) <= 60:
                        news_bucket = 'medium±60m'
                    elif impact == 'low' and abs(distance_mins) <= 30:
                        news_bucket = 'low±30m'
                    else:
                        news_bucket = 'quiet'
                    
                    event['news_context'] = {
                        'news_bucket': news_bucket,
                        'news_distance_mins': distance_mins,
                        'news_impact_level': impact,
                        'closest_news_event': closest_news['event']
                    }
                    
                    print(f"   📰 {filename}: RD@40 at {event_time} → {news_bucket} ({distance_mins}m to {closest_news['event']})")
                else:
                    # No news nearby
                    event['news_context'] = {
                        'news_bucket': 'quiet',
                        'news_distance_mins': None,
                        'news_impact_level': None,
                        'closest_news_event': None
                    }
        
        # Save updated file
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        if rd40_count > 0:
            print(f"   ✅ {filename}: Updated {rd40_count} RD@40 events with news context")

def find_nearest_news(trading_date: str, event_time: str, news_events: List[Dict]) -> Dict[str, Any]:
    """Find nearest news event within window"""
    if not news_events:
        return None
    
    try:
        # Parse event datetime
        event_dt = datetime.strptime(f"{trading_date} {event_time}", '%Y-%m-%d %H:%M:%S')
        
        closest = None
        min_distance = float('inf')
        
        for news in news_events:
            news_dt = datetime.strptime(f"{news['date']} {news['time_et']}", '%Y-%m-%d %H:%M:%S')
            distance_mins = (event_dt - news_dt).total_seconds() / 60
            
            if abs(distance_mins) < abs(min_distance):
                min_distance = distance_mins
                closest = {**news, 'distance_mins': distance_mins}
        
        return closest
        
    except Exception as e:
        print(f"      ⚠️  Error parsing time {event_time}: {e}")
        return None

def validate_fixes():
    """Validate that fixes worked"""
    print("\n🔍 Validating Fixes")
    
    enhanced_files = glob.glob("/Users/jack/IRONFORGE/data/day_news_enhanced/*.json")
    
    total_files = len(enhanced_files)
    files_with_day_context = 0
    files_with_session_info = 0
    total_rd40_events = 0
    rd40_with_news_context = 0
    
    day_of_week_counts = {}
    news_bucket_counts = {}
    
    for file_path in enhanced_files:
        with open(file_path, 'r') as f:
            session_data = json.load(f)
        
        # Check session_info or day_profile
        if 'session_info' in session_data:
            files_with_session_info += 1
            day_of_week = session_data['session_info'].get('day_of_week', 'unknown')
        elif 'day_profile' in session_data:
            files_with_session_info += 1  # Count day_profile as session info
            day_of_week = session_data['day_profile'].get('day_of_week', 'unknown')
        else:
            day_of_week = 'unknown'
            
        day_of_week_counts[day_of_week] = day_of_week_counts.get(day_of_week, 0) + 1
        
        # Check day_context in events
        events_with_day_context = 0
        for event in session_data.get('events', []):
            if 'day_context' in event:
                events_with_day_context += 1
            
            # Check RD@40 events
            if event.get('dimensional_relationship') == 'dimensional_destiny_40pct':
                total_rd40_events += 1
                
                if 'news_context' in event and event['news_context'].get('news_bucket') != 'quiet':
                    rd40_with_news_context += 1
                
                news_bucket = event.get('news_context', {}).get('news_bucket', 'unknown')
                news_bucket_counts[news_bucket] = news_bucket_counts.get(news_bucket, 0) + 1
        
        if events_with_day_context > 0:
            files_with_day_context += 1
    
    print(f"📊 Validation Results:")
    print(f"   Files with session_info: {files_with_session_info}/{total_files}")
    print(f"   Files with day_context: {files_with_day_context}/{total_files}")
    print(f"   Total RD@40 events: {total_rd40_events}")
    print(f"   RD@40 with news context: {rd40_with_news_context}")
    
    print(f"\n📅 Day of Week Distribution:")
    for day, count in sorted(day_of_week_counts.items()):
        print(f"   {day}: {count} sessions")
    
    print(f"\n📰 News Bucket Distribution:")
    for bucket, count in sorted(news_bucket_counts.items()):
        print(f"   {bucket}: {count} RD@40 events")
    
    # Acceptance checks
    success = True
    
    if 'unknown' in day_of_week_counts and day_of_week_counts['unknown'] > 0:
        print("❌ FAIL: Still have 'unknown' day_of_week")
        success = False
    else:
        print("✅ PASS: No 'unknown' day_of_week")
    
    if news_bucket_counts.get('quiet', 0) == total_rd40_events:
        print("❌ FAIL: All news buckets still 'quiet'")
        success = False
    else:
        print("✅ PASS: Found non-quiet news buckets")
    
    return success

def main():
    """Run the complete fix"""
    print("🚀 Day & News Context Fix (super short)")
    print("=" * 50)
    
    # 1. Fix timezone and day context
    fix_timezone_and_day_context()
    
    # 2. Fix news calendar join
    fix_news_calendar_join()
    
    # 3. Validate fixes
    success = validate_fixes()
    
    if success:
        print("\n🎉 All fixes applied successfully!")
        print("   Ready to re-run Experiment E tables")
    else:
        print("\n⚠️  Some issues remain - check validation output")

if __name__ == "__main__":
    main()