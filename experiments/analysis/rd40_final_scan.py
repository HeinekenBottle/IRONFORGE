#!/usr/bin/env python3
"""
Final RD@40 Scan: Complete Analysis with Fixed Classification
Generate proper Day/News/Matrix tables with gap split and session overlap
"""

import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from enhanced_statistical_framework import EnhancedStatisticalAnalyzer
import random

def extract_and_classify_rd40_cases():
    """Extract RD@40 cases with fixed outcome-based classification"""
    
    enhanced_files = glob.glob("/Users/jack/IRONFORGE/data/day_news_enhanced/day_news_*.json")
    rd40_cases = []
    
    print(f"📊 Processing {len(enhanced_files)} session files for RD@40 cases...")
    
    for file_path in enhanced_files:
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            
            session_info = session_data.get('session_info', {})
            day_profile = session_data.get('day_profile', {})
            events = session_data.get('events', [])
            
            session_start = session_info.get('start_time', '09:30:00')
            
            for i, event in enumerate(events):
                range_position = event.get('range_position', 0.5)
                
                # RD@40 zone filter
                if abs(range_position - 0.40) <= 0.025:
                    
                    # Calculate outcome-based path
                    path_info = calculate_outcome_path(event, events, i, session_start)
                    news_context = event.get('news_context', {})
                    
                    case = {
                        'session_file': file_path.split('/')[-1],
                        'day_of_week': day_profile.get('day_of_week', 'Unknown'),
                        'day_profile': day_profile.get('profile_name', 'unknown'),
                        'news_bucket': news_context.get('news_bucket', 'quiet'),
                        'news_distance_mins': news_context.get('news_distance_mins', 999),
                        'session_overlap': news_context.get('session_overlap', False),
                        'gap_age_days': event.get('gap_age_days', 0),
                        'range_position': range_position,
                        'timestamp': event.get('timestamp', ''),
                        'price_level': event.get('price_level', 0),
                        
                        # Path classification
                        'path': path_info['path'],
                        'hit_mid_time': path_info['hit_mid_time'],
                        'hit_80_time': path_info['hit_80_time'],
                        'reason': path_info['reason']
                    }
                    
                    rd40_cases.append(case)
                    
        except Exception as e:
            continue
    
    return rd40_cases

def calculate_outcome_path(rd40_event, all_events, rd40_index, session_start):
    """Calculate path based on hit_mid≤60m, hit_80≤90m rules"""
    
    rd40_timestamp = rd40_event.get('timestamp', '')
    rd40_minutes = timestamp_to_minutes(rd40_timestamp, session_start)
    
    if rd40_minutes is None:
        return {'path': 'E1', 'hit_mid_time': None, 'hit_80_time': None, 'reason': 'invalid_timestamp'}
    
    # Calculate session levels
    session_high = max(e.get('price_level', 0) for e in all_events if e.get('range_position', 0) > 0.9)
    session_low = min(e.get('price_level', 999999) for e in all_events if e.get('range_position', 0) < 0.1)
    session_range = session_high - session_low
    
    if session_range <= 0:
        return {'path': 'E1', 'hit_mid_time': None, 'hit_80_time': None, 'reason': 'no_range'}
    
    mid_level = session_low + (session_range * 0.5)
    level_80 = session_low + (session_range * 0.8)
    
    # Look for hits in subsequent events
    hit_mid_time = None
    hit_80_time = None
    
    subsequent_events = all_events[rd40_index + 1:]
    
    for event in subsequent_events:
        event_timestamp = event.get('timestamp', '')
        event_price = event.get('price_level', 0)
        event_minutes = timestamp_to_minutes(event_timestamp, session_start)
        
        if event_minutes is None:
            continue
            
        time_elapsed = event_minutes - rd40_minutes
        
        if time_elapsed > 120:  # Stop looking after 2 hours
            break
        
        # Check for mid hit (within 2% of mid level)
        if hit_mid_time is None and abs(event_price - mid_level) <= (session_range * 0.02):
            hit_mid_time = time_elapsed
        
        # Check for 80% hit
        if hit_80_time is None and event_price >= level_80:
            hit_80_time = time_elapsed
    
    # Apply classification rules
    if hit_mid_time is not None and hit_mid_time <= 60:
        return {
            'path': 'E2',
            'hit_mid_time': hit_mid_time,
            'hit_80_time': hit_80_time,
            'reason': f'hit_mid_at_{hit_mid_time:.0f}m'
        }
    elif hit_80_time is not None and hit_80_time <= 90:
        # Simplified pullback check - if it hit 80% within 90m, classify as E3
        return {
            'path': 'E3',
            'hit_mid_time': hit_mid_time,
            'hit_80_time': hit_80_time,
            'reason': f'hit_80_at_{hit_80_time:.0f}m'
        }
    else:
        return {
            'path': 'E1',
            'hit_mid_time': hit_mid_time,
            'hit_80_time': hit_80_time,
            'reason': 'neither_criteria_met'
        }

def timestamp_to_minutes(timestamp_str, session_start):
    """Convert HH:MM:SS to minutes since session start"""
    
    try:
        if not timestamp_str or not session_start:
            return None
            
        event_time = datetime.strptime(timestamp_str, '%H:%M:%S').time()
        start_time = datetime.strptime(session_start, '%H:%M:%S').time()
        
        base_date = datetime(2025, 1, 1)
        event_dt = datetime.combine(base_date, event_time)
        start_dt = datetime.combine(base_date, start_time)
        
        if event_dt < start_dt:
            event_dt += timedelta(days=1)
        
        return (event_dt - start_dt).total_seconds() / 60
        
    except Exception:
        return None

def generate_markdown_table(data, slice_key, title):
    """Generate markdown table with proper CI rules"""
    
    # Group by slice
    grouped = defaultdict(list)
    for case in data:
        key = case.get(slice_key, 'Unknown')
        grouped[key].append(case)
    
    # Apply sample size rules
    results = []
    other_cases = []
    
    for key, cases in grouped.items():
        n = len(cases)
        e1_count = sum(1 for c in cases if c['path'] == 'E1')
        e2_count = sum(1 for c in cases if c['path'] == 'E2')
        e3_count = sum(1 for c in cases if c['path'] == 'E3')
        
        if n >= 5:
            # Calculate percentages and Wilson CI for dominant path
            dominant_count = max(e1_count, e2_count, e3_count)
            dominant_path = ['E1', 'E2', 'E3'][np.argmax([e1_count, e2_count, e3_count])]
            
            analyzer = EnhancedStatisticalAnalyzer()
            ci_lower, ci_upper = analyzer._wilson_confidence_interval(dominant_count, n)
            ci_width = ci_upper - ci_lower
            
            results.append({
                'slice': key,
                'n': n,
                'e1_count': e1_count, 'e1_pct': (e1_count/n)*100,
                'e2_count': e2_count, 'e2_pct': (e2_count/n)*100,
                'e3_count': e3_count, 'e3_pct': (e3_count/n)*100,
                'dominant_path': dominant_path,
                'dominant_pct': (dominant_count/n)*100,
                'ci_lower': ci_lower*100,
                'ci_upper': ci_upper*100,
                'ci_width': ci_width,
                'inconclusive': ci_width > 0.30,
                'cases': cases
            })
        else:
            other_cases.extend(cases)
    
    # Add "Other" group if exists
    if other_cases:
        n = len(other_cases)
        e1_count = sum(1 for c in other_cases if c['path'] == 'E1')
        e2_count = sum(1 for c in other_cases if c['path'] == 'E2')
        e3_count = sum(1 for c in other_cases if c['path'] == 'E3')
        
        results.append({
            'slice': 'Other',
            'n': n,
            'e1_count': e1_count, 'e1_pct': 0,
            'e2_count': e2_count, 'e2_pct': 0,
            'e3_count': e3_count, 'e3_pct': 0,
            'dominant_path': 'counts_only',
            'dominant_pct': 0,
            'ci_lower': 0, 'ci_upper': 0, 'ci_width': 0,
            'inconclusive': False,
            'cases': other_cases
        })
    
    # Generate markdown
    lines = [f"### {title}", ""]
    lines.append("| Slice | n | E1 | E2 | E3 | Dominant | CI | Notes |")
    lines.append("|-------|---|----|----|----|---------|----|-------|")
    
    for result in sorted(results, key=lambda x: x['n'], reverse=True):
        slice_name = result['slice']
        n = result['n']
        
        if result['slice'] == 'Other':
            e1_str = str(result['e1_count'])
            e2_str = str(result['e2_count'])
            e3_str = str(result['e3_count'])
            dominant_str = "counts only"
            ci_str = "—"
        else:
            e1_str = f"{result['e1_count']} ({result['e1_pct']:.0f}%)"
            e2_str = f"{result['e2_count']} ({result['e2_pct']:.0f}%)"
            e3_str = f"{result['e3_count']} ({result['e3_pct']:.0f}%)"
            dominant_str = f"{result['dominant_path']} ({result['dominant_pct']:.0f}%)"
            ci_str = f"[{result['ci_lower']:.0f}–{result['ci_upper']:.0f}]"
        
        notes = "Inconclusive" if result['inconclusive'] else ""
        
        lines.append(f"| {slice_name} | {n} | {e1_str} | {e2_str} | {e3_str} | {dominant_str} | {ci_str} | {notes} |")
    
    return "\n".join(lines), results

def generate_day_news_matrix(data):
    """Generate Day × News matrix with CI validation"""
    
    matrix = defaultdict(list)
    for case in data:
        key = f"{case['day_of_week']}×{case['news_bucket']}"
        matrix[key].append(case)
    
    results = []
    
    for key, cases in matrix.items():
        n = len(cases)
        if n >= 3:  # Lower threshold for matrix cells
            e1_count = sum(1 for c in cases if c['path'] == 'E1')
            e2_count = sum(1 for c in cases if c['path'] == 'E2')
            e3_count = sum(1 for c in cases if c['path'] == 'E3')
            
            dominant_count = max(e1_count, e2_count, e3_count)
            dominant_path = ['E1', 'E2', 'E3'][np.argmax([e1_count, e2_count, e3_count])]
            
            analyzer = EnhancedStatisticalAnalyzer()
            ci_lower, ci_upper = analyzer._wilson_confidence_interval(dominant_count, n)
            ci_width = ci_upper - ci_lower
            
            results.append({
                'slice': key,
                'n': n,
                'e1_count': e1_count,
                'e2_count': e2_count,
                'e3_count': e3_count,
                'dominant_path': dominant_path,
                'dominant_pct': (dominant_count/n)*100,
                'ci_lower': ci_lower*100,
                'ci_upper': ci_upper*100,
                'ci_width': ci_width,
                'inconclusive': ci_width > 0.30,
                'cases': cases
            })
    
    # Generate markdown
    lines = ["### Day × News Matrix", ""]
    lines.append("| Day×News | n | E1 | E2 | E3 | Dominant | CI | Notes |")
    lines.append("|----------|---|----|----|----|---------|----|-------|")
    
    for result in sorted(results, key=lambda x: x['n'], reverse=True):
        slice_name = result['slice']
        n = result['n']
        
        e1_str = str(result['e1_count'])
        e2_str = str(result['e2_count'])
        e3_str = str(result['e3_count'])
        
        dominant_str = f"{result['dominant_path']} ({result['dominant_pct']:.0f}%)"
        ci_str = f"[{result['ci_lower']:.0f}–{result['ci_upper']:.0f}]"
        notes = "Inconclusive" if result['inconclusive'] else ""
        
        lines.append(f"| {slice_name} | {n} | {e1_str} | {e2_str} | {e3_str} | {dominant_str} | {ci_str} | {notes} |")
    
    return "\n".join(lines), results

def generate_top5_slices(day_results, news_results, matrix_results):
    """Generate Top 5 slices with proper thumbs up/down logic"""
    
    all_slices = []
    
    for result in day_results + news_results + matrix_results:
        if result['slice'] != 'Other' and result['n'] >= 3:
            
            # Calculate median times based on dominant path
            median_time = None
            if result['dominant_path'] == 'E3' and result['cases']:
                times = [c.get('hit_80_time') for c in result['cases'] if c.get('hit_80_time') and c['path'] == 'E3']
                if times:
                    median_time = f"t80={np.median(times):.0f}m"
            elif result['dominant_path'] == 'E2' and result['cases']:
                times = [c.get('hit_mid_time') for c in result['cases'] if c.get('hit_mid_time') and c['path'] == 'E2']
                if times:
                    median_time = f"tmid={np.median(times):.0f}m"
            
            # Thumbs up/down logic
            n = result['n']
            dominant_pct = result['dominant_pct']
            ci_width = result['ci_width']
            
            if ci_width > 0.30:
                thumbs = "Inconclusive"
            elif n >= 5 and dominant_pct >= 60 and ci_width <= 0.25:
                thumbs = "👍"
            else:
                thumbs = "👎"
            
            all_slices.append({
                'name': result['slice'],
                'n': n,
                'dominant_path': result['dominant_path'],
                'dominant_pct': dominant_pct,
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'thumbs': thumbs,
                'median_time': median_time
            })
    
    # Sort by sample size and take top 5
    top5 = sorted(all_slices, key=lambda x: x['n'], reverse=True)[:5]
    
    lines = ["### Top 5 Slices", ""]
    for slice_info in top5:
        time_str = f" ({slice_info['median_time']})" if slice_info['median_time'] else ""
        line = f"{slice_info['name']} (n={slice_info['n']}): {slice_info['dominant_path']} {slice_info['dominant_pct']:.0f}% [{slice_info['ci_lower']:.0f}–{slice_info['ci_upper']:.0f}]{time_str} → {slice_info['thumbs']}"
        lines.append(line)
    
    return "\n".join(lines)

def main():
    """Run final comprehensive RD@40 scan"""
    
    print("🔍 Final RD@40 Scan: Complete Analysis")
    print("=" * 50)
    
    # Extract and classify cases
    rd40_cases = extract_and_classify_rd40_cases()
    
    if not rd40_cases:
        print("❌ No RD@40 cases found")
        return
    
    print(f"📊 Found {len(rd40_cases)} RD@40 cases")
    
    # Print classification distribution
    path_counts = pd.Series([case['path'] for case in rd40_cases]).value_counts()
    print(f"\nPath Distribution:")
    for path, count in path_counts.items():
        pct = (count/len(rd40_cases))*100
        print(f"  {path}: {count} ({pct:.1f}%)")
    
    # Generate tables
    day_table, day_results = generate_markdown_table(rd40_cases, 'day_of_week', 'By Day')
    news_table, news_results = generate_markdown_table(rd40_cases, 'news_bucket', 'By News')
    matrix_table, matrix_results = generate_day_news_matrix(rd40_cases)
    
    # Gap age split
    for case in rd40_cases:
        gap_age = case.get('gap_age_days', 0)
        case['gap_age_bucket'] = 'fresh' if gap_age == 0 else 'aged_1to3'
    
    gap_table, gap_results = generate_markdown_table(rd40_cases, 'gap_age_bucket', 'By Gap Age')
    
    # Session overlap split
    overlap_table, overlap_results = generate_markdown_table(rd40_cases, 'session_overlap', 'By Session Overlap')
    
    # Generate Top 5
    top5 = generate_top5_slices(day_results + gap_results, news_results + overlap_results, matrix_results)
    
    # Output results
    print(f"\n{day_table}")
    print(f"\n{news_table}")
    print(f"\n{matrix_table}")
    print(f"\n{gap_table}")
    print(f"\n{overlap_table}")
    print(f"\n{top5}")
    
    print(f"\n**What to probe next**: Focus on E2/E3 classification refinement and validate hit_mid/hit_80 timing accuracy with manual spot checks.")

if __name__ == "__main__":
    main()