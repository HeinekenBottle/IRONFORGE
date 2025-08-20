#!/usr/bin/env python3
"""
Quick Scan: RD@40 by Day & News (Exploration Only)
Extract and analyze all RD@40 cases with day/news context
"""

import json
import glob
from collections import defaultdict
from enhanced_statistical_framework import EnhancedStatisticalAnalyzer
import numpy as np

def extract_rd40_cases():
    """Extract all RD@40 cases with required fields"""
    
    enhanced_files = glob.glob("/Users/jack/IRONFORGE/data/day_news_enhanced/day_news_*.json")
    
    if not enhanced_files:
        print("❌ No enhanced data files found")
        return []
    
    rd40_cases = []
    
    for file_path in enhanced_files:
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            
            day_profile = session_data.get('day_profile', {})
            day_of_week = day_profile.get('day_of_week', 'Unknown')
            profile_name = day_profile.get('profile_name', 'unknown')
            
            events = session_data.get('events', [])
            
            for event in events:
                range_position = event.get('range_position', 0.5)
                
                # Filter RD@40 events (40% ± 2.5%)
                if abs(range_position - 0.40) <= 0.025:
                    news_context = event.get('news_context', {})
                    
                    # Extract required fields
                    case = {
                        'day_of_week': day_of_week,
                        'day_profile': profile_name,
                        'news_bucket': news_context.get('news_bucket', 'quiet'),
                        'news_distance_mins': news_context.get('news_distance_mins', 999),
                        'regime': event.get('regime', 'unknown'),
                        'f8_level': event.get('f8_level', 0.0),
                        'f8_slope': event.get('f8_slope', 0.0),
                        'gap_age_days': event.get('gap_age_days', 0),
                        'session_overlap': news_context.get('session_overlap', False),
                        'energy_density': event.get('energy_density', 0.5),
                        'time_to_60': event.get('time_to_60_pct'),
                        'time_to_80': event.get('time_to_80_pct'),
                        'mid_hit': event.get('mid_hit', False),
                        'timestamp': event.get('timestamp', '')
                    }
                    
                    # Classify path (E1/E2/E3)
                    case['path'] = classify_path(case)
                    
                    rd40_cases.append(case)
                    
        except Exception as e:
            continue
    
    return rd40_cases

def classify_path(case):
    """Classify RD@40 case into E1/E2/E3 path"""
    
    energy = case['energy_density']
    f8_level = case['f8_level']
    
    # Simple classification logic
    if energy > 0.7 or f8_level > 0.6:
        return 'E3'  # ACCEL
    elif energy < 0.3 and case['mid_hit']:
        return 'E2'  # MR
    else:
        return 'E1'  # CONT

def generate_analysis_table(data, slice_key, title):
    """Generate analysis table with sample size rules"""
    
    # Group by slice key
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
            # Calculate percentages and CIs
            e1_pct = (e1_count / n) * 100
            e2_pct = (e2_count / n) * 100  
            e3_pct = (e3_count / n) * 100
            
            # Wilson CI for dominant path
            dominant_count = max(e1_count, e2_count, e3_count)
            dominant_path = ['E1', 'E2', 'E3'][np.argmax([e1_count, e2_count, e3_count])]
            
            analyzer = EnhancedStatisticalAnalyzer()
            ci_lower, ci_upper = analyzer._wilson_confidence_interval(dominant_count, n)
            ci_width = ci_upper - ci_lower
            
            inconclusive = "Inconclusive" if ci_width > 0.30 else ""
            
            results.append({
                'slice': key,
                'n': n,
                'e1_count': e1_count, 'e1_pct': e1_pct,
                'e2_count': e2_count, 'e2_pct': e2_pct,
                'e3_count': e3_count, 'e3_pct': e3_pct,
                'dominant_path': dominant_path,
                'dominant_pct': (dominant_count / n) * 100,
                'ci_lower': ci_lower * 100,
                'ci_upper': ci_upper * 100,
                'ci_width': ci_width,
                'inconclusive': inconclusive,
                'cases': cases
            })
        else:
            # Small sample - merge to Other
            other_cases.extend(cases)
    
    # Handle "Other" group if exists
    if other_cases:
        n = len(other_cases)
        e1_count = sum(1 for c in other_cases if c['path'] == 'E1')
        e2_count = sum(1 for c in other_cases if c['path'] == 'E2')
        e3_count = sum(1 for c in other_cases if c['path'] == 'E3')
        
        results.append({
            'slice': 'Other',
            'n': n,
            'e1_count': e1_count, 'e1_pct': 0,  # No % for small samples
            'e2_count': e2_count, 'e2_pct': 0,
            'e3_count': e3_count, 'e3_pct': 0,
            'dominant_path': 'counts only',
            'dominant_pct': 0,
            'ci_lower': 0, 'ci_upper': 0, 'ci_width': 0,
            'inconclusive': '',
            'cases': other_cases
        })
    
    # Generate markdown table
    table_lines = [f"## {title}", ""]
    table_lines.append("| Slice | n | E1 | E2 | E3 | Dominant | CI | Notes |")
    table_lines.append("|-------|---|----|----|----|---------|----|-------|")
    
    for result in sorted(results, key=lambda x: x['n'], reverse=True):
        slice_name = result['slice']
        n = result['n']
        
        if result['slice'] == 'Other':
            # Show counts only for small samples
            e1_str = str(result['e1_count'])
            e2_str = str(result['e2_count'])
            e3_str = str(result['e3_count'])
            dominant_str = "counts only"
            ci_str = "—"
        else:
            # Show percentages for n≥5
            e1_str = f"{result['e1_count']} ({result['e1_pct']:.0f}%)"
            e2_str = f"{result['e2_count']} ({result['e2_pct']:.0f}%)"
            e3_str = f"{result['e3_count']} ({result['e3_pct']:.0f}%)"
            dominant_str = f"{result['dominant_path']} ({result['dominant_pct']:.0f}%)"
            ci_str = f"[{result['ci_lower']:.0f}–{result['ci_upper']:.0f}]"
        
        notes = result['inconclusive']
        
        table_lines.append(f"| {slice_name} | {n} | {e1_str} | {e2_str} | {e3_str} | {dominant_str} | {ci_str} | {notes} |")
    
    return "\n".join(table_lines), results

def generate_day_news_matrix(data):
    """Generate Day × News matrix"""
    
    # Group by day + news combination
    matrix = defaultdict(list)
    for case in data:
        key = f"{case['day_of_week']}×{case['news_bucket']}"
        matrix[key].append(case)
    
    # Apply sample size rules and create results
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
                'dominant_pct': (dominant_count / n) * 100,
                'ci_lower': ci_lower * 100,
                'ci_upper': ci_upper * 100,
                'ci_width': ci_width,
                'inconclusive': "Inconclusive" if ci_width > 0.30 else "",
                'cases': cases
            })
    
    # Generate table
    table_lines = ["## Day × News Matrix", ""]
    table_lines.append("| Day×News | n | E1 | E2 | E3 | Dominant | CI | Notes |")
    table_lines.append("|----------|---|----|----|----|---------|----|-------|")
    
    for result in sorted(results, key=lambda x: x['n'], reverse=True):
        slice_name = result['slice']
        n = result['n']
        
        e1_str = str(result['e1_count'])
        e2_str = str(result['e2_count'])
        e3_str = str(result['e3_count'])
        
        dominant_str = f"{result['dominant_path']} ({result['dominant_pct']:.0f}%)"
        ci_str = f"[{result['ci_lower']:.0f}–{result['ci_upper']:.0f}]"
        notes = result['inconclusive']
        
        table_lines.append(f"| {slice_name} | {n} | {e1_str} | {e2_str} | {e3_str} | {dominant_str} | {ci_str} | {notes} |")
    
    return "\n".join(table_lines), results

def generate_top5_slices(day_results, news_results, matrix_results):
    """Generate Top 5 slices with thumbs up/down"""
    
    all_slices = []
    
    # Collect all slices
    for result in day_results + news_results + matrix_results:
        if result['slice'] != 'Other' and result['n'] >= 3:
            
            # Calculate median times based on dominant path
            median_time = None
            if result['dominant_path'] == 'E3':
                # ACCEL - time to 80
                times = [c.get('time_to_80') for c in result['cases'] if c.get('time_to_80')]
                if times:
                    median_time = f"t80={np.median(times):.0f}m"
            elif result['dominant_path'] == 'E2':
                # MR - time to mid  
                times = [c.get('time_to_60') for c in result['cases'] if c.get('time_to_60')]  # Using 60 as proxy for mid
                if times:
                    median_time = f"tmid={np.median(times):.0f}m"
            
            # Determine thumbs up/down
            n = result['n']
            dominant_pct = result['dominant_pct']
            ci_width = result['ci_width']
            
            if ci_width > 0.30:
                thumbs = "Inconclusive"
            elif n >= 5 and dominant_pct >= 60 and ci_width <= 0.25:
                thumbs = "👍"
            else:
                thumbs = "👎"
            
            slice_info = {
                'name': result['slice'],
                'n': n,
                'dominant_path': result['dominant_path'],
                'dominant_pct': dominant_pct,
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'thumbs': thumbs,
                'median_time': median_time
            }
            
            all_slices.append(slice_info)
    
    # Sort by sample size (descending) and take top 5
    top5 = sorted(all_slices, key=lambda x: x['n'], reverse=True)[:5]
    
    lines = ["## Top 5 Slices", ""]
    for slice_info in top5:
        time_str = f" ({slice_info['median_time']})" if slice_info['median_time'] else ""
        
        line = f"{slice_info['name']} (n={slice_info['n']}): {slice_info['dominant_path']} {slice_info['dominant_pct']:.0f}% [{slice_info['ci_lower']:.0f}–{slice_info['ci_upper']:.0f}]{time_str} → {slice_info['thumbs']}"
        lines.append(line)
    
    return "\n".join(lines)

def main():
    """Run quick RD@40 scan"""
    
    print("🔍 Quick Scan: RD@40 by Day & News")
    print("=" * 40)
    
    # Extract RD@40 cases
    rd40_cases = extract_rd40_cases()
    
    if not rd40_cases:
        print("❌ No RD@40 cases found")
        return
    
    print(f"📊 Found {len(rd40_cases)} RD@40 cases")
    
    # Generate tables
    day_table, day_results = generate_analysis_table(rd40_cases, 'day_of_week', 'By Day')
    news_table, news_results = generate_analysis_table(rd40_cases, 'news_bucket', 'By News')
    matrix_table, matrix_results = generate_day_news_matrix(rd40_cases)
    
    # Generate Top 5
    top5 = generate_top5_slices(day_results, news_results, matrix_results)
    
    # Output results
    print("\n" + day_table)
    print("\n" + news_table)  
    print("\n" + matrix_table)
    print("\n" + top5)
    
    print("\n**What to probe next**: Focus on Tuesday×high±120m and Monday×quiet combinations for deeper statistical validation.")

if __name__ == "__main__":
    main()