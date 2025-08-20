#!/usr/bin/env python3
"""
Day & News Pattern Analyzer
Analyzes day-of-week and economic news impact on RD@40 path patterns
"""

import sys
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
import traceback

# Import the existing experiment E analyzer
from experiment_e_analyzer import ExperimentEAnalyzer

class DayProfileAnalyzer:
    """Analyzer for day-of-week patterns in RD@40 path selection"""
    
    def __init__(self):
        self.experiment_e = ExperimentEAnalyzer()
        self.enhanced_sessions = self._load_enhanced_sessions()
    
    def _load_enhanced_sessions(self) -> Dict[str, Any]:
        """Load day/news enhanced session data"""
        enhanced_files = glob.glob("/Users/jack/IRONFORGE/data/day_news_enhanced/day_news_*.json")
        sessions = {}
        
        print(f"📊 Loading {len(enhanced_files)} enhanced sessions...")
        
        for file_path in enhanced_files:
            session_name = file_path.split('/')[-1].replace('day_news_', '').replace('.json', '')
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    sessions[session_name] = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"✅ Loaded {len(sessions)} enhanced sessions")
        return sessions
    
    def analyze_by_day_profile(self) -> Dict[str, Any]:
        """Analyze E1/E2/E3 path distribution by day of week"""
        print("📅 ANALYZING RD@40 PATTERNS BY DAY PROFILE")
        print("=" * 60)
        
        day_analysis = defaultdict(lambda: {
            'sessions': 0,
            'rd40_events': 0,
            'e1_cont': 0,
            'e2_mr': 0,
            'e3_accel': 0,
            'unknown': 0,
            'timing_to_60': [],
            'timing_to_80': [],
            'events_detail': []
        })
        
        total_rd40_events = 0
        
        for session_name, session_data in self.enhanced_sessions.items():
            day_profile = session_data.get('day_profile', {})
            day_name = day_profile.get('day_of_week', 'Unknown')
            
            day_analysis[day_name]['sessions'] += 1
            
            # Analyze RD@40 events in this session
            events = session_data.get('events', [])
            session_df = pd.DataFrame(events)
            
            if session_df.empty:
                continue
            
            # Find RD@40 events (±2.5% tolerance around 40%)
            rd40_events = session_df[
                (session_df['range_position'] >= 0.375) & 
                (session_df['range_position'] <= 0.425)
            ]
            
            for idx, rd40_event in rd40_events.iterrows():
                day_analysis[day_name]['rd40_events'] += 1
                total_rd40_events += 1
                
                # Classify path using existing E analyzer
                try:
                    enhanced_data = self.experiment_e.derive_advanced_features(session_df)
                    
                    # Try E1 CONT classification
                    e1_result = self.experiment_e.classify_e1_cont_path(enhanced_data, idx)
                    if e1_result.get('path') == 'E1_CONT':
                        day_analysis[day_name]['e1_cont'] += 1
                        continue
                    
                    # Try E2 MR classification  
                    e2_result = self.experiment_e.classify_e2_mr_path(enhanced_data, idx)
                    if e2_result.get('path') == 'E2_MR':
                        day_analysis[day_name]['e2_mr'] += 1
                        continue
                    
                    # Try E3 ACCEL classification
                    e3_result = self.experiment_e.classify_e3_accel_path(enhanced_data, idx)
                    if e3_result.get('path') == 'E3_ACCEL':
                        day_analysis[day_name]['e3_accel'] += 1
                        continue
                    
                    # If no clear classification
                    day_analysis[day_name]['unknown'] += 1
                    
                    # Store event details for further analysis
                    day_analysis[day_name]['events_detail'].append({
                        'session': session_name,
                        'timestamp': rd40_event.get('timestamp'),
                        'range_position': rd40_event.get('range_position'),
                        'energy_density': rd40_event.get('energy_density'),
                        'archaeological_significance': rd40_event.get('archaeological_significance'),
                        'news_context': rd40_event.get('news_context', {})
                    })
                    
                except Exception as e:
                    print(f"Error classifying event in {session_name}: {e}")
                    day_analysis[day_name]['unknown'] += 1
        
        # Calculate percentages and summary statistics
        day_summary = {}
        
        for day_name, data in day_analysis.items():
            total_events = data['rd40_events']
            if total_events == 0:
                continue
            
            day_summary[day_name] = {
                'profile_name': self._get_profile_name_for_day(day_name),
                'sessions': data['sessions'],
                'rd40_events': total_events,
                'path_distribution': {
                    'E1_CONT': {
                        'count': data['e1_cont'],
                        'percentage': (data['e1_cont'] / total_events) * 100
                    },
                    'E2_MR': {
                        'count': data['e2_mr'], 
                        'percentage': (data['e2_mr'] / total_events) * 100
                    },
                    'E3_ACCEL': {
                        'count': data['e3_accel'],
                        'percentage': (data['e3_accel'] / total_events) * 100
                    },
                    'UNKNOWN': {
                        'count': data['unknown'],
                        'percentage': (data['unknown'] / total_events) * 100
                    }
                }
            }
        
        # Print results
        print(f"\n📊 DAY PROFILE ANALYSIS RESULTS:")
        print(f"   Total RD@40 events analyzed: {total_rd40_events}")
        
        for day_name, summary in day_summary.items():
            profile = summary['profile_name']
            total = summary['rd40_events']
            
            print(f"\n🗓️ {day_name} ({profile}):")
            print(f"   Sessions: {summary['sessions']}, RD@40 events: {total}")
            
            dist = summary['path_distribution']
            print(f"   E1 CONT: {dist['E1_CONT']['count']} ({dist['E1_CONT']['percentage']:.1f}%)")
            print(f"   E2 MR:   {dist['E2_MR']['count']} ({dist['E2_MR']['percentage']:.1f}%)")
            print(f"   E3 ACCEL:{dist['E3_ACCEL']['count']} ({dist['E3_ACCEL']['percentage']:.1f}%)")
            print(f"   UNKNOWN: {dist['UNKNOWN']['count']} ({dist['UNKNOWN']['percentage']:.1f}%)")
        
        return {
            'total_rd40_events': total_rd40_events,
            'day_analysis': dict(day_analysis),
            'day_summary': day_summary
        }
    
    def _get_profile_name_for_day(self, day_name: str) -> str:
        """Get profile name for day"""
        profiles = {
            'Monday': 'gap_fill_bias',
            'Tuesday': 'trend_continuation',
            'Wednesday': 'balanced', 
            'Thursday': 'reversal_setup',
            'Friday': 'profit_taking'
        }
        return profiles.get(day_name, 'unknown')

class NewsImpactAnalyzer:
    """Analyzer for economic news impact on RD@40 path selection"""
    
    def __init__(self):
        self.experiment_e = ExperimentEAnalyzer()
        self.enhanced_sessions = self._load_enhanced_sessions()
    
    def _load_enhanced_sessions(self) -> Dict[str, Any]:
        """Load day/news enhanced session data"""
        enhanced_files = glob.glob("/Users/jack/IRONFORGE/data/day_news_enhanced/day_news_*.json")
        sessions = {}
        
        for file_path in enhanced_files:
            session_name = file_path.split('/')[-1].replace('day_news_', '').replace('.json', '')
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    sessions[session_name] = data
            except Exception as e:
                continue
        
        return sessions
    
    def analyze_by_news_bucket(self) -> Dict[str, Any]:
        """Analyze E1/E2/E3 path distribution by news proximity"""
        print("\n📰 ANALYZING RD@40 PATTERNS BY NEWS PROXIMITY")
        print("=" * 60)
        
        news_analysis = defaultdict(lambda: {
            'rd40_events': 0,
            'e1_cont': 0,
            'e2_mr': 0,
            'e3_accel': 0,
            'unknown': 0,
            'news_events': [],
            'volatility_multipliers': []
        })
        
        total_analyzed = 0
        
        for session_name, session_data in self.enhanced_sessions.items():
            events = session_data.get('events', [])
            
            # Find RD@40 events with news context
            for event in events:
                range_position = event.get('range_position', 0.5)
                
                if abs(range_position - 0.40) <= 0.025 and 'news_context' in event:
                    news_context = event.get('news_context', {})
                    news_bucket = news_context.get('news_bucket', 'quiet')
                    
                    news_analysis[news_bucket]['rd40_events'] += 1
                    total_analyzed += 1
                    
                    # Store news event details
                    news_analysis[news_bucket]['news_events'].append({
                        'event': news_context.get('news_event'),
                        'distance_mins': news_context.get('news_distance_mins'),
                        'impact': news_context.get('news_impact')
                    })
                    
                    if 'volatility_multiplier' in news_context:
                        news_analysis[news_bucket]['volatility_multipliers'].append(
                            news_context['volatility_multiplier']
                        )
                    
                    # Simple classification based on news context biases
                    expected_accel_bias = news_context.get('expected_accel_bias', 0.0)
                    expected_mr_bias = news_context.get('expected_mr_bias', 0.0)
                    
                    # Use bias to determine most likely path
                    if expected_accel_bias > 0.15:  # Strong ACCEL bias
                        news_analysis[news_bucket]['e3_accel'] += 1
                    elif expected_mr_bias > 0.10:   # Strong MR bias
                        news_analysis[news_bucket]['e2_mr'] += 1
                    elif expected_accel_bias > 0.05: # Mild ACCEL bias
                        news_analysis[news_bucket]['e3_accel'] += 1
                    elif expected_mr_bias > 0.0:    # Any MR bias
                        news_analysis[news_bucket]['e2_mr'] += 1
                    else:
                        news_analysis[news_bucket]['unknown'] += 1
        
        # Calculate summaries
        news_summary = {}
        
        for news_bucket, data in news_analysis.items():
            total_events = data['rd40_events']
            if total_events == 0:
                continue
            
            news_summary[news_bucket] = {
                'rd40_events': total_events,
                'path_distribution': {
                    'E1_CONT': {
                        'count': data['e1_cont'],
                        'percentage': (data['e1_cont'] / total_events) * 100
                    },
                    'E2_MR': {
                        'count': data['e2_mr'],
                        'percentage': (data['e2_mr'] / total_events) * 100
                    },
                    'E3_ACCEL': {
                        'count': data['e3_accel'], 
                        'percentage': (data['e3_accel'] / total_events) * 100
                    },
                    'UNKNOWN': {
                        'count': data['unknown'],
                        'percentage': (data['unknown'] / total_events) * 100
                    }
                },
                'avg_volatility_multiplier': np.mean(data['volatility_multipliers']) if data['volatility_multipliers'] else 1.0,
                'sample_news_events': list(set([ne['event'] for ne in data['news_events'] if ne['event']][:5]))
            }
        
        # Print results
        print(f"\n📊 NEWS IMPACT ANALYSIS RESULTS:")
        print(f"   Total RD@40 events with news context: {total_analyzed}")
        
        for news_bucket, summary in news_summary.items():
            total = summary['rd40_events']
            
            print(f"\n📰 {news_bucket.upper()}:")
            print(f"   RD@40 events: {total}")
            
            dist = summary['path_distribution']
            print(f"   E1 CONT: {dist['E1_CONT']['count']} ({dist['E1_CONT']['percentage']:.1f}%)")
            print(f"   E2 MR:   {dist['E2_MR']['count']} ({dist['E2_MR']['percentage']:.1f}%)")
            print(f"   E3 ACCEL:{dist['E3_ACCEL']['count']} ({dist['E3_ACCEL']['percentage']:.1f}%)")
            print(f"   UNKNOWN: {dist['UNKNOWN']['count']} ({dist['UNKNOWN']['percentage']:.1f}%)")
            
            print(f"   Avg volatility multiplier: {summary['avg_volatility_multiplier']:.2f}")
            if summary['sample_news_events']:
                print(f"   Sample events: {', '.join(summary['sample_news_events'][:3])}")
        
        return {
            'total_analyzed': total_analyzed,
            'news_analysis': dict(news_analysis),
            'news_summary': news_summary
        }

class DayNewsMatrixAnalyzer:
    """Combined analyzer for Day × News interaction patterns"""
    
    def __init__(self):
        self.day_analyzer = DayProfileAnalyzer()
        self.news_analyzer = NewsImpactAnalyzer()
    
    def create_day_news_matrix(self) -> Dict[str, Any]:
        """Create Day × News matrix analysis"""
        print("\n🗓️📰 CREATING DAY × NEWS INTERACTION MATRIX")
        print("=" * 60)
        
        matrix_data = defaultdict(lambda: defaultdict(lambda: {
            'rd40_events': 0,
            'e1_cont': 0,
            'e2_mr': 0,
            'e3_accel': 0,
            'unknown': 0
        }))
        
        total_matrix_events = 0
        
        for session_name, session_data in self.day_analyzer.enhanced_sessions.items():
            day_profile = session_data.get('day_profile', {})
            day_name = day_profile.get('day_of_week', 'Unknown')
            
            events = session_data.get('events', [])
            
            for event in events:
                range_position = event.get('range_position', 0.5)
                
                if abs(range_position - 0.40) <= 0.025 and 'news_context' in event:
                    news_context = event.get('news_context', {})
                    news_bucket = news_context.get('news_bucket', 'quiet')
                    
                    matrix_data[day_name][news_bucket]['rd40_events'] += 1
                    total_matrix_events += 1
                    
                    # Simple path classification based on combined biases
                    day_profiles = {
                        'Monday': {'mr_bias': 0.15, 'accel_bias': -0.10},
                        'Tuesday': {'mr_bias': -0.10, 'accel_bias': 0.20},
                        'Wednesday': {'mr_bias': 0.0, 'accel_bias': 0.0},
                        'Thursday': {'mr_bias': 0.12, 'accel_bias': -0.08},
                        'Friday': {'mr_bias': 0.08, 'accel_bias': 0.05}
                    }
                    
                    day_bias = day_profiles.get(day_name, {'mr_bias': 0.0, 'accel_bias': 0.0})
                    news_accel_bias = news_context.get('expected_accel_bias', 0.0)
                    news_mr_bias = news_context.get('expected_mr_bias', 0.0)
                    
                    # Combined bias calculation
                    total_accel_bias = day_bias['accel_bias'] + news_accel_bias
                    total_mr_bias = day_bias['mr_bias'] + news_mr_bias
                    
                    # Classify based on combined bias
                    if total_accel_bias > 0.15:
                        matrix_data[day_name][news_bucket]['e3_accel'] += 1
                    elif total_mr_bias > 0.10:
                        matrix_data[day_name][news_bucket]['e2_mr'] += 1
                    elif total_accel_bias > 0.05:
                        matrix_data[day_name][news_bucket]['e3_accel'] += 1
                    elif total_mr_bias > 0.0:
                        matrix_data[day_name][news_bucket]['e2_mr'] += 1
                    else:
                        matrix_data[day_name][news_bucket]['unknown'] += 1
        
        # Create matrix summary
        matrix_summary = {}
        
        for day_name, news_buckets in matrix_data.items():
            matrix_summary[day_name] = {}
            
            for news_bucket, data in news_buckets.items():
                total_events = data['rd40_events']
                if total_events >= 3:  # Only include cells with sufficient sample size
                    matrix_summary[day_name][news_bucket] = {
                        'rd40_events': total_events,
                        'e2_mr_percentage': (data['e2_mr'] / total_events) * 100,
                        'e3_accel_percentage': (data['e3_accel'] / total_events) * 100,
                        'dominant_path': 'E2_MR' if data['e2_mr'] > data['e3_accel'] else 'E3_ACCEL'
                    }
        
        # Print matrix
        print(f"\n📊 DAY × NEWS MATRIX RESULTS:")
        print(f"   Total matrix events: {total_matrix_events}")
        
        print(f"\n🗓️📰 MATRIX SUMMARY (minimum 3 events per cell):")
        print("=" * 80)
        print(f"{'Day':<12} {'News Bucket':<15} {'Events':<8} {'MR%':<6} {'ACCEL%':<8} {'Dominant':<10}")
        print("-" * 80)
        
        for day_name, news_buckets in matrix_summary.items():
            for news_bucket, summary in news_buckets.items():
                events = summary['rd40_events']
                mr_pct = summary['e2_mr_percentage']
                accel_pct = summary['e3_accel_percentage']
                dominant = summary['dominant_path']
                
                print(f"{day_name:<12} {news_bucket:<15} {events:<8} {mr_pct:<6.1f} {accel_pct:<8.1f} {dominant:<10}")
        
        return {
            'total_matrix_events': total_matrix_events,
            'matrix_data': dict(matrix_data),
            'matrix_summary': matrix_summary
        }

def main():
    """Main analysis function"""
    print("🚀 IRONFORGE: Day & News Pattern Analysis")
    print("🎯 Analyzing day-of-week and news impact on RD@40 path patterns")
    print("=" * 80)
    
    try:
        # Day profile analysis
        print("\n🔄 Running Day Profile Analysis...")
        day_analyzer = DayProfileAnalyzer()
        day_results = day_analyzer.analyze_by_day_profile()
        
        # News impact analysis
        print("\n🔄 Running News Impact Analysis...")
        news_analyzer = NewsImpactAnalyzer()
        news_results = news_analyzer.analyze_by_news_bucket()
        
        # Day × News matrix analysis
        print("\n🔄 Running Day × News Matrix Analysis...")
        matrix_analyzer = DayNewsMatrixAnalyzer()
        matrix_results = matrix_analyzer.create_day_news_matrix()
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"day_news_pattern_analysis_{timestamp}.json"
        
        comprehensive_results = {
            'timestamp': timestamp,
            'analysis_type': 'Day_News_Pattern_Analysis',
            'day_profile_results': day_results,
            'news_impact_results': news_results,
            'matrix_results': matrix_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n💾 Analysis results saved to: {output_file}")
        print(f"\n✅ Day & News Pattern Analysis Complete!")
        
        return comprehensive_results
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure experiment_e_analyzer.py is available")
        
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()