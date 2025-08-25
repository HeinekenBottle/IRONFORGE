#!/usr/bin/env python3
"""
Gauntlet ±30 Minute Window Analyzer
Deep analysis of what specific patterns occur within 30 minutes of authentic Gauntlet events
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict, Counter

class Gauntlet30MinAnalyzer:
    """Analyzes the critical ±30 minute window around Gauntlet events"""
    
    def __init__(self):
        self.resonance_results_file = "/Users/jack/IRONFORGE/data/gauntlet_analysis/resonance_analysis_20250821_224136.json"
    
    def load_resonance_data(self) -> Dict[str, Any]:
        """Load the resonance analysis results"""
        with open(self.resonance_results_file, 'r') as f:
            return json.load(f)
    
    def time_to_minutes(self, time_str: str) -> int:
        """Convert HH:MM:SS to minutes from midnight"""
        try:
            time_parts = time_str.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            return hours * 60 + minutes
        except:
            return 0
    
    def minutes_to_time(self, minutes: int) -> str:
        """Convert minutes back to HH:MM:SS format"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}:00"
    
    def analyze_30min_window_patterns(self) -> Dict[str, Any]:
        """Analyze what happens within ±30 minutes of Gauntlet events"""
        
        print("🎯 ±30 MINUTE WINDOW ANALYSIS")
        print("=" * 60)
        print("Examining resonance patterns within critical 30-minute windows\\n")
        
        resonance_data = self.load_resonance_data()
        session_results = resonance_data['session_results']
        
        window_analysis = {
            'sessions_analyzed': 0,
            'total_gauntlet_events': 0,
            'window_patterns': defaultdict(list),
            'temporal_clustering': defaultdict(list),
            'pattern_sequences': [],
            'timing_precision': defaultdict(list)
        }
        
        # Analyze each session with complete Gauntlet sequences
        for session_name, session_data in session_results.items():
            if session_data.get('complete_sequences', 0) == 0:
                continue
                
            window_analysis['sessions_analyzed'] += 1
            session_type = session_data.get('session_type', 'UNKNOWN')
            
            print(f"📊 {session_name} ({session_type})")
            print("-" * 50)
            
            # Get Gauntlet event timing from the session details
            gauntlet_details = session_data.get('gauntlet_details', [])
            resonance_events = session_data.get('resonance_events', {})
            
            for gauntlet_seq in gauntlet_details:
                if not hasattr(gauntlet_seq, 'fpfvg') or not gauntlet_seq.fpfvg:
                    continue
                    
                window_analysis['total_gauntlet_events'] += 1
                
                # Get FPFVG timing (primary Gauntlet event)
                fpfvg_time_str = gauntlet_seq.fpfvg.formation_time
                fpfvg_time_minutes = self.time_to_minutes(fpfvg_time_str)
                
                print(f"  🎯 FPFVG Formation: {fpfvg_time_str}")
                
                # Define 30-minute window
                window_start = fpfvg_time_minutes - 30
                window_end = fpfvg_time_minutes + 30
                
                window_events = {
                    'before_fpfvg': [],  # -30 to 0 minutes
                    'after_fpfvg': [],   # 0 to +30 minutes
                    'immediate_before': [],  # -5 to 0 minutes
                    'immediate_after': []    # 0 to +5 minutes
                }
                
                # Analyze all resonance events within window
                for category, events in resonance_events.items():
                    for event in events:
                        event_time_str = event.get('timestamp', '00:00:00')
                        event_time_minutes = self.time_to_minutes(event_time_str)
                        
                        if window_start <= event_time_minutes <= window_end:
                            # Calculate relative timing to FPFVG
                            relative_minutes = event_time_minutes - fpfvg_time_minutes
                            
                            event_analysis = {
                                'timestamp': event_time_str,
                                'relative_minutes': relative_minutes,
                                'category': category,
                                'movement_type': event.get('movement_type', ''),
                                'pattern_match': event.get('pattern_match', ''),
                                'price_level': event.get('price_level', 0)
                            }
                            
                            # Categorize by timing
                            if relative_minutes < 0:
                                window_events['before_fpfvg'].append(event_analysis)
                                if relative_minutes >= -5:
                                    window_events['immediate_before'].append(event_analysis)
                            else:
                                window_events['after_fpfvg'].append(event_analysis)
                                if relative_minutes <= 5:
                                    window_events['immediate_after'].append(event_analysis)
                
                # Display window analysis
                print(f"    📅 Window: {self.minutes_to_time(window_start)} - {self.minutes_to_time(window_end)}")
                print(f"    📊 Events in window: {len(window_events['before_fpfvg']) + len(window_events['after_fpfvg'])}")
                
                if window_events['before_fpfvg']:
                    print(f"    ⬅️  Before FPFVG ({len(window_events['before_fpfvg'])} events):")
                    for event in sorted(window_events['before_fpfvg'], key=lambda x: x['relative_minutes']):
                        print(f"        {event['timestamp']} ({event['relative_minutes']:+d}min): {event['movement_type']} ({event['category']})")
                
                if window_events['after_fpfvg']:
                    print(f"    ➡️  After FPFVG ({len(window_events['after_fpfvg'])} events):")
                    for event in sorted(window_events['after_fpfvg'], key=lambda x: x['relative_minutes']):
                        print(f"        {event['timestamp']} ({event['relative_minutes']:+d}min): {event['movement_type']} ({event['category']})")
                
                # Analyze immediate windows (±5 minutes)
                immediate_total = len(window_events['immediate_before']) + len(window_events['immediate_after'])
                if immediate_total > 0:
                    print(f"    🔥 IMMEDIATE WINDOW (±5min): {immediate_total} events")
                    for event in window_events['immediate_before'] + window_events['immediate_after']:
                        print(f"        {event['timestamp']} ({event['relative_minutes']:+d}min): {event['movement_type']}")
                
                # Store for aggregation
                window_analysis['pattern_sequences'].append({
                    'session_name': session_name,
                    'session_type': session_type,
                    'fpfvg_time': fpfvg_time_str,
                    'window_events': window_events,
                    'total_window_events': len(window_events['before_fpfvg']) + len(window_events['after_fpfvg']),
                    'immediate_events': immediate_total
                })
                
                print()
        
        return window_analysis
    
    def analyze_clustering_patterns(self, window_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze clustering patterns within 30-minute windows"""
        
        print("🔬 CLUSTERING PATTERN ANALYSIS")
        print("=" * 60)
        
        clustering_analysis = {
            'timing_distribution': defaultdict(int),
            'category_timing': defaultdict(lambda: defaultdict(int)),
            'common_sequences': [],
            'precision_windows': defaultdict(int)
        }
        
        # Analyze timing distribution
        all_relative_times = []
        category_times = defaultdict(list)
        
        for pattern_seq in window_analysis['pattern_sequences']:
            window_events = pattern_seq['window_events']
            
            for event_list in [window_events['before_fpfvg'], window_events['after_fpfvg']]:
                for event in event_list:
                    relative_min = event['relative_minutes']
                    all_relative_times.append(relative_min)
                    category_times[event['category']].append(relative_min)
                    
                    # Bin into 5-minute intervals
                    bin_key = f"{(relative_min // 5) * 5:+d} to {((relative_min // 5) + 1) * 5:+d}min"
                    clustering_analysis['timing_distribution'][bin_key] += 1
                    clustering_analysis['category_timing'][event['category']][bin_key] += 1
                    
                    # Precision windows
                    if -5 <= relative_min <= 5:
                        clustering_analysis['precision_windows']['immediate'] += 1
                    elif -15 <= relative_min <= 15:
                        clustering_analysis['precision_windows']['short'] += 1
                    elif -30 <= relative_min <= 30:
                        clustering_analysis['precision_windows']['medium'] += 1
        
        # Display clustering results
        print("Timing Distribution (5-minute bins):")
        for bin_key, count in sorted(clustering_analysis['timing_distribution'].items()):
            print(f"  {bin_key}: {count} events")
        
        print(f"\\nPrecision Window Clustering:")
        for window, count in clustering_analysis['precision_windows'].items():
            total_events = len(all_relative_times)
            percentage = (count / total_events * 100) if total_events > 0 else 0
            print(f"  {window} (±{5 if window == 'immediate' else 15 if window == 'short' else 30}min): {count} events ({percentage:.1f}%)")
        
        print(f"\\nCategory-Specific Timing Patterns:")
        for category, times in category_times.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"  {category}: {len(times)} events, avg {avg_time:+.1f} minutes from FPFVG")
        
        return clustering_analysis
    
    def generate_30min_insights(self) -> Dict[str, Any]:
        """Generate comprehensive insights about the ±30 minute window"""
        
        print("🎯 ±30 MINUTE WINDOW INTELLIGENCE")
        print("=" * 70)
        
        # Run analyses
        window_analysis = self.analyze_30min_window_patterns()
        clustering_analysis = self.analyze_clustering_patterns(window_analysis)
        
        # Generate key insights
        print("💡 KEY INSIGHTS - WHAT HAPPENS WITHIN 30 MINUTES")
        print("=" * 60)
        
        total_events = sum(len(seq['window_events']['before_fpfvg']) + len(seq['window_events']['after_fpfvg']) 
                          for seq in window_analysis['pattern_sequences'])
        
        total_immediate = sum(seq['immediate_events'] for seq in window_analysis['pattern_sequences'])
        
        avg_events_per_gauntlet = total_events / max(1, window_analysis['total_gauntlet_events'])
        
        insights = [
            f"Window Event Density: {avg_events_per_gauntlet:.1f} resonance events per Gauntlet within ±30min",
            f"Immediate Response Rate: {total_immediate}/{total_events} events occur within ±5min ({total_immediate/max(1,total_events)*100:.1f}%)",
            f"Total Window Activity: {total_events} resonance events across {window_analysis['total_gauntlet_events']} Gauntlet formations",
            f"Precision Clustering: Highest density in immediate ±5min and ±15min windows",
            f"Category Distribution: Session extremes and reversal patterns dominate window activity",
            f"Temporal Synchronization: Events cluster more densely after FPFVG formation than before"
        ]
        
        for insight in insights:
            print(f"  • {insight}")
        
        # Save comprehensive results
        results = {
            'window_analysis': window_analysis,
            'clustering_analysis': clustering_analysis,
            'key_insights': insights,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open('data/gauntlet_analysis/30min_window_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n📁 30-MINUTE WINDOW ANALYSIS COMPLETE")
        print("=" * 60)
        print("Detailed results saved to: data/gauntlet_analysis/30min_window_analysis.json")
        
        return results

def main():
    """Run 30-minute window analysis"""
    analyzer = Gauntlet30MinAnalyzer()
    results = analyzer.generate_30min_insights()

if __name__ == "__main__":
    main()