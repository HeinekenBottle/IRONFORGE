#!/usr/bin/env python3
"""
Macro Window Volatility Analysis
Phase I Data Exploration: Behavioral Pattern Classification

Analyzes volatility characteristics across 89 detected macro windows from H8 research.
Classifies macro windows by behavioral patterns and investigates RD@40 clustering.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from collections import defaultdict, Counter
import sys
from pathlib import Path


class MacroWindowVolatilityAnalyzer:
    """
    Analyzes volatility characteristics across macro window classifications.
    
    Behavioral Pattern Classifications:
    - Asian Session: Low volatility consolidation (19:00-04:00 ET)
    - London Session: Institutional manipulation patterns (02:00-11:00 ET)  
    - NY AM/PM: Expansion and high volatility (09:30-16:00 ET)
    - Lunch/Counter: Reversal and counter-trend zones (12:00-13:00 ET)
    """
    
    def __init__(self, data_path):
        """Initialize analyzer with confluence analysis data."""
        self.data_path = data_path
        self.df = None
        self.macro_windows = None
        self.session_patterns = {}
        
        # Define behavioral pattern classifications
        self.session_behaviors = {
            'ASIA': {'pattern': 'low_volatility', 'description': 'Consolidation and range-bound'},
            'LONDON': {'pattern': 'manipulation', 'description': 'Institutional liquidity sweeps'},
            'PREMARKET': {'pattern': 'setup', 'description': 'Pre-market positioning'},
            'NY_AM': {'pattern': 'expansion', 'description': 'Morning volatility expansion'}, 
            'NY_PM': {'pattern': 'continuation', 'description': 'Afternoon trend continuation'},
            'LUNCH': {'pattern': 'consolidation', 'description': 'Mid-day consolidation'},
            'MIDNIGHT': {'pattern': 'quiet', 'description': 'Overnight quiet periods'},
            'UNKNOWN': {'pattern': 'mixed', 'description': 'Mixed or transitional'}
        }
        
        # Define ICT macro window classifications
        self.macro_window_behaviors = {
            'LONDON': {'volatility': 'medium', 'behavior': 'manipulation'},
            'PRE_MKT_1': {'volatility': 'low', 'behavior': 'setup'},
            'PRE_MKT_2': {'volatility': 'medium', 'behavior': 'setup'},
            'AM_1': {'volatility': 'high', 'behavior': 'expansion'},
            'AM_2': {'volatility': 'high', 'behavior': 'expansion'},
            'AM_3': {'volatility': 'medium', 'behavior': 'expansion'},
            'AM_4': {'volatility': 'medium', 'behavior': 'transition'},
            'LUNCH_COUNTER': {'volatility': 'low', 'behavior': 'counter_trend'},
            'PM_1': {'volatility': 'high', 'behavior': 'continuation'},
            'PM_2': {'volatility': 'high', 'behavior': 'continuation'}, 
            'PM_3': {'volatility': 'medium', 'behavior': 'continuation'},
            'PM_4': {'volatility': 'medium', 'behavior': 'closing'}
        }
    
    def load_data(self):
        """Load macro window confluence analysis data."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} events from confluence analysis")
            
            # Filter to macro window events only
            self.macro_windows = self.df[self.df['in_macro_window'] == True].copy()
            print(f"✅ Found {len(self.macro_windows)} macro window events")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_session_patterns(self):
        """Analyze behavioral patterns by session type."""
        print("\n🔍 ANALYZING SESSION BEHAVIORAL PATTERNS")
        print("=" * 60)
        
        session_stats = {}
        
        for session_type in self.df['session_type'].unique():
            if pd.isna(session_type) or session_type == 'unknown':
                continue
                
            session_data = self.df[self.df['session_type'] == session_type]
            macro_data = session_data[session_data['in_macro_window'] == True]
            
            # Calculate volatility metrics
            energy_stats = {
                'mean_energy': session_data['energy_density'].mean(),
                'std_energy': session_data['energy_density'].std(),
                'max_energy': session_data['energy_density'].max(),
                'min_energy': session_data['energy_density'].min()
            }
            
            # Range position clustering
            range_stats = {
                'mean_position': session_data['range_position'].mean(),
                'std_position': session_data['range_position'].std(),
                'position_40_count': len(session_data[abs(session_data['range_position'] - 0.4) < 0.025])
            }
            
            # Macro window penetration
            macro_stats = {
                'total_events': len(session_data),
                'macro_events': len(macro_data),
                'macro_penetration': len(macro_data) / len(session_data) if len(session_data) > 0 else 0,
                'compound_events': len(session_data[session_data['compound_type'] != 'baseline'])
            }
            
            session_stats[session_type] = {
                'behavior': self.session_behaviors.get(session_type, {}).get('pattern', 'unknown'),
                'description': self.session_behaviors.get(session_type, {}).get('description', 'Unknown pattern'),
                'energy': energy_stats,
                'range': range_stats, 
                'macro': macro_stats
            }
        
        self.session_patterns = session_stats
        self._print_session_analysis()
        
        return session_stats
    
    def analyze_macro_window_clustering(self):
        """Analyze RD@40 archaeological zone clustering across macro windows."""
        print("\n🔍 RD@40 ARCHAEOLOGICAL ZONE CLUSTERING ANALYSIS")
        print("=" * 60)
        
        # Group by macro window name
        window_analysis = {}
        
        for window_name in self.macro_windows['macro_window_name'].unique():
            if pd.isna(window_name):
                continue
                
            window_data = self.macro_windows[self.macro_windows['macro_window_name'] == window_name]
            
            # Analyze clustering patterns
            clustering_stats = {
                'event_count': len(window_data),
                'unique_timestamps': len(window_data['timestamp_et'].unique()),
                'unique_price_levels': len(window_data['price_level'].unique()),
                'date_span': len(window_data['trading_day'].unique()),
                'avg_energy_density': window_data['energy_density'].mean(),
                'range_position_mean': window_data['range_position'].mean(),
                'range_position_std': window_data['range_position'].std(),
                'exact_40_hits': len(window_data[window_data['range_position'] == 0.4]),
                'near_40_hits': len(window_data[abs(window_data['range_position'] - 0.4) < 0.025])
            }
            
            # Session distribution
            session_dist = window_data['session_type'].value_counts().to_dict()
            
            # News proximity analysis  
            news_dist = window_data['news_bucket'].value_counts().to_dict()
            compound_dist = window_data['compound_type'].value_counts().to_dict()
            
            window_analysis[window_name] = {
                'behavior': self.macro_window_behaviors.get(window_name, {}).get('behavior', 'unknown'),
                'volatility': self.macro_window_behaviors.get(window_name, {}).get('volatility', 'unknown'),
                'clustering': clustering_stats,
                'session_distribution': session_dist,
                'news_distribution': news_dist,
                'compound_distribution': compound_dist
            }
        
        self._print_clustering_analysis(window_analysis)
        return window_analysis
    
    def analyze_timestamp_distributions(self):
        """Explore news proximity patterns and timestamp distributions."""
        print("\n🔍 TIMESTAMP DISTRIBUTION & NEWS PROXIMITY ANALYSIS")
        print("=" * 60)
        
        # TODO(human) - Implement timestamp pattern analysis
        # This section needs human input on the analysis approach
        print("⚠️  TODO: Implement detailed timestamp pattern analysis")
        print("   - Hour-by-hour event distribution")
        print("   - News proximity correlation patterns")
        print("   - Day-of-week clustering effects")
        print("   - Cross-session temporal persistence")
        
        return {}
    
    def _print_session_analysis(self):
        """Print formatted session behavioral pattern analysis."""
        print("\n📊 SESSION BEHAVIORAL PATTERNS")
        print("-" * 50)
        
        for session_type, stats in self.session_patterns.items():
            behavior = stats['behavior']
            description = stats['description']
            energy = stats['energy']
            macro = stats['macro']
            
            print(f"\n🎯 {session_type} Session - {behavior.upper()}")
            print(f"   Pattern: {description}")
            print(f"   Energy Density: {energy['mean_energy']:.3f} ± {energy['std_energy']:.3f}")
            print(f"   Total Events: {macro['total_events']}")
            print(f"   Macro Penetration: {macro['macro_penetration']:.1%} ({macro['macro_events']} events)")
            print(f"   Compound Events: {macro['compound_events']}")
    
    def _print_clustering_analysis(self, analysis):
        """Print formatted macro window clustering analysis."""
        print("\n📊 MACRO WINDOW CLUSTERING PATTERNS")
        print("-" * 50)
        
        for window_name, stats in analysis.items():
            behavior = stats['behavior']
            volatility = stats['volatility'] 
            clustering = stats['clustering']
            
            print(f"\n🎯 {window_name} - {behavior.upper()} ({volatility} vol)")
            print(f"   Events: {clustering['event_count']} across {clustering['date_span']} days")
            print(f"   Unique Timestamps: {clustering['unique_timestamps']}")
            print(f"   Price Levels: {clustering['unique_price_levels']}")
            print(f"   Avg Energy: {clustering['avg_energy_density']:.3f}")
            print(f"   Range Position: {clustering['range_position_mean']:.3f} ± {clustering['range_position_std']:.3f}")
            print(f"   Exact 40% Hits: {clustering['exact_40_hits']}")
            print(f"   Near 40% Hits: {clustering['near_40_hits']}")
            
            # Top session types
            sessions = stats['session_distribution']
            if sessions:
                top_session = max(sessions, key=sessions.get)
                print(f"   Primary Session: {top_session} ({sessions[top_session]} events)")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive volatility analysis report."""
        print("\n🚀 GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        # Run all analyses
        session_patterns = self.analyze_session_patterns()
        window_clustering = self.analyze_macro_window_clustering() 
        timestamp_patterns = self.analyze_timestamp_distributions()
        
        # Generate insights
        self._generate_insights(session_patterns, window_clustering)
        
        return True
    
    def _generate_insights(self, session_patterns, window_clustering):
        """Generate key insights from the analysis."""
        print("\n💡 KEY INSIGHTS & BEHAVIORAL DISCOVERIES")
        print("=" * 60)
        
        # Macro window hit rate analysis
        total_events = len(self.df)
        macro_events = len(self.macro_windows)
        hit_rate = macro_events / total_events
        
        print(f"📈 Macro Window Hit Rate: {hit_rate:.1%} ({macro_events}/{total_events})")
        
        # Highest energy sessions
        if session_patterns:
            energy_sessions = [(k, v['energy']['mean_energy']) for k, v in session_patterns.items()]
            energy_sessions.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n⚡ Highest Energy Sessions:")
            for session, energy in energy_sessions[:3]:
                behavior = session_patterns[session]['behavior'] 
                print(f"   {session}: {energy:.3f} ({behavior})")
        
        # Most active macro windows
        if window_clustering:
            window_events = [(k, v['clustering']['event_count']) for k, v in window_clustering.items()]
            window_events.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🎯 Most Active Macro Windows:")
            for window, count in window_events[:5]:
                behavior = window_clustering[window]['behavior']
                volatility = window_clustering[window]['volatility']
                print(f"   {window}: {count} events ({behavior}, {volatility} vol)")
        
        print(f"\n✅ Analysis complete. Ready for Phase II implementation.")


def main():
    """Main execution function for macro window volatility analysis."""
    
    # Data path from H8 research results
    data_path = "/Users/jack/IRONFORGE/runs/RUN_20250824_182221_NEWSCLUST_3P/artifacts/macro_window_confluence_analysis.csv"
    
    # Initialize analyzer
    analyzer = MacroWindowVolatilityAnalyzer(data_path)
    
    # Generate comprehensive report
    success = analyzer.generate_comprehensive_report()
    
    if success:
        print("\n🎉 Macro Window Volatility Analysis Complete!")
        return 0
    else:
        print("\n❌ Analysis failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())