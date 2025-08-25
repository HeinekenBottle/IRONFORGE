#!/usr/bin/env python3
"""
PO3 Macro Window Classifier - Phase II Analysis
Classifies 89 macro windows by PO3 phases using news amplification baseline

PO3 Classification Framework:
- Accumulation: Low amplification, quiet news periods, setup positioning
- Manipulation: High news amplification, volatility expansion, institutional moves  
- Distribution: Mixed amplification, trend continuation, closing patterns

Uses H8 compound amplification results as baseline:
- Baseline: 17.5x amplification
- Macro-only: 16.0x amplification (sub-baseline = accumulation)
- News-only: 51.0x amplification (3x baseline = manipulation)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from collections import defaultdict, Counter
import sys
from pathlib import Path


class PO3MacroWindowClassifier:
    """
    Classifies macro windows by PO3 phases using news amplification baseline.
    
    Classification Logic:
    - Accumulation: amplification < baseline (17.5x), quiet periods, consolidation
    - Manipulation: amplification > 2x baseline (35x), high news activity, expansion
    - Distribution: baseline to 2x baseline, mixed signals, continuation/closing
    """
    
    def __init__(self, confluence_path, amplification_path):
        """Initialize classifier with confluence and amplification data."""
        self.confluence_path = confluence_path
        self.amplification_path = amplification_path
        self.confluence_df = None
        self.amplification_df = None
        self.macro_windows = None
        
        # PO3 classification thresholds (from H8 amplification results)
        self.baseline_amplification = 17.53  # Baseline events
        self.accumulation_threshold = self.baseline_amplification  # Below baseline
        self.manipulation_threshold = 2 * self.baseline_amplification  # 35x+
        
        # PO3 phase definitions
        self.po3_phases = {
            'ACCUMULATION': {
                'amplification_range': (0, self.accumulation_threshold),
                'news_pattern': 'quiet',
                'behavior': 'setup_positioning',
                'volatility': 'low_medium',
                'description': 'Institutional accumulation and positioning'
            },
            'MANIPULATION': {
                'amplification_range': (self.manipulation_threshold, 100),
                'news_pattern': 'high_impact',
                'behavior': 'volatility_expansion',
                'volatility': 'high',
                'description': 'News-driven manipulation and expansion'
            },
            'DISTRIBUTION': {
                'amplification_range': (self.accumulation_threshold, self.manipulation_threshold),
                'news_pattern': 'mixed',
                'behavior': 'trend_continuation',
                'volatility': 'medium',
                'description': 'Trend continuation and distribution'
            }
        }
        
        # Macro window to PO3 mapping based on ICT framework
        self.window_po3_mapping = {
            'LONDON': 'MANIPULATION',  # Institutional manipulation
            'PRE_MKT_1': 'ACCUMULATION',  # Pre-market positioning
            'PRE_MKT_2': 'ACCUMULATION',  # Pre-market setup
            'AM_1': 'MANIPULATION',  # Opening manipulation
            'AM_2': 'MANIPULATION',  # Morning expansion
            'AM_3': 'DISTRIBUTION',  # Morning continuation
            'AM_4': 'DISTRIBUTION',  # Transition to lunch
            'LUNCH_COUNTER': 'ACCUMULATION',  # Counter-trend setup
            'PM_1': 'MANIPULATION',  # Afternoon manipulation
            'PM_2': 'MANIPULATION',  # Afternoon expansion
            'PM_3': 'DISTRIBUTION',  # Afternoon continuation
            'PM_4': 'DISTRIBUTION'  # Closing distribution
        }
        
    def load_data(self):
        """Load confluence and amplification data."""
        try:
            # Load confluence analysis
            self.confluence_df = pd.read_csv(self.confluence_path)
            self.macro_windows = self.confluence_df[
                self.confluence_df['in_macro_window'] == True
            ].copy()
            
            # Load amplification results  
            self.amplification_df = pd.read_csv(self.amplification_path)
            
            print(f"✅ Loaded {len(self.confluence_df)} confluence events")
            print(f"✅ Found {len(self.macro_windows)} macro window events")
            print(f"✅ Loaded {len(self.amplification_df)} amplification results")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def classify_po3_phases(self):
        """Classify macro windows by PO3 phases using news amplification."""
        print("\n🔍 CLASSIFYING MACRO WINDOWS BY PO3 PHASES")
        print("=" * 60)
        
        po3_classifications = {}
        
        # Group by macro window for classification
        for window_name in self.macro_windows['macro_window_name'].unique():
            if pd.isna(window_name):
                continue
                
            window_data = self.macro_windows[
                self.macro_windows['macro_window_name'] == window_name
            ]
            
            # Calculate amplification metrics
            amplification_metrics = self._calculate_window_amplification(window_data)
            
            # Classify PO3 phase
            po3_phase = self._determine_po3_phase(window_name, amplification_metrics, window_data)
            
            # Compile classification results
            po3_classifications[window_name] = {
                'po3_phase': po3_phase,
                'amplification_score': amplification_metrics['effective_amplification'],
                'news_activity': amplification_metrics['news_activity_score'],
                'event_count': len(window_data),
                'session_distribution': window_data['session_type'].value_counts().to_dict(),
                'compound_distribution': window_data['compound_type'].value_counts().to_dict(),
                'energy_metrics': {
                    'mean_energy': window_data['energy_density'].mean(),
                    'energy_volatility': window_data['energy_density'].std()
                },
                'range_clustering': {
                    'mean_position': window_data['range_position'].mean(),
                    'position_std': window_data['range_position'].std(),
                    'near_40_count': len(window_data[abs(window_data['range_position'] - 0.4) < 0.025])
                }
            }
        
        self._print_po3_classifications(po3_classifications)
        return po3_classifications
    
    def _calculate_window_amplification(self, window_data):
        """Calculate amplification metrics for a macro window."""
        # News activity scoring
        news_buckets = window_data['news_bucket'].value_counts()
        news_activity_score = 0
        
        if 'high±120m' in news_buckets:
            news_activity_score += news_buckets['high±120m'] * 3  # High impact
        if 'medium±60m' in news_buckets:  
            news_activity_score += news_buckets['medium±60m'] * 2  # Medium impact
        if 'low±30m' in news_buckets:
            news_activity_score += news_buckets['low±30m'] * 1  # Low impact
        
        # Event density and clustering
        event_count = len(window_data)
        unique_sessions = len(window_data['session_type'].unique())
        temporal_density = event_count / max(unique_sessions, 1)
        
        # Range position clustering (closer to 0.4 = higher precision)
        range_precision = 1 - window_data['range_position'].apply(lambda x: abs(x - 0.4)).mean()
        
        # Compound event ratio
        compound_events = len(window_data[window_data['compound_type'] != 'baseline'])
        compound_ratio = compound_events / event_count if event_count > 0 else 0
        
        # Effective amplification calculation
        base_amplification = self.baseline_amplification
        news_multiplier = 1 + (news_activity_score / event_count) if event_count > 0 else 1
        density_multiplier = min(temporal_density / 2, 2.0)  # Cap at 2x
        precision_multiplier = 1 + range_precision
        
        effective_amplification = (
            base_amplification * 
            news_multiplier * 
            density_multiplier * 
            precision_multiplier
        )
        
        return {
            'effective_amplification': effective_amplification,
            'news_activity_score': news_activity_score,
            'temporal_density': temporal_density,
            'range_precision': range_precision,
            'compound_ratio': compound_ratio,
            'news_multiplier': news_multiplier
        }
    
    def _determine_po3_phase(self, window_name, metrics, window_data):
        """Determine PO3 phase based on amplification and window characteristics."""
        amplification = metrics['effective_amplification']
        news_activity = metrics['news_activity_score']
        
        # Primary classification by amplification
        if amplification >= self.manipulation_threshold:
            primary_phase = 'MANIPULATION'
        elif amplification <= self.accumulation_threshold:
            primary_phase = 'ACCUMULATION'
        else:
            primary_phase = 'DISTRIBUTION'
        
        # Secondary validation by ICT window mapping
        ict_phase = self.window_po3_mapping.get(window_name, primary_phase)
        
        # News activity validation
        if news_activity > len(window_data):  # High news activity
            if primary_phase == 'ACCUMULATION':
                # High news + low amp = potential manipulation setup
                validated_phase = 'MANIPULATION'
            else:
                validated_phase = primary_phase
        elif news_activity == 0:  # No news activity
            if primary_phase == 'MANIPULATION':
                # No news + high amp = likely distribution
                validated_phase = 'DISTRIBUTION'
            else:
                validated_phase = primary_phase
        else:
            validated_phase = primary_phase
        
        # Final phase determination (ICT framework takes precedence for edge cases)
        if abs(amplification - self.baseline_amplification) < 5:  # Close to baseline
            final_phase = ict_phase
        else:
            final_phase = validated_phase
        
        return final_phase
    
    def analyze_po3_distributions(self, classifications):
        """Analyze PO3 phase distributions across macro windows."""
        print("\n🔍 PO3 PHASE DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        # Phase distribution
        phase_counts = Counter([c['po3_phase'] for c in classifications.values()])
        total_windows = len(classifications)
        
        print(f"📊 PO3 Phase Distribution ({total_windows} macro windows):")
        for phase, count in phase_counts.items():
            percentage = count / total_windows * 100
            description = self.po3_phases[phase]['description']
            print(f"   {phase}: {count} windows ({percentage:.1f}%) - {description}")
        
        # Amplification analysis by phase
        print(f"\n📈 Amplification Analysis by PO3 Phase:")
        for phase in ['ACCUMULATION', 'MANIPULATION', 'DISTRIBUTION']:
            phase_windows = [w for w in classifications.values() if w['po3_phase'] == phase]
            if not phase_windows:
                continue
                
            amplifications = [w['amplification_score'] for w in phase_windows]
            mean_amp = np.mean(amplifications)
            std_amp = np.std(amplifications)
            
            print(f"   {phase}: {mean_amp:.1f}x ± {std_amp:.1f}x amplification")
        
        # Session type analysis by phase
        print(f"\n🎯 Session Distribution by PO3 Phase:")
        for phase in phase_counts.keys():
            phase_windows = [w for w in classifications.values() if w['po3_phase'] == phase]
            
            # Aggregate session distributions
            session_totals = defaultdict(int)
            for window in phase_windows:
                for session, count in window['session_distribution'].items():
                    session_totals[session] += count
            
            if session_totals:
                top_session = max(session_totals, key=session_totals.get)
                total_events = sum(session_totals.values())
                print(f"   {phase}: {total_events} events, primary session: {top_session}")
        
        return {
            'phase_distribution': dict(phase_counts),
            'total_windows': total_windows
        }
    
    def _print_po3_classifications(self, classifications):
        """Print formatted PO3 classification results."""
        print("\n📊 PO3 MACRO WINDOW CLASSIFICATIONS")
        print("-" * 60)
        
        # Sort by PO3 phase for organized output
        sorted_classifications = sorted(
            classifications.items(),
            key=lambda x: (x[1]['po3_phase'], x[1]['amplification_score']),
            reverse=True
        )
        
        current_phase = None
        for window_name, data in sorted_classifications:
            if data['po3_phase'] != current_phase:
                current_phase = data['po3_phase']
                phase_desc = self.po3_phases[current_phase]['description']
                print(f"\n🎯 {current_phase} PHASE - {phase_desc}")
                print("-" * 40)
            
            amp_score = data['amplification_score']
            event_count = data['event_count']
            news_activity = data['news_activity']
            
            print(f"   {window_name}: {amp_score:.1f}x amplification ({event_count} events)")
            print(f"     News Activity: {news_activity}, Range Clustering: {data['range_clustering']['near_40_count']}")
    
    def generate_po3_classification_report(self):
        """Generate comprehensive PO3 classification report."""
        print("\n🚀 GENERATING PO3 CLASSIFICATION REPORT")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        # Run PO3 classification
        classifications = self.classify_po3_phases()
        
        # Analyze distributions
        distribution_analysis = self.analyze_po3_distributions(classifications)
        
        # Generate insights
        self._generate_po3_insights(classifications, distribution_analysis)
        
        return classifications
    
    def _generate_po3_insights(self, classifications, distribution_analysis):
        """Generate key insights from PO3 classification."""
        print("\n💡 KEY PO3 CLASSIFICATION INSIGHTS")
        print("=" * 60)
        
        phase_dist = distribution_analysis['phase_distribution']
        total_windows = distribution_analysis['total_windows']
        
        # Most dominant phase
        dominant_phase = max(phase_dist, key=phase_dist.get)
        dominant_count = phase_dist[dominant_phase]
        dominant_pct = dominant_count / total_windows * 100
        
        print(f"🎯 Dominant Phase: {dominant_phase} ({dominant_count}/{total_windows} windows, {dominant_pct:.1f}%)")
        
        # Highest amplification windows
        high_amp_windows = sorted(
            [(name, data['amplification_score']) for name, data in classifications.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\n⚡ Highest Amplification Macro Windows:")
        for window, amp in high_amp_windows[:5]:
            phase = classifications[window]['po3_phase']
            print(f"   {window}: {amp:.1f}x amplification ({phase})")
        
        # Phase-specific insights
        for phase, count in phase_dist.items():
            phase_windows = [w for w in classifications.values() if w['po3_phase'] == phase]
            avg_events = np.mean([w['event_count'] for w in phase_windows])
            
            phase_desc = self.po3_phases[phase]['description']
            print(f"\n📊 {phase} ({count} windows): {phase_desc}")
            print(f"   Average Events per Window: {avg_events:.1f}")
        
        print(f"\n✅ PO3 Classification complete. {total_windows} macro windows classified.")


def main():
    """Main execution function for PO3 macro window classification."""
    
    # Data paths from H8 research results
    confluence_path = "/Users/jack/IRONFORGE/runs/RUN_20250824_182221_NEWSCLUST_3P/artifacts/macro_window_confluence_analysis.csv"
    amplification_path = "/Users/jack/IRONFORGE/runs/RUN_20250824_182221_NEWSCLUST_3P/artifacts/compound_amplification_results.csv"
    
    # Initialize classifier
    classifier = PO3MacroWindowClassifier(confluence_path, amplification_path)
    
    # Generate PO3 classification report
    classifications = classifier.generate_po3_classification_report()
    
    if classifications:
        print("\n🎉 PO3 Macro Window Classification Complete!")
        return 0
    else:
        print("\n❌ Classification failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())