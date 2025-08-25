#!/usr/bin/env python3
"""
Macro Window News Resonance Analysis - H8 Framework Implementation
Tests whether news events during ICT macro windows create compound amplification >25x baseline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MacroWindow:
    """ICT Macro Window Definition"""
    name: str
    start_time: str  # HH:MM format in ET
    end_time: str    # HH:MM format in ET
    
    def contains_time(self, time_str: str) -> bool:
        """Check if given time falls within this macro window"""
        try:
            # Parse time from various formats
            if 'ET' in time_str:
                # Format: "2025-07-31 09:30:00 ET"
                time_part = time_str.split(' ')[1]  # Get "09:30:00" part
            elif 'T' in time_str:
                # ISO format: "2025-07-31T09:30:00"
                time_part = time_str.split('T')[1]
            else:
                # Just time: "09:30:00"
                time_part = time_str
            
            # Extract HH:MM from time_part
            time_hm = time_part[:5]  # First 5 chars for HH:MM
            
            # Handle cross-midnight windows for Asia session
            if self.name == "ASIA" and self.start_time > self.end_time:
                # Window crosses midnight (19:50 to 20:10 next day)
                return time_hm >= self.start_time or time_hm <= self.end_time
            else:
                # Normal window within same day
                return self.start_time <= time_hm <= self.end_time
        except Exception as e:
            print(f"Warning: Error parsing time '{time_str}': {e}")
            return False
    
    def get_phase_classification(self, event_time: str, news_time: str) -> Optional[str]:
        """
        Classify news event relative to macro window
        Returns: 'PRE-macro', 'EXACT-macro', 'POST-macro', or None
        """
        try:
            # Parse times
            event_dt = pd.to_datetime(event_time)
            news_dt = pd.to_datetime(news_time)
            
            # Calculate difference in minutes
            diff_mins = (news_dt - event_dt).total_seconds() / 60
            
            # Check if event is in macro window
            if self.contains_time(event_time):
                # Classify news relative to macro window
                if -30 <= diff_mins <= -10:
                    return 'PRE-macro'
                elif -10 <= diff_mins <= 10:
                    return 'EXACT-macro'  
                elif 0 <= diff_mins <= 15:
                    return 'POST-macro'
            
            return None
        except:
            return None

class MacroWindowNewsAnalyzer:
    """Analyzes compound amplification of news events during ICT macro windows"""
    
    def __init__(self):
        # Complete ICT Macro Windows Framework (ET timezone)
        self.macro_windows = [
            # Asia Session
            MacroWindow("ASIA", "19:50", "20:10"),         # Asia Major
            
            # London Session
            MacroWindow("LONDON", "02:50", "03:10"),       # London Open
            
            # Pre-Market Windows
            MacroWindow("PRE_MKT_1", "06:50", "07:10"),    # Pre-Market Early
            MacroWindow("PRE_MKT_2", "07:50", "08:10"),    # Pre-Market Late
            
            # AM Session Windows
            MacroWindow("AM_1", "08:50", "09:10"),         # AM Opening
            MacroWindow("AM_2", "09:50", "10:10"),         # AM Major (most significant)
            MacroWindow("AM_3", "10:50", "11:10"),         # AM Late
            MacroWindow("AM_4", "11:50", "12:10"),         # AM Final
            
            # Lunch Counter-Trend Window (extended)
            MacroWindow("LUNCH_COUNTER", "11:30", "13:30"), # Lunch Counter-Trend
            
            # PM Session Windows
            MacroWindow("PM_1", "13:50", "14:10"),         # PM Opening
            MacroWindow("PM_2", "14:50", "15:10"),         # PM Mid
            MacroWindow("PM_3", "15:15", "15:45"),         # PM Power Hour Start
            MacroWindow("PM_4", "15:50", "16:10"),         # PM Close
        ]
        
        # Statistical parameters
        self.confidence_level = 0.95
        self.fdr_alpha = 0.05
        
    def load_enhanced_session_data(self, data_dir: str = "/Users/jack/IRONFORGE/data/day_news_enhanced") -> List[Dict]:
        """Load and process enhanced session data"""
        
        session_files = glob.glob(f"{data_dir}/day_news_*.json")
        all_events = []
        
        print(f"📁 Loading {len(session_files)} enhanced session files...")
        
        for file_path in session_files:
            try:
                with open(file_path, 'r') as f:
                    session_data = json.load(f)
                
                events = session_data.get("events", [])
                session_info = session_data.get("session_info", {})
                
                # Extract RD@40 events with news context
                for event in events:
                    range_pos = event.get('range_position', 0.5)
                    
                    # Target RD@40 archaeological zone events
                    if abs(range_pos - 0.40) <= 0.025:
                        event_data = self._extract_event_features(event, session_data)
                        if event_data:
                            all_events.append(event_data)
                            
            except Exception as e:
                print(f"⚠️ Error processing {Path(file_path).name}: {e}")
                continue
        
        print(f"✅ Loaded {len(all_events)} RD@40 events for analysis")
        return all_events
    
    def _extract_event_features(self, event: Dict, session_data: Dict) -> Optional[Dict]:
        """Extract relevant features for macro window analysis"""
        
        try:
            # Base event data
            event_data = {
                'timestamp': event.get('timestamp', ''),
                'timestamp_et': event.get('timestamp_et', ''),
                'range_position': event.get('range_position', 0.5),
                'energy_density': event.get('energy_density', 0.5),
                'price_level': event.get('price_level', 0),
                'magnitude': event.get('magnitude', 0),
            }
            
            # Session context
            session_info = session_data.get("session_info", {})
            event_data.update({
                'session_id': session_info.get('session_id', 'unknown'),
                'session_type': session_info.get('session_type', 'unknown'),
                'trading_day': session_info.get('trading_day', ''),
            })
            
            # News context
            news_context = event.get('news_context', {})
            real_news = event.get('real_news_context', {})
            
            if real_news:
                # Use real news data if available
                event_data.update({
                    'news_bucket': real_news.get('news_bucket', 'quiet'),
                    'news_event_time': real_news.get('event_time_utc', ''),
                    'news_distance_mins': real_news.get('news_distance_mins', 999),
                    'impact_level': real_news.get('impact_level', 'low'),
                    'has_news': real_news.get('news_bucket', 'quiet') != 'quiet'
                })
            else:
                # Fallback to synthetic news
                event_data.update({
                    'news_bucket': news_context.get('news_bucket', 'quiet'),
                    'news_event_time': '',
                    'news_distance_mins': 999,
                    'impact_level': 'low',
                    'has_news': news_context.get('news_bucket', 'quiet') != 'quiet'
                })
            
            # Day context
            day_context = event.get('day_context', {})
            event_data['day_of_week'] = day_context.get('day_of_week', 'unknown')
            
            return event_data
            
        except Exception as e:
            print(f"⚠️ Error extracting event features: {e}")
            return None
    
    def classify_macro_window_events(self, events: List[Dict]) -> pd.DataFrame:
        """Classify events by macro window timing and news confluence"""
        
        print("🔍 Classifying events by macro window timing...")
        print(f"Testing against {len(self.macro_windows)} macro windows:")
        for w in self.macro_windows:
            print(f"  {w.name}: {w.start_time}-{w.end_time}")
        
        classified_events = []
        macro_window_hits = {}
        
        for event in events:
            event_time = event.get('timestamp_et', event.get('timestamp', ''))
            news_time = event.get('news_event_time', '')
            
            # Base classification
            event_class = {
                **event,
                'in_macro_window': False,
                'macro_window_name': None,
                'news_phase': None,
                'compound_type': 'baseline'
            }
            
            # Check each macro window
            for window in self.macro_windows:
                if window.contains_time(event_time):
                    event_class['in_macro_window'] = True
                    event_class['macro_window_name'] = window.name
                    
                    # Track hits per window
                    macro_window_hits[window.name] = macro_window_hits.get(window.name, 0) + 1
                    
                    # Classify news phase if news exists
                    if event['has_news'] and news_time:
                        phase = window.get_phase_classification(event_time, news_time)
                        if phase:
                            event_class['news_phase'] = phase
                            event_class['compound_type'] = f"macro+{phase}"
                        else:
                            event_class['compound_type'] = 'macro+news_other'
                    else:
                        event_class['compound_type'] = 'macro_only'
                    break
            
            # Non-macro events
            if not event_class['in_macro_window']:
                if event['has_news']:
                    event_class['compound_type'] = 'news_only'
                # else remains 'baseline'
            
            classified_events.append(event_class)
        
        df = pd.DataFrame(classified_events)
        
        # Summary statistics
        print(f"\n📊 Classification Summary:")
        print(f"Total events: {len(df)}")
        print(f"Macro window events: {df['in_macro_window'].sum()}")
        print(f"News events: {df['has_news'].sum()}")
        print(f"Compound events (macro+news): {len(df[df['compound_type'].str.contains('macro\\+', regex=True)])}")
        
        # Report macro window hits
        if macro_window_hits:
            print(f"\n📍 Macro Window Hits:")
            for window, count in sorted(macro_window_hits.items()):
                print(f"  {window}: {count} events")
        else:
            print(f"\n⚠️ No macro window hits detected")
        
        print(f"\nCompound type distribution:")
        print(df['compound_type'].value_counts())
        
        return df
    
    def calculate_clustering_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate clustering strength for different compound types"""
        
        print("\n🧮 Calculating clustering strength by compound type...")
        
        results = []
        
        # Get baseline clustering strength (quiet periods)
        baseline_subset = df[df['compound_type'] == 'baseline'].copy()
        baseline_density = self._calculate_temporal_density(baseline_subset)
        
        if baseline_density == 0:
            baseline_density = 0.01  # Prevent division by zero
        
        # Define analysis groups
        compound_types = df['compound_type'].unique()
        
        for compound_type in compound_types:
            subset = df[df['compound_type'] == compound_type].copy()
            
            if len(subset) == 0:
                continue
                
            # Calculate temporal clustering density
            temporal_density = self._calculate_temporal_density(subset)
            
            # Calculate clustering coefficient (how tightly events cluster)
            clustering_coefficient = self._calculate_clustering_coefficient(subset)
            
            # Calculate event concentration (events per unique time window)
            time_concentration = self._calculate_time_concentration(subset)
            
            # Composite clustering strength (weighted average of metrics)
            clustering_strength = (
                0.4 * temporal_density + 
                0.4 * clustering_coefficient + 
                0.2 * time_concentration
            )
            
            # Calculate amplification factor vs baseline
            amplification_factor = clustering_strength / baseline_density if baseline_density > 0 else 1.0
            
            # Statistical confidence interval (bootstrap estimation)
            ci_lower, ci_upper = self._calculate_confidence_interval(subset, baseline_subset)
            
            # Calculate p-value using permutation test
            p_value = self._permutation_test(subset, baseline_subset)
            
            results.append({
                'compound_type': compound_type,
                'event_count': len(subset),
                'temporal_density': temporal_density,
                'clustering_coefficient': clustering_coefficient,
                'time_concentration': time_concentration,
                'clustering_strength': clustering_strength,
                'amplification_factor': amplification_factor,
                'baseline_ratio': amplification_factor,
                'confidence_interval': f"[{ci_lower:.1f}, {ci_upper:.1f}]",
                'p_value': p_value,
                'statistical_significance': 'significant' if p_value < 0.05 else 'not_significant'
            })
        
        results_df = pd.DataFrame(results)
        
        # Apply FDR correction for multiple testing
        if len(results_df) > 1:
            p_values = results_df['p_value'].values
            _, p_corrected, _, _ = multipletests(p_values, alpha=self.fdr_alpha, method='fdr_bh')
            results_df['p_value_corrected'] = p_corrected
            results_df['fdr_significant'] = p_corrected < self.fdr_alpha
        
        return results_df
    
    def _calculate_temporal_density(self, subset: pd.DataFrame) -> float:
        """Calculate temporal density (events per minute within active periods)"""
        if len(subset) <= 1:
            return 0.0
            
        try:
            # Convert timestamps to datetime
            timestamps = pd.to_datetime(subset['timestamp_et'].fillna(subset['timestamp']))
            
            # Calculate time span in minutes
            time_span_minutes = (timestamps.max() - timestamps.min()).total_seconds() / 60
            
            if time_span_minutes <= 0:
                return len(subset)  # All events at same time = high density
                
            # Events per minute
            density = len(subset) / time_span_minutes
            return density
            
        except:
            return 0.0
    
    def _calculate_clustering_coefficient(self, subset: pd.DataFrame) -> float:
        """Calculate how tightly events cluster (inverse of temporal variance)"""
        if len(subset) <= 1:
            return 1.0
            
        try:
            # Convert to timestamps
            timestamps = pd.to_datetime(subset['timestamp_et'].fillna(subset['timestamp']))
            
            # Calculate inter-event intervals in minutes
            intervals = timestamps.sort_values().diff().dt.total_seconds() / 60
            intervals = intervals.dropna()
            
            if len(intervals) == 0:
                return 1.0
                
            # Clustering coefficient = 1 / (1 + coefficient_of_variation)
            cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else 0
            clustering_coef = 1.0 / (1.0 + cv)
            
            return clustering_coef
            
        except:
            return 0.0
    
    def _calculate_time_concentration(self, subset: pd.DataFrame) -> float:
        """Calculate event concentration within time windows"""
        if len(subset) <= 1:
            return 1.0
            
        try:
            # Group events by 5-minute time windows
            timestamps = pd.to_datetime(subset['timestamp_et'].fillna(subset['timestamp']))
            
            # Round to 5-minute intervals
            rounded_times = timestamps.dt.floor('5min')
            time_counts = rounded_times.value_counts()
            
            # Concentration = max events in single window / total events
            max_in_window = time_counts.max()
            concentration = max_in_window / len(subset)
            
            return concentration
            
        except:
            return 0.0
    
    def _calculate_confidence_interval(self, subset: pd.DataFrame, baseline: pd.DataFrame) -> Tuple[float, float]:
        """Calculate confidence interval for amplification factor using bootstrap"""
        try:
            n_bootstrap = 1000
            amplifications = []
            
            baseline_density = self._calculate_temporal_density(baseline)
            if baseline_density == 0:
                baseline_density = 0.01
                
            for _ in range(n_bootstrap):
                # Bootstrap sample
                if len(subset) > 0:
                    bootstrap_sample = subset.sample(n=len(subset), replace=True)
                    sample_density = self._calculate_temporal_density(bootstrap_sample)
                    amplification = sample_density / baseline_density
                    amplifications.append(amplification)
            
            if amplifications:
                ci_lower = np.percentile(amplifications, 2.5)
                ci_upper = np.percentile(amplifications, 97.5)
                return ci_lower, ci_upper
            else:
                return 1.0, 1.0
                
        except:
            return 1.0, 1.0
    
    def _permutation_test(self, subset: pd.DataFrame, baseline: pd.DataFrame, n_permutations: int = 1000) -> float:
        """Calculate p-value using permutation test"""
        try:
            if len(subset) == 0 or len(baseline) == 0:
                return 1.0
                
            # Observed difference
            subset_density = self._calculate_temporal_density(subset)
            baseline_density = self._calculate_temporal_density(baseline)
            observed_diff = subset_density - baseline_density
            
            # Combine datasets
            combined = pd.concat([subset, baseline])
            
            # Permutation test
            greater_count = 0
            
            for _ in range(n_permutations):
                # Randomly shuffle labels
                shuffled = combined.sample(frac=1).reset_index(drop=True)
                perm_subset = shuffled.iloc[:len(subset)]
                perm_baseline = shuffled.iloc[len(subset):]
                
                # Calculate difference
                perm_subset_density = self._calculate_temporal_density(perm_subset)
                perm_baseline_density = self._calculate_temporal_density(perm_baseline)
                perm_diff = perm_subset_density - perm_baseline_density
                
                if perm_diff >= observed_diff:
                    greater_count += 1
            
            p_value = greater_count / n_permutations
            return p_value
            
        except:
            return 1.0
    
    def test_h8_hypotheses(self, df: pd.DataFrame, clustering_results: pd.DataFrame) -> Dict:
        """Test specific H8 framework hypotheses"""
        
        print("\n🧪 Testing H8 Framework Hypotheses...")
        
        # Extract amplification factors for each hypothesis
        h8_results = {}
        
        # H8.1: Direct confluence >40x amplification (EXACT-macro timing)
        exact_macro_results = clustering_results[
            clustering_results['compound_type'] == 'macro+EXACT-macro'
        ]
        
        if not exact_macro_results.empty:
            h8_1_amplification = exact_macro_results['amplification_factor'].iloc[0]
            h8_1_pvalue = exact_macro_results['p_value'].iloc[0] if 'p_value' in exact_macro_results.columns else 1.0
            h8_results['H8.1'] = {
                'hypothesis': 'Direct confluence >40x amplification',
                'measured_amplification': h8_1_amplification,
                'target_amplification': 40,
                'result': 'CONFIRMED' if h8_1_amplification > 40 else 'NOT_CONFIRMED',
                'p_value': h8_1_pvalue,
                'effect_size': h8_1_amplification / 40
            }
        else:
            h8_results['H8.1'] = {
                'hypothesis': 'Direct confluence >40x amplification',
                'measured_amplification': 0,
                'target_amplification': 40,
                'result': 'NO_DATA',
                'p_value': 1.0,
                'effect_size': 0
            }
        
        # H8.2: PRE-macro setup 35x amplification
        pre_macro_results = clustering_results[
            clustering_results['compound_type'] == 'macro+PRE-macro'
        ]
        
        if not pre_macro_results.empty:
            h8_2_amplification = pre_macro_results['amplification_factor'].iloc[0]
            h8_2_pvalue = pre_macro_results['p_value'].iloc[0] if 'p_value' in pre_macro_results.columns else 1.0
            h8_results['H8.2'] = {
                'hypothesis': 'PRE-macro setup 35x amplification',
                'measured_amplification': h8_2_amplification,
                'target_amplification': 35,
                'result': 'CONFIRMED' if h8_2_amplification > 35 else 'NOT_CONFIRMED',
                'p_value': h8_2_pvalue,
                'effect_size': h8_2_amplification / 35
            }
        else:
            h8_results['H8.2'] = {
                'hypothesis': 'PRE-macro setup 35x amplification',
                'measured_amplification': 0,
                'target_amplification': 35,
                'result': 'NO_DATA',
                'p_value': 1.0,
                'effect_size': 0
            }
        
        # H8.3: POST-macro expansion 30x amplification
        post_macro_results = clustering_results[
            clustering_results['compound_type'] == 'macro+POST-macro'
        ]
        
        if not post_macro_results.empty:
            h8_3_amplification = post_macro_results['amplification_factor'].iloc[0]
            h8_3_pvalue = post_macro_results['p_value'].iloc[0] if 'p_value' in post_macro_results.columns else 1.0
            h8_results['H8.3'] = {
                'hypothesis': 'POST-macro expansion 30x amplification',
                'measured_amplification': h8_3_amplification,
                'target_amplification': 30,
                'result': 'CONFIRMED' if h8_3_amplification > 30 else 'NOT_CONFIRMED',
                'p_value': h8_3_pvalue,
                'effect_size': h8_3_amplification / 30
            }
        else:
            h8_results['H8.3'] = {
                'hypothesis': 'POST-macro expansion 30x amplification',
                'measured_amplification': 0,
                'target_amplification': 30,
                'result': 'NO_DATA',
                'p_value': 1.0,
                'effect_size': 0
            }
        
        # H8.4: Counter-trend lunch window reverse correlation <5x amplification
        lunch_counter_results = clustering_results[
            clustering_results['compound_type'].str.contains('LUNCH_COUNTER', na=False)
        ]
        
        if not lunch_counter_results.empty:
            h8_4_amplification = lunch_counter_results['amplification_factor'].iloc[0]
            h8_4_pvalue = lunch_counter_results['p_value'].iloc[0] if 'p_value' in lunch_counter_results.columns else 1.0
            h8_results['H8.4'] = {
                'hypothesis': 'Counter-trend lunch window <5x amplification',
                'measured_amplification': h8_4_amplification,
                'target_amplification': 5,
                'result': 'CONFIRMED' if h8_4_amplification < 5 else 'NOT_CONFIRMED',
                'p_value': h8_4_pvalue,
                'effect_size': h8_4_amplification / 5
            }
        else:
            h8_results['H8.4'] = {
                'hypothesis': 'Counter-trend lunch window <5x amplification',
                'measured_amplification': 0,
                'target_amplification': 5,
                'result': 'NO_DATA',
                'p_value': 1.0,
                'effect_size': 0
            }
        
        # Additional analysis: Identify highest amplification window
        macro_only_results = clustering_results[
            clustering_results['compound_type'].str.contains('macro', na=False)
        ]
        
        if not macro_only_results.empty:
            max_amplification = macro_only_results['amplification_factor'].max()
            max_window = macro_only_results.loc[
                macro_only_results['amplification_factor'].idxmax(), 'compound_type'
            ]
            h8_results['MAX_WINDOW'] = {
                'window': max_window,
                'amplification': max_amplification
            }
        
        return h8_results
    
    def generate_analysis_outputs(self, df: pd.DataFrame, clustering_results: pd.DataFrame, 
                                h8_results: Dict) -> None:
        """Generate required analysis output files"""
        
        print("\n📊 Generating analysis output files...")
        
        # Ensure output directory exists
        output_dir = Path("/Users/jack/IRONFORGE/runs/RUN_20250824_182221_NEWSCLUST_3P/artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Raw data analysis
        macro_confluence_path = output_dir / "macro_window_confluence_analysis.csv"
        df.to_csv(macro_confluence_path, index=False)
        print(f"✅ Saved: {macro_confluence_path}")
        
        # 2. Compound amplification results
        amplification_path = output_dir / "compound_amplification_results.csv"
        clustering_results.to_csv(amplification_path, index=False)
        print(f"✅ Saved: {amplification_path}")
        
        # 3. Temporal phase comparison
        phase_df = df[df['news_phase'].notna()].copy()
        if not phase_df.empty:
            phase_comparison = phase_df.groupby(['news_phase', 'compound_type']).agg({
                'range_position': ['count', 'mean', 'std'],
                'energy_density': ['mean', 'std']
            }).round(3)
        else:
            # Create empty dataframe with proper structure
            phase_comparison = pd.DataFrame({
                ('range_position', 'count'): [],
                ('range_position', 'mean'): [],
                ('range_position', 'std'): [],
                ('energy_density', 'mean'): [],
                ('energy_density', 'std'): []
            })
        
        phase_comparison_path = output_dir / "temporal_phase_comparison.csv"
        phase_comparison.to_csv(phase_comparison_path)
        print(f"✅ Saved: {phase_comparison_path}")
        
        # 4. H8 hypothesis test results
        h8_summary = []
        for hypothesis, results in h8_results.items():
            if hypothesis != 'MAX_WINDOW':  # Skip the MAX_WINDOW entry
                h8_summary.append({
                    'hypothesis_id': hypothesis,
                    'description': results.get('hypothesis', f'{hypothesis} hypothesis'),
                    'target_amplification': results.get('target_amplification', 0),
                    'measured_amplification': results.get('measured_amplification', 0),
                    'result': results.get('result', 'NO_DATA'),
                    'effect_size': results.get('effect_size', 0),
                    'p_value': results.get('p_value', 1.0)
                })
        
        h8_results_path = output_dir / "h8_hypothesis_results.csv"
        pd.DataFrame(h8_summary).to_csv(h8_results_path, index=False)
        print(f"✅ Saved: {h8_results_path}")
    
    def run_complete_analysis(self) -> Dict:
        """Run the complete macro window news resonance analysis"""
        
        print("🚀 Starting Macro Window News Resonance Analysis")
        print("=" * 60)
        
        # Load data
        events = self.load_enhanced_session_data()
        
        # Classify events by macro window timing
        classified_df = self.classify_macro_window_events(events)
        
        # Calculate clustering strength
        clustering_results = self.calculate_clustering_strength(classified_df)
        
        # Test H8 hypotheses
        h8_results = self.test_h8_hypotheses(classified_df, clustering_results)
        
        # Generate outputs
        self.generate_analysis_outputs(classified_df, clustering_results, h8_results)
        
        # Summary results
        analysis_summary = {
            'total_events': len(classified_df),
            'macro_window_events': classified_df['in_macro_window'].sum(),
            'compound_events': len(classified_df[classified_df['compound_type'].str.contains('macro\\+', regex=True)]),
            'h8_results': h8_results,
            'clustering_amplification': clustering_results['amplification_factor'].max(),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        print(f"\n🎯 Analysis Complete!")
        print(f"📊 Processed {analysis_summary['total_events']} events")
        print(f"🔍 Found {analysis_summary['compound_events']} compound confluence events")
        print(f"🚀 Maximum amplification: {analysis_summary['clustering_amplification']:.1f}x")
        
        return analysis_summary

def main():
    """Main execution function"""
    analyzer = MacroWindowNewsAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*60)
    print("H8 FRAMEWORK MACRO WINDOW ANALYSIS COMPLETE")
    print("="*60)
    
    # Print H8 results summary
    for hypothesis_id, result in results['h8_results'].items():
        if hypothesis_id != 'MAX_WINDOW':  # Skip MAX_WINDOW entry
            status = "✅ CONFIRMED" if result.get('result') == 'CONFIRMED' else "❌ NOT CONFIRMED"
            amplification = result.get('measured_amplification', 0)
            print(f"{hypothesis_id}: {status} ({amplification:.1f}x)")
    
    # Print maximum window result if available
    if 'MAX_WINDOW' in results['h8_results']:
        max_info = results['h8_results']['MAX_WINDOW']
        print(f"🎯 Highest amplification window: {max_info.get('window', 'N/A')} ({max_info.get('amplification', 0):.1f}x)")

if __name__ == "__main__":
    main()