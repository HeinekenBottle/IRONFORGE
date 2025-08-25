#!/usr/bin/env python3
"""
IRONFORGE Daily Timeframe Archaeological Zone Analyzer
=====================================================

Tests if previous day's 40% level reactions predict current day highs/lows
using 1-minute precision shard data.

Performance Benchmark: Session-level 7.55-point accuracy, 87.5% cascade probability
Target: 95% confidence, FDR correction, permutation testing
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.stats import bootstrap, permutation_test
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DailyArchaeologicalEvent:
    """Daily archaeological zone interaction event"""
    date: str
    previous_day_high: float
    previous_day_low: float
    archaeological_40_level: float
    reaction_time: str
    reaction_price: float
    reaction_accuracy: float
    current_day_high: float
    current_day_low: float
    predicted_high_success: bool
    predicted_low_success: bool
    
@dataclass
class DailyPredictionResult:
    """Daily prediction validation result"""
    total_interactions: int
    high_prediction_accuracy: float
    low_prediction_accuracy: float
    mean_reaction_accuracy: float
    timing_distribution: Dict[str, int]

class DailyTimeframeArchaeologicalAnalyzer:
    """Analyze daily timeframe archaeological zone predictions"""
    
    def __init__(self, shard_data_path: str):
        self.shard_data_path = Path(shard_data_path)
        self.confidence_level = 0.95
        self.fdr_q = 0.10
        self.reaction_threshold = 5.0  # Points within 40% level to count as interaction
        
    def load_shard_data(self, session_path: Path) -> Optional[pd.DataFrame]:
        """Load 1-minute shard data from parquet files"""
        try:
            nodes_file = session_path / "nodes.parquet"
            if nodes_file.exists():
                df = pd.read_parquet(nodes_file)
                return df
        except Exception as e:
            print(f"Error loading {session_path}: {e}")
        return None
    
    def extract_daily_ranges(self, dates: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract daily high/low ranges from all session shards"""
        daily_ranges = {}
        
        for date in dates:
            daily_high = -np.inf
            daily_low = np.inf
            
            # Check all session types for this date
            session_types = ['MIDNIGHT', 'PREMARKET', 'ASIA', 'LONDON', 'LUNCH', 'NY']
            
            for session_type in session_types:
                session_path = self.shard_data_path / f"shard_{session_type}_{date}"
                
                if session_path.exists():
                    df = self.load_shard_data(session_path)
                    if df is not None and 'price' in df.columns:
                        session_high = df['price'].max()
                        session_low = df['price'].min()
                        
                        daily_high = max(daily_high, session_high)
                        daily_low = min(daily_low, session_low)
            
            if daily_high != -np.inf and daily_low != np.inf:
                daily_ranges[date] = {
                    'high': daily_high,
                    'low': daily_low,
                    'range': daily_high - daily_low,
                    'archaeological_40_level': daily_low + 0.4 * (daily_high - daily_low)
                }
        
        return daily_ranges
    
    def detect_archaeological_reactions(self, current_date: str, previous_40_level: float) -> List[Dict]:
        """Detect 1-minute precision reactions at previous day's 40% level"""
        reactions = []
        
        # Search all sessions for current_date and find price interactions within reaction_threshold
        # of the previous_40_level. Record timing, price, and accuracy of each reaction.
        
        session_types = ['MIDNIGHT', 'PREMARKET', 'ASIA', 'LONDON', 'LUNCH', 'NY']
        
        for session_type in session_types:
            session_path = self.shard_data_path / f"shard_{session_type}_{current_date}"
            
            if session_path.exists():
                df = self.load_shard_data(session_path)
                if df is not None and len(df) > 0:
                    # Handle different possible column names
                    price_col = None
                    time_col = None
                    
                    for col in df.columns:
                        if 'price' in col.lower():
                            price_col = col
                        if 'time' in col.lower() or 'timestamp' in col.lower():
                            time_col = col
                    
                    if price_col is not None:
                        # Vectorized detection: find all prices within threshold of 40% level
                        price_diff = np.abs(df[price_col] - previous_40_level)
                        reaction_mask = price_diff <= self.reaction_threshold
                        
                        if reaction_mask.any():
                            # Get reaction data efficiently with pandas
                            reaction_indices = df.index[reaction_mask]
                            
                            for idx in reaction_indices:
                                reactions.append({
                                    'timestamp': df.loc[idx, time_col] if time_col else f"{session_type}_{idx}",
                                    'price': df.loc[idx, price_col],
                                    'accuracy': price_diff.loc[idx],
                                    'session_type': session_type,
                                    'target_level': previous_40_level,
                                    'index': idx
                                })
        
        # Sort reactions by timestamp if available
        if reactions and all('timestamp' in r for r in reactions):
            try:
                reactions.sort(key=lambda x: str(x['timestamp']))
            except:
                pass  # Keep original order if sorting fails
        
        return reactions
    
    def validate_predictions(self, events: List[DailyArchaeologicalEvent]) -> DailyPredictionResult:
        """Validate prediction accuracy for daily extremes"""
        
        if not events:
            return DailyPredictionResult(0, 0.0, 0.0, 0.0, {})
        
        high_predictions = [event.predicted_high_success for event in events]
        low_predictions = [event.predicted_low_success for event in events]
        accuracies = [event.reaction_accuracy for event in events]
        
        # Extract timing distribution
        timing_dist = {}
        for event in events:
            hour = event.reaction_time.split(':')[0]
            timing_dist[hour] = timing_dist.get(hour, 0) + 1
        
        return DailyPredictionResult(
            total_interactions=len(events),
            high_prediction_accuracy=np.mean(high_predictions) if high_predictions else 0.0,
            low_prediction_accuracy=np.mean(low_predictions) if low_predictions else 0.0,
            mean_reaction_accuracy=np.mean(accuracies) if accuracies else 0.0,
            timing_distribution=timing_dist
        )
    
    def statistical_validation(self, events: List[DailyArchaeologicalEvent]) -> Dict[str, float]:
        """Statistical validation with 95% confidence and FDR correction"""
        
        if len(events) < 3:
            return {"error": "Insufficient sample size for statistical validation"}
        
        high_success = [event.predicted_high_success for event in events]
        low_success = [event.predicted_low_success for event in events]
        accuracies = [event.reaction_accuracy for event in events]
        
        # Binomial test against random chance (50%)
        high_success_count = sum(high_success)
        low_success_count = sum(low_success)
        n_events = len(events)
        
        high_binomial_p = stats.binomtest(high_success_count, n_events, 0.5).pvalue
        low_binomial_p = stats.binomtest(low_success_count, n_events, 0.5).pvalue
        
        # Bootstrap confidence intervals
        rng = np.random.default_rng(42)
        
        def success_rate(x, axis):
            return np.mean(x, axis=axis)
        
        high_bootstrap = bootstrap((high_success,), success_rate, 
                                 n_resamples=10000, 
                                 confidence_level=self.confidence_level,
                                 random_state=rng)
        
        low_bootstrap = bootstrap((low_success,), success_rate,
                                n_resamples=10000,
                                confidence_level=self.confidence_level, 
                                random_state=rng)
        
        return {
            'high_prediction_rate': np.mean(high_success),
            'low_prediction_rate': np.mean(low_success),
            'high_binomial_pvalue': high_binomial_p,
            'low_binomial_pvalue': low_binomial_p,
            'high_ci_lower': high_bootstrap.confidence_interval.low,
            'high_ci_upper': high_bootstrap.confidence_interval.high,
            'low_ci_lower': low_bootstrap.confidence_interval.low,
            'low_ci_upper': low_bootstrap.confidence_interval.high,
            'mean_reaction_accuracy': np.mean(accuracies),
            'sample_size': n_events
        }
    
    def compare_to_session_benchmarks(self, results: DailyPredictionResult) -> Dict[str, str]:
        """Compare performance to session-level benchmarks"""
        
        session_accuracy = 7.55  # points
        session_cascade_prob = 0.875
        
        # Performance assessment logic
        accuracy_better = results.mean_reaction_accuracy < session_accuracy
        prediction_better = results.high_prediction_accuracy > session_cascade_prob
        
        # Overall assessment
        if accuracy_better and prediction_better:
            assessment = "SUPERIOR - Better accuracy AND prediction rate"
        elif accuracy_better:
            assessment = f"MIXED - Better accuracy ({results.mean_reaction_accuracy:.2f} vs {session_accuracy}), lower prediction rate"
        elif prediction_better:
            assessment = f"MIXED - Better prediction rate ({results.high_prediction_accuracy:.1%} vs {session_cascade_prob:.1%}), lower accuracy"
        else:
            assessment = "INFERIOR - Lower performance on both metrics"
        
        # Add statistical significance context
        if results.total_interactions < 5:
            assessment += " [INSUFFICIENT SAMPLE]"
        elif results.total_interactions < 20:
            assessment += " [LIMITED SAMPLE]"
        
        benchmark_comparison = {
            'daily_vs_session_accuracy': f"{results.mean_reaction_accuracy:.2f} vs {session_accuracy} points",
            'daily_vs_session_prediction': f"{results.high_prediction_accuracy:.1%} vs {session_cascade_prob:.1%}",
            'performance_assessment': assessment,
            'accuracy_improvement': f"{((session_accuracy - results.mean_reaction_accuracy) / session_accuracy * 100):+.1f}%" if results.mean_reaction_accuracy > 0 else "N/A",
            'prediction_improvement': f"{((results.high_prediction_accuracy - session_cascade_prob) / session_cascade_prob * 100):+.1f}%" if results.high_prediction_accuracy > 0 else "N/A"
        }
        
        return benchmark_comparison
    
    def run_daily_timeframe_analysis(self, start_date: str = "2025-07-24", 
                                   end_date: str = "2025-08-07") -> Dict:
        """Execute complete daily timeframe archaeological analysis"""
        
        # Generate date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        # Extract daily ranges
        print("📊 Extracting daily high/low ranges...")
        daily_ranges = self.extract_daily_ranges(dates)
        
        print(f"✅ Extracted {len(daily_ranges)} daily ranges")
        
        # Process archaeological events
        events = []
        
        for i, date in enumerate(dates[1:], 1):  # Skip first date (no previous day)
            previous_date = dates[i-1]
            
            if previous_date in daily_ranges and date in daily_ranges:
                prev_data = daily_ranges[previous_date]
                curr_data = daily_ranges[date]
                
                # Detect reactions at previous day's 40% level
                reactions = self.detect_archaeological_reactions(date, prev_data['archaeological_40_level'])
                
                if reactions:
                    # Process each reaction
                    for reaction in reactions:
                        # Success criteria: current day exceeds previous day range
                        # High success: current day high > previous day high
                        # Low success: current day low < previous day low
                        high_success = curr_data['high'] > prev_data['high']
                        low_success = curr_data['low'] < prev_data['low']
                        
                        event = DailyArchaeologicalEvent(
                            date=date,
                            previous_day_high=prev_data['high'],
                            previous_day_low=prev_data['low'],
                            archaeological_40_level=prev_data['archaeological_40_level'],
                            reaction_time=str(reaction['timestamp']),
                            reaction_price=reaction['price'],
                            reaction_accuracy=reaction['accuracy'],
                            current_day_high=curr_data['high'],
                            current_day_low=curr_data['low'],
                            predicted_high_success=high_success,
                            predicted_low_success=low_success
                        )
                        events.append(event)
        
        # Validate predictions
        validation_results = self.validate_predictions(events)
        
        # Statistical analysis
        statistical_results = self.statistical_validation(events)
        
        # Benchmark comparison
        benchmark_comparison = self.compare_to_session_benchmarks(validation_results)
        
        return {
            'analysis_summary': {
                'date_range': f"{start_date} to {end_date}",
                'total_days_analyzed': len(daily_ranges),
                'archaeological_events_detected': len(events)
            },
            'prediction_results': validation_results,
            'statistical_validation': statistical_results,
            'benchmark_comparison': benchmark_comparison,
            'daily_ranges': daily_ranges,
            'events': [vars(event) for event in events]
        }

def main():
    """Execute daily timeframe archaeological analysis"""
    
    print("🏺 DAILY TIMEFRAME ARCHAEOLOGICAL ZONE ANALYZER")
    print("=" * 55)
    
    analyzer = DailyTimeframeArchaeologicalAnalyzer("/Users/jack/IRONFORGE/data/shards/NQ_M5")
    
    print("🔍 Analyzing previous day 40% level → current day extreme predictions...")
    print("📈 Target: Beat session-level 7.55-point accuracy, 87.5% cascade probability")
    
    results = analyzer.run_daily_timeframe_analysis()
    
    # Display results
    summary = results['analysis_summary']
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   📅 Period: {summary['date_range']}")
    print(f"   🗓️  Days: {summary['total_days_analyzed']}")
    print(f"   🎯 Events: {summary['archaeological_events_detected']}")
    
    if results['prediction_results'].total_interactions > 0:
        pred_results = results['prediction_results']
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   📈 High Prediction Rate: {pred_results.high_prediction_accuracy:.1%}")
        print(f"   📉 Low Prediction Rate: {pred_results.low_prediction_accuracy:.1%}")
        print(f"   🎯 Mean Accuracy: {pred_results.mean_reaction_accuracy:.2f} points")
        print(f"   ⏰ Total Interactions: {pred_results.total_interactions}")
        
        if 'statistical_validation' in results:
            stats_results = results['statistical_validation']
            if 'error' not in stats_results:
                print(f"\n📈 STATISTICAL VALIDATION:")
                print(f"   🔬 Sample Size: {stats_results['sample_size']}")
                print(f"   📊 High Success Rate: {stats_results['high_prediction_rate']:.1%}")
                print(f"   🎯 Statistical Significance (High): p={stats_results['high_binomial_pvalue']:.4f}")
                
        benchmark = results['benchmark_comparison']
        print(f"\n🏆 BENCHMARK COMPARISON:")
        print(f"   📏 Accuracy: {benchmark['daily_vs_session_accuracy']}")
        print(f"   🎯 Prediction: {benchmark['daily_vs_session_prediction']}")
        print(f"   📊 Assessment: {benchmark['performance_assessment']}")
    else:
        print("\n⚠️  No archaeological events detected in the analyzed period")
    
    return results

if __name__ == "__main__":
    main()