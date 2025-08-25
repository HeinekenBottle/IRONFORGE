#!/usr/bin/env python3
"""
IRONFORGE Statistical Prediction Generator Agent
===============================================

Probability synthesizer - converts tracking data into predictive intelligence.
Generates actionable probability forecasts for post-40% liquidity target completion.

Multi-Agent Role: Statistical Prediction Generator
Focus: Success rates, timing patterns, session-specific behaviors for live trading
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TargetPrediction:
    """Predictive intelligence for specific target type"""
    target_type: str
    overall_completion_probability: float
    current_session_probability: float
    next_session_probability: float
    average_completion_time_minutes: Optional[float]
    confidence_interval: Tuple[float, float]
    sample_size: int
    statistical_significance: str

@dataclass
class SessionSpecificPrediction:
    """Session-specific targeting behavior patterns"""
    session_type: str
    target_completion_rate: float
    preferred_targets: List[Dict]  # [{'target_type': 'session_high', 'probability': 0.85}]
    timing_patterns: Dict
    sequence_preferences: List[str]  # Order of target completion
    risk_factors: List[str]

@dataclass
class LiquidityProgressionForecast:
    """Complete predictive intelligence framework"""
    target_predictions: List[TargetPrediction]
    session_predictions: List[SessionSpecificPrediction]
    compound_scenarios: Dict  # Multiple target completion scenarios
    timing_intelligence: Dict
    risk_assessment: Dict
    forecast_confidence: float
    live_trading_recommendations: Dict
    forecast_timestamp: str

class StatisticalPredictionGeneratorAgent:
    """Agent specialized in generating predictive intelligence from tracking data"""
    
    def __init__(self, tracking_results_file: str):
        self.tracking_results_file = Path(tracking_results_file)
        
        # Statistical parameters
        self.confidence_level = 0.95
        self.minimum_sample_size = 3
        self.significance_threshold = 0.05
        
        # Prediction categories
        self.target_types = ['session_high', 'session_low', 'daily_high', 'daily_low', 'fvg_redelivery']
        self.session_types = ['LONDON', 'NY', 'ASIA', 'LUNCH', 'PREMARKET', 'MIDNIGHT']
        self.timing_classifications = ['current_session', 'next_session', 'delayed', 'never']
    
    def load_tracking_results(self) -> List[Dict]:
        """Load target progression tracking results"""
        
        print("📊 Loading target progression tracking results...")
        
        # TODO(human): Replace sample data with real IRONFORGE tracking results
        # This should load actual target progression tracking data from:
        # - Historical archaeological zone touches (40% liquidity events)
        # - Their subsequent target completions (session highs/lows, daily extremes, FVG redeliveries)
        # - Timing classifications (current_session, next_session, delayed, never)
        # - Session types and completion statistics
        # Consider loading from multiple tracking files for robust statistical analysis
        
        sample_tracking_results = [
            {
                'archaeological_touch': {
                    'timestamp': '2025-07-25 14:35:00',
                    'session_type': 'NY'
                },
                'target_completions': [
                    {
                        'target_type': 'session_high',
                        'target_id': 'LONDON_HIGH',
                        'completed': True,
                        'timing_classification': 'current_session',
                        'minutes_after_touch': 25
                    },
                    {
                        'target_type': 'daily_high',
                        'target_id': 'DAILY_HIGH',
                        'completed': True,
                        'timing_classification': 'next_session',
                        'minutes_after_touch': 180
                    },
                    {
                        'target_type': 'fvg_redelivery',
                        'target_id': 'LONDON_FVG_1',
                        'completed': False,
                        'timing_classification': 'never',
                        'minutes_after_touch': None
                    }
                ],
                'overall_statistics': {
                    'total_targets': 3,
                    'completed_targets': 2,
                    'overall_completion_rate': 0.67
                }
            }
        ]
        
        print(f"✅ Loaded {len(sample_tracking_results)} tracking results for statistical analysis")
        return sample_tracking_results
    
    def calculate_target_predictions(self, tracking_results: List[Dict]) -> List[TargetPrediction]:
        """Calculate predictive probabilities for each target type"""
        
        target_predictions = []
        
        for target_type in self.target_types:
            # Collect all completions for this target type
            target_completions = []
            
            for result in tracking_results:
                type_completions = [c for c in result['target_completions'] if c['target_type'] == target_type]
                target_completions.extend(type_completions)
            
            if len(target_completions) < self.minimum_sample_size:
                # Insufficient data for reliable prediction
                prediction = TargetPrediction(
                    target_type=target_type,
                    overall_completion_probability=0.0,
                    current_session_probability=0.0,
                    next_session_probability=0.0,
                    average_completion_time_minutes=None,
                    confidence_interval=(0.0, 0.0),
                    sample_size=len(target_completions),
                    statistical_significance='insufficient_data'
                )
                target_predictions.append(prediction)
                continue
            
            # Calculate completion statistics
            completed = [c for c in target_completions if c['completed']]
            overall_rate = len(completed) / len(target_completions)
            
            # Timing breakdown
            current_session_completed = [c for c in completed if c['timing_classification'] == 'current_session']
            next_session_completed = [c for c in completed if c['timing_classification'] == 'next_session']
            
            current_session_rate = len(current_session_completed) / len(target_completions)
            next_session_rate = len(next_session_completed) / len(target_completions)
            
            # Average completion time
            completion_times = [c['minutes_after_touch'] for c in completed if c['minutes_after_touch'] is not None]
            avg_time = np.mean(completion_times) if completion_times else None
            
            # Calculate confidence interval using binomial proportion
            if len(target_completions) >= 10:
                # Use normal approximation for large samples
                se = np.sqrt(overall_rate * (1 - overall_rate) / len(target_completions))
                z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
                ci_lower = max(0, overall_rate - z_score * se)
                ci_upper = min(1, overall_rate + z_score * se)
                significance = 'significant' if len(target_completions) >= 20 else 'moderate'
            else:
                # Use exact binomial confidence interval for small samples
                ci_lower, ci_upper = stats.binom.interval(self.confidence_level, len(target_completions), overall_rate)
                ci_lower /= len(target_completions)
                ci_upper /= len(target_completions)
                significance = 'limited'
            
            prediction = TargetPrediction(
                target_type=target_type,
                overall_completion_probability=overall_rate,
                current_session_probability=current_session_rate,
                next_session_probability=next_session_rate,
                average_completion_time_minutes=avg_time,
                confidence_interval=(ci_lower, ci_upper),
                sample_size=len(target_completions),
                statistical_significance=significance
            )
            
            target_predictions.append(prediction)
        
        return target_predictions
    
    def calculate_session_predictions(self, tracking_results: List[Dict]) -> List[SessionSpecificPrediction]:
        """Calculate session-specific targeting behavior predictions"""
        
        session_predictions = []
        
        for session_type in self.session_types:
            # Filter results for this session type
            session_results = [r for r in tracking_results if r['archaeological_touch']['session_type'] == session_type]
            
            if not session_results:
                continue
            
            # Calculate overall completion rate for this session
            total_targets = sum(r['overall_statistics']['total_targets'] for r in session_results)
            total_completed = sum(r['overall_statistics']['completed_targets'] for r in session_results)
            session_completion_rate = total_completed / total_targets if total_targets > 0 else 0
            
            # Identify preferred targets
            target_preferences = {}
            for target_type in self.target_types:
                type_completions = []
                for result in session_results:
                    type_comps = [c for c in result['target_completions'] if c['target_type'] == target_type]
                    type_completions.extend(type_comps)
                
                if type_completions:
                    completion_rate = len([c for c in type_completions if c['completed']]) / len(type_completions)
                    target_preferences[target_type] = completion_rate
            
            # Sort by preference
            preferred_targets = [
                {'target_type': target, 'probability': prob} 
                for target, prob in sorted(target_preferences.items(), key=lambda x: x[1], reverse=True)
            ]
            
            # Analyze timing patterns
            all_completions = []
            for result in session_results:
                all_completions.extend([c for c in result['target_completions'] if c['completed']])
            
            timing_patterns = {}
            if all_completions:
                completion_times = [c['minutes_after_touch'] for c in all_completions if c['minutes_after_touch'] is not None]
                if completion_times:
                    timing_patterns = {
                        'average_time': np.mean(completion_times),
                        'median_time': np.median(completion_times),
                        'fastest': min(completion_times),
                        'slowest': max(completion_times),
                        'std_deviation': np.std(completion_times)
                    }
            
            # Determine sequence preferences (simplified)
            sequence_preferences = [item['target_type'] for item in preferred_targets[:3]]
            
            # Identify risk factors
            risk_factors = []
            if session_completion_rate < 0.5:
                risk_factors.append('low_completion_rate')
            if len(session_results) < 5:
                risk_factors.append('limited_sample_size')
            if timing_patterns.get('std_deviation', 0) > 60:  # High time variability
                risk_factors.append('inconsistent_timing')
            
            prediction = SessionSpecificPrediction(
                session_type=session_type,
                target_completion_rate=session_completion_rate,
                preferred_targets=preferred_targets,
                timing_patterns=timing_patterns,
                sequence_preferences=sequence_preferences,
                risk_factors=risk_factors
            )
            
            session_predictions.append(prediction)
        
        return session_predictions
    
    def calculate_compound_scenarios(self, target_predictions: List[TargetPrediction]) -> Dict:
        """Calculate probabilities for multiple target completion scenarios"""
        
        # Create scenarios based on target combinations
        scenarios = {
            'session_sweep': {
                'description': 'Both session high and low completion',
                'targets': ['session_high', 'session_low'],
                'probability': 1.0
            },
            'daily_progression': {
                'description': 'Session high followed by daily high',
                'targets': ['session_high', 'daily_high'],
                'probability': 1.0
            },
            'fvg_with_liquidity': {
                'description': 'FVG redelivery with session high/low',
                'targets': ['fvg_redelivery', 'session_high'],
                'probability': 1.0
            },
            'complete_targeting': {
                'description': 'All target types completed',
                'targets': self.target_types,
                'probability': 1.0
            }
        }
        
        # Calculate compound probabilities (simplified independence assumption)
        prediction_map = {pred.target_type: pred.overall_completion_probability for pred in target_predictions}
        
        for scenario_name, scenario in scenarios.items():
            compound_prob = 1.0
            for target in scenario['targets']:
                if target in prediction_map:
                    compound_prob *= prediction_map[target]
            
            scenario['probability'] = compound_prob
        
        return scenarios
    
    def generate_timing_intelligence(self, tracking_results: List[Dict]) -> Dict:
        """Generate timing intelligence for optimal entry/exit strategies"""
        
        # Collect all completion times
        all_completions = []
        for result in tracking_results:
            completed_targets = [c for c in result['target_completions'] if c['completed']]
            all_completions.extend(completed_targets)
        
        if not all_completions:
            return {'error': 'no_completions_data'}
        
        # Calculate timing statistics
        completion_times = [c['minutes_after_touch'] for c in all_completions if c['minutes_after_touch'] is not None]
        
        timing_intelligence = {
            'optimal_monitoring_window': {
                'minutes': int(np.percentile(completion_times, 90)) if completion_times else 60,
                'description': '90th percentile completion time'
            },
            'early_completion_threshold': {
                'minutes': int(np.percentile(completion_times, 25)) if completion_times else 15,
                'description': '25th percentile - early completions'
            },
            'peak_completion_window': {
                'start_minutes': int(np.percentile(completion_times, 25)) if completion_times else 10,
                'end_minutes': int(np.percentile(completion_times, 75)) if completion_times else 45,
                'description': 'Interquartile range - highest probability window'
            },
            'timing_distribution': {
                'mean': np.mean(completion_times) if completion_times else 0,
                'median': np.median(completion_times) if completion_times else 0,
                'std': np.std(completion_times) if completion_times else 0
            },
            'session_timing_preferences': self._calculate_session_timing_preferences(all_completions)
        }
        
        return timing_intelligence
    
    def _calculate_session_timing_preferences(self, completions: List[Dict]) -> Dict:
        """Calculate timing preferences by session type"""
        
        session_timing = {}
        
        for session_type in self.session_types:
            session_completions = [c for c in completions if c.get('completion_session') == session_type]
            
            if session_completions:
                times = [c['minutes_after_touch'] for c in session_completions if c['minutes_after_touch'] is not None]
                
                if times:
                    session_timing[session_type] = {
                        'average_time': np.mean(times),
                        'fastest': min(times),
                        'completion_count': len(session_completions)
                    }
        
        return session_timing
    
    def assess_prediction_risks(self, target_predictions: List[TargetPrediction], 
                              session_predictions: List[SessionSpecificPrediction]) -> Dict:
        """Assess risks and limitations of predictions"""
        
        risks = {
            'data_quality_risks': [],
            'statistical_risks': [],
            'market_condition_risks': [],
            'overall_risk_level': 'low'
        }
        
        # Data quality assessment
        total_sample_size = sum(pred.sample_size for pred in target_predictions)
        if total_sample_size < 50:
            risks['data_quality_risks'].append('limited_sample_size')
        
        insufficient_data_targets = len([pred for pred in target_predictions if pred.statistical_significance == 'insufficient_data'])
        if insufficient_data_targets > 2:
            risks['data_quality_risks'].append('multiple_insufficient_targets')
        
        # Statistical significance risks
        low_significance_predictions = len([pred for pred in target_predictions if pred.statistical_significance in ['limited', 'insufficient_data']])
        if low_significance_predictions > len(target_predictions) * 0.5:
            risks['statistical_risks'].append('low_statistical_confidence')
        
        # Wide confidence intervals
        wide_ci_predictions = len([pred for pred in target_predictions if pred.confidence_interval[1] - pred.confidence_interval[0] > 0.4])
        if wide_ci_predictions > 0:
            risks['statistical_risks'].append('wide_confidence_intervals')
        
        # Session-specific risks
        session_risk_count = sum(len(pred.risk_factors) for pred in session_predictions)
        if session_risk_count > len(session_predictions):
            risks['market_condition_risks'].append('session_variability')
        
        # Overall risk assessment
        total_risks = len(risks['data_quality_risks']) + len(risks['statistical_risks']) + len(risks['market_condition_risks'])
        if total_risks >= 4:
            risks['overall_risk_level'] = 'high'
        elif total_risks >= 2:
            risks['overall_risk_level'] = 'moderate'
        
        return risks
    
    def generate_live_trading_recommendations(self, target_predictions: List[TargetPrediction],
                                            timing_intelligence: Dict, risk_assessment: Dict) -> Dict:
        """Generate actionable recommendations for live trading"""
        
        recommendations = {
            'primary_targets': [],
            'timing_strategy': {},
            'risk_management': {},
            'session_specific_guidance': {}
        }
        
        # Identify primary targets (>60% probability)
        high_probability_targets = [pred for pred in target_predictions if pred.overall_completion_probability > 0.6]
        recommendations['primary_targets'] = [
            {
                'target_type': pred.target_type,
                'probability': pred.overall_completion_probability,
                'preferred_timing': 'current_session' if pred.current_session_probability > pred.next_session_probability else 'next_session',
                'expected_time_minutes': pred.average_completion_time_minutes
            }
            for pred in high_probability_targets
        ]
        
        # Timing strategy
        if 'peak_completion_window' in timing_intelligence:
            window = timing_intelligence['peak_completion_window']
            recommendations['timing_strategy'] = {
                'monitor_period_minutes': timing_intelligence.get('optimal_monitoring_window', {}).get('minutes', 60),
                'highest_probability_window': f"{window['start_minutes']}-{window['end_minutes']} minutes post-touch",
                'early_signal_threshold': timing_intelligence.get('early_completion_threshold', {}).get('minutes', 15)
            }
        
        # Risk management
        recommendations['risk_management'] = {
            'overall_risk_level': risk_assessment['overall_risk_level'],
            'position_sizing_factor': 1.0 if risk_assessment['overall_risk_level'] == 'low' else 0.7 if risk_assessment['overall_risk_level'] == 'moderate' else 0.5,
            'key_risks': risk_assessment['data_quality_risks'] + risk_assessment['statistical_risks'],
            'monitoring_requirements': 'enhanced' if risk_assessment['overall_risk_level'] == 'high' else 'standard'
        }
        
        return recommendations
    
    def generate_liquidity_progression_forecast(self, tracking_results: List[Dict]) -> LiquidityProgressionForecast:
        """Generate complete liquidity progression forecast"""
        
        print("📈 Generating liquidity progression forecast...")
        
        # Calculate all prediction components
        target_predictions = self.calculate_target_predictions(tracking_results)
        session_predictions = self.calculate_session_predictions(tracking_results)
        compound_scenarios = self.calculate_compound_scenarios(target_predictions)
        timing_intelligence = self.generate_timing_intelligence(tracking_results)
        risk_assessment = self.assess_prediction_risks(target_predictions, session_predictions)
        live_trading_recommendations = self.generate_live_trading_recommendations(target_predictions, timing_intelligence, risk_assessment)
        
        # Calculate overall forecast confidence
        significant_predictions = len([pred for pred in target_predictions if pred.statistical_significance in ['significant', 'moderate']])
        total_predictions = len(target_predictions)
        forecast_confidence = significant_predictions / total_predictions if total_predictions > 0 else 0
        
        forecast = LiquidityProgressionForecast(
            target_predictions=target_predictions,
            session_predictions=session_predictions,
            compound_scenarios=compound_scenarios,
            timing_intelligence=timing_intelligence,
            risk_assessment=risk_assessment,
            forecast_confidence=forecast_confidence,
            live_trading_recommendations=live_trading_recommendations,
            forecast_timestamp=datetime.now().isoformat()
        )
        
        print("✅ Liquidity progression forecast generated")
        return forecast
    
    def export_forecast(self, forecast: LiquidityProgressionForecast) -> str:
        """Export complete forecast to JSON"""
        
        output_path = Path("/Users/jack/IRONFORGE/liquidity_progression_forecasts")
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"liquidity_progression_forecast_{timestamp}.json"
        
        export_data = {
            'agent_role': 'Statistical Prediction Generator Agent',
            'forecast_timestamp': forecast.forecast_timestamp,
            'forecast_confidence': forecast.forecast_confidence,
            'liquidity_progression_forecast': asdict(forecast),
            'summary_statistics': self._create_forecast_summary(forecast)
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📁 Liquidity progression forecast exported to: {output_file}")
        return str(output_file)
    
    def _create_forecast_summary(self, forecast: LiquidityProgressionForecast) -> Dict:
        """Create executive summary of forecast"""
        
        # Top target predictions
        top_targets = sorted(forecast.target_predictions, key=lambda x: x.overall_completion_probability, reverse=True)[:3]
        
        # Best session predictions
        top_sessions = sorted(forecast.session_predictions, key=lambda x: x.target_completion_rate, reverse=True)[:2]
        
        return {
            'highest_probability_targets': [
                {'target': pred.target_type, 'probability': f"{pred.overall_completion_probability:.1%}"} 
                for pred in top_targets
            ],
            'best_performing_sessions': [
                {'session': pred.session_type, 'completion_rate': f"{pred.target_completion_rate:.1%}"} 
                for pred in top_sessions
            ],
            'optimal_monitoring_window': forecast.timing_intelligence.get('optimal_monitoring_window', {}),
            'overall_risk_level': forecast.risk_assessment['overall_risk_level'],
            'forecast_confidence': f"{forecast.forecast_confidence:.1%}"
        }
    
    def run_statistical_prediction_generation(self) -> Dict:
        """Execute complete statistical prediction generation"""
        
        print("📈 STATISTICAL PREDICTION GENERATOR AGENT STARTING...")
        print("=" * 60)
        
        # Load tracking results
        tracking_results = self.load_tracking_results()
        
        # Generate forecast
        forecast = self.generate_liquidity_progression_forecast(tracking_results)
        
        # Export forecast
        export_file = self.export_forecast(forecast)
        
        return {
            'agent_role': 'Statistical Prediction Generator Agent',
            'forecast_generated': True,
            'forecast_confidence': forecast.forecast_confidence,
            'export_file': export_file,
            'predictive_intelligence_ready': True
        }

def main():
    """Execute statistical prediction generation"""
    
    print("📈 STATISTICAL PREDICTION GENERATOR AGENT")
    print("=" * 50)
    
    agent = StatisticalPredictionGeneratorAgent(
        tracking_results_file="/Users/jack/IRONFORGE/target_progression_tracking/latest.json"
    )
    
    results = agent.run_statistical_prediction_generation()
    
    # Display results
    print(f"\n📈 STATISTICAL PREDICTION GENERATION COMPLETE:")
    print(f"   🎯 Forecast Generated: {results['forecast_generated']}")
    print(f"   📊 Forecast Confidence: {results['forecast_confidence']:.1%}")
    print(f"   📁 Export File: {results['export_file']}")
    print(f"   ✅ Predictive Intelligence Ready: {results['predictive_intelligence_ready']}")
    
    return results

if __name__ == "__main__":
    main()