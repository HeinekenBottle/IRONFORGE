#!/usr/bin/env python3
"""
ExperimentE Advanced Path Analysis Module
Implements E1/E2/E3 post-RD@40% sequence analysis with statistical rigor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ExperimentEAnalyzer:
    """Advanced post-RD@40% path analysis with E1/E2/E3 specifications"""
    
    def __init__(self):
        self.isotonic_regressors = {
            'CONT': IsotonicRegression(out_of_bounds='clip'),
            'MR': IsotonicRegression(out_of_bounds='clip'),
            'ACCEL': IsotonicRegression(out_of_bounds='clip')
        }
        self.feature_importance = {}
        self.path_thresholds = {
            'E1_CONT': {
                'f8_q_min': 0.90,  # ≥P90
                'theory_b_max_dt': 30,  # ≤30m
                'gap_age_max': 2,  # ≤2 days aligned
                'time_to_60_max': 45,  # ≤45m
                'time_to_80_max': 90   # ≤90m
            },
            'E2_MR': {
                'news_window': 15,  # ±15m
                'time_to_mid_max': 60,  # ≤60m
                'second_rd_window': 120,  # ≤120m
                'failure_window': 120   # ≤120m
            },
            'E3_ACCEL': {
                'f8_q_min': 0.95,  # ≥P95
                'h1_window': 15,  # ±15m
                'time_to_80_max': 60,  # ≤60m
                'max_pullback_atr': 0.25  # ≤0.25·ATR(M5)
            }
        }
    
    def derive_advanced_features(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """Derive f8_q, f8_slope_sign, and other advanced features from base data"""
        enhanced_data = session_data.copy()
        
        # Derive f8_q percentile ranking if f8 column exists
        if 'f8' in enhanced_data.columns:
            enhanced_data['f8_q'] = enhanced_data['f8'].rank(pct=True)
            
            # Calculate f8_slope_sign (3-bar rolling slope)
            f8_slope = enhanced_data['f8'].rolling(window=3, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
            enhanced_data['f8_slope_sign'] = np.sign(f8_slope)
        else:
            # Create placeholder features from available data
            enhanced_data['f8_q'] = 0.5  # Neutral percentile
            enhanced_data['f8_slope_sign'] = 0
        
        # Derive gap_age_d from available temporal features
        if 'time_since_session_open' in enhanced_data.columns:
            # Proxy gap age using time since session open and price momentum
            enhanced_data['gap_age_d'] = np.where(
                enhanced_data.get('price_momentum', 0).abs() > 0.02,
                0,  # Fresh gap (high momentum)
                1   # Aged gap
            )
        else:
            enhanced_data['gap_age_d'] = 1  # Conservative default
        
        # Calculate ATR(M5) for pullback analysis
        if 'price_level' in enhanced_data.columns:
            price_series = enhanced_data['price_level']
            high_low_range = price_series.rolling(window=5).max() - price_series.rolling(window=5).min()
            enhanced_data['atr_m5'] = high_low_range.rolling(window=14).mean()
        else:
            enhanced_data['atr_m5'] = enhanced_data.get('magnitude', 1.0) * 10  # Proxy ATR
        
        # Derive Theory B delta time from archaeological significance
        enhanced_data['theory_b_dt'] = np.where(
            enhanced_data.get('archaeological_significance', 0) > 0.8,
            15,  # High significance = recent Theory B event
            45   # Lower significance = older Theory B event
        )
        
        return enhanced_data
    
    def classify_e1_cont_path(self, session_data: pd.DataFrame, rd40_event_idx: int) -> Dict[str, Any]:
        """Classify E1 CONT path: RD@40 → 60% within ≤45m, 80% within ≤90m"""
        thresholds = self.path_thresholds['E1_CONT']
        
        # Get enhanced features
        enhanced_data = self.derive_advanced_features(session_data)
        
        if rd40_event_idx >= len(enhanced_data):
            return {"path": "UNKNOWN", "reason": "Invalid event index", "confidence": 0.0}
        
        rd40_event = enhanced_data.iloc[rd40_event_idx]
        
        # Check preconditions
        f8_q = rd40_event.get('f8_q', 0.5)
        f8_slope_sign = rd40_event.get('f8_slope_sign', 0)
        theory_b_dt = rd40_event.get('theory_b_dt', 50)
        gap_age_d = rd40_event.get('gap_age_d', 3)
        f50_regime = rd40_event.get('f50_htf_regime', 1)
        
        # E1 CONT precondition scoring
        precondition_score = 0.0
        precondition_details = {}
        
        # f8_q ≥ P90 with positive slope
        if f8_q >= thresholds['f8_q_min'] and f8_slope_sign > 0:
            precondition_score += 0.3
            precondition_details['f8_condition'] = True
        else:
            precondition_details['f8_condition'] = False
        
        # TheoryB_Δt ≤ 30m
        if theory_b_dt <= thresholds['theory_b_max_dt']:
            precondition_score += 0.25
            precondition_details['theory_b_timing'] = True
        else:
            precondition_details['theory_b_timing'] = False
        
        # gap_age_d ≤ 2 aligned
        if gap_age_d <= thresholds['gap_age_max']:
            precondition_score += 0.2
            precondition_details['gap_freshness'] = True
        else:
            precondition_details['gap_freshness'] = False
        
        # f50 ∈ {trend, transition} (regime codes 2=trend, 1=transition, 0=consolidation)
        if f50_regime in [1, 2]:
            precondition_score += 0.25
            precondition_details['regime_favorable'] = True
        else:
            precondition_details['regime_favorable'] = False
        
        # Analyze post-RD progression
        session_stats = self._calculate_session_stats(enhanced_data)
        progression_analysis = self._analyze_progression_timing(
            enhanced_data, rd40_event_idx, session_stats
        )
        
        # E1 CONT classification logic
        time_to_60 = progression_analysis.get('time_to_60', float('inf'))
        time_to_80 = progression_analysis.get('time_to_80', float('inf'))
        drawdown_to_mid = progression_analysis.get('max_drawdown_to_mid', 0)
        
        # Check E1 CONT criteria
        cont_criteria_met = (
            time_to_60 <= thresholds['time_to_60_max'] and
            time_to_80 <= thresholds['time_to_80_max'] and
            drawdown_to_mid < 0.15  # Less than 15% drawdown to mid
        )
        
        if cont_criteria_met and precondition_score >= 0.6:
            confidence = min(0.95, precondition_score + 0.2)
            return {
                "path": "E1_CONT",
                "confidence": confidence,
                "precondition_score": precondition_score,
                "precondition_details": precondition_details,
                "timing_analysis": {
                    "time_to_60": time_to_60,
                    "time_to_80": time_to_80,
                    "drawdown_to_mid": drawdown_to_mid
                },
                "kpis": {
                    "probability": confidence,
                    "expected_time_to_60": time_to_60,
                    "expected_time_to_80": time_to_80,
                    "drawdown_risk": drawdown_to_mid
                }
            }
        
        return {
            "path": "NOT_E1_CONT",
            "confidence": 1 - precondition_score,
            "reason": "Failed E1 CONT criteria",
            "precondition_score": precondition_score,
            "precondition_details": precondition_details,
            "timing_analysis": progression_analysis
        }
    
    def classify_e2_mr_path(self, session_data: pd.DataFrame, rd40_event_idx: int) -> Dict[str, Any]:
        """Classify E2 MR path: RD@40 → mid (50-60%) within ≤60m with branching analysis"""
        thresholds = self.path_thresholds['E2_MR']
        
        enhanced_data = self.derive_advanced_features(session_data)
        
        if rd40_event_idx >= len(enhanced_data):
            return {"path": "UNKNOWN", "reason": "Invalid event index", "confidence": 0.0}
        
        rd40_event = enhanced_data.iloc[rd40_event_idx]
        
        # Check MR preconditions
        f50_regime = rd40_event.get('f50_htf_regime', 1)
        f8_slope_sign = rd40_event.get('f8_slope_sign', 0)
        news_impact = self._assess_news_proximity(rd40_event, thresholds['news_window'])
        
        precondition_score = 0.0
        precondition_details = {}
        
        # f50 = mean-revert (regime code 0) OR high news impact
        if f50_regime == 0 or news_impact['has_high_impact']:
            precondition_score += 0.4
            precondition_details['mr_conditions'] = True
        else:
            precondition_details['mr_conditions'] = False
        
        # f8_slope_sign ≤ 0
        if f8_slope_sign <= 0:
            precondition_score += 0.3
            precondition_details['negative_slope'] = True
        else:
            precondition_details['negative_slope'] = False
        
        # No H1 breakout
        h1_breakout = self._detect_h1_breakout(enhanced_data, rd40_event_idx, 15)
        if not h1_breakout['detected']:
            precondition_score += 0.3
            precondition_details['no_h1_breakout'] = True
        else:
            precondition_details['no_h1_breakout'] = False
        
        # Analyze progression to mid-range
        session_stats = self._calculate_session_stats(enhanced_data)
        mid_analysis = self._analyze_mid_reversion(
            enhanced_data, rd40_event_idx, session_stats, thresholds
        )
        
        if mid_analysis['reaches_mid'] and precondition_score >= 0.6:
            # Further analyze for second_rd and failure branches
            branch_analysis = self._analyze_mr_branches(
                enhanced_data, rd40_event_idx, session_stats, thresholds
            )
            
            confidence = min(0.90, precondition_score + 0.2)
            return {
                "path": "E2_MR",
                "confidence": confidence,
                "precondition_score": precondition_score,
                "precondition_details": precondition_details,
                "mid_analysis": mid_analysis,
                "branch_analysis": branch_analysis,
                "kpis": {
                    "probability_mr": confidence,
                    "probability_second_rd": branch_analysis.get('second_rd_probability', 0),
                    "probability_failure": branch_analysis.get('failure_probability', 0),
                    "time_to_mid": mid_analysis.get('time_to_mid', float('inf')),
                    "max_adverse_excursion": mid_analysis.get('max_adverse_excursion', 0)
                }
            }
        
        return {
            "path": "NOT_E2_MR",
            "confidence": 1 - precondition_score,
            "reason": "Failed E2 MR criteria",
            "precondition_score": precondition_score,
            "precondition_details": precondition_details,
            "mid_analysis": mid_analysis
        }
    
    def classify_e3_accel_path(self, session_data: pd.DataFrame, rd40_event_idx: int) -> Dict[str, Any]:
        """Classify E3 ACCEL path: RD@40 + H1 breakout → 80% within ≤60m"""
        thresholds = self.path_thresholds['E3_ACCEL']
        
        enhanced_data = self.derive_advanced_features(session_data)
        
        if rd40_event_idx >= len(enhanced_data):
            return {"path": "UNKNOWN", "reason": "Invalid event index", "confidence": 0.0}
        
        rd40_event = enhanced_data.iloc[rd40_event_idx]
        
        # Check ACCEL preconditions
        f8_q = rd40_event.get('f8_q', 0.5)
        theory_b_dt = rd40_event.get('theory_b_dt', 50)
        
        precondition_score = 0.0
        precondition_details = {}
        
        # H1 breakout (same direction) within ±15m
        h1_breakout = self._detect_h1_breakout(enhanced_data, rd40_event_idx, thresholds['h1_window'])
        if h1_breakout['detected'] and h1_breakout['direction_aligned']:
            precondition_score += 0.4
            precondition_details['h1_breakout_aligned'] = True
        else:
            precondition_details['h1_breakout_aligned'] = False
        
        # f8_q ≥ P95
        if f8_q >= thresholds['f8_q_min']:
            precondition_score += 0.3
            precondition_details['f8_elite'] = True
        else:
            precondition_details['f8_elite'] = False
        
        # TheoryB_Δt ≤ 30m
        if theory_b_dt <= 30:
            precondition_score += 0.2
            precondition_details['theory_b_timing'] = True
        else:
            precondition_details['theory_b_timing'] = False
        
        # Tight dist_to_zone_atr at RD
        zone_distance = rd40_event.get('range_position', 0.5) - 0.4  # Distance from 40%
        if abs(zone_distance) <= 0.02:  # Within 2% of 40% zone
            precondition_score += 0.1
            precondition_details['tight_zone_distance'] = True
        else:
            precondition_details['tight_zone_distance'] = False
        
        # Analyze acceleration to 80%
        session_stats = self._calculate_session_stats(enhanced_data)
        accel_analysis = self._analyze_acceleration(
            enhanced_data, rd40_event_idx, session_stats, thresholds
        )
        
        if accel_analysis['reaches_80_fast'] and precondition_score >= 0.7:
            confidence = min(0.95, precondition_score + 0.25)
            return {
                "path": "E3_ACCEL",
                "confidence": confidence,
                "precondition_score": precondition_score,
                "precondition_details": precondition_details,
                "h1_breakout": h1_breakout,
                "acceleration_analysis": accel_analysis,
                "kpis": {
                    "probability": confidence,
                    "time_to_80": accel_analysis.get('time_to_80', float('inf')),
                    "pullback_depth": accel_analysis.get('max_pullback_depth', 0),
                    "continuation_beyond_80": accel_analysis.get('continuation_probability', 0)
                }
            }
        
        return {
            "path": "NOT_E3_ACCEL",
            "confidence": 1 - precondition_score,
            "reason": "Failed E3 ACCEL criteria",
            "precondition_score": precondition_score,
            "precondition_details": precondition_details,
            "acceleration_analysis": accel_analysis
        }
    
    def _calculate_session_stats(self, session_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate session high, low, range, and zone levels"""
        price_col = 'price_level' if 'price_level' in session_data.columns else 'absolute_price'
        if price_col not in session_data.columns:
            return {}
        
        prices = session_data[price_col].dropna()
        if len(prices) == 0:
            return {}
        
        session_high = prices.max()
        session_low = prices.min()
        session_range = session_high - session_low
        
        return {
            'session_high': session_high,
            'session_low': session_low,
            'session_range': session_range,
            'zone_40': session_low + (session_range * 0.4),
            'zone_50': session_low + (session_range * 0.5),
            'zone_60': session_low + (session_range * 0.6),
            'zone_80': session_low + (session_range * 0.8)
        }
    
    def _analyze_progression_timing(self, data: pd.DataFrame, start_idx: int, 
                                  session_stats: Dict[str, float]) -> Dict[str, Any]:
        """Analyze timing to reach various zones after RD@40%"""
        if not session_stats or start_idx >= len(data):
            return {}
        
        price_col = 'price_level' if 'price_level' in data.columns else 'absolute_price'
        post_event_data = data.iloc[start_idx:].copy()
        
        zone_60 = session_stats['zone_60']
        zone_80 = session_stats['zone_80']
        zone_50 = session_stats['zone_50']
        
        time_to_60 = None
        time_to_80 = None
        max_drawdown_to_mid = 0
        
        for i, (_, row) in enumerate(post_event_data.iterrows()):
            price = row.get(price_col, 0)
            if price == 0:
                continue
            
            # Check zone reaches
            if time_to_60 is None and abs(price - zone_60) / session_stats['session_range'] <= 0.03:
                time_to_60 = i
            
            if time_to_80 is None and abs(price - zone_80) / session_stats['session_range'] <= 0.03:
                time_to_80 = i
            
            # Track drawdown to mid
            drawdown_pct = abs(price - zone_50) / session_stats['session_range']
            max_drawdown_to_mid = max(max_drawdown_to_mid, drawdown_pct)
        
        return {
            'time_to_60': time_to_60 if time_to_60 is not None else float('inf'),
            'time_to_80': time_to_80 if time_to_80 is not None else float('inf'),
            'max_drawdown_to_mid': max_drawdown_to_mid
        }
    
    def _assess_news_proximity(self, event: pd.Series, window_minutes: int) -> Dict[str, Any]:
        """Assess news impact within ±window_minutes of event"""
        # Placeholder implementation - would integrate with news calendar
        # For now, use energy_density as proxy for news impact
        energy_density = event.get('energy_density', 0.5)
        
        return {
            'has_high_impact': energy_density > 0.8,
            'impact_score': energy_density,
            'distance_to_news': 30  # Placeholder
        }
    
    def _detect_h1_breakout(self, data: pd.DataFrame, event_idx: int, 
                          window_minutes: int) -> Dict[str, Any]:
        """Detect H1 breakout within ±window_minutes"""
        # Placeholder H1 breakout detection
        # Would integrate with HTF analysis
        
        if event_idx >= len(data):
            return {'detected': False}
        
        event_row = data.iloc[event_idx]
        htf_confluence = event_row.get('htf_confluence', 0.5)
        magnitude = event_row.get('magnitude', 0)
        
        # Detect based on HTF confluence and magnitude
        detected = htf_confluence > 0.7 and magnitude > 0.5
        direction_aligned = magnitude > 0  # Simplified direction check
        
        return {
            'detected': detected,
            'direction_aligned': direction_aligned,
            'confluence_score': htf_confluence,
            'breakout_strength': magnitude
        }
    
    def _analyze_mid_reversion(self, data: pd.DataFrame, start_idx: int, 
                             session_stats: Dict[str, float], thresholds: Dict) -> Dict[str, Any]:
        """Analyze reversion to mid-range (50-60%)"""
        if not session_stats or start_idx >= len(data):
            return {'reaches_mid': False}
        
        price_col = 'price_level' if 'price_level' in data.columns else 'absolute_price'
        post_event_data = data.iloc[start_idx:start_idx + thresholds['time_to_mid_max']].copy()
        
        zone_50 = session_stats['zone_50']
        zone_60 = session_stats['zone_60']
        
        reaches_mid = False
        time_to_mid = None
        max_adverse_excursion = 0
        
        for i, (_, row) in enumerate(post_event_data.iterrows()):
            price = row.get(price_col, 0)
            if price == 0:
                continue
            
            # Check if in mid-range (50-60%)
            if zone_50 <= price <= zone_60:
                reaches_mid = True
                if time_to_mid is None:
                    time_to_mid = i
            
            # Track adverse excursion
            adverse_move = abs(price - zone_50) / session_stats['session_range']
            max_adverse_excursion = max(max_adverse_excursion, adverse_move)
        
        return {
            'reaches_mid': reaches_mid,
            'time_to_mid': time_to_mid if time_to_mid is not None else float('inf'),
            'max_adverse_excursion': max_adverse_excursion
        }
    
    def _analyze_mr_branches(self, data: pd.DataFrame, start_idx: int, 
                           session_stats: Dict[str, float], thresholds: Dict) -> Dict[str, Any]:
        """Analyze MR branching: second_rd vs failure"""
        # Placeholder for second RD and failure detection
        # Would require more sophisticated imbalance analysis
        
        return {
            'second_rd_probability': 0.3,
            'failure_probability': 0.2,
            'branch_detected_at': None
        }
    
    def _analyze_acceleration(self, data: pd.DataFrame, start_idx: int, 
                            session_stats: Dict[str, float], thresholds: Dict) -> Dict[str, Any]:
        """Analyze acceleration to 80% zone"""
        if not session_stats or start_idx >= len(data):
            return {'reaches_80_fast': False}
        
        price_col = 'price_level' if 'price_level' in data.columns else 'absolute_price'
        post_event_data = data.iloc[start_idx:start_idx + thresholds['time_to_80_max']].copy()
        
        zone_80 = session_stats['zone_80']
        atr_threshold = thresholds['max_pullback_atr']
        
        reaches_80_fast = False
        time_to_80 = None
        max_pullback_depth = 0
        continuation_probability = 0
        
        for i, (_, row) in enumerate(post_event_data.iterrows()):
            price = row.get(price_col, 0)
            atr_m5 = row.get('atr_m5', 1.0)
            
            if price == 0:
                continue
            
            # Check if reaches 80% zone
            if abs(price - zone_80) / session_stats['session_range'] <= 0.03:
                reaches_80_fast = True
                if time_to_80 is None:
                    time_to_80 = i
                    continuation_probability = 0.6  # Base continuation probability
            
            # Track pullback depth relative to ATR
            if time_to_80 is not None:
                pullback = abs(price - zone_80) / (atr_m5 * atr_threshold)
                max_pullback_depth = max(max_pullback_depth, pullback)
        
        return {
            'reaches_80_fast': reaches_80_fast and time_to_80 <= thresholds['time_to_80_max'],
            'time_to_80': time_to_80 if time_to_80 is not None else float('inf'),
            'max_pullback_depth': max_pullback_depth,
            'continuation_probability': continuation_probability
        }
    
    def analyze_pattern_switches(self, session_data: pd.DataFrame, 
                               rd40_events: List[Dict]) -> Dict[str, Any]:
        """Analyze pattern-switch diagnostics for regime transitions"""
        diagnostics = {
            'regime_flip_analysis': {},
            'news_proximity_effects': {},
            'h1_confirmation_impact': {},
            'gap_context_patterns': {},
            'micro_momentum_switches': {}
        }
        
        enhanced_data = self.derive_advanced_features(session_data)
        
        for event in rd40_events:
            event_idx = event.get('event_index', 0)
            if event_idx >= len(enhanced_data):
                continue
            
            # Δf50 regime analysis
            regime_changes = self._analyze_regime_flip(enhanced_data, event_idx)
            diagnostics['regime_flip_analysis'][f"event_{event_idx}"] = regime_changes
            
            # News proximity effects
            news_effects = self._analyze_news_effects(enhanced_data, event_idx)
            diagnostics['news_proximity_effects'][f"event_{event_idx}"] = news_effects
            
            # H1 confirmation impact
            h1_impact = self._analyze_h1_confirmation(enhanced_data, event_idx)
            diagnostics['h1_confirmation_impact'][f"event_{event_idx}"] = h1_impact
            
            # Gap context patterns
            gap_patterns = self._analyze_gap_context(enhanced_data, event_idx)
            diagnostics['gap_context_patterns'][f"event_{event_idx}"] = gap_patterns
            
            # Micro momentum switches
            momentum_switches = self._analyze_micro_momentum(enhanced_data, event_idx)
            diagnostics['micro_momentum_switches'][f"event_{event_idx}"] = momentum_switches
        
        return diagnostics
    
    def _analyze_regime_flip(self, data: pd.DataFrame, event_idx: int) -> Dict[str, Any]:
        """Analyze Δf50 regime changes near RD40"""
        if event_idx >= len(data) - 5:
            return {}
        
        pre_window = data.iloc[max(0, event_idx-5):event_idx]
        post_window = data.iloc[event_idx:min(len(data), event_idx+5)]
        
        if len(pre_window) > 0 and 'f50_htf_regime' in pre_window.columns:
            pre_regime = pre_window['f50_htf_regime'].mean()
        else:
            pre_regime = 1  # Default transition
        
        if len(post_window) > 0 and 'f50_htf_regime' in post_window.columns:
            post_regime = post_window['f50_htf_regime'].mean()
        else:
            post_regime = 1  # Default transition
        
        regime_change = abs(post_regime - pre_regime)
        
        return {
            'pre_regime_avg': pre_regime,
            'post_regime_avg': post_regime,
            'regime_change_magnitude': regime_change,
            'flips_cont_to_mr': regime_change > 0.5 and post_regime < pre_regime,
            'flips_mr_to_cont': regime_change > 0.5 and post_regime > pre_regime
        }
    
    def _analyze_news_effects(self, data: pd.DataFrame, event_idx: int) -> Dict[str, Any]:
        """Analyze news proximity effects on path selection"""
        event_row = data.iloc[event_idx] if event_idx < len(data) else pd.Series()
        
        # Use energy_density as proxy for news impact
        energy_density = event_row.get('energy_density', 0.5)
        
        return {
            'high_impact_detected': energy_density > 0.8,
            'suppresses_cont_accel': energy_density > 0.8,
            'boosts_mr_failure': energy_density > 0.8,
            'energy_density': energy_density
        }
    
    def _analyze_h1_confirmation(self, data: pd.DataFrame, event_idx: int) -> Dict[str, Any]:
        """Analyze H1 confirmation impact on ACCEL probability"""
        h1_breakout = self._detect_h1_breakout(data, event_idx, 15)
        
        return {
            'adds_accel_probability': h1_breakout['detected'] and h1_breakout['direction_aligned'],
            'breakout_strength': h1_breakout.get('breakout_strength', 0),
            'confluence_score': h1_breakout.get('confluence_score', 0.5)
        }
    
    def _analyze_gap_context(self, data: pd.DataFrame, event_idx: int) -> Dict[str, Any]:
        """Analyze gap context effects on path selection"""
        event_row = data.iloc[event_idx] if event_idx < len(data) else pd.Series()
        
        gap_age_d = event_row.get('gap_age_d', 3)
        magnitude = event_row.get('magnitude', 0)
        
        return {
            'fresh_gap_detected': gap_age_d <= 2,
            'aligned_unresolved_rd': gap_age_d <= 2 and magnitude > 0.3,
            'favors_cont': gap_age_d <= 2 and magnitude > 0.3,
            'stale_gap_favors_mr': gap_age_d > 2
        }
    
    def _analyze_micro_momentum(self, data: pd.DataFrame, event_idx: int) -> Dict[str, Any]:
        """Analyze micro momentum changes in 3-5 bars after RD40"""
        if event_idx >= len(data) - 5:
            return {}
        
        post_event_window = data.iloc[event_idx:min(len(data), event_idx+5)]
        
        if len(post_event_window) > 0 and 'f8_slope_sign' in post_event_window.columns:
            f8_slope_changes = post_event_window['f8_slope_sign'].tolist()
        else:
            f8_slope_changes = [0] * min(5, len(data) - event_idx)
        
        # Detect sign changes
        sign_changes = 0
        for i in range(1, len(f8_slope_changes)):
            if f8_slope_changes[i] != f8_slope_changes[i-1]:
                sign_changes += 1
        
        return {
            'sign_changes_detected': sign_changes > 0,
            'predicts_mr': sign_changes >= 2,
            'slope_sequence': f8_slope_changes,
            'change_count': sign_changes
        }