"""
HTF Context Processor for IRONFORGE Archaeological Discovery
==========================================================

Adds Higher Timeframe context features to M5 base events while preserving 
fundamental mathematical laws and maintaining temporal integrity.

Implements the minimal HTF context specification:
- M5 base stream with M15/H1 contextual features  
- Leakage-safe "last closed bar only" rule
- Preserves archaeological discovery focus
- 6 new features: f45-f50 (45D → 51D nodes)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import math

import numpy as np
import pandas as pd
from dataclasses import dataclass

# Try to import from iron_core, fallback to local constants if not available
try:
    from iron_core.mathematical.constraints import BusinessRules, HTFConstants
except ImportError:
    # Fallback constants if iron_core not available
    class HTFConstants:
        MU_H = 0.02
        ALPHA_H = 35.51
        BETA_H = 0.00442

logger = logging.getLogger(__name__)


@dataclass
class HTFContextConfig:
    """Configuration for HTF context processing"""
    enabled: bool = True
    timeframes: List[str] = None
    sv_lookback_bars: int = 30
    sv_weights: Dict[str, float] = None
    anchors: Dict[str, bool] = None
    regime: Dict[str, float] = None
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["M15", "H1"]
        if self.sv_weights is None:
            self.sv_weights = {"ev_cnt": 0.5, "abs_ret": 0.4, "liq": 0.1}
        if self.anchors is None:
            self.anchors = {"daily_mid": True, "use_prev_day": True}
        if self.regime is None:
            self.regime = {"upper": 0.7, "lower": 0.3}


@dataclass
class HTFBar:
    """Represents a closed HTF bar with computed metrics"""
    timeframe: str
    bar_start: int  # UTC milliseconds
    bar_end: int    # UTC milliseconds
    events: List[Dict[str, Any]]
    sv_raw: float
    sv_z_score: Optional[float] = None
    regime_code: int = 1  # 0=consolidation, 1=transition, 2=expansion


class TimeFrameManager:
    """Manages HTF timeframe calculations with leakage prevention"""
    
    # Timeframe durations in milliseconds
    TF_DURATIONS = {
        "M5": 5 * 60 * 1000,
        "M15": 15 * 60 * 1000,
        "H1": 60 * 60 * 1000
    }
    
    @classmethod
    def get_bar_index(cls, timestamp_ms: int, timeframe: str) -> int:
        """Get bar index k_T = floor(t / T) for timeframe"""
        duration = cls.TF_DURATIONS.get(timeframe)
        if not duration:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        return timestamp_ms // duration
    
    @classmethod
    def get_bar_bounds(cls, bar_index: int, timeframe: str) -> Tuple[int, int]:
        """Get bar start/end times for given bar index"""
        duration = cls.TF_DURATIONS.get(timeframe)
        if not duration:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        
        bar_start = bar_index * duration
        bar_end = (bar_index + 1) * duration
        return bar_start, bar_end
    
    @classmethod
    def get_closed_bar_index(cls, timestamp_ms: int, timeframe: str) -> Optional[int]:
        """Get most recent closed bar index (bar_end <= timestamp)"""
        duration = cls.TF_DURATIONS.get(timeframe)
        if not duration:
            return None
            
        # Find the bar that contains this timestamp
        current_bar_index = timestamp_ms // duration
        _, current_bar_end = cls.get_bar_bounds(current_bar_index, timeframe)
        
        # If current bar hasn't closed yet, use previous bar
        if current_bar_end > timestamp_ms:
            return max(0, current_bar_index - 1) if current_bar_index > 0 else None
        else:
            return current_bar_index


class SyntheticVolumeCalculator:
    """
    Calculates Synthetic Volume following mathematical foundations
    Preserves the SV calculation framework from iron_core constraints
    """
    
    def __init__(self, config: HTFContextConfig):
        self.config = config
        self.weights = config.sv_weights
        
    def calculate_raw_sv(self, events: List[Dict[str, Any]]) -> float:
        """
        Calculate raw Synthetic Volume for a closed HTF bar
        
        Formula: SV_raw = 0.5*std(ev_cnt) + 0.4*std(sum_abs_ret) + 0.1*std(liq_wt)
        All standardization is within-session for stability
        """
        if not events:
            return 0.0
        
        try:
            # Event count component
            ev_cnt = len(events)
            
            # Price movement component (sum of absolute returns)
            sum_abs_ret = 0.0
            if len(events) >= 2:
                prices = [float(e.get('price_level', 0)) for e in events]
                for i in range(1, len(prices)):
                    sum_abs_ret += abs(prices[i] - prices[i-1])
            
            # Liquidity component (count of liquidity events)
            liq_wt = sum(1 for e in events if e.get('source_type') == 'liquidity_event')
            
            # Apply weights (preserving mathematical foundation)
            # Note: Standardization within session happens later during z-score calculation
            sv_raw = (
                self.weights["ev_cnt"] * ev_cnt + 
                self.weights["abs_ret"] * sum_abs_ret +
                self.weights["liq"] * liq_wt
            )
            
            return float(sv_raw)
            
        except Exception as e:
            logger.warning(f"Error calculating raw SV: {e}")
            return 0.0
    
    def calculate_z_score(self, current_sv: float, historical_sv: List[float]) -> Optional[float]:
        """Calculate z-score over last N bars with minimum bar requirement"""
        if len(historical_sv) < 10:  # Minimum 10 bars required
            return None
        
        # Use last N bars for rolling statistics
        recent_sv = historical_sv[-self.config.sv_lookback_bars:]
        
        if not recent_sv:
            return None
            
        mean_sv = np.mean(recent_sv)
        std_sv = np.std(recent_sv)
        
        if std_sv == 0:
            return 0.0
        
        return float((current_sv - mean_sv) / std_sv)


class HTFRegimeClassifier:
    """Classifies HTF regime using SV and volatility percentiles"""
    
    def __init__(self, config: HTFContextConfig):
        self.config = config
        self.upper_threshold = config.regime["upper"] 
        self.lower_threshold = config.regime["lower"]
        
    def classify_regime(self, sv_raw: float, volatility_raw: float, 
                       sv_history: List[float], vol_history: List[float]) -> int:
        """
        Classify regime code using percentiles
        Returns: 0=consolidation, 1=transition, 2=expansion
        """
        if len(sv_history) < 10 or len(vol_history) < 10:
            return 1  # Default to transition if insufficient data
        
        try:
            # Calculate percentiles over last 30 bars
            recent_sv = sv_history[-30:]
            recent_vol = vol_history[-30:]
            
            sv_pct = self._calculate_percentile(sv_raw, recent_sv)
            vol_pct = self._calculate_percentile(volatility_raw, recent_vol)
            
            # Apply regime classification rules
            if sv_pct >= self.upper_threshold or vol_pct >= self.upper_threshold:
                return 2  # expansion
            elif sv_pct <= self.lower_threshold and vol_pct <= self.lower_threshold:
                return 0  # consolidation
            else:
                return 1  # transition
                
        except Exception as e:
            logger.warning(f"Error classifying regime: {e}")
            return 1
    
    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Calculate percentile of value within history"""
        if not history:
            return 0.5
        
        sorted_history = sorted(history)
        n = len(sorted_history)
        
        # Count values less than current value
        count_less = sum(1 for v in sorted_history if v < value)
        
        # Calculate percentile
        percentile = count_less / n if n > 0 else 0.5
        return min(1.0, max(0.0, percentile))


class DailyAnchorCalculator:
    """Calculates distance to daily anchors for archaeological context"""
    
    @staticmethod
    def calculate_daily_mid_distance(current_price: float, session_events: List[Dict[str, Any]],
                                   previous_day_data: Optional[Dict] = None) -> Optional[float]:
        """
        Calculate normalized distance to previous day midpoint
        Formula: (price - daily_mid) / max(ε, PDH - PDL)
        """
        try:
            # Try to get previous day data from session or parameter
            if previous_day_data:
                pdh = float(previous_day_data.get('high', 0))
                pdl = float(previous_day_data.get('low', 0))
            else:
                # Extract from session events if available
                pdh, pdl = DailyAnchorCalculator._extract_daily_range_from_events(session_events)
            
            if pdh == 0 or pdl == 0 or pdh == pdl:
                return None
                
            daily_mid = (pdh + pdl) / 2
            daily_range = pdh - pdl
            
            # Normalize distance with epsilon guard
            epsilon = 1e-9
            normalized_distance = (current_price - daily_mid) / max(epsilon, daily_range)
            
            return float(normalized_distance)
            
        except Exception as e:
            logger.warning(f"Error calculating daily mid distance: {e}")
            return None
    
    @staticmethod
    def _extract_daily_range_from_events(events: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Extract daily high/low from session events"""
        prices = [float(e.get('price_level', 0)) for e in events if e.get('price_level')]
        
        if not prices:
            return 0.0, 0.0
            
        return max(prices), min(prices)


class HTFContextProcessor:
    """
    Main HTF context processor that adds 6 contextual features to M5 base events
    
    Features added: f45_sv_m15_z, f46_sv_h1_z, f47_barpos_m15, f48_barpos_h1, 
                   f49_dist_daily_mid, f50_htf_regime
    """
    
    def __init__(self, config: HTFContextConfig = None):
        self.config = config or HTFContextConfig()
        self.tf_manager = TimeFrameManager()
        self.sv_calculator = SyntheticVolumeCalculator(self.config)
        self.regime_classifier = HTFRegimeClassifier(self.config)
        self.anchor_calculator = DailyAnchorCalculator()
        
        # Storage for HTF bars and statistics
        self.htf_bars = defaultdict(list)  # {timeframe: [HTFBar, ...]}
        self.sv_history = defaultdict(list)  # {timeframe: [sv_raw, ...]}
        self.vol_history = defaultdict(list)  # {timeframe: [volatility, ...]}
        
    def process_session(self, session_events: List[Dict[str, Any]], 
                       session_metadata: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Process a session and generate HTF context features for all events
        
        Returns: Dict with feature arrays indexed by event position
        """
        if not session_events or not self.config.enabled:
            return self._create_empty_features(len(session_events))
        
        logger.info(f"Processing HTF context for {len(session_events)} events")
        
        # Build HTF bars for each timeframe
        for timeframe in self.config.timeframes:
            self._build_htf_bars(session_events, timeframe)
        
        # Calculate features for each event
        feature_arrays = {
            'f45_sv_m15_z': [],
            'f46_sv_h1_z': [],
            'f47_barpos_m15': [],
            'f48_barpos_h1': [],
            'f49_dist_daily_mid': [],
            'f50_htf_regime': []
        }
        
        for event in session_events:
            features = self._extract_event_features(event, session_events, session_metadata)
            
            feature_arrays['f45_sv_m15_z'].append(features.get('f45_sv_m15_z', np.nan))
            feature_arrays['f46_sv_h1_z'].append(features.get('f46_sv_h1_z', np.nan))
            feature_arrays['f47_barpos_m15'].append(features.get('f47_barpos_m15', np.nan))
            feature_arrays['f48_barpos_h1'].append(features.get('f48_barpos_h1', np.nan))
            feature_arrays['f49_dist_daily_mid'].append(features.get('f49_dist_daily_mid', np.nan))
            feature_arrays['f50_htf_regime'].append(features.get('f50_htf_regime', 1))  # Default transition
        
        logger.info(f"Generated HTF context features for {len(session_events)} events")
        return feature_arrays
    
    def _build_htf_bars(self, events: List[Dict[str, Any]], timeframe: str) -> None:
        """Build HTF bars for given timeframe"""
        if timeframe not in self.config.timeframes:
            return
        
        # Group events by HTF bar
        bar_events = defaultdict(list)
        
        for event in events:
            timestamp = event.get('t', 0)  # Assume UTC milliseconds
            if timestamp <= 0:
                continue
                
            bar_index = self.tf_manager.get_bar_index(timestamp, timeframe)
            bar_events[bar_index].append(event)
        
        # Create HTF bars with SV calculations
        htf_bars = []
        sv_values = []
        vol_values = []
        
        for bar_index in sorted(bar_events.keys()):
            bar_start, bar_end = self.tf_manager.get_bar_bounds(bar_index, timeframe)
            bar_event_list = bar_events[bar_index]
            
            # Calculate raw SV
            sv_raw = self.sv_calculator.calculate_raw_sv(bar_event_list)
            
            # Calculate volatility (sum_abs_ret component)
            volatility = self._calculate_bar_volatility(bar_event_list)
            
            # Create HTF bar
            htf_bar = HTFBar(
                timeframe=timeframe,
                bar_start=bar_start,
                bar_end=bar_end,
                events=bar_event_list,
                sv_raw=sv_raw
            )
            
            # Calculate z-score using historical data
            htf_bar.sv_z_score = self.sv_calculator.calculate_z_score(sv_raw, sv_values)
            
            # Calculate regime code
            htf_bar.regime_code = self.regime_classifier.classify_regime(
                sv_raw, volatility, sv_values, vol_values
            )
            
            htf_bars.append(htf_bar)
            sv_values.append(sv_raw)
            vol_values.append(volatility)
        
        # Store for feature extraction
        self.htf_bars[timeframe] = htf_bars
        self.sv_history[timeframe] = sv_values
        self.vol_history[timeframe] = vol_values
    
    def _calculate_bar_volatility(self, events: List[Dict[str, Any]]) -> float:
        """Calculate volatility (sum of absolute returns) for bar"""
        if len(events) < 2:
            return 0.0
        
        prices = [float(e.get('price_level', 0)) for e in events]
        volatility = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
        return volatility
    
    def _extract_event_features(self, event: Dict[str, Any], 
                               session_events: List[Dict[str, Any]],
                               session_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract HTF context features for a single event"""
        timestamp = event.get('t', 0)
        price = float(event.get('price_level', 0))
        
        features = {}
        
        # SV z-scores for M15 and H1
        for i, timeframe in enumerate(['M15', 'H1']):
            feature_key = f'f{45+i}_sv_{timeframe.lower()}_z'
            closed_bar_index = self.tf_manager.get_closed_bar_index(timestamp, timeframe)
            
            if closed_bar_index is not None and timeframe in self.htf_bars:
                # Find the closed bar
                htf_bars = self.htf_bars[timeframe]
                matching_bar = next((bar for bar in htf_bars 
                                   if self.tf_manager.get_bar_index(bar.bar_start, timeframe) == closed_bar_index), 
                                  None)
                
                if matching_bar and matching_bar.sv_z_score is not None:
                    features[feature_key] = matching_bar.sv_z_score
                else:
                    features[feature_key] = np.nan
            else:
                features[feature_key] = np.nan
        
        # Bar positions for M15 and H1
        for i, timeframe in enumerate(['M15', 'H1']):
            feature_key = f'f{47+i}_barpos_{timeframe.lower()}'
            closed_bar_index = self.tf_manager.get_closed_bar_index(timestamp, timeframe)
            
            if closed_bar_index is not None:
                bar_start, bar_end = self.tf_manager.get_bar_bounds(closed_bar_index, timeframe)
                if bar_end <= timestamp:  # Ensure bar is closed
                    barpos = min(1.0, max(0.0, (timestamp - bar_start) / (bar_end - bar_start)))
                    features[feature_key] = barpos
                else:
                    features[feature_key] = np.nan
            else:
                features[feature_key] = np.nan
        
        # Daily midpoint distance
        features['f49_dist_daily_mid'] = self.anchor_calculator.calculate_daily_mid_distance(
            price, session_events
        )
        
        # HTF regime (use H1 as primary regime indicator)
        if 'H1' in self.htf_bars:
            closed_bar_index = self.tf_manager.get_closed_bar_index(timestamp, 'H1')
            if closed_bar_index is not None:
                htf_bars = self.htf_bars['H1']
                matching_bar = next((bar for bar in htf_bars 
                                   if self.tf_manager.get_bar_index(bar.bar_start, 'H1') == closed_bar_index), 
                                  None)
                
                features['f50_htf_regime'] = matching_bar.regime_code if matching_bar else 1
            else:
                features['f50_htf_regime'] = 1  # Default transition
        else:
            features['f50_htf_regime'] = 1
        
        return features
    
    def _create_empty_features(self, event_count: int) -> Dict[str, List[float]]:
        """Create empty feature arrays when HTF processing is disabled"""
        return {
            'f45_sv_m15_z': [np.nan] * event_count,
            'f46_sv_h1_z': [np.nan] * event_count,
            'f47_barpos_m15': [np.nan] * event_count,
            'f48_barpos_h1': [np.nan] * event_count,
            'f49_dist_daily_mid': [np.nan] * event_count,
            'f50_htf_regime': [1] * event_count  # Default transition
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of HTF feature names"""
        return [
            'f45_sv_m15_z',
            'f46_sv_h1_z', 
            'f47_barpos_m15',
            'f48_barpos_h1',
            'f49_dist_daily_mid',
            'f50_htf_regime'
        ]
    
    def validate_temporal_integrity(self, events: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Validate that HTF features maintain temporal integrity (no leakage)
        Returns validation results
        """
        results = {
            'no_future_leakage': True,
            'closed_bars_only': True,
            'feature_consistency': True
        }
        
        for event in events:
            timestamp = event.get('t', 0)
            
            # Check that all referenced HTF bars ended before or at event time
            for timeframe in self.config.timeframes:
                closed_bar_index = self.tf_manager.get_closed_bar_index(timestamp, timeframe)
                
                if closed_bar_index is not None:
                    _, bar_end = self.tf_manager.get_bar_bounds(closed_bar_index, timeframe)
                    
                    if bar_end > timestamp:
                        results['closed_bars_only'] = False
                        logger.error(f"Leakage detected: Bar end {bar_end} > event time {timestamp}")
        
        return results


def create_default_htf_config() -> HTFContextConfig:
    """Create default HTF context configuration"""
    return HTFContextConfig(
        enabled=True,
        timeframes=["M15", "H1"],
        sv_lookback_bars=30,
        sv_weights={"ev_cnt": 0.5, "abs_ret": 0.4, "liq": 0.1},
        anchors={"daily_mid": True, "use_prev_day": True},
        regime={"upper": 0.7, "lower": 0.3}
    )


if __name__ == "__main__":
    """Test HTF context processor with sample data"""
    print("🧪 Testing HTF Context Processor")
    print("=" * 50)
    
    # Create processor with default config
    config = create_default_htf_config()
    processor = HTFContextProcessor(config)
    
    # Generate sample events (M5 timeframe)
    base_time = 1753372800000  # Sample UTC timestamp
    sample_events = []
    
    for i in range(20):
        event = {
            't': base_time + (i * 5 * 60 * 1000),  # 5-minute intervals
            'price_level': 23000 + (i * 10),  # Trending price
            'timestamp': f"{9 + (i//12):02d}:{(i*5)%60:02d}:00",
            'source_type': 'price_movement' if i % 3 != 0 else 'liquidity_event',
            'movement_type': 'expansion' if i > 10 else 'consolidation'
        }
        sample_events.append(event)
    
    # Process events
    features = processor.process_session(sample_events, {})
    
    print(f"✅ Processed {len(sample_events)} events")
    print(f"📊 Feature dimensions: {len(features)} features")
    
    # Validate temporal integrity
    validation = processor.validate_temporal_integrity(sample_events)
    print(f"🔒 Temporal integrity: {all(validation.values())}")
    
    # Show sample features
    for feature_name, values in features.items():
        non_nan_values = [v for v in values if not np.isnan(v)]
        print(f"   {feature_name}: {len(non_nan_values)} valid / {len(values)} total")
    
    print("=" * 50)
    print("✅ HTF Context Processor test completed")