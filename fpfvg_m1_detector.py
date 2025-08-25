#!/usr/bin/env python3
"""
First Presented Fair Value Gap (FPFVG) M1 Detection Engine
Detects ICT Gauntlet FVG patterns at 1-minute resolution with archaeological integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FPFVGEvent:
    """First Presented Fair Value Gap event with precise M1 timing"""
    formation_time: str
    premium_high: float
    discount_low: float
    consequent_encroachment: float  # Exact midpoint
    gap_size: float
    candle_sequence: List[Dict[str, float]]  # The 3-candle formation
    session_context: Dict[str, Any]
    archaeological_proximity: Optional[float] = None
    validation_status: str = "detected"

@dataclass
class GauntletSequence:
    """Complete ICT Gauntlet sequence with M1 precision"""
    fpfvg: FPFVGEvent
    liquidity_hunt_time: Optional[str] = None
    liquidity_hunt_price: Optional[float] = None
    reversal_time: Optional[str] = None
    ce_breach_time: Optional[str] = None
    ce_breach_price: Optional[float] = None
    completion_status: str = "partial"
    archaeological_confluence: Optional[Dict[str, Any]] = None

class FPFVGM1Detector:
    """M1 Resolution FPFVG Detection for ICT Gauntlet Pattern Discovery"""
    
    def __init__(self):
        # ICT FPFVG Detection Parameters
        self.MIN_GAP_SIZE_POINTS = 2.0  # Minimum gap size to avoid noise
        self.SESSION_START_TIME = "09:30:00"  # ET - Cash market open
        self.MACRO_WINDOW_START = "09:50:00"  # ET - ICT macro window begins
        self.MACRO_WINDOW_END = "10:10:00"    # ET - ICT macro window ends
        
        # Archaeological Zone Integration
        self.ARCHAEOLOGICAL_ZONES = [0.40, 0.60, 0.80]  # RD@40%, RD@60%, RD@80%
        self.ZONE_PROXIMITY_THRESHOLD = 10.0  # Points for confluence
        
    def detect_fpfvg_from_m1_data(self, m1_ohlc_data: pd.DataFrame, 
                                session_type: str, session_date: str) -> List[FPFVGEvent]:
        """
        Detect First Presented Fair Value Gaps in M1 OHLC data
        
        Args:
            m1_ohlc_data: DataFrame with M1 OHLC data (timestamp, open, high, low, close)
            session_type: Session type (ny_am, ny_pm, etc.)
            session_date: Session date (YYYY-MM-DD)
            
        Returns:
            List of detected FPFVG events
        """
        if len(m1_ohlc_data) < 3:
            return []
        
        # Filter for session timeframe (9:30 AM - 10:10 AM ET for AM session)
        session_data = self._filter_session_timeframe(m1_ohlc_data, session_type)
        
        if len(session_data) < 3:
            return []
        
        fpfvg_events = []
        session_fpfvgs_found = {}  # Track first presented by gap level
        
        # Scan for 3-candle FVG patterns
        for i in range(len(session_data) - 2):
            candle1 = session_data.iloc[i]
            candle2 = session_data.iloc[i + 1]
            candle3 = session_data.iloc[i + 2]
            
            # Check for bullish FVG pattern
            bullish_fvg = self._detect_bullish_fvg(candle1, candle2, candle3)
            if bullish_fvg and self._is_first_presented(bullish_fvg, session_fpfvgs_found, 'bullish'):
                fpfvg_event = self._create_fpfvg_event(
                    bullish_fvg, [candle1, candle2, candle3], 
                    session_type, session_date, 'bullish'
                )
                fpfvg_events.append(fpfvg_event)
                session_fpfvgs_found[bullish_fvg['gap_key']] = True
            
            # Check for bearish FVG pattern
            bearish_fvg = self._detect_bearish_fvg(candle1, candle2, candle3)
            if bearish_fvg and self._is_first_presented(bearish_fvg, session_fpfvgs_found, 'bearish'):
                fpfvg_event = self._create_fpfvg_event(
                    bearish_fvg, [candle1, candle2, candle3], 
                    session_type, session_date, 'bearish'
                )
                fpfvg_events.append(fpfvg_event)
                session_fpfvgs_found[bearish_fvg['gap_key']] = True
        
        return fpfvg_events
    
    def _filter_session_timeframe(self, data: pd.DataFrame, session_type: str) -> pd.DataFrame:
        """Filter data for relevant session timeframe"""
        if session_type.lower() == 'ny_am':
            # Focus on 9:30 AM - 10:10 AM ET for AM session FVG formation
            start_time = self.SESSION_START_TIME
            end_time = self.MACRO_WINDOW_END
        else:
            # For other sessions, use full data
            return data
        
        # Filter by time (assuming timestamp column exists)
        if 'timestamp' in data.columns:
            data['time_only'] = pd.to_datetime(data['timestamp']).dt.strftime('%H:%M:%S')
            filtered = data[
                (data['time_only'] >= start_time) & 
                (data['time_only'] <= end_time)
            ].copy()
            return filtered
        
        return data
    
    def _detect_bullish_fvg(self, candle1: pd.Series, candle2: pd.Series, 
                           candle3: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Detect bullish Fair Value Gap pattern
        Pattern: candle1.high < candle3.low (gap between them)
        """
        if candle1['high'] < candle3['low']:
            gap_size = candle3['low'] - candle1['high']
            
            if gap_size >= self.MIN_GAP_SIZE_POINTS:
                return {
                    'direction': 'bullish',
                    'premium_high': candle3['low'],
                    'discount_low': candle1['high'],
                    'gap_size': gap_size,
                    'consequent_encroachment': (candle1['high'] + candle3['low']) / 2,
                    'formation_time': candle3['timestamp'],
                    'gap_key': f"bullish_{candle1['high']:.2f}_{candle3['low']:.2f}"
                }
        return None
    
    def _detect_bearish_fvg(self, candle1: pd.Series, candle2: pd.Series, 
                           candle3: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Detect bearish Fair Value Gap pattern  
        Pattern: candle1.low > candle3.high (gap between them)
        """
        if candle1['low'] > candle3['high']:
            gap_size = candle1['low'] - candle3['high']
            
            if gap_size >= self.MIN_GAP_SIZE_POINTS:
                return {
                    'direction': 'bearish',
                    'premium_high': candle1['low'],
                    'discount_low': candle3['high'],
                    'gap_size': gap_size,
                    'consequent_encroachment': (candle1['low'] + candle3['high']) / 2,
                    'formation_time': candle3['timestamp'],
                    'gap_key': f"bearish_{candle1['low']:.2f}_{candle3['high']:.2f}"
                }
        return None
    
    def _is_first_presented(self, fvg_data: Dict[str, Any], 
                           session_found: Dict[str, bool], direction: str) -> bool:
        """
        Check if this is the first presentation of this FVG level in the session
        """
        gap_key = fvg_data['gap_key']
        return gap_key not in session_found
    
    def _create_fpfvg_event(self, fvg_data: Dict[str, Any], candles: List[pd.Series],
                           session_type: str, session_date: str, direction: str) -> FPFVGEvent:
        """Create FPFVG event from detected pattern"""
        candle_sequence = []
        for i, candle in enumerate(candles):
            candle_sequence.append({
                'sequence': i + 1,
                'timestamp': candle['timestamp'],
                'open': candle['open'],
                'high': candle['high'], 
                'low': candle['low'],
                'close': candle['close']
            })
        
        session_context = {
            'session_type': session_type,
            'session_date': session_date,
            'direction': direction,
            'formation_phase': self._get_formation_phase(fvg_data['formation_time']),
            'macro_window_proximity': self._get_macro_window_proximity(fvg_data['formation_time'])
        }
        
        return FPFVGEvent(
            formation_time=fvg_data['formation_time'],
            premium_high=fvg_data['premium_high'],
            discount_low=fvg_data['discount_low'],
            consequent_encroachment=fvg_data['consequent_encroachment'],
            gap_size=fvg_data['gap_size'],
            candle_sequence=candle_sequence,
            session_context=session_context
        )
    
    def _get_formation_phase(self, formation_time: str) -> str:
        """Determine session phase when FVG formed"""
        time_only = formation_time.split()[1] if ' ' in formation_time else formation_time
        
        if time_only <= self.MACRO_WINDOW_START:
            return "pre_macro"
        elif time_only <= self.MACRO_WINDOW_END:
            return "macro_window"
        else:
            return "post_macro"
    
    def _get_macro_window_proximity(self, formation_time: str) -> Dict[str, Any]:
        """Calculate proximity to macro window timing"""
        time_only = formation_time.split()[1] if ' ' in formation_time else formation_time
        
        # Convert to datetime for calculation
        formation_dt = datetime.strptime(time_only, '%H:%M:%S')
        macro_start_dt = datetime.strptime(self.MACRO_WINDOW_START, '%H:%M:%S')
        macro_end_dt = datetime.strptime(self.MACRO_WINDOW_END, '%H:%M:%S')
        
        if formation_dt <= macro_start_dt:
            minutes_to_macro = (macro_start_dt - formation_dt).seconds // 60
            return {'phase': 'before_macro', 'minutes_to_macro': minutes_to_macro}
        elif formation_dt <= macro_end_dt:
            minutes_into_macro = (formation_dt - macro_start_dt).seconds // 60
            return {'phase': 'within_macro', 'minutes_into_macro': minutes_into_macro}
        else:
            minutes_after_macro = (formation_dt - macro_end_dt).seconds // 60
            return {'phase': 'after_macro', 'minutes_after_macro': minutes_after_macro}
    
    def detect_gauntlet_sequence(self, fpfvg_events: List[FPFVGEvent], 
                                m1_ohlc_data: pd.DataFrame,
                                session_high: float, session_low: float) -> List[GauntletSequence]:
        """
        Detect complete ICT Gauntlet sequences from FPFVG events
        
        Sequence: FPFVG → Liquidity Hunt → Reversal → CE Breach
        """
        gauntlet_sequences = []
        
        for fpfvg in fpfvg_events:
            sequence = GauntletSequence(fpfvg=fpfvg)
            
            # Add archaeological confluence analysis
            sequence.archaeological_confluence = self._analyze_archaeological_confluence(
                fpfvg, session_high, session_low
            )
            
            # Look for liquidity hunt after FVG formation
            hunt_result = self._detect_liquidity_hunt(fpfvg, m1_ohlc_data)
            if hunt_result:
                sequence.liquidity_hunt_time = hunt_result['time']
                sequence.liquidity_hunt_price = hunt_result['price']
                
                # Look for reversal after liquidity hunt
                reversal_result = self._detect_reversal(fpfvg, hunt_result, m1_ohlc_data)
                if reversal_result:
                    sequence.reversal_time = reversal_result['time']
                    
                    # Look for CE breach confirmation
                    ce_breach = self._detect_ce_breach(fpfvg, reversal_result, m1_ohlc_data)
                    if ce_breach:
                        sequence.ce_breach_time = ce_breach['time']
                        sequence.ce_breach_price = ce_breach['price']
                        sequence.completion_status = "complete"
                    else:
                        sequence.completion_status = "reversal_only"
                else:
                    sequence.completion_status = "hunt_only"
            
            gauntlet_sequences.append(sequence)
        
        return gauntlet_sequences
    
    def _analyze_archaeological_confluence(self, fpfvg: FPFVGEvent, 
                                         session_high: float, session_low: float) -> Dict[str, Any]:
        """Analyze proximity to archaeological zones for confluence"""
        session_range = session_high - session_low
        if session_range <= 0:
            return {'error': 'Invalid session range'}
        
        confluence_analysis = {
            'session_range': session_range,
            'fpfvg_position_pct': (fpfvg.consequent_encroachment - session_low) / session_range,
            'zone_proximities': {},
            'closest_zone': None,
            'confluence_strength': 0.0
        }
        
        # Check proximity to each archaeological zone
        min_distance = float('inf')
        closest_zone_pct = None
        
        for zone_pct in self.ARCHAEOLOGICAL_ZONES:
            zone_level = session_low + (session_range * zone_pct)
            distance = abs(fpfvg.consequent_encroachment - zone_level)
            
            confluence_analysis['zone_proximities'][f"{int(zone_pct*100)}%"] = {
                'level': zone_level,
                'distance': distance,
                'within_threshold': distance <= self.ZONE_PROXIMITY_THRESHOLD
            }
            
            if distance < min_distance:
                min_distance = distance
                closest_zone_pct = zone_pct
        
        if closest_zone_pct:
            confluence_analysis['closest_zone'] = {
                'zone': f"{int(closest_zone_pct*100)}%",
                'distance': min_distance,
                'confluence': min_distance <= self.ZONE_PROXIMITY_THRESHOLD
            }
            
            # Calculate confluence strength (inverse of distance, normalized)
            if min_distance <= self.ZONE_PROXIMITY_THRESHOLD:
                confluence_analysis['confluence_strength'] = max(0, 
                    (self.ZONE_PROXIMITY_THRESHOLD - min_distance) / self.ZONE_PROXIMITY_THRESHOLD
                )
        
        return confluence_analysis
    
    def _detect_liquidity_hunt(self, fpfvg: FPFVGEvent, 
                              m1_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect liquidity hunt phase after FPFVG formation"""
        # Implementation placeholder - detect expansion lower for bullish FVG
        # This would analyze price action after FVG formation looking for liquidity sweep
        return None  # TODO: Implement liquidity hunt detection
    
    def _detect_reversal(self, fpfvg: FPFVGEvent, hunt_result: Dict[str, Any],
                        m1_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect reversal after liquidity hunt"""
        # Implementation placeholder - detect reversal pattern
        return None  # TODO: Implement reversal detection
    
    def _detect_ce_breach(self, fpfvg: FPFVGEvent, reversal_result: Dict[str, Any],
                         m1_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect close beyond consequent encroachment"""
        # Implementation placeholder - detect CE breach confirmation
        return None  # TODO: Implement CE breach detection

    def process_session(self, session_file_path: str) -> Dict[str, Any]:
        """
        Process a complete session for FPFVG and Gauntlet detection
        
        Args:
            session_file_path: Path to session data (JSON or parquet)
            
        Returns:
            Complete session analysis with FPFVG and Gauntlet patterns
        """
        # TODO: Load session data, detect FPFVGs, analyze Gauntlet sequences
        # TODO: Integrate with existing enhanced session format
        # TODO: Return results compatible with IRONFORGE architecture
        
        return {
            'session_analysis': 'TODO: Implement session processing',
            'fpfvg_events': [],
            'gauntlet_sequences': [],
            'archaeological_confluence': {}
        }

# Testing and validation
def demo_fpfvg_detection():
    """Demonstrate FPFVG M1 Detection with sample data"""
    print("🔍 FPFVG M1 Detection Engine Demo")
    print("=" * 50)
    
    detector = FPFVGM1Detector()
    
    # Create sample M1 OHLC data for testing
    sample_data = pd.DataFrame({
        'timestamp': ['2025-07-29 09:32:00', '2025-07-29 09:33:00', '2025-07-29 09:34:00'],
        'open': [23590.0, 23595.0, 23602.0],
        'high': [23595.0, 23600.0, 23610.0],
        'low': [23588.0, 23593.0, 23601.0],
        'close': [23594.0, 23599.0, 23608.0]
    })
    
    print("\n📊 Sample M1 OHLC Data:")
    print(sample_data.to_string(index=False))
    
    # Detect FPFVGs
    fpfvg_events = detector.detect_fpfvg_from_m1_data(
        sample_data, 'ny_am', '2025-07-29'
    )
    
    print(f"\n🎯 FPFVG Detection Results:")
    print(f"   Events detected: {len(fpfvg_events)}")
    
    for i, event in enumerate(fpfvg_events):
        print(f"   Event {i+1}:")
        print(f"     Formation: {event.formation_time}")
        print(f"     Premium: {event.premium_high}")
        print(f"     Discount: {event.discount_low}")
        print(f"     CE: {event.consequent_encroachment}")
        print(f"     Gap Size: {event.gap_size:.2f} points")
        print(f"     Direction: {event.session_context['direction']}")

if __name__ == "__main__":
    demo_fpfvg_detection()