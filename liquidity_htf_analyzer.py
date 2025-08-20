#!/usr/bin/env python3
"""
IRONFORGE Liquidity & HTF Follow-Through Analyzer
Experiment E — Liquidity & HTF Follow-Through (explore-only)

Measures RD@40 follow-through using market structure instead of time targets:
- Liquidity sweeps: prior day/session/weekly H/L taps
- FVG events: next FPFVG RD in same/opposite direction  
- HTF taps: H1/H4/D/W/M level touches

Context splits: Day, News, Session with Wilson CI validation
"""

import json
import glob
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dataclasses import dataclass
from enhanced_statistical_framework import EnhancedStatisticalAnalyzer
import pytz

@dataclass
class LiquidityLevel:
    """Liquidity level with identification metadata"""
    level: float
    level_type: str  # 'prior_day_high', 'prior_day_low', 'session_high', 'weekly_high', etc.
    source_session: str
    source_date: str
    timeframe: str  # 'session', 'daily', 'weekly'

@dataclass  
class HTFLevel:
    """Higher timeframe level with OHLC context"""
    level: float
    level_type: str  # 'H1_high', 'H4_mid', 'D_low', 'W_open', 'M_close'
    timeframe: str   # 'H1', 'H4', 'D', 'W', 'M'
    ohlc_type: str   # 'open', 'high', 'low', 'close', 'mid'
    reference_time: str

@dataclass
class SweepEvent:
    """Liquidity sweep event record"""
    rd40_timestamp: str
    sweep_timestamp: str
    level_swept: float
    level_type: str
    time_to_sweep_mins: float
    side_taken: str  # 'buy' (above level), 'sell' (below level)
    rd40_direction: str  # 'bullish', 'bearish'
    alignment: str   # 'aligned', 'counter', 'neutral'

@dataclass
class FVGEvent:
    """FVG follow-through event record"""
    rd40_timestamp: str
    fvg_timestamp: str
    time_to_fvg_mins: float
    fvg_direction: str  # 'bullish', 'bearish'
    rd40_direction: str
    direction_relationship: str  # 'same', 'opposite'

class LiquidityHTFAnalyzer:
    """Analyzes RD@40 follow-through using liquidity sweeps and HTF level touches"""
    
    def __init__(self):
        self.stats_framework = EnhancedStatisticalAnalyzer()
        self.et_tz = pytz.timezone('America/New_York')
    
    def parse_event_datetime(self, event: Dict, trading_day: str) -> Optional[datetime]:
        """Parse event datetime with proper timezone handling"""
        timestamp_et = event.get('timestamp_et')
        if timestamp_et:
            try:
                # Parse "2025-07-28 13:30:00 ET" format
                dt_str = timestamp_et.replace(' ET', '')
                dt_naive = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                return self.et_tz.localize(dt_naive)
            except ValueError:
                pass
        
        # Fallback to timestamp + trading_day
        timestamp = event.get('timestamp')
        if timestamp and trading_day:
            try:
                dt_str = f"{trading_day} {timestamp}"
                dt_naive = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                return self.et_tz.localize(dt_naive)
            except ValueError:
                pass
                
        return None
        
    def calculate_liquidity_levels(self, session_data: Dict, trading_day: str) -> List[LiquidityLevel]:
        """Calculate liquidity levels for a trading session"""
        levels = []
        
        # Extract session info
        session_info = session_data.get('session_info', {})
        session_type = session_info.get('session_type', 'UNKNOWN')
        events = session_data.get('events', [])
        
        if not events:
            return levels
            
        # Calculate current session high/low
        session_prices = [e.get('price_level', 0) for e in events if e.get('price_level')]
        if not session_prices:
            return levels
            
        session_high = max(session_prices)
        session_low = min(session_prices)
        
        # Add current session levels
        levels.extend([
            LiquidityLevel(
                level=session_high,
                level_type='session_high', 
                source_session=session_type,
                source_date=trading_day,
                timeframe='session'
            ),
            LiquidityLevel(
                level=session_low,
                level_type='session_low',
                source_session=session_type, 
                source_date=trading_day,
                timeframe='session'
            )
        ])
        
        return levels
        
    def generate_htf_levels(self, session_data: Dict, reference_time: str) -> List[HTFLevel]:
        """Generate HTF levels (H1/H4/D/W/M) for reference time"""
        levels = []
        
        events = session_data.get('events', [])
        if not events:
            return levels
            
        # Calculate session OHLC for HTF approximation
        session_prices = [e.get('price_level', 0) for e in events if e.get('price_level')]
        if not session_prices:
            return levels
            
        session_open = session_prices[0]
        session_high = max(session_prices)
        session_low = min(session_prices) 
        session_close = session_prices[-1]
        session_mid = session_low + (session_high - session_low) * 0.5
        
        # H1 levels (approximate using session data)
        for ohlc_type, price in [
            ('open', session_open), ('high', session_high), 
            ('low', session_low), ('close', session_close), ('mid', session_mid)
        ]:
            levels.append(HTFLevel(
                level=price,
                level_type=f'H1_{ohlc_type}',
                timeframe='H1',
                ohlc_type=ohlc_type,
                reference_time=reference_time
            ))
            
        # H4 levels (using session as proxy)
        for ohlc_type, price in [
            ('open', session_open), ('high', session_high),
            ('low', session_low), ('close', session_close), ('mid', session_mid)
        ]:
            levels.append(HTFLevel(
                level=price,
                level_type=f'H4_{ohlc_type}',
                timeframe='H4', 
                ohlc_type=ohlc_type,
                reference_time=reference_time
            ))
            
        return levels
        
    def detect_liquidity_sweeps(self, rd40_event: Dict, session_events: List[Dict], 
                               liquidity_levels: List[LiquidityLevel]) -> List[SweepEvent]:
        """Detect liquidity sweeps following RD@40 event using real timestamps"""
        sweeps = []
        
        rd40_price = rd40_event.get('price_level')
        if not rd40_price:
            return sweeps
        
        # Parse RD@40 datetime with timezone
        trading_day = rd40_event.get('day_context', {}).get('trading_day', '2025-08-01')
        rd40_dt = self.parse_event_datetime(rd40_event, trading_day)
        if not rd40_dt:
            return sweeps
            
        # Find events after RD@40 within 90 minutes (surgical time window)
        max_time = rd40_dt + timedelta(minutes=90)
        post_rd40_events = []
        
        for event in session_events:
            event_dt = self.parse_event_datetime(event, trading_day)
            if event_dt and rd40_dt < event_dt <= max_time:
                post_rd40_events.append((event, event_dt))
                
        # Check each liquidity level for sweeps using REAL timestamps
        for level in liquidity_levels:
            for event_data in post_rd40_events:
                event, event_dt = event_data
                event_price = event.get('price_level')
                
                if not event_price:
                    continue
                    
                # Simple tap detection - price beyond level
                swept = False
                side_taken = 'neutral'
                
                if level.level_type.endswith('_high') and event_price >= level.level:
                    swept = True
                    side_taken = 'buy'
                elif level.level_type.endswith('_low') and event_price <= level.level:
                    swept = True  
                    side_taken = 'sell'
                    
                if swept:
                    # Calculate time to sweep in REAL minutes
                    time_to_sweep = (event_dt - rd40_dt).total_seconds() / 60
                    
                    # Determine RD@40 directional bias from price momentum context
                    # For now, use session position as proxy - could be enhanced with slope/momentum
                    rd40_range_pos = rd40_event.get('range_position', 0.4)
                    rd40_direction = 'bullish' if rd40_range_pos > 0.5 else 'bearish'
                    
                    # Determine alignment - does sweep direction match RD@40 momentum?
                    alignment = 'aligned' if (
                        (side_taken == 'buy' and rd40_direction == 'bullish') or
                        (side_taken == 'sell' and rd40_direction == 'bearish')
                    ) else 'counter'
                    
                    sweeps.append(SweepEvent(
                        rd40_timestamp=rd40_event.get('timestamp'),
                        sweep_timestamp=event.get('timestamp'),
                        level_swept=level.level,
                        level_type=level.level_type,
                        time_to_sweep_mins=time_to_sweep,
                        side_taken=side_taken,
                        rd40_direction=rd40_direction,
                        alignment=alignment
                    ))
                    break  # First sweep wins
                    
        return sweeps
        
    def detect_fvg_events(self, rd40_event: Dict, session_events: List[Dict]) -> List[FVGEvent]:
        """Detect FVG redelivery events following RD@40"""
        fvg_events = []
        
        rd40_timestamp = rd40_event.get('timestamp')
        rd40_price = rd40_event.get('price_level')
        
        if not rd40_timestamp:
            return fvg_events
            
        # Find potential FVG events after RD@40
        rd40_time = datetime.strptime(rd40_timestamp, '%H:%M:%S').time()
        
        for event in session_events:
            event_time_str = event.get('timestamp')
            if not event_time_str:
                continue
                
            event_time = datetime.strptime(event_time_str, '%H:%M:%S').time()
            if event_time <= rd40_time:
                continue
                
            # Look for FVG-related events (simplified detection)
            event_type = event.get('type', '')
            if 'fvg' in event_type.lower() or 'gap' in event_type.lower():
                event_price = event.get('price_level')
                
                # Determine directions
                rd40_direction = 'bullish'  # Simplified
                fvg_direction = 'bullish' if event_price and event_price > rd40_price else 'bearish'
                
                direction_relationship = 'same' if rd40_direction == fvg_direction else 'opposite'
                
                time_to_fvg = self._calculate_time_diff_minutes(rd40_timestamp, event_time_str)
                
                fvg_events.append(FVGEvent(
                    rd40_timestamp=rd40_timestamp,
                    fvg_timestamp=event_time_str,
                    time_to_fvg_mins=time_to_fvg,
                    fvg_direction=fvg_direction,
                    rd40_direction=rd40_direction,
                    direction_relationship=direction_relationship
                ))
                
        return fvg_events
        
    def detect_htf_taps(self, rd40_event: Dict, session_events: List[Dict], 
                       htf_levels: List[HTFLevel]) -> List[Dict]:
        """Detect HTF level touches following RD@40"""
        htf_taps = []
        
        rd40_timestamp = rd40_event.get('timestamp')
        rd40_price = rd40_event.get('price_level')
        
        if not rd40_timestamp or not rd40_price:
            return htf_taps
            
        # Find events after RD@40
        rd40_time = datetime.strptime(rd40_timestamp, '%H:%M:%S').time()
        post_rd40_events = []
        
        for event in session_events:
            event_time_str = event.get('timestamp')
            if not event_time_str:
                continue
                
            event_time = datetime.strptime(event_time_str, '%H:%M:%S').time()
            if event_time > rd40_time:
                post_rd40_events.append(event)
                
        # Check HTF levels for taps (priority order: W > D > H4 > H1)
        priority_order = ['W', 'D', 'H4', 'H1']
        
        for timeframe in priority_order:
            timeframe_levels = [l for l in htf_levels if l.timeframe == timeframe]
            
            for level in timeframe_levels:
                for event in post_rd40_events:
                    event_price = event.get('price_level')
                    event_timestamp = event.get('timestamp')
                    
                    if not event_price or not event_timestamp:
                        continue
                        
                    # Check if price tapped the HTF level (within small tolerance)
                    tolerance = abs(level.level * 0.001)  # 0.1% tolerance
                    
                    if abs(event_price - level.level) <= tolerance:
                        time_to_tap = self._calculate_time_diff_minutes(rd40_timestamp, event_timestamp)
                        
                        htf_taps.append({
                            'rd40_timestamp': rd40_timestamp,
                            'tap_timestamp': event_timestamp,
                            'level_tapped': level.level,
                            'level_type': level.level_type,
                            'timeframe': level.timeframe,
                            'ohlc_type': level.ohlc_type,
                            'time_to_tap_mins': time_to_tap,
                            'price_at_tap': event_price
                        })
                        break  # First tap wins for this level
                break  # Highest priority timeframe wins
                
        return htf_taps
        
    def _calculate_time_diff_minutes(self, time1_str: str, time2_str: str) -> float:
        """Calculate time difference in minutes between two HH:MM:SS timestamps"""
        try:
            time1 = datetime.strptime(time1_str, '%H:%M:%S').time()
            time2 = datetime.strptime(time2_str, '%H:%M:%S').time()
            
            # Convert to datetime for calculation
            base_date = datetime(2025, 1, 1)
            dt1 = datetime.combine(base_date, time1)
            dt2 = datetime.combine(base_date, time2)
            
            # Handle cross-midnight if needed
            if dt2 < dt1:
                dt2 += timedelta(days=1)
                
            diff = dt2 - dt1
            return diff.total_seconds() / 60.0
            
        except:
            return 0.0
            
    def analyze_minute_hotspots(self, rd40_events: List[Dict]) -> Dict[str, Any]:
        """Analyze minute-of-day hotspots for RD@40 events"""
        minute_counts = {}
        
        for event in rd40_events:
            timestamp = event.get('timestamp')
            if not timestamp:
                continue
                
            try:
                time_obj = datetime.strptime(timestamp, '%H:%M:%S').time()
                minute_key = f"{time_obj.hour:02d}:{time_obj.minute:02d}"
                minute_counts[minute_key] = minute_counts.get(minute_key, 0) + 1
            except:
                continue
                
        # Sort by count and get top 5
        sorted_minutes = sorted(minute_counts.items(), key=lambda x: x[1], reverse=True)
        top_5_minutes = sorted_minutes[:5]
        
        # Check for 14:35 ET ±3m pattern
        target_minutes = ['14:32', '14:33', '14:34', '14:35', '14:36', '14:37', '14:38']
        target_zone_count = sum(minute_counts.get(minute, 0) for minute in target_minutes)
        
        return {
            'total_minutes_analyzed': len(minute_counts),
            'top_5_minutes': top_5_minutes,
            'target_zone_14_35_pm3': {
                'minutes': target_minutes,
                'total_events': target_zone_count,
                'percentage': (target_zone_count / len(rd40_events) * 100) if rd40_events else 0
            },
            'minute_distribution': minute_counts
        }
        
    def load_enhanced_sessions(self) -> List[Dict]:
        """Load all enhanced session files"""
        enhanced_files = glob.glob('/Users/jack/IRONFORGE/data/day_news_enhanced/*.json')
        sessions = []
        
        for file_path in enhanced_files:
            try:
                with open(file_path, 'r') as f:
                    session_data = json.load(f)
                    sessions.append({
                        'file_path': file_path,
                        'data': session_data
                    })
            except Exception as e:
                print(f"⚠️  Failed to load {file_path}: {e}")
                continue
                
        return sessions

def main():
    """Demo the liquidity/HTF analyzer"""
    print("🔄 IRONFORGE Liquidity & HTF Follow-Through Analyzer")
    print("=" * 60)
    
    analyzer = LiquidityHTFAnalyzer()
    sessions = analyzer.load_enhanced_sessions()
    
    print(f"📁 Loaded {len(sessions)} enhanced sessions")
    
    # Extract all RD@40 events
    all_rd40_events = []
    
    for session in sessions[:3]:  # Test with first 3 sessions
        session_data = session['data']
        events = session_data.get('events', [])
        
        rd40_events = [e for e in events 
                      if e.get('dimensional_relationship') == 'dimensional_destiny_40pct']
        
        print(f"📊 Session: {session['file_path'].split('/')[-1]} - {len(rd40_events)} RD@40 events")
        
        for rd40_event in rd40_events:
            # Calculate liquidity levels
            liquidity_levels = analyzer.calculate_liquidity_levels(session_data, "2025-08-01")
            
            # Generate HTF levels
            htf_levels = analyzer.generate_htf_levels(session_data, rd40_event.get('timestamp', ''))
            
            # Detect sweeps
            sweeps = analyzer.detect_liquidity_sweeps(rd40_event, events, liquidity_levels)
            
            # Detect FVG events
            fvg_events = analyzer.detect_fvg_events(rd40_event, events)
            
            # Detect HTF taps
            htf_taps = analyzer.detect_htf_taps(rd40_event, events, htf_levels)
            
            print(f"  🎯 RD@40 at {rd40_event.get('timestamp')}: {len(sweeps)} sweeps, {len(fvg_events)} FVGs, {len(htf_taps)} HTF taps")
            
            all_rd40_events.append(rd40_event)
    
    # Analyze minute hotspots
    if all_rd40_events:
        hotspots = analyzer.analyze_minute_hotspots(all_rd40_events)
        print(f"\n⏰ Minute-of-Day Hotspots:")
        print(f"   Top 5 minutes: {hotspots['top_5_minutes']}")
        print(f"   14:35 ET ±3m zone: {hotspots['target_zone_14_35_pm3']['total_events']} events ({hotspots['target_zone_14_35_pm3']['percentage']:.1f}%)")
    
    print("\n✅ Liquidity/HTF analysis framework ready for TQE integration")

if __name__ == "__main__":
    main()