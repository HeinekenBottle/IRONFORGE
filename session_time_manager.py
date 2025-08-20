#!/usr/bin/env python3
"""
IRONFORGE Session Time Manager
Handles precise session timing and temporal calculations for price relativity
"""

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

@dataclass
class SessionSpec:
    """Session specification with timing and characteristics"""
    name: str
    start_time: time
    end_time: time
    duration_minutes: int
    timezone: str
    characteristics: Dict[str, Any]

class SessionTimeManager:
    """Manages session timing, duration, and temporal percentage calculations"""
    
    def __init__(self):
        self.session_specs = {
            # Trading day sequence (all times in ET)
            'ASIA': SessionSpec(
                name='ASIA',
                start_time=time(19, 0),   # 19:00:00 ET (starts on calendar day before)
                end_time=time(23, 59),    # 23:59:00 ET
                duration_minutes=299,     # 4 hours 59 minutes
                timezone='ET',
                characteristics={'volatility': 'low', 'participation': 'regional', 'cross_day': True}
            ),
            'MIDNIGHT': SessionSpec(
                name='MIDNIGHT',
                start_time=time(0, 0),    # 00:00:00 ET
                end_time=time(0, 29),     # 00:29:00 ET
                duration_minutes=29,      # 29 minutes
                timezone='ET',
                characteristics={'volatility': 'very_low', 'participation': 'minimal'}
            ),
            'LONDON': SessionSpec(
                name='LONDON',
                start_time=time(2, 0),    # 02:00:00 ET
                end_time=time(4, 59),     # 04:59:00 ET
                duration_minutes=179,     # 2 hours 59 minutes
                timezone='ET',
                characteristics={'volatility': 'medium', 'participation': 'institutional'}
            ),
            'PREMARKET': SessionSpec(
                name='PREMARKET',
                start_time=time(7, 0),    # 07:00:00 ET
                end_time=time(9, 29),     # 09:29:00 ET
                duration_minutes=149,     # 2 hours 29 minutes
                timezone='ET',
                characteristics={'volatility': 'low', 'participation': 'institutional_prep'}
            ),
            'NYAM': SessionSpec(
                name='NYAM',
                start_time=time(9, 30),   # 09:30:00 ET
                end_time=time(11, 59),    # 11:59:00 ET
                duration_minutes=149,     # 2 hours 29 minutes
                timezone='ET',
                characteristics={'volatility': 'high', 'participation': 'institutional'}
            ),
            'LUNCH': SessionSpec(
                name='LUNCH',
                start_time=time(12, 0),   # 12:00:00 ET
                end_time=time(12, 59),    # 12:59:00 ET
                duration_minutes=59,      # 59 minutes
                timezone='ET',
                characteristics={'volatility': 'very_low', 'participation': 'minimal'}
            ),
            'NYPM': SessionSpec(
                name='NYPM', 
                start_time=time(13, 30),  # 13:30:00 ET (algorithmic start)
                end_time=time(16, 10),    # 16:10:00 ET
                duration_minutes=160,     # 2 hours 40 minutes
                timezone='ET',
                characteristics={'volatility': 'medium', 'participation': 'mixed'}
            )
        }
    
    def get_session_spec(self, session_type: str) -> Optional[SessionSpec]:
        """Get session specification by type"""
        return self.session_specs.get(session_type.upper())
    
    def calculate_session_progress(self, session_type: str, event_time: str) -> Dict[str, Any]:
        """
        Calculate session progress for an event
        
        Args:
            session_type: Session type (NYAM, NYPM, etc.)
            event_time: Time string (e.g., "14:35:00" or "2025-08-05T14:35:00")
            
        Returns:
            Dict with absolute time, relative percentage, and session context
        """
        spec = self.get_session_spec(session_type)
        if not spec:
            return {"error": f"Unknown session type: {session_type}"}
        
        # Parse event time
        if 'T' in event_time:
            # Full timestamp format
            dt = datetime.fromisoformat(event_time.replace('T', ' '))
            event_time_obj = dt.time()
        else:
            # Just time format
            event_time_obj = datetime.strptime(event_time, "%H:%M:%S").time()
        
        # Calculate minutes from session start
        session_start_minutes = spec.start_time.hour * 60 + spec.start_time.minute
        event_minutes = event_time_obj.hour * 60 + event_time_obj.minute
        
        # Handle cross-day sessions (ASIA starts at 19:00, ends at 23:59 same day)
        if spec.name == 'ASIA':
            # ASIA runs 19:00-23:59 ET on same calendar day (no cross-day math needed)
            # The cross_day characteristic refers to it starting on calendar day before trading day
            pass
        elif spec.name == 'MIDNIGHT':
            # MIDNIGHT runs 00:00-00:29 ET (early morning hours)
            pass
        
        minutes_from_start = event_minutes - session_start_minutes
        
        # Calculate percentage through session
        session_progress_pct = min(minutes_from_start / spec.duration_minutes, 1.0) * 100
        
        # Classify session phase
        if session_progress_pct < 20:
            phase = "opening"
        elif session_progress_pct < 40:
            phase = "early"
        elif session_progress_pct < 60:
            phase = "mid"
        elif session_progress_pct < 80:
            phase = "late"
        else:
            phase = "closing"
        
        return {
            "absolute_time": event_time,
            "session_type": session_type,
            "session_progress_pct": round(session_progress_pct, 1),
            "minutes_from_start": minutes_from_start,
            "session_phase": phase,
            "session_duration": spec.duration_minutes,
            "session_start": spec.start_time.strftime("%H:%M:%S"),
            "session_end": spec.end_time.strftime("%H:%M:%S"),
            "timezone": spec.timezone,
            "characteristics": spec.characteristics
        }
    
    def calculate_archaeological_zones(self, session_high: float, session_low: float, 
                                     current_price: float) -> Dict[str, Any]:
        """
        Calculate archaeological zone positioning for Theory B analysis
        
        Args:
            session_high: Session high price
            session_low: Session low price  
            current_price: Current event price
            
        Returns:
            Zone analysis with both absolute and relative positioning
        """
        session_range = session_high - session_low
        
        if session_range <= 0:
            return {"error": "Invalid session range"}
        
        # Calculate zone levels
        zone_20 = session_low + (session_range * 0.2)
        zone_40 = session_low + (session_range * 0.4)
        zone_60 = session_low + (session_range * 0.6)
        zone_80 = session_low + (session_range * 0.8)
        
        # Current price percentage in range
        price_pct = ((current_price - session_low) / session_range) * 100
        
        # Determine which zone we're in
        if current_price >= zone_80:
            current_zone = "80%+ (momentum_threshold)"
            zone_type = "momentum_threshold_80pct"
        elif current_price >= zone_60:
            current_zone = "60-80% (resistance_confluence)"
            zone_type = "resistance_confluence_60pct"
        elif current_price >= zone_40:
            current_zone = "40-60% (dimensional_destiny)"
            zone_type = "dimensional_destiny_40pct"
        elif current_price >= zone_20:
            current_zone = "20-40% (structural_support)"
            zone_type = "structural_support_20pct"
        else:
            current_zone = "0-20% (base)"
            zone_type = "transitional_zone"
        
        # Calculate distances to key zones (Theory B precision analysis)
        distance_to_40 = abs(current_price - zone_40)
        distance_to_60 = abs(current_price - zone_60)
        distance_to_80 = abs(current_price - zone_80)
        
        return {
            "current_price": current_price,
            "session_range": session_range,
            "session_high": session_high,
            "session_low": session_low,
            "price_pct_in_range": round(price_pct, 1),
            "current_zone": current_zone,
            "zone_type": zone_type,
            "archaeological_zones": {
                "20%": {"level": zone_20, "distance": abs(current_price - zone_20)},
                "40%": {"level": zone_40, "distance": distance_to_40},
                "60%": {"level": zone_60, "distance": distance_to_60},
                "80%": {"level": zone_80, "distance": distance_to_80}
            },
            "theory_b_precision": {
                "distance_to_40pct": distance_to_40,
                "theory_b_threshold": 7.55,  # Known precision from your discovery
                "meets_precision": distance_to_40 <= 7.55
            }
        }
    
    def analyze_event_context(self, session_type: str, event_time: str, 
                            event_price: float, session_stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Complete contextual analysis combining temporal and price relativity
        
        Args:
            session_type: Session type
            event_time: Event timestamp
            event_price: Event price
            session_stats: Dict with session_high, session_low, session_open, session_close
            
        Returns:
            Complete analysis with temporal and archaeological context
        """
        # Get temporal context
        temporal_context = self.calculate_session_progress(session_type, event_time)
        
        # Get archaeological zone context
        zone_context = self.calculate_archaeological_zones(
            session_stats['session_high'],
            session_stats['session_low'],
            event_price
        )
        
        # Combine contexts
        return {
            **temporal_context,
            **zone_context,
            "event_price": event_price,
            "session_stats": session_stats,
            "combined_analysis": {
                "temporal_phase": temporal_context.get("session_phase"),
                "archaeological_zone": zone_context.get("current_zone"),
                "zone_type": zone_context.get("zone_type"),
                "theory_b_candidate": (
                    zone_context.get("theory_b_precision", {}).get("meets_precision", False) 
                    and temporal_context.get("session_progress_pct", 0) > 20
                ),
                "positioning_description": f"{temporal_context.get('session_phase', '')} session, {zone_context.get('current_zone', '')}"
            }
        }
    
    def get_session_taxonomy(self) -> Dict[str, Any]:
        """Return complete session taxonomy for reference"""
        return {
            "session_types": list(self.session_specs.keys()),
            "session_specs": {k: {
                "duration": v.duration_minutes,
                "start": v.start_time.strftime("%H:%M:%S"),
                "end": v.end_time.strftime("%H:%M:%S"),
                "timezone": v.timezone,
                "characteristics": v.characteristics
            } for k, v in self.session_specs.items()},
            "phase_definitions": {
                "opening": "0-20% through session",
                "early": "20-40% through session", 
                "mid": "40-60% through session",
                "late": "60-80% through session",
                "closing": "80-100% through session"
            },
            "archaeological_zones": {
                "0-20%": "transitional_zone / base level",
                "20-40%": "structural_support_20pct",
                "40-60%": "dimensional_destiny_40pct (Theory B zone)",
                "60-80%": "resistance_confluence_60pct", 
                "80-100%": "momentum_threshold_80pct"
            },
            "trading_day_info": {
                "trading_day_start": "18:00 ET (technical start)",
                "first_tracked_session": "ASIA at 19:00 ET",
                "asia_cross_day_note": "ASIA session starts on calendar day before trading day label",
                "session_sequence": "ASIA→MIDNIGHT→LONDON→PREMARKET→NYAM→LUNCH→NYPM"
            }
        }

# Testing and demo functions
def demo_session_time_manager():
    """Demonstrate SessionTimeManager capabilities"""
    print("🕐 IRONFORGE Session Time Manager Demo")
    print("=" * 50)
    
    manager = SessionTimeManager()
    
    # Test Theory B example from your discovery
    print("\n1. Theory B Example (2025-08-05 PM session)")
    context = manager.analyze_event_context(
        session_type="NYPM",
        event_time="14:35:00", 
        event_price=23162.25,
        session_stats={
            'session_high': 23375.5,
            'session_low': 23148.5, 
            'session_open': 23169.25,
            'session_close': 23368.0
        }
    )
    
    print(f"Time: {context['absolute_time']} ({context['session_progress_pct']}% through session)")
    print(f"Phase: {context['session_phase']}")
    print(f"Zone: {context['current_zone']}")
    print(f"Theory B precision: {context['theory_b_precision']['distance_to_40pct']:.2f} points from 40% zone")
    print(f"Meets precision: {context['theory_b_precision']['meets_precision']}")
    
    # Test different session types
    print("\n2. Session Type Examples")
    test_cases = [
        ("ASIA", "21:00:00", "Evening Asia trading (7PM ET)"),
        ("MIDNIGHT", "00:15:00", "Late night continuation"),
        ("LONDON", "03:30:00", "European market hours"), 
        ("PREMARKET", "08:00:00", "Pre-market preparation"),
        ("NYAM", "09:45:00", "Early morning institutional activity"),
        ("LUNCH", "12:30:00", "Low-volume lunch period"),
        ("NYPM", "14:00:00", "Afternoon algorithmic trading")
    ]
    
    for session, time_str, description in test_cases:
        progress = manager.calculate_session_progress(session, time_str)
        if 'error' not in progress:
            print(f"{session}: {time_str} = {progress['session_progress_pct']}% ({progress['session_phase']}) - {description}")
        else:
            print(f"{session}: {time_str} = ERROR: {progress['error']}")
    
    # Show taxonomy
    print("\n3. Session Taxonomy")
    taxonomy = manager.get_session_taxonomy()
    for session_type, spec in taxonomy['session_specs'].items():
        print(f"{session_type}: {spec['start']}-{spec['end']} ({spec['duration']}min, {spec['timezone']})")

if __name__ == "__main__":
    demo_session_time_manager()