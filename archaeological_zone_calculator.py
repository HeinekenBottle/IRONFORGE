#!/usr/bin/env python3
"""
IRONFORGE Archaeological Zone Calculator
Theory B integration for temporal non-locality and dimensional destiny analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, time
from session_time_manager import SessionTimeManager

@dataclass
class ZoneEvent:
    """Archaeological zone event with Theory B characteristics"""
    timestamp: str
    price: float
    zone_type: str
    dimensional_destiny: bool
    session_progress_pct: float
    distance_to_final_level: float
    precision_score: float
    event_family: str = "archaeological_zones"

class ArchaeologicalZoneCalculator:
    """
    Calculates and analyzes archaeological zones with Theory B temporal non-locality
    
    Key Discovery: 40% zone events position with 7.55 point precision to final session ranges
    before the session range is fully established (temporal non-locality)
    """
    
    def __init__(self):
        self.session_manager = SessionTimeManager()
        
        # Theory B constants from your empirical discovery
        self.THEORY_B_PRECISION_THRESHOLD = 7.55  # Points
        self.DIMENSIONAL_DESTINY_ZONES = [0.40, 0.60, 0.80]  # 40%, 60%, 80%
        
        # Zone type mappings from existing data
        self.ZONE_MAPPINGS = {
            (0.0, 0.2): {"type": "transitional_zone", "significance": 0.1},
            (0.2, 0.4): {"type": "structural_support_20pct", "significance": 0.3},
            (0.4, 0.6): {"type": "dimensional_destiny_40pct", "significance": 0.8},
            (0.6, 0.8): {"type": "resistance_confluence_60pct", "significance": 0.7},
            (0.8, 1.0): {"type": "momentum_threshold_80pct", "significance": 0.9}
        }
    
    def calculate_zones_for_session(self, session_high: float, session_low: float) -> Dict[str, Any]:
        """
        Calculate all archaeological zones for a session
        
        Args:
            session_high: Final session high
            session_low: Final session low
            
        Returns:
            Dict with zone levels and Theory B analysis framework
        """
        session_range = session_high - session_low
        
        if session_range <= 0:
            return {"error": "Invalid session range", "session_range": session_range}
        
        zones = {}
        for pct in [0.2, 0.4, 0.6, 0.8, 1.0]:
            level = session_low + (session_range * pct)
            zones[f"{int(pct * 100)}%"] = {
                "level": level,
                "percentage": pct,
                "distance_from_low": session_range * pct,
                "is_dimensional_destiny": pct in self.DIMENSIONAL_DESTINY_ZONES
            }
        
        return {
            "session_range": session_range,
            "session_high": session_high,
            "session_low": session_low,
            "zones": zones,
            "theory_b_framework": {
                "precision_threshold": self.THEORY_B_PRECISION_THRESHOLD,
                "dimensional_zones": [f"{int(z*100)}%" for z in self.DIMENSIONAL_DESTINY_ZONES],
                "temporal_non_locality": "Events position relative to eventual completion"
            }
        }
    
    def analyze_event_positioning(self, event_price: float, event_time: str, 
                                session_type: str, final_session_stats: Dict[str, float],
                                current_session_stats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Analyze event positioning with Theory B temporal non-locality
        
        Args:
            event_price: Price at event occurrence
            event_time: Event timestamp
            session_type: Session type (NYAM, NYPM, etc.)
            final_session_stats: Final session high/low/open/close
            current_session_stats: Session stats at time of event (for non-locality analysis)
            
        Returns:
            Complete Theory B analysis
        """
        # Get session zones based on final range
        zone_analysis = self.calculate_zones_for_session(
            final_session_stats['session_high'],
            final_session_stats['session_low']
        )
        
        # Get temporal context
        temporal_context = self.session_manager.calculate_session_progress(session_type, event_time)
        
        # Calculate event's position in final range
        final_range = final_session_stats['session_high'] - final_session_stats['session_low']
        position_in_final_range = (event_price - final_session_stats['session_low']) / final_range
        
        # Find closest archaeological zone
        closest_zone = None
        min_distance = float('inf')
        
        for zone_pct, zone_data in zone_analysis['zones'].items():
            distance = abs(event_price - zone_data['level'])
            if distance < min_distance:
                min_distance = distance
                closest_zone = {
                    "zone": zone_pct,
                    "level": zone_data['level'],
                    "distance": distance,
                    "is_dimensional_destiny": zone_data['is_dimensional_destiny']
                }
        
        # Theory B temporal non-locality analysis
        theory_b_analysis = self._analyze_temporal_non_locality(
            event_price, event_time, final_session_stats, current_session_stats
        )
        
        # Determine zone classification
        zone_classification = self._classify_zone_position(position_in_final_range)
        
        return {
            "event_price": event_price,
            "event_time": event_time,
            "session_type": session_type,
            "temporal_context": temporal_context,
            "position_in_final_range": round(position_in_final_range * 100, 1),
            "closest_zone": closest_zone,
            "zone_classification": zone_classification,
            "theory_b_analysis": theory_b_analysis,
            "dimensional_relationship": zone_classification["dimensional_relationship"],
            "archaeological_significance": zone_classification["significance"],
            "session_stats": final_session_stats
        }
    
    def _analyze_temporal_non_locality(self, event_price: float, event_time: str,
                                     final_stats: Dict[str, float], 
                                     current_stats: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze temporal non-locality - events positioning relative to eventual completion
        
        This is the core Theory B analysis from your discovery
        """
        final_range = final_stats['session_high'] - final_stats['session_low']
        final_40_zone = final_stats['session_low'] + (final_range * 0.4)
        
        # Distance to eventual 40% zone
        distance_to_final_40 = abs(event_price - final_40_zone)
        
        analysis = {
            "distance_to_final_40pct": distance_to_final_40,
            "meets_theory_b_precision": distance_to_final_40 <= self.THEORY_B_PRECISION_THRESHOLD,
            "precision_score": max(0, (self.THEORY_B_PRECISION_THRESHOLD - distance_to_final_40) / self.THEORY_B_PRECISION_THRESHOLD),
            "temporal_non_locality": distance_to_final_40 <= self.THEORY_B_PRECISION_THRESHOLD
        }
        
        # If we have current stats, analyze the non-locality effect
        if current_stats:
            current_range = current_stats.get('session_high', event_price) - current_stats.get('session_low', event_price)
            if current_range > 0:
                current_40_zone = current_stats.get('session_low', event_price) + (current_range * 0.4)
                distance_to_current_40 = abs(event_price - current_40_zone)
                
                analysis.update({
                    "distance_to_current_40pct": distance_to_current_40,
                    "current_vs_final_precision_ratio": distance_to_final_40 / distance_to_current_40 if distance_to_current_40 > 0 else 0,
                    "demonstrates_non_locality": distance_to_final_40 < distance_to_current_40,
                    "non_locality_strength": max(0, distance_to_current_40 - distance_to_final_40)
                })
        
        return analysis
    
    def _classify_zone_position(self, position_pct: float) -> Dict[str, Any]:
        """Classify zone position based on percentage in range"""
        for (min_pct, max_pct), zone_info in self.ZONE_MAPPINGS.items():
            if min_pct <= position_pct < max_pct:
                return {
                    "zone_range": f"{int(min_pct*100)}-{int(max_pct*100)}%",
                    "dimensional_relationship": zone_info["type"],
                    "significance": zone_info["significance"],
                    "position_pct": round(position_pct * 100, 1)
                }
        
        # Handle edge case for 100%
        if position_pct >= 1.0:
            return {
                "zone_range": "80-100%",
                "dimensional_relationship": "momentum_threshold_80pct",
                "significance": 0.9,
                "position_pct": round(position_pct * 100, 1)
            }
        
        return {
            "zone_range": "unknown",
            "dimensional_relationship": "transitional_zone", 
            "significance": 0.1,
            "position_pct": round(position_pct * 100, 1)
        }
    
    def find_theory_b_candidates(self, session_events: List[Dict[str, Any]], 
                               final_session_stats: Dict[str, float]) -> List[ZoneEvent]:
        """
        Find events that demonstrate Theory B temporal non-locality
        
        Args:
            session_events: List of events with price, timestamp data
            final_session_stats: Final session statistics
            
        Returns:
            List of ZoneEvent objects that meet Theory B criteria
        """
        candidates = []
        
        for event in session_events:
            if 'price' not in event or 'timestamp' not in event:
                continue
            
            analysis = self.analyze_event_positioning(
                event['price'], 
                event['timestamp'],
                event.get('session_type', 'UNKNOWN'),
                final_session_stats
            )
            
            # Check if this meets Theory B criteria
            theory_b = analysis['theory_b_analysis']
            if theory_b['meets_theory_b_precision'] and analysis['closest_zone']['is_dimensional_destiny']:
                
                zone_event = ZoneEvent(
                    timestamp=event['timestamp'],
                    price=event['price'],
                    zone_type=analysis['dimensional_relationship'],
                    dimensional_destiny=True,
                    session_progress_pct=analysis['temporal_context']['session_progress_pct'],
                    distance_to_final_level=theory_b['distance_to_final_40pct'],
                    precision_score=theory_b['precision_score']
                )
                
                candidates.append(zone_event)
        
        # Sort by precision score (best precision first)
        candidates.sort(key=lambda x: x.precision_score, reverse=True)
        
        return candidates
    
    def generate_zone_report(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive archaeological zone report for a session
        
        Args:
            session_data: Complete session data with events and final stats
            
        Returns:
            Detailed report on zone activity and Theory B findings
        """
        final_stats = session_data['session_stats']
        events = session_data.get('events', [])
        session_type = session_data.get('session_type', 'UNKNOWN')
        
        # Calculate zones
        zone_framework = self.calculate_zones_for_session(
            final_stats['session_high'],
            final_stats['session_low']
        )
        
        # Find Theory B candidates
        theory_b_candidates = self.find_theory_b_candidates(events, final_stats)
        
        # Analyze zone activity distribution
        zone_activity = {}
        for event in events:
            if 'price' not in event:
                continue
                
            analysis = self.analyze_event_positioning(
                event['price'],
                event.get('timestamp', '00:00:00'),
                session_type,
                final_stats
            )
            
            zone = analysis['dimensional_relationship']
            if zone not in zone_activity:
                zone_activity[zone] = 0
            zone_activity[zone] += 1
        
        return {
            "session_type": session_type,
            "session_range": final_stats['session_high'] - final_stats['session_low'],
            "zone_framework": zone_framework,
            "theory_b_candidates": len(theory_b_candidates),
            "best_theory_b_event": theory_b_candidates[0] if theory_b_candidates else None,
            "zone_activity_distribution": zone_activity,
            "dimensional_destiny_events": len([c for c in theory_b_candidates if c.dimensional_destiny]),
            "average_precision": np.mean([c.precision_score for c in theory_b_candidates]) if theory_b_candidates else 0,
            "temporal_non_locality_confirmed": any(c.precision_score > 0.5 for c in theory_b_candidates)
        }

# Testing and demonstration
def demo_archaeological_zones():
    """Demonstrate Archaeological Zone Calculator with Theory B example"""
    print("🏛️ IRONFORGE Archaeological Zone Calculator Demo")
    print("=" * 60)
    
    calculator = ArchaeologicalZoneCalculator()
    
    # Your Theory B discovery example
    print("\n1. Theory B Discovery Analysis (2025-08-05 PM)")
    print("   Event: 14:35:00 at 23162.25 (40% zone)")
    print("   Session Low established 18 minutes later at 14:53:00")
    
    final_stats = {
        'session_high': 23375.5,
        'session_low': 23148.5,
        'session_open': 23169.25,
        'session_close': 23368.0
    }
    
    analysis = calculator.analyze_event_positioning(
        event_price=23162.25,
        event_time="14:35:00",
        session_type="NYPM",
        final_session_stats=final_stats
    )
    
    print(f"\n   📊 Analysis Results:")
    print(f"   Position in final range: {analysis['position_in_final_range']}%")
    print(f"   Closest zone: {analysis['closest_zone']['zone']} ({analysis['closest_zone']['distance']:.2f} points away)")
    print(f"   Theory B precision: {analysis['theory_b_analysis']['distance_to_final_40pct']:.2f} points")
    print(f"   Meets Theory B threshold: {analysis['theory_b_analysis']['meets_theory_b_precision']}")
    print(f"   Dimensional relationship: {analysis['dimensional_relationship']}")
    print(f"   Temporal progress: {analysis['temporal_context']['session_progress_pct']}% through session")
    
    # Zone framework analysis
    print("\n2. Complete Zone Framework")
    zones = calculator.calculate_zones_for_session(final_stats['session_high'], final_stats['session_low'])
    
    for zone_pct, zone_data in zones['zones'].items():
        destiny_marker = " ⭐ DIMENSIONAL DESTINY" if zone_data['is_dimensional_destiny'] else ""
        print(f"   {zone_pct}: {zone_data['level']:.2f}{destiny_marker}")
    
    print(f"\n   Session Range: {zones['session_range']:.1f} points")
    print(f"   Theory B Threshold: {zones['theory_b_framework']['precision_threshold']} points")
    
    # Test multiple events
    print("\n3. Multiple Event Analysis")
    test_events = [
        {"price": 23162.25, "timestamp": "14:35:00", "description": "Original Theory B event"},
        {"price": 23260.00, "timestamp": "15:00:00", "description": "60% zone test"},
        {"price": 23330.00, "timestamp": "15:30:00", "description": "80% zone test"}
    ]
    
    for event in test_events:
        analysis = calculator.analyze_event_positioning(
            event['price'], event['timestamp'], "NYPM", final_stats
        )
        theory_b = analysis['theory_b_analysis']['meets_theory_b_precision']
        zone = analysis['dimensional_relationship'].replace('_', ' ').replace('pct', '%')
        
        print(f"   {event['timestamp']}: {event['price']:.2f} → {zone} {'✅ Theory B' if theory_b else ''}")

if __name__ == "__main__":
    demo_archaeological_zones()