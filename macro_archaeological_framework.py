#!/usr/bin/env python3
"""
Macro-Archaeological Trading Framework
Integration of IRONFORGE temporal intelligence with ICT macro timing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ArchaeologicalZone:
    """Theory B archaeological zone with precision metrics"""
    zone_pct: float  # 0.4, 0.6, 0.8
    level: float
    precision_score: float  # Distance to final session level
    timestamp: datetime
    zone_type: str  # "dimensional_destiny_40pct", etc.
    significance: float
    session_context: Dict[str, Any]

@dataclass
class MacroWindow:
    """ICT macro time window with orbital phases"""
    start_time: time
    end_time: time
    name: str
    orbital_phases: Dict[str, Tuple[int, int]]  # phase_name -> (start_offset, end_offset) in minutes
    
@dataclass
class GauntletSetup:
    """Gauntlet FVG setup with archaeological context"""
    fvg_level: float
    ce_level: float  # Consequent encroachment
    sweep_level: float
    archaeological_proximity: float  # Distance to nearest zone
    confidence_multiplier: float  # 1x or 2x based on archaeological alignment
    formation_time: datetime
    session_context: Dict[str, Any]

class MacroArchaeologicalFramework:
    """
    Enhanced AM scalping framework integrating IRONFORGE microstructure 
    with ICT macro timing and Gauntlet methodology
    """
    
    def __init__(self, data_dir="/Users/jack/IRONFORGE/data/shards/NQ_M5"):
        self.data_dir = Path(data_dir)
        
        # ICT Macro Windows (ET)
        self.macro_windows = {
            'morning_primary': MacroWindow(
                start_time=time(9, 50),
                end_time=time(10, 10),
                name='9:50-10:10 AM Primary',
                orbital_phases={
                    'setup': (-5, -2),
                    'entry': (-2, 2),
                    'extension': (2, 7),
                    'completion': (7, 10)
                }
            ),
            'morning_variant': MacroWindow(
                start_time=time(10, 20),
                end_time=time(10, 40),
                name='10:20-10:40 AM Variant',
                orbital_phases={
                    'setup': (-5, -2),
                    'entry': (-2, 2),
                    'extension': (2, 7),
                    'completion': (7, 10)
                }
            )
        }
        
        # Theory B constants from IRONFORGE validation
        self.THEORY_B_PRECISION_THRESHOLD = 7.55  # Points
        self.ARCHAEOLOGICAL_ZONES = [0.40, 0.60, 0.80]
        
        # Risk parameters
        self.BASE_CONFIDENCE = 1.0
        self.ARCHAEOLOGICAL_CONFIDENCE_MULTIPLIER = 2.0
        self.PROFIT_TARGETS = [10, 20, 30]  # Handles
        
        # Session data cache
        self.session_cache = {}
        self.current_archaeological_zones = []
        
    def load_prior_session_data(self, current_date: str, session_type: str = "NYPM") -> Optional[Dict]:
        """Load prior session data for archaeological zone calculation"""
        try:
            # Find prior session file
            session_files = list(self.data_dir.glob(f"shard_{session_type}_*"))
            prior_session_file = None
            
            for session_file in sorted(session_files):
                session_date = session_file.name.split('_')[-1]
                if session_date < current_date:
                    prior_session_file = session_file
            
            if not prior_session_file:
                return None
                
            # Load nodes data
            nodes_file = prior_session_file / "nodes.parquet"
            if not nodes_file.exists():
                return None
                
            nodes = pd.read_parquet(nodes_file)
            nodes['timestamp_et'] = pd.to_datetime(nodes['t'], unit='ms')
            
            return {
                'nodes': nodes,
                'session_name': prior_session_file.name.replace('shard_', ''),
                'session_high': nodes['price'].max(),
                'session_low': nodes['price'].min(),
                'session_range': nodes['price'].max() - nodes['price'].min()
            }
            
        except Exception as e:
            print(f"Error loading prior session: {e}")
            return None
    
    def calculate_archaeological_zones(self, session_data: Dict) -> List[ArchaeologicalZone]:
        """Calculate Theory B archaeological zones from prior session completion"""
        if not session_data or session_data['session_range'] <= 0:
            return []
        
        zones = []
        session_high = session_data['session_high']
        session_low = session_data['session_low']
        session_range = session_data['session_range']
        nodes = session_data['nodes']
        
        # Calculate archaeological zones
        for zone_pct in self.ARCHAEOLOGICAL_ZONES:
            zone_level = session_low + (session_range * zone_pct)
            
            # Find closest actual price point to zone level
            closest_idx = (nodes['price'] - zone_level).abs().idxmin()
            closest_row = nodes.loc[closest_idx]
            precision_score = abs(closest_row['price'] - zone_level)
            
            # Determine zone significance based on Theory B principles
            significance = 0.8 if zone_pct == 0.40 else (0.7 if zone_pct == 0.60 else 0.9)
            zone_type = f"dimensional_destiny_{int(zone_pct*100)}pct"
            
            zone = ArchaeologicalZone(
                zone_pct=zone_pct,
                level=zone_level,
                precision_score=precision_score,
                timestamp=closest_row['timestamp_et'],
                zone_type=zone_type,
                significance=significance,
                session_context={
                    'session_name': session_data['session_name'],
                    'session_high': session_high,
                    'session_low': session_low,
                    'session_range': session_range
                }
            )
            
            zones.append(zone)
        
        return zones
    
    def map_pre_open_anchors(self, current_date: str) -> Dict[str, Any]:
        """
        Enhanced pre-open anchor mapping including archaeological zones
        08:45–09:29 ET preparation phase
        """
        print(f"🗺️  Mapping pre-open anchors for {current_date}")
        
        anchors = {
            'timestamp': datetime.now(),
            'date': current_date,
            'traditional_anchors': {},
            'archaeological_zones': [],
            'confluence_levels': [],
            'risk_parameters': {}
        }
        
        # Load prior session for archaeological zones
        prior_session = self.load_prior_session_data(current_date)
        if prior_session:
            archaeological_zones = self.calculate_archaeological_zones(prior_session)
            anchors['archaeological_zones'] = archaeological_zones
            self.current_archaeological_zones = archaeological_zones
            
            print(f"  📍 Mapped {len(archaeological_zones)} archaeological zones:")
            for zone in archaeological_zones:
                print(f"    {zone.zone_type}: {zone.level:.2f} (precision: {zone.precision_score:.2f})")
        
        # Traditional anchor placeholders (would integrate with actual data feeds)
        anchors['traditional_anchors'] = {
            'london_high': None,  # Would pull from London session data
            'london_low': None,
            'asia_high': None,
            'asia_low': None,
            'midnight_open': None,
            'prior_day_imbalances': []
        }
        
        # Identify confluence levels (archaeological + traditional)
        confluence_levels = []
        for zone in anchors['archaeological_zones']:
            confluence_levels.append({
                'level': zone.level,
                'type': 'archaeological',
                'significance': zone.significance,
                'zone_info': zone
            })
        
        anchors['confluence_levels'] = sorted(confluence_levels, key=lambda x: x['significance'], reverse=True)
        
        return anchors
    
    def detect_gauntlet_archaeological_convergence(self, gauntlet_level: float, 
                                                  archaeological_zones: List[ArchaeologicalZone]) -> Dict[str, Any]:
        """
        Detect convergence between Gauntlet FVG and archaeological zones
        Returns confidence multiplier and convergence details
        """
        if not archaeological_zones:
            return {
                'convergence_detected': False,
                'confidence_multiplier': self.BASE_CONFIDENCE,
                'nearest_zone': None,
                'distance': float('inf')
            }
        
        # Find nearest archaeological zone
        distances = [(abs(gauntlet_level - zone.level), zone) for zone in archaeological_zones]
        nearest_distance, nearest_zone = min(distances, key=lambda x: x[0])
        
        # Check for Theory B precision convergence
        convergence_detected = nearest_distance <= self.THEORY_B_PRECISION_THRESHOLD
        confidence_multiplier = self.ARCHAEOLOGICAL_CONFIDENCE_MULTIPLIER if convergence_detected else self.BASE_CONFIDENCE
        
        return {
            'convergence_detected': convergence_detected,
            'confidence_multiplier': confidence_multiplier,
            'nearest_zone': nearest_zone,
            'distance': nearest_distance,
            'precision_analysis': {
                'within_theory_b_threshold': nearest_distance <= self.THEORY_B_PRECISION_THRESHOLD,
                'zone_significance': nearest_zone.significance,
                'zone_type': nearest_zone.zone_type
            }
        }
    
    def analyze_macro_orbital_phase(self, current_time: datetime, macro_window: MacroWindow) -> Dict[str, Any]:
        """
        Determine current orbital phase within macro window
        Returns phase context and timing recommendations
        """
        window_start = datetime.combine(current_time.date(), macro_window.start_time)
        window_end = datetime.combine(current_time.date(), macro_window.end_time)
        
        if current_time < window_start or current_time > window_end:
            return {'in_window': False, 'phase': 'outside_window'}
        
        # Calculate minutes from window start
        minutes_from_start = (current_time - window_start).total_seconds() / 60
        
        # Determine orbital phase
        current_phase = 'unknown'
        phase_context = {}
        
        for phase_name, (start_offset, end_offset) in macro_window.orbital_phases.items():
            adjusted_start = start_offset + 10  # Adjust for -5 to +10 range
            adjusted_end = end_offset + 10
            
            if adjusted_start <= minutes_from_start <= adjusted_end:
                current_phase = phase_name
                phase_context = {
                    'phase_start': window_start + timedelta(minutes=start_offset),
                    'phase_end': window_start + timedelta(minutes=end_offset),
                    'phase_progress': (minutes_from_start - adjusted_start) / (adjusted_end - adjusted_start),
                    'recommended_action': self._get_phase_action(phase_name)
                }
                break
        
        return {
            'in_window': True,
            'window_name': macro_window.name,
            'phase': current_phase,
            'phase_context': phase_context,
            'minutes_from_start': minutes_from_start
        }
    
    def _get_phase_action(self, phase_name: str) -> str:
        """Return recommended action for orbital phase"""
        actions = {
            'setup': 'Monitor for liquidity accumulation and range compression',
            'entry': 'Execute sweep → reclaim setups with archaeological confluence',
            'extension': 'Manage positions, look for momentum continuation',
            'completion': 'Prepare for retracement, scale out positions'
        }
        return actions.get(phase_name, 'Monitor market structure')
    
    def create_gauntlet_setup(self, fvg_level: float, ce_level: float, 
                             sweep_level: float, formation_time: datetime) -> GauntletSetup:
        """
        Create enhanced Gauntlet setup with archaeological context
        """
        # Check archaeological convergence
        convergence = self.detect_gauntlet_archaeological_convergence(
            ce_level, self.current_archaeological_zones
        )
        
        setup = GauntletSetup(
            fvg_level=fvg_level,
            ce_level=ce_level,
            sweep_level=sweep_level,
            archaeological_proximity=convergence['distance'],
            confidence_multiplier=convergence['confidence_multiplier'],
            formation_time=formation_time,
            session_context={
                'convergence_analysis': convergence,
                'archaeological_zones_count': len(self.current_archaeological_zones)
            }
        )
        
        return setup
    
    def calculate_archaeological_risk_levels(self, entry_price: float, 
                                           direction: str) -> Dict[str, float]:
        """
        Calculate risk management levels incorporating archaeological zones
        """
        risk_levels = {
            'traditional_stop': None,
            'archaeological_stop': None,
            'profit_targets': [],
            'trail_levels': []
        }
        
        # Find relevant archaeological zones for risk management
        relevant_zones = []
        for zone in self.current_archaeological_zones:
            if direction == 'long' and zone.level < entry_price:
                relevant_zones.append(zone)
            elif direction == 'short' and zone.level > entry_price:
                relevant_zones.append(zone)
        
        if relevant_zones:
            # Use nearest zone as additional risk reference
            nearest_zone = min(relevant_zones, key=lambda z: abs(z.level - entry_price))
            risk_levels['archaeological_stop'] = nearest_zone.level
        
        # Calculate profit targets with archaeological context
        for target_handles in self.PROFIT_TARGETS:
            target_price = entry_price + (target_handles if direction == 'long' else -target_handles)
            
            # Check if target aligns with archaeological zones
            archaeological_confluence = False
            for zone in self.current_archaeological_zones:
                if abs(zone.level - target_price) <= self.THEORY_B_PRECISION_THRESHOLD:
                    archaeological_confluence = True
                    break
            
            risk_levels['profit_targets'].append({
                'price': target_price,
                'handles': target_handles,
                'archaeological_confluence': archaeological_confluence
            })
        
        return risk_levels
    
    def generate_execution_checklist(self, current_time: datetime) -> Dict[str, Any]:
        """
        Generate enhanced execution checklist with archaeological context
        """
        checklist = {
            'timestamp': current_time,
            'pre_open_complete': False,
            'archaeological_zones_mapped': len(self.current_archaeological_zones) > 0,
            'macro_window_status': {},
            'gauntlet_readiness': {},
            'risk_parameters_set': False,
            'execution_recommendations': []
        }
        
        # Check macro window status
        for window_name, macro_window in self.macro_windows.items():
            orbital_analysis = self.analyze_macro_orbital_phase(current_time, macro_window)
            checklist['macro_window_status'][window_name] = orbital_analysis
        
        # Generate recommendations
        recommendations = []
        
        if not checklist['archaeological_zones_mapped']:
            recommendations.append("⚠️  Map archaeological zones from prior session completion")
        
        active_windows = [w for w in checklist['macro_window_status'].values() if w.get('in_window', False)]
        if active_windows:
            for window in active_windows:
                phase = window.get('phase', 'unknown')
                action = window.get('phase_context', {}).get('recommended_action', 'Monitor')
                recommendations.append(f"🎯 {window['window_name']}: {phase.upper()} phase - {action}")
        
        if len(self.current_archaeological_zones) > 0:
            primary_zone = max(self.current_archaeological_zones, key=lambda z: z.significance)
            recommendations.append(f"📍 Primary archaeological target: {primary_zone.zone_type} at {primary_zone.level:.2f}")
        
        checklist['execution_recommendations'] = recommendations
        
        return checklist

def main():
    """Demo/test the framework"""
    print("🚀 Macro-Archaeological Trading Framework")
    print("=" * 50)
    
    framework = MacroArchaeologicalFramework()
    
    # Test with sample date
    test_date = "2025-08-06"
    current_time = datetime(2025, 8, 6, 9, 55)  # 9:55 AM ET
    
    # Map pre-open anchors
    anchors = framework.map_pre_open_anchors(test_date)
    print(f"\n📊 Pre-open anchors mapped: {len(anchors['archaeological_zones'])} zones")
    
    # Test Gauntlet setup
    if anchors['archaeological_zones']:
        gauntlet = framework.create_gauntlet_setup(
            fvg_level=18500.0,
            ce_level=18495.0,
            sweep_level=18485.0,
            formation_time=current_time
        )
        print(f"\n🎯 Gauntlet setup confidence: {gauntlet.confidence_multiplier}x")
        print(f"   Archaeological proximity: {gauntlet.archaeological_proximity:.2f} points")
    
    # Generate execution checklist
    checklist = framework.generate_execution_checklist(current_time)
    print(f"\n✅ Execution checklist generated:")
    for recommendation in checklist['execution_recommendations']:
        print(f"   {recommendation}")

if __name__ == "__main__":
    main()