#!/usr/bin/env python3
"""
AM Trading Interface: Enhanced Scalping with Macro-Archaeological Integration
Real-time interface for ICT Gauntlet + IRONFORGE temporal intelligence
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from macro_archaeological_framework import MacroArchaeologicalFramework, ArchaeologicalZone, GauntletSetup

@dataclass
class TradeSignal:
    """Enhanced trade signal with archaeological context"""
    timestamp: datetime
    signal_type: str  # 'gauntlet_entry', 'archaeological_confluence', 'orbital_exit'
    entry_price: float
    direction: str  # 'long', 'short'
    confidence: float  # 1x or 2x
    stop_loss: float
    profit_targets: List[Dict[str, Any]]
    archaeological_context: Dict[str, Any]
    macro_context: Dict[str, Any]
    risk_reward: float
    notes: str

@dataclass
class PositionManager:
    """Position management with orbital and archaeological awareness"""
    entry_time: datetime
    entry_price: float
    direction: str
    position_size: float
    current_targets: List[Dict[str, Any]]
    archaeological_stops: List[float]
    macro_phase_context: str
    scaling_completed: Dict[str, bool]

class AMTradingInterface:
    """
    Enhanced AM trading interface integrating:
    - ICT Gauntlet methodology
    - IRONFORGE archaeological zones
    - Macro window orbital timing
    - Risk management with Theory B precision
    """
    
    def __init__(self):
        self.framework = MacroArchaeologicalFramework()
        self.active_positions: List[PositionManager] = []
        self.trade_signals: List[TradeSignal] = []
        self.session_stats = {
            'trades_taken': 0,
            'archaeological_confluences': 0,
            'macro_window_hits': 0,
            'avg_confidence': 0.0
        }
        
        # Trading parameters
        self.PRIMARY_WINDOW = (time(9, 35), time(9, 58))  # Cash open digestion
        self.MACRO_WINDOW = (time(9, 50), time(10, 10))   # Macro follow-through/fade
        self.VARIANT_WINDOW = (time(10, 20), time(10, 40))  # Half-hour variant
        
        # Risk management
        self.MAX_POSITIONS = 2
        self.BASE_POSITION_SIZE = 1.0
        self.ARCHAEOLOGICAL_SIZE_MULTIPLIER = 1.5
        
    def initialize_session(self, trading_date: str) -> Dict[str, Any]:
        """
        Initialize trading session with pre-open archaeological mapping
        08:45–09:29 ET preparation phase
        """
        print(f"🚀 Initializing AM Session: {trading_date}")
        print("=" * 50)
        
        # Map pre-open anchors
        anchors = self.framework.map_pre_open_anchors(trading_date)
        
        session_init = {
            'date': trading_date,
            'initialization_time': datetime.now(),
            'archaeological_zones': anchors['archaeological_zones'],
            'confluence_levels': anchors['confluence_levels'],
            'session_ready': len(anchors['archaeological_zones']) > 0,
            'htf_bias': None,  # Would be set from external analysis
            'risk_on': True
        }
        
        # Display pre-open setup
        self._display_preopen_setup(session_init)
        
        return session_init
    
    def _display_preopen_setup(self, session_init: Dict[str, Any]):
        """Display pre-open archaeological setup"""
        print(f"\n📍 Archaeological Zones Mapped: {len(session_init['archaeological_zones'])}")
        
        for zone in session_init['archaeological_zones']:
            precision_status = "🎯 HIGH" if zone.precision_score <= 2.0 else "⚠️  MED" if zone.precision_score <= 5.0 else "❌ LOW"
            print(f"  {zone.zone_type}: {zone.level:.2f} | Precision: {zone.precision_score:.2f} | {precision_status}")
        
        print(f"\n🎯 Primary Confluence Levels:")
        for level in session_init['confluence_levels'][:3]:  # Top 3
            print(f"  Level: {level['level']:.2f} | Type: {level['type']} | Significance: {level['significance']:.1f}")
        
        print(f"\n⏰ Trading Windows:")
        print(f"  Primary: {self.PRIMARY_WINDOW[0].strftime('%H:%M')}-{self.PRIMARY_WINDOW[1].strftime('%H:%M')} ET")
        print(f"  Macro: {self.MACRO_WINDOW[0].strftime('%H:%M')}-{self.MACRO_WINDOW[1].strftime('%H:%M')} ET")
        print(f"  Variant: {self.VARIANT_WINDOW[0].strftime('%H:%M')}-{self.VARIANT_WINDOW[1].strftime('%H:%M')} ET")
    
    def analyze_gauntlet_setup(self, price_data: Dict[str, float], 
                              current_time: datetime) -> Optional[TradeSignal]:
        """
        Analyze potential Gauntlet setup with archaeological enhancement
        
        Args:
            price_data: {'current': float, 'fvg_low': float, 'fvg_high': float, 'ce': float, 'sweep_level': float}
            current_time: Current market time
        """
        
        # Check if in valid trading window
        current_time_only = current_time.time()
        in_primary = self.PRIMARY_WINDOW[0] <= current_time_only <= self.PRIMARY_WINDOW[1]
        in_macro = self.MACRO_WINDOW[0] <= current_time_only <= self.MACRO_WINDOW[1]
        in_variant = self.VARIANT_WINDOW[0] <= current_time_only <= self.VARIANT_WINDOW[1]
        
        if not (in_primary or in_macro or in_variant):
            return None
        
        # Create Gauntlet setup
        gauntlet = self.framework.create_gauntlet_setup(
            fvg_level=price_data['fvg_low'],
            ce_level=price_data['ce'],
            sweep_level=price_data['sweep_level'],
            formation_time=current_time
        )
        
        # Analyze macro orbital context
        macro_window = self.framework.macro_windows['morning_primary']
        orbital_analysis = self.framework.analyze_macro_orbital_phase(current_time, macro_window)
        
        # Determine entry viability
        entry_viable = self._assess_entry_viability(gauntlet, orbital_analysis, price_data)
        
        if not entry_viable:
            return None
        
        # Calculate enhanced risk management
        direction = 'long'  # Simplified - would determine from context
        risk_levels = self.framework.calculate_archaeological_risk_levels(
            price_data['ce'], direction
        )
        
        # Create trade signal
        signal = TradeSignal(
            timestamp=current_time,
            signal_type='gauntlet_archaeological_entry',
            entry_price=price_data['ce'],
            direction=direction,
            confidence=gauntlet.confidence_multiplier,
            stop_loss=price_data['sweep_level'],
            profit_targets=risk_levels['profit_targets'],
            archaeological_context={
                'convergence_detected': gauntlet.archaeological_proximity <= 7.55,
                'nearest_zone_distance': gauntlet.archaeological_proximity,
                'zone_count': len(self.framework.current_archaeological_zones)
            },
            macro_context={
                'window_type': 'macro' if in_macro else 'primary' if in_primary else 'variant',
                'orbital_phase': orbital_analysis.get('phase', 'unknown'),
                'phase_progress': orbital_analysis.get('phase_context', {}).get('phase_progress', 0)
            },
            risk_reward=self._calculate_risk_reward(price_data['ce'], price_data['sweep_level'], risk_levels['profit_targets']),
            notes=f"Gauntlet CE: {price_data['ce']:.2f}, Archaeological proximity: {gauntlet.archaeological_proximity:.1f}pts"
        )
        
        return signal
    
    def _assess_entry_viability(self, gauntlet: GauntletSetup, orbital_analysis: Dict, 
                               price_data: Dict[str, float]) -> bool:
        """Assess if Gauntlet entry is viable given current context"""
        
        # Basic Gauntlet criteria
        if gauntlet.archaeological_proximity > 50:  # Too far from any archaeological zone
            return False
        
        # Orbital phase criteria
        orbital_phase = orbital_analysis.get('phase', 'unknown')
        if orbital_phase in ['completion']:  # Avoid entries during completion phase
            return False
        
        # Enhanced criteria for archaeological confluence
        if gauntlet.confidence_multiplier > 1.0:
            return True  # High confidence due to archaeological alignment
        
        # Standard criteria for non-archaeological setups
        if orbital_phase in ['entry', 'extension']:
            return True
        
        return False
    
    def _calculate_risk_reward(self, entry: float, stop: float, targets: List[Dict]) -> float:
        """Calculate risk-reward ratio for first target"""
        risk = abs(entry - stop)
        if not targets or risk == 0:
            return 0.0
        
        first_target_reward = abs(targets[0]['price'] - entry)
        return first_target_reward / risk
    
    def manage_position(self, position: PositionManager, current_price: float, 
                       current_time: datetime) -> Dict[str, Any]:
        """
        Enhanced position management with orbital and archaeological awareness
        """
        management_actions = {
            'scale_out': [],
            'stop_adjustment': None,
            'full_exit': False,
            'trail_stop': None,
            'notes': []
        }
        
        # Check orbital phase for scaling decisions
        macro_window = self.framework.macro_windows['morning_primary']
        orbital_analysis = self.framework.analyze_macro_orbital_phase(current_time, macro_window)
        current_phase = orbital_analysis.get('phase', 'unknown')
        
        # Time-based management (ICT 10-12 minute rule)
        time_in_trade = (current_time - position.entry_time).total_seconds() / 60
        
        # Scale out logic
        for i, target in enumerate(position.current_targets):
            target_hit = (
                (position.direction == 'long' and current_price >= target['price']) or
                (position.direction == 'short' and current_price <= target['price'])
            )
            
            if target_hit and not position.scaling_completed.get(f'target_{i}', False):
                scale_percentage = 0.5 if i == 0 else 0.3  # 50% at first target, 30% at second
                management_actions['scale_out'].append({
                    'target_level': target['price'],
                    'scale_percentage': scale_percentage,
                    'archaeological_confluence': target.get('archaeological_confluence', False)
                })
                position.scaling_completed[f'target_{i}'] = True
                management_actions['notes'].append(f"Scaled {scale_percentage*100}% at target {i+1}")
        
        # Orbital phase-based management
        if current_phase == 'completion':
            # Consider scaling out during completion phase
            if not position.scaling_completed.get('orbital_completion', False):
                management_actions['scale_out'].append({
                    'target_level': current_price,
                    'scale_percentage': 0.3,
                    'reason': 'orbital_completion_phase'
                })
                position.scaling_completed['orbital_completion'] = True
                management_actions['notes'].append("Orbital completion phase scaling")
        
        # Archaeological zone-based exits
        for zone in self.framework.current_archaeological_zones:
            zone_proximity = abs(current_price - zone.level)
            if zone_proximity <= 7.55:  # Theory B precision threshold
                management_actions['notes'].append(f"Near {zone.zone_type} (±{zone_proximity:.1f}pts)")
                # Consider partial exit due to archaeological significance
                if zone.significance > 0.8:
                    management_actions['scale_out'].append({
                        'target_level': zone.level,
                        'scale_percentage': 0.4,
                        'reason': 'high_significance_archaeological_zone'
                    })
        
        # Time-based exit (ICT 10-12 minute rule)
        if time_in_trade >= 12 and len(position.scaling_completed) == 0:
            management_actions['full_exit'] = True
            management_actions['notes'].append("12-minute time limit reached, no expansion")
        elif time_in_trade >= 10 and current_phase == 'completion':
            management_actions['full_exit'] = True
            management_actions['notes'].append("10+ minutes in completion phase")
        
        return management_actions
    
    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate end-of-session summary with archaeological insights"""
        
        archaeological_hits = sum(1 for signal in self.trade_signals 
                                 if signal.archaeological_context.get('convergence_detected', False))
        
        macro_window_trades = sum(1 for signal in self.trade_signals 
                                 if signal.macro_context.get('window_type') == 'macro')
        
        avg_confidence = np.mean([signal.confidence for signal in self.trade_signals]) if self.trade_signals else 0
        
        summary = {
            'session_date': datetime.now().strftime('%Y-%m-%d'),
            'total_signals': len(self.trade_signals),
            'archaeological_confluences': archaeological_hits,
            'macro_window_trades': macro_window_trades,
            'average_confidence': avg_confidence,
            'confidence_distribution': {
                '1x': sum(1 for s in self.trade_signals if s.confidence == 1.0),
                '2x': sum(1 for s in self.trade_signals if s.confidence == 2.0)
            },
            'orbital_phase_distribution': {},
            'key_insights': []
        }
        
        # Orbital phase analysis
        phase_counts = {}
        for signal in self.trade_signals:
            phase = signal.macro_context.get('orbital_phase', 'unknown')
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        summary['orbital_phase_distribution'] = phase_counts
        
        # Key insights
        insights = []
        if archaeological_hits > 0:
            insights.append(f"Archaeological confluence enhanced {archaeological_hits} setups (+{(2.0-1.0)*100:.0f}% confidence)")
        
        if macro_window_trades > len(self.trade_signals) * 0.5:
            insights.append("Majority of trades occurred during macro windows (high probability timing)")
        
        most_common_phase = max(phase_counts, key=phase_counts.get) if phase_counts else 'none'
        if most_common_phase != 'none':
            insights.append(f"Most productive orbital phase: {most_common_phase}")
        
        summary['key_insights'] = insights
        
        return summary

def demo_trading_session():
    """Demo/test the AM trading interface"""
    print("🎯 AM Trading Interface Demo")
    print("=" * 40)
    
    # Initialize interface
    interface = AMTradingInterface()
    
    # Initialize session
    session_init = interface.initialize_session("2025-08-07")
    
    if not session_init['session_ready']:
        print("❌ Session not ready - no archaeological zones mapped")
        return
    
    # Simulate Gauntlet setup analysis
    print(f"\n🔍 Analyzing Gauntlet Setup at 9:52 AM ET...")
    
    mock_price_data = {
        'current': 23400.0,
        'fvg_low': 23395.0,
        'fvg_high': 23405.0,
        'ce': 23400.0,
        'sweep_level': 23385.0
    }
    
    signal = interface.analyze_gauntlet_setup(
        mock_price_data, 
        datetime(2025, 8, 7, 9, 52)
    )
    
    if signal:
        print(f"✅ Trade Signal Generated:")
        print(f"   Entry: {signal.entry_price:.2f}")
        print(f"   Direction: {signal.direction}")
        print(f"   Confidence: {signal.confidence}x")
        print(f"   Archaeological Context: {signal.archaeological_context}")
        print(f"   Macro Context: {signal.macro_context}")
        print(f"   R:R: {signal.risk_reward:.2f}")
        
        interface.trade_signals.append(signal)
    else:
        print("❌ No viable signal generated")
    
    # Generate session summary
    print(f"\n📊 Session Summary:")
    summary = interface.generate_session_summary()
    for key, value in summary.items():
        if key not in ['orbital_phase_distribution', 'confidence_distribution']:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    demo_trading_session()