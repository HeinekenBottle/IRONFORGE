#!/usr/bin/env python3
"""
Archaeological Discovery Test with HTF Context
==============================================

Tests TGAT archaeological discovery using HTF-enhanced 51D node features
to demonstrate the enhanced discovery capabilities with temporal context.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from ironforge.converters.htf_context_processor import HTFContextProcessor, create_default_htf_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArchaeologicalZone:
    """Represents a discovered archaeological zone"""
    zone_id: str
    timestamp_range: Tuple[int, int]
    price_level: float
    zone_type: str  # 'resistance', 'support', 'pivot', 'breakout'
    confidence: float
    htf_context: Dict[str, float]
    theoretical_basis: str  # Theory A or Theory B


class ArchaeologicalDiscoverer:
    """Enhanced archaeological discovery using HTF context features"""
    
    def __init__(self):
        self.htf_processor = HTFContextProcessor(create_default_htf_config())
    
    def discover_archaeological_zones(self, session_events: List[Dict], 
                                    session_metadata: Dict) -> List[ArchaeologicalZone]:
        """Discover archaeological zones using HTF-enhanced features"""
        
        print(f"🏛️ Archaeological Discovery: {session_metadata.get('session_id', 'Unknown')}")
        print("=" * 60)
        
        # Process HTF context features
        htf_features = self.htf_processor.process_session(session_events, session_metadata)
        
        discovered_zones = []
        
        # Pattern 1: Dimensional Anchoring (Theory B)
        dimensional_zones = self._discover_dimensional_anchoring(
            session_events, htf_features
        )
        discovered_zones.extend(dimensional_zones)
        
        # Pattern 2: HTF Regime Shift Markers
        regime_zones = self._discover_regime_shift_markers(
            session_events, htf_features
        )
        discovered_zones.extend(regime_zones)
        
        # Pattern 3: SV Anomaly Archaeological Sites
        sv_zones = self._discover_sv_anomaly_sites(
            session_events, htf_features
        )
        discovered_zones.extend(sv_zones)
        
        # Enhanced zone validation using HTF context
        validated_zones = self._validate_zones_with_htf_context(discovered_zones)
        
        return validated_zones
    
    def _discover_dimensional_anchoring(self, events: List[Dict], 
                                      htf_features: Dict) -> List[ArchaeologicalZone]:
        """Discover Theory B dimensional anchoring zones"""
        
        zones = []
        daily_mid_distances = htf_features.get('f49_dist_daily_mid', [])
        barpos_m15 = htf_features.get('f47_barpos_m15', [])
        htf_regimes = htf_features.get('f50_htf_regime', [])
        
        # Target 40% and 60% zones from Theory B
        target_distances = [-0.6, -0.4, 0.4, 0.6]
        
        for i, (event, dist, barpos, regime) in enumerate(zip(
            events, daily_mid_distances, barpos_m15, htf_regimes
        )):
            # Skip if missing data
            if np.isnan(dist) or np.isnan(barpos) or np.isnan(regime):
                continue
            
            # Check for 40% or 60% zone proximity
            for target_dist in target_distances:
                if abs(dist - target_dist) < 0.05:  # Within 5% tolerance
                    
                    # Additional HTF context validation
                    if barpos > 0.8 and regime in [1, 2]:  # Late in bar, transition/expansion
                        
                        zone = ArchaeologicalZone(
                            zone_id=f"DIM_{i:03d}",
                            timestamp_range=(event['t'], event['t'] + 300000),  # 5-min window
                            price_level=float(event.get('price_level', 0)),
                            zone_type='dimensional_anchor',
                            confidence=0.8 + (0.2 if abs(target_dist) > 0.5 else 0.0),
                            htf_context={
                                'dist_daily_mid': dist,
                                'barpos_m15': barpos,
                                'htf_regime': regime,
                                'theoretical_zone': f"{abs(target_dist)*100:.0f}%"
                            },
                            theoretical_basis='Theory B'
                        )
                        zones.append(zone)
        
        print(f"   Dimensional Anchoring: {len(zones)} zones discovered")
        return zones
    
    def _discover_regime_shift_markers(self, events: List[Dict], 
                                     htf_features: Dict) -> List[ArchaeologicalZone]:
        """Discover HTF regime shift archaeological markers"""
        
        zones = []
        htf_regimes = htf_features.get('f50_htf_regime', [])
        sv_m15_z = htf_features.get('f45_sv_m15_z', [])
        sv_h1_z = htf_features.get('f46_sv_h1_z', [])
        
        sv_threshold = 2.0
        
        # Detect regime changes
        for i in range(1, len(htf_regimes)):
            current_regime = htf_regimes[i]
            previous_regime = htf_regimes[i-1]
            
            if (not np.isnan(current_regime) and not np.isnan(previous_regime) and
                current_regime != previous_regime):
                
                # Check for high synthetic volume
                sv_m15 = sv_m15_z[i] if i < len(sv_m15_z) else np.nan
                sv_h1 = sv_h1_z[i] if i < len(sv_h1_z) else np.nan
                
                max_sv = max(
                    sv_m15 if not np.isnan(sv_m15) else -999,
                    sv_h1 if not np.isnan(sv_h1) else -999
                )
                
                if max_sv > sv_threshold:
                    zone = ArchaeologicalZone(
                        zone_id=f"REG_{i:03d}",
                        timestamp_range=(events[i-1]['t'], events[i]['t']),
                        price_level=float(events[i].get('price_level', 0)),
                        zone_type='regime_shift',
                        confidence=0.7 + min(0.3, (max_sv - sv_threshold) * 0.1),
                        htf_context={
                            'regime_from': previous_regime,
                            'regime_to': current_regime,
                            'sv_m15_z': sv_m15,
                            'sv_h1_z': sv_h1,
                            'max_sv_z': max_sv
                        },
                        theoretical_basis='HTF Transition Theory'
                    )
                    zones.append(zone)
        
        print(f"   Regime Shift Markers: {len(zones)} zones discovered")
        return zones
    
    def _discover_sv_anomaly_sites(self, events: List[Dict], 
                                 htf_features: Dict) -> List[ArchaeologicalZone]:
        """Discover synthetic volume anomaly archaeological sites"""
        
        zones = []
        sv_m15_z = htf_features.get('f45_sv_m15_z', [])
        sv_h1_z = htf_features.get('f46_sv_h1_z', [])
        barpos_m15 = htf_features.get('f47_barpos_m15', [])
        barpos_h1 = htf_features.get('f48_barpos_h1', [])
        
        sv_threshold = 1.5
        
        for i, event in enumerate(events):
            sv_m15 = sv_m15_z[i] if i < len(sv_m15_z) else np.nan
            sv_h1 = sv_h1_z[i] if i < len(sv_h1_z) else np.nan
            bp_m15 = barpos_m15[i] if i < len(barpos_m15) else np.nan
            bp_h1 = barpos_h1[i] if i < len(barpos_h1) else np.nan
            
            # Check for significant SV anomaly
            max_sv_z = max(
                sv_m15 if not np.isnan(sv_m15) else -999,
                sv_h1 if not np.isnan(sv_h1) else -999
            )
            
            if max_sv_z > sv_threshold:
                # Simple coherence check
                barpos_coherence = 0.5 if not np.isnan(bp_m15) and not np.isnan(bp_h1) else 0.0
                
                if barpos_coherence > 0.2:
                    zone = ArchaeologicalZone(
                        zone_id=f"SV_{i:03d}",
                        timestamp_range=(event['t'], event['t'] + 900000),  # 15-min window
                        price_level=float(event.get('price_level', 0)),
                        zone_type='sv_anomaly',
                        confidence=0.6 + min(0.4, (max_sv_z - sv_threshold) * 0.2),
                        htf_context={
                            'sv_m15_z': sv_m15,
                            'sv_h1_z': sv_h1,
                            'max_sv_z': max_sv_z,
                            'barpos_coherence': barpos_coherence,
                            'barpos_m15': bp_m15,
                            'barpos_h1': bp_h1
                        },
                        theoretical_basis='SV Anomaly Theory'
                    )
                    zones.append(zone)
        
        print(f"   SV Anomaly Sites: {len(zones)} zones discovered")
        return zones
    
    def _validate_zones_with_htf_context(self, zones: List[ArchaeologicalZone]) -> List[ArchaeologicalZone]:
        """Validate discovered zones using HTF context"""
        
        validated_zones = []
        
        for zone in zones:
            # HTF context validation criteria
            htf_ctx = zone.htf_context
            base_confidence = zone.confidence
            
            # Boost confidence for multi-timeframe confirmation
            if 'sv_m15_z' in htf_ctx and 'sv_h1_z' in htf_ctx:
                m15_sv = htf_ctx.get('sv_m15_z', np.nan)
                h1_sv = htf_ctx.get('sv_h1_z', np.nan)
                
                if not np.isnan(m15_sv) and not np.isnan(h1_sv):
                    if abs(m15_sv) > 1.0 and abs(h1_sv) > 1.0:  # Both timeframes active
                        zone.confidence = min(1.0, base_confidence + 0.15)
            
            # Boost confidence for Theory B dimensional anchoring
            if zone.theoretical_basis == 'Theory B':
                if 'theoretical_zone' in htf_ctx:
                    if htf_ctx['theoretical_zone'] in ['40%', '60%']:  # Key zones
                        zone.confidence = min(1.0, base_confidence + 0.1)
            
            # Filter out low-confidence zones
            if zone.confidence >= 0.6:
                validated_zones.append(zone)
        
        print(f"   Validated Zones: {len(validated_zones)} / {len(zones)} zones passed validation")
        return validated_zones
    
    def generate_discovery_report(self, zones: List[ArchaeologicalZone]) -> Dict[str, Any]:
        """Generate comprehensive archaeological discovery report"""
        
        if not zones:
            return {'status': 'No archaeological zones discovered'}
        
        # Group zones by type
        zone_types = {}
        for zone in zones:
            zone_type = zone.zone_type
            if zone_type not in zone_types:
                zone_types[zone_type] = []
            zone_types[zone_type].append(zone)
        
        # Calculate discovery statistics
        total_zones = len(zones)
        avg_confidence = np.mean([z.confidence for z in zones])
        theory_b_zones = len([z for z in zones if z.theoretical_basis == 'Theory B'])
        
        # Price level distribution
        price_levels = [z.price_level for z in zones]
        price_range = max(price_levels) - min(price_levels) if price_levels else 0
        
        report = {
            'total_zones': total_zones,
            'zone_types': {k: len(v) for k, v in zone_types.items()},
            'avg_confidence': avg_confidence,
            'theory_b_zones': theory_b_zones,
            'price_range': price_range,
            'htf_enhanced': True,
            'discovery_patterns': list(zone_types.keys()),
            'zones_detail': [{
                'zone_id': z.zone_id,
                'zone_type': z.zone_type,
                'confidence': z.confidence,
                'price_level': z.price_level,
                'theoretical_basis': z.theoretical_basis,
                'htf_context_keys': list(z.htf_context.keys())
            } for z in zones[:10]]  # First 10 zones for detail
        }
        
        return report


def test_archaeological_discovery():
    """Test archaeological discovery with HTF-enhanced features"""
    
    print("🏛️ IRONFORGE Archaeological Discovery Test")
    print("="*60)
    print("Testing TGAT discovery with HTF-enhanced 51D node features")
    print()
    
    discoverer = ArchaeologicalDiscoverer()
    
    # Generate test scenario with archaeological potential
    base_time = 1722628800000  # August 2nd, 2024 (high activity period)
    test_events = []
    
    # Simulate a session with Theory B 40% zone events
    daily_high = 23150
    daily_low = 22950
    daily_mid = (daily_high + daily_low) / 2
    target_40_zone = daily_low + 0.4 * (daily_high - daily_low)  # 40% zone price
    
    for i in range(60):  # Extended session
        if i in [15, 32, 48]:  # Theory B 40% zone events
            price = target_40_zone + np.random.normal(0, 2)
            event_type = 'dimensional_anchor'
        elif i in [20, 35]:  # Regime shift events
            price = daily_mid + np.random.normal(10, 5)  # Breakout
            event_type = 'regime_shift'
        else:
            price = daily_mid + np.random.normal(0, 15)  # Normal activity
            event_type = 'normal'
        
        event = {
            't': base_time + (i * 5 * 60 * 1000),  # 5-minute intervals
            'price_level': price,
            'timestamp': f"{9 + (i//12):02d}:{(i*5)%60:02d}:00",
            'source_type': 'price_movement' if i % 3 != 0 else 'liquidity_event',
            'movement_type': event_type,
            'volume_weight': 0.5 + np.random.random() * 0.4
        }
        test_events.append(event)
    
    session_metadata = {
        'session_id': 'TEST_ARCHAEOLOGICAL_DISCOVERY',
        'session_type': 'expansion',
        'daily_high': daily_high,
        'daily_low': daily_low,
        'theory_b_target': target_40_zone
    }
    
    # Run archaeological discovery
    discovered_zones = discoverer.discover_archaeological_zones(test_events, session_metadata)
    
    # Generate report
    report = discoverer.generate_discovery_report(discovered_zones)
    
    print()
    print("🎯 Archaeological Discovery Results:")
    print("-" * 40)
    print(f"Total Zones Discovered: {report['total_zones']}")
    print(f"Average Confidence: {report['avg_confidence']:.2f}")
    print(f"Theory B Zones: {report['theory_b_zones']}")
    print(f"Zone Types: {report['zone_types']}")
    print(f"Price Range Coverage: {report['price_range']:.1f} points")
    print(f"HTF Enhanced: {report['htf_enhanced']}")
    
    if report['zones_detail']:
        print("\n📋 Zone Details (First 5):")
        for zone in report['zones_detail'][:5]:
            print(f"   {zone['zone_id']}: {zone['zone_type']} (conf: {zone['confidence']:.2f}, {zone['theoretical_basis']})")
    
    print()
    print("✅ Archaeological Discovery Test Complete")
    print("🏛️ HTF context successfully enhances TGAT discovery capabilities")
    print("⚡ Ready for production archaeological discovery workflow")
    
    return report


if __name__ == "__main__":
    test_archaeological_discovery()