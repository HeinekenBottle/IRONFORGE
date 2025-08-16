#!/usr/bin/env python3
"""
Enhanced Session Data Adapter for IRONFORGE Archaeological Discovery
====================================================================

Bridges the gap between enhanced session data format and archaeological
event detector expectations, enabling full event discovery from existing
enhanced sessions.

Key Features:
- 60 event type mappings across 7 archaeological families
- Multi-source magnitude calculation with archaeological zone boosting
- Preserves Theory B dimensional destiny (40% zone relationships)
- Non-invasive integration with existing archaeology system
- Full compatibility with TGAT neural network discovery

Author: IRONFORGE Archaeological Discovery System
Date: August 15, 2025
"""

from typing import Dict, List, Any, Optional
import json
import math
from datetime import datetime
from pathlib import Path

class EnhancedSessionAdapter:
    """
    Adapter to convert enhanced session format to archaeology-compatible format
    """
    
    # Enhanced session event types → Archaeological event types
    # Comprehensive mapping covering 60+ event types across 7 families
    EVENT_TYPE_MAPPING = {
        # FVG Family (Fair Value Gap events)
        'pm_fpfvg_formation_premium': 'fvg_formation',
        'pm_fpfvg_formation_discount': 'fvg_formation',
        'pm_fpfvg_rebalance': 'fvg_rebalance',
        'fpfvg_formation': 'fvg_formation',
        'fpfvg_redelivery': 'fvg_redelivery',
        'fpfvg_continuation': 'fvg_continuation',
        'FVG_first_presented': 'fvg_formation',
        'FVG_redelivery': 'fvg_redelivery',
        'FVG_continuation': 'fvg_continuation',
        'fvg_formation': 'fvg_formation',
        'fvg_rebalance': 'fvg_rebalance',
        'fvg_completion': 'fvg_completion',
        
        # Liquidity Sweep Family
        'price_gap': 'liquidity_sweep',
        'liquidity_sweep': 'liquidity_sweep',
        'double_sweep': 'sweep_sequence',
        'sell_side_sweep': 'liquidity_sweep',
        'buy_side_sweep': 'liquidity_sweep',
        'session_low_sweep': 'liquidity_sweep',
        'session_high_sweep': 'liquidity_sweep',
        'sweep_completion': 'liquidity_sweep',
        
        # Expansion Family
        'expansion_start_higher': 'expansion_phase',
        'expansion_start_lower': 'expansion_phase',
        'expansion_low': 'expansion_event',
        'expansion_high': 'expansion_event',
        'expansion_continuation': 'expansion_phase',
        'expansion_acceleration': 'expansion_event',
        'expansion_completion': 'expansion_event',
        'expansion_reversal': 'expansion_event',
        'expansion_phase_1': 'expansion_phase',
        'expansion_phase_2': 'expansion_phase',
        'expansion_phase_3': 'expansion_phase',
        'expansion_climax': 'expansion_event',
        
        # Consolidation Family
        'consolidation_start_high': 'consolidation_phase',
        'consolidation_start_low': 'consolidation_phase',
        'consolidation_high': 'consolidation_event',
        'consolidation_low': 'consolidation_event',
        'consolidation_continuation': 'consolidation_phase',
        'consolidation_compression': 'consolidation_event',
        'consolidation_break': 'consolidation_event',
        'consolidation_end': 'consolidation_event',
        
        # Retracement Family
        'retracement_start': 'retracement_phase',
        'retracement_23_6': 'retracement_level',
        'retracement_38_2': 'retracement_level',
        'retracement_50_0': 'retracement_level',
        'retracement_61_8': 'retracement_level',
        'retracement_completion': 'retracement_event',
        
        # Structural Family
        'momentum_shift': 'regime_shift',
        'volume_spike': 'liquidity_event',
        'volatility_burst': 'volatility_event',
        'trend_continuation': 'structural_trend',
        'trend_reversal': 'structural_reversal',
        'structural_break': 'structural_event',
        
        # Session Markers
        'session_open': 'session_marker',
        'session_close': 'session_marker',
        'session_high': 'session_extremes',
        'session_low': 'session_extremes',
        'open': 'session_marker',
        'close': 'session_marker',
        'high': 'session_extremes',
        'low': 'session_extremes',
        
        # Archaeological Zone Events (Theory B Dimensional Destiny)
        'zone_20_percent': 'archaeological_zone',
        'zone_40_percent': 'archaeological_zone',
        'zone_60_percent': 'archaeological_zone', 
        'zone_80_percent': 'archaeological_zone'
    }
    
    # Archaeological zone boost factors (Theory B preservation)
    ARCHAEOLOGICAL_ZONE_BOOSTS = {
        'zone_40_percent': 0.9,  # Highest boost for dimensional destiny
        'zone_20_percent': 0.7,
        'zone_60_percent': 0.7,
        'zone_80_percent': 0.8
    }
    
    def __init__(self):
        """Initialize the adapter with comprehensive tracking"""
        self.stats = {
            'sessions_processed': 0,
            'events_extracted': 0,
            'price_movements_converted': 0,
            'liquidity_events_converted': 0,
            'fpfvg_events_converted': 0,
            'archaeological_zones_detected': 0,
            'magnitude_calculation_methods': {
                'intensity_direct': 0,
                'price_momentum': 0,
                'range_position': 0,
                'energy_density': 0,
                'composite': 0
            },
            'unmapped_event_types': set(),
            'event_family_distribution': {
                'fvg_family': 0,
                'liquidity_family': 0,
                'expansion_family': 0,
                'consolidation_family': 0,
                'retracement_family': 0,
                'structural_family': 0,
                'session_markers': 0,
                'archaeological_zones': 0
            }
        }
    
    def adapt_enhanced_session(self, session_data: Dict) -> Dict:
        """
        Convert enhanced session format to archaeology-compatible format
        
        Args:
            session_data: Enhanced session in original format
            
        Returns:
            Dict in archaeological format with 'events' and 'enhanced_features'
        """
        archaeological_events = []
        
        # Extract events from price_movements
        price_events = self._extract_price_movement_events(
            session_data.get('price_movements', [])
        )
        archaeological_events.extend(price_events)
        
        # Extract events from session_liquidity_events
        liquidity_events = self._extract_liquidity_events(
            session_data.get('session_liquidity_events', []), session_data
        )
        archaeological_events.extend(liquidity_events)
        
        # Extract events from session_fpfvg (if present)
        fpfvg_events = self._extract_fpfvg_events(
            session_data.get('session_fpfvg', {})
        )
        archaeological_events.extend(fpfvg_events)
        
        # Create enhanced_features from session metadata
        enhanced_features = self._create_enhanced_features(session_data)
        
        # Detect archaeological zones
        archaeological_zone_events = self._detect_archaeological_zones(
            archaeological_events, session_data
        )
        archaeological_events.extend(archaeological_zone_events)
        
        # Update statistics
        self._update_statistics(archaeological_events, price_events, liquidity_events, fpfvg_events)
        
        return {
            'events': archaeological_events,
            'enhanced_features': enhanced_features,
            'session_metadata': session_data.get('session_metadata', {}),
            'relativity_stats': session_data.get('relativity_stats', {}),
            'original_format': 'enhanced_session',
            'adapter_version': '2.0.0',
            'adaptation_timestamp': datetime.now().isoformat()
        }
    
    def _extract_price_movement_events(self, price_movements: List[Dict]) -> List[Dict]:
        """Extract archaeological events from price_movements array"""
        events = []
        
        for movement in price_movements:
            # Map the event type
            original_type = movement.get('movement_type', '')
            archaeological_type = self.EVENT_TYPE_MAPPING.get(original_type, original_type)
            
            # Track unmapped types
            if original_type and original_type not in self.EVENT_TYPE_MAPPING:
                self.stats['unmapped_event_types'].add(original_type)
            
            # Calculate magnitude using multiple strategies
            magnitude = self._calculate_magnitude_from_movement(movement, original_type)
            
            # Determine range position and archaeological significance
            range_position = movement.get('range_position', 0.5)
            archaeological_significance = self._calculate_archaeological_significance(
                range_position, original_type, magnitude
            )
            
            # Create comprehensive archaeological event
            event = {
                'type': archaeological_type,
                'original_type': original_type,
                'magnitude': magnitude,
                'value': movement.get('normalized_price', 0.5),
                'timestamp': movement.get('timestamp'),
                'price_level': movement.get('price_level'),
                'range_position': range_position,
                'duration': self._calculate_duration(movement),
                'htf_confluence': movement.get('htf_confluence_strength', 0.5),
                
                # Enhanced session specific fields
                'pct_from_open': movement.get('pct_from_open', 0.0),
                'pct_from_high': movement.get('pct_from_high', 0.0),
                'pct_from_low': movement.get('pct_from_low', 0.0),
                'price_momentum': movement.get('price_momentum', 0.0),
                'energy_density': movement.get('energy_density', 0.5),
                'time_since_session_open': movement.get('time_since_session_open', 0),
                'normalized_time': movement.get('normalized_time', 0.0),
                'absolute_price': movement.get('absolute_price', movement.get('price_level', 0)),
                
                # Archaeological analysis
                'archaeological_significance': archaeological_significance,
                'event_family': self._determine_event_family(archaeological_type),
                'structural_role': self._determine_structural_role(original_type, range_position),
                'dimensional_relationship': self._calculate_dimensional_relationship(range_position),
                
                # Source tracking
                'source': 'price_movements',
                'enhanced_session_data': True,
                'relativity_enhanced': movement.get('relativity_enhanced', False)
            }
            
            events.append(event)
        
        return events
    
    def _resolve_target_level_to_price(self, target_level: str, session_data: Dict) -> float:
        """Resolve target_level string to actual price value"""
        if not target_level:
            return None
            
        # Get relativity stats for session extremes
        relativity_stats = session_data.get('relativity_stats', {})
        
        # Direct mapping for exact matches
        level_mapping = {
            'session_open': relativity_stats.get('session_open'),
            'session_close': relativity_stats.get('session_close'),
            'session_high': relativity_stats.get('session_high'),
            'session_low': relativity_stats.get('session_low'),
            'session_session_high': relativity_stats.get('session_high'),
            'session_session_low': relativity_stats.get('session_low'),
        }
        
        # Check for exact matches first
        exact_match = level_mapping.get(target_level)
        if exact_match is not None:
            return exact_match
            
        # Check for partial matches in target_level string
        target_lower = target_level.lower()
        if 'session_high' in target_lower or 'high' in target_lower:
            return relativity_stats.get('session_high')
        elif 'session_low' in target_lower or 'low' in target_lower:
            return relativity_stats.get('session_low')
        elif 'session_open' in target_lower or 'open' in target_lower:
            return relativity_stats.get('session_open')
        elif 'session_close' in target_lower or 'close' in target_lower:
            return relativity_stats.get('session_close')
            
        # Return None if no mapping found (will be handled gracefully)
        return None
    
    def _extract_liquidity_events(self, liquidity_events: List[Dict], session_data: Dict = None) -> List[Dict]:
        """Extract archaeological events from session_liquidity_events array"""
        events = []
        
        for liq_event in liquidity_events:
            # Map the event type
            original_type = liq_event.get('event_type', '')
            archaeological_type = self.EVENT_TYPE_MAPPING.get(original_type, original_type)
            
            # Track unmapped types
            if original_type and original_type not in self.EVENT_TYPE_MAPPING:
                self.stats['unmapped_event_types'].add(original_type)
            
            # Use intensity as primary magnitude source
            magnitude = liq_event.get('intensity', 0.5)
            
            # Resolve price level from target_level if price_level is not available
            price_level = liq_event.get('price_level')
            if price_level is None and session_data:
                target_level = liq_event.get('target_level')
                price_level = self._resolve_target_level_to_price(target_level, session_data)
            
            # Calculate archaeological significance
            archaeological_significance = self._calculate_archaeological_significance(
                0.5, original_type, magnitude  # Default range position for liquidity events
            )
            
            # Create comprehensive archaeological event
            event = {
                'type': archaeological_type,
                'original_type': original_type,
                'magnitude': magnitude,
                'value': magnitude,  # Use intensity as value
                'timestamp': liq_event.get('timestamp'),
                'price_level': price_level,
                'absolute_price': price_level,  # Use same value for absolute_price
                'duration': liq_event.get('impact_duration', 60),
                'range_position': liq_event.get('range_position', 0.5),
                'htf_confluence': liq_event.get('htf_confluence', 0.5),
                
                # Liquidity specific fields
                'intensity': magnitude,
                'impact_duration': liq_event.get('impact_duration', 60),
                'liquidity_type': liq_event.get('liquidity_type', 'unknown'),
                'sweep_direction': liq_event.get('sweep_direction', 'unknown'),
                'target_level': liq_event.get('target_level', 'unknown'),
                
                # Archaeological analysis
                'archaeological_significance': archaeological_significance,
                'event_family': self._determine_event_family(archaeological_type),
                'structural_role': self._determine_structural_role(original_type, 0.5),
                'liquidity_archetype': self._determine_liquidity_archetype(original_type),
                
                # Source tracking
                'source': 'session_liquidity_events',
                'enhanced_session_data': True
            }
            
            events.append(event)
        
        return events
    
    def _extract_fpfvg_events(self, fpfvg_data: Dict) -> List[Dict]:
        """Extract archaeological events from session_fpfvg data"""
        events = []
        
        if not fpfvg_data or not fpfvg_data.get('fpfvg_present', False):
            return events
        
        fpfvg_formation = fpfvg_data.get('fpfvg_formation', {})
        
        # Create FVG formation event
        if fpfvg_formation:
            formation_event = {
                'type': 'fvg_formation',
                'original_type': 'fpfvg_formation',
                'magnitude': 0.8,  # High significance for FPFVG
                'value': 0.8,
                'timestamp': fpfvg_formation.get('formation_time'),
                'price_level': fpfvg_formation.get('premium_high', 0),
                'duration': 300,  # 5 minutes default for FVG formation
                'range_position': 0.5,  # Default positioning
                'htf_confluence': 0.7,  # FVGs typically have HTF significance
                
                # FVG specific fields
                'premium_high': fpfvg_formation.get('premium_high'),
                'discount_low': fpfvg_formation.get('discount_low'),
                'gap_size': fpfvg_formation.get('gap_size', 0),
                'fvg_type': 'first_presented_fvg',
                
                # Archaeological analysis
                'archaeological_significance': 0.85,  # High significance
                'event_family': 'fvg_family',
                'structural_role': 'liquidity_anchor',
                
                # Source tracking
                'source': 'session_fpfvg',
                'enhanced_session_data': True
            }
            events.append(formation_event)
        
        # Extract interaction events
        interactions = fpfvg_formation.get('interactions', [])
        for interaction in interactions:
            interaction_event = {
                'type': 'fvg_rebalance',
                'original_type': 'fpfvg_rebalance',
                'magnitude': 0.6,
                'value': 0.6,
                'timestamp': interaction.get('interaction_time'),
                'price_level': interaction.get('price_level', 0),
                'duration': 120,  # 2 minutes for rebalance
                'range_position': 0.5,
                'htf_confluence': 0.6,
                
                # Interaction specific fields
                'interaction_type': interaction.get('interaction_type'),
                'interaction_context': interaction.get('interaction_context'),
                'rebalance_price': interaction.get('price_level'),
                
                # Archaeological analysis
                'archaeological_significance': 0.7,
                'event_family': 'fvg_family',
                'structural_role': 'rebalance_mechanism',
                
                # Source tracking
                'source': 'session_fpfvg_interactions',
                'enhanced_session_data': True
            }
            events.append(interaction_event)
        
        return events
    
    def _detect_archaeological_zones(self, events: List[Dict], session_data: Dict) -> List[Dict]:
        """
        Detect archaeological zones based on Theory B dimensional destiny
        40% zone represents dimensional relationship to FINAL session range
        """
        zone_events = []
        
        # Get session range information
        relativity_stats = session_data.get('relativity_stats', {})
        session_high = relativity_stats.get('session_high', 0)
        session_low = relativity_stats.get('session_low', 0)
        session_range = relativity_stats.get('session_range', 0)
        
        if session_range is None or session_range <= 0:
            return zone_events
        
        # Define archaeological zone levels (Theory B)
        zone_levels = {
            'zone_20_percent': session_low + (session_range * 0.20),
            'zone_40_percent': session_low + (session_range * 0.40),  # Dimensional destiny
            'zone_60_percent': session_low + (session_range * 0.60),
            'zone_80_percent': session_low + (session_range * 0.80)
        }
        
        # Check events for archaeological zone proximity
        for event in events:
            price_level = event.get('price_level')
            if price_level is None or price_level <= 0:
                continue
            
            # Find closest archaeological zone
            closest_zone = None
            min_distance = float('inf')
            
            for zone_name, zone_price in zone_levels.items():
                distance = abs(price_level - zone_price)
                if distance < min_distance and distance < (session_range * 0.05):  # Within 5% of range
                    min_distance = distance
                    closest_zone = zone_name
            
            # Create archaeological zone event if close match found
            if closest_zone:
                zone_event = {
                    'type': 'archaeological_zone',
                    'original_type': closest_zone,
                    'magnitude': 0.8 + self.ARCHAEOLOGICAL_ZONE_BOOSTS.get(closest_zone, 0.0),
                    'value': float(closest_zone.split('_')[1]) / 100.0,  # Extract percentage
                    'timestamp': event.get('timestamp'),
                    'price_level': zone_levels[closest_zone],
                    'duration': 60,
                    'range_position': float(closest_zone.split('_')[1]) / 100.0,
                    'htf_confluence': 0.9 if closest_zone == 'zone_40_percent' else 0.7,
                    
                    # Archaeological zone specific fields
                    'zone_level': closest_zone,
                    'zone_percentage': float(closest_zone.split('_')[1]),
                    'distance_to_zone': min_distance,
                    'dimensional_destiny': closest_zone == 'zone_40_percent',  # Theory B
                    'final_range_relationship': True,
                    
                    # Archaeological analysis
                    'archaeological_significance': 0.95 if closest_zone == 'zone_40_percent' else 0.8,
                    'event_family': 'archaeological_zones',
                    'structural_role': 'dimensional_anchor',
                    'theory_b_validated': closest_zone == 'zone_40_percent',
                    
                    # Source tracking
                    'source': 'archaeological_zone_detection',
                    'enhanced_session_data': True,
                    'related_event_id': event.get('event_id', 'unknown')
                }
                zone_events.append(zone_event)
        
        return zone_events
    
    def _calculate_magnitude_from_movement(self, movement: Dict, original_type: str) -> float:
        """Calculate event magnitude using multiple strategies"""
        
        # Strategy 1: Direct intensity (for liquidity events)
        intensity = movement.get('intensity')
        if intensity is not None:
            self.stats['magnitude_calculation_methods']['intensity_direct'] += 1
            return min(float(intensity), 1.0)
        
        # Strategy 2: Price momentum
        price_momentum = movement.get('price_momentum')
        if price_momentum is not None:
            self.stats['magnitude_calculation_methods']['price_momentum'] += 1
            return min(abs(float(price_momentum)) * 10, 1.0)  # Scale momentum
        
        # Strategy 3: Range position significance
        range_position = movement.get('range_position', 0.5)
        if abs(range_position - 0.5) > 0.2:  # Significant deviation from center
            self.stats['magnitude_calculation_methods']['range_position'] += 1
            return min(abs(range_position - 0.5) * 2, 1.0)
        
        # Strategy 4: Energy density
        energy_density = movement.get('energy_density')
        if energy_density is not None:
            self.stats['magnitude_calculation_methods']['energy_density'] += 1
            return min(float(energy_density), 1.0)
        
        # Strategy 5: Composite calculation
        self.stats['magnitude_calculation_methods']['composite'] += 1
        magnitude_sources = [
            abs(movement.get('pct_from_open', 0.0)) * 20,  # Scale percentage
            abs(movement.get('normalized_price', 0.5) - 0.5) * 2,  # Distance from center
            0.7 if 'fvg' in original_type.lower() else 0.5,  # Type-based boost
            0.8 if 'expansion' in original_type.lower() else 0.5,  # Type-based boost
            0.6 if 'consolidation' in original_type.lower() else 0.5  # Type-based boost
        ]
        
        return min(max(magnitude_sources), 1.0)
    
    def _calculate_duration(self, movement: Dict) -> int:
        """Calculate event duration from movement data"""
        # Try multiple duration sources
        duration_sources = [
            movement.get('time_to_next_event'),
            movement.get('impact_duration'),
            60  # Default 1 minute
        ]
        
        for duration in duration_sources:
            if duration is not None and duration > 0:
                return int(duration)
        
        return 60  # Default duration
    
    def _calculate_archaeological_significance(self, range_position: float, event_type: str, magnitude: float) -> float:
        """Calculate archaeological significance score"""
        base_significance = magnitude
        
        # Boost for archaeological zones
        if 'zone' in event_type or abs(range_position - 0.4) < 0.05:  # Near 40% zone
            base_significance *= 1.3
        
        # Boost for FVG events
        if 'fvg' in event_type.lower():
            base_significance *= 1.2
        
        # Boost for expansion/consolidation
        if any(term in event_type.lower() for term in ['expansion', 'consolidation']):
            base_significance *= 1.1
        
        return min(base_significance, 1.0)
    
    def _determine_event_family(self, archaeological_type: str) -> str:
        """Determine the event family for categorization"""
        if 'fvg' in archaeological_type:
            return 'fvg_family'
        elif 'liquidity' in archaeological_type or 'sweep' in archaeological_type:
            return 'liquidity_family'
        elif 'expansion' in archaeological_type:
            return 'expansion_family'
        elif 'consolidation' in archaeological_type:
            return 'consolidation_family'
        elif 'retracement' in archaeological_type:
            return 'retracement_family'
        elif any(term in archaeological_type for term in ['regime', 'structural', 'volatility']):
            return 'structural_family'
        elif 'session' in archaeological_type:
            return 'session_markers'
        elif 'archaeological' in archaeological_type:
            return 'archaeological_zones'
        else:
            return 'unknown_family'
    
    def _determine_structural_role(self, original_type: str, range_position: float) -> str:
        """Determine the structural role of the event"""
        if 'formation' in original_type:
            return 'structure_formation'
        elif 'rebalance' in original_type:
            return 'equilibrium_mechanism'
        elif 'sweep' in original_type:
            return 'liquidity_mechanism'
        elif 'expansion' in original_type:
            return 'momentum_driver'
        elif 'consolidation' in original_type:
            return 'compression_mechanism'
        elif abs(range_position - 0.4) < 0.05:  # Near 40% zone
            return 'dimensional_anchor'
        else:
            return 'market_participant'
    
    def _determine_liquidity_archetype(self, original_type: str) -> str:
        """Determine liquidity archetype"""
        if 'sweep' in original_type and 'low' in original_type:
            return 'session_low_sweep'
        elif 'sweep' in original_type and 'high' in original_type:
            return 'session_high_sweep'
        elif 'gap' in original_type:
            return 'liquidity_gap'
        elif 'volume' in original_type:
            return 'volume_liquidity'
        else:
            return 'general_liquidity'
    
    def _calculate_dimensional_relationship(self, range_position: float) -> str:
        """Calculate dimensional relationship based on Theory B"""
        if abs(range_position - 0.4) < 0.02:
            return 'dimensional_destiny_40pct'
        elif abs(range_position - 0.2) < 0.02:
            return 'structural_support_20pct'
        elif abs(range_position - 0.6) < 0.02:
            return 'resistance_confluence_60pct'
        elif abs(range_position - 0.8) < 0.02:
            return 'momentum_threshold_80pct'
        else:
            return 'transitional_zone'
    
    def _create_enhanced_features(self, session_data: Dict) -> Dict:
        """Create enhanced_features from session metadata and computed metrics"""
        session_metadata = session_data.get('session_metadata', {})
        session_fpfvg = session_data.get('session_fpfvg', {})
        relativity_stats = session_data.get('relativity_stats', {})
        
        # Calculate session-level metrics
        price_movements = session_data.get('price_movements', [])
        liquidity_events = session_data.get('session_liquidity_events', [])
        
        total_events = len(price_movements) + len(liquidity_events)
        avg_energy = self._calculate_average_energy(price_movements)
        session_volatility = self._calculate_session_volatility(price_movements)
        archaeological_density = self._calculate_archaeological_density(price_movements)
        
        enhanced_features = {
            # Session characteristics
            'session_type': session_metadata.get('session_type', 'unknown'),
            'session_date': session_metadata.get('session_date', ''),
            'session_duration': session_metadata.get('session_duration', 0),
            'total_events': total_events,
            'price_event_count': len(price_movements),
            'liquidity_event_count': len(liquidity_events),
            
            # Energy and volatility metrics
            'average_energy_density': avg_energy,
            'session_volatility': session_volatility,
            'event_density': total_events / max(1, session_metadata.get('session_duration', 60)),
            'archaeological_density': archaeological_density,
            
            # FPFVG characteristics
            'fpfvg_present': bool(session_fpfvg.get('fpfvg_present', False)),
            'fpfvg_characteristics': session_fpfvg,
            'fpfvg_interaction_count': len(session_fpfvg.get('fpfvg_formation', {}).get('interactions', [])),
            
            # HTF analysis
            'htf_carryover': session_metadata.get('htf_carryover', 0.0),
            'htf_influence': session_metadata.get('htf_influence', 0.0),
            'htf_confluence_events': sum(1 for mv in price_movements if mv.get('htf_confluence_strength', 0) > 0.7),
            
            # Session completion metrics
            'session_completion_rate': session_metadata.get('completion_rate', 1.0),
            'range_development': session_metadata.get('range_development', 0.0),
            'session_range': relativity_stats.get('session_range', 0),
            'session_high': relativity_stats.get('session_high', 0),
            'session_low': relativity_stats.get('session_low', 0),
            
            # Archaeological features
            'dimensional_anchoring_potential': self._calculate_dimensional_anchoring_potential(price_movements),
            'theory_b_validation_score': self._calculate_theory_b_score(price_movements, relativity_stats),
            'cross_session_inheritance': session_metadata.get('cross_session_inheritance', 0.0),
            'temporal_non_locality_index': self._calculate_temporal_non_locality(price_movements),
            
            # Enhanced session metadata
            'relativity_enhanced': True,
            'normalization_applied': relativity_stats.get('normalization_applied', False),
            'structural_relationships_enabled': relativity_stats.get('structural_relationships_enabled', False),
            'permanent_pattern_capability': relativity_stats.get('permanent_pattern_capability', False)
        }
        
        return enhanced_features
    
    def _calculate_average_energy(self, price_movements: List[Dict]) -> float:
        """Calculate average energy density from price movements"""
        if not price_movements:
            return 0.5
        
        energy_values = []
        for movement in price_movements:
            energy = movement.get('energy_density', 
                                abs(movement.get('price_momentum', 0.0)))
            energy_values.append(energy)
        
        return sum(energy_values) / len(energy_values) if energy_values else 0.5
    
    def _calculate_session_volatility(self, price_movements: List[Dict]) -> float:
        """Calculate session volatility from price movements"""
        if len(price_movements) < 2:
            return 0.0
        
        price_levels = [m.get('price_level', 0) for m in price_movements if m.get('price_level')]
        if len(price_levels) < 2:
            return 0.0
        
        # Calculate range as percentage
        price_range = max(price_levels) - min(price_levels)
        avg_price = sum(price_levels) / len(price_levels)
        
        return (price_range / avg_price) if avg_price > 0 else 0.0
    
    def _calculate_archaeological_density(self, price_movements: List[Dict]) -> float:
        """Calculate density of archaeologically significant events"""
        if not price_movements:
            return 0.0
        
        significant_events = 0
        for movement in price_movements:
            movement_type = movement.get('movement_type', '').lower()
            range_pos = movement.get('range_position', 0.5)
            
            # Check for archaeological significance
            if (any(term in movement_type for term in ['fvg', 'expansion', 'consolidation']) or
                abs(range_pos - 0.4) < 0.05 or abs(range_pos - 0.2) < 0.05 or
                abs(range_pos - 0.6) < 0.05 or abs(range_pos - 0.8) < 0.05):
                significant_events += 1
        
        return significant_events / len(price_movements)
    
    def _calculate_dimensional_anchoring_potential(self, price_movements: List[Dict]) -> float:
        """Calculate potential for dimensional anchoring based on Theory B"""
        if not price_movements:
            return 0.0
        
        anchoring_events = 0
        for movement in price_movements:
            range_pos = movement.get('range_position', 0.5)
            # Check proximity to key archaeological zones
            if any(abs(range_pos - zone) < 0.05 for zone in [0.2, 0.4, 0.6, 0.8]):
                anchoring_events += 1
        
        return anchoring_events / len(price_movements)
    
    def _calculate_theory_b_score(self, price_movements: List[Dict], relativity_stats: Dict) -> float:
        """Calculate Theory B validation score"""
        if not price_movements:
            return 0.0
        
        # Count events near 40% zone (dimensional destiny)
        forty_percent_events = 0
        for movement in price_movements:
            range_pos = movement.get('range_position', 0.5)
            if abs(range_pos - 0.4) < 0.02:  # Within 2% of 40% zone
                forty_percent_events += 1
        
        # Score based on 40% zone concentration
        theory_b_score = forty_percent_events / len(price_movements)
        
        # Boost for FPFVG presence (related to dimensional destiny)
        if relativity_stats.get('fpfvg_present', False):
            theory_b_score *= 1.2
        
        return min(theory_b_score, 1.0)
    
    def _calculate_temporal_non_locality(self, price_movements: List[Dict]) -> float:
        """Calculate temporal non-locality index"""
        if not price_movements:
            return 0.0
        
        # Look for events that position relative to eventual completion
        forward_positioning_events = 0
        total_positioned_events = 0
        
        for movement in price_movements:
            range_pos = movement.get('range_position', 0.5)
            time_normalized = movement.get('normalized_time', 0.0)
            
            # Events that show forward-looking positioning
            if (abs(range_pos - 0.4) < 0.05 and time_normalized < 0.5):  # Early 40% positioning
                forward_positioning_events += 1
            
            if range_pos != 0.5:  # Any positioned event
                total_positioned_events += 1
        
        return forward_positioning_events / max(1, total_positioned_events)
    
    def _update_statistics(self, all_events: List[Dict], price_events: List[Dict], 
                          liquidity_events: List[Dict], fpfvg_events: List[Dict]):
        """Update adapter statistics"""
        self.stats['sessions_processed'] += 1
        self.stats['events_extracted'] += len(all_events)
        self.stats['price_movements_converted'] += len(price_events)
        self.stats['liquidity_events_converted'] += len(liquidity_events)
        self.stats['fpfvg_events_converted'] += len(fpfvg_events)
        
        # Count archaeological zones
        self.stats['archaeological_zones_detected'] += sum(
            1 for event in all_events if event.get('type') == 'archaeological_zone'
        )
        
        # Update event family distribution
        for event in all_events:
            family = event.get('event_family', 'unknown_family')
            if family in self.stats['event_family_distribution']:
                self.stats['event_family_distribution'][family] += 1
    
    def get_adapter_stats(self) -> Dict:
        """Get comprehensive statistics about the adaptation process"""
        stats = dict(self.stats)
        stats['unmapped_event_types'] = list(self.stats['unmapped_event_types'])
        stats['event_type_mapping_coverage'] = len(self.EVENT_TYPE_MAPPING)
        stats['total_magnitude_calculations'] = sum(self.stats['magnitude_calculation_methods'].values())
        
        return stats
    
    def reset_stats(self):
        """Reset adapter statistics"""
        self.stats = {
            'sessions_processed': 0,
            'events_extracted': 0,
            'price_movements_converted': 0,
            'liquidity_events_converted': 0,
            'fpfvg_events_converted': 0,
            'archaeological_zones_detected': 0,
            'magnitude_calculation_methods': {
                'intensity_direct': 0,
                'price_momentum': 0,
                'range_position': 0,
                'energy_density': 0,
                'composite': 0
            },
            'unmapped_event_types': set(),
            'event_family_distribution': {
                'fvg_family': 0,
                'liquidity_family': 0,
                'expansion_family': 0,
                'consolidation_family': 0,
                'retracement_family': 0,
                'structural_family': 0,
                'session_markers': 0,
                'archaeological_zones': 0
            }
        }
    
    def process_multiple_sessions(self, session_files: List[str]) -> List[Dict]:
        """Process multiple enhanced session files and return adapted format"""
        adapted_sessions = []
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                adapted_session = self.adapt_enhanced_session(session_data)
                adapted_sessions.append(adapted_session)
                
            except Exception as e:
                print(f"Error processing {session_file}: {e}")
                continue
        
        return adapted_sessions


# Integration patch for broad_spectrum_archaeology.py
class ArchaeologySystemPatch:
    """
    Non-invasive patch for broad_spectrum_archaeology.py to use enhanced session adapter
    """
    
    @staticmethod
    def patch_extract_timeframe_events(archaeology_instance):
        """
        Patch the _extract_timeframe_events method to use enhanced session adapter
        """
        # Store original method
        if not hasattr(archaeology_instance, '_original_extract_timeframe_events'):
            archaeology_instance._original_extract_timeframe_events = archaeology_instance._extract_timeframe_events
        
        # Create adapter instance
        adapter = EnhancedSessionAdapter()
        archaeology_instance.adapter = adapter
        
        def patched_extract_timeframe_events(session_data, timeframe, config):
            """Patched version that handles enhanced session format"""
            
            # Check if this is enhanced session format
            if ('price_movements' in session_data or 
                'session_liquidity_events' in session_data or
                'session_fpfvg' in session_data):
                
                # Adapt to archaeological format
                adapted_data = adapter.adapt_enhanced_session(session_data)
                
                # Use original method with adapted data
                return archaeology_instance._original_extract_timeframe_events(
                    adapted_data, timeframe, config
                )
            else:
                # Use original method for non-enhanced sessions
                return archaeology_instance._original_extract_timeframe_events(
                    session_data, timeframe, config
                )
        
        # Replace the method
        archaeology_instance._extract_timeframe_events = patched_extract_timeframe_events
        
        print("✅ Enhanced Session Adapter integrated with archaeology system")
        return archaeology_instance
    
    @staticmethod
    def remove_patch(archaeology_instance):
        """Remove the patch and restore original functionality"""
        if hasattr(archaeology_instance, '_original_extract_timeframe_events'):
            archaeology_instance._extract_timeframe_events = archaeology_instance._original_extract_timeframe_events
            delattr(archaeology_instance, '_original_extract_timeframe_events')
            
            if hasattr(archaeology_instance, 'adapter'):
                delattr(archaeology_instance, 'adapter')
            
            print("✅ Enhanced Session Adapter patch removed")
        else:
            print("⚠️  No patch to remove")


# Quick test function
def test_adapter_with_sample():
    """Test the adapter with a sample enhanced session"""
    
    # Sample enhanced session structure (based on real data)
    sample_session = {
        "session_metadata": {
            "session_type": "ny_pm",
            "session_date": "2025-08-05",
            "session_start": "13:30:00",
            "session_end": "16:09:00",
            "session_duration": 159,
            "htf_carryover": 0.7
        },
        "session_fpfvg": {
            "fpfvg_present": True,
            "fpfvg_formation": {
                "formation_time": "13:31:00",
                "premium_high": 23208.75,
                "discount_low": 23208.0,
                "gap_size": 0.75,
                "interactions": [
                    {
                        "interaction_time": "13:33:00",
                        "interaction_type": "rebalance",
                        "price_level": 23208.75,
                        "interaction_context": "pm_fpfvg_rebalance_during_consolidation"
                    }
                ]
            }
        },
        "price_movements": [
            {
                "timestamp": "13:31:00",
                "price_level": 23208.75,
                "movement_type": "pm_fpfvg_formation_premium",
                "normalized_price": 0.6843065693430657,
                "pct_from_open": -0.008616697004620703,
                "pct_from_high": 31.569343065693428,
                "pct_from_low": 68.43065693430657,
                "range_position": 0.6843065693430657,
                "price_momentum": 0.15,
                "energy_density": 0.8,
                "time_since_session_open": 60,
                "normalized_time": 0.006289308176100629,
                "time_to_next_event": 120
            },
            {
                "timestamp": "13:35:00",
                "price_level": 23201.25,
                "movement_type": "expansion_start_higher",
                "normalized_price": 0.6295620437956204,
                "range_position": 0.6295620437956204,
                "price_momentum": -0.03231539828728389,
                "time_since_session_open": 300,
                "time_to_next_event": 240
            }
        ],
        "session_liquidity_events": [
            {
                "timestamp": "13:30:00",
                "event_type": "price_gap",
                "intensity": 0.926,
                "price_level": 23210.75,
                "impact_duration": 13
            },
            {
                "timestamp": "13:31:00",
                "event_type": "momentum_shift",
                "intensity": 0.93,
                "price_level": 23208.75,
                "impact_duration": 26
            }
        ],
        "relativity_stats": {
            "session_high": 23252.0,
            "session_low": 23115.0,
            "session_open": 23210.75,
            "session_close": 23146.25,
            "session_range": 137.0,
            "normalization_applied": True,
            "structural_relationships_enabled": True,
            "permanent_pattern_capability": True
        }
    }
    
    # Test the adapter
    adapter = EnhancedSessionAdapter()
    adapted = adapter.adapt_enhanced_session(sample_session)
    
    print("=== ENHANCED SESSION ADAPTER TEST RESULTS ===")
    print(f"📊 Events extracted: {len(adapted['events'])}")
    print(f"🎯 Enhanced features created: {len(adapted['enhanced_features'])}")
    print(f"⚡ Archaeological zones detected: {sum(1 for e in adapted['events'] if e.get('type') == 'archaeological_zone')}")
    
    print("\n📋 Event breakdown by family:")
    event_families = {}
    for event in adapted['events']:
        family = event.get('event_family', 'unknown')
        event_families[family] = event_families.get(family, 0) + 1
    
    for family, count in event_families.items():
        print(f"  {family}: {count} events")
    
    print(f"\n📈 Sample adapted event:")
    if adapted['events']:
        sample_event = adapted['events'][0]
        print(f"  Type: {sample_event.get('type')} (from {sample_event.get('original_type')})")
        print(f"  Magnitude: {sample_event.get('magnitude'):.3f}")
        print(f"  Archaeological significance: {sample_event.get('archaeological_significance'):.3f}")
        print(f"  Event family: {sample_event.get('event_family')}")
        print(f"  Structural role: {sample_event.get('structural_role')}")
    
    print(f"\n📊 Adapter statistics:")
    stats = adapter.get_adapter_stats()
    print(f"  Event type mappings: {stats['event_type_mapping_coverage']}")
    print(f"  Total magnitude calculations: {stats['total_magnitude_calculations']}")
    print(f"  Archaeological zones: {stats['archaeological_zones_detected']}")
    
    return adapted


if __name__ == "__main__":
    # Run comprehensive test
    print("🧪 Testing Enhanced Session Adapter...")
    test_result = test_adapter_with_sample()
    print(f"\n✅ Enhanced Session Adapter ready for integration!")
    print(f"   Expected improvement: 0 → 15-25+ events per session")
    print(f"   Actual test result: {len(test_result['events'])} events extracted")