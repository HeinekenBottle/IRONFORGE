#!/usr/bin/env python3
"""
IRONFORGE Edge Scoring Functions - Archaeological Discovery Enhancement
======================================================================

Implements 5 specific edge scoring functions for the IRONFORGE archaeological 
discovery system. These functions calculate enhanced relationship scores between
nodes in the temporal graph attention network (TGAT) to discover permanent 
market relationships across time and price distance.

Edge Scorers:
1. temporal_resonance: Detects harmonic time relationships
2. semantic_weight: Calculates semantic relationship importance  
3. causality_score: Estimates causal relationship strength
4. hierarchy_distance: Calculates distance in multi-timeframe hierarchy
5. permanence_score: Estimates cross-regime stability

These functions are plugged into the EnhancedGraphBuilder architecture via:
builder.set_edge_scorers(get_all_scorers())
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

# Import the RichNodeFeature class from the enhanced graph builder
try:
    from .enhanced_graph_builder import RichNodeFeature
except ImportError:
    # Fallback if running standalone
    from enhanced_graph_builder import RichNodeFeature

logger = logging.getLogger(__name__)


def temporal_resonance(source_feat: RichNodeFeature, 
                      target_feat: RichNodeFeature, 
                      time_delta: float) -> float:
    """
    Detect harmonic time relationships between market events.
    
    Looks for patterns like same minute marks across sessions, hourly intervals,
    session boundaries, and cyclic timing patterns that indicate temporal resonance.
    
    Args:
        source_feat: Source node rich features (27 dimensions)
        target_feat: Target node rich features (27 dimensions) 
        time_delta: Time difference in minutes between nodes
        
    Returns:
        float: 0.0-1.0 where 1.0 = perfect harmonic resonance
        
    Mathematical Foundation:
        Uses harmonic analysis with sin/cos relationships to detect:
        - Same minute marks (e.g., both at :15 minutes of different hours)
        - Hourly intervals (60min, 120min, etc.)
        - Session boundary alignments
        - Daily phase alignment (morning/afternoon cycles)
    """
    try:
        # Perfect same-time resonance
        if abs(time_delta) < 0.5:  # Same minute
            return 1.0
            
        # Harmonic minute mark resonance (same minute within hour)
        source_minute_mark = source_feat.time_minutes % 60
        target_minute_mark = target_feat.time_minutes % 60
        minute_mark_diff = min(abs(source_minute_mark - target_minute_mark),
                              60 - abs(source_minute_mark - target_minute_mark))
        
        if minute_mark_diff < 1.0:  # Same minute mark within hour
            minute_resonance = 1.0 - (minute_mark_diff / 1.0)
        else:
            minute_resonance = 0.0
            
        # Hourly interval resonance (multiples of 60 minutes)
        hourly_intervals = [60, 120, 180, 240, 300, 360]  # 1-6 hours
        hourly_resonance = 0.0
        
        for interval in hourly_intervals:
            interval_remainder = abs(time_delta) % interval
            interval_proximity = min(interval_remainder, interval - interval_remainder)
            if interval_proximity < 5.0:  # Within 5 minutes of hourly boundary
                hourly_resonance = max(hourly_resonance, 1.0 - (interval_proximity / 5.0))
                
        # Daily phase resonance using sin/cos features
        phase_correlation = (source_feat.daily_phase_sin * target_feat.daily_phase_sin + 
                           source_feat.daily_phase_cos * target_feat.daily_phase_cos)
        phase_resonance = max(0.0, phase_correlation)  # 0 to 1 range
        
        # Session position resonance (same relative position in session)
        session_pos_diff = abs(source_feat.session_position - target_feat.session_position)
        session_resonance = 1.0 - min(1.0, session_pos_diff)
        
        # Combine resonances with weighted importance
        total_resonance = (
            0.4 * minute_resonance +      # Minute mark alignment (highest weight)
            0.3 * hourly_resonance +      # Hourly boundary alignment  
            0.2 * phase_resonance +       # Daily cycle alignment
            0.1 * session_resonance       # Session position alignment
        )
        
        return min(1.0, total_resonance)
        
    except Exception as e:
        logger.warning(f"temporal_resonance calculation failed: {e}")
        return 0.5  # Neutral default


def semantic_weight(source_feat: RichNodeFeature, 
                   target_feat: RichNodeFeature, 
                   edge_type: str) -> float:
    """
    Calculate semantic relationship importance between market events.
    
    Considers event types, liquidity types, session characters, and structural
    importance to weight edges based on their semantic meaning in market structure.
    
    Args:
        source_feat: Source node rich features
        target_feat: Target node rich features
        edge_type: Type of edge ('temporal', 'scale', 'cascade', etc.)
        
    Returns:
        float: 0.0-2.0 where 2.0 = maximum semantic importance
        
    Mathematical Foundation:
        Uses information theory concepts and market structure hierarchy:
        - Event type compatibility matrix
        - Timeframe hierarchy importance  
        - Liquidity type interaction strength
        - Fisher regime and session character alignment
    """
    try:
        base_weight = 1.0
        
        # Edge type base weights
        edge_type_weights = {
            'temporal': 1.0,           # Standard temporal relationships
            'scale': 1.5,              # Cross-timeframe relationships (higher importance)
            'cascade': 1.8,            # Cascade relationships (very important)
            'pd_array': 1.6,           # Price action structure
            'cross_tf_confluence': 2.0, # Cross-timeframe confluence (maximum)
            'temporal_echo': 1.2,      # Temporal echoes
            'discovered': 1.4          # Archaeological discoveries
        }
        
        edge_weight = edge_type_weights.get(edge_type, 1.0)
        
        # Event type compatibility (certain event types are more semantically related)
        event_compatibility = 1.0
        if source_feat.event_type_id == target_feat.event_type_id:
            event_compatibility = 1.5  # Same event types are more related
            
        # Timeframe hierarchy importance (connections across more distant timeframes are more significant)
        tf_distance = abs(source_feat.timeframe_source - target_feat.timeframe_source)
        if tf_distance > 0:
            # Cross-timeframe connections are semantically more important
            tf_importance = 1.0 + (0.2 * tf_distance)  # Up to 2.0 for max TF distance
        else:
            tf_importance = 1.0  # Same timeframe
            
        # Fisher regime alignment (events in different regimes indicate transitions)
        regime_factor = 1.0
        if source_feat.fisher_regime != target_feat.fisher_regime:
            regime_factor = 1.3  # Regime transitions are semantically important
            
        # Session character alignment
        session_factor = 1.0
        if source_feat.session_character == target_feat.session_character:
            session_factor = 1.1  # Same session character reinforces relationship
        else:
            session_factor = 1.2  # Different session characters indicate transitions
            
        # Liquidity type interaction strength
        liquidity_factor = 1.0
        if source_feat.liquidity_type != target_feat.liquidity_type:
            liquidity_factor = 1.2  # Different liquidity types indicate complex interactions
            
        # Structural importance weighting
        structural_factor = 1.0 + 0.1 * (source_feat.structural_importance + 
                                        target_feat.structural_importance) / 2.0
        
        # First presentation events are semantically more important
        presentation_factor = 1.0
        if source_feat.first_presentation_flag > 0.5 or target_feat.first_presentation_flag > 0.5:
            presentation_factor = 1.2
            
        # Combine all factors
        total_weight = (base_weight * edge_weight * event_compatibility * 
                       tf_importance * regime_factor * session_factor * 
                       liquidity_factor * structural_factor * presentation_factor)
        
        return min(2.0, total_weight)
        
    except Exception as e:
        logger.warning(f"semantic_weight calculation failed: {e}")
        return 1.0  # Default semantic weight


def causality_score(source_feat: RichNodeFeature, 
                   target_feat: RichNodeFeature, 
                   graph_context: Dict[str, Any]) -> float:
    """
    Estimate causal relationship strength between market events.
    
    Earlier events with higher energy/contamination that lead to later cascades
    indicate strong causal relationships. Considers temporal ordering, price
    movements, and event significance.
    
    Args:
        source_feat: Source node rich features
        target_feat: Target node rich features  
        graph_context: Graph context with additional information
        
    Returns:
        float: 0.0-1.0 where 1.0 = strong causal relationship
        
    Mathematical Foundation:
        Uses causal inference patterns:
        - Temporal precedence (cause precedes effect)
        - Energy/contamination transfer
        - Price movement directionality
        - Event magnitude correlation
    """
    try:
        # Temporal precedence is required for causality
        time_diff = target_feat.time_minutes - source_feat.time_minutes
        if time_diff <= 0:
            return 0.0  # No causality if target precedes source
            
        # Time decay factor (closer events have stronger causal potential)
        time_decay = np.exp(-time_diff / 60.0)  # 1-hour half-life
        
        # Energy/contamination transfer strength
        source_energy = source_feat.energy_state + source_feat.contamination_coefficient
        target_energy = target_feat.energy_state + target_feat.contamination_coefficient
        
        # Strong source energy with subsequent target activity suggests causality
        energy_causality = 0.0
        if source_energy > 0.1:  # Significant source energy
            if target_energy > source_energy * 0.5:  # Target shows elevated energy
                energy_causality = min(1.0, source_energy * 2.0)
                
        # Price movement directionality (same direction suggests causal relationship)
        price_direction_causality = 0.0
        source_price_momentum = (source_feat.price_delta_1m + source_feat.price_delta_5m) / 2.0
        target_price_momentum = (target_feat.price_delta_1m + target_feat.price_delta_5m) / 2.0
        
        # Mathematical fix: Use epsilon to prevent division by zero and handle actual zero momentum
        MOMENTUM_EPSILON = 1e-8
        abs_source = max(abs(source_price_momentum), MOMENTUM_EPSILON)
        abs_target = max(abs(target_price_momentum), MOMENTUM_EPSILON)
        
        if abs_source > 0.001 and abs_target > 0.001:
            # Same direction momentum suggests causality - now division by zero safe
            direction_correlation = (source_price_momentum * target_price_momentum) / (abs_source * abs_target)
            if direction_correlation > 0:
                price_direction_causality = direction_correlation
                
        # Event magnitude correlation (significant source events causing target events)
        magnitude_causality = 0.0
        source_magnitude = source_feat.pd_array_strength + source_feat.structural_importance
        target_magnitude = target_feat.pd_array_strength + target_feat.structural_importance
        
        if source_magnitude > 0.5 and target_magnitude > 0.3:  # Both significant
            magnitude_causality = min(1.0, source_magnitude * target_magnitude)
            
        # Fisher regime transitions suggest causal relationships
        regime_causality = 0.0
        if source_feat.fisher_regime != target_feat.fisher_regime:
            regime_causality = 0.3  # Regime transitions indicate causality
            
        # Volatility increase suggests causal impact
        volatility_causality = 0.0
        if target_feat.volatility_window > source_feat.volatility_window * 1.2:
            # Mathematical fix: Prevent division by zero in volatility calculation
            if source_feat.volatility_window > 1e-8:
                volatility_causality = min(1.0, (target_feat.volatility_window / 
                                               source_feat.volatility_window) - 1.0)
            else:
                # Handle zero volatility source: use target volatility as causality measure
                volatility_causality = min(1.0, target_feat.volatility_window * 10.0)
                                           
        # Cross-timeframe causality (higher TF events causing lower TF effects)
        tf_causality = 0.0
        if source_feat.timeframe_source > target_feat.timeframe_source:
            tf_distance = source_feat.timeframe_source - target_feat.timeframe_source
            tf_causality = 0.2 * tf_distance  # Higher TF causing lower TF
            
        # Combine causality factors with time decay
        total_causality = time_decay * (
            0.25 * energy_causality +
            0.20 * price_direction_causality +  
            0.20 * magnitude_causality +
            0.15 * regime_causality +
            0.10 * volatility_causality +
            0.10 * tf_causality
        )
        
        return min(1.0, total_causality)
        
    except Exception as e:
        logger.warning(f"causality_score calculation failed: {e}")
        return 0.5  # Neutral default


def hierarchy_distance(source_feat: RichNodeFeature, 
                      target_feat: RichNodeFeature, 
                      timeframe_jump: int) -> float:
    """
    Calculate distance in multi-timeframe hierarchy.
    
    Normalizes timeframe distance by hierarchy span. Higher distances indicate
    more significant cross-timeframe relationships.
    
    Args:
        source_feat: Source node rich features
        target_feat: Target node rich features
        timeframe_jump: Raw timeframe jump (difference in TF indices)
        
    Returns:
        float: 0.0-1.0 where 0.0 = same timeframe, 1.0 = maximum distance
        
    Mathematical Foundation:
        Uses graph distance metrics in timeframe hierarchy:
        - Linear distance: |TF_target - TF_source| / max_distance
        - Logarithmic scaling for non-linear TF relationships
        - Directional weighting (lower to higher vs higher to lower)
    """
    try:
        # Maximum possible timeframe distance (0=1m to 5=W)
        MAX_TF_DISTANCE = 5
        
        # Same timeframe = 0 distance
        if timeframe_jump == 0:
            return 0.0
            
        # Raw distance normalized by maximum
        raw_distance = abs(timeframe_jump) / MAX_TF_DISTANCE
        
        # Apply logarithmic scaling for non-linear TF relationships
        # Timeframes have non-linear jumps: 1m->5m (5x), 5m->15m (3x), 15m->1h (4x), etc.
        tf_multipliers = [1, 5, 15, 60, 1440, 10080]  # Minutes in each timeframe
        
        source_tf_idx = int(source_feat.timeframe_source)
        target_tf_idx = int(target_feat.timeframe_source)
        
        if 0 <= source_tf_idx < len(tf_multipliers) and 0 <= target_tf_idx < len(tf_multipliers):
            source_minutes = tf_multipliers[source_tf_idx]
            target_minutes = tf_multipliers[target_tf_idx]
            
            # Mathematical fix: Prevent division by zero and log10(0) in timeframe calculations
            if source_minutes > 0 and target_minutes > 0:
                # Logarithmic relationship between timeframes
                log_ratio = abs(np.log10(target_minutes / source_minutes))
                log_distance = log_ratio / np.log10(tf_multipliers[-1] / tf_multipliers[0])  # Normalize
            else:
                # Fallback to raw distance if timeframe values are invalid
                log_distance = raw_distance
        else:
            log_distance = raw_distance
            
        # Directional weighting (lower to higher TF vs higher to lower TF)
        direction_weight = 1.0
        if target_feat.timeframe_source > source_feat.timeframe_source:
            # Lower to higher timeframe (aggregation)
            direction_weight = 1.0
        else:
            # Higher to lower timeframe (drill-down)  
            direction_weight = 1.1  # Slightly higher weight for drill-down
            
        # Cross-timeframe confluence adjustment
        confluence_factor = 1.0
        if hasattr(source_feat, 'cross_tf_confluence') and hasattr(target_feat, 'cross_tf_confluence'):
            avg_confluence = (source_feat.cross_tf_confluence + target_feat.cross_tf_confluence) / 2.0
            if avg_confluence > 0.8:  # High confluence reduces perceived distance
                confluence_factor = 0.8
                
        # Price proximity adjustment (similar prices across TFs reduce hierarchy distance)
        price_proximity = 1.0 - min(1.0, abs(source_feat.normalized_price - 
                                            target_feat.normalized_price) / 0.01)
        price_factor = 1.0 - (0.2 * price_proximity)  # Max 20% distance reduction
        
        # Combine factors
        final_distance = (0.6 * raw_distance + 0.4 * log_distance) * direction_weight * confluence_factor * price_factor
        
        return min(1.0, final_distance)
        
    except Exception as e:
        logger.warning(f"hierarchy_distance calculation failed: {e}")
        return float(min(1.0, abs(timeframe_jump) / 5.0))  # Simple fallback


def permanence_score(source_feat: RichNodeFeature, 
                    target_feat: RichNodeFeature, 
                    graph_context: Dict[str, Any]) -> float:
    """
    Estimate cross-regime stability (archaeological permanence).
    
    Relationships that survive across different market conditions indicate
    permanent structural features. Considers structural importance, event tiers,
    price levels, and regime independence.
    
    Args:
        source_feat: Source node rich features
        target_feat: Target node rich features  
        graph_context: Graph context with market regime information
        
    Returns:
        float: 0.0-1.0 where 1.0 = maximum cross-regime permanence
        
    Mathematical Foundation:
        Uses stability analysis:
        - Structural importance persistence
        - Fisher regime independence  
        - Cross-session stability
        - Price level significance
    """
    try:
        # Base permanence from structural importance
        avg_structural = (source_feat.structural_importance + target_feat.structural_importance) / 2.0
        base_permanence = min(1.0, avg_structural)
        
        # Fisher regime independence (relationships that persist across regimes)
        regime_independence = 0.0
        if source_feat.fisher_regime != target_feat.fisher_regime:
            # Different regimes suggest the relationship transcends market conditions
            regime_independence = 0.4
        else:
            # Same regime, but check if both are in transitional regime
            if source_feat.fisher_regime == 2:  # Transitional regime
                regime_independence = 0.3  # Moderate permanence in transitions
            elif source_feat.fisher_regime == 0:  # Baseline regime
                regime_independence = 0.2  # Lower permanence in quiet conditions
            else:  # Elevated regime
                regime_independence = 0.1  # Least permanent in stressed conditions
                
        # Session character independence
        session_independence = 0.0
        if source_feat.session_character != target_feat.session_character:
            # Relationships across different session types are more permanent
            session_independence = 0.3
        else:
            # Same session character gets moderate permanence
            session_independence = 0.15
            
        # Price level significance (major price levels create permanent relationships)
        price_significance = 0.0
        major_levels = [0.23000, 0.23500, 0.24000, 0.24500, 0.25000]  # Example major levels
        
        for level in major_levels:
            source_distance = abs(source_feat.normalized_price - level)
            target_distance = abs(target_feat.normalized_price - level)
            
            if source_distance < 0.001 or target_distance < 0.001:  # Near major level
                price_significance = max(price_significance, 0.3)
            elif source_distance < 0.005 or target_distance < 0.005:  # Moderately near
                price_significance = max(price_significance, 0.15)
                
        # Cross-timeframe permanence (multi-TF relationships are more permanent)
        tf_permanence = 0.0
        if source_feat.timeframe_source != target_feat.timeframe_source:
            tf_distance = abs(source_feat.timeframe_source - target_feat.timeframe_source)
            tf_permanence = min(0.4, 0.1 * tf_distance)  # Higher TF distance = more permanent
            
        # First presentation permanence (original events create permanent patterns)
        presentation_permanence = 0.0
        if source_feat.first_presentation_flag > 0.5 or target_feat.first_presentation_flag > 0.5:
            presentation_permanence = 0.2
            
        # PD array strength permanence (strong patterns create permanent relationships)
        pd_permanence = 0.0
        avg_pd_strength = (source_feat.pd_array_strength + target_feat.pd_array_strength) / 2.0
        pd_permanence = min(0.3, avg_pd_strength * 0.3)
        
        # Energy state independence (permanent relationships work across energy levels)
        energy_independence = 0.0
        energy_diff = abs(source_feat.energy_state - target_feat.energy_state)
        if energy_diff > 0.5:  # Significant energy difference
            energy_independence = 0.2  # Relationship transcends energy levels
            
        # Weekend proximity permanence (relationships near market structure boundaries)
        weekend_permanence = 0.0
        if source_feat.weekend_proximity < 0.1 or target_feat.weekend_proximity < 0.1:
            weekend_permanence = 0.1  # Near weekend boundaries create permanent patterns
            
        # Combine permanence factors
        total_permanence = (
            0.20 * base_permanence +        # Structural importance base
            0.15 * regime_independence +    # Fisher regime transcendence
            0.15 * session_independence +   # Session character transcendence  
            0.15 * price_significance +     # Major price level significance
            0.10 * tf_permanence +          # Cross-timeframe relationships
            0.10 * presentation_permanence + # First presentation events
            0.10 * pd_permanence +          # PD array strength
            0.03 * energy_independence +    # Energy level transcendence
            0.02 * weekend_permanence       # Market boundary effects
        )
        
        # Archaeological discovery bonus (discovered relationships are more permanent)
        if graph_context.get('edge_type') == 'discovered':
            total_permanence *= 1.2
            
        return min(1.0, total_permanence)
        
    except Exception as e:
        logger.warning(f"permanence_score calculation failed: {e}")
        return 0.0  # Conservative default for permanence


def get_all_scorers() -> Dict[str, Callable]:
    """
    Get all edge scoring functions for integration with EnhancedGraphBuilder.
    
    Returns:
        Dict mapping scorer names to callable functions
        
    Usage:
        builder = EnhancedGraphBuilder()
        builder.set_edge_scorers(get_all_scorers())
    """
    return {
        'temporal_resonance': temporal_resonance,
        'semantic_weight': semantic_weight, 
        'causality_score': causality_score,
        'hierarchy_distance': hierarchy_distance,
        'permanence_score': permanence_score
    }


def validate_scorers() -> bool:
    """
    Validate all scorer functions with sample data.
    
    Returns:
        bool: True if all scorers pass validation
    """
    try:
        logger.info("🔍 Validating IRONFORGE edge scorers...")
        
        # Create sample RichNodeFeature objects
        sample_source = RichNodeFeature(
            # Temporal (9)
            time_minutes=60.0, daily_phase_sin=0.5, daily_phase_cos=0.866,
            session_position=0.4, time_to_close=90.0, weekend_proximity=0.3,
            absolute_timestamp=1672531200, day_of_week=2, month_phase=0.15,
            
            # Price & Market (10)
            normalized_price=0.235, price_delta_1m=0.002, price_delta_5m=0.005,
            price_delta_15m=0.008, volatility_window=0.03, energy_state=1.2,
            contamination_coefficient=0.4, fisher_regime=1, session_character=0,
            cross_tf_confluence=0.8,
            
            # Event & Structure (8)  
            event_type_id=1, timeframe_source=2, liquidity_type=1,
            fpfvg_gap_size=15.0, fpfvg_interaction_count=3, first_presentation_flag=1.0,
            pd_array_strength=0.7, structural_importance=0.6,
            
            raw_json={'test': 'data'}
        )
        
        sample_target = RichNodeFeature(
            # Temporal (9)
            time_minutes=75.0, daily_phase_sin=0.7, daily_phase_cos=0.714,
            session_position=0.5, time_to_close=75.0, weekend_proximity=0.3,
            absolute_timestamp=1672532100, day_of_week=2, month_phase=0.15,
            
            # Price & Market (10)
            normalized_price=0.237, price_delta_1m=0.003, price_delta_5m=0.006,
            price_delta_15m=0.009, volatility_window=0.035, energy_state=1.5,
            contamination_coefficient=0.5, fisher_regime=2, session_character=1,
            cross_tf_confluence=0.9,
            
            # Event & Structure (8)
            event_type_id=2, timeframe_source=3, liquidity_type=2,
            fpfvg_gap_size=20.0, fpfvg_interaction_count=4, first_presentation_flag=0.0,
            pd_array_strength=0.8, structural_importance=0.7,
            
            raw_json={'test': 'data2'}
        )
        
        sample_context = {
            'graph_type': 'validation',
            'edge_type': 'temporal',
            'tf_from': '15m',
            'tf_to': '1h'
        }
        
        scorers = get_all_scorers()
        results = {}
        
        # Test temporal_resonance
        score = scorers['temporal_resonance'](sample_source, sample_target, 15.0)
        results['temporal_resonance'] = score
        assert 0.0 <= score <= 1.0, f"temporal_resonance out of range: {score}"
        
        # Test semantic_weight  
        score = scorers['semantic_weight'](sample_source, sample_target, 'scale')
        results['semantic_weight'] = score
        assert 0.0 <= score <= 2.0, f"semantic_weight out of range: {score}"
        
        # Test causality_score
        score = scorers['causality_score'](sample_source, sample_target, sample_context)
        results['causality_score'] = score  
        assert 0.0 <= score <= 1.0, f"causality_score out of range: {score}"
        
        # Test hierarchy_distance
        score = scorers['hierarchy_distance'](sample_source, sample_target, 1)
        results['hierarchy_distance'] = score
        assert 0.0 <= score <= 1.0, f"hierarchy_distance out of range: {score}"
        
        # Test permanence_score
        score = scorers['permanence_score'](sample_source, sample_target, sample_context)
        results['permanence_score'] = score
        assert 0.0 <= score <= 1.0, f"permanence_score out of range: {score}"
        
        logger.info("✅ All edge scorers passed validation")
        logger.info(f"📊 Sample scores: {results}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Edge scorer validation failed: {e}")
        return False


if __name__ == "__main__":
    """Test edge scoring functions"""
    
    logging.basicConfig(level=logging.INFO)
    print("🏗️ IRONFORGE EDGE SCORERS - Function Testing")
    print("=" * 60)
    
    # Validate all scorers
    if validate_scorers():
        print("\n✅ All edge scoring functions validated successfully")
        
        scorers = get_all_scorers()
        print(f"\n📊 Available Scorers: {list(scorers.keys())}")
        
        print("\n🎯 SCORER DESCRIPTIONS:")
        print("   temporal_resonance: Harmonic time relationships (0.0-1.0)")
        print("   semantic_weight: Relationship importance (0.0-2.0)")
        print("   causality_score: Causal relationship strength (0.0-1.0)")
        print("   hierarchy_distance: Multi-TF hierarchy distance (0.0-1.0)")
        print("   permanence_score: Cross-regime stability (0.0-1.0)")
        
        print(f"\n🚀 Integration Ready:")
        print("   from edge_scorers import get_all_scorers")
        print("   builder.set_edge_scorers(get_all_scorers())")
        
    else:
        print("\n❌ Scorer validation failed - check logs for details")
    
    print(f"\n🔗 IRONFORGE archaeological discovery edge scoring complete")