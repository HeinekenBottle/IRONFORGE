"""
Feature Utilities for Enhanced Graph Builder
Shared utilities and calculations for archaeological zone analysis and ICT concepts
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class FeatureUtils:
    """Utility class for feature calculations and archaeological zone analysis"""

    @staticmethod
    def parse_timestamp_to_seconds(timestamp) -> float:
        """Parse timestamp to seconds from start of day"""
        if isinstance(timestamp, (int, float)):
            # Assume Unix timestamp, convert to seconds from start of day
            return float(timestamp % 86400)  # 86400 seconds in a day
        elif isinstance(timestamp, str):
            # Try to parse string timestamp
            try:
                # Simple HH:MM:SS format
                if ":" in timestamp:
                    parts = timestamp.split(":")
                    hours = int(parts[0])
                    minutes = int(parts[1]) if len(parts) > 1 else 0
                    seconds = int(parts[2]) if len(parts) > 2 else 0
                    return float(hours * 3600 + minutes * 60 + seconds)
                else:
                    return float(timestamp)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse timestamp: {timestamp}")
                return 0.0
        else:
            return 0.0

    @staticmethod
    def calculate_session_phase(time_since_open: float, session_duration: float) -> float:
        """Calculate session phase (opening/trending/closing)"""
        if session_duration <= 0:
            return 0.0

        phase_position = time_since_open / session_duration

        # Opening phase (0-20%)
        if phase_position < 0.2:
            return 0.0
        # Trending phase (20-80%)
        elif phase_position < 0.8:
            return 1.0
        # Closing phase (80-100%)
        else:
            return 2.0

    @staticmethod
    def calculate_zone_proximity(
        price: float,
        session_low: float,
        session_range: float,
        zone_level: float,
        session_context: Dict[str, Any] = None,
    ) -> float:
        """Calculate proximity to archaeological zone (Theory B implementation)"""
        if session_range <= 0:
            return 0.0

        # THEORY B: Calculate position in FINAL session range
        position = (price - session_low) / session_range

        # Distance from target zone level
        distance = abs(position - zone_level)

        # Theory B enhanced proximity scoring
        theory_b_active = session_context and session_context.get("theory_b_final_range", False)

        # Special handling for 40% zone (Theory B breakthrough zone)
        if zone_level == 0.40:
            if distance < 0.01:  # Within 1% of 40% zone
                proximity = 1.0
                if theory_b_active:
                    logger.debug(
                        f"Theory B 40% zone: price {price} within 1% (position {position:.3f})"
                    )
            elif distance < 0.02:  # Within 2% of 40% zone
                proximity = 0.95
            elif distance < 0.05:  # Within 5% of 40% zone
                proximity = 0.8
            else:
                proximity = max(0.0, 1.0 - (distance / 0.1))  # Gradual decline over 10%
        else:
            # Standard proximity for other zones
            if distance < 0.02:  # Within 2%
                proximity = 1.0
            elif distance < 0.05:  # Within 5%
                proximity = max(0.0, 1.0 - (distance / 0.05))
            else:
                proximity = 0.0

        # Apply Theory B temporal non-locality bonus for final range calculations
        if theory_b_active and proximity > 0.5:
            proximity = min(1.0, proximity * 1.05)  # 5% bonus for Theory B compliance

        return proximity

    @staticmethod
    def calculate_dimensional_relationship(
        price: float,
        session_low: float,
        session_range: float,
        session_context: Dict[str, Any] = None,
    ) -> float:
        """Calculate Theory B dimensional relationship score"""
        if session_range <= 0:
            return 0.0

        # THEORY B CORE IMPLEMENTATION
        # Position relative to FINAL session range (not current range)
        position = (price - session_low) / session_range

        # Log Theory B application for validation
        theory_b_active = session_context and session_context.get("theory_b_final_range", False)
        if theory_b_active:
            logger.debug(
                f"Theory B: Price {price} positioned at {position:.3f} of FINAL range {session_range:.1f}"
            )

        # Key archaeological levels with Theory B emphasis on 40% zone
        key_levels = [0.20, 0.40, 0.50, 0.618, 0.80]

        # Find closest key level
        min_distance = min(abs(position - level) for level in key_levels)

        # THEORY B DIMENSIONAL RELATIONSHIP SCORING
        # Events at 40% zone have highest significance due to temporal non-locality
        if abs(position - 0.40) < 0.02:  # Within 2% of 40% zone
            score = 1.0  # Perfect Theory B compliance
            if theory_b_active:
                logger.info(
                    f"Theory B 40% ZONE HIT: Price {price} at position {position:.3f} (final range)"
                )
        elif abs(position - 0.40) < 0.05:  # Within 5% of 40% zone
            score = 0.9  # Strong Theory B relationship
        elif min_distance < 0.02:  # Within 2% of any key level
            score = 0.8  # Strong archaeological zone
        elif min_distance < 0.05:  # Within 5% of any key level
            score = 0.6  # Good archaeological zone
        else:
            # Gradual decline based on distance from nearest key level
            score = max(0.0, 1.0 - (min_distance / 0.5))  # Scaled to 50% range

        # Apply Theory B temporal non-locality bonus
        if theory_b_active and score > 0.5:
            # Events positioned accurately relative to final range get bonus
            score = min(1.0, score * 1.1)  # 10% bonus for Theory B compliance

        return score

    @staticmethod
    def calculate_premium_discount_position(
        price: float, session_low: float, session_range: float
    ) -> float:
        """Calculate ICT Premium/Discount Array position with enhanced precision"""
        if session_range <= 0:
            return 0.5  # Equilibrium

        position = (price - session_low) / session_range

        # ICT Enhanced PD Array with granular positioning
        if position < 0.20:
            return 0.0  # Deep Discount
        elif position < 0.40:
            return 0.25  # Discount
        elif position < 0.50:
            return 0.45  # Lower Equilibrium
        elif position < 0.60:
            return 0.55  # Upper Equilibrium
        elif position < 0.80:
            return 0.75  # Premium
        else:
            return 1.0  # Deep Premium

    @staticmethod
    def estimate_liquidity_proximity(price: float, session_high: float, session_low: float) -> float:
        """Calculate ICT liquidity proximity with sweep detection"""
        if session_high <= session_low:
            return 0.0

        session_range = session_high - session_low

        # Calculate distances to key liquidity levels
        distance_to_high = abs(price - session_high) / session_range
        distance_to_low = abs(price - session_low) / session_range

        # ICT Liquidity Zones: liquidity pools exist around key levels
        # Equal High/Low (EQH/EQL) zones
        eqh_zone_threshold = 0.02  # Within 2% of session high
        eql_zone_threshold = 0.02  # Within 2% of session low

        # Check for liquidity zones
        liquidity_score = 0.0

        # High liquidity zone
        if distance_to_high < eqh_zone_threshold:
            liquidity_score = 1.0 - (distance_to_high / eqh_zone_threshold)
        # Low liquidity zone
        elif distance_to_low < eql_zone_threshold:
            liquidity_score = 1.0 - (distance_to_low / eql_zone_threshold)
        # Mid-range liquidity (less significant)
        else:
            # Calculate position relative to key levels
            position = (price - session_low) / session_range
            # Psychological levels (50%, 25%, 75%)
            key_levels = [0.25, 0.50, 0.75]
            min_distance_to_key = min(abs(position - level) for level in key_levels)
            if min_distance_to_key < 0.05:  # Within 5% of key level
                liquidity_score = 0.5 * (1.0 - min_distance_to_key / 0.05)

        return liquidity_score

    @staticmethod
    def calculate_range_extension(price: float, session_high: float, session_low: float) -> float:
        """Calculate ICT range extension with liquidity sweep detection"""
        if session_high <= session_low:
            return 0.0

        session_range = session_high - session_low
        sweep_threshold = session_range * 0.005  # 0.5% beyond range qualifies as sweep

        # Calculate extension beyond range
        if price > session_high:
            extension_distance = price - session_high
            if extension_distance > sweep_threshold:
                # ICT Liquidity Sweep Above (Buy Side Liquidity Sweep)
                return 1.0  # Strong bullish sweep
            else:
                # Minor extension
                return 0.7  # Weak extension
        elif price < session_low:
            extension_distance = session_low - price
            if extension_distance > sweep_threshold:
                # ICT Liquidity Sweep Below (Sell Side Liquidity Sweep)
                return -1.0  # Strong bearish sweep
            else:
                # Minor extension
                return -0.7  # Weak extension
        else:
            # Price within range - check proximity to boundaries
            position = (price - session_low) / session_range
            if position > 0.95:
                return 0.5  # Approaching high
            elif position < 0.05:
                return -0.5  # Approaching low
            else:
                return 0.0  # Within range

    @staticmethod
    def estimate_momentum(event: Dict[str, Any], _session_context: Dict[str, Any]) -> float:
        """Estimate price momentum (simplified)"""
        # This would normally require multiple price points
        # For now, use event significance as proxy
        return event.get("significance", 0.5)

    @staticmethod
    def calculate_volatility_measure(session_range: float, session_context: Dict[str, Any]) -> float:
        """Calculate session volatility measure"""
        # Normalize range by typical market volatility
        typical_range = session_context.get("typical_range", session_range)

        if typical_range > 0:
            return session_range / typical_range
        else:
            return 1.0

    @staticmethod
    def estimate_volume_profile(
        _price: float, volume: float, session_context: Dict[str, Any]
    ) -> float:
        """Estimate volume profile at price level"""
        # Simplified volume profile
        avg_volume = session_context.get("average_volume", 1.0)

        if avg_volume > 0:
            return volume / avg_volume
        else:
            return 1.0

    @staticmethod
    def estimate_pattern_completion(event: Dict[str, Any], _session_context: Dict[str, Any]) -> float:
        """Estimate pattern completion probability"""
        # Based on event type and session context
        event_type = event.get("event_type", "")

        if "redelivery" in event_type:
            return 0.8  # High completion probability
        elif "sweep" in event_type:
            return 0.6  # Medium completion probability
        else:
            return 0.3  # Low completion probability

    @staticmethod
    def calculate_fractal_similarity(_price: float, _session_context: Dict[str, Any]) -> float:
        """Calculate fractal similarity score"""
        # Simplified fractal calculation
        # Would normally compare patterns across timeframes
        return 0.5  # Placeholder

    @staticmethod
    def calculate_archaeological_significance(
        event: Dict[str, Any], _session_context: Dict[str, Any]
    ) -> float:
        """Calculate overall archaeological significance"""
        significance = event.get("significance", 0.5)
        event_type = event.get("event_type", "")

        # Weight by event type importance
        if "fpfvg" in event_type:
            return min(1.0, significance * 1.5)  # FPFVG events more significant
        elif "liquidity_sweep" in event_type:
            return min(1.0, significance * 1.3)  # Liquidity sweeps important
        else:
            return significance

    @staticmethod
    def calculate_fvg_proximity(price: float, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Calculate proximity to Fair Value Gaps (ICT concept)"""
        # Simplified FVG proximity calculation
        event_type = event.get("event_type", "").lower()

        if "fvg" in event_type or "fair_value_gap" in event_type:
            return 1.0  # Direct FVG event
        elif "redelivery" in event_type:
            return 0.8  # Likely FVG redelivery
        else:
            return 0.3  # Default proximity

    @staticmethod
    def calculate_order_block_strength(price: float, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Calculate ICT Order Block strength"""
        # Simplified order block strength calculation
        event_type = event.get("event_type", "").lower()

        if "order_block" in event_type or "ob" in event_type:
            return 1.0  # Direct order block
        elif "liquidity" in event_type:
            return 0.7  # Related to liquidity
        else:
            return 0.4  # Default strength

    @staticmethod
    def calculate_ict_market_structure_shift(price: float, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Calculate ICT Market Structure Shift significance"""
        # Simplified MSS calculation
        event_type = event.get("event_type", "").lower()

        if any(term in event_type for term in ["mss", "structure_shift", "change_of_character"]):
            return 1.0  # Direct structure shift
        elif "reversal" in event_type:
            return 0.8  # Likely structure change
        else:
            return 0.2  # Default significance
