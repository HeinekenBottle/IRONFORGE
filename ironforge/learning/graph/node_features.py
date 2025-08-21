"""
Node Features Module for Enhanced Graph Builder
45D node feature vector with semantic preservation and ICT pattern detection
"""

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


class RichNodeFeature:
    """45D node feature vector with semantic preservation"""

    def __init__(self):
        # 45D total: 8 semantic + 37 traditional
        self.features = torch.zeros(45, dtype=torch.float32)

        # Semantic event flags (first 8 dimensions)
        self.semantic_indices = {
            "fvg_redelivery_flag": 0,
            "expansion_phase_flag": 1,
            "consolidation_flag": 2,
            "retracement_flag": 3,
            "reversal_flag": 4,
            "liq_sweep_flag": 5,
            "pd_array_interaction_flag": 6,
            "semantic_reserved": 7,
        }

        # Traditional features (indices 8-44)
        self.traditional_start = 8

    def set_semantic_event(self, event_type: str, value: float = 1.0):
        """Set semantic event flag"""
        if event_type in self.semantic_indices:
            self.features[self.semantic_indices[event_type]] = value

    def set_traditional_features(self, features: torch.Tensor):
        """Set traditional features (37D)"""
        if features.size(0) == 37:
            self.features[self.traditional_start :] = features
        else:
            logger.warning(f"Expected 37D traditional features, got {features.size(0)}D")


class NodeFeatureProcessor:
    """Processes events into 45D node features with ICT semantic detection"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_node_feature(
        self, event: Dict[str, Any], session_context: Dict[str, Any] = None
    ) -> RichNodeFeature:
        """Create 45D node feature from event data"""
        feature = RichNodeFeature()

        # Set semantic events based on event data - Enhanced ICT Detection
        self._detect_semantic_events(feature, event)

        # Generate traditional features (37D)
        traditional_features = self._calculate_traditional_features(event, session_context)
        feature.set_traditional_features(traditional_features)

        return feature

    def _detect_semantic_events(self, feature: RichNodeFeature, event: Dict[str, Any]):
        """Detect and set ICT semantic events"""
        event_type = event.get("event_type", "").lower()
        original_type = event.get("original_type", "").lower()
        semantic_context = event.get("semantic_context", {})

        # ICT FVG Events (Fair Value Gaps)
        if any(
            term in event_type
            for term in ["fvg_redelivery", "fvg_formation", "fair_value_gap", "fpfvg"]
        ):
            feature.set_semantic_event("fvg_redelivery_flag", 1.0)

        # ICT Market Phase Detection
        phase = semantic_context.get("phase", "").lower()
        if "expansion" in phase or "trending" in event_type or "impulse" in event_type:
            feature.set_semantic_event("expansion_phase_flag", 1.0)
        if "consolidation" in phase or "ranging" in event_type or "sideways" in event_type:
            feature.set_semantic_event("consolidation_flag", 1.0)

        # ICT Retracement Detection
        if any(term in event_type for term in ["retracement", "pullback", "correction", "retrace"]):
            feature.set_semantic_event("retracement_flag", 1.0)

        # ICT Reversal Detection (Market Structure Shift)
        if any(
            term in event_type
            for term in ["reversal", "structure_shift", "mss", "change_of_character"]
        ):
            feature.set_semantic_event("reversal_flag", 1.0)

        # ICT Liquidity Sweep Detection
        if any(
            term in event_type for term in ["liquidity_sweep", "sweep", "liq_sweep", "stop_hunt"]
        ):
            feature.set_semantic_event("liq_sweep_flag", 1.0)

        # ICT Premium/Discount Array Interaction
        if any(term in event_type for term in ["premium", "discount", "pd_array", "equilibrium"]):
            feature.set_semantic_event("pd_array_interaction_flag", 1.0)

        # Additional ICT pattern detection from original_type
        if original_type:
            if any(term in original_type for term in ["fpfvg", "premium", "discount"]):
                feature.set_semantic_event("fvg_redelivery_flag", 1.0)
            if any(term in original_type for term in ["sweep", "liquidity"]):
                feature.set_semantic_event("liq_sweep_flag", 1.0)

    def _calculate_traditional_features(
        self, event: Dict[str, Any], session_context: Dict[str, Any] = None
    ) -> torch.Tensor:
        """Calculate 37D traditional features"""
        from .feature_utils import FeatureUtils

        traditional = torch.zeros(37)

        # Price features
        price = event.get("price", 0.0)
        volume = event.get("volume", 0.0)
        timestamp = event.get("timestamp", 0)

        traditional[0] = float(price)
        traditional[1] = float(volume)
        traditional[2] = float(FeatureUtils.parse_timestamp_to_seconds(timestamp))

        # Extract session context for relative calculations (with fallbacks)
        if session_context is None:
            session_context = {}

        session_open = session_context.get("session_open", price)
        session_high = session_context.get("session_high", price)
        session_low = session_context.get("session_low", price)
        session_range = max(session_high - session_low, 0.01)  # Avoid division by zero
        session_duration = session_context.get("session_duration", 3600)  # Default 1 hour
        session_start_time = session_context.get("session_start_time", 0)

        # PRICE RELATIVITY FEATURES (indices 3-9) - Theory B Foundation
        traditional[3] = (price - session_low) / session_range  # Normalized position (0.0-1.0)
        traditional[4] = (
            ((price - session_open) / session_open) * 100 if session_open > 0 else 0.0
        )  # % from open
        traditional[5] = ((session_high - price) / session_range) * 100  # % from high
        traditional[6] = ((price - session_low) / session_range) * 100  # % from low
        traditional[7] = session_range  # Session volatility measure
        traditional[8] = price / session_open if session_open > 0 else 1.0  # Price to open ratio
        traditional[9] = 1.0 if price > session_open else 0.0  # Above/below open flag

        # TEMPORAL FEATURES (indices 10-15) - Session Character
        timestamp_seconds = FeatureUtils.parse_timestamp_to_seconds(timestamp)
        time_since_open = max(timestamp_seconds - session_start_time, 0)
        traditional[10] = float(time_since_open)  # Seconds from session start
        traditional[11] = min(time_since_open / session_duration, 1.0)  # Normalized time (0-1)
        traditional[12] = float(timestamp_seconds // 3600 % 24)  # Hour of day (0-23)
        traditional[13] = float((timestamp_seconds // 86400) % 7)  # Day of week (0-6)
        traditional[14] = 1.0 if 9 <= (timestamp_seconds // 3600 % 24) <= 16 else 0.0  # Market hours flag
        traditional[15] = FeatureUtils.calculate_session_phase(time_since_open, session_duration)

        # ARCHAEOLOGICAL ZONE FEATURES (indices 16-21) - Theory B Implementation
        traditional[16] = FeatureUtils.calculate_zone_proximity(
            price, session_low, session_range, 0.40, session_context
        )  # 40% zone
        traditional[17] = FeatureUtils.calculate_zone_proximity(
            price, session_low, session_range, 0.618, session_context
        )  # Golden ratio
        traditional[18] = FeatureUtils.calculate_zone_proximity(
            price, session_low, session_range, 0.50, session_context
        )  # Equilibrium
        traditional[19] = FeatureUtils.calculate_zone_proximity(
            price, session_low, session_range, 0.20, session_context
        )  # Discount
        traditional[20] = FeatureUtils.calculate_zone_proximity(
            price, session_low, session_range, 0.80, session_context
        )  # Premium
        traditional[21] = FeatureUtils.calculate_dimensional_relationship(
            price, session_low, session_range, session_context
        )

        # MARKET STRUCTURE FEATURES (indices 22-27) - ICT Concepts
        traditional[22] = FeatureUtils.calculate_premium_discount_position(
            price, session_low, session_range
        )
        traditional[23] = FeatureUtils.estimate_liquidity_proximity(price, session_high, session_low)
        traditional[24] = FeatureUtils.calculate_range_extension(price, session_high, session_low)
        traditional[25] = FeatureUtils.estimate_momentum(event, session_context)
        traditional[26] = FeatureUtils.calculate_volatility_measure(session_range, session_context)
        traditional[27] = FeatureUtils.estimate_volume_profile(price, volume, session_context)

        # ICT ADVANCED FEATURES (indices 28-33) - Enhanced ICT concepts
        traditional[28] = FeatureUtils.calculate_fvg_proximity(price, event, session_context)
        traditional[29] = FeatureUtils.calculate_order_block_strength(price, event, session_context)
        traditional[30] = FeatureUtils.calculate_ict_market_structure_shift(price, event, session_context)
        traditional[31] = 1.0  # HTF range relationship (placeholder)
        traditional[32] = 1.0  # HTF body ratio (placeholder)
        traditional[33] = 1.0  # HTF wick ratio (placeholder)

        # PATTERN COMPLETION FEATURES (indices 34-36)
        traditional[34] = FeatureUtils.estimate_pattern_completion(event, session_context)
        traditional[35] = FeatureUtils.calculate_fractal_similarity(price, session_context)
        traditional[36] = FeatureUtils.calculate_archaeological_significance(event, session_context)

        return traditional
