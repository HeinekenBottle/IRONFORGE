"""
Edge Features Module for Enhanced Graph Builder
20D edge feature vector with semantic relationships and temporal analysis
"""

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


class RichEdgeFeature:
    """20D edge feature vector with semantic relationships"""

    def __init__(self):
        # 20D total: 3 semantic + 17 traditional
        self.features = torch.zeros(20, dtype=torch.float32)

        # Semantic relationship indices (first 3 dimensions)
        self.semantic_indices = {
            "semantic_event_link": 0,
            "event_causality": 1,
            "relationship_type": 2,
        }

        # Traditional features (indices 3-19)
        self.traditional_start = 3

    def set_semantic_relationship(self, rel_type: str, value: float):
        """Set semantic relationship strength"""
        if rel_type in self.semantic_indices:
            self.features[self.semantic_indices[rel_type]] = value


class EdgeFeatureProcessor:
    """Processes event pairs into 20D edge features with ICT relationship analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_edge_feature(
        self, event1: Dict[str, Any], event2: Dict[str, Any]
    ) -> RichEdgeFeature:
        """Create 20D edge feature from event pair"""
        feature = RichEdgeFeature()

        # Set semantic relationships - Enhanced ICT Analysis
        self._detect_semantic_relationships(feature, event1, event2)

        # Generate traditional edge features (17D)
        traditional_features = self._calculate_traditional_edge_features(event1, event2)
        feature.features[feature.traditional_start :] = traditional_features

        return feature

    def _detect_semantic_relationships(
        self, feature: RichEdgeFeature, event1: Dict[str, Any], event2: Dict[str, Any]
    ):
        """Detect and set ICT semantic relationships between events"""
        type1 = event1.get("event_type", "").lower()
        type2 = event2.get("event_type", "").lower()
        original_type1 = event1.get("original_type", "").lower()

        # ICT Causal Relationship Detection
        causality_score = 0.0
        event_link_score = 0.0
        relationship_type_score = 0.0

        # Strong ICT causal relationships
        if "liquidity_sweep" in type1 and any(
            term in type2 for term in ["redelivery", "fvg", "reversal"]
        ):
            causality_score = 0.9  # Sweep → Redelivery/Reversal
            event_link_score = 1.0
            relationship_type_score = 0.8  # Strong causal
        elif "fvg_formation" in type1 and "fvg_redelivery" in type2:
            causality_score = 0.95  # FVG Formation → Redelivery
            event_link_score = 1.0
            relationship_type_score = 0.9  # Very strong causal
        elif any(term in type1 for term in ["sweep", "liquidity"]) and "structure" in type2:
            causality_score = 0.8  # Sweep → Structure change
            event_link_score = 0.9
            relationship_type_score = 0.7

        # ICT Premium/Discount Array relationships
        if any(term in type1 for term in ["premium", "discount"]) and any(
            term in type2 for term in ["premium", "discount"]
        ):
            event_link_score = 0.7  # PD Array interactions
            relationship_type_score = 0.6

        # Market structure continuity
        if "expansion" in type1 and "consolidation" in type2:
            causality_score = 0.7  # Natural market flow
            event_link_score = 0.8
            relationship_type_score = 0.5
        elif "consolidation" in type1 and any(term in type2 for term in ["expansion", "breakout"]):
            causality_score = 0.75  # Consolidation → Expansion
            event_link_score = 0.8
            relationship_type_score = 0.6

        # Order block and liquidity relationships
        if any(term in original_type1 for term in ["fpfvg", "premium"]) and "sweep" in type2:
            causality_score = 0.6  # FPFVG → Sweep
            event_link_score = 0.7

        # Set the semantic relationships
        feature.set_semantic_relationship("event_causality", causality_score)
        feature.set_semantic_relationship("semantic_event_link", event_link_score)
        feature.set_semantic_relationship("relationship_type", relationship_type_score)

    def _calculate_traditional_edge_features(
        self, event1: Dict[str, Any], event2: Dict[str, Any]
    ) -> torch.Tensor:
        """Calculate 17D traditional edge features"""
        from .feature_utils import FeatureUtils

        traditional = torch.zeros(17)

        # Temporal distance
        time1 = FeatureUtils.parse_timestamp_to_seconds(event1.get("timestamp", 0))
        time2 = FeatureUtils.parse_timestamp_to_seconds(event2.get("timestamp", 0))
        traditional[0] = abs(time2 - time1)

        # Price distance
        price1 = event1.get("price", 0.0)
        price2 = event2.get("price", 0.0)
        traditional[1] = abs(price2 - price1)

        # Price direction and momentum
        price_direction = 1.0 if price2 > price1 else -1.0 if price2 < price1 else 0.0
        traditional[2] = price_direction

        # Normalized temporal distance (0-1 over typical session duration)
        traditional[3] = min(traditional[0] / 3600.0, 1.0)  # Normalize to hours

        # Price momentum strength
        traditional[4] = abs(price2 - price1) / max(abs(price1), 1.0) if price1 != 0 else 0.0

        # Temporal coherence (events close in time = higher coherence)
        temporal_coherence = max(0.0, 1.0 - (traditional[0] / 1800.0))  # 30min window
        traditional[5] = temporal_coherence

        # Event type relationship strength
        type1 = event1.get("event_type", "").lower()
        type2 = event2.get("event_type", "").lower()
        traditional[6] = self._calculate_event_relationship_strength(type1, type2)

        # Session phase consistency
        traditional[7] = 1.0 if abs(time1 % 3600 - time2 % 3600) < 1800 else 0.0  # Same hour

        # Price level significance
        traditional[8] = self._calculate_price_level_significance(price1, price2)

        # Structural relationship (simplified)
        traditional[9] = 0.5 if "structure" in type1 or "structure" in type2 else 0.3

        # Archaeological zone relationship
        traditional[10] = self._calculate_zone_relationship(price1, price2)

        # Liquidity flow direction
        traditional[11] = 1.0 if "sweep" in type1 and "redelivery" in type2 else 0.5

        # Pattern completion link
        traditional[12] = 0.8 if "expansion" in type1 and "consolidation" in type2 else 0.4

        # Multi-timeframe consistency
        traditional[13] = 0.7  # Placeholder for HTF consistency

        # Session character preservation
        traditional[14] = 0.6  # Placeholder for session character

        # Fractal relationship
        traditional[15] = 0.5  # Placeholder for fractal similarity

        # Overall archaeological significance
        traditional[16] = min(1.0, (traditional[6] + traditional[10]) / 2.0)

        return traditional

    def _calculate_event_relationship_strength(self, type1: str, type2: str) -> float:
        """Calculate relationship strength between event types"""
        # Strong relationships
        strong_pairs = [
            ("liquidity_sweep", "redelivery"),
            ("fvg_formation", "fvg_redelivery"),
            ("expansion", "consolidation"),
            ("premium", "discount"),
            ("structure_shift", "reversal"),
        ]

        # Check for strong relationships
        for pair in strong_pairs:
            if (pair[0] in type1 and pair[1] in type2) or (pair[1] in type1 and pair[0] in type2):
                return 0.9

        # Medium relationships
        if any(term in type1 for term in ["sweep", "liquidity"]) and any(
            term in type2 for term in ["structure", "reversal"]
        ):
            return 0.7

        # Weak relationships
        if any(term in type1 for term in ["expansion", "consolidation"]) and any(
            term in type2 for term in ["expansion", "consolidation"]
        ):
            return 0.5

        return 0.3  # Default weak relationship

    def _calculate_price_level_significance(self, price1: float, price2: float) -> float:
        """Calculate significance of price levels"""
        # Simple implementation - could be enhanced with support/resistance levels
        price_diff = abs(price2 - price1)
        avg_price = (price1 + price2) / 2.0

        if avg_price == 0:
            return 0.0

        # Normalize by average price
        normalized_diff = price_diff / avg_price

        # Higher significance for larger price moves
        return min(1.0, normalized_diff * 10.0)

    def _calculate_zone_relationship(self, price1: float, price2: float) -> float:
        """Calculate archaeological zone relationship between prices"""
        # Simple zone relationship calculation
        # In a full implementation, this would use session context
        price_ratio = price2 / price1 if price1 != 0 else 1.0

        # Check if prices are in similar zones (simplified)
        if 0.95 <= price_ratio <= 1.05:
            return 0.9  # Same zone
        elif 0.90 <= price_ratio <= 1.10:
            return 0.7  # Adjacent zones
        else:
            return 0.3  # Distant zones
