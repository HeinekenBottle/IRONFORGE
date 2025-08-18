"""
Enhanced Graph Builder for 45D/20D architecture
Archaeological discovery graph construction with semantic features
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
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


class EnhancedGraphBuilder:
    """
    Enhanced graph builder for archaeological discovery
    Transforms JSON session data into rich 45D/20D graph representations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Graph Builder initialized")

    def build_session_graph(self, session_data: Dict[str, Any]) -> nx.Graph:
        """
        Build enhanced graph from session JSON data

        Args:
            session_data: Session JSON with events and metadata

        Returns:
            NetworkX graph with rich 45D/20D features
        """
        try:
            graph = nx.Graph()

            # Extract session metadata
            session_name = session_data.get("session_name", "unknown")
            events = session_data.get("events", [])

            # Extract session context for calculations
            session_context = self._extract_session_context(session_data, events)

            self.logger.info(f"Building graph for session {session_name} with {len(events)} events")

            # Add nodes with rich features
            for i, event in enumerate(events):
                node_feature = self._create_node_feature(event, session_context)
                graph.add_node(
                    i, feature=node_feature.features, raw_data=event, session_name=session_name
                )

            # Add edges with rich features
            for i in range(len(events)):
                for j in range(i + 1, min(i + 5, len(events))):  # Connect to next 4 events
                    edge_feature = self._create_edge_feature(events[i], events[j])
                    graph.add_edge(i, j, feature=edge_feature.features, temporal_distance=j - i)

            self.logger.info(
                f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            return graph

        except Exception as e:
            self.logger.error(f"Failed to build graph: {e}")
            raise

    def _create_node_feature(
        self, event: Dict[str, Any], session_context: Dict[str, Any] = None
    ) -> RichNodeFeature:
        """Create 45D node feature from event data"""
        feature = RichNodeFeature()

        # Set semantic events based on event data - Enhanced ICT Detection
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

        # Generate traditional features (37D)
        traditional = torch.zeros(37)

        # Price features
        price = event.get("price", 0.0)
        volume = event.get("volume", 0.0)
        timestamp = event.get("timestamp", 0)

        traditional[0] = float(price)
        traditional[1] = float(volume)
        traditional[2] = float(timestamp % 86400)  # Time of day

        # REPLACE SYNTHETIC FEATURES WITH REAL MARKET CALCULATIONS

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
        time_since_open = max(timestamp - session_start_time, 0)
        traditional[10] = float(time_since_open)  # Seconds from session start
        traditional[11] = min(time_since_open / session_duration, 1.0)  # Normalized time (0-1)
        traditional[12] = float(timestamp // 3600 % 24)  # Hour of day (0-23)
        traditional[13] = float((timestamp // 86400) % 7)  # Day of week (0-6)
        traditional[14] = 1.0 if 9 <= (timestamp // 3600 % 24) <= 16 else 0.0  # Market hours flag
        traditional[15] = self._calculate_session_phase(time_since_open, session_duration)

        # ARCHAEOLOGICAL ZONE FEATURES (indices 16-21) - Theory B Implementation
        traditional[16] = self._calculate_zone_proximity(
            price, session_low, session_range, 0.40, session_context
        )  # 40% zone
        traditional[17] = self._calculate_zone_proximity(
            price, session_low, session_range, 0.618, session_context
        )  # Golden ratio
        traditional[18] = self._calculate_zone_proximity(
            price, session_low, session_range, 0.50, session_context
        )  # Equilibrium
        traditional[19] = self._calculate_zone_proximity(
            price, session_low, session_range, 0.20, session_context
        )  # Discount
        traditional[20] = self._calculate_zone_proximity(
            price, session_low, session_range, 0.80, session_context
        )  # Premium
        traditional[21] = self._calculate_dimensional_relationship(
            price, session_low, session_range, session_context
        )

        # MARKET STRUCTURE FEATURES (indices 22-27) - ICT Concepts
        traditional[22] = self._calculate_premium_discount_position(
            price, session_low, session_range
        )
        traditional[23] = self._estimate_liquidity_proximity(price, session_high, session_low)
        traditional[24] = self._calculate_range_extension(price, session_high, session_low)
        traditional[25] = self._estimate_momentum(event, session_context)
        traditional[26] = self._calculate_volatility_measure(session_range, session_context)
        traditional[27] = self._estimate_volume_profile(price, volume, session_context)

        # ICT ADVANCED FEATURES (indices 28-33) - Enhanced ICT concepts
        traditional[28] = self._calculate_fvg_proximity(price, event, session_context)
        traditional[29] = self._calculate_order_block_strength(price, event, session_context)
        traditional[30] = self._calculate_ict_market_structure_shift(price, event, session_context)
        traditional[31] = 1.0  # HTF range relationship (placeholder)
        traditional[32] = 1.0  # HTF body ratio (placeholder)
        traditional[33] = 1.0  # HTF wick ratio (placeholder)

        # PATTERN COMPLETION FEATURES (indices 34-36)
        traditional[34] = self._estimate_pattern_completion(event, session_context)
        traditional[35] = self._calculate_fractal_similarity(price, session_context)
        traditional[36] = self._calculate_archaeological_significance(event, session_context)

        feature.set_traditional_features(traditional)
        return feature

    def _create_edge_feature(
        self, event1: Dict[str, Any], event2: Dict[str, Any]
    ) -> RichEdgeFeature:
        """Create 20D edge feature from event pair"""
        feature = RichEdgeFeature()

        # Set semantic relationships - Enhanced ICT Analysis
        type1 = event1.get("event_type", "").lower()
        type2 = event2.get("event_type", "").lower()
        original_type1 = event1.get("original_type", "").lower()
        event2.get("original_type", "").lower()

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

        # Generate traditional edge features (17D)
        traditional = torch.zeros(17)

        # Temporal distance
        time1 = event1.get("timestamp", 0)
        time2 = event2.get("timestamp", 0)
        traditional[0] = abs(time2 - time1)

        # Price distance
        price1 = event1.get("price", 0.0)
        price2 = event2.get("price", 0.0)
        traditional[1] = abs(price2 - price1)

        # REPLACE SYNTHETIC EDGE FEATURES WITH REAL CALCULATIONS

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

        feature.features[feature.traditional_start :] = traditional
        return feature

    # HELPER METHODS FOR REAL MARKET CALCULATIONS

    def _calculate_session_phase(self, time_since_open: float, session_duration: float) -> float:
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

    def _calculate_zone_proximity(
        self,
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
                    self.logger.debug(
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

    def _calculate_dimensional_relationship(
        self,
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
            self.logger.debug(
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
                self.logger.info(
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

    def _calculate_premium_discount_position(
        self, price: float, session_low: float, session_range: float
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

    def _estimate_liquidity_proximity(
        self, price: float, session_high: float, session_low: float
    ) -> float:
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

    def _calculate_range_extension(
        self, price: float, session_high: float, session_low: float
    ) -> float:
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

    def _estimate_momentum(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Estimate price momentum (simplified)"""
        # This would normally require multiple price points
        # For now, use event significance as proxy
        return event.get("significance", 0.5)

    def _calculate_volatility_measure(
        self, session_range: float, session_context: Dict[str, Any]
    ) -> float:
        """Calculate session volatility measure"""
        # Normalize range by typical market volatility
        typical_range = session_context.get("typical_range", session_range)

        if typical_range > 0:
            return session_range / typical_range
        else:
            return 1.0

    def _estimate_volume_profile(
        self, price: float, volume: float, session_context: Dict[str, Any]
    ) -> float:
        """Estimate volume profile at price level"""
        # Simplified volume profile
        avg_volume = session_context.get("average_volume", 1.0)

        if avg_volume > 0:
            return volume / avg_volume
        else:
            return 1.0

    def _estimate_pattern_completion(
        self, event: Dict[str, Any], session_context: Dict[str, Any]
    ) -> float:
        """Estimate pattern completion probability"""
        # Based on event type and session context
        event_type = event.get("event_type", "")

        if "redelivery" in event_type:
            return 0.8  # High completion probability
        elif "sweep" in event_type:
            return 0.6  # Medium completion probability
        else:
            return 0.3  # Low completion probability

    def _calculate_fractal_similarity(self, price: float, session_context: Dict[str, Any]) -> float:
        """Calculate fractal similarity score"""
        # Simplified fractal calculation
        # Would normally compare patterns across timeframes
        return 0.5  # Placeholder

    def _calculate_archaeological_significance(
        self, event: Dict[str, Any], session_context: Dict[str, Any]
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

    def _calculate_event_relationship_strength(self, type1: str, type2: str) -> float:
        """Calculate relationship strength between two event types - Enhanced ICT"""
        # Enhanced ICT relationship matrix
        ict_relationships = {
            # Liquidity Sweep relationships
            ("liquidity_sweep", "fvg_redelivery"): 0.95,
            ("liquidity_sweep", "reversal"): 0.9,
            ("liquidity_sweep", "structure_shift"): 0.9,
            ("sweep", "fvg_redelivery"): 0.9,
            ("liq_sweep", "reversal"): 0.85,
            # FVG relationships
            ("fvg_formation", "fvg_redelivery"): 0.95,
            ("fvg_redelivery", "reversal"): 0.8,
            ("fpfvg", "premium"): 0.85,
            ("fpfvg", "discount"): 0.85,
            # Market structure relationships
            ("expansion_phase", "consolidation"): 0.8,
            ("consolidation", "expansion_phase"): 0.75,
            ("expansion", "retracement"): 0.7,
            ("retracement", "reversal"): 0.8,
            # Premium/Discount Array relationships
            ("premium", "discount"): 0.6,
            ("premium", "equilibrium"): 0.7,
            ("discount", "equilibrium"): 0.7,
            ("pd_array", "liquidity_sweep"): 0.8,
            # Order Block relationships
            ("order_block", "liquidity_sweep"): 0.85,
            ("order_block", "fvg_formation"): 0.8,
            ("structure", "liquidity"): 0.75,
            # Market Structure Shift relationships
            ("mss", "liquidity_sweep"): 0.9,
            ("change_of_character", "reversal"): 0.85,
            ("structure_shift", "fvg_redelivery"): 0.8,
        }

        # Check both directions and partial matches
        type1_lower = type1.lower()
        type2_lower = type2.lower()

        # Direct matches
        key1 = (type1_lower, type2_lower)
        key2 = (type2_lower, type1_lower)

        if key1 in ict_relationships:
            return ict_relationships[key1]
        elif key2 in ict_relationships:
            return ict_relationships[key2]

        # Partial term matching for ICT concepts
        for (term1, term2), strength in ict_relationships.items():
            if (term1 in type1_lower and term2 in type2_lower) or (
                term1 in type2_lower and term2 in type1_lower
            ):
                return strength * 0.9  # Slight discount for partial match

        # Enhanced categorization
        ict_structural = [
            "liquidity_sweep",
            "fvg_redelivery",
            "expansion_phase",
            "fpfvg",
            "premium",
            "discount",
            "order_block",
        ]
        ict_temporal = ["consolidation", "retracement", "reversal", "mss", "structure_shift"]
        ict_flow = ["sweep", "liquidity", "momentum", "impulse"]

        # Category-based relationships
        if any(term in type1_lower for term in ict_structural) and any(
            term in type2_lower for term in ict_structural
        ):
            return 0.7  # ICT structural events relate strongly
        elif any(term in type1_lower for term in ict_temporal) and any(
            term in type2_lower for term in ict_temporal
        ):
            return 0.6  # ICT temporal events relate moderately
        elif any(term in type1_lower for term in ict_flow) and any(
            term in type2_lower for term in ict_flow
        ):
            return 0.65  # ICT flow events relate well
        elif (
            any(term in type1_lower for term in ict_structural)
            and any(term in type2_lower for term in ict_temporal)
        ) or (
            any(term in type1_lower for term in ict_temporal)
            and any(term in type2_lower for term in ict_structural)
        ):
            return 0.75  # Cross-category ICT relationships are important
        else:
            return 0.3  # Default weak relationship

    def _calculate_price_level_significance(self, price1: float, price2: float) -> float:
        """Calculate significance of price level relationship"""
        if price1 <= 0 or price2 <= 0:
            return 0.0

        # Calculate price difference as percentage
        price_diff_pct = abs(price2 - price1) / max(price1, price2) * 100

        # Significance inversely related to distance (closer = more significant)
        if price_diff_pct < 0.1:  # Very close prices
            return 1.0
        elif price_diff_pct < 0.5:  # Close prices
            return 0.8
        elif price_diff_pct < 1.0:  # Moderately close
            return 0.6
        elif price_diff_pct < 2.0:  # Somewhat related
            return 0.4
        else:  # Distant prices
            return max(0.1, 1.0 - (price_diff_pct / 10.0))

    def _calculate_zone_relationship(self, price1: float, price2: float) -> float:
        """Calculate archaeological zone relationship between two prices"""
        if price1 <= 0 or price2 <= 0:
            return 0.0

        # Simplified zone relationship - in real implementation would use session context
        # For now, calculate if prices are in similar zone (e.g., both premium, both discount)

        # Use a simplified range assumption for zone calculation
        range_estimate = max(price1, price2) * 0.02  # 2% range estimate
        low_estimate = min(price1, price2) - range_estimate / 2
        max(price1, price2) + range_estimate / 2

        # Calculate normalized positions
        pos1 = (price1 - low_estimate) / range_estimate if range_estimate > 0 else 0.5
        pos2 = (price2 - low_estimate) / range_estimate if range_estimate > 0 else 0.5

        # Determine zones (simplified)
        def get_zone(pos):
            if pos < 0.4:
                return "discount"
            elif pos < 0.6:
                return "equilibrium"
            else:
                return "premium"

        zone1 = get_zone(pos1)
        zone2 = get_zone(pos2)

        # Same zone = strong relationship
        if zone1 == zone2:
            return 0.8
        # Adjacent zones = moderate relationship
        elif (
            (zone1 == "discount" and zone2 == "equilibrium")
            or (zone1 == "equilibrium" and zone2 == "premium")
            or (zone2 == "discount" and zone1 == "equilibrium")
            or (zone2 == "equilibrium" and zone1 == "premium")
        ):
            return 0.5
        else:
            return 0.2  # Opposite zones = weak relationship

    def _extract_session_context(
        self, session_data: Dict[str, Any], events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract session context for feature calculations"""
        context = {}

        # THEORY B IMPLEMENTATION: Extract FINAL session values for dimensional relationships
        # Check relativity_stats first (has final completed session data)
        relativity_stats = session_data.get("relativity_stats", {})

        if relativity_stats:
            # Use final session values for Theory B dimensional calculations
            context["session_open"] = relativity_stats.get("session_open", 0.0)
            context["session_high"] = relativity_stats.get("session_high", 0.0)
            context["session_low"] = relativity_stats.get("session_low", 0.0)
            context["session_close"] = relativity_stats.get("session_close", 0.0)
            context["session_range"] = relativity_stats.get("session_range", 0.0)
            context["session_duration"] = relativity_stats.get("session_duration_seconds", 3600)
            context["theory_b_final_range"] = True  # Flag indicating we have final values
        else:
            # Fallback to session metadata
            context["session_open"] = session_data.get("session_open", 0.0)
            context["session_high"] = session_data.get("session_high", 0.0)
            context["session_low"] = session_data.get("session_low", 0.0)
            context["session_duration"] = session_data.get("session_duration", 3600)
            context["theory_b_final_range"] = False

        # If still no values, calculate from events (least preferred for Theory B)
        if events and (context["session_high"] == 0.0 or context["session_low"] == 0.0):
            prices = [event.get("price", 0.0) for event in events if event.get("price", 0.0) > 0]

            if prices:
                if context["session_high"] == 0.0:
                    context["session_high"] = max(prices)
                if context["session_low"] == 0.0:
                    context["session_low"] = min(prices)
                if context["session_open"] == 0.0:
                    context["session_open"] = prices[0]
                context["theory_b_final_range"] = False

        # Ensure session_range is calculated
        if context.get("session_range", 0.0) == 0.0:
            context["session_range"] = max(context["session_high"] - context["session_low"], 0.01)

        # Calculate additional context
        context["typical_range"] = session_data.get("typical_range", context["session_range"])
        context["average_volume"] = session_data.get("average_volume", 1.0)
        context["session_start_time"] = session_data.get("session_start_time", 0)

        return context

    def _calculate_fvg_proximity(
        self, price: float, event: Dict[str, Any], session_context: Dict[str, Any]
    ) -> float:
        """Calculate proximity to Fair Value Gaps (ICT concept)"""
        event_type = event.get("type", "").lower()

        # Check if this event is related to FVG
        fvg_score = 0.0

        if "fvg" in event_type or "fair_value_gap" in event_type:
            # Direct FVG event
            fvg_score = 1.0
        elif "redelivery" in event_type:
            # FVG redelivery event (ICT concept)
            fvg_score = 0.9
        elif event.get("archaeological_significance", 0.0) > 0.7:
            # High significance events may be near FVGs
            fvg_score = 0.6

        # Apply ICT logic: FVGs in premium/discount arrays are more significant
        session_range = session_context.get("session_range", 1.0)
        session_low = session_context.get("session_low", price)

        if session_range > 0:
            position = (price - session_low) / session_range
            # FVGs in extreme zones (premium/discount) are more significant
            if position < 0.3 or position > 0.7:
                fvg_score *= 1.2  # Boost for extreme zones

        return min(1.0, fvg_score)

    def _calculate_order_block_strength(
        self, price: float, event: Dict[str, Any], session_context: Dict[str, Any]
    ) -> float:
        """Calculate ICT Order Block strength"""
        session_range = session_context.get("session_range", 1.0)
        session_low = session_context.get("session_low", price)

        if session_range <= 0:
            return 0.0

        position = (price - session_low) / session_range
        event_type = event.get("type", "").lower()

        # ICT Order Block characteristics
        order_block_score = 0.0

        # Check for order block indicators
        if "liquidity" in event_type or "sweep" in event_type:
            order_block_score = 0.8  # High potential for order block
        elif "reversal" in event_type or "structure" in event_type:
            order_block_score = 0.7  # Good potential
        elif event.get("archaeological_significance", 0.0) > 0.8:
            order_block_score = 0.6  # Moderate potential

        # ICT: Order blocks are more significant at key levels
        key_levels = [0.20, 0.25, 0.50, 0.75, 0.80]
        min_distance = min(abs(position - level) for level in key_levels)

        if min_distance < 0.05:  # Within 5% of key level
            order_block_score *= 1.3  # Boost for key levels

        # Time-based decay for order block strength
        timestamp = event.get("timestamp", "00:00:00")
        if timestamp and ":" in timestamp:
            try:
                hour = int(timestamp.split(":")[0])
                # ICT: Order blocks during key sessions are stronger
                if 9 <= hour <= 16:  # Market hours
                    order_block_score *= 1.1
                elif 13 <= hour <= 16:  # PM session (often stronger)
                    order_block_score *= 1.2
            except (ValueError, IndexError):
                pass

        return min(1.0, order_block_score)

    def _calculate_ict_market_structure_shift(
        self, price: float, event: Dict[str, Any], session_context: Dict[str, Any]
    ) -> float:
        """Calculate ICT Market Structure Shift (MSS) potential"""
        event_type = event.get("type", "").lower()
        session_range = session_context.get("session_range", 1.0)
        session_low = session_context.get("session_low", price)
        session_high = session_context.get("session_high", price)

        mss_score = 0.0

        # ICT MSS indicators
        if "reversal" in event_type or "structure" in event_type:
            mss_score = 0.8
        elif "sweep" in event_type and "liquidity" in event_type:
            mss_score = 0.9  # Liquidity sweeps often precede MSS
        elif event.get("archaeological_significance", 0.0) > 0.8:
            mss_score = 0.6

        # Position-based MSS potential
        if session_range > 0:
            position = (price - session_low) / session_range

            # MSS more likely at extremes after liquidity sweeps
            if position > 0.9:  # Near session high
                if price > session_high:  # Breaking high
                    mss_score *= 1.4  # Strong MSS potential
            elif position < 0.1:  # Near session low
                if price < session_low:  # Breaking low
                    mss_score *= 1.4  # Strong MSS potential

        return min(1.0, mss_score)

    def validate_theory_b_implementation(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Theory B implementation for a session

        Returns:
            Validation results showing Theory B compliance
        """
        try:
            session_context = self._extract_session_context(
                session_data, session_data.get("events", [])
            )
            events = session_data.get("events", [])

            if not events:
                return {"error": "No events to validate"}

            validation_results = {
                "session_name": session_data.get("session_name", "unknown"),
                "theory_b_active": session_context.get("theory_b_final_range", False),
                "final_range_available": session_context.get("session_range", 0.0) > 0,
                "session_high": session_context.get("session_high", 0.0),
                "session_low": session_context.get("session_low", 0.0),
                "session_range": session_context.get("session_range", 0.0),
                "forty_percent_events": [],
                "theory_b_violations": [],
                "total_events": len(events),
            }

            # Calculate 40% zone boundaries
            session_low = session_context.get("session_low", 0.0)
            session_range = session_context.get("session_range", 0.0)

            if session_range > 0:
                forty_percent_level = session_low + (session_range * 0.40)
                forty_percent_tolerance = session_range * 0.02  # 2% tolerance

                validation_results["forty_percent_level"] = forty_percent_level
                validation_results["forty_percent_tolerance"] = forty_percent_tolerance

                # Check events for Theory B compliance
                for i, event in enumerate(events):
                    price = event.get("price", 0.0)
                    if price <= 0:
                        continue

                    # Check if event is near 40% zone
                    distance_from_40pct = abs(price - forty_percent_level)
                    if distance_from_40pct <= forty_percent_tolerance:
                        validation_results["forty_percent_events"].append(
                            {
                                "event_index": i,
                                "timestamp": event.get("timestamp", "unknown"),
                                "price": price,
                                "distance_from_40pct": distance_from_40pct,
                                "distance_points": distance_from_40pct,
                                "position_in_range": (price - session_low) / session_range,
                                "event_type": event.get("type", "unknown"),
                            }
                        )

                # Theory B validation summary
                validation_results["forty_percent_event_count"] = len(
                    validation_results["forty_percent_events"]
                )
                validation_results["theory_b_compliance_rate"] = (
                    len(validation_results["forty_percent_events"]) / len(events) * 100
                    if events
                    else 0.0
                )

            return validation_results

        except Exception as e:
            return {"error": f"Theory B validation failed: {e}"}

    def extract_features_for_tgat(self, graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract feature tensors for TGAT processing

        Returns:
            node_features: [N, 45] tensor
            edge_features: [E, 20] tensor
        """
        nodes = list(graph.nodes())
        edges = list(graph.edges())

        # Extract node features
        node_features = torch.stack([graph.nodes[node]["feature"] for node in nodes])

        # Extract edge features
        edge_features = torch.stack([graph.edges[edge]["feature"] for edge in edges])

        self.logger.info(
            f"Extracted features: nodes {node_features.shape}, edges {edge_features.shape}"
        )
        return node_features, edge_features
