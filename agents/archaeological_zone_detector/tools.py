"""
Archaeological Zone Analysis Tools
==================================

Core analytical tools for archaeological zone detection and temporal non-locality analysis.
Implements the mathematical foundations of archaeological intelligence within IRONFORGE.

Key Components:
- ZoneAnalyzer: 40% dimensional anchor detection and analysis
- TemporalNonLocalityValidator: Theory B forward positioning validation
- DimensionalAnchorCalculator: Archaeological constant calculations
- TheoryBValidator: Forward-looking temporal coherence validation

Mathematical Foundation:
- Dimensional anchors: anchor_zone = previous_day_range * 0.40
- Temporal non-locality: events position relative to FINAL range completion
- Precision target: 7.55-point accuracy to eventual completion
- Theory B: Information propagates through forward temporal echoes
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import euclidean_distances
import torch

from .ironforge_config import ArchaeologicalConfig

logger = logging.getLogger(__name__)


class RangeCalculationMethod(Enum):
    """Methods for calculating session range for dimensional anchoring"""
    HIGH_LOW = "high_low"
    OPEN_CLOSE = "open_close" 
    BODY_RANGE = "body_range"
    WEIGHTED_AVERAGE = "weighted_average"


@dataclass
class ArchaeologicalZoneData:
    """Internal data structure for zone calculations"""
    anchor_point: float
    zone_range: Tuple[float, float]
    confidence: float
    session_id: str
    discovery_timestamp: float
    previous_range: float
    zone_width: float
    precision_score: float
    theory_b_valid: bool = False
    temporal_offset: float = 0.0
    forward_positioning: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.forward_positioning is None:
            self.forward_positioning = {}


@dataclass  
class TemporalEcho:
    """Temporal echo data for non-locality analysis"""
    echo_id: str
    source_event: Dict[str, Any]
    target_event: Dict[str, Any]
    propagation_strength: float
    temporal_distance: int
    coherence_score: float
    forward_validation: bool


@dataclass
class TheoryBResult:
    """Theory B forward positioning validation result"""
    zone_id: str
    forward_positioning_valid: bool
    completion_accuracy: float
    intermediate_filtering_score: float
    final_state_alignment: float
    precision_score: float
    temporal_coherence: float


class DimensionalAnchorCalculator:
    """
    Calculates 40% dimensional anchor points from previous session ranges
    
    Implements the archaeological constant discovery: anchor_zone = previous_range * 0.40
    Maintains session isolation while providing dimensional reference points for
    temporal non-locality analysis.
    """
    
    def __init__(self, config: ArchaeologicalConfig):
        self.config = config
        self.anchor_config = config.dimensional_anchor
        logger.debug("DimensionalAnchorCalculator initialized")
    
    def calculate_session_range(
        self, 
        session_data: pd.DataFrame,
        method: Optional[RangeCalculationMethod] = None
    ) -> float:
        """
        Calculate session range for dimensional anchor calculation
        
        Args:
            session_data: Session price/event data
            method: Range calculation method (defaults to config setting)
            
        Returns:
            Session range value for 40% anchor calculation
        """
        if method is None:
            method = RangeCalculationMethod(self.anchor_config.range_calculation_method)
        
        try:
            if method == RangeCalculationMethod.HIGH_LOW:
                return self._calculate_high_low_range(session_data)
            elif method == RangeCalculationMethod.OPEN_CLOSE:
                return self._calculate_open_close_range(session_data)
            elif method == RangeCalculationMethod.BODY_RANGE:
                return self._calculate_body_range(session_data)
            elif method == RangeCalculationMethod.WEIGHTED_AVERAGE:
                return self._calculate_weighted_average_range(session_data)
            else:
                logger.warning(f"Unknown range calculation method: {method}, using HIGH_LOW")
                return self._calculate_high_low_range(session_data)
        
        except Exception as e:
            logger.error(f"Range calculation failed: {e}")
            return 0.0
    
    def calculate_dimensional_anchors(
        self,
        previous_range: float,
        current_session_data: pd.DataFrame,
        anchor_percentage: float = 0.40
    ) -> List[Dict[str, Any]]:
        """
        Calculate 40% dimensional anchor points for archaeological zone detection
        
        Args:
            previous_range: Range from previous session for anchor calculation
            current_session_data: Current session data for zone positioning
            anchor_percentage: Archaeological constant (default: 0.40)
            
        Returns:
            List of dimensional anchor zone dictionaries
        """
        if previous_range <= 0:
            logger.warning("Invalid previous range for anchor calculation")
            return []
        
        # Calculate anchor zone width (40% of previous range)
        anchor_zone_width = previous_range * anchor_percentage
        
        # Extract session price information
        session_id = self._extract_session_id(current_session_data)
        session_high, session_low = self._extract_session_high_low(current_session_data)
        
        if session_high <= session_low:
            logger.warning(f"Invalid session high/low for {session_id}")
            return []
        
        # Generate anchor points within session range
        anchor_zones = []
        
        # Primary anchor: Session midpoint
        session_mid = (session_high + session_low) / 2.0
        primary_anchor = self._create_anchor_zone(
            anchor_point=session_mid,
            zone_width=anchor_zone_width,
            previous_range=previous_range,
            session_id=session_id,
            anchor_type="primary_midpoint"
        )
        anchor_zones.append(primary_anchor)
        
        # Secondary anchors: Key levels within session
        # Upper anchor (75th percentile)
        upper_anchor_point = session_low + (session_high - session_low) * 0.75
        upper_anchor = self._create_anchor_zone(
            anchor_point=upper_anchor_point,
            zone_width=anchor_zone_width * 0.8,  # Slightly smaller
            previous_range=previous_range,
            session_id=session_id,
            anchor_type="secondary_upper"
        )
        anchor_zones.append(upper_anchor)
        
        # Lower anchor (25th percentile) 
        lower_anchor_point = session_low + (session_high - session_low) * 0.25
        lower_anchor = self._create_anchor_zone(
            anchor_point=lower_anchor_point,
            zone_width=anchor_zone_width * 0.8,
            previous_range=previous_range,
            session_id=session_id,
            anchor_type="secondary_lower"
        )
        anchor_zones.append(lower_anchor)
        
        # Filter anchors by configuration constraints
        filtered_anchors = self._filter_anchor_zones(anchor_zones)
        
        logger.debug(f"Generated {len(filtered_anchors)} dimensional anchors for {session_id}")
        return filtered_anchors
    
    def _calculate_high_low_range(self, session_data: pd.DataFrame) -> float:
        """Calculate range using session high and low"""
        high_col = self._find_price_column(session_data, ["high", "High", "HIGH"])
        low_col = self._find_price_column(session_data, ["low", "Low", "LOW"])
        
        if high_col and low_col:
            session_high = session_data[high_col].max()
            session_low = session_data[low_col].min()
            return max(0.0, session_high - session_low)
        
        # Fallback: use any price column
        price_col = self._find_price_column(session_data, ["price", "close", "Close"])
        if price_col:
            return session_data[price_col].max() - session_data[price_col].min()
        
        return 0.0
    
    def _calculate_open_close_range(self, session_data: pd.DataFrame) -> float:
        """Calculate range using session open and close"""
        open_col = self._find_price_column(session_data, ["open", "Open", "OPEN"])
        close_col = self._find_price_column(session_data, ["close", "Close", "CLOSE"])
        
        if open_col and close_col:
            session_open = session_data[open_col].iloc[0]
            session_close = session_data[close_col].iloc[-1]
            return abs(session_close - session_open)
        
        return 0.0
    
    def _calculate_body_range(self, session_data: pd.DataFrame) -> float:
        """Calculate range using candlestick body (open-close range)"""
        open_col = self._find_price_column(session_data, ["open", "Open", "OPEN"])
        close_col = self._find_price_column(session_data, ["close", "Close", "CLOSE"])
        
        if open_col and close_col:
            body_ranges = abs(session_data[close_col] - session_data[open_col])
            return body_ranges.mean()
        
        return 0.0
    
    def _calculate_weighted_average_range(self, session_data: pd.DataFrame) -> float:
        """Calculate weighted average of different range methods"""
        high_low_range = self._calculate_high_low_range(session_data)
        open_close_range = self._calculate_open_close_range(session_data)
        body_range = self._calculate_body_range(session_data)
        
        # Weighted average: High-Low (60%), Open-Close (25%), Body (15%)
        weighted_range = (
            high_low_range * 0.60 +
            open_close_range * 0.25 + 
            body_range * 0.15
        )
        
        return weighted_range
    
    def _find_price_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find price column from candidate names"""
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None
    
    def _extract_session_id(self, session_data: pd.DataFrame) -> str:
        """Extract session identifier"""
        if 'session_id' in session_data.columns:
            return str(session_data['session_id'].iloc[0])
        elif 'session_date' in session_data.columns:
            return f"session_{session_data['session_date'].iloc[0]}"
        else:
            return f"session_{hash(str(session_data.iloc[0].values))}"[:16]
    
    def _extract_session_high_low(self, session_data: pd.DataFrame) -> Tuple[float, float]:
        """Extract session high and low prices"""
        high_col = self._find_price_column(session_data, ["high", "High", "HIGH"])
        low_col = self._find_price_column(session_data, ["low", "Low", "LOW"])
        
        if high_col and low_col:
            return session_data[high_col].max(), session_data[low_col].min()
        
        # Fallback: use any price column
        price_col = self._find_price_column(session_data, ["price", "close", "Close"])
        if price_col:
            return session_data[price_col].max(), session_data[price_col].min()
        
        return 0.0, 0.0
    
    def _create_anchor_zone(
        self,
        anchor_point: float,
        zone_width: float,
        previous_range: float,
        session_id: str,
        anchor_type: str
    ) -> Dict[str, Any]:
        """Create anchor zone data structure"""
        # Apply width constraints
        zone_width = max(
            self.anchor_config.min_zone_width,
            min(self.anchor_config.max_zone_width, zone_width)
        )
        
        # Calculate zone boundaries
        half_width = zone_width / 2.0
        zone_low = anchor_point - half_width
        zone_high = anchor_point + half_width
        
        # Calculate confidence based on zone properties
        confidence = self._calculate_zone_confidence(
            anchor_point, zone_width, previous_range, anchor_type
        )
        
        # Calculate precision score
        precision_score = self._calculate_initial_precision_score(
            anchor_point, zone_width, previous_range
        )
        
        return {
            "anchor_point": anchor_point,
            "zone_range": (zone_low, zone_high),
            "zone_width": zone_width,
            "confidence": confidence,
            "precision_score": precision_score,
            "session_id": session_id,
            "anchor_type": anchor_type,
            "previous_range": previous_range,
            "discovery_timestamp": time.time(),
            "theory_b_valid": False,  # Will be validated later
            "temporal_offset": 0.0,
            "forward_positioning": {}
        }
    
    def _calculate_zone_confidence(
        self,
        anchor_point: float,
        zone_width: float, 
        previous_range: float,
        anchor_type: str
    ) -> float:
        """Calculate confidence score for anchor zone"""
        base_confidence = 0.7  # Base confidence level
        
        # Adjust based on zone width relative to previous range
        width_ratio = zone_width / previous_range if previous_range > 0 else 0
        if 0.35 <= width_ratio <= 0.45:  # Close to 40% target
            width_bonus = 0.2
        else:
            width_penalty = abs(width_ratio - 0.40) * 0.5
            width_bonus = -min(0.15, width_penalty)
        
        # Adjust based on anchor type
        type_bonuses = {
            "primary_midpoint": 0.1,
            "secondary_upper": 0.05,
            "secondary_lower": 0.05
        }
        type_bonus = type_bonuses.get(anchor_type, 0.0)
        
        confidence = base_confidence + width_bonus + type_bonus
        return max(0.1, min(1.0, confidence))
    
    def _calculate_initial_precision_score(
        self,
        anchor_point: float,
        zone_width: float,
        previous_range: float
    ) -> float:
        """Calculate initial precision score (target: 7.55)"""
        # Base precision inversely related to zone width
        if zone_width > 0:
            base_precision = min(8.0, 100.0 / zone_width)
        else:
            base_precision = 0.0
        
        # Adjust based on archaeological ratio (40% target)
        if previous_range > 0:
            ratio = zone_width / previous_range
            ratio_accuracy = 1.0 - abs(ratio - 0.40) / 0.40
            precision_adjustment = ratio_accuracy * 2.0
        else:
            precision_adjustment = 0.0
        
        precision_score = base_precision + precision_adjustment
        return max(0.0, min(10.0, precision_score))
    
    def _filter_anchor_zones(self, anchor_zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter anchor zones by configuration constraints"""
        filtered = []
        
        for zone in anchor_zones:
            # Check confidence threshold
            if zone["confidence"] < self.anchor_config.zone_confidence_threshold:
                continue
            
            # Check zone width constraints
            if (zone["zone_width"] < self.anchor_config.min_zone_width or
                zone["zone_width"] > self.anchor_config.max_zone_width):
                continue
            
            filtered.append(zone)
        
        return filtered


class TemporalNonLocalityValidator:
    """
    Validates temporal non-locality patterns using Theory B forward positioning
    
    Implements archaeological principle: events position relative to FINAL session
    completion, not intermediate states. Detects temporal echoes and forward-propagating
    market structure information.
    """
    
    def __init__(self, config: ArchaeologicalConfig):
        self.config = config
        self.temporal_config = config.temporal_nonlocality
        logger.debug("TemporalNonLocalityValidator initialized")
    
    def analyze_nonlocality(
        self,
        session_data: pd.DataFrame,
        anchor_zones: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze temporal non-locality patterns in session data
        
        Args:
            session_data: Session event/price data
            anchor_zones: Dimensional anchor zones for reference
            
        Returns:
            Temporal non-locality analysis results
        """
        try:
            # Extract temporal event sequence
            event_sequence = self._extract_event_sequence(session_data)
            
            # Detect causality patterns
            causality_patterns = self._detect_causality_patterns(event_sequence, anchor_zones)
            
            # Analyze temporal coherence
            coherence_analysis = self._analyze_temporal_coherence(event_sequence, anchor_zones)
            
            # Forward positioning validation
            forward_positioning = self._analyze_forward_positioning(
                event_sequence, anchor_zones, session_data
            )
            
            analysis_results = {
                "event_sequence_length": len(event_sequence),
                "causality_patterns": causality_patterns,
                "temporal_coherence": coherence_analysis,
                "forward_positioning": forward_positioning,
                "nonlocality_score": self._calculate_nonlocality_score(
                    causality_patterns, coherence_analysis, forward_positioning
                ),
                "analysis_timestamp": time.time()
            }
            
            logger.debug(f"Temporal non-locality analysis complete: {len(causality_patterns)} patterns")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Temporal non-locality analysis failed: {e}")
            return {
                "event_sequence_length": 0,
                "causality_patterns": [],
                "temporal_coherence": {"score": 0.0},
                "forward_positioning": {},
                "nonlocality_score": 0.0,
                "error": str(e)
            }
    
    def detect_temporal_echoes(
        self,
        session_data: pd.DataFrame,
        anchor_zones: List[Dict[str, Any]]
    ) -> List[TemporalEcho]:
        """
        Detect temporal echoes - forward-propagating market structure information
        
        Args:
            session_data: Session data for echo detection
            anchor_zones: Reference anchor zones
            
        Returns:
            List of detected temporal echoes
        """
        if not self.temporal_config.temporal_echo_detection:
            return []
        
        try:
            event_sequence = self._extract_event_sequence(session_data)
            temporal_echoes = []
            
            for i, source_event in enumerate(event_sequence[:-self.temporal_config.short_term_window]):
                # Look for echo propagation in forward events
                for j in range(i + 1, min(i + self.temporal_config.long_term_window, len(event_sequence))):
                    target_event = event_sequence[j]
                    
                    # Calculate echo properties
                    echo_strength = self._calculate_echo_strength(source_event, target_event, anchor_zones)
                    temporal_distance = j - i
                    coherence_score = self._calculate_echo_coherence(source_event, target_event)
                    
                    # Filter significant echoes
                    if echo_strength > 0.5 and coherence_score > self.temporal_config.temporal_coherence_threshold:
                        echo = TemporalEcho(
                            echo_id=f"echo_{i}_{j}",
                            source_event=source_event,
                            target_event=target_event,
                            propagation_strength=echo_strength,
                            temporal_distance=temporal_distance,
                            coherence_score=coherence_score,
                            forward_validation=self._validate_forward_echo(
                                source_event, target_event, anchor_zones
                            )
                        )
                        temporal_echoes.append(echo)
            
            logger.debug(f"Detected {len(temporal_echoes)} temporal echoes")
            return temporal_echoes
            
        except Exception as e:
            logger.error(f"Temporal echo detection failed: {e}")
            return []
    
    def _extract_event_sequence(self, session_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract temporal event sequence from session data"""
        events = []
        
        # Sort by timestamp if available
        if 'timestamp' in session_data.columns:
            sorted_data = session_data.sort_values('timestamp')
        elif 'time' in session_data.columns:
            sorted_data = session_data.sort_values('time')
        else:
            sorted_data = session_data.sort_index()
        
        for idx, row in sorted_data.iterrows():
            event = {
                "index": idx,
                "timestamp": row.get('timestamp', idx),
                "price": self._extract_event_price(row),
                "event_type": row.get('event_type', 'unknown'),
                "volume": row.get('volume', 0.0),
                "features": self._extract_event_features(row)
            }
            events.append(event)
        
        return events
    
    def _extract_event_price(self, row: pd.Series) -> float:
        """Extract price from event row"""
        price_candidates = ['price', 'close', 'Close', 'mid', 'avg_price']
        for candidate in price_candidates:
            if candidate in row.index and pd.notna(row[candidate]):
                return float(row[candidate])
        return 0.0
    
    def _extract_event_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract numerical features from event row"""
        features = {}
        for col in row.index:
            if col.startswith('f') and col[1:].isdigit():  # Feature columns (f0, f1, etc.)
                features[col] = float(row[col]) if pd.notna(row[col]) else 0.0
        return features
    
    def _detect_causality_patterns(
        self,
        event_sequence: List[Dict[str, Any]], 
        anchor_zones: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect causality patterns in event sequence"""
        causality_patterns = []
        
        for anchor in anchor_zones:
            anchor_point = anchor["anchor_point"]
            zone_range = anchor["zone_range"]
            
            # Find events within or near anchor zone
            zone_events = []
            for event in event_sequence:
                event_price = event["price"]
                if zone_range[0] <= event_price <= zone_range[1]:
                    zone_events.append(event)
                elif abs(event_price - anchor_point) / anchor_point < 0.01:  # Within 1%
                    zone_events.append(event)
            
            if len(zone_events) >= 2:
                # Analyze causality between zone events
                for i in range(len(zone_events) - 1):
                    source = zone_events[i]
                    target = zone_events[i + 1]
                    
                    causality_strength = self._calculate_causality_strength(source, target, anchor)
                    if causality_strength > self.temporal_config.causality_threshold:
                        pattern = {
                            "source_event": source,
                            "target_event": target,
                            "causality_strength": causality_strength,
                            "anchor_zone": anchor,
                            "temporal_distance": target["timestamp"] - source["timestamp"]
                        }
                        causality_patterns.append(pattern)
        
        return causality_patterns
    
    def _analyze_temporal_coherence(
        self,
        event_sequence: List[Dict[str, Any]],
        anchor_zones: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal coherence of event sequence"""
        if len(event_sequence) < 3:
            return {"score": 0.0, "analysis": "insufficient_data"}
        
        # Calculate price momentum coherence
        prices = [event["price"] for event in event_sequence]
        price_diffs = np.diff(prices)
        momentum_coherence = self._calculate_momentum_coherence(price_diffs)
        
        # Calculate temporal spacing coherence
        timestamps = [event["timestamp"] for event in event_sequence]
        time_diffs = np.diff(timestamps)
        temporal_coherence = self._calculate_temporal_spacing_coherence(time_diffs)
        
        # Calculate anchor zone interaction coherence
        zone_coherence = self._calculate_zone_interaction_coherence(event_sequence, anchor_zones)
        
        # Combined coherence score
        combined_score = (
            momentum_coherence * 0.4 +
            temporal_coherence * 0.3 +
            zone_coherence * 0.3
        )
        
        return {
            "score": combined_score,
            "momentum_coherence": momentum_coherence,
            "temporal_coherence": temporal_coherence, 
            "zone_coherence": zone_coherence,
            "event_count": len(event_sequence)
        }
    
    def _analyze_forward_positioning(
        self,
        event_sequence: List[Dict[str, Any]],
        anchor_zones: List[Dict[str, Any]],
        session_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze forward positioning patterns (Theory B)"""
        if not self.temporal_config.enable_theory_b_validation:
            return {"enabled": False}
        
        forward_patterns = []
        
        # Analyze each anchor zone for forward positioning
        for anchor in anchor_zones:
            anchor_point = anchor["anchor_point"]
            zone_events = [
                event for event in event_sequence
                if anchor["zone_range"][0] <= event["price"] <= anchor["zone_range"][1]
            ]
            
            if zone_events:
                # Calculate forward positioning metrics
                positioning_strength = self._calculate_forward_positioning_strength(
                    zone_events, event_sequence, anchor
                )
                
                completion_accuracy = self._calculate_completion_accuracy(
                    zone_events, session_data, anchor
                )
                
                pattern = {
                    "anchor_zone": anchor,
                    "zone_event_count": len(zone_events),
                    "positioning_strength": positioning_strength,
                    "completion_accuracy": completion_accuracy,
                    "forward_validation": positioning_strength > 0.7 and completion_accuracy > 0.8
                }
                forward_patterns.append(pattern)
        
        return {
            "enabled": True,
            "patterns": forward_patterns,
            "pattern_count": len(forward_patterns),
            "average_positioning_strength": np.mean([
                p["positioning_strength"] for p in forward_patterns
            ]) if forward_patterns else 0.0
        }
    
    def _calculate_nonlocality_score(
        self,
        causality_patterns: List[Dict[str, Any]],
        coherence_analysis: Dict[str, Any],
        forward_positioning: Dict[str, Any]
    ) -> float:
        """Calculate overall non-locality score"""
        # Base score from temporal coherence
        base_score = coherence_analysis.get("score", 0.0) * 0.4
        
        # Causality pattern contribution
        causality_score = len(causality_patterns) / 10.0  # Normalize
        causality_contribution = min(0.3, causality_score * 0.3)
        
        # Forward positioning contribution
        positioning_score = forward_positioning.get("average_positioning_strength", 0.0) * 0.3
        
        total_score = base_score + causality_contribution + positioning_score
        return min(1.0, max(0.0, total_score))
    
    def _calculate_echo_strength(
        self,
        source_event: Dict[str, Any],
        target_event: Dict[str, Any],
        anchor_zones: List[Dict[str, Any]]
    ) -> float:
        """Calculate temporal echo propagation strength"""
        # Price correlation component
        price_diff = abs(source_event["price"] - target_event["price"])
        relative_price_diff = price_diff / source_event["price"] if source_event["price"] > 0 else 1.0
        price_strength = max(0.0, 1.0 - relative_price_diff)
        
        # Anchor zone proximity component
        zone_proximity = 0.0
        for anchor in anchor_zones:
            source_proximity = self._calculate_zone_proximity(source_event["price"], anchor)
            target_proximity = self._calculate_zone_proximity(target_event["price"], anchor)
            zone_proximity = max(zone_proximity, (source_proximity + target_proximity) / 2.0)
        
        # Feature similarity component
        feature_similarity = self._calculate_feature_similarity(
            source_event.get("features", {}),
            target_event.get("features", {})
        )
        
        # Combined echo strength
        echo_strength = (
            price_strength * 0.4 +
            zone_proximity * 0.4 +
            feature_similarity * 0.2
        )
        
        # Apply decay based on temporal distance
        temporal_distance = target_event["timestamp"] - source_event["timestamp"]
        decay_factor = self.temporal_config.echo_propagation_decay ** temporal_distance
        
        return echo_strength * decay_factor
    
    def _calculate_echo_coherence(
        self,
        source_event: Dict[str, Any],
        target_event: Dict[str, Any]
    ) -> float:
        """Calculate coherence score for temporal echo"""
        # Event type consistency
        type_coherence = 1.0 if source_event.get("event_type") == target_event.get("event_type") else 0.5
        
        # Volume consistency
        source_vol = source_event.get("volume", 1.0)
        target_vol = target_event.get("volume", 1.0)
        if source_vol > 0 and target_vol > 0:
            volume_ratio = min(source_vol, target_vol) / max(source_vol, target_vol)
        else:
            volume_ratio = 0.5
        
        # Combined coherence
        coherence = (type_coherence * 0.6 + volume_ratio * 0.4)
        return coherence
    
    def _validate_forward_echo(
        self,
        source_event: Dict[str, Any],
        target_event: Dict[str, Any],
        anchor_zones: List[Dict[str, Any]]
    ) -> bool:
        """Validate forward echo using Theory B principles"""
        if not self.temporal_config.completion_validation_enabled:
            return True
        
        # Check if echo represents forward positioning
        price_movement = target_event["price"] - source_event["price"]
        
        # Validate against anchor zone positioning
        for anchor in anchor_zones:
            zone_center = anchor["anchor_point"]
            
            # Check if movement is consistent with zone positioning
            source_distance = abs(source_event["price"] - zone_center)
            target_distance = abs(target_event["price"] - zone_center)
            
            # Forward validation: movement should be coherent with zone structure
            if source_distance > target_distance:  # Moving toward zone
                return True
            elif price_movement != 0 and target_distance < source_distance * 1.1:  # Maintaining zone proximity
                return True
        
        return False
    
    def _calculate_causality_strength(
        self,
        source_event: Dict[str, Any],
        target_event: Dict[str, Any],
        anchor: Dict[str, Any]
    ) -> float:
        """Calculate causality strength between events"""
        # Temporal consistency (closer in time = stronger causality)
        time_diff = abs(target_event["timestamp"] - source_event["timestamp"])
        temporal_strength = 1.0 / (1.0 + time_diff / 10.0)  # Decay over time
        
        # Price relationship strength
        price_correlation = self._calculate_price_correlation(source_event, target_event)
        
        # Anchor zone influence
        zone_influence = self._calculate_zone_influence(source_event, target_event, anchor)
        
        # Combined causality strength
        causality = (
            temporal_strength * 0.4 +
            price_correlation * 0.4 +
            zone_influence * 0.2
        )
        
        return causality
    
    def _calculate_momentum_coherence(self, price_diffs: np.ndarray) -> float:
        """Calculate momentum coherence from price differences"""
        if len(price_diffs) < 2:
            return 0.0
        
        # Calculate consistency of momentum direction
        positive_moves = np.sum(price_diffs > 0)
        negative_moves = np.sum(price_diffs < 0)
        total_moves = len(price_diffs)
        
        # Coherence based on momentum consistency
        dominant_direction = max(positive_moves, negative_moves)
        coherence = dominant_direction / total_moves if total_moves > 0 else 0.0
        
        return coherence
    
    def _calculate_temporal_spacing_coherence(self, time_diffs: np.ndarray) -> float:
        """Calculate temporal spacing coherence"""
        if len(time_diffs) < 2:
            return 0.0
        
        # Calculate consistency of temporal spacing
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        
        # Coherence inversely related to variance in spacing
        if mean_diff > 0:
            coherence = 1.0 / (1.0 + std_diff / mean_diff)
        else:
            coherence = 0.0
        
        return min(1.0, coherence)
    
    def _calculate_zone_interaction_coherence(
        self,
        event_sequence: List[Dict[str, Any]],
        anchor_zones: List[Dict[str, Any]]
    ) -> float:
        """Calculate coherence of event interactions with anchor zones"""
        if not anchor_zones:
            return 0.0
        
        zone_interactions = 0
        total_events = len(event_sequence)
        
        for event in event_sequence:
            event_price = event["price"]
            
            # Check interaction with any anchor zone
            for anchor in anchor_zones:
                zone_range = anchor["zone_range"]
                if zone_range[0] <= event_price <= zone_range[1]:
                    zone_interactions += 1
                    break  # Count each event only once
        
        # Coherence based on proportion of events interacting with zones
        coherence = zone_interactions / total_events if total_events > 0 else 0.0
        return coherence
    
    def _calculate_forward_positioning_strength(
        self,
        zone_events: List[Dict[str, Any]],
        all_events: List[Dict[str, Any]],
        anchor: Dict[str, Any]
    ) -> float:
        """Calculate forward positioning strength for anchor zone"""
        if not zone_events:
            return 0.0
        
        # Look ahead from each zone event
        positioning_scores = []
        
        for zone_event in zone_events:
            zone_event_idx = None
            for i, event in enumerate(all_events):
                if event["timestamp"] == zone_event["timestamp"]:
                    zone_event_idx = i
                    break
            
            if zone_event_idx is not None:
                # Analyze forward events within window
                forward_window = min(
                    len(all_events) - zone_event_idx - 1,
                    self.temporal_config.forward_positioning_window
                )
                
                if forward_window > 0:
                    forward_events = all_events[zone_event_idx + 1:zone_event_idx + 1 + forward_window]
                    positioning_score = self._calculate_event_positioning_score(
                        zone_event, forward_events, anchor
                    )
                    positioning_scores.append(positioning_score)
        
        return np.mean(positioning_scores) if positioning_scores else 0.0
    
    def _calculate_completion_accuracy(
        self,
        zone_events: List[Dict[str, Any]],
        session_data: pd.DataFrame,
        anchor: Dict[str, Any]
    ) -> float:
        """Calculate accuracy relative to session completion"""
        if not zone_events:
            return 0.0
        
        # Get final session state
        final_price = self._extract_final_session_price(session_data)
        anchor_point = anchor["anchor_point"]
        
        # Calculate accuracy of zone events relative to final state
        accuracies = []
        for event in zone_events:
            event_price = event["price"]
            
            # Accuracy based on how well event price predicts final positioning
            predicted_direction = 1.0 if event_price > anchor_point else -1.0
            actual_direction = 1.0 if final_price > anchor_point else -1.0
            
            # Direction accuracy
            direction_accuracy = 1.0 if predicted_direction == actual_direction else 0.0
            
            # Distance accuracy (closer predictions score higher)
            distance_to_final = abs(event_price - final_price)
            max_possible_distance = abs(anchor_point - final_price)
            if max_possible_distance > 0:
                distance_accuracy = 1.0 - (distance_to_final / max_possible_distance)
            else:
                distance_accuracy = 1.0
            
            # Combined accuracy
            combined_accuracy = (direction_accuracy * 0.6 + distance_accuracy * 0.4)
            accuracies.append(combined_accuracy)
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def _extract_final_session_price(self, session_data: pd.DataFrame) -> float:
        """Extract final session price for completion analysis"""
        price_col = None
        for col in ['close', 'Close', 'price', 'last']:
            if col in session_data.columns:
                price_col = col
                break
        
        if price_col:
            return float(session_data[price_col].iloc[-1])
        else:
            return 0.0
    
    def _calculate_zone_proximity(self, price: float, anchor: Dict[str, Any]) -> float:
        """Calculate proximity to anchor zone"""
        zone_range = anchor["zone_range"]
        
        if zone_range[0] <= price <= zone_range[1]:
            return 1.0  # Inside zone
        
        # Distance to nearest zone boundary
        distance_to_zone = min(
            abs(price - zone_range[0]),
            abs(price - zone_range[1])
        )
        
        # Proximity inversely related to distance
        proximity = 1.0 / (1.0 + distance_to_zone / anchor["zone_width"])
        return proximity
    
    def _calculate_feature_similarity(
        self,
        features1: Dict[str, float],
        features2: Dict[str, float]
    ) -> float:
        """Calculate similarity between event features"""
        if not features1 or not features2:
            return 0.0
        
        # Find common features
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
        
        # Calculate correlation of common features
        values1 = [features1[key] for key in common_features]
        values2 = [features2[key] for key in common_features]
        
        if len(values1) >= 2:
            correlation = np.corrcoef(values1, values2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            # Single feature: use inverse of relative difference
            correlation = 1.0 - abs(values1[0] - values2[0]) / (abs(values1[0]) + abs(values2[0]) + 1e-8)
        
        return max(0.0, correlation)
    
    def _calculate_price_correlation(
        self,
        source_event: Dict[str, Any],
        target_event: Dict[str, Any]
    ) -> float:
        """Calculate price correlation between events"""
        source_price = source_event["price"]
        target_price = target_event["price"]
        
        if source_price == 0:
            return 0.0
        
        # Correlation based on relative price change
        relative_change = abs(target_price - source_price) / source_price
        correlation = 1.0 / (1.0 + relative_change)
        
        return correlation
    
    def _calculate_zone_influence(
        self,
        source_event: Dict[str, Any],
        target_event: Dict[str, Any],
        anchor: Dict[str, Any]
    ) -> float:
        """Calculate anchor zone influence on event relationship"""
        source_proximity = self._calculate_zone_proximity(source_event["price"], anchor)
        target_proximity = self._calculate_zone_proximity(target_event["price"], anchor)
        
        # Influence based on average proximity to zone
        influence = (source_proximity + target_proximity) / 2.0
        return influence
    
    def _calculate_event_positioning_score(
        self,
        zone_event: Dict[str, Any],
        forward_events: List[Dict[str, Any]],
        anchor: Dict[str, Any]
    ) -> float:
        """Calculate positioning score for zone event based on forward events"""
        if not forward_events:
            return 0.0
        
        zone_price = zone_event["price"]
        anchor_point = anchor["anchor_point"]
        
        # Analyze forward movement patterns
        positioning_scores = []
        
        for forward_event in forward_events:
            forward_price = forward_event["price"]
            
            # Score based on movement consistency with zone positioning
            zone_direction = 1.0 if zone_price > anchor_point else -1.0
            forward_direction = 1.0 if forward_price > zone_price else -1.0
            
            # Consistency score
            consistency = 1.0 if zone_direction == forward_direction else 0.5
            
            # Distance score (movement away from zone can be positive positioning)
            distance_score = abs(forward_price - anchor_point) / abs(zone_price - anchor_point) if zone_price != anchor_point else 1.0
            distance_score = min(1.0, distance_score)
            
            # Combined positioning score
            event_score = consistency * 0.7 + distance_score * 0.3
            positioning_scores.append(event_score)
        
        return np.mean(positioning_scores) if positioning_scores else 0.0


class TheoryBValidator:
    """
    Validates Theory B forward positioning principles
    
    Theory B: Events position relative to eventual session completion, not intermediate states.
    Validates that archaeological zones exhibit forward-looking temporal coherence with
    eventual range completion and 7.55-point precision targets.
    """
    
    def __init__(self, config: ArchaeologicalConfig):
        self.config = config
        self.temporal_config = config.temporal_nonlocality
        self.anchor_config = config.dimensional_anchor
        logger.debug("TheoryBValidator initialized")
    
    def validate_forward_positioning(
        self,
        session_data: pd.DataFrame,
        anchor_zones: List[Dict[str, Any]],
        temporal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate Theory B forward positioning for anchor zones
        
        Args:
            session_data: Complete session data for validation
            anchor_zones: Dimensional anchor zones to validate
            temporal_analysis: Temporal non-locality analysis results
            
        Returns:
            Theory B validation results for all anchor zones
        """
        if not self.temporal_config.enable_theory_b_validation:
            return {"enabled": False, "validation_results": []}
        
        validation_results = []
        
        for anchor in anchor_zones:
            result = self._validate_single_zone_theory_b(
                anchor, session_data, temporal_analysis
            )
            validation_results.append(result)
        
        # Calculate aggregate validation metrics
        valid_zones = [r for r in validation_results if r.forward_positioning_valid]
        validation_summary = {
            "enabled": True,
            "validation_results": validation_results,
            "total_zones": len(anchor_zones),
            "valid_zones": len(valid_zones),
            "validation_rate": len(valid_zones) / len(anchor_zones) if anchor_zones else 0.0,
            "average_completion_accuracy": np.mean([
                r.completion_accuracy for r in validation_results
            ]) if validation_results else 0.0,
            "average_precision_score": np.mean([
                r.precision_score for r in validation_results
            ]) if validation_results else 0.0
        }
        
        logger.debug(f"Theory B validation complete: {len(valid_zones)}/{len(anchor_zones)} zones valid")
        return validation_summary
    
    def calculate_precision_scores(
        self,
        anchor_zones: List[Dict[str, Any]],
        session_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Calculate precision scores for anchor zones (target: 7.55 points)
        
        Args:
            anchor_zones: Anchor zones for precision calculation
            session_data: Session data for precision validation
            
        Returns:
            Precision scores for each anchor zone
        """
        precision_scores = []
        
        # Extract session completion data
        final_price = self._extract_session_final_price(session_data)
        session_range = self._extract_session_range(session_data)
        
        for anchor in anchor_zones:
            precision_data = self._calculate_single_zone_precision(
                anchor, final_price, session_range, session_data
            )
            precision_scores.append(precision_data)
        
        logger.debug(f"Calculated precision scores for {len(anchor_zones)} zones")
        return precision_scores
    
    def _validate_single_zone_theory_b(
        self,
        anchor: Dict[str, Any],
        session_data: pd.DataFrame,
        temporal_analysis: Dict[str, Any]
    ) -> TheoryBResult:
        """Validate Theory B principles for a single anchor zone"""
        anchor_point = anchor["anchor_point"]
        zone_range = anchor["zone_range"]
        session_id = anchor.get("session_id", "unknown")
        
        # Extract session completion data
        final_price = self._extract_session_final_price(session_data)
        session_high = self._extract_session_high(session_data)
        session_low = self._extract_session_low(session_data)
        
        # Calculate forward positioning validation
        forward_positioning_valid = self._validate_forward_positioning_logic(
            anchor, final_price, session_high, session_low
        )
        
        # Calculate completion accuracy (how well zone predicted final state)
        completion_accuracy = self._calculate_completion_accuracy_score(
            anchor, final_price, session_high, session_low
        )
        
        # Calculate intermediate filtering score
        intermediate_filtering_score = self._calculate_intermediate_filtering(
            anchor, session_data, temporal_analysis
        )
        
        # Calculate final state alignment
        final_state_alignment = self._calculate_final_state_alignment(
            anchor, final_price, session_high, session_low
        )
        
        # Calculate precision score (target: 7.55)
        precision_score = self._calculate_zone_precision_score(
            anchor, final_price, session_high, session_low
        )
        
        # Calculate temporal coherence
        temporal_coherence = self._calculate_temporal_coherence_score(
            anchor, temporal_analysis
        )
        
        return TheoryBResult(
            zone_id=f"{session_id}_zone_{anchor_point:.2f}",
            forward_positioning_valid=forward_positioning_valid,
            completion_accuracy=completion_accuracy,
            intermediate_filtering_score=intermediate_filtering_score,
            final_state_alignment=final_state_alignment,
            precision_score=precision_score,
            temporal_coherence=temporal_coherence
        )
    
    def _validate_forward_positioning_logic(
        self,
        anchor: Dict[str, Any],
        final_price: float,
        session_high: float,
        session_low: float
    ) -> bool:
        """Validate forward positioning logic for Theory B compliance"""
        anchor_point = anchor["anchor_point"]
        zone_range = anchor["zone_range"]
        
        # Theory B validation: anchor positioning should relate to FINAL range
        final_range = session_high - session_low
        
        # Check if anchor positioning is coherent with final range
        anchor_position_in_range = (anchor_point - session_low) / final_range if final_range > 0 else 0.5
        
        # Valid positioning: anchor should be meaningfully positioned in final range
        valid_positioning = 0.1 <= anchor_position_in_range <= 0.9
        
        # Check zone coverage relative to final range
        zone_coverage = (zone_range[1] - zone_range[0]) / final_range if final_range > 0 else 0.0
        valid_coverage = 0.05 <= zone_coverage <= 0.75  # Zone should be reasonable portion of range
        
        # Check final price interaction with zone
        final_price_interaction = zone_range[0] <= final_price <= zone_range[1] or \
                                abs(final_price - anchor_point) / anchor_point < 0.02
        
        # All criteria must be met for valid forward positioning
        return valid_positioning and valid_coverage and final_price_interaction
    
    def _calculate_completion_accuracy_score(
        self,
        anchor: Dict[str, Any],
        final_price: float,
        session_high: float,
        session_low: float
    ) -> float:
        """Calculate accuracy score relative to session completion"""
        anchor_point = anchor["anchor_point"]
        zone_range = anchor["zone_range"]
        
        # Accuracy based on final price positioning relative to anchor
        if zone_range[0] <= final_price <= zone_range[1]:
            # Final price within zone: high accuracy
            zone_accuracy = 1.0
        else:
            # Distance-based accuracy
            distance_to_zone = min(
                abs(final_price - zone_range[0]),
                abs(final_price - zone_range[1])
            )
            zone_width = zone_range[1] - zone_range[0]
            zone_accuracy = max(0.0, 1.0 - distance_to_zone / zone_width) if zone_width > 0 else 0.0
        
        # Range prediction accuracy
        predicted_range_position = (anchor_point - session_low) / (session_high - session_low) if session_high > session_low else 0.5
        actual_range_position = (final_price - session_low) / (session_high - session_low) if session_high > session_low else 0.5
        range_accuracy = 1.0 - abs(predicted_range_position - actual_range_position)
        
        # Combined accuracy score
        completion_accuracy = (zone_accuracy * 0.7 + range_accuracy * 0.3)
        return max(0.0, min(1.0, completion_accuracy))
    
    def _calculate_intermediate_filtering(
        self,
        anchor: Dict[str, Any],
        session_data: pd.DataFrame,
        temporal_analysis: Dict[str, Any]
    ) -> float:
        """Calculate intermediate state filtering score"""
        if not self.temporal_config.intermediate_state_filtering:
            return 1.0
        
        # Analyze how well anchor filters out intermediate noise
        anchor_point = anchor["anchor_point"]
        
        # Extract price series
        prices = self._extract_price_series(session_data)
        if len(prices) < 3:
            return 0.5
        
        # Calculate intermediate state deviations
        intermediate_deviations = []
        for price in prices[1:-1]:  # Exclude start and end
            deviation = abs(price - anchor_point) / anchor_point if anchor_point != 0 else 0.0
            intermediate_deviations.append(deviation)
        
        # Filtering score: lower average deviation = better filtering
        if intermediate_deviations:
            average_deviation = np.mean(intermediate_deviations)
            filtering_score = 1.0 / (1.0 + average_deviation * 10.0)  # Scale and invert
        else:
            filtering_score = 0.5
        
        return max(0.0, min(1.0, filtering_score))
    
    def _calculate_final_state_alignment(
        self,
        anchor: Dict[str, Any],
        final_price: float,
        session_high: float,
        session_low: float
    ) -> float:
        """Calculate alignment with final session state"""
        anchor_point = anchor["anchor_point"]
        
        # Apply final state weighting factor
        final_state_weight = self.temporal_config.final_state_weight
        
        # Calculate alignment based on final price positioning
        if session_high > session_low:
            anchor_percentile = (anchor_point - session_low) / (session_high - session_low)
            final_percentile = (final_price - session_low) / (session_high - session_low)
            
            # Alignment score based on percentile similarity
            percentile_alignment = 1.0 - abs(anchor_percentile - final_percentile)
        else:
            percentile_alignment = 0.5
        
        # Distance alignment
        distance_alignment = 1.0 / (1.0 + abs(anchor_point - final_price) / final_price) if final_price != 0 else 0.0
        
        # Weighted final state alignment
        alignment = (
            percentile_alignment * 0.6 +
            distance_alignment * 0.4
        ) * final_state_weight
        
        return max(0.0, min(1.0, alignment))
    
    def _calculate_zone_precision_score(
        self,
        anchor: Dict[str, Any],
        final_price: float,
        session_high: float,
        session_low: float
    ) -> float:
        """Calculate zone precision score (target: 7.55)"""
        anchor_point = anchor["anchor_point"]
        zone_width = anchor.get("zone_width", 0.0)
        
        # Base precision from zone width (smaller zones = higher precision)
        if zone_width > 0:
            width_precision = min(10.0, 50.0 / zone_width)  # Scale factor for precision
        else:
            width_precision = 0.0
        
        # Final price accuracy precision
        final_distance = abs(final_price - anchor_point)
        if anchor_point != 0:
            accuracy_precision = max(0.0, 5.0 - (final_distance / anchor_point) * 20.0)
        else:
            accuracy_precision = 0.0
        
        # Range positioning precision
        session_range = session_high - session_low
        if session_range > 0:
            range_precision = max(0.0, 3.0 - abs(zone_width - session_range * 0.4) / session_range * 10.0)
        else:
            range_precision = 0.0
        
        # Combined precision score
        precision_score = width_precision * 0.4 + accuracy_precision * 0.4 + range_precision * 0.2
        
        # Target adjustment: scale toward 7.55 target
        target_precision = 7.55
        if precision_score > target_precision:
            precision_score = target_precision + (precision_score - target_precision) * 0.1
        
        return max(0.0, min(10.0, precision_score))
    
    def _calculate_temporal_coherence_score(
        self,
        anchor: Dict[str, Any],
        temporal_analysis: Dict[str, Any]
    ) -> float:
        """Calculate temporal coherence score from analysis"""
        if not temporal_analysis:
            return 0.0
        
        # Extract coherence metrics
        base_coherence = temporal_analysis.get("temporal_coherence", {}).get("score", 0.0)
        
        # Forward positioning coherence
        forward_patterns = temporal_analysis.get("forward_positioning", {}).get("patterns", [])
        if forward_patterns:
            anchor_patterns = [
                p for p in forward_patterns 
                if p.get("anchor_zone", {}).get("anchor_point") == anchor["anchor_point"]
            ]
            if anchor_patterns:
                positioning_coherence = np.mean([
                    p.get("positioning_strength", 0.0) for p in anchor_patterns
                ])
            else:
                positioning_coherence = 0.0
        else:
            positioning_coherence = 0.0
        
        # Combined temporal coherence
        coherence_score = base_coherence * 0.6 + positioning_coherence * 0.4
        return max(0.0, min(1.0, coherence_score))
    
    def _calculate_single_zone_precision(
        self,
        anchor: Dict[str, Any],
        final_price: float,
        session_range: float,
        session_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate precision data for single anchor zone"""
        anchor_point = anchor["anchor_point"]
        zone_id = f"zone_{anchor_point:.2f}"
        
        # Calculate precision score
        precision_score = self._calculate_zone_precision_score(
            anchor, final_price, final_price + session_range/2, final_price - session_range/2
        )
        
        # Calculate precision accuracy relative to target
        precision_target = self.anchor_config.precision_target
        precision_accuracy = 1.0 - abs(precision_score - precision_target) / precision_target
        precision_accuracy = max(0.0, min(1.0, precision_accuracy))
        
        # Additional precision metrics
        precision_data = {
            "zone_id": zone_id,
            "anchor_point": anchor_point,
            "precision": precision_score,
            "precision_target": precision_target,
            "precision_accuracy": precision_accuracy,
            "target_achievement": precision_score >= precision_target * 0.9,
            "precision_grade": self._grade_precision_score(precision_score),
            "calculation_timestamp": time.time()
        }
        
        return precision_data
    
    def _grade_precision_score(self, precision_score: float) -> str:
        """Grade precision score for qualitative assessment"""
        if precision_score >= 8.0:
            return "excellent"
        elif precision_score >= 7.0:
            return "good"
        elif precision_score >= 5.0:
            return "fair"
        elif precision_score >= 3.0:
            return "poor"
        else:
            return "inadequate"
    
    def _extract_session_final_price(self, session_data: pd.DataFrame) -> float:
        """Extract final session price"""
        for col in ['close', 'Close', 'price', 'last']:
            if col in session_data.columns:
                return float(session_data[col].iloc[-1])
        return 0.0
    
    def _extract_session_high(self, session_data: pd.DataFrame) -> float:
        """Extract session high price"""
        for col in ['high', 'High', 'HIGH']:
            if col in session_data.columns:
                return float(session_data[col].max())
        
        # Fallback to max of any price column
        for col in ['price', 'close', 'Close']:
            if col in session_data.columns:
                return float(session_data[col].max())
        
        return 0.0
    
    def _extract_session_low(self, session_data: pd.DataFrame) -> float:
        """Extract session low price"""
        for col in ['low', 'Low', 'LOW']:
            if col in session_data.columns:
                return float(session_data[col].min())
        
        # Fallback to min of any price column
        for col in ['price', 'close', 'Close']:
            if col in session_data.columns:
                return float(session_data[col].min())
        
        return 0.0
    
    def _extract_session_range(self, session_data: pd.DataFrame) -> float:
        """Extract session range"""
        session_high = self._extract_session_high(session_data)
        session_low = self._extract_session_low(session_data)
        return max(0.0, session_high - session_low)
    
    def _extract_price_series(self, session_data: pd.DataFrame) -> List[float]:
        """Extract price series from session data"""
        for col in ['price', 'close', 'Close', 'mid']:
            if col in session_data.columns:
                return session_data[col].tolist()
        return []


class ZoneAnalyzer:
    """
    Comprehensive archaeological zone analysis system
    
    Integrates dimensional anchoring, temporal non-locality, and Theory B validation
    to provide complete archaeological intelligence for IRONFORGE integration.
    """
    
    def __init__(self, config: ArchaeologicalConfig):
        self.config = config
        self.anchor_calculator = DimensionalAnchorCalculator(config)
        self.temporal_validator = TemporalNonLocalityValidator(config)
        self.theory_b_validator = TheoryBValidator(config)
        logger.debug("ZoneAnalyzer initialized with comprehensive analysis capabilities")
    
    def analyze_session_zones(
        self,
        session_data: pd.DataFrame,
        previous_session_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Complete archaeological zone analysis for session
        
        Args:
            session_data: Current session data
            previous_session_data: Previous session for dimensional anchoring
            
        Returns:
            Comprehensive zone analysis results
        """
        analysis_start = time.time()
        
        try:
            # Step 1: Calculate dimensional anchors
            previous_range = 0.0
            if previous_session_data is not None:
                previous_range = self.anchor_calculator.calculate_session_range(previous_session_data)
            
            anchor_zones = self.anchor_calculator.calculate_dimensional_anchors(
                previous_range, session_data
            )
            
            # Step 2: Temporal non-locality analysis
            temporal_analysis = self.temporal_validator.analyze_nonlocality(
                session_data, anchor_zones
            )
            
            # Step 3: Detect temporal echoes
            temporal_echoes = self.temporal_validator.detect_temporal_echoes(
                session_data, anchor_zones
            )
            
            # Step 4: Theory B validation
            theory_b_results = self.theory_b_validator.validate_forward_positioning(
                session_data, anchor_zones, temporal_analysis
            )
            
            # Step 5: Precision scoring
            precision_scores = self.theory_b_validator.calculate_precision_scores(
                anchor_zones, session_data
            )
            
            # Compile comprehensive analysis
            analysis_time = time.time() - analysis_start
            
            comprehensive_analysis = {
                "session_analysis": {
                    "session_id": self._extract_session_id(session_data),
                    "analysis_timestamp": time.time(),
                    "processing_time": analysis_time,
                    "previous_range": previous_range
                },
                "dimensional_anchors": {
                    "anchor_zones": anchor_zones,
                    "zone_count": len(anchor_zones),
                    "anchor_percentage": self.config.dimensional_anchor.anchor_percentage,
                    "total_anchor_confidence": sum(zone.get("confidence", 0.0) for zone in anchor_zones)
                },
                "temporal_analysis": {
                    "nonlocality_analysis": temporal_analysis,
                    "temporal_echoes": temporal_echoes,
                    "echo_count": len(temporal_echoes),
                    "nonlocality_score": temporal_analysis.get("nonlocality_score", 0.0)
                },
                "theory_b_validation": {
                    "validation_results": theory_b_results,
                    "precision_scores": precision_scores,
                    "validation_rate": theory_b_results.get("validation_rate", 0.0),
                    "average_precision": np.mean([
                        score.get("precision", 0.0) for score in precision_scores
                    ]) if precision_scores else 0.0
                },
                "quality_metrics": {
                    "processing_performance": analysis_time < self.config.performance.max_session_processing_time,
                    "precision_target_achievement": any(
                        score.get("precision", 0.0) >= self.config.dimensional_anchor.precision_target
                        for score in precision_scores
                    ),
                    "authenticity_ready": len([
                        zone for zone in anchor_zones 
                        if zone.get("confidence", 0.0) >= 0.8
                    ]) > 0,
                    "theory_b_compliance": theory_b_results.get("validation_rate", 0.0) >= 0.5
                }
            }
            
            logger.debug(
                f"Zone analysis complete: {len(anchor_zones)} zones, "
                f"{len(temporal_echoes)} echoes, {analysis_time:.3f}s"
            )
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Zone analysis failed: {e}")
            return {
                "session_analysis": {"error": str(e)},
                "dimensional_anchors": {"anchor_zones": [], "zone_count": 0},
                "temporal_analysis": {"nonlocality_score": 0.0},
                "theory_b_validation": {"validation_rate": 0.0},
                "quality_metrics": {"processing_performance": False}
            }
    
    def _extract_session_id(self, session_data: pd.DataFrame) -> str:
        """Extract session identifier"""
        if 'session_id' in session_data.columns:
            return str(session_data['session_id'].iloc[0])
        elif 'session_date' in session_data.columns:
            return f"session_{session_data['session_date'].iloc[0]}"
        else:
            return f"session_{time.time():.0f}"


# Export all tools
__all__ = [
    "ZoneAnalyzer",
    "TemporalNonLocalityValidator",
    "DimensionalAnchorCalculator", 
    "TheoryBValidator",
    "ArchaeologicalZoneData",
    "TemporalEcho",
    "TheoryBResult",
    "RangeCalculationMethod"
]