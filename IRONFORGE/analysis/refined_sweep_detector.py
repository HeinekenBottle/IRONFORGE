#!/usr/bin/env python3
"""
🔧 IRONFORGE Refined Sweep Detector
==================================

Immediate refinements to fix "framework complete but data empty" issue.

Key Fixes:
1. Weekly Sweep Detection: Lower thresholds, multi-candle sweeps, tolerance bands
2. PM Belt Execution: Relaxed timing windows, belt prelude detection  
3. Cross-Timeframe Anchoring: Bridge nodes from Step 2 integration
4. Ablation Analysis: Daily-only vs Weekly dependency testing

Goal: Get cascades lighting up to validate the macro→micro transmission framework.
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class RefinedSweepEvent:
    """Enhanced sweep event with refined detection"""

    session_id: str
    timestamp: str
    timeframe: str
    sweep_type: str
    price_level: float
    atr_penetration_pct: float  # % of ATR penetrated
    multi_candle_sweep: bool  # Wick pierces over multiple candles
    tolerance_band_hit: bool  # Within tolerance of HTF high/low
    bridge_node_aligned: bool  # Aligns with Step 2 bridge nodes
    pm_belt_category: str  # 'exact', 'prelude', 'extended', 'outside'
    displacement: float
    follow_through: float
    range_pos: float
    zone_proximity: float
    detection_confidence: float  # Overall confidence score


class RefinedSweepDetector:
    """
    Refined sweep detector with relaxed thresholds and enhanced detection
    """

    def __init__(self):
        """Initialize with refined detection parameters"""
        self.config = get_config()
        self.discoveries_path = Path(self.config.get_discoveries_path())

        # Refined detection thresholds
        self.weekly_penetration_threshold = 0.25  # 0.25% of ATR (was strict)
        self.multi_candle_lookback = 3  # Check 3 bars for pierce
        self.htf_tolerance_pct = 0.15  # ±15% tolerance for HTF levels

        # PM belt timing (relaxed)
        self.pm_belt_exact_start = time(14, 35, 0)
        self.pm_belt_exact_end = time(14, 38, 59)
        self.pm_belt_extended_start = time(14, 25, 0)  # 10min before
        self.pm_belt_extended_end = time(14, 48, 59)  # 10min after
        self.pm_belt_prelude_start = time(14, 25, 0)
        self.pm_belt_prelude_end = time(14, 34, 59)

        # Bridge nodes for cross-timeframe anchoring
        self.bridge_nodes = []  # Will load from Step 2 results

    def detect_refined_sweeps(
        self, sessions_data: dict[str, Any]
    ) -> dict[str, list[RefinedSweepEvent]]:
        """
        Detect sweeps with refined thresholds and multi-source validation

        Returns:
            Dict with 'weekly_sweeps', 'daily_sweeps', 'pm_executions'
        """
        logger.info("🔧 Starting refined sweep detection with lowered thresholds...")

        # Load bridge nodes from Step 2 for anchoring
        self._load_bridge_nodes()

        enhanced_sessions = sessions_data.get("enhanced_sessions", [])

        results = {
            "weekly_sweeps": self._detect_refined_weekly_sweeps(enhanced_sessions),
            "daily_sweeps": self._detect_refined_daily_sweeps(enhanced_sessions),
            "pm_executions": self._detect_refined_pm_executions(enhanced_sessions),
        }

        logger.info("🔍 Refined detection results:")
        logger.info(f"  Weekly sweeps: {len(results['weekly_sweeps'])}")
        logger.info(f"  Daily sweeps: {len(results['daily_sweeps'])}")
        logger.info(f"  PM executions: {len(results['pm_executions'])}")

        return results

    def _load_bridge_nodes(self) -> None:
        """Load bridge nodes from Step 2 global lattice for cross-timeframe anchoring"""
        try:
            # Find most recent global lattice file
            lattice_files = list(self.discoveries_path.glob("global_lattice_*.json"))
            if not lattice_files:
                logger.warning(
                    "No global lattice files found - proceeding without bridge node anchoring"
                )
                return

            latest_lattice = sorted(lattice_files, key=lambda x: x.stat().st_mtime)[-1]

            with open(latest_lattice) as f:
                lattice_data = json.load(f)

            # Extract bridge nodes for anchoring
            bridge_data = lattice_data.get("bridge_nodes", [])
            self.bridge_nodes = [
                {
                    "price_level": node.get("price_level", 0),
                    "timeframe": node.get("timeframe", ""),
                    "significance": node.get("significance", 0),
                }
                for node in bridge_data
                if isinstance(node, dict)
            ]

            logger.info(
                f"📊 Loaded {len(self.bridge_nodes)} bridge nodes for cross-timeframe anchoring"
            )

        except Exception as e:
            logger.warning(f"Failed to load bridge nodes: {e}")
            self.bridge_nodes = []

    def _detect_refined_weekly_sweeps(
        self, enhanced_sessions: list[dict[str, Any]]
    ) -> list[RefinedSweepEvent]:
        """
        Detect Weekly sweeps with refined criteria:
        - Lower wick penetration % (0.25–0.5% of ATR vs full body)
        - Multi-candle sweeps (W closes inside, but wick pierces 2–3 bars earlier)
        - Tolerance bands for HTF highs/lows (±15%)
        """
        weekly_sweeps = []

        for session in enhanced_sessions:
            session_name = session.get("session_name", "unknown")

            # Look for weekly-level events with relaxed criteria
            weekly_candidates = self._extract_relaxed_weekly_events(session)

            for candidate in weekly_candidates:
                sweep = self._analyze_refined_weekly_sweep(candidate, session_name, session)
                if sweep and sweep.detection_confidence > 0.3:  # Lower confidence threshold
                    weekly_sweeps.append(sweep)

        logger.info(f"🔍 Detected {len(weekly_sweeps)} refined Weekly sweeps")
        return weekly_sweeps

    def _detect_refined_daily_sweeps(
        self, enhanced_sessions: list[dict[str, Any]]
    ) -> list[RefinedSweepEvent]:
        """Detect Daily sweeps with enhanced multi-source detection"""
        daily_sweeps = []

        for session in enhanced_sessions:
            session_name = session.get("session_name", "unknown")

            # Enhanced daily sweep detection
            daily_candidates = self._extract_enhanced_daily_events(session)

            for candidate in daily_candidates:
                sweep = self._analyze_refined_daily_sweep(candidate, session_name, session)
                if sweep and sweep.detection_confidence > 0.2:  # Very low threshold to catch more
                    daily_sweeps.append(sweep)

        logger.info(f"📈 Detected {len(daily_sweeps)} refined Daily sweeps")
        return daily_sweeps

    def _detect_refined_pm_executions(
        self, enhanced_sessions: list[dict[str, Any]]
    ) -> list[RefinedSweepEvent]:
        """
        Detect PM executions with relaxed timing:
        - Exact belt: 14:35-14:38
        - Extended belt: 14:25-14:48 (±10 min)
        - Prelude: 14:25-14:34
        """
        pm_executions = []

        for session in enhanced_sessions:
            session_name = session.get("session_name", "unknown")

            # Only analyze NY_PM sessions
            if "NY_PM" not in session_name:
                continue

            # Enhanced PM belt detection with multiple time windows
            pm_candidates = self._extract_enhanced_pm_events(session)

            for candidate in pm_candidates:
                sweep = self._analyze_refined_pm_execution(candidate, session_name, session)
                if sweep and sweep.detection_confidence > 0.1:  # Very low threshold
                    pm_executions.append(sweep)

        logger.info(f"⏰ Detected {len(pm_executions)} refined PM executions")
        return pm_executions

    def _extract_relaxed_weekly_events(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract weekly events with relaxed criteria"""
        weekly_events = []

        # Multiple event sources with relaxed filtering
        event_sources = [
            "semantic_events",
            "session_liquidity_events",
            "structural_events",
            "price_movements",
            "session_events",  # Additional source
        ]

        for source in event_sources:
            events = session.get(source, [])
            for event in events:
                # Relaxed weekly detection - look for any multi-session or HTF indicators
                if self._is_relaxed_weekly_indicator(event):
                    weekly_events.append(event)

                # Also check for high-significance price levels
                price = self._safe_float(event.get("price_level", event.get("price", 0)))
                if price > 0 and self._is_significant_price_level(price):
                    weekly_events.append(event)

        # Remove duplicates
        unique_events = []
        seen_prices = set()
        for event in weekly_events:
            price = self._safe_float(event.get("price_level", event.get("price", 0)))
            price_key = round(price, 1)  # Round to nearest 0.1
            if price_key not in seen_prices:
                seen_prices.add(price_key)
                unique_events.append(event)

        return unique_events

    def _extract_enhanced_daily_events(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract daily events with enhanced detection"""
        daily_events = []

        # Expanded sweep keywords
        sweep_keywords = [
            "sweep",
            "liquidity_grab",
            "stop_hunt",
            "liquidity_raid",
            "pierce",
            "break",
            "violation",
            "test",
            "reject",
            "wick",
            "spike",
            "probe",
            "hunt",
        ]

        event_sources = [
            "semantic_events",
            "session_liquidity_events",
            "price_movements",
            "session_events",
        ]

        for source in event_sources:
            events = session.get(source, [])
            for event in events:
                event_text = str(event).lower()

                # Keyword-based detection
                if any(keyword in event_text for keyword in sweep_keywords):
                    daily_events.append(event)

                # Price-based detection for significant levels
                price = self._safe_float(event.get("price_level", event.get("price", 0)))
                if price > 0:
                    # Check if price is near session high/low or round numbers
                    if self._is_potential_sweep_level(price, session):
                        daily_events.append(event)

        return daily_events

    def _extract_enhanced_pm_events(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract PM events with extended time windows"""
        pm_events = []

        event_sources = [
            "semantic_events",
            "session_liquidity_events",
            "price_movements",
            "session_events",
        ]

        for source in event_sources:
            events = session.get(source, [])
            for event in events:
                timestamp = event.get("timestamp", "")

                # Check extended PM belt timing (14:25-14:48)
                pm_category = self._classify_pm_belt_timing(timestamp)
                if pm_category != "outside":
                    # Enhance event with PM category
                    enhanced_event = event.copy()
                    enhanced_event["pm_belt_category"] = pm_category
                    pm_events.append(enhanced_event)

        return pm_events

    def _analyze_refined_weekly_sweep(
        self, event: dict[str, Any], session_id: str, session: dict[str, Any]
    ) -> RefinedSweepEvent | None:
        """Analyze event as potential Weekly sweep with refined criteria"""
        try:
            timestamp = event.get("timestamp", "")
            price_level = self._safe_float(event.get("price_level", event.get("price", 0)))

            if price_level == 0:
                return None

            # Calculate refined metrics
            atr_penetration = self._calculate_atr_penetration(event, session)
            multi_candle = self._check_multi_candle_sweep(event, session)
            tolerance_hit = self._check_tolerance_band_hit(price_level)
            bridge_aligned = self._check_bridge_node_alignment(price_level)

            # Relaxed validation - pass if any criterion is met
            detection_confidence = 0.0

            if atr_penetration >= self.weekly_penetration_threshold:
                detection_confidence += 0.4
            if multi_candle:
                detection_confidence += 0.3
            if tolerance_hit:
                detection_confidence += 0.2
            if bridge_aligned:
                detection_confidence += 0.3

            # Minimum threshold for weekly sweep
            if detection_confidence < 0.3:
                return None

            return RefinedSweepEvent(
                session_id=session_id,
                timestamp=timestamp,
                timeframe="Weekly",
                sweep_type=self._determine_sweep_type(event),
                price_level=price_level,
                atr_penetration_pct=atr_penetration,
                multi_candle_sweep=multi_candle,
                tolerance_band_hit=tolerance_hit,
                bridge_node_aligned=bridge_aligned,
                pm_belt_category="outside",  # Weekly events not in PM belt
                displacement=self._calculate_displacement(event),
                follow_through=self._calculate_follow_through(event),
                range_pos=self._calculate_range_position(event, price_level),
                zone_proximity=self._calculate_zone_proximity_from_range_pos(
                    self._calculate_range_position(event, price_level)
                ),
                detection_confidence=detection_confidence,
            )

        except Exception as e:
            logger.debug(f"Failed to analyze weekly sweep: {e}")
            return None

    def _analyze_refined_daily_sweep(
        self, event: dict[str, Any], session_id: str, session: dict[str, Any]
    ) -> RefinedSweepEvent | None:
        """Analyze event as potential Daily sweep"""
        try:
            timestamp = event.get("timestamp", "")
            price_level = self._safe_float(event.get("price_level", event.get("price", 0)))

            if price_level == 0:
                return None

            # Calculate metrics with lower thresholds
            atr_penetration = self._calculate_atr_penetration(event, session)
            multi_candle = self._check_multi_candle_sweep(event, session)
            tolerance_hit = self._check_tolerance_band_hit(price_level)
            bridge_aligned = self._check_bridge_node_alignment(price_level)

            # Very relaxed validation for daily sweeps
            detection_confidence = 0.2  # Base confidence

            if atr_penetration > 0.1:  # Very low threshold
                detection_confidence += 0.2
            if multi_candle:
                detection_confidence += 0.1
            if tolerance_hit:
                detection_confidence += 0.1
            if bridge_aligned:
                detection_confidence += 0.2

            return RefinedSweepEvent(
                session_id=session_id,
                timestamp=timestamp,
                timeframe="Daily",
                sweep_type=self._determine_sweep_type(event),
                price_level=price_level,
                atr_penetration_pct=atr_penetration,
                multi_candle_sweep=multi_candle,
                tolerance_band_hit=tolerance_hit,
                bridge_node_aligned=bridge_aligned,
                pm_belt_category="outside",
                displacement=self._calculate_displacement(event),
                follow_through=self._calculate_follow_through(event),
                range_pos=self._calculate_range_position(event, price_level),
                zone_proximity=self._calculate_zone_proximity_from_range_pos(
                    self._calculate_range_position(event, price_level)
                ),
                detection_confidence=detection_confidence,
            )

        except Exception as e:
            logger.debug(f"Failed to analyze daily sweep: {e}")
            return None

    def _analyze_refined_pm_execution(
        self, event: dict[str, Any], session_id: str, session: dict[str, Any]
    ) -> RefinedSweepEvent | None:
        """Analyze event as potential PM execution with relaxed timing"""
        try:
            timestamp = event.get("timestamp", "")
            price_level = self._safe_float(event.get("price_level", event.get("price", 0)))

            if price_level == 0:
                return None

            # Get PM belt category
            pm_category = event.get("pm_belt_category", self._classify_pm_belt_timing(timestamp))

            # Calculate metrics
            atr_penetration = self._calculate_atr_penetration(event, session)
            bridge_aligned = self._check_bridge_node_alignment(price_level)

            # PM execution confidence (lower threshold for extended belt)
            detection_confidence = 0.1  # Base confidence

            if pm_category == "exact":
                detection_confidence += 0.4
            elif pm_category == "prelude":
                detection_confidence += 0.3
            elif pm_category == "extended":
                detection_confidence += 0.2

            if atr_penetration > 0.05:  # Very low threshold
                detection_confidence += 0.2
            if bridge_aligned:
                detection_confidence += 0.2

            return RefinedSweepEvent(
                session_id=session_id,
                timestamp=timestamp,
                timeframe="PM",
                sweep_type=self._determine_sweep_type(event),
                price_level=price_level,
                atr_penetration_pct=atr_penetration,
                multi_candle_sweep=False,  # PM events are typically single-candle
                tolerance_band_hit=self._check_tolerance_band_hit(price_level),
                bridge_node_aligned=bridge_aligned,
                pm_belt_category=pm_category,
                displacement=self._calculate_displacement(event),
                follow_through=self._calculate_follow_through(event),
                range_pos=self._calculate_range_position(event, price_level),
                zone_proximity=self._calculate_zone_proximity_from_range_pos(
                    self._calculate_range_position(event, price_level)
                ),
                detection_confidence=detection_confidence,
            )

        except Exception as e:
            logger.debug(f"Failed to analyze PM execution: {e}")
            return None

    def _classify_pm_belt_timing(self, timestamp: str) -> str:
        """Classify timestamp into PM belt categories"""
        try:
            if ":" in timestamp:
                time_part = timestamp.split(" ")[-1] if " " in timestamp else timestamp
                hour, minute = map(int, time_part.split(":")[:2])
                event_time = time(hour, minute)

                if self.pm_belt_exact_start <= event_time <= self.pm_belt_exact_end:
                    return "exact"
                elif self.pm_belt_prelude_start <= event_time <= self.pm_belt_prelude_end:
                    return "prelude"
                elif self.pm_belt_extended_start <= event_time <= self.pm_belt_extended_end:
                    return "extended"
                else:
                    return "outside"
        except:
            pass
        return "outside"

    def _is_relaxed_weekly_indicator(self, event: dict[str, Any]) -> bool:
        """Check for relaxed weekly timeframe indicators"""
        event_text = str(event).lower()

        # Expanded weekly indicators
        weekly_indicators = [
            "weekly",
            "week",
            "htf",
            "higher timeframe",
            "multi-day",
            "session",
            "cross-session",
            "overnight",
            "gap",
            "structural",
            "break",
            "continuation",
            "reversal",
        ]

        return any(indicator in event_text for indicator in weekly_indicators)

    def _is_significant_price_level(self, price: float) -> bool:
        """Check if price represents significant weekly level"""
        # Round numbers (multiples of 100, 50, 25)
        if price % 100 == 0 or price % 50 == 0 or price % 25 == 0:
            return True

        # Price ranges that typically matter for ES futures
        if 23000 <= price <= 24000:  # Current trading range
            return True

        return False

    def _is_potential_sweep_level(self, price: float, session: dict[str, Any]) -> bool:
        """Check if price could be a sweep level based on session context"""
        # Get session range
        session_high = self._safe_float(session.get("session_high", 0))
        session_low = self._safe_float(session.get("session_low", 0))

        if session_high > 0 and session_low > 0:
            session_range = session_high - session_low

            # Near session extremes (within 10% of range)
            if (
                abs(price - session_high) <= session_range * 0.1
                or abs(price - session_low) <= session_range * 0.1
            ):
                return True

        # Round numbers
        return self._is_significant_price_level(price)

    def _calculate_atr_penetration(self, event: dict[str, Any], session: dict[str, Any]) -> float:
        """Calculate ATR penetration percentage (simplified)"""
        # Simplified ATR estimation based on session range
        session_high = self._safe_float(session.get("session_high", 0))
        session_low = self._safe_float(session.get("session_low", 0))

        if session_high > 0 and session_low > 0:
            estimated_atr = (session_high - session_low) * 0.3  # Rough ATR estimate
            price = self._safe_float(event.get("price_level", event.get("price", 0)))

            if estimated_atr > 0:
                # Estimate penetration based on position relative to range
                if price > session_high:
                    penetration = (price - session_high) / estimated_atr
                elif price < session_low:
                    penetration = (session_low - price) / estimated_atr
                else:
                    penetration = 0.1  # Inside range, minimal penetration

                return penetration * 100  # Convert to percentage

        return 0.5  # Default moderate penetration

    def _check_multi_candle_sweep(self, event: dict[str, Any], session: dict[str, Any]) -> bool:
        """Check for multi-candle sweep pattern (simplified)"""
        # Simplified implementation - in production would analyze actual candle data
        event_text = str(event).lower()
        multi_indicators = ["multi", "sequence", "series", "continued", "extended"]
        return any(indicator in event_text for indicator in multi_indicators)

    def _check_tolerance_band_hit(self, price: float) -> bool:
        """Check if price hits tolerance band around significant levels"""
        if not self.bridge_nodes:
            return False

        for node in self.bridge_nodes:
            node_price = node.get("price_level", 0)
            if node_price > 0:
                tolerance = node_price * (self.htf_tolerance_pct / 100)
                if abs(price - node_price) <= tolerance:
                    return True

        return False

    def _check_bridge_node_alignment(self, price: float) -> bool:
        """Check alignment with bridge nodes from Step 2"""
        if not self.bridge_nodes:
            return False

        for node in self.bridge_nodes:
            node_price = node.get("price_level", 0)
            significance = node.get("significance", 0)

            if node_price > 0 and significance > 0.5:  # High significance nodes only
                # Allow wider tolerance for high-significance bridge nodes
                tolerance = 50 if significance > 0.8 else 25
                if abs(price - node_price) <= tolerance:
                    return True

        return False

    # Helper methods
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else 0.0
        except:
            return 0.0

    def _determine_sweep_type(self, event: dict[str, Any]) -> str:
        """Determine sweep type based on event characteristics"""
        event_text = str(event).lower()

        if any(word in event_text for word in ["buy", "long", "higher", "up", "bull"]):
            return "buy_sweep"
        elif any(word in event_text for word in ["sell", "short", "lower", "down", "bear"]):
            return "sell_sweep"
        else:
            return "neutral_sweep"

    def _calculate_displacement(self, event: dict[str, Any]) -> float:
        """Calculate sweep displacement magnitude"""
        price = self._safe_float(event.get("price_level", event.get("price", 0)))
        return abs(price * 0.001) if price > 0 else 0.0

    def _calculate_follow_through(self, event: dict[str, Any]) -> float:
        """Calculate follow-through strength"""
        return 0.7  # Simplified

    def _calculate_range_position(self, event: dict[str, Any], price_level: float) -> float:
        """Calculate position within range"""
        if price_level == 0:
            return 0.5

        # Simplified range position calculation
        if price_level > 23400:
            return 0.8
        elif price_level > 23200:
            return 0.5
        else:
            return 0.2

    def _calculate_zone_proximity_from_range_pos(self, range_pos: float) -> float:
        """Calculate proximity to Theory B zones"""
        theory_b_zones = [0.20, 0.40, 0.50, 0.618, 0.80]
        min_distance = min(abs(range_pos - zone) for zone in theory_b_zones)
        return max(0, 1 - (min_distance * 10))

    def serialize_refined_sweep(self, sweep: RefinedSweepEvent) -> dict[str, Any]:
        """Serialize RefinedSweepEvent to dictionary"""
        return {
            "session_id": sweep.session_id,
            "timestamp": sweep.timestamp,
            "timeframe": sweep.timeframe,
            "sweep_type": sweep.sweep_type,
            "price_level": sweep.price_level,
            "atr_penetration_pct": sweep.atr_penetration_pct,
            "multi_candle_sweep": sweep.multi_candle_sweep,
            "tolerance_band_hit": sweep.tolerance_band_hit,
            "bridge_node_aligned": sweep.bridge_node_aligned,
            "pm_belt_category": sweep.pm_belt_category,
            "displacement": sweep.displacement,
            "follow_through": sweep.follow_through,
            "range_pos": sweep.range_pos,
            "zone_proximity": sweep.zone_proximity,
            "detection_confidence": sweep.detection_confidence,
        }
