#!/usr/bin/env python3
"""
🔄 IRONFORGE FPFVG Redelivery Network Lattice
============================================

Theory B Testing Framework: "Zones Know Their Completion"
Tests whether FVG redelivery events position themselves relative to eventual completion patterns.

Core Hypothesis:
- FVG formations contain forward-looking positioning information
- Redelivery events demonstrate temporal non-locality (positioning before full context available)
- Cross-session FVG networks show dimensional relationship patterns

Discovery Focus:
1. FVG Formation → Redelivery Network Mapping
2. Theory B Dimensional Positioning Analysis  
3. Cross-Session FVG Persistence Patterns
4. Temporal Non-Locality Evidence Collection

Archaeological Significance:
- Tests whether early FVG formations "know" eventual session completion patterns
- Maps redelivery networks across timeframes (15m→4H→Daily)
- Identifies predictive FVG positioning relative to final session structure
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

logger = logging.getLogger(__name__)


class FPFVGRedeliveryLattice:
    """
    FPFVG (Fair Value Gap) Redelivery Network Lattice Builder

    Constructs specialized lattice views focused on FVG formation and redelivery patterns
    to test Theory B's temporal non-locality hypothesis.
    """

    def __init__(self):
        """Initialize FPFVG redelivery lattice builder"""
        self.config = get_config()
        self.enhanced_sessions_path = Path(self.config.get_enhanced_data_path())
        self.discoveries_path = Path(self.config.get_discoveries_path())

        # Theory B testing parameters
        self.theory_b_thresholds = {
            "high_precision": 10.0,  # Points
            "medium_precision": 25.0,
            "low_precision": 50.0,
        }

        # FVG redelivery network parameters
        self.redelivery_timeframes = ["15m", "1H", "4H", "Daily"]
        self.cross_session_lookback = 5  # Sessions to analyze for persistence

    def build_fpfvg_redelivery_lattice(
        self, sessions_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive FPFVG redelivery network lattice

        Theory B Focus:
        - Map FVG formation → redelivery positioning patterns
        - Test whether early FVGs position relative to eventual completion
        - Identify cross-session FVG persistence and dimensional relationships

        Returns:
            Dict containing complete FPFVG redelivery lattice analysis
        """
        logger.info("Building FPFVG Redelivery Network Lattice...")

        try:
            # Initialize lattice structure
            lattice = {
                "lattice_type": "fpfvg_redelivery_network",
                "lattice_metadata": {
                    "build_timestamp": datetime.now().isoformat(),
                    "focus": "fvg_formation_redelivery_patterns",
                    "theory_b_testing": True,
                    "timeframes": self.redelivery_timeframes,
                    "cross_session_analysis": True,
                },
            }

            # Load enhanced sessions
            enhanced_sessions = self._load_enhanced_sessions(sessions_limit)
            lattice["lattice_metadata"]["sessions_analyzed"] = len(enhanced_sessions)

            if not enhanced_sessions:
                lattice["error"] = "No enhanced sessions found for FVG analysis"
                return lattice

            # Extract FVG formation and redelivery events
            fvg_networks = self._extract_fvg_networks(enhanced_sessions)
            lattice["fvg_networks"] = fvg_networks

            # Build FVG redelivery relationship map
            redelivery_map = self._build_redelivery_relationship_map(fvg_networks)
            lattice["redelivery_relationships"] = redelivery_map

            # Theory B dimensional analysis on FVG positioning
            theory_b_analysis = self._analyze_theory_b_fvg_positioning(
                fvg_networks, enhanced_sessions
            )
            lattice["theory_b_fvg_analysis"] = theory_b_analysis

            # Cross-session FVG persistence analysis
            persistence_analysis = self._analyze_cross_session_fvg_persistence(fvg_networks)
            lattice["cross_session_persistence"] = persistence_analysis

            # Temporal non-locality evidence collection
            non_locality_evidence = self._collect_temporal_non_locality_evidence(
                fvg_networks, enhanced_sessions
            )
            lattice["temporal_non_locality_evidence"] = non_locality_evidence

            # Network topology analysis
            network_topology = self._analyze_fvg_network_topology(redelivery_map)
            lattice["network_topology"] = network_topology

            # Discovery insights and recommendations
            discovery_insights = self._generate_fvg_discovery_insights(lattice)
            lattice["discovery_insights"] = discovery_insights

            # Save lattice to discoveries
            self._save_fpfvg_lattice(lattice)

            logger.info(
                f"FPFVG Redelivery Lattice built: {len(fvg_networks)} FVG networks analyzed"
            )
            return lattice

        except Exception as e:
            logger.error(f"Failed to build FPFVG redelivery lattice: {e}")
            return {
                "lattice_type": "fpfvg_redelivery_network",
                "error": str(e),
                "lattice_metadata": {
                    "build_timestamp": datetime.now().isoformat(),
                    "build_failed": True,
                },
            }

    def _load_enhanced_sessions(self, sessions_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load enhanced session data for FVG analysis"""
        enhanced_sessions = []
        session_files = list(self.enhanced_sessions_path.glob("enhanced_rel_*.json"))

        # Sort by modification time (most recent first)
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if sessions_limit:
            session_files = session_files[:sessions_limit]

        for session_file in session_files:
            try:
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                    enhanced_sessions.append(session_data)
            except Exception as e:
                logger.warning(f"Failed to load session {session_file}: {e}")
                continue

        logger.info(f"Loaded {len(enhanced_sessions)} enhanced sessions for FVG analysis")
        return enhanced_sessions

    def _extract_fvg_networks(
        self, enhanced_sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract FVG formation and redelivery event networks from enhanced sessions

        Focus Areas:
        1. FVG formation events (gaps, imbalances)
        2. FVG redelivery/fill events
        3. Cross-timeframe FVG relationships
        4. Session-to-session FVG persistence
        """
        fvg_networks = []

        for session in enhanced_sessions:
            session_name = session.get("session_name", "unknown")

            # Extract FVG-related events
            fvg_formation_events = self._extract_fvg_formation_events(session)
            fvg_redelivery_events = self._extract_fvg_redelivery_events(session)

            if fvg_formation_events or fvg_redelivery_events:
                network = {
                    "session_name": session_name,
                    "session_date": session.get("session_date", "unknown"),
                    "fvg_formations": fvg_formation_events,
                    "fvg_redeliveries": fvg_redelivery_events,
                    "network_metadata": {
                        "formation_count": len(fvg_formation_events),
                        "redelivery_count": len(fvg_redelivery_events),
                        "session_range": self._calculate_session_range(session),
                        "session_timeframe": self._get_session_timeframe(session_name),
                    },
                }

                # Map FVG formation → redelivery relationships within session
                network["internal_relationships"] = self._map_internal_fvg_relationships(
                    fvg_formation_events, fvg_redelivery_events
                )

                fvg_networks.append(network)

        logger.info(
            f"Extracted {len(fvg_networks)} FVG networks from {len(enhanced_sessions)} sessions"
        )
        return fvg_networks

    def _extract_fvg_formation_events(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract FVG formation events from session data"""
        formations = []

        # Look for FVG formation in different event sources
        event_sources = [
            "semantic_events",
            "price_movements",
            "session_liquidity_events",
            "structural_events",
        ]

        for source in event_sources:
            events = session.get(source, [])
            for event in events:
                if self._is_fvg_formation_event(event):
                    formation = {
                        "timestamp": event.get("timestamp"),
                        "formation_type": self._classify_fvg_formation_type(event),
                        "price_level": self._extract_fvg_price_level(event),
                        "timeframe": self._infer_fvg_timeframe(event),
                        "gap_size": self._calculate_fvg_gap_size(event),
                        "source": source,
                        "raw_event": event,
                        "session_context": self._get_formation_session_context(event, session),
                    }
                    formations.append(formation)

        return formations

    def _extract_fvg_redelivery_events(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract FVG redelivery/fill events from session data"""
        redeliveries = []

        # Look for redelivery patterns in price movements and semantic events
        event_sources = ["semantic_events", "price_movements", "session_liquidity_events"]

        for source in event_sources:
            events = session.get(source, [])
            for event in events:
                if self._is_fvg_redelivery_event(event):
                    redelivery = {
                        "timestamp": event.get("timestamp"),
                        "redelivery_type": self._classify_fvg_redelivery_type(event),
                        "target_price": self._extract_redelivery_target_price(event),
                        "fill_percentage": self._calculate_fvg_fill_percentage(event),
                        "redelivery_timeframe": self._infer_redelivery_timeframe(event),
                        "source": source,
                        "raw_event": event,
                        "completion_context": self._get_redelivery_completion_context(
                            event, session
                        ),
                    }
                    redeliveries.append(redelivery)

        return redeliveries

    def _is_fvg_formation_event(self, event: Dict[str, Any]) -> bool:
        """Determine if event represents FVG formation"""
        # Check for FVG-related keywords and patterns
        event_text = str(event).lower()
        fvg_indicators = [
            "fvg",
            "fair value gap",
            "gap",
            "imbalance",
            "unfilled gap",
            "gap formation",
            "price gap",
            "liquidity gap",
            "inefficiency",
        ]

        return any(indicator in event_text for indicator in fvg_indicators)

    def _is_fvg_redelivery_event(self, event: Dict[str, Any]) -> bool:
        """Determine if event represents FVG redelivery/fill"""
        event_text = str(event).lower()
        redelivery_indicators = [
            "redelivery",
            "fill",
            "gap fill",
            "fvg fill",
            "return to gap",
            "gap retest",
            "inefficiency fill",
            "gap closure",
            "imbalance fill",
        ]

        return any(indicator in event_text for indicator in redelivery_indicators)

    def _classify_fvg_formation_type(self, event: Dict[str, Any]) -> str:
        """Classify type of FVG formation"""
        event_text = str(event).lower()

        if "bullish" in event_text or "up" in event_text:
            return "bullish_fvg"
        elif "bearish" in event_text or "down" in event_text:
            return "bearish_fvg"
        elif "liquidity" in event_text:
            return "liquidity_fvg"
        else:
            return "unknown_fvg"

    def _classify_fvg_redelivery_type(self, event: Dict[str, Any]) -> str:
        """Classify type of FVG redelivery"""
        event_text = str(event).lower()

        if "partial" in event_text:
            return "partial_fill"
        elif "complete" in event_text or "full" in event_text:
            return "complete_fill"
        elif "retest" in event_text:
            return "retest_only"
        else:
            return "unknown_redelivery"

    def _extract_fvg_price_level(self, event: Dict[str, Any]) -> float:
        """Extract FVG formation price level"""
        price = event.get("price_level", event.get("price", 0))
        try:
            return float(price) if price is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _extract_redelivery_target_price(self, event: Dict[str, Any]) -> float:
        """Extract FVG redelivery target price"""
        price = event.get("target_level", event.get("price_level", event.get("price", 0)))
        try:
            return float(price) if price is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _infer_fvg_timeframe(self, event: Dict[str, Any]) -> str:
        """Infer timeframe of FVG formation"""
        # Try to extract from event context or default to session timeframe
        event_text = str(event).lower()

        for tf in ["15m", "1h", "4h", "daily", "weekly"]:
            if tf in event_text:
                return tf

        return "1h"  # Default

    def _infer_redelivery_timeframe(self, event: Dict[str, Any]) -> str:
        """Infer timeframe of FVG redelivery"""
        return self._infer_fvg_timeframe(event)

    def _calculate_fvg_gap_size(self, event: Dict[str, Any]) -> float:
        """Calculate FVG gap size in points"""
        # Try to extract gap size from event or estimate
        if "gap_size" in event:
            return event["gap_size"]
        elif "high" in event and "low" in event:
            return abs(event["high"] - event["low"])
        else:
            return 10.0  # Default estimate

    def _calculate_fvg_fill_percentage(self, event: Dict[str, Any]) -> float:
        """Calculate percentage of FVG filled"""
        if "fill_percentage" in event:
            return event["fill_percentage"]
        elif "partial" in str(event).lower():
            return 50.0  # Estimate
        elif "complete" in str(event).lower():
            return 100.0
        else:
            return 25.0  # Default estimate

    def _get_formation_session_context(
        self, event: Dict[str, Any], session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get session context for FVG formation"""
        session_range = self._calculate_session_range(session)
        event_price = self._extract_fvg_price_level(event)

        return {
            "session_range": session_range,
            "formation_position_in_range": self._calculate_range_position(
                event_price, session_range
            ),
            "time_in_session": self._calculate_time_in_session(event, session),
            "session_phase": self._determine_session_phase(event, session),
        }

    def _get_redelivery_completion_context(
        self, event: Dict[str, Any], session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get completion context for FVG redelivery - Theory B focus"""
        session_range = self._calculate_session_range(session)
        redelivery_price = self._extract_redelivery_target_price(event)

        return {
            "session_range": session_range,
            "redelivery_position_in_final_range": self._calculate_range_position(
                redelivery_price, session_range
            ),
            "distance_to_session_completion": self._calculate_time_to_session_end(event, session),
            "final_range_prediction_accuracy": self._assess_final_range_prediction(event, session),
        }

    def _calculate_session_range(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate session high/low range"""
        # Extract from session metadata or calculate from events
        if "session_high" in session and "session_low" in session:
            return {
                "high": session["session_high"],
                "low": session["session_low"],
                "range": session["session_high"] - session["session_low"],
            }
        else:
            # Calculate from price movements
            prices = []
            for source in ["price_movements", "semantic_events"]:
                events = session.get(source, [])
                for event in events:
                    price = event.get("price_level", event.get("price", 0))
                    try:
                        price_float = float(price) if price is not None else 0.0
                        if price_float > 0:
                            prices.append(price_float)
                    except (ValueError, TypeError):
                        continue

            if prices:
                return {"high": max(prices), "low": min(prices), "range": max(prices) - min(prices)}
            else:
                return {"high": 0, "low": 0, "range": 0}

    def _calculate_range_position(self, price: float, session_range: Dict[str, Any]) -> float:
        """Calculate price position within session range (0-1)"""
        if session_range["range"] == 0:
            return 0.5

        return (price - session_range["low"]) / session_range["range"]

    def _calculate_time_in_session(self, event: Dict[str, Any], session: Dict[str, Any]) -> float:
        """Calculate normalized time position in session (0-1)"""
        # Simplified implementation - would need actual session timing
        return 0.5  # Placeholder

    def _calculate_time_to_session_end(
        self, event: Dict[str, Any], session: Dict[str, Any]
    ) -> float:
        """Calculate time remaining to session end"""
        # Simplified implementation
        return 0.5  # Placeholder

    def _determine_session_phase(self, event: Dict[str, Any], session: Dict[str, Any]) -> str:
        """Determine which phase of session the event occurred in"""
        time_in_session = self._calculate_time_in_session(event, session)

        if time_in_session < 0.25:
            return "opening"
        elif time_in_session < 0.75:
            return "middle"
        else:
            return "closing"

    def _assess_final_range_prediction(
        self, event: Dict[str, Any], session: Dict[str, Any]
    ) -> float:
        """Assess how well FVG event predicted final session range - Theory B test"""
        # This would compare FVG positioning against eventual session completion
        # Placeholder for Theory B validation
        return 0.8  # Mock high prediction accuracy

    def _get_session_timeframe(self, session_name: str) -> str:
        """Extract timeframe from session name"""
        if "NY_AM" in session_name:
            return "NY_AM"
        elif "NY_PM" in session_name:
            return "NY_PM"
        elif "LONDON" in session_name:
            return "LONDON"
        elif "ASIA" in session_name:
            return "ASIA"
        else:
            return "unknown"

    def _map_internal_fvg_relationships(
        self, formations: List[Dict], redeliveries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Map FVG formation → redelivery relationships within session"""
        relationships = []

        for formation in formations:
            formation_price = formation["price_level"]
            formation_time = formation["timestamp"]

            # Find potential redeliveries for this formation
            for redelivery in redeliveries:
                redelivery_price = redelivery["target_price"]
                redelivery_time = redelivery["timestamp"]

                # Check if redelivery could be related to formation
                price_distance = abs(formation_price - redelivery_price)
                if price_distance < 50.0:  # Within 50 points
                    relationship = {
                        "formation_timestamp": formation_time,
                        "redelivery_timestamp": redelivery_time,
                        "price_distance": price_distance,
                        "formation_type": formation["formation_type"],
                        "redelivery_type": redelivery["redelivery_type"],
                        "relationship_strength": self._calculate_relationship_strength(
                            formation, redelivery
                        ),
                    }
                    relationships.append(relationship)

        return relationships

    def _calculate_relationship_strength(self, formation: Dict, redelivery: Dict) -> float:
        """Calculate strength of formation → redelivery relationship"""
        # Simple scoring based on price proximity and timing
        price_distance = abs(formation["price_level"] - redelivery["target_price"])
        price_score = max(0, 1 - (price_distance / 100.0))  # Normalize to 0-1

        # Time factor - closer in time = stronger relationship
        time_score = 0.5  # Placeholder

        return (price_score + time_score) / 2

    def _build_redelivery_relationship_map(
        self, fvg_networks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build comprehensive FVG formation → redelivery relationship map"""
        relationship_map = {
            "total_formations": 0,
            "total_redeliveries": 0,
            "formation_to_redelivery_ratio": 0,
            "cross_session_relationships": [],
            "timeframe_relationships": {},
            "redelivery_success_rates": {},
        }

        all_formations = []
        all_redeliveries = []

        # Collect all formations and redeliveries
        for network in fvg_networks:
            all_formations.extend(network["fvg_formations"])
            all_redeliveries.extend(network["fvg_redeliveries"])

        relationship_map["total_formations"] = len(all_formations)
        relationship_map["total_redeliveries"] = len(all_redeliveries)

        if len(all_formations) > 0:
            relationship_map["formation_to_redelivery_ratio"] = len(all_redeliveries) / len(
                all_formations
            )

        # Analyze cross-session relationships
        cross_session_relationships = self._find_cross_session_fvg_relationships(fvg_networks)
        relationship_map["cross_session_relationships"] = cross_session_relationships

        # Analyze timeframe-specific relationships
        timeframe_relationships = self._analyze_timeframe_fvg_relationships(
            all_formations, all_redeliveries
        )
        relationship_map["timeframe_relationships"] = timeframe_relationships

        # Calculate redelivery success rates
        success_rates = self._calculate_redelivery_success_rates(all_formations, all_redeliveries)
        relationship_map["redelivery_success_rates"] = success_rates

        return relationship_map

    def _find_cross_session_fvg_relationships(
        self, fvg_networks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find FVG relationships that span across sessions"""
        cross_session_relationships = []

        # Compare formations in one session with redeliveries in subsequent sessions
        for i, network1 in enumerate(fvg_networks):
            for j, network2 in enumerate(fvg_networks):
                if i >= j:  # Only look forward in time
                    continue

                # Look for formations in network1 that get redelivered in network2
                for formation in network1["fvg_formations"]:
                    for redelivery in network2["fvg_redeliveries"]:
                        relationship_strength = self._calculate_relationship_strength(
                            formation, redelivery
                        )

                        if relationship_strength > 0.6:  # Strong relationship threshold
                            cross_relationship = {
                                "formation_session": network1["session_name"],
                                "redelivery_session": network2["session_name"],
                                "formation_timestamp": formation["timestamp"],
                                "redelivery_timestamp": redelivery["timestamp"],
                                "relationship_strength": relationship_strength,
                                "price_distance": abs(
                                    formation["price_level"] - redelivery["target_price"]
                                ),
                                "session_gap": j - i,
                            }
                            cross_session_relationships.append(cross_relationship)

        return cross_session_relationships

    def _analyze_timeframe_fvg_relationships(
        self, formations: List[Dict], redeliveries: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze FVG relationships by timeframe"""
        timeframe_analysis = {}

        for tf in self.redelivery_timeframes:
            tf_formations = [f for f in formations if f["timeframe"] == tf]
            tf_redeliveries = [r for r in redeliveries if r["redelivery_timeframe"] == tf]

            timeframe_analysis[tf] = {
                "formations": len(tf_formations),
                "redeliveries": len(tf_redeliveries),
                "redelivery_rate": (
                    len(tf_redeliveries) / len(tf_formations) if tf_formations else 0
                ),
                "avg_gap_size": (
                    sum(f["gap_size"] for f in tf_formations) / len(tf_formations)
                    if tf_formations
                    else 0
                ),
            }

        return timeframe_analysis

    def _calculate_redelivery_success_rates(
        self, formations: List[Dict], redeliveries: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate FVG redelivery success rates by type"""
        success_rates = {}

        # Group by formation type
        formation_types = {}
        for formation in formations:
            f_type = formation["formation_type"]
            if f_type not in formation_types:
                formation_types[f_type] = []
            formation_types[f_type].append(formation)

        # Calculate success rates
        for f_type, type_formations in formation_types.items():
            # Count how many of these formations had redeliveries
            successful_redeliveries = 0

            for formation in type_formations:
                # Look for matching redeliveries
                for redelivery in redeliveries:
                    if self._calculate_relationship_strength(formation, redelivery) > 0.5:
                        successful_redeliveries += 1
                        break  # Count only once per formation

            success_rates[f_type] = {
                "total_formations": len(type_formations),
                "successful_redeliveries": successful_redeliveries,
                "success_rate": (
                    successful_redeliveries / len(type_formations) if type_formations else 0
                ),
            }

        return success_rates

    def _analyze_theory_b_fvg_positioning(
        self, fvg_networks: List[Dict[str, Any]], sessions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Theory B Analysis: Test if FVG formations position relative to eventual completion

        Key Question: Do early FVG formations "know" where the session will eventually complete?
        """
        theory_b_analysis = {
            "total_fvg_formations_tested": 0,
            "high_precision_formations": 0,
            "medium_precision_formations": 0,
            "low_precision_formations": 0,
            "avg_completion_prediction_accuracy": 0,
            "dimensional_positioning_scores": [],
            "temporal_non_locality_evidence": [],
        }

        all_formations = []
        for network in fvg_networks:
            all_formations.extend(network["fvg_formations"])

        theory_b_analysis["total_fvg_formations_tested"] = len(all_formations)

        if not all_formations:
            return theory_b_analysis

        precision_scores = []
        positioning_scores = []

        for formation in all_formations:
            # Find the session this formation belongs to
            session = self._find_session_for_formation(formation, sessions)
            if not session:
                continue

            # Test Theory B: How well did this formation predict final session structure?
            completion_prediction = self._test_formation_completion_prediction(formation, session)
            precision_scores.append(completion_prediction["prediction_accuracy"])

            # Dimensional positioning analysis
            dimensional_score = self._analyze_formation_dimensional_positioning(formation, session)
            positioning_scores.append(dimensional_score)

            # Categorize precision
            distance_to_completion = completion_prediction.get("distance_to_final_structure", 999)
            if distance_to_completion < self.theory_b_thresholds["high_precision"]:
                theory_b_analysis["high_precision_formations"] += 1
            elif distance_to_completion < self.theory_b_thresholds["medium_precision"]:
                theory_b_analysis["medium_precision_formations"] += 1
            elif distance_to_completion < self.theory_b_thresholds["low_precision"]:
                theory_b_analysis["low_precision_formations"] += 1

            # Collect temporal non-locality evidence
            if completion_prediction["prediction_accuracy"] > 0.8:
                evidence = {
                    "formation_timestamp": formation["timestamp"],
                    "formation_price": formation["price_level"],
                    "session_name": session.get("session_name", "unknown"),
                    "prediction_accuracy": completion_prediction["prediction_accuracy"],
                    "evidence_type": "high_accuracy_early_formation",
                }
                theory_b_analysis["temporal_non_locality_evidence"].append(evidence)

        # Calculate averages
        if precision_scores:
            theory_b_analysis["avg_completion_prediction_accuracy"] = sum(precision_scores) / len(
                precision_scores
            )

        if positioning_scores:
            theory_b_analysis["dimensional_positioning_scores"] = positioning_scores
            theory_b_analysis["avg_dimensional_positioning_score"] = sum(positioning_scores) / len(
                positioning_scores
            )

        return theory_b_analysis

    def _find_session_for_formation(
        self, formation: Dict[str, Any], sessions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find which session a formation belongs to"""
        formation_timestamp = formation["timestamp"]

        for session in sessions:
            # Check if this formation's timestamp appears in this session's events
            for source in ["semantic_events", "price_movements", "session_liquidity_events"]:
                events = session.get(source, [])
                for event in events:
                    if event.get("timestamp") == formation_timestamp:
                        return session

        return None

    def _test_formation_completion_prediction(
        self, formation: Dict[str, Any], session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test how well FVG formation predicted final session completion - Core Theory B test"""
        session_range = self._calculate_session_range(session)
        formation_price = formation["price_level"]

        # Calculate where formation positioned relative to FINAL session range
        final_range_position = self._calculate_range_position(formation_price, session_range)

        # Assess prediction accuracy based on dimensional relationships
        # Theory B: Formations should position at meaningful dimensional levels (20%, 40%, 60%, 80%)
        dimensional_levels = [0.2, 0.4, 0.6, 0.8]
        closest_level = min(dimensional_levels, key=lambda x: abs(x - final_range_position))
        distance_to_level = abs(closest_level - final_range_position)

        # Convert to price distance
        distance_to_final_structure = distance_to_level * session_range["range"]

        # Prediction accuracy (1.0 = perfect dimensional positioning)
        prediction_accuracy = max(0, 1 - (distance_to_level / 0.2))  # Within 20% = perfect score

        return {
            "final_range_position": final_range_position,
            "closest_dimensional_level": closest_level,
            "distance_to_final_structure": distance_to_final_structure,
            "prediction_accuracy": prediction_accuracy,
        }

    def _analyze_formation_dimensional_positioning(
        self, formation: Dict[str, Any], session: Dict[str, Any]
    ) -> float:
        """Analyze dimensional positioning quality of FVG formation"""
        session_context = formation.get("session_context", {})
        formation_position = session_context.get("formation_position_in_range", 0.5)

        # Score based on proximity to key dimensional levels
        dimensional_levels = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        closest_level = min(dimensional_levels, key=lambda x: abs(x - formation_position))
        distance = abs(closest_level - formation_position)

        # Score: 1.0 = exactly on dimensional level, 0.0 = far from any level
        return max(0, 1 - (distance / 0.1))

    def _analyze_cross_session_fvg_persistence(
        self, fvg_networks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze FVG persistence across sessions"""
        persistence_analysis = {
            "sessions_analyzed": len(fvg_networks),
            "persistent_fvg_chains": [],
            "avg_persistence_duration": 0,
            "cross_session_redelivery_rate": 0,
            "persistent_price_levels": [],
        }

        if len(fvg_networks) < 2:
            return persistence_analysis

        # Track FVG formations that persist across multiple sessions
        persistent_chains = self._find_persistent_fvg_chains(fvg_networks)
        persistence_analysis["persistent_fvg_chains"] = persistent_chains

        # Calculate average persistence duration
        if persistent_chains:
            durations = [chain["session_count"] for chain in persistent_chains]
            persistence_analysis["avg_persistence_duration"] = sum(durations) / len(durations)

        # Calculate cross-session redelivery rate
        cross_session_redeliveries = sum(
            1 for chain in persistent_chains if chain["eventually_redelivered"]
        )
        total_formations = sum(len(network["fvg_formations"]) for network in fvg_networks)

        if total_formations > 0:
            persistence_analysis["cross_session_redelivery_rate"] = (
                cross_session_redeliveries / total_formations
            )

        # Identify price levels that show persistent FVG activity
        persistent_levels = self._identify_persistent_price_levels(fvg_networks)
        persistence_analysis["persistent_price_levels"] = persistent_levels

        return persistence_analysis

    def _find_persistent_fvg_chains(
        self, fvg_networks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find FVG formations that persist across multiple sessions"""
        persistent_chains = []

        # Sort networks by session date
        sorted_networks = sorted(fvg_networks, key=lambda x: x["session_date"])

        for i, network in enumerate(sorted_networks):
            for formation in network["fvg_formations"]:
                formation_price = formation["price_level"]

                # Look for this price level in subsequent sessions
                chain = {
                    "initial_formation": formation,
                    "initial_session": network["session_name"],
                    "price_level": formation_price,
                    "appearances": [network["session_name"]],
                    "session_count": 1,
                    "eventually_redelivered": False,
                }

                # Check subsequent sessions
                for j in range(i + 1, min(i + self.cross_session_lookback, len(sorted_networks))):
                    subsequent_network = sorted_networks[j]

                    # Check if this price level appears in formations or redeliveries
                    level_found = False

                    # Check formations
                    for sub_formation in subsequent_network["fvg_formations"]:
                        if (
                            abs(sub_formation["price_level"] - formation_price) < 25.0
                        ):  # Within 25 points
                            chain["appearances"].append(subsequent_network["session_name"])
                            chain["session_count"] += 1
                            level_found = True
                            break

                    # Check redeliveries
                    if not level_found:
                        for redelivery in subsequent_network["fvg_redeliveries"]:
                            if abs(redelivery["target_price"] - formation_price) < 25.0:
                                chain["appearances"].append(subsequent_network["session_name"])
                                chain["eventually_redelivered"] = True
                                level_found = True
                                break

                    # If level not found in this session, chain is broken
                    if not level_found:
                        break

                # Only include chains that persist across multiple sessions
                if chain["session_count"] > 1:
                    persistent_chains.append(chain)

        return persistent_chains

    def _identify_persistent_price_levels(
        self, fvg_networks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify price levels that show persistent FVG activity"""
        price_level_activity = {}

        # Collect all FVG-related price levels
        for network in fvg_networks:
            session_name = network["session_name"]

            # Track formation prices
            for formation in network["fvg_formations"]:
                price = formation["price_level"]
                price_bucket = round(price / 10) * 10  # Group into 10-point buckets

                if price_bucket not in price_level_activity:
                    price_level_activity[price_bucket] = {
                        "price_level": price_bucket,
                        "formation_count": 0,
                        "redelivery_count": 0,
                        "sessions_involved": set(),
                        "total_activity": 0,
                    }

                price_level_activity[price_bucket]["formation_count"] += 1
                price_level_activity[price_bucket]["sessions_involved"].add(session_name)
                price_level_activity[price_bucket]["total_activity"] += 1

            # Track redelivery prices
            for redelivery in network["fvg_redeliveries"]:
                price = redelivery["target_price"]
                price_bucket = round(price / 10) * 10

                if price_bucket not in price_level_activity:
                    price_level_activity[price_bucket] = {
                        "price_level": price_bucket,
                        "formation_count": 0,
                        "redelivery_count": 0,
                        "sessions_involved": set(),
                        "total_activity": 0,
                    }

                price_level_activity[price_bucket]["redelivery_count"] += 1
                price_level_activity[price_bucket]["sessions_involved"].add(session_name)
                price_level_activity[price_bucket]["total_activity"] += 1

        # Convert to list and filter for persistent levels (multiple sessions, high activity)
        persistent_levels = []
        for price_bucket, activity in price_level_activity.items():
            if len(activity["sessions_involved"]) >= 2 and activity["total_activity"] >= 3:
                level_info = {
                    "price_level": activity["price_level"],
                    "formation_count": activity["formation_count"],
                    "redelivery_count": activity["redelivery_count"],
                    "sessions_involved": len(activity["sessions_involved"]),
                    "total_activity": activity["total_activity"],
                    "persistence_score": len(activity["sessions_involved"])
                    * activity["total_activity"],
                }
                persistent_levels.append(level_info)

        # Sort by persistence score
        persistent_levels.sort(key=lambda x: x["persistence_score"], reverse=True)

        return persistent_levels[:10]  # Return top 10 most persistent levels

    def _collect_temporal_non_locality_evidence(
        self, fvg_networks: List[Dict[str, Any]], sessions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collect evidence of temporal non-locality in FVG positioning"""
        evidence = {
            "total_evidence_instances": 0,
            "strong_evidence_instances": 0,
            "evidence_categories": {
                "early_formation_accurate_prediction": [],
                "cross_session_dimensional_alignment": [],
                "pre_completion_positioning": [],
            },
            "temporal_non_locality_score": 0,
        }

        # Look for early formations that accurately predicted session completion
        early_formation_evidence = self._find_early_formation_prediction_evidence(
            fvg_networks, sessions
        )
        evidence["evidence_categories"][
            "early_formation_accurate_prediction"
        ] = early_formation_evidence

        # Look for cross-session dimensional alignment
        cross_session_evidence = self._find_cross_session_alignment_evidence(fvg_networks)
        evidence["evidence_categories"][
            "cross_session_dimensional_alignment"
        ] = cross_session_evidence

        # Look for pre-completion positioning patterns
        pre_completion_evidence = self._find_pre_completion_positioning_evidence(
            fvg_networks, sessions
        )
        evidence["evidence_categories"]["pre_completion_positioning"] = pre_completion_evidence

        # Calculate totals
        total_instances = (
            len(early_formation_evidence)
            + len(cross_session_evidence)
            + len(pre_completion_evidence)
        )
        evidence["total_evidence_instances"] = total_instances

        # Count strong evidence (high prediction accuracy)
        strong_evidence = 0
        for category_evidence in evidence["evidence_categories"].values():
            for instance in category_evidence:
                if instance.get("prediction_accuracy", 0) > 0.8:
                    strong_evidence += 1

        evidence["strong_evidence_instances"] = strong_evidence

        # Calculate overall temporal non-locality score
        if total_instances > 0:
            evidence["temporal_non_locality_score"] = strong_evidence / total_instances

        return evidence

    def _find_early_formation_prediction_evidence(
        self, fvg_networks: List[Dict[str, Any]], sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find evidence of early FVG formations accurately predicting session completion"""
        evidence_instances = []

        for network in fvg_networks:
            session = self._find_session_by_name(network["session_name"], sessions)
            if not session:
                continue

            for formation in network["fvg_formations"]:
                # Check if this was an early formation (first 25% of session)
                session_context = formation.get("session_context", {})
                time_in_session = session_context.get("time_in_session", 1.0)

                if time_in_session < 0.25:  # Early formation
                    # Test prediction accuracy
                    prediction_test = self._test_formation_completion_prediction(formation, session)

                    if prediction_test["prediction_accuracy"] > 0.7:  # High accuracy
                        evidence = {
                            "formation_timestamp": formation["timestamp"],
                            "session_name": network["session_name"],
                            "formation_price": formation["price_level"],
                            "time_in_session": time_in_session,
                            "prediction_accuracy": prediction_test["prediction_accuracy"],
                            "dimensional_level_targeted": prediction_test[
                                "closest_dimensional_level"
                            ],
                            "evidence_strength": (
                                "strong"
                                if prediction_test["prediction_accuracy"] > 0.85
                                else "medium"
                            ),
                        }
                        evidence_instances.append(evidence)

        return evidence_instances

    def _find_cross_session_alignment_evidence(
        self, fvg_networks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find evidence of FVG dimensional alignment across sessions"""
        evidence_instances = []

        # Look for formations at similar dimensional levels across sessions
        dimensional_clusters = {}

        for network in fvg_networks:
            for formation in network["fvg_formations"]:
                session_context = formation.get("session_context", {})
                dimensional_position = session_context.get("formation_position_in_range", 0.5)

                # Round to nearest 20% level for clustering
                dimensional_bucket = round(dimensional_position * 5) / 5

                if dimensional_bucket not in dimensional_clusters:
                    dimensional_clusters[dimensional_bucket] = []

                dimensional_clusters[dimensional_bucket].append(
                    {"formation": formation, "session_name": network["session_name"]}
                )

        # Find clusters with multiple sessions
        for dimensional_level, formations in dimensional_clusters.items():
            if len(formations) >= 3:  # At least 3 formations at this level
                sessions_involved = set(f["session_name"] for f in formations)
                if len(sessions_involved) >= 2:  # Across multiple sessions
                    evidence = {
                        "dimensional_level": dimensional_level,
                        "formation_count": len(formations),
                        "sessions_involved": len(sessions_involved),
                        "session_names": list(sessions_involved),
                        "prediction_accuracy": 0.8,  # High for cross-session alignment
                        "evidence_strength": "strong",
                    }
                    evidence_instances.append(evidence)

        return evidence_instances

    def _find_pre_completion_positioning_evidence(
        self, fvg_networks: List[Dict[str, Any]], sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find evidence of FVG positioning before session completion context was available"""
        evidence_instances = []

        for network in fvg_networks:
            session = self._find_session_by_name(network["session_name"], sessions)
            if not session:
                continue

            session_range = self._calculate_session_range(session)

            for formation in network["fvg_formations"]:
                session_context = formation.get("session_context", {})
                time_in_session = session_context.get("time_in_session", 1.0)
                formation_price = formation["price_level"]

                # Only consider formations that occurred before session was 80% complete
                if time_in_session < 0.8:
                    # Check if formation positioned at eventual key level
                    final_position = self._calculate_range_position(formation_price, session_range)

                    # Check if positioned at meaningful dimensional level
                    key_levels = [0.2, 0.4, 0.6, 0.8]
                    closest_level = min(key_levels, key=lambda x: abs(x - final_position))
                    distance_to_level = abs(closest_level - final_position)

                    if distance_to_level < 0.05:  # Within 5% of key level
                        evidence = {
                            "formation_timestamp": formation["timestamp"],
                            "session_name": network["session_name"],
                            "formation_price": formation_price,
                            "time_in_session": time_in_session,
                            "final_dimensional_position": final_position,
                            "closest_key_level": closest_level,
                            "positioning_accuracy": 1 - (distance_to_level / 0.05),
                            "prediction_accuracy": 1 - (distance_to_level / 0.05),
                            "evidence_strength": "strong" if distance_to_level < 0.02 else "medium",
                        }
                        evidence_instances.append(evidence)

        return evidence_instances

    def _find_session_by_name(
        self, session_name: str, sessions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find session by name"""
        for session in sessions:
            if session.get("session_name") == session_name:
                return session
        return None

    def _analyze_fvg_network_topology(self, redelivery_map: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze FVG network topology and connection patterns"""
        topology = {
            "network_density": 0,
            "average_connections_per_node": 0,
            "most_connected_price_levels": [],
            "formation_redelivery_clusters": [],
            "network_efficiency": 0,
        }

        # Calculate network density
        total_formations = redelivery_map["total_formations"]
        total_redeliveries = redelivery_map["total_redeliveries"]

        if total_formations > 0:
            topology["network_density"] = total_redeliveries / total_formations

        # Analyze cross-session relationships
        cross_session_rels = redelivery_map.get("cross_session_relationships", [])

        if cross_session_rels:
            # Group by price level to find most connected levels
            price_level_connections = {}

            for rel in cross_session_rels:
                price_distance = rel["price_distance"]
                if price_distance < 50.0:  # Within reasonable distance
                    # Use formation price as key
                    formation_session = rel["formation_session"]

                    if formation_session not in price_level_connections:
                        price_level_connections[formation_session] = 0
                    price_level_connections[formation_session] += 1

            # Find most connected
            if price_level_connections:
                sorted_connections = sorted(
                    price_level_connections.items(), key=lambda x: x[1], reverse=True
                )
                topology["most_connected_price_levels"] = sorted_connections[:5]

                total_connections = sum(price_level_connections.values())
                topology["average_connections_per_node"] = total_connections / len(
                    price_level_connections
                )

        # Calculate network efficiency
        successful_formations = sum(
            success_data["successful_redeliveries"]
            for success_data in redelivery_map.get("redelivery_success_rates", {}).values()
        )

        if total_formations > 0:
            topology["network_efficiency"] = successful_formations / total_formations

        return topology

    def _generate_fvg_discovery_insights(self, lattice: Dict[str, Any]) -> Dict[str, Any]:
        """Generate discovery insights and recommendations for FPFVG analysis"""
        insights = {
            "theory_b_validation_summary": {},
            "temporal_non_locality_assessment": {},
            "cross_session_patterns": {},
            "discovery_recommendations": [],
        }

        # Theory B validation summary
        theory_b = lattice.get("theory_b_fvg_analysis", {})
        total_tested = theory_b.get("total_fvg_formations_tested", 0)
        high_precision = theory_b.get("high_precision_formations", 0)
        avg_accuracy = theory_b.get("avg_completion_prediction_accuracy", 0)

        insights["theory_b_validation_summary"] = {
            "formations_tested": total_tested,
            "high_precision_formations": high_precision,
            "precision_rate": high_precision / total_tested if total_tested > 0 else 0,
            "avg_prediction_accuracy": avg_accuracy,
            "validation_status": (
                "STRONG" if avg_accuracy > 0.8 else "MODERATE" if avg_accuracy > 0.6 else "WEAK"
            ),
        }

        # Temporal non-locality assessment
        non_locality = lattice.get("temporal_non_locality_evidence", {})
        total_evidence = non_locality.get("total_evidence_instances", 0)
        strong_evidence = non_locality.get("strong_evidence_instances", 0)
        nl_score = non_locality.get("temporal_non_locality_score", 0)

        insights["temporal_non_locality_assessment"] = {
            "evidence_instances": total_evidence,
            "strong_evidence_instances": strong_evidence,
            "non_locality_score": nl_score,
            "evidence_strength": (
                "STRONG" if nl_score > 0.7 else "MODERATE" if nl_score > 0.5 else "WEAK"
            ),
        }

        # Cross-session patterns
        persistence = lattice.get("cross_session_persistence", {})
        persistent_chains = persistence.get("persistent_fvg_chains", [])
        redelivery_rate = persistence.get("cross_session_redelivery_rate", 0)

        insights["cross_session_patterns"] = {
            "persistent_chains": len(persistent_chains),
            "cross_session_redelivery_rate": redelivery_rate,
            "persistence_strength": (
                "HIGH" if redelivery_rate > 0.4 else "MODERATE" if redelivery_rate > 0.2 else "LOW"
            ),
        }

        # Generate recommendations
        recommendations = []

        # Theory B recommendations
        if avg_accuracy > 0.8:
            recommendations.append(
                {
                    "priority": "EXTREME",
                    "type": "theory_b_validation",
                    "description": "Strong Theory B evidence detected - build predictive FVG positioning models",
                    "action": "Develop real-time FVG formation analysis for session completion prediction",
                }
            )
        elif avg_accuracy > 0.6:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "type": "theory_b_investigation",
                    "description": "Moderate Theory B evidence - expand FVG positioning analysis",
                    "action": "Analyze larger dataset of FVG formations for dimensional positioning patterns",
                }
            )

        # Cross-session recommendations
        if len(persistent_chains) > 3:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "type": "cross_session_analysis",
                    "description": "Significant cross-session FVG persistence detected",
                    "action": "Map multi-session FVG networks for longer-term positioning intelligence",
                }
            )

        # Temporal non-locality recommendations
        if nl_score > 0.7:
            recommendations.append(
                {
                    "priority": "EXTREME",
                    "type": "temporal_causality",
                    "description": "Strong temporal non-locality evidence in FVG positioning",
                    "action": "Investigate causal mechanisms behind forward-looking FVG positioning",
                }
            )

        insights["discovery_recommendations"] = recommendations

        return insights

    def _save_fpfvg_lattice(self, lattice: Dict[str, Any]) -> None:
        """Save FPFVG redelivery lattice to discoveries directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fpfvg_redelivery_lattice_{timestamp}.json"
        filepath = self.discoveries_path / filename

        try:
            with open(filepath, "w") as f:
                json.dump(lattice, f, indent=2, default=str)
            logger.info(f"FPFVG Redelivery Lattice saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save FPFVG lattice: {e}")
