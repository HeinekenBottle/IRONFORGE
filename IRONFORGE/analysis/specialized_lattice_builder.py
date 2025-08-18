"""
IRONFORGE Specialized Lattice Builder
====================================

STEP 3: Builds specialized lattice views based on global terrain findings.
Focuses on candidate areas identified in the terrain analysis.

Priority 1: NY PM Archaeological Belt (14:35-38 belt with 1m resolution)
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from config import get_config
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import get_config

logger = logging.getLogger(__name__)


class SpecializedLatticeBuilder:
    """
    Builds specialized lattice views for deep archaeological analysis.

    Focus Areas:
    1. NY PM Archaeological Belt (14:35-38 with Theory B validation)
    2. FPFVG Redelivery Networks (4H → 1H → 15m cascades)
    3. Weekly → Daily Liquidity Sweep Cascades
    """

    def __init__(self):
        self.config = get_config()

        # Enhanced sessions paths
        self.enhanced_path = Path(self.config.get_data_path()) / "enhanced"
        self.discoveries_path = Path(self.config.get_discoveries_path())

        logger.info("Specialized Lattice Builder initialized for archaeological deep dive")

    def build_ny_pm_archaeological_belt(self) -> Dict[str, Any]:
        """
        Build specialized lattice for NY PM 14:35-38 archaeological belt

        Focus: Theory B 40% zone validation with 1m resolution
        """
        try:
            logger.info("🏛️ Building NY PM Archaeological Belt lattice (1m resolution)")

            # Find all NY_PM sessions
            ny_pm_files = list(self.enhanced_path.glob("enhanced_rel_NY_PM_*.json"))
            logger.info(f"Found {len(ny_pm_files)} NY PM sessions for belt analysis")

            if not ny_pm_files:
                return self._create_empty_specialized_lattice(
                    "ny_pm_belt", "No NY PM sessions found"
                )

            belt_lattice = {
                "lattice_type": "ny_pm_archaeological_belt",
                "lattice_metadata": {
                    "build_timestamp": datetime.now().isoformat(),
                    "focus_timeframe": "14:35:00 - 14:38:59",
                    "resolution": "1m",
                    "theory_b_validation": True,
                    "sessions_analyzed": len(ny_pm_files),
                },
                "belt_events": [],
                "theory_b_analysis": {},
                "dimensional_relationships": {},
                "archaeological_zones": {},
                "belt_statistics": {},
                "discovery_insights": {},
            }

            # Process each NY PM session
            all_belt_events = []
            theory_b_measurements = []

            for session_file in ny_pm_files:
                session_data = self._load_session_data(session_file)
                if session_data:
                    belt_events = self._extract_belt_events(session_data)
                    theory_b_data = self._analyze_theory_b_relationships(session_data, belt_events)

                    all_belt_events.extend(belt_events)
                    if theory_b_data:
                        theory_b_measurements.append(theory_b_data)

            # Build specialized belt lattice components
            belt_lattice["belt_events"] = all_belt_events
            belt_lattice["theory_b_analysis"] = self._aggregate_theory_b_analysis(
                theory_b_measurements
            )
            belt_lattice["dimensional_relationships"] = self._analyze_dimensional_relationships(
                all_belt_events
            )
            belt_lattice["archaeological_zones"] = self._identify_archaeological_zones(
                all_belt_events
            )
            belt_lattice["belt_statistics"] = self._calculate_belt_statistics(all_belt_events)
            belt_lattice["discovery_insights"] = self._generate_belt_insights(belt_lattice)

            # Save specialized lattice
            self._save_specialized_lattice(belt_lattice, "ny_pm_archaeological_belt")

            logger.info(
                f"✅ NY PM Belt lattice complete: {len(all_belt_events)} belt events, {len(theory_b_measurements)} Theory B measurements"
            )
            return belt_lattice

        except Exception as e:
            logger.error(f"NY PM belt lattice build failed: {e}")
            return {"error": str(e)}

    def _extract_belt_events(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract events from the 14:35-38 archaeological belt"""
        belt_events = []
        session_name = session_data.get("session_name", "unknown")

        # Target belt timeframe
        belt_start = time(14, 35, 0)
        belt_end = time(14, 38, 59)

        # Extract from various event sources
        event_sources = [
            ("session_liquidity_events", "liquidity"),
            ("price_movements", "price_movement"),
            ("session_fpfvg", "fpfvg_interaction"),
        ]

        for source_key, event_type_base in event_sources:
            events = session_data.get(source_key, [])

            # Handle FPFVG special structure
            if source_key == "session_fpfvg":
                fpfvg_data = session_data.get("session_fpfvg", {})
                if fpfvg_data.get("fpfvg_present") and "fpfvg_formation" in fpfvg_data:
                    events = fpfvg_data["fpfvg_formation"].get("interactions", [])
                else:
                    events = []

            for event in events:
                event_time_str = event.get("timestamp") or event.get("interaction_time", "00:00:00")

                # Parse time string
                try:
                    if ":" in event_time_str:
                        time_parts = event_time_str.split(":")
                        hour = int(time_parts[0])
                        minute = int(time_parts[1])
                        second = int(time_parts[2]) if len(time_parts) > 2 else 0
                        event_time = time(hour, minute, second)

                        # Check if event is in belt timeframe
                        if belt_start <= event_time <= belt_end:
                            belt_event = {
                                "session_name": session_name,
                                "timestamp": event_time_str,
                                "time_object": event_time,
                                "event_type": event_type_base,
                                "price": event.get("price_level") or event.get("price", 0),
                                "significance": event.get("intensity")
                                or event.get("significance", 0.5),
                                "archaeological_zone": event.get("zone", "belt_zone"),
                                "source": source_key,
                                "raw_event": event,
                            }
                            belt_events.append(belt_event)

                except (ValueError, IndexError):
                    continue

        return belt_events

    def _analyze_theory_b_relationships(
        self, session_data: Dict[str, Any], belt_events: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze Theory B dimensional relationships for belt events"""
        if not belt_events:
            return None

        session_name = session_data.get("session_name", "unknown")

        # Extract session range information
        energy_state = session_data.get("energy_state", {})
        session_high = energy_state.get("session_high", 0)
        session_low = energy_state.get("session_low", 0)
        session_range = session_high - session_low if session_high > session_low else 0

        if session_range <= 0:
            return None

        # Calculate dimensional levels (Theory B)
        range_40_percent = session_low + (session_range * 0.4)
        range_20_percent = session_low + (session_range * 0.2)
        range_80_percent = session_low + (session_range * 0.8)

        theory_b_analysis = {
            "session_name": session_name,
            "session_high": session_high,
            "session_low": session_low,
            "session_range": session_range,
            "dimensional_levels": {
                "20_percent": range_20_percent,
                "40_percent": range_40_percent,
                "80_percent": range_80_percent,
            },
            "belt_event_measurements": [],
        }

        # Measure each belt event against dimensional levels
        for event in belt_events:
            event_price = event.get("price", 0)
            if event_price <= 0:
                continue

            # Calculate distances to dimensional levels
            distance_to_40 = abs(event_price - range_40_percent)
            distance_to_20 = abs(event_price - range_20_percent)
            distance_to_80 = abs(event_price - range_80_percent)

            # Find closest dimensional level
            distances = {
                "40_percent": distance_to_40,
                "20_percent": distance_to_20,
                "80_percent": distance_to_80,
            }

            closest_level = min(distances, key=distances.get)
            closest_distance = distances[closest_level]

            measurement = {
                "timestamp": event.get("timestamp"),
                "event_price": event_price,
                "distances_to_levels": distances,
                "closest_dimensional_level": closest_level,
                "closest_distance": closest_distance,
                "dimensional_precision": (
                    closest_distance / session_range if session_range > 0 else 1.0
                ),
                "theory_b_score": max(
                    0, 1.0 - (closest_distance / 50)
                ),  # Score based on 50-point tolerance
            }

            theory_b_analysis["belt_event_measurements"].append(measurement)

        return theory_b_analysis

    def _aggregate_theory_b_analysis(
        self, theory_b_measurements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate Theory B analysis across all sessions"""
        if not theory_b_measurements:
            return {}

        aggregated = {
            "total_sessions": len(theory_b_measurements),
            "total_belt_events": 0,
            "dimensional_level_preferences": defaultdict(int),
            "precision_statistics": {},
            "theory_b_scores": [],
            "session_summaries": [],
        }

        all_measurements = []
        all_scores = []

        for session_analysis in theory_b_measurements:
            session_measurements = session_analysis.get("belt_event_measurements", [])
            aggregated["total_belt_events"] += len(session_measurements)

            session_scores = []
            for measurement in session_measurements:
                closest_level = measurement.get("closest_dimensional_level", "unknown")
                aggregated["dimensional_level_preferences"][closest_level] += 1

                score = measurement.get("theory_b_score", 0)
                all_scores.append(score)
                session_scores.append(score)
                all_measurements.append(measurement)

            # Session summary
            session_summary = {
                "session_name": session_analysis.get("session_name", "unknown"),
                "belt_events": len(session_measurements),
                "avg_theory_b_score": np.mean(session_scores) if session_scores else 0,
                "max_theory_b_score": max(session_scores) if session_scores else 0,
                "session_range": session_analysis.get("session_range", 0),
            }
            aggregated["session_summaries"].append(session_summary)

        # Calculate precision statistics
        if all_measurements:
            precisions = [m.get("dimensional_precision", 1.0) for m in all_measurements]
            distances_40 = [
                m.get("distances_to_levels", {}).get("40_percent", 999) for m in all_measurements
            ]

            aggregated["precision_statistics"] = {
                "avg_dimensional_precision": np.mean(precisions),
                "median_dimensional_precision": np.median(precisions),
                "best_dimensional_precision": min(precisions),
                "avg_distance_to_40_percent": np.mean(distances_40),
                "median_distance_to_40_percent": np.median(distances_40),
                "best_distance_to_40_percent": min(distances_40),
            }

        # Theory B score statistics
        if all_scores:
            aggregated["theory_b_scores"] = {
                "avg_score": np.mean(all_scores),
                "median_score": np.median(all_scores),
                "max_score": max(all_scores),
                "min_score": min(all_scores),
                "high_score_events": len([s for s in all_scores if s > 0.8]),
                "total_events": len(all_scores),
            }

        return aggregated

    def _analyze_dimensional_relationships(
        self, belt_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze dimensional relationships in belt events"""
        if not belt_events:
            return {}

        # Group events by session
        sessions = defaultdict(list)
        for event in belt_events:
            session_name = event.get("session_name", "unknown")
            sessions[session_name].append(event)

        dimensional_analysis = {
            "session_count": len(sessions),
            "total_belt_events": len(belt_events),
            "temporal_patterns": {},
            "price_clustering": {},
            "significance_distribution": {},
        }

        # Analyze temporal patterns within belt
        minute_distribution = defaultdict(int)
        for event in belt_events:
            timestamp = event.get("timestamp", "00:00:00")
            if ":" in timestamp:
                minute = timestamp.split(":")[1]
                minute_distribution[minute] += 1

        dimensional_analysis["temporal_patterns"] = {
            "minute_distribution": dict(minute_distribution),
            "most_active_minute": (
                max(minute_distribution.items(), key=lambda x: x[1])[0]
                if minute_distribution
                else "unknown"
            ),
        }

        # Analyze price clustering
        prices = [event.get("price", 0) for event in belt_events if event.get("price", 0) > 0]
        if prices:
            dimensional_analysis["price_clustering"] = {
                "price_range": max(prices) - min(prices),
                "avg_price": np.mean(prices),
                "median_price": np.median(prices),
                "price_std": np.std(prices),
                "unique_price_levels": len(set(prices)),
            }

        # Analyze significance distribution
        significances = [event.get("significance", 0) for event in belt_events]
        if significances:
            dimensional_analysis["significance_distribution"] = {
                "avg_significance": np.mean(significances),
                "median_significance": np.median(significances),
                "high_significance_events": len([s for s in significances if s > 0.7]),
                "total_events": len(significances),
            }

        return dimensional_analysis

    def _identify_archaeological_zones(self, belt_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify archaeological zones within the belt"""
        zones = {
            "zone_identification": {},
            "zone_characteristics": {},
            "cross_session_persistence": {},
        }

        # Group by archaeological zone if available
        zone_events = defaultdict(list)
        for event in belt_events:
            zone = event.get("archaeological_zone", "unknown")
            zone_events[zone].append(event)

        zones["zone_identification"] = {zone: len(events) for zone, events in zone_events.items()}

        # Analyze each zone
        for zone, events in zone_events.items():
            if len(events) < 2:
                continue

            sessions_involved = set(event.get("session_name", "unknown") for event in events)

            zones["zone_characteristics"][zone] = {
                "event_count": len(events),
                "sessions_involved": len(sessions_involved),
                "cross_session_persistence": len(sessions_involved) > 1,
                "avg_significance": np.mean([event.get("significance", 0) for event in events]),
            }

        return zones

    def _calculate_belt_statistics(self, belt_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive belt statistics"""
        if not belt_events:
            return {}

        # Session-level statistics
        sessions = set(event.get("session_name", "unknown") for event in belt_events)

        # Event type distribution
        event_types = defaultdict(int)
        for event in belt_events:
            event_type = event.get("event_type", "unknown")
            event_types[event_type] += 1

        # Temporal distribution
        [event.get("timestamp", "00:00:00") for event in belt_events]

        statistics = {
            "belt_coverage": {
                "sessions_with_belt_events": len(sessions),
                "total_belt_events": len(belt_events),
                "avg_events_per_session": len(belt_events) / len(sessions) if sessions else 0,
            },
            "event_type_distribution": dict(event_types),
            "temporal_characteristics": {
                "belt_timeframe": "14:35:00 - 14:38:59",
                "resolution": "1m",
                "temporal_span_minutes": 4,
            },
            "archaeological_significance": {
                "high_significance_events": len(
                    [e for e in belt_events if e.get("significance", 0) > 0.7]
                ),
                "medium_significance_events": len(
                    [e for e in belt_events if 0.4 < e.get("significance", 0) <= 0.7]
                ),
                "total_events": len(belt_events),
            },
        }

        return statistics

    def _generate_belt_insights(self, belt_lattice: Dict[str, Any]) -> Dict[str, Any]:
        """Generate discovery insights from belt analysis"""
        insights = {
            "theory_b_validation": {},
            "temporal_non_locality": {},
            "archaeological_patterns": {},
            "discovery_recommendations": [],
        }

        # Theory B validation insights
        theory_b = belt_lattice.get("theory_b_analysis", {})
        if theory_b:
            scores = theory_b.get("theory_b_scores", {})
            precision = theory_b.get("precision_statistics", {})

            insights["theory_b_validation"] = {
                "validation_status": (
                    "CONFIRMED" if scores.get("avg_score", 0) > 0.6 else "NEEDS_INVESTIGATION"
                ),
                "avg_dimensional_precision": precision.get("avg_dimensional_precision", 1.0),
                "best_40_percent_distance": precision.get("best_distance_to_40_percent", 999),
                "high_score_event_percentage": (
                    (scores.get("high_score_events", 0) / scores.get("total_events", 1)) * 100
                    if scores.get("total_events", 0) > 0
                    else 0
                ),
            }

        # Temporal non-locality evidence
        dimensional = belt_lattice.get("dimensional_relationships", {})
        if dimensional:
            temporal = dimensional.get("temporal_patterns", {})
            insights["temporal_non_locality"] = {
                "evidence_strength": "STRONG" if temporal.get("most_active_minute") else "MODERATE",
                "clustering_detected": dimensional.get("price_clustering", {}).get(
                    "unique_price_levels", 0
                )
                > 10,
                "minute_concentration": temporal.get("most_active_minute", "unknown"),
            }

        # Archaeological pattern insights
        zones = belt_lattice.get("archaeological_zones", {})
        if zones:
            zone_chars = zones.get("zone_characteristics", {})
            persistent_zones = len(
                [z for z in zone_chars.values() if z.get("cross_session_persistence", False)]
            )

            insights["archaeological_patterns"] = {
                "persistent_zones_detected": persistent_zones,
                "cross_session_reproducibility": persistent_zones > 0,
                "total_archaeological_zones": len(zone_chars),
            }

        # Generate recommendations
        recommendations = []

        if insights["theory_b_validation"].get("validation_status") == "CONFIRMED":
            recommendations.append(
                {
                    "priority": "HIGH",
                    "type": "theory_b_expansion",
                    "description": "Expand Theory B validation to other session types",
                    "action": "Apply belt analysis to LONDON and ASIA sessions",
                }
            )

        if insights["temporal_non_locality"].get("evidence_strength") == "STRONG":
            recommendations.append(
                {
                    "priority": "EXTREME",
                    "type": "temporal_causality_investigation",
                    "description": "Investigate temporal causality mechanisms",
                    "action": "Build predictive models based on belt event positioning",
                }
            )

        insights["discovery_recommendations"] = recommendations

        return insights

    def _load_session_data(self, session_file: Path) -> Optional[Dict[str, Any]]:
        """Load session data from file"""
        try:
            with open(session_file, "r") as f:
                session_data = json.load(f)

            session_name = session_file.stem.replace("enhanced_rel_", "")
            session_data["session_name"] = session_name

            return session_data

        except Exception as e:
            logger.error(f"Failed to load session {session_file}: {e}")
            return None

    def _save_specialized_lattice(self, lattice_data: Dict[str, Any], lattice_type: str):
        """Save specialized lattice to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"specialized_lattice_{lattice_type}_{timestamp}.json"
            filepath = self.discoveries_path / filename

            with open(filepath, "w") as f:
                json.dump(lattice_data, f, indent=2, default=str)

            logger.info(f"💾 Specialized lattice saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save specialized lattice: {e}")

    def _create_empty_specialized_lattice(
        self, lattice_type: str, error_message: str
    ) -> Dict[str, Any]:
        """Create empty specialized lattice for error cases"""
        return {
            "lattice_type": lattice_type,
            "lattice_metadata": {
                "build_timestamp": datetime.now().isoformat(),
                "status": "empty",
                "error": error_message,
            },
            "belt_events": [],
            "theory_b_analysis": {},
            "dimensional_relationships": {},
            "archaeological_zones": {},
            "belt_statistics": {},
            "discovery_insights": {},
        }

    def get_specialized_builder_summary(self) -> Dict[str, Any]:
        """Get summary of specialized lattice builder capabilities"""
        return {
            "specialized_lattice_builder": "IRONFORGE Archaeological Deep Dive",
            "specialized_views": [
                "ny_pm_archaeological_belt",
                "fpfvg_redelivery_networks",
                "weekly_daily_liquidity_sweeps",
            ],
            "focus_areas": {
                "ny_pm_belt": "Theory B 40% zone validation with 1m resolution",
                "fpfvg_networks": "Fair Value Gap redelivery cascade analysis",
                "liquidity_sweeps": "HTF → tactical timeframe cascade tracing",
            },
            "archaeological_capabilities": [
                "theory_b_dimensional_analysis",
                "temporal_non_locality_detection",
                "cross_session_pattern_persistence",
                "archaeological_zone_identification",
            ],
        }
