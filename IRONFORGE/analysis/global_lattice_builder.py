"""
IRONFORGE Global Lattice Builder
===============================

Creates comprehensive Monthly→1m lattice mappings across all enhanced sessions.
Builds the global terrain map with nodes, edges, hot zones, and vertical cascades.

Purpose: Discover the broadest scope patterns before drilling into details.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
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

from ironforge.analysis.timeframe_lattice_mapper import TimeframeLatticeMapper

logger = logging.getLogger(__name__)


class GlobalLatticeBuilder:
    """
    Global lattice builder for comprehensive Monthly→1m analysis across all sessions.

    Features:
    - Multi-session lattice aggregation
    - Hot zone identification and clustering
    - Vertical cascade tracing (Monthly→Weekly→Daily→PM)
    - Statistical enrichment analysis
    - Bridge node discovery
    """

    def __init__(self):
        self.config = get_config()
        self.lattice_mapper = TimeframeLatticeMapper()

        # Timeframes in hierarchical order
        self.timeframes = ["Monthly", "Weekly", "Daily", "4H", "1H", "15m", "5m", "1m"]

        # Initialize global structures
        self.global_nodes = []
        self.global_edges = []
        self.global_hot_zones = []
        self.cascade_chains = []
        self.bridge_nodes = []

        # Statistics tracking
        self.session_stats = {
            "sessions_processed": 0,
            "total_events": 0,
            "timeframe_distribution": {tf: 0 for tf in self.timeframes},
            "zone_types": defaultdict(int),
            "cascade_patterns": defaultdict(int),
        }

        logger.info("Global Lattice Builder initialized for Monthly→1m analysis")

    def build_global_lattice(self, enhanced_sessions_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Build comprehensive global lattice from all enhanced sessions

        Args:
            enhanced_sessions_path: Path to enhanced sessions directory

        Returns:
            Complete global lattice with nodes, edges, hot zones, and analysis
        """
        try:
            logger.info("🌐 Starting global lattice build - Monthly→1m scope")

            if enhanced_sessions_path is None:
                enhanced_sessions_path = Path(self.config.get_data_path()) / "enhanced"
            else:
                enhanced_sessions_path = Path(enhanced_sessions_path)

            # Find all enhanced session files
            session_files = list(enhanced_sessions_path.glob("enhanced_rel_*.json"))
            logger.info(f"Found {len(session_files)} enhanced sessions for global analysis")

            if not session_files:
                logger.warning("No enhanced session files found")
                return self._create_empty_lattice()

            # Process each session and build lattice components
            for session_file in session_files:
                try:
                    session_data = self._load_session_data(session_file)
                    if session_data:
                        session_lattice = self._build_session_lattice(session_data)
                        self._integrate_session_lattice(session_lattice)
                        self.session_stats["sessions_processed"] += 1

                except Exception as e:
                    logger.error(f"Failed to process session {session_file}: {e}")
                    continue

            # Post-processing: identify patterns, cascades, and hot zones
            self._identify_hot_zones()
            self._trace_vertical_cascades()
            self._identify_bridge_nodes()
            self._calculate_enrichment_statistics()

            # Build final global lattice structure
            global_lattice = self._construct_final_lattice()

            # Save global lattice results
            self._save_global_lattice(global_lattice)

            logger.info(
                f"✅ Global lattice complete: {len(self.global_nodes)} nodes, {len(self.global_edges)} edges, {len(self.global_hot_zones)} hot zones"
            )

            return global_lattice

        except Exception as e:
            logger.error(f"Global lattice build failed: {e}")
            return {"error": str(e), "status": "failed"}

    def _load_session_data(self, session_file: Path) -> Optional[Dict[str, Any]]:
        """Load and validate session data"""
        try:
            with open(session_file, "r") as f:
                session_data = json.load(f)

            # Extract session name from filename
            session_name = session_file.stem.replace("enhanced_rel_", "")
            session_data["session_name"] = session_name

            return session_data

        except Exception as e:
            logger.error(f"Failed to load session {session_file}: {e}")
            return None

    def _build_session_lattice(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build lattice for individual session using TimeframeLatticeMapper"""
        try:
            # Use existing lattice mapper
            session_lattice = self.lattice_mapper.map_session_lattice(session_data)

            # Enhance with global context
            session_lattice["global_context"] = self._extract_global_context(session_data)

            return session_lattice

        except Exception as e:
            logger.error(f"Session lattice build failed: {e}")
            return {"error": str(e)}

    def _extract_global_context(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract global context information from session"""
        context = {
            "session_name": session_data.get("session_name", "unknown"),
            "market_period": self._determine_market_period(session_data.get("session_name", "")),
            "energy_state": session_data.get("energy_state", {}),
            "session_range": self._extract_session_range(session_data),
            "archaeological_significance": self._assess_archaeological_significance(session_data),
        }

        return context

    def _determine_market_period(self, session_name: str) -> str:
        """Determine market period from session name"""
        if "ASIA" in session_name:
            return "ASIA"
        elif "LONDON" in session_name:
            return "LONDON"
        elif "NY_AM" in session_name:
            return "NY_AM"
        elif "NY_PM" in session_name:
            return "NY_PM"
        elif "LUNCH" in session_name:
            return "LUNCH"
        elif "MIDNIGHT" in session_name:
            return "MIDNIGHT"
        elif "PREMARKET" in session_name:
            return "PREMARKET"
        else:
            return "UNKNOWN"

    def _extract_session_range(self, session_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract session high/low/range information"""
        energy_state = session_data.get("energy_state", {})

        return {
            "session_high": energy_state.get("session_high", 0),
            "session_low": energy_state.get("session_low", 0),
            "session_range": energy_state.get("session_range", 0),
            "range_midpoint": (
                energy_state.get("session_high", 0) + energy_state.get("session_low", 0) / 2
                if energy_state.get("session_high", 0) > 0
                else 0
            ),
        }

    def _assess_archaeological_significance(self, session_data: Dict[str, Any]) -> float:
        """Assess archaeological significance of session"""
        significance_factors = []

        # FPFVG presence
        fpfvg = session_data.get("session_fpfvg", {})
        if fpfvg.get("fpfvg_present"):
            significance_factors.append(0.3)

        # Liquidity events count
        liquidity_events = session_data.get("session_liquidity_events", [])
        if len(liquidity_events) > 5:
            significance_factors.append(0.2)

        # Price movement intensity
        price_movements = session_data.get("price_movements", [])
        if len(price_movements) > 10:
            significance_factors.append(0.2)

        # Energy state indicators
        energy_state = session_data.get("energy_state", {})
        if energy_state.get("session_range", 0) > 100:  # Significant range
            significance_factors.append(0.3)

        return min(sum(significance_factors), 1.0)

    def _integrate_session_lattice(self, session_lattice: Dict[str, Any]):
        """Integrate session lattice into global structures"""
        if "error" in session_lattice:
            return

        # Add nodes with global IDs
        nodes = session_lattice.get("nodes", [])
        for node in nodes:
            node["global_id"] = f"global_{len(self.global_nodes)}"
            node["session_context"] = session_lattice.get("global_context", {})
            self.global_nodes.append(node)

            # Update timeframe statistics
            tf = node.get("timeframe", "unknown")
            if tf in self.session_stats["timeframe_distribution"]:
                self.session_stats["timeframe_distribution"][tf] += 1

        # Add edges with global node references
        edges = session_lattice.get("edges", [])
        for edge in edges:
            edge["global_id"] = f"global_edge_{len(self.global_edges)}"
            self.global_edges.append(edge)

        # Track event count
        metrics = session_lattice.get("metrics", {})
        self.session_stats["total_events"] += metrics.get("events_processed", 0)

    def _identify_hot_zones(self):
        """Identify and cluster hot zones across all sessions"""
        logger.info("🔥 Identifying global hot zones...")

        # Group nodes by price levels and timeframes
        price_clusters = defaultdict(list)

        for node in self.global_nodes:
            price = node.get("price", 0)
            timeframe = node.get("timeframe", "unknown")
            significance = node.get("significance", 0)

            # Only consider significant events
            if significance > 0.6:
                # Use precise price clustering (10-point tolerance)
                price_key = round(price / 10) * 10
                cluster_key = f"{timeframe}_{price_key}"
                price_clusters[cluster_key].append(node)

        # Create hot zones from clusters
        for cluster_key, cluster_nodes in price_clusters.items():
            if len(cluster_nodes) >= 2:  # At least 2 events to form hot zone
                timeframe, price_level = cluster_key.split("_", 1)
                price_level = float(price_level)

                hot_zone = {
                    "zone_id": f"hot_zone_{len(self.global_hot_zones)}",
                    "timeframe": timeframe,
                    "price_level": price_level,
                    "event_count": len(cluster_nodes),
                    "avg_significance": np.mean([n.get("significance", 0) for n in cluster_nodes]),
                    "max_significance": max([n.get("significance", 0) for n in cluster_nodes]),
                    "zone_type": self._determine_zone_type(cluster_nodes),
                    "sessions_involved": list(
                        set(
                            [
                                n.get("session_context", {}).get("session_name", "unknown")
                                for n in cluster_nodes
                            ]
                        )
                    ),
                    "market_periods": list(
                        set(
                            [
                                n.get("session_context", {}).get("market_period", "unknown")
                                for n in cluster_nodes
                            ]
                        )
                    ),
                }

                self.global_hot_zones.append(hot_zone)

                # Update zone type statistics
                zone_type = hot_zone["zone_type"]
                self.session_stats["zone_types"][zone_type] += 1

        logger.info(f"Identified {len(self.global_hot_zones)} global hot zones")

    def _determine_zone_type(self, nodes: List[Dict[str, Any]]) -> str:
        """Determine hot zone type based on constituent events"""
        event_types = [n.get("event_type", "unknown") for n in nodes]

        # Count event type frequencies
        type_counts = defaultdict(int)
        for event_type in event_types:
            type_counts[event_type] += 1

        # Determine dominant type
        if not type_counts:
            return "mixed"

        dominant_type = max(type_counts, key=type_counts.get)

        if "fpfvg" in dominant_type:
            return "fpfvg_cluster"
        elif "liquidity" in dominant_type:
            return "liquidity_cluster"
        elif "price_movement" in dominant_type:
            return "movement_cluster"
        else:
            return "mixed_cluster"

    def _trace_vertical_cascades(self):
        """Trace vertical cascades from Monthly down to 1m"""
        logger.info("📈 Tracing vertical cascades Monthly→Weekly→Daily→PM...")

        # Group nodes by timeframe
        tf_nodes = {tf: [] for tf in self.timeframes}
        for node in self.global_nodes:
            tf = node.get("timeframe", "unknown")
            if tf in tf_nodes:
                tf_nodes[tf].append(node)

        # Trace cascades through timeframe hierarchy
        for i in range(len(self.timeframes) - 1):
            higher_tf = self.timeframes[i]
            lower_tf = self.timeframes[i + 1]

            higher_nodes = tf_nodes[higher_tf]
            lower_nodes = tf_nodes[lower_tf]

            # Find cascade connections
            for h_node in higher_nodes:
                connected_nodes = []
                h_price = h_node.get("price", 0)

                # Find lower timeframe nodes within cascade tolerance
                for l_node in lower_nodes:
                    l_price = l_node.get("price", 0)
                    price_diff = abs(h_price - l_price)

                    # Cascade tolerance based on timeframe gap
                    tolerance = self._get_cascade_tolerance(higher_tf, lower_tf)

                    if price_diff <= tolerance:
                        connected_nodes.append(l_node)

                # Create cascade if connections found
                if connected_nodes:
                    cascade = {
                        "cascade_id": f"cascade_{len(self.cascade_chains)}",
                        "origin_timeframe": higher_tf,
                        "target_timeframe": lower_tf,
                        "origin_node": h_node,
                        "connected_nodes": connected_nodes,
                        "cascade_strength": len(connected_nodes),
                        "price_center": h_price,
                        "cascade_spread": (
                            max([abs(h_price - n.get("price", 0)) for n in connected_nodes])
                            if connected_nodes
                            else 0
                        ),
                    }

                    self.cascade_chains.append(cascade)

                    # Update cascade pattern statistics
                    pattern_key = f"{higher_tf}_to_{lower_tf}"
                    self.session_stats["cascade_patterns"][pattern_key] += 1

        logger.info(f"Traced {len(self.cascade_chains)} vertical cascades")

    def _get_cascade_tolerance(self, higher_tf: str, lower_tf: str) -> float:
        """Get price tolerance for cascade connections between timeframes"""
        # Tighter tolerances for closer timeframes
        tolerance_map = {
            ("Monthly", "Weekly"): 200,
            ("Weekly", "Daily"): 100,
            ("Daily", "4H"): 75,
            ("4H", "1H"): 50,
            ("1H", "15m"): 30,
            ("15m", "5m"): 20,
            ("5m", "1m"): 10,
        }

        return tolerance_map.get((higher_tf, lower_tf), 50)

    def _identify_bridge_nodes(self):
        """Identify bridge nodes that connect multiple timeframes"""
        logger.info("🔗 Identifying bridge nodes across timeframes...")

        # Nodes that appear in multiple cascade chains are potential bridges
        node_cascade_count = defaultdict(int)

        for cascade in self.cascade_chains:
            origin_node = cascade["origin_node"]
            node_id = origin_node.get("global_id", "unknown")
            node_cascade_count[node_id] += 1

            for connected_node in cascade["connected_nodes"]:
                conn_id = connected_node.get("global_id", "unknown")
                node_cascade_count[conn_id] += 1

        # Identify bridge nodes (appear in multiple cascades)
        for node in self.global_nodes:
            global_id = node.get("global_id", "unknown")
            cascade_count = node_cascade_count[global_id]

            if cascade_count >= 2:  # Bridge threshold
                bridge_node = {
                    "bridge_id": f"bridge_{len(self.bridge_nodes)}",
                    "node": node,
                    "cascade_connections": cascade_count,
                    "bridge_type": self._classify_bridge_node(node, cascade_count),
                    "structural_importance": min(cascade_count / 5.0, 1.0),  # Normalize to 0-1
                }

                self.bridge_nodes.append(bridge_node)

        logger.info(f"Identified {len(self.bridge_nodes)} bridge nodes")

    def _classify_bridge_node(self, node: Dict[str, Any], cascade_count: int) -> str:
        """Classify bridge node type based on properties"""
        timeframe = node.get("timeframe", "unknown")
        significance = node.get("significance", 0)

        if timeframe in ["Monthly", "Weekly"]:
            return "htf_anchor"
        elif timeframe in ["Daily", "4H"]:
            return "structural_pivot"
        elif cascade_count >= 4:
            return "super_connector"
        elif significance > 0.8:
            return "high_significance_bridge"
        else:
            return "standard_bridge"

    def _calculate_enrichment_statistics(self):
        """Calculate enrichment statistics for hot zones and patterns"""
        logger.info("📊 Calculating enrichment statistics...")

        # Calculate baseline event density
        total_price_range = 1000  # Approximate total price range observed
        baseline_density = (
            len(self.global_nodes) / total_price_range if total_price_range > 0 else 0
        )

        # Calculate enrichment for each hot zone
        for hot_zone in self.global_hot_zones:
            event_count = hot_zone["event_count"]
            zone_density = event_count / 10  # 10-point zone width

            enrichment_ratio = zone_density / baseline_density if baseline_density > 0 else 0

            hot_zone["enrichment_ratio"] = enrichment_ratio
            hot_zone["statistical_significance"] = self._calculate_significance(
                event_count, baseline_density
            )
            hot_zone["enrichment_category"] = self._categorize_enrichment(enrichment_ratio)

    def _calculate_significance(self, observed_count: int, baseline_density: float) -> float:
        """Calculate statistical significance of enrichment"""
        # Simple statistical significance based on deviation from baseline
        expected_count = baseline_density * 10  # For 10-point zone
        if expected_count <= 0:
            return 0.0

        # Z-score approximation
        z_score = abs(observed_count - expected_count) / np.sqrt(expected_count)

        # Convert to significance (0-1 scale)
        return min(z_score / 3.0, 1.0)  # Normalize to 0-1

    def _categorize_enrichment(self, enrichment_ratio: float) -> str:
        """Categorize enrichment level"""
        if enrichment_ratio >= 5.0:
            return "extreme_enrichment"
        elif enrichment_ratio >= 3.0:
            return "high_enrichment"
        elif enrichment_ratio >= 2.0:
            return "moderate_enrichment"
        elif enrichment_ratio >= 1.5:
            return "slight_enrichment"
        else:
            return "baseline"

    def _construct_final_lattice(self) -> Dict[str, Any]:
        """Construct final global lattice structure"""

        global_lattice = {
            "lattice_metadata": {
                "build_timestamp": datetime.now().isoformat(),
                "lattice_type": "global_monthly_to_1m",
                "scope": "all_enhanced_sessions",
                "timeframes": self.timeframes,
            },
            "global_nodes": self.global_nodes,
            "global_edges": self.global_edges,
            "hot_zones": self.global_hot_zones,
            "vertical_cascades": self.cascade_chains,
            "bridge_nodes": self.bridge_nodes,
            "statistics": self.session_stats,
            "enrichment_analysis": {
                "hot_zones_by_enrichment": self._group_zones_by_enrichment(),
                "cascade_patterns": dict(self.session_stats["cascade_patterns"]),
                "zone_type_distribution": dict(self.session_stats["zone_types"]),
            },
            "discovery_recommendations": self._generate_discovery_recommendations(),
        }

        return global_lattice

    def _group_zones_by_enrichment(self) -> Dict[str, List[Dict]]:
        """Group hot zones by enrichment category"""
        enrichment_groups = defaultdict(list)

        for zone in self.global_hot_zones:
            category = zone.get("enrichment_category", "baseline")
            enrichment_groups[category].append(zone)

        return dict(enrichment_groups)

    def _generate_discovery_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for next discovery steps"""
        recommendations = []

        # Recommend investigating extreme enrichment zones
        extreme_zones = [
            z for z in self.global_hot_zones if z.get("enrichment_category") == "extreme_enrichment"
        ]
        if extreme_zones:
            recommendations.append(
                {
                    "type": "hot_zone_investigation",
                    "priority": "high",
                    "description": f"Investigate {len(extreme_zones)} extreme enrichment zones",
                    "zones": extreme_zones[:5],  # Top 5
                }
            )

        # Recommend investigating strongest cascades
        strong_cascades = sorted(
            self.cascade_chains, key=lambda x: x.get("cascade_strength", 0), reverse=True
        )[:10]
        if strong_cascades:
            recommendations.append(
                {
                    "type": "cascade_analysis",
                    "priority": "medium",
                    "description": f"Analyze top {len(strong_cascades)} vertical cascades",
                    "cascades": strong_cascades,
                }
            )

        # Recommend investigating bridge nodes
        if self.bridge_nodes:
            super_connectors = [
                b for b in self.bridge_nodes if b.get("bridge_type") == "super_connector"
            ]
            if super_connectors:
                recommendations.append(
                    {
                        "type": "bridge_node_analysis",
                        "priority": "high",
                        "description": f"Analyze {len(super_connectors)} super connector bridge nodes",
                        "bridges": super_connectors,
                    }
                )

        return recommendations

    def _save_global_lattice(self, global_lattice: Dict[str, Any]):
        """Save global lattice results to file"""
        try:
            discoveries_path = Path(self.config.get_discoveries_path())
            discoveries_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"global_lattice_monthly_to_1m_{timestamp}.json"
            filepath = discoveries_path / filename

            with open(filepath, "w") as f:
                json.dump(global_lattice, f, indent=2, default=str)

            logger.info(f"💾 Global lattice saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save global lattice: {e}")

    def _create_empty_lattice(self) -> Dict[str, Any]:
        """Create empty lattice structure for error cases"""
        return {
            "lattice_metadata": {
                "build_timestamp": datetime.now().isoformat(),
                "lattice_type": "global_monthly_to_1m",
                "scope": "no_sessions_found",
                "status": "empty",
            },
            "global_nodes": [],
            "global_edges": [],
            "hot_zones": [],
            "vertical_cascades": [],
            "bridge_nodes": [],
            "statistics": self.session_stats,
            "error": "No enhanced sessions found",
        }

    def get_lattice_summary(self) -> Dict[str, Any]:
        """Get summary of current lattice state"""
        return {
            "global_lattice_builder": "IRONFORGE Monthly→1m Analysis",
            "timeframes_analyzed": self.timeframes,
            "current_stats": self.session_stats,
            "structures_built": {
                "nodes": len(self.global_nodes),
                "edges": len(self.global_edges),
                "hot_zones": len(self.global_hot_zones),
                "cascades": len(self.cascade_chains),
                "bridges": len(self.bridge_nodes),
            },
        }
