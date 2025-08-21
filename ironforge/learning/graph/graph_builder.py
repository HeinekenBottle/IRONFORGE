"""
Enhanced Graph Builder Core Module
Main graph construction logic with session context extraction and Theory B validation
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
import torch

from .node_features import NodeFeatureProcessor
from .edge_features import EdgeFeatureProcessor

logger = logging.getLogger(__name__)


class EnhancedGraphBuilder:
    """
    Enhanced graph builder for archaeological discovery
    Transforms JSON session data into rich 45D/20D graph representations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Graph Builder initialized")
        
        # Initialize feature processors
        self.node_processor = NodeFeatureProcessor()
        self.edge_processor = EdgeFeatureProcessor()

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
                node_feature = self.node_processor.create_node_feature(event, session_context)
                graph.add_node(
                    i, feature=node_feature.features, raw_data=event, session_name=session_name
                )

            # Add edges with rich features
            for i in range(len(events)):
                for j in range(i + 1, min(i + 5, len(events))):  # Connect to next 4 events
                    edge_feature = self.edge_processor.create_edge_feature(events[i], events[j])
                    graph.add_edge(i, j, feature=edge_feature.features, temporal_distance=j - i)

            self.logger.info(
                f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            return graph

        except Exception as e:
            self.logger.error(f"Failed to build graph: {e}")
            raise

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

    def validate_theory_b_compliance(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Theory B compliance for session data

        Args:
            session_data: Session JSON with events and metadata

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

            # Check for 40% zone events (Theory B breakthrough zone)
            session_low = session_context.get("session_low", 0.0)
            session_range = session_context.get("session_range", 0.0)

            if session_range > 0:
                for i, event in enumerate(events):
                    price = event.get("price", 0.0)
                    position = (price - session_low) / session_range

                    # Check for 40% zone proximity (Theory B critical zone)
                    if abs(position - 0.40) < 0.02:  # Within 2% of 40% zone
                        validation_results["forty_percent_events"].append({
                            "event_index": i,
                            "price": price,
                            "position": position,
                            "distance_from_40pct": abs(position - 0.40),
                            "event_type": event.get("event_type", "unknown"),
                        })

                    # Check for Theory B violations (events outside expected zones)
                    key_zones = [0.20, 0.40, 0.50, 0.618, 0.80]
                    min_distance_to_zone = min(abs(position - zone) for zone in key_zones)

                    if min_distance_to_zone > 0.10:  # More than 10% from any key zone
                        validation_results["theory_b_violations"].append({
                            "event_index": i,
                            "price": price,
                            "position": position,
                            "min_distance_to_zone": min_distance_to_zone,
                            "event_type": event.get("event_type", "unknown"),
                        })

            # Calculate compliance metrics
            validation_results["forty_percent_compliance"] = len(validation_results["forty_percent_events"])
            validation_results["theory_b_violation_rate"] = (
                len(validation_results["theory_b_violations"]) / len(events) if events else 0.0
            )
            validation_results["overall_compliance"] = (
                1.0 - validation_results["theory_b_violation_rate"]
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

    # Legacy method aliases for backward compatibility
    def _create_node_feature(self, event: Dict[str, Any], session_context: Dict[str, Any] = None):
        """Legacy method - delegates to node processor"""
        return self.node_processor.create_node_feature(event, session_context)

    def _create_edge_feature(self, event1: Dict[str, Any], event2: Dict[str, Any]):
        """Legacy method - delegates to edge processor"""
        return self.edge_processor.create_edge_feature(event1, event2)
