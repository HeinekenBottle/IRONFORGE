"""FPFVG Analysis Runner - Thin Orchestration and I/O Logic."""

import json
import logging
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_config
from ironforge.analysis.fpfvg.chain_builder import (
    calculate_network_density,
    construct_directed_network,
    identify_network_motifs,
)
from ironforge.analysis.fpfvg.features import (
    analyze_score_distribution,
    calculate_range_position,
    extract_magnitude,
    generate_summary_insights,
    get_candidate_summary_stats,
    get_zone_proximity,
    score_redelivery_strength,
    test_pm_belt_interaction,
    test_reproducibility,
    test_zone_enrichment,
)
from ironforge.analysis.fpfvg.validators import (
    is_in_pm_belt,
    safe_float,
    validate_candidates,
    validate_network_graph,
)

logger = logging.getLogger(__name__)


class FPFVGNetworkAnalyzer:
    """
    FPFVG Redelivery Network Analyzer - Main orchestration class
    
    Coordinates the analysis pipeline while delegating heavy lifting to specialized modules.
    """

    def __init__(self):
        """Initialize FPFVG Network Analyzer with configuration"""
        self.config = get_config()
        
        # Paths
        self.enhanced_path = Path(self.config.get_enhanced_data_path())
        self.discoveries_path = Path(self.config.get_discoveries_path())
        
        # Analysis parameters
        self.price_epsilon = 5.0  # Points for price proximity
        self.range_pos_delta = 0.05  # Range position proximity threshold
        self.max_temporal_gap_hours = 12.0  # Maximum time gap for connections
        self.zone_tolerance = 0.03  # Theory B zone tolerance
        self.alpha = 0.05  # Statistical significance level
        
        # Theory B zones (dimensional anchors)
        self.theory_b_zones = [0.2, 0.4, 0.5, 0.618, 0.8]
        
        # PM belt timing
        self.pm_belt_start = time(14, 35)
        self.pm_belt_end = time(14, 38)
        
        # Scoring weights
        self.scoring_weights = {
            "price_proximity": 0.3,
            "range_pos_proximity": 0.3,
            "zone_confluence": 0.25,
            "temporal_penalty": 0.15,
        }
        
        # Session range cache (would be populated from data)
        self.session_ranges = {}
        
        logger.info("FPFVG Network Analyzer initialized")

    def analyze_fpfvg_network(self) -> dict[str, Any]:
        """
        Main analysis entrypoint - orchestrates the full FPFVG network analysis pipeline
        
        Returns comprehensive analysis results with statistical validation.
        """
        logger.info("Starting FPFVG network analysis")
        
        analysis_results = {
            "analysis_type": "fpfvg_network_analysis",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "price_epsilon": self.price_epsilon,
                "range_pos_delta": self.range_pos_delta,
                "max_temporal_gap_hours": self.max_temporal_gap_hours,
                "zone_tolerance": self.zone_tolerance,
                "theory_b_zones": self.theory_b_zones,
                "scoring_weights": self.scoring_weights,
            },
        }

        try:
            # Step 1: Extract FPFVG candidates
            logger.info("Extracting FPFVG candidates from enhanced sessions")
            candidates = self._extract_fpfvg_candidates()
            
            # Validate candidates
            validation_results = validate_candidates(candidates)
            analysis_results["candidate_validation"] = validation_results
            
            if not validation_results["valid"]:
                analysis_results["error"] = "Candidate validation failed"
                return analysis_results
            
            # Step 2: Get candidate summary statistics
            candidate_stats = get_candidate_summary_stats(candidates)
            analysis_results["candidate_extraction"] = candidate_stats
            
            if not candidates:
                analysis_results["error"] = "No FPFVG candidates found"
                return analysis_results

            # Step 3: Construct directed network
            logger.info("Constructing directed FPFVG network")
            network_graph = construct_directed_network(
                candidates,
                self.price_epsilon,
                self.range_pos_delta,
                self.max_temporal_gap_hours
            )
            
            # Validate network
            network_validation = validate_network_graph(network_graph)
            analysis_results["network_validation"] = network_validation
            
            if not network_validation["valid"]:
                analysis_results["error"] = "Network validation failed"
                return analysis_results
            
            # Calculate network metrics
            network_density = calculate_network_density(network_graph)
            network_motifs = identify_network_motifs(network_graph)
            
            analysis_results["network_construction"] = {
                "network_summary": network_graph["metadata"],
                "network_density": network_density,
                "network_motifs": network_motifs,
            }

            # Step 4: Score redelivery strength
            logger.info("Scoring redelivery strength")
            redelivery_scores = score_redelivery_strength(
                network_graph,
                self.scoring_weights,
                self.price_epsilon,
                self.range_pos_delta,
                self.max_temporal_gap_hours,
                self.theory_b_zones
            )
            
            score_distribution = analyze_score_distribution(redelivery_scores)
            
            analysis_results["redelivery_scoring"] = {
                "score_distribution": score_distribution,
                "high_strength_edges": [s for s in redelivery_scores if s["strength"] > 0.7],
                "edge_count": len(redelivery_scores),
            }

            # Step 5: Statistical tests
            logger.info("Running statistical validation tests")
            
            # Zone enrichment test
            zone_enrichment = test_zone_enrichment(
                candidates, self.theory_b_zones, self.zone_tolerance, self.alpha
            )
            analysis_results["zone_enrichment_test"] = zone_enrichment
            
            # PM belt interaction test
            pm_belt_interaction = test_pm_belt_interaction(
                candidates, network_graph, self.alpha
            )
            analysis_results["pm_belt_interaction_test"] = pm_belt_interaction
            
            # Reproducibility test
            reproducibility = test_reproducibility(candidates, network_graph)
            analysis_results["reproducibility_test"] = reproducibility

            # Step 6: Generate insights
            logger.info("Generating summary insights")
            summary_insights = generate_summary_insights(analysis_results)
            analysis_results["summary_insights"] = summary_insights

            # Step 7: Save results
            self._save_analysis_results(analysis_results, network_graph, redelivery_scores)

            logger.info("FPFVG network analysis completed successfully")
            return analysis_results

        except Exception as e:
            error_msg = f"FPFVG network analysis failed: {e}"
            logger.error(error_msg, exc_info=True)
            analysis_results["error"] = error_msg
            return analysis_results

    def _extract_fpfvg_candidates(self) -> list[dict[str, Any]]:
        """
        Extract FPFVG candidates from enhanced sessions
        
        This is a simplified extraction - real implementation would parse lattice summaries
        """
        candidates = []
        
        try:
            # Find enhanced session files
            enhanced_files = list(self.enhanced_path.glob("enhanced_rel_*.json"))
            logger.info(f"Found {len(enhanced_files)} enhanced session files")
            
            for file_path in enhanced_files[:10]:  # Limit for testing
                try:
                    with open(file_path) as f:
                        session_data = json.load(f)
                    
                    session_candidates = self._extract_session_candidates(session_data, file_path.stem)
                    candidates.extend(session_candidates)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
            
            logger.info(f"Extracted {len(candidates)} FPFVG candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to extract FPFVG candidates: {e}")
            return []

    def _extract_session_candidates(self, session_data: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
        """Extract FPFVG candidates from a single session"""
        candidates = []
        
        # This is a simplified implementation
        # Real implementation would parse specific FPFVG event types from graph data
        
        # Mock some candidates for testing
        if "nodes" in session_data:
            nodes = session_data["nodes"][:5]  # Limit for testing
            
            for i, node in enumerate(nodes):
                # Create mock FPFVG candidates
                price_level = safe_float(node.get("price", 23000 + i * 10))
                timestamp = node.get("timestamp", f"2025-08-01T14:{30+i}:00")
                
                # Calculate session range (simplified)
                if session_id not in self.session_ranges:
                    self.session_ranges[session_id] = {"low": price_level - 50, "high": price_level + 50}
                
                range_pos = calculate_range_position(price_level, session_id, self.session_ranges)
                zone_proximity = get_zone_proximity(range_pos, self.theory_b_zones, self.zone_tolerance)
                in_pm_belt = is_in_pm_belt(timestamp, self.pm_belt_start, self.pm_belt_end)
                
                candidate = {
                    "id": f"{session_id}_fpfvg_{i}",
                    "session_id": session_id,
                    "event_type": "formation" if i % 2 == 0 else "redelivery",
                    "price_level": price_level,
                    "range_pos": range_pos,
                    "start_ts": timestamp,
                    "in_pm_belt": in_pm_belt,
                    "zone_proximity": zone_proximity,
                    "timeframe": "15m",
                    "magnitude": extract_magnitude(node),
                }
                
                candidates.append(candidate)
        
        return candidates

    def _save_analysis_results(
        self, 
        analysis_results: dict[str, Any], 
        network_graph: dict[str, Any], 
        redelivery_scores: list[dict[str, Any]]  # noqa: ARG002
    ) -> None:
        """Save analysis results to discovery files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main analysis results
        results_filename = f"fpfvg_network_analysis_{timestamp}.json"
        results_filepath = self.discoveries_path / results_filename
        
        try:
            with open(results_filepath, "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)
            logger.info(f"Analysis results saved to {results_filepath}")
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")

        # Save network graph
        network_filename = f"fpfvg_network_graph_{timestamp}.json"
        network_filepath = self.discoveries_path / network_filename

        try:
            with open(network_filepath, "w") as f:
                json.dump(network_graph, f, indent=2, default=str)
            logger.info(f"Network graph saved to {network_filepath}")
        except Exception as e:
            logger.error(f"Failed to save network data: {e}")

        # Save statistical summary
        stats_filename = f"fpfvg_network_stats_{timestamp}.json"
        stats_filepath = self.discoveries_path / stats_filename

        try:
            stats_summary = {
                "zone_enrichment": analysis_results.get("zone_enrichment_test", {}),
                "pm_belt_interaction": analysis_results.get("pm_belt_interaction_test", {}),
                "network_motifs": analysis_results.get("network_construction", {}).get(
                    "network_motifs", {}
                ),
                "summary_insights": analysis_results.get("summary_insights", {}),
                "timestamp": timestamp,
            }

            with open(stats_filepath, "w") as f:
                json.dump(stats_summary, f, indent=2, default=str)
            logger.info(f"Network statistics saved to {stats_filepath}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")


# Public API functions (for backward compatibility)
def build_chains(adjacency: dict[str, list[str]], min_length: int = 3) -> list[list[str]]:
    """Build chains from adjacency list (re-exported from chain_builder)"""
    from ironforge.analysis.fpfvg.chain_builder import find_chains
    return find_chains(adjacency, min_length)


def validate_chain(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate FPFVG chain data (re-exported from validators)"""
    return validate_candidates(candidates)


def compute_chain_features(network_graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute chain features (re-exported from features)"""
    return score_redelivery_strength(network_graph)