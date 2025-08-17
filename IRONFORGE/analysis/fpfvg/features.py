"""FPFVG Feature Extraction and Scoring Logic."""

import logging
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)


def score_redelivery_strength(
    network_graph: dict[str, Any],
    scoring_weights: dict[str, float] = None,
    price_epsilon: float = 5.0,
    range_pos_delta: float = 0.05,
    max_temporal_gap_hours: float = 12.0,
    theory_b_zones: list[float] = None,
) -> list[dict[str, Any]]:
    """
    Score re-delivery strength using weighted factors

    Formula: w1·(price proximity) + w2·(range_pos proximity) + w3·(zone_confluence) - w4·(Δt penalty)
    """
    if scoring_weights is None:
        scoring_weights = {
            "price_proximity": 0.3,
            "range_pos_proximity": 0.3,
            "zone_confluence": 0.25,
            "temporal_penalty": 0.15,
        }

    if theory_b_zones is None:
        theory_b_zones = [0.2, 0.4, 0.5, 0.618, 0.8]

    redelivery_scores = []

    for edge in network_graph["edges"]:
        # Component scores (0-1 scale)
        price_proximity_score = calculate_price_proximity_score(edge, price_epsilon)
        range_pos_proximity_score = calculate_range_pos_proximity_score(edge, range_pos_delta)
        zone_confluence_score = calculate_zone_confluence_score(edge, theory_b_zones)
        temporal_penalty_score = calculate_temporal_penalty_score(edge, max_temporal_gap_hours)

        # Weighted total score
        total_score = (
            scoring_weights["price_proximity"] * price_proximity_score
            + scoring_weights["range_pos_proximity"] * range_pos_proximity_score
            + scoring_weights["zone_confluence"] * zone_confluence_score
            - scoring_weights["temporal_penalty"] * temporal_penalty_score
        )

        score_record = {
            "edge_id": f"{edge['source']}_{edge['target']}",
            "source": edge["source"],
            "target": edge["target"],
            "strength": max(0.0, min(1.0, total_score)),  # Clamp to [0,1]
            "components": {
                "price_proximity": price_proximity_score,
                "range_pos_proximity": range_pos_proximity_score,
                "zone_confluence": zone_confluence_score,
                "temporal_penalty": temporal_penalty_score,
            },
            "edge_features": edge,
        }
        redelivery_scores.append(score_record)

    return redelivery_scores


def calculate_price_proximity_score(edge: dict[str, Any], price_epsilon: float = 5.0) -> float:
    """Calculate price proximity score (1.0 = identical prices)"""
    price_distance = edge.get("price_distance", 0)

    if price_distance == 0:
        return 1.0

    # Exponential decay with epsilon
    return np.exp(-price_distance / price_epsilon)


def calculate_range_pos_proximity_score(
    edge: dict[str, Any], range_pos_delta: float = 0.05
) -> float:
    """Calculate range position proximity score"""
    delta_range_pos = edge.get("delta_range_pos", 0)

    if delta_range_pos == 0:
        return 1.0

    # Exponential decay with delta
    return np.exp(-delta_range_pos / range_pos_delta)


def calculate_zone_confluence_score(
    edge: dict[str, Any], theory_b_zones: list[float] = None
) -> float:
    """Calculate zone confluence score"""
    if theory_b_zones is None:
        theory_b_zones = [0.2, 0.4, 0.5, 0.618, 0.8]

    zone_flags = edge.get("same_zone_flags", {})

    if not zone_flags:
        return 0.0

    # Score based on number of zones where both nodes align
    aligned_zones = sum(1 for flag in zone_flags.values() if flag)
    total_zones = len(theory_b_zones)

    return aligned_zones / total_zones if total_zones > 0 else 0.0


def calculate_temporal_penalty_score(
    edge: dict[str, Any], max_temporal_gap_hours: float = 12.0
) -> float:
    """Calculate temporal penalty score (higher for longer delays)"""
    delta_t_minutes = edge.get("delta_t_minutes", 0)

    if delta_t_minutes == 0:
        return 0.0

    # Normalize by maximum temporal gap (in minutes)
    max_gap_minutes = max_temporal_gap_hours * 60
    return min(1.0, delta_t_minutes / max_gap_minutes)


def analyze_score_distribution(redelivery_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze distribution of redelivery scores"""
    if not redelivery_scores:
        return {}

    scores = [s["strength"] for s in redelivery_scores]

    return {
        "count": len(scores),
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "percentiles": {
            "25th": np.percentile(scores, 25),
            "75th": np.percentile(scores, 75),
            "90th": np.percentile(scores, 90),
        },
    }


def calculate_range_position(
    price_level: float, session_id: str, session_ranges: dict[str, dict] = None
) -> float:
    """
    Calculate range position (0-1) for a price level within session range

    Args:
        price_level: Price to calculate position for
        session_id: Session identifier
        session_ranges: Dict of session ranges {session_id: {"low": float, "high": float}}

    Returns:
        float: Position in range (0 = low, 1 = high)
    """
    if session_ranges is None or session_id not in session_ranges:
        logger.warning(f"No session range data for session {session_id}")
        return 0.5  # Default to middle

    session_range = session_ranges[session_id]
    low = session_range.get("low", price_level)
    high = session_range.get("high", price_level)

    if high == low:
        return 0.5  # Avoid division by zero

    position = (price_level - low) / (high - low)
    return max(0.0, min(1.0, position))  # Clamp to [0,1]


def get_zone_proximity(
    range_pos: float, theory_b_zones: list[float] = None, zone_tolerance: float = 0.03
) -> dict[str, Any]:
    """
    Calculate proximity to Theory B zones

    Args:
        range_pos: Range position (0-1)
        theory_b_zones: List of zone positions
        zone_tolerance: Tolerance for zone proximity

    Returns:
        dict: Zone proximity information
    """
    if theory_b_zones is None:
        theory_b_zones = [0.2, 0.4, 0.5, 0.618, 0.8]

    # Find closest zones
    zone_distances = {}
    closest_zones = []

    for zone in theory_b_zones:
        distance = abs(range_pos - zone)
        zone_name = f"{zone*100:.1f}%"
        zone_distances[zone_name] = distance

        if distance <= zone_tolerance:
            closest_zones.append(zone_name)

    # Find overall closest zone
    closest_zone = min(zone_distances.keys(), key=lambda z: zone_distances[z])
    min_distance = zone_distances[closest_zone]

    return {
        "range_position": range_pos,
        "closest_zone": closest_zone,
        "distance_to_closest": min_distance,
        "in_zone": len(closest_zones) > 0,
        "closest_zones": closest_zones,
        "all_distances": zone_distances,
    }


def extract_magnitude(event_data: dict[str, Any]) -> float:
    """Extract magnitude/importance of event from event data"""
    # This is a simplified implementation
    # Real implementation would parse event-specific magnitude fields

    if "magnitude" in event_data:
        try:
            return float(event_data["magnitude"])
        except (ValueError, TypeError):
            pass

    if "gap_size" in event_data:
        try:
            return float(event_data["gap_size"])
        except (ValueError, TypeError):
            pass

    if "volume" in event_data:
        try:
            return float(event_data["volume"])
        except (ValueError, TypeError):
            pass

    # Default magnitude
    return 1.0


def get_candidate_summary_stats(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate summary statistics for FPFVG candidates"""
    if not candidates:
        return {}

    formation_count = len([c for c in candidates if c.get("event_type") == "formation"])
    redelivery_count = len([c for c in candidates if c.get("event_type") == "redelivery"])

    # Session distribution
    sessions = [c.get("session_id") for c in candidates if c.get("session_id")]
    unique_sessions = len(set(sessions))

    # PM belt distribution
    pm_belt_candidates = [c for c in candidates if c.get("in_pm_belt", False)]
    pm_belt_count = len(pm_belt_candidates)

    # Range position distribution
    range_positions = [
        c.get("range_pos", 0.5) for c in candidates if c.get("range_pos") is not None
    ]

    # Zone proximity analysis
    in_zone_count = len(
        [c for c in candidates if c.get("zone_proximity", {}).get("in_zone", False)]
    )

    stats = {
        "total_candidates": len(candidates),
        "formation_count": formation_count,
        "redelivery_count": redelivery_count,
        "unique_sessions": unique_sessions,
        "pm_belt_count": pm_belt_count,
        "pm_belt_percentage": pm_belt_count / len(candidates) * 100 if candidates else 0,
        "in_zone_count": in_zone_count,
        "zone_percentage": in_zone_count / len(candidates) * 100 if candidates else 0,
        "range_position_stats": {
            "mean": np.mean(range_positions) if range_positions else 0,
            "median": np.median(range_positions) if range_positions else 0,
            "std": np.std(range_positions) if range_positions else 0,
            "min": np.min(range_positions) if range_positions else 0,
            "max": np.max(range_positions) if range_positions else 0,
        },
    }

    return stats


def test_zone_enrichment(
    candidates: list[dict[str, Any]],
    theory_b_zones: list[float] = None,
    zone_tolerance: float = 0.03,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Test zone enrichment: are redeliveries enriched in Theory B zones?

    Uses Fisher exact test to compare observed vs expected redeliveries in zones.
    """
    if theory_b_zones is None:
        theory_b_zones = [0.2, 0.4, 0.5, 0.618, 0.8]

    zone_enrichment_results = {
        "test_type": "zone_enrichment_fisher_exact",
        "hypothesis": "redeliveries_enrich_in_theory_b_zones",
        "zones_tested": theory_b_zones,
    }

    # Count redeliveries in zones vs outside zones
    redelivery_candidates = [c for c in candidates if c["event_type"] == "redelivery"]

    if not redelivery_candidates:
        zone_enrichment_results["error"] = "No redelivery candidates found"
        return zone_enrichment_results

    total_redeliveries = len(redelivery_candidates)
    redeliveries_in_zones = len(
        [c for c in redelivery_candidates if c["zone_proximity"]["in_zone"]]
    )
    redeliveries_outside_zones = total_redeliveries - redeliveries_in_zones

    # Calculate baseline expectation (zone coverage)
    total_zone_coverage = len(theory_b_zones) * zone_tolerance * 2  # ±tolerance for each zone
    expected_in_zones = total_redeliveries * total_zone_coverage
    expected_outside_zones = total_redeliveries - expected_in_zones

    # Contingency table for Fisher exact test
    try:
        # Fisher exact test
        odds_ratio, p_value = fisher_exact(
            [
                [redeliveries_in_zones, redeliveries_outside_zones],
                [max(1, int(expected_in_zones)), max(1, int(expected_outside_zones))],
            ]
        )

        zone_enrichment_results.update(
            {
                "observed_in_zones": redeliveries_in_zones,
                "observed_outside_zones": redeliveries_outside_zones,
                "expected_in_zones": expected_in_zones,
                "expected_outside_zones": expected_outside_zones,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
                "significant": p_value < alpha,
                "enrichment_factor": redeliveries_in_zones / max(1, expected_in_zones),
                "zone_coverage": total_zone_coverage,
            }
        )

    except Exception as e:
        zone_enrichment_results["error"] = f"Statistical test failed: {e}"

    return zone_enrichment_results


def test_pm_belt_interaction(
    candidates: list[dict[str, Any]],
    network_graph: dict[str, Any],  # noqa: ARG001
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Test PM-belt interaction: P(redelivery hits 14:35-:38 | prior FVG in session) vs baseline

    H0: No increased PM belt interaction after FVG formation
    H1: FVG formations increase probability of PM belt redelivery
    """
    pm_belt_results = {
        "test_type": "pm_belt_interaction_analysis",
        "hypothesis": "fvg_formations_increase_pm_belt_redelivery_probability",
        "pm_belt_window": "14:35 - 14:38",
    }

    # Calculate conditional probabilities
    sessions_with_fvg = set()
    sessions_with_pm_belt_redelivery = set()
    sessions_with_both = set()

    for candidate in candidates:
        session_id = candidate["session_id"]

        if candidate["event_type"] == "formation":
            sessions_with_fvg.add(session_id)

        if candidate["event_type"] == "redelivery" and candidate["in_pm_belt"]:
            sessions_with_pm_belt_redelivery.add(session_id)

    # Find sessions with both
    sessions_with_both = sessions_with_fvg.intersection(sessions_with_pm_belt_redelivery)

    # Calculate probabilities
    total_sessions = len({c["session_id"] for c in candidates})

    if total_sessions == 0:
        pm_belt_results["error"] = "No sessions found"
        return pm_belt_results

    # P(PM belt redelivery | FVG formation in session)
    if len(sessions_with_fvg) > 0:
        p_pm_given_fvg = len(sessions_with_both) / len(sessions_with_fvg)
    else:
        p_pm_given_fvg = 0.0

    # P(PM belt redelivery) - baseline probability
    p_pm_baseline = len(sessions_with_pm_belt_redelivery) / total_sessions

    # Chi-square test for independence
    try:
        # Contingency table: [FVG & PM, FVG & ~PM], [~FVG & PM, ~FVG & ~PM]
        fvg_and_pm = len(sessions_with_both)
        fvg_not_pm = len(sessions_with_fvg) - fvg_and_pm
        not_fvg_and_pm = len(sessions_with_pm_belt_redelivery) - fvg_and_pm
        not_fvg_not_pm = total_sessions - fvg_and_pm - fvg_not_pm - not_fvg_and_pm

        contingency_table = [[fvg_and_pm, fvg_not_pm], [not_fvg_and_pm, not_fvg_not_pm]]

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        pm_belt_results.update(
            {
                "sessions_with_fvg": len(sessions_with_fvg),
                "sessions_with_pm_belt_redelivery": len(sessions_with_pm_belt_redelivery),
                "sessions_with_both": len(sessions_with_both),
                "total_sessions": total_sessions,
                "p_pm_given_fvg": p_pm_given_fvg,
                "p_pm_baseline": p_pm_baseline,
                "relative_risk": p_pm_given_fvg / max(0.001, p_pm_baseline),
                "contingency_table": contingency_table,
                "chi2_statistic": chi2,
                "p_value": p_value,
                "significant": p_value < alpha,
            }
        )

    except Exception as e:
        pm_belt_results["error"] = f"Statistical test failed: {e}"

    return pm_belt_results


def test_reproducibility(
    candidates: list[dict[str, Any]],
    network_graph: dict[str, Any],  # noqa: ARG001
    bootstrap_iterations: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """
    Test reproducibility with per-session bootstrap analysis

    Goal: Validate that findings are reproducible across sessions
    """
    reproducibility_results = {
        "test_type": "reproducibility_bootstrap_analysis",
        "bootstrap_iterations": bootstrap_iterations,
        "confidence_level": confidence_level,
    }

    # Group candidates by session
    sessions = {}
    for candidate in candidates:
        session_id = candidate["session_id"]
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(candidate)

    if len(sessions) < 2:
        reproducibility_results["error"] = "Need at least 2 sessions for reproducibility test"
        return reproducibility_results

    # Bootstrap sampling across sessions
    session_ids = list(sessions.keys())
    bootstrap_scores = []

    try:
        for _ in range(bootstrap_iterations):
            # Sample sessions with replacement
            sampled_sessions = np.random.choice(session_ids, size=len(session_ids), replace=True)

            # Combine candidates from sampled sessions
            bootstrap_candidates = []
            for session_id in sampled_sessions:
                bootstrap_candidates.extend(sessions[session_id])

            # Calculate summary metric (e.g., zone enrichment rate)
            if bootstrap_candidates:
                redeliveries = [c for c in bootstrap_candidates if c["event_type"] == "redelivery"]
                if redeliveries:
                    in_zone_rate = len(
                        [
                            c
                            for c in redeliveries
                            if c.get("zone_proximity", {}).get("in_zone", False)
                        ]
                    ) / len(redeliveries)
                    bootstrap_scores.append(in_zone_rate)

        if bootstrap_scores:
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            ci_lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
            ci_upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

            reproducibility_results.update(
                {
                    "sessions_analyzed": len(sessions),
                    "bootstrap_scores": {
                        "mean": np.mean(bootstrap_scores),
                        "std": np.std(bootstrap_scores),
                        "median": np.median(bootstrap_scores),
                        "min": np.min(bootstrap_scores),
                        "max": np.max(bootstrap_scores),
                    },
                    "confidence_interval": {
                        "lower": ci_lower,
                        "upper": ci_upper,
                        "level": confidence_level,
                    },
                    "coefficient_of_variation": (
                        np.std(bootstrap_scores) / np.mean(bootstrap_scores)
                        if np.mean(bootstrap_scores) > 0
                        else float("inf")
                    ),
                }
            )
        else:
            reproducibility_results["error"] = "No valid bootstrap scores generated"

    except Exception as e:
        reproducibility_results["error"] = f"Bootstrap analysis failed: {e}"

    return reproducibility_results


def generate_summary_insights(analysis_results: dict[str, Any]) -> dict[str, Any]:
    """Generate high-level insights from analysis results"""
    insights = {
        "key_findings": [],
        "statistical_significance": {},
        "recommendations": [],
        "data_quality": {},
    }

    # Zone enrichment insights
    zone_test = analysis_results.get("zone_enrichment_test", {})
    if not zone_test.get("error"):
        insights["statistical_significance"]["zone_enrichment"] = zone_test.get(
            "significant", False
        )
        if zone_test.get("significant"):
            enrichment_factor = zone_test.get("enrichment_factor", 1.0)
            insights["key_findings"].append(
                f"Significant zone enrichment detected (factor: {enrichment_factor:.2f})"
            )
        else:
            insights["key_findings"].append("No significant zone enrichment detected")

    # PM belt interaction insights
    pm_test = analysis_results.get("pm_belt_interaction_test", {})
    if not pm_test.get("error"):
        insights["statistical_significance"]["pm_belt_interaction"] = pm_test.get(
            "significant", False
        )
        relative_risk = pm_test.get("relative_risk", 1.0)
        if pm_test.get("significant"):
            insights["key_findings"].append(
                f"Significant PM belt interaction (relative risk: {relative_risk:.2f})"
            )

    # Network topology insights
    network_construction = analysis_results.get("network_construction", {})
    if network_construction:
        density = network_construction.get("network_density", 0)
        motifs = network_construction.get("network_motifs", {})

        insights["key_findings"].append(f"Network density: {density:.3f}")

        if motifs:
            chain_count = motifs.get("chain_count", 0)
            if chain_count > 0:
                insights["key_findings"].append(f"Found {chain_count} redelivery chains")

    # Data quality assessment
    candidate_extraction = analysis_results.get("candidate_extraction", {})
    if candidate_extraction:
        total_candidates = candidate_extraction.get("total_candidates", 0)
        formation_count = candidate_extraction.get("formation_count", 0)
        redelivery_count = candidate_extraction.get("redelivery_count", 0)

        insights["data_quality"]["total_candidates"] = total_candidates
        insights["data_quality"]["formation_redelivery_ratio"] = formation_count / max(
            1, redelivery_count
        )

    # Recommendations
    if len(insights["key_findings"]) > 0:
        insights["recommendations"].append("Continue monitoring FPFVG patterns")

    if insights["statistical_significance"].get("zone_enrichment"):
        insights["recommendations"].append("Focus on Theory B zone events for trading signals")

    if insights["statistical_significance"].get("pm_belt_interaction"):
        insights["recommendations"].append("Monitor PM belt timing for enhanced signal detection")

    return insights
