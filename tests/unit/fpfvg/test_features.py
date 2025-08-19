"""Test FPFVG features functionality."""

import pytest

from ironforge.analysis.fpfvg.features import (
    analyze_score_distribution,
    calculate_price_proximity_score,
    calculate_range_pos_proximity_score,
    calculate_range_position,
    calculate_temporal_penalty_score,
    calculate_zone_confluence_score,
    extract_magnitude,
    get_candidate_summary_stats,
    get_zone_proximity,
)


def test_calculate_price_proximity_score():
    """Test price proximity scoring."""
    # Identical prices should score 1.0
    edge_identical = {"price_distance": 0}
    assert calculate_price_proximity_score(edge_identical) == 1.0

    # Exponential decay for non-zero distances
    edge_close = {"price_distance": 2.5}
    score = calculate_price_proximity_score(edge_close, price_epsilon=5.0)
    assert 0.0 < score < 1.0

    # Larger distance should have lower score
    edge_far = {"price_distance": 10.0}
    score_far = calculate_price_proximity_score(edge_far, price_epsilon=5.0)
    assert score_far < score


def test_calculate_range_pos_proximity_score():
    """Test range position proximity scoring."""
    # Identical range positions should score 1.0
    edge_identical = {"delta_range_pos": 0}
    assert calculate_range_pos_proximity_score(edge_identical) == 1.0

    # Exponential decay for non-zero deltas
    edge_close = {"delta_range_pos": 0.02}
    score = calculate_range_pos_proximity_score(edge_close, range_pos_delta=0.05)
    assert 0.0 < score < 1.0

    # Larger delta should have lower score
    edge_far = {"delta_range_pos": 0.1}
    score_far = calculate_range_pos_proximity_score(edge_far, range_pos_delta=0.05)
    assert score_far < score


def test_calculate_zone_confluence_score():
    """Test zone confluence scoring."""
    theory_b_zones = [0.2, 0.4, 0.5, 0.618, 0.8]

    # No zone flags should score 0.0
    edge_no_zones = {"same_zone_flags": {}}
    assert calculate_zone_confluence_score(edge_no_zones, theory_b_zones) == 0.0

    # All zones aligned should score 1.0
    edge_all_zones = {
        "same_zone_flags": {
            "20.0%": True,
            "40.0%": True,
            "50.0%": True,
            "61.8%": True,
            "80.0%": True,
        }
    }
    assert calculate_zone_confluence_score(edge_all_zones, theory_b_zones) == 1.0

    # Partial alignment should score proportionally
    edge_partial = {
        "same_zone_flags": {
            "20.0%": True,
            "40.0%": True,
            "50.0%": False,
            "61.8%": False,
            "80.0%": False,
        }
    }
    expected_score = 2 / 5  # 2 aligned out of 5 zones
    assert calculate_zone_confluence_score(edge_partial, theory_b_zones) == expected_score


def test_calculate_temporal_penalty_score():
    """Test temporal penalty scoring."""
    # No time delta should have no penalty
    edge_no_delta = {"delta_t_minutes": 0}
    assert calculate_temporal_penalty_score(edge_no_delta) == 0.0

    # Maximum gap should have maximum penalty (1.0)
    edge_max_gap = {"delta_t_minutes": 720}  # 12 hours * 60 minutes
    assert calculate_temporal_penalty_score(edge_max_gap, max_temporal_gap_hours=12.0) == 1.0

    # Half the maximum gap should have penalty 0.5
    edge_half_gap = {"delta_t_minutes": 360}  # 6 hours * 60 minutes
    expected_penalty = 360 / 720  # 0.5
    assert (
        calculate_temporal_penalty_score(edge_half_gap, max_temporal_gap_hours=12.0)
        == expected_penalty
    )


def test_calculate_range_position():
    """Test range position calculation."""
    session_ranges = {"session_1": {"low": 23000.0, "high": 23100.0}}

    # Price at low should give position 0.0
    assert calculate_range_position(23000.0, "session_1", session_ranges) == 0.0

    # Price at high should give position 1.0
    assert calculate_range_position(23100.0, "session_1", session_ranges) == 1.0

    # Price at midpoint should give position 0.5
    assert calculate_range_position(23050.0, "session_1", session_ranges) == 0.5

    # Price below low should be clamped to 0.0
    assert calculate_range_position(22900.0, "session_1", session_ranges) == 0.0

    # Price above high should be clamped to 1.0
    assert calculate_range_position(23200.0, "session_1", session_ranges) == 1.0

    # Unknown session should return default (0.5)
    assert calculate_range_position(23050.0, "unknown_session", session_ranges) == 0.5


def test_get_zone_proximity():
    """Test zone proximity calculation."""
    theory_b_zones = [0.2, 0.4, 0.5, 0.618, 0.8]

    # Exact zone match should have zero distance and be in zone
    result = get_zone_proximity(0.4, theory_b_zones, zone_tolerance=0.03)
    assert result["range_position"] == 0.4
    assert result["closest_zone"] == "40.0%"
    assert result["distance_to_closest"] == 0.0
    assert result["in_zone"] is True
    assert "40.0%" in result["closest_zones"]

    # Near zone (within tolerance) should be in zone
    result = get_zone_proximity(0.42, theory_b_zones, zone_tolerance=0.03)
    assert result["closest_zone"] == "40.0%"
    assert abs(result["distance_to_closest"] - 0.02) < 1e-10  # Handle floating point precision
    assert result["in_zone"] is True

    # Far from all zones should not be in zone
    result = get_zone_proximity(0.3, theory_b_zones, zone_tolerance=0.03)
    assert result["in_zone"] is False
    assert len(result["closest_zones"]) == 0


def test_extract_magnitude():
    """Test magnitude extraction from event data."""
    # Direct magnitude field
    event_with_magnitude = {"magnitude": 5.5}
    assert extract_magnitude(event_with_magnitude) == 5.5

    # Gap size field
    event_with_gap_size = {"gap_size": 12.3}
    assert extract_magnitude(event_with_gap_size) == 12.3

    # Volume field
    event_with_volume = {"volume": 1000.0}
    assert extract_magnitude(event_with_volume) == 1000.0

    # No magnitude fields should return default
    event_no_magnitude = {"other_field": "value"}
    assert extract_magnitude(event_no_magnitude) == 1.0

    # Invalid magnitude should return default
    event_invalid_magnitude = {"magnitude": "not_a_number"}
    assert extract_magnitude(event_invalid_magnitude) == 1.0


def test_get_candidate_summary_stats():
    """Test candidate summary statistics."""
    candidates = [
        {
            "event_type": "formation",
            "session_id": "session_1",
            "in_pm_belt": False,
            "range_pos": 0.3,
            "zone_proximity": {"in_zone": False},
        },
        {
            "event_type": "redelivery",
            "session_id": "session_1",
            "in_pm_belt": True,
            "range_pos": 0.7,
            "zone_proximity": {"in_zone": True},
        },
        {
            "event_type": "formation",
            "session_id": "session_2",
            "in_pm_belt": False,
            "range_pos": 0.5,
            "zone_proximity": {"in_zone": False},
        },
    ]

    stats = get_candidate_summary_stats(candidates)

    assert stats["total_candidates"] == 3
    assert stats["formation_count"] == 2
    assert stats["redelivery_count"] == 1
    assert stats["unique_sessions"] == 2
    assert stats["pm_belt_count"] == 1
    assert stats["pm_belt_percentage"] == pytest.approx(33.33, rel=1e-2)
    assert stats["in_zone_count"] == 1
    assert stats["zone_percentage"] == pytest.approx(33.33, rel=1e-2)

    # Check range position stats
    range_stats = stats["range_position_stats"]
    assert range_stats["mean"] == pytest.approx(0.5)  # (0.3 + 0.7 + 0.5) / 3
    assert range_stats["min"] == 0.3
    assert range_stats["max"] == 0.7


def test_get_candidate_summary_stats_empty():
    """Test candidate summary statistics with empty list."""
    stats = get_candidate_summary_stats([])

    assert stats == {}


def test_analyze_score_distribution():
    """Test score distribution analysis."""
    redelivery_scores = [
        {"strength": 0.8},
        {"strength": 0.6},
        {"strength": 0.4},
        {"strength": 0.9},
        {"strength": 0.5},
    ]

    distribution = analyze_score_distribution(redelivery_scores)

    assert distribution["count"] == 5
    assert distribution["mean"] == pytest.approx(0.64)  # (0.8+0.6+0.4+0.9+0.5)/5
    assert distribution["median"] == pytest.approx(0.6)
    assert distribution["min"] == 0.4
    assert distribution["max"] == 0.9
    assert "percentiles" in distribution
    assert "25th" in distribution["percentiles"]
    assert "75th" in distribution["percentiles"]
    assert "90th" in distribution["percentiles"]


def test_analyze_score_distribution_empty():
    """Test score distribution analysis with empty list."""
    distribution = analyze_score_distribution([])

    assert distribution == {}
