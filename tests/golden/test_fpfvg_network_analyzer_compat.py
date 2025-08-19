"""Golden compatibility test for FPFVG Network Analyzer refactoring."""

import json
import tempfile
from pathlib import Path

from ironforge.analysis.fpfvg_network_analyzer import FPFVGNetworkAnalyzer


def create_test_session_data():
    """Create test session data for golden test."""
    return {
        "session_id": "test_session_20250801_14",
        "nodes": [
            {
                "id": "node_1",
                "timestamp": "2025-08-01T14:30:00",
                "price": 23000.0,
                "event_type": "fvg_formation",
                "magnitude": 5.5,
            },
            {
                "id": "node_2",
                "timestamp": "2025-08-01T14:35:00",
                "price": 23005.0,
                "event_type": "fvg_redelivery",
                "magnitude": 3.2,
            },
            {
                "id": "node_3",
                "timestamp": "2025-08-01T14:40:00",
                "price": 23010.0,
                "event_type": "fvg_formation",
                "magnitude": 2.1,
            },
        ],
        "edges": [
            {
                "source": "node_1",
                "target": "node_2",
                "weight": 0.8,
            },
            {
                "source": "node_2",
                "target": "node_3",
                "weight": 0.6,
            },
        ],
        "metadata": {
            "session_start": "2025-08-01T14:00:00",
            "session_end": "2025-08-01T15:00:00",
            "session_low": 22950.0,
            "session_high": 23100.0,
        },
    }


def test_fpfvg_analyzer_initialization():
    """Test that FPFVGNetworkAnalyzer can be initialized without errors."""
    analyzer = FPFVGNetworkAnalyzer()

    # Verify key attributes are set
    assert analyzer.price_epsilon == 5.0
    assert analyzer.range_pos_delta == 0.05
    assert analyzer.max_temporal_gap_hours == 12.0
    assert analyzer.zone_tolerance == 0.03
    assert analyzer.alpha == 0.05
    assert len(analyzer.theory_b_zones) == 5
    assert 0.4 in analyzer.theory_b_zones  # Theory B zone
    assert 0.618 in analyzer.theory_b_zones  # Golden ratio zone


def test_fpfvg_analyzer_backwards_compatibility():
    """Test that old import patterns still work."""
    # Test main class import
    from ironforge.analysis.fpfvg_network_analyzer import FPFVGNetworkAnalyzer

    analyzer = FPFVGNetworkAnalyzer()
    assert analyzer is not None

    # Test function imports
    from ironforge.analysis.fpfvg_network_analyzer import (
        build_chains,
        compute_chain_features,
        validate_chain,
    )

    assert callable(build_chains)
    assert callable(validate_chain)
    assert callable(compute_chain_features)


def test_analyze_fpfvg_network_structure():
    """Test that analyze_fpfvg_network returns expected structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup temporary paths
        temp_path = Path(temp_dir)
        enhanced_path = temp_path / "enhanced"
        discoveries_path = temp_path / "discoveries"
        enhanced_path.mkdir()
        discoveries_path.mkdir()

        # Create test session file
        test_session = create_test_session_data()
        session_file = enhanced_path / "enhanced_rel_test_session.json"
        with open(session_file, "w") as f:
            json.dump(test_session, f)

        # Initialize analyzer with test paths
        analyzer = FPFVGNetworkAnalyzer()
        analyzer.enhanced_path = enhanced_path
        analyzer.discoveries_path = discoveries_path

        # Run analysis
        result = analyzer.analyze_fpfvg_network()

        # Verify result structure
        assert isinstance(result, dict)
        assert "analysis_type" in result
        assert result["analysis_type"] == "fpfvg_network_analysis"
        assert "timestamp" in result
        assert "parameters" in result

        # Verify parameters are preserved
        params = result["parameters"]
        assert params["price_epsilon"] == 5.0
        assert params["range_pos_delta"] == 0.05
        assert params["max_temporal_gap_hours"] == 12.0
        assert params["zone_tolerance"] == 0.03
        assert params["theory_b_zones"] == [0.2, 0.4, 0.5, 0.618, 0.8]

        # Check for main analysis sections
        if "error" not in result:
            # Only check these if analysis succeeded
            expected_sections = [
                "candidate_validation",
                "candidate_extraction",
                "network_construction",
                "redelivery_scoring",
                "zone_enrichment_test",
                "pm_belt_interaction_test",
                "reproducibility_test",
                "summary_insights",
            ]

            for section in expected_sections:
                assert section in result, f"Missing expected section: {section}"


def test_fpfvg_public_functions():
    """Test that public functions work as expected."""
    from ironforge.analysis.fpfvg_network_analyzer import (
        build_chains,
        compute_chain_features,
        validate_chain,
    )

    # Test build_chains
    adjacency = {"A": ["B"], "B": ["C"], "C": []}
    chains = build_chains(adjacency, min_length=2)
    assert isinstance(chains, list)

    # Test validate_chain
    candidates = [
        {
            "id": "test_1",
            "session_id": "session_1",
            "event_type": "formation",
            "price_level": 23000.0,
            "range_pos": 0.5,
            "start_ts": "2025-08-01T14:30:00",
            "in_pm_belt": False,
            "zone_proximity": {"in_zone": False},
            "timeframe": "15m",
        }
    ]
    validation = validate_chain(candidates)
    assert isinstance(validation, dict)
    assert "valid" in validation

    # Test compute_chain_features
    network_graph = {
        "nodes": [{"id": "1"}, {"id": "2"}],
        "edges": [
            {
                "source": "1",
                "target": "2",
                "price_distance": 5.0,
                "delta_range_pos": 0.1,
                "delta_t_minutes": 30.0,
                "same_zone_flags": {"40.0%": True},
            }
        ],
    }
    features = compute_chain_features(network_graph)
    assert isinstance(features, list)


def test_module_docstring_preservation():
    """Test that module docstring is preserved for documentation."""
    import ironforge.analysis.fpfvg_network_analyzer as module

    assert hasattr(module, "__doc__")
    assert module.__doc__ is not None
    assert "BACKWARD COMPATIBILITY" in module.__doc__
    assert "Compatibility Shim" in module.__doc__


def test_all_exports_available():
    """Test that all expected exports are available."""
    import ironforge.analysis.fpfvg_network_analyzer as module

    expected_exports = [
        "FPFVGNetworkAnalyzer",
        "build_chains",
        "validate_chain",
        "compute_chain_features",
        "find_chains",
        "calculate_network_density",
        "identify_network_motifs",
        "validate_network_graph",
        "is_in_pm_belt",
        "safe_float",
        "analyze_score_distribution",
        "calculate_range_position",
        "get_zone_proximity",
        "extract_magnitude",
        "get_candidate_summary_stats",
        "test_zone_enrichment",
        "test_pm_belt_interaction",
        "test_reproducibility",
        "generate_summary_insights",
    ]

    for export in expected_exports:
        assert hasattr(module, export), f"Missing export: {export}"
        assert export in module.__all__, f"Export {export} not in __all__"


def test_no_behavior_change_smoke_test():
    """Smoke test to ensure basic functionality hasn't changed."""
    # This test ensures that the refactoring hasn't broken basic usage patterns
    analyzer = FPFVGNetworkAnalyzer()

    # Test that key methods are still callable
    assert hasattr(analyzer, "analyze_fpfvg_network")
    assert callable(analyzer.analyze_fpfvg_network)

    # Test that configuration is still accessible
    assert hasattr(analyzer, "price_epsilon")
    assert hasattr(analyzer, "theory_b_zones")
    assert hasattr(analyzer, "scoring_weights")

    # Test that the main analysis method returns a dict structure
    # (This will create mock data since we don't have real enhanced sessions)
    result = analyzer.analyze_fpfvg_network()
    assert isinstance(result, dict)
    assert "analysis_type" in result
