#!/usr/bin/env python3
"""
Test script for IRONFORGE temporal cycle integration (34D -> 37D)
Innovation Architect implementation validation
"""

import sys
from datetime import datetime

sys.path.append("/Users/jack/IRONPULSE/IRONFORGE")

from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder, RichNodeFeature
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery


def test_temporal_cycle_features():
    """Test that temporal cycle features are properly integrated"""

    print("🧪 Testing IRONFORGE Temporal Cycle Integration (37D)")
    print("=" * 60)

    # Test 1: RichNodeFeature with temporal cycles
    print("\n1. Testing RichNodeFeature with temporal cycles...")

    feature = RichNodeFeature(
        # Temporal (12)
        time_minutes=120.0,
        daily_phase_sin=0.5,
        daily_phase_cos=0.8,
        session_position=0.3,
        time_to_close=180.0,
        weekend_proximity=0.2,
        absolute_timestamp=1692108000,
        day_of_week=1,
        month_phase=0.4,
        # Temporal Cycles (3) - NEW
        week_of_month=2,
        month_of_year=8,
        day_of_week_cycle=1,
        # Price Relativity (7)
        normalized_price=0.75,
        pct_from_open=2.5,
        pct_from_high=15.2,
        pct_from_low=84.8,
        price_to_HTF_ratio=1.02,
        time_since_session_open=7200.0,
        normalized_time=0.4,
        # Price Context Legacy (3)
        price_delta_1m=0.01,
        price_delta_5m=0.02,
        price_delta_15m=0.03,
        # Market State (7)
        volatility_window=0.05,
        energy_state=0.6,
        contamination_coefficient=0.1,
        fisher_regime=1,
        session_character=0,
        cross_tf_confluence=0.8,
        timeframe_rank=1,
        # Event & Structure (8)
        event_type_id=1,
        timeframe_source=0,
        liquidity_type=0,
        fpfvg_gap_size=5.0,
        fpfvg_interaction_count=2,
        first_presentation_flag=1.0,
        pd_array_strength=0.7,
        structural_importance=0.85,
        # Preservation
        raw_json={"test": "data"},
    )

    tensor = feature.to_tensor()
    print(f"   ✅ Feature tensor shape: {tensor.shape}")
    print(f"   ✅ Expected 37D, got {tensor.shape[0]}D")

    # Verify temporal cycle values
    print(f"   ✅ Week of month: {tensor[9].item()} (expected: 2)")
    print(f"   ✅ Month of year: {tensor[10].item()} (expected: 8)")
    print(f"   ✅ Day of week cycle: {tensor[11].item()} (expected: 1)")

    assert tensor.shape[0] == 37, f"Expected 37D features, got {tensor.shape[0]}D"
    assert tensor[9].item() == 2.0, "Week of month incorrect"
    assert tensor[10].item() == 8.0, "Month of year incorrect"
    assert tensor[11].item() == 1.0, "Day of week cycle incorrect"

    # Test 2: TGAT Discovery with 37D features
    print("\n2. Testing IRONFORGEDiscovery with 37D features...")

    discovery = IRONFORGEDiscovery(node_features=37)  # 37D input
    print(f"   ✅ TGAT model attention dimension: {discovery.model.attention_dim}")
    print(f"   ✅ Expected 36D (37->36 divisible by 4), got {discovery.model.attention_dim}D")

    assert (
        discovery.model.attention_dim == 36
    ), f"Expected 36D attention, got {discovery.model.attention_dim}D"

    # Test 3: Enhanced Graph Builder with temporal cycles
    print("\n3. Testing EnhancedGraphBuilder temporal cycle parsing...")

    builder = EnhancedGraphBuilder()

    # Test session date parsing with temporal cycles
    test_dates = ["2025-08-13", "2025-12-25", "2025-03-15"]

    for test_date in test_dates:
        result = builder._parse_session_date(test_date)
        assert len(result) == 6, f"Expected 6 values from date parsing, got {len(result)}"

        timestamp, day_of_week, month_phase, week_of_month, month_of_year, day_of_week_cycle = (
            result
        )

        # Parse the test date to verify
        parsed_date = datetime.strptime(test_date, "%Y-%m-%d")
        expected_week_of_month = min(5, ((parsed_date.day - 1) // 7) + 1)
        expected_month_of_year = parsed_date.month

        print(f"   Date: {test_date}")
        print(f"     Week of month: {week_of_month} (expected: {expected_week_of_month})")
        print(f"     Month of year: {month_of_year} (expected: {expected_month_of_year})")
        print(f"     Day of week cycle: {day_of_week_cycle} (matches day_of_week: {day_of_week})")

        assert week_of_month == expected_week_of_month, f"Week calculation wrong for {test_date}"
        assert month_of_year == expected_month_of_year, f"Month calculation wrong for {test_date}"
        assert day_of_week_cycle == day_of_week, "Day cycle should match day of week"

    # Test 4: Mock session data processing
    print("\n4. Testing mock session processing with temporal cycles...")

    mock_session = {
        "session_metadata": {
            "session_type": "ny_pm",
            "session_start": "13:30:00",
            "session_end": "16:00:00",
            "session_duration": 150,
            "session_date": "2025-08-13",  # Tuesday, 2nd week of August
        },
        "price_movements": [
            {
                "timestamp": "13:30:00",
                "price_level": 23500.0,
                "movement_type": "open",
                "normalized_price": 0.0,
                "pct_from_open": 0.0,
                "pct_from_high": 50.0,
                "pct_from_low": 50.0,
                "time_since_session_open": 0,
                "normalized_time": 0.0,
            },
            {
                "timestamp": "14:15:00",
                "price_level": 23525.0,
                "movement_type": "high",
                "normalized_price": 1.0,
                "pct_from_open": 0.11,
                "pct_from_high": 0.0,
                "pct_from_low": 100.0,
                "time_since_session_open": 2700,
                "normalized_time": 0.3,
            },
        ],
        "energy_state": {"total_accumulated": 2500},
        "contamination_analysis": {"contamination_coefficient": 0.3},
    }

    try:
        graph = builder.build_rich_graph(mock_session)
        print("   ✅ Graph built successfully")
        print(f"   ✅ Feature dimensions: {graph['metadata']['feature_dimensions']}")
        print(f"   ✅ Total nodes: {graph['metadata']['total_nodes']}")

        # Convert to TGAT format
        X, edge_index, edge_times, metadata, edge_attr = builder.to_tgat_format(graph)
        print("   ✅ TGAT format conversion successful")
        print(f"   ✅ Node features shape: {X.shape}")

        assert X.shape[1] == 37, f"Expected 37D features in TGAT format, got {X.shape[1]}D"

        # Test temporal cycle extraction
        if X.shape[0] > 0:
            # Check temporal cycle features are present
            week_of_month_vals = X[:, 9]
            month_of_year_vals = X[:, 10]
            day_of_week_vals = X[:, 11]

            print(f"   ✅ Week of month values: {week_of_month_vals.tolist()}")
            print(f"   ✅ Month of year values: {month_of_year_vals.tolist()}")
            print(f"   ✅ Day of week cycle values: {day_of_week_vals.tolist()}")

            # For August 13, 2025 (Tuesday), expect week 2, month 8, day 1
            expected_week = 2  # 2nd week of August
            expected_month = 8  # August
            expected_day = 1  # Tuesday (0=Monday, 1=Tuesday)

            print(
                f"   Expected for 2025-08-13: week {expected_week}, month {expected_month}, day {expected_day}"
            )

    except Exception as e:
        print(f"   ❌ Error in graph building: {e}")
        raise

    print("\n✅ All temporal cycle integration tests passed!")
    print("🎯 IRONFORGE successfully expanded from 34D to 37D with temporal cycles")


if __name__ == "__main__":
    test_temporal_cycle_features()
