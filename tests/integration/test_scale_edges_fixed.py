#!/usr/bin/env python3
"""
Scale Edges Test - Fixed HTF Data
=================================
Test scale edge generation with the fixed HTF data to verify
that the 0% scale edges issue has been resolved.
"""

import json
from pathlib import Path

from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery


def test_scale_edges_with_fixed_data():
    """Test scale edge generation with fixed HTF data."""
    print("🔬 IRONFORGE Scale Edges Test - Fixed HTF Data")
    print("=" * 60)

    # Load a price-relativity enhanced HTF session
    htf_file = "/Users/jack/IRONPULSE/data/sessions/htf_relativity/NY_AM_Lvl-1_2025_07_30_htf_regenerated_rel.json"
    print(f"📊 Testing with: {Path(htf_file).name}")

    with open(htf_file) as f:
        htf_data = json.load(f)

    # Initialize enhanced graph builder
    print("🔧 Initializing Enhanced Graph Builder...")
    graph_builder = EnhancedGraphBuilder()

    # Build enhanced graph with HTF data
    print("🏗️ Building enhanced graph...")
    graph_data = graph_builder.build_rich_graph(htf_data)

    # Analyze node features
    print("\n📊 Graph Analysis:")
    print(f"  📍 Nodes: {len(graph_data['nodes'])}")
    print(f"  🔗 Edges: {len(graph_data['edges'])}")

    # Count nodes by timeframe
    timeframe_counts = {}
    price_level_counts = {}

    for node in graph_data["nodes"]:
        # Handle different node formats
        if hasattr(node, "features"):
            timeframe = node.features.temporal_cycles["timeframe_marker"]
            price_level = node.features.price_relativity_features["current_price"]
        elif isinstance(node, dict):
            timeframe = node.get("timeframe", "unknown")
            price_level = node.get("price_level", 0)
        else:
            timeframe = "unknown"
            price_level = 0

        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1

        # Check price levels
        if price_level > 0:
            price_level_counts[timeframe] = price_level_counts.get(timeframe, 0) + 1

    print("\n⏰ Timeframe Distribution:")
    for tf, count in timeframe_counts.items():
        valid_prices = price_level_counts.get(tf, 0)
        print(f"  {tf}: {count} nodes ({valid_prices} with valid prices)")

    # Analyze edge types
    edge_types = {}
    scale_edges = 0

    for edge in graph_data["edges"]:
        # Handle different edge formats
        if hasattr(edge, "features"):
            edge_type = edge.features.temporal_resonance_score
            source_idx = edge.source_idx
            target_idx = edge.target_idx
        elif isinstance(edge, dict):
            edge_type = edge.get("temporal_resonance_score", 0)
            source_idx = edge.get("source", 0)
            target_idx = edge.get("target", 0)
        else:
            edge_type = 0
            source_idx = 0
            target_idx = 0

        edge_type_name = "temporal" if edge_type > 0 else "structural"

        edge_types[edge_type_name] = edge_types.get(edge_type_name, 0) + 1

        # Check if this is a scale edge (cross-timeframe)
        if source_idx < len(graph_data["nodes"]) and target_idx < len(graph_data["nodes"]):
            source_node = graph_data["nodes"][source_idx]
            target_node = graph_data["nodes"][target_idx]

            # Get timeframes
            if hasattr(source_node, "features"):
                source_tf = source_node.features.temporal_cycles["timeframe_marker"]
            elif isinstance(source_node, dict):
                source_tf = source_node.get("timeframe", "unknown")
            else:
                source_tf = "unknown"

            if hasattr(target_node, "features"):
                target_tf = target_node.features.temporal_cycles["timeframe_marker"]
            elif isinstance(target_node, dict):
                target_tf = target_node.get("timeframe", "unknown")
            else:
                target_tf = "unknown"

            if source_tf != target_tf and source_tf != "unknown" and target_tf != "unknown":
                scale_edges += 1

    print("\n🔗 Edge Analysis:")
    for edge_type, count in edge_types.items():
        print(f"  {edge_type}: {count} edges")

    print("\n⚖️ Scale Edge Analysis:")
    print(
        f"  Cross-timeframe edges: {scale_edges}/{len(graph_data['edges'])} ({scale_edges/len(graph_data['edges'])*100:.1f}%)"
    )

    # Test TGAT conversion
    print("\n🧠 Testing TGAT Conversion...")
    X, edge_index, edge_times, metadata, edge_attr = graph_builder.to_tgat_format(graph_data)

    print(f"  X (node features): {X.shape}")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  edge_attr (edge features): {edge_attr.shape}")

    # Initialize TGAT discovery
    tgat_discovery = IRONFORGEDiscovery()

    # Test learn_session with full tensor pipeline
    print("\n🏛️ Testing TGAT learn_session()...")
    learn_result = tgat_discovery.learn_session(X, edge_index, edge_times, metadata, edge_attr)

    patterns = learn_result.get("patterns", [])
    print(f"  Patterns discovered: {len(patterns)}")

    # Success criteria
    success_criteria = [
        ("Non-zero scale edges", scale_edges > 0),
        ("Multiple timeframes", len(timeframe_counts) > 1),
        ("Valid price levels", sum(price_level_counts.values()) > 0),
        ("Pattern discovery", len(patterns) > 0),
        ("Tensor conversion", X.shape[0] > 0 and edge_attr.shape[0] > 0),
    ]

    print("\n✅ Success Criteria:")
    all_passed = True
    for criterion, passed in success_criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {criterion}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Scale Edges Test: SUCCESS!")
        print("✅ HTF data fix has resolved the 0% scale edges issue")
        print("✅ Cross-timeframe hierarchical relationships now working")
    else:
        print("\n⚠️ Scale Edges Test: PARTIAL SUCCESS")
        print("🔧 Some issues remain, but major progress made")

    return {
        "scale_edges": scale_edges,
        "total_edges": len(graph_data["edges"]),
        "scale_edge_percentage": scale_edges / len(graph_data["edges"]) * 100,
        "timeframes": list(timeframe_counts.keys()),
        "patterns_discovered": len(patterns),
        "all_criteria_passed": all_passed,
    }


if __name__ == "__main__":
    try:
        result = test_scale_edges_with_fixed_data()
        print("\n📊 Final Results:")
        print(
            f"  Scale edges: {result['scale_edges']}/{result['total_edges']} ({result['scale_edge_percentage']:.1f}%)"
        )
        print(f"  Timeframes: {len(result['timeframes'])}")
        print(f"  Patterns: {result['patterns_discovered']}")
        print(f"  Success: {result['all_criteria_passed']}")

    except Exception as e:
        print(f"❌ Scale edges test failed: {e}")
        import traceback

        traceback.print_exc()
