#!/usr/bin/env python3
"""
Scale Edges Final Test - Complete HTF Pipeline
==============================================
Final test of scale edge generation with the complete HTF data pipeline.
Tests the full Level 1 → HTF → TGAT archaeological discovery flow.
"""

import json
from pathlib import Path

from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery


def test_complete_archaeological_pipeline():
    """Test the complete archaeological discovery pipeline."""
    print("🏛️ IRONFORGE Complete Archaeological Pipeline Test")
    print("=" * 70)

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

    # Analyze graph structure
    print("\n📊 Graph Analysis:")
    print(f"  📍 Node dictionary: {len(graph_data['nodes'])} entries")
    print(f"  🔗 Edge dictionary: {len(graph_data['edges'])} entries")
    print(f"  🧠 Rich node features: {len(graph_data['rich_node_features'])}")
    print(f"  ⚡ Rich edge features: {len(graph_data['rich_edge_features'])}")

    # Analyze timeframes from rich node features
    timeframe_counts = {}
    price_level_counts = {}

    timeframe_mapping = {0: "1m", 1: "5m", 2: "15m", 3: "1h", 4: "D", 5: "W"}

    for node_features in graph_data["rich_node_features"]:
        timeframe_id = node_features.timeframe_source
        timeframe = timeframe_mapping.get(timeframe_id, f"unknown_{timeframe_id}")
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1

        # Check price levels
        price_level = node_features.normalized_price
        if price_level > 0:
            price_level_counts[timeframe] = price_level_counts.get(timeframe, 0) + 1

    print("\n⏰ Timeframe Distribution:")
    for tf, count in timeframe_counts.items():
        valid_prices = price_level_counts.get(tf, 0)
        print(f"  {tf}: {count} nodes ({valid_prices} with valid prices)")

    # Convert to TGAT format for tensor analysis
    print("\n🧠 Converting to TGAT Format...")
    X, edge_index, edge_times, metadata, edge_attr = graph_builder.to_tgat_format(graph_data)

    print(f"  X (node features): {X.shape}")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  edge_attr (edge features): {edge_attr.shape}")

    # Analyze scale edges from tensor data
    scale_edges = 0
    total_edges = edge_index.shape[1]

    # Get timeframes for each node from rich features
    timeframes = []
    timeframe_mapping = {0: "1m", 1: "5m", 2: "15m", 3: "1h", 4: "D", 5: "W"}

    for node_features in graph_data["rich_node_features"]:
        timeframe_id = node_features.timeframe_source
        timeframe = timeframe_mapping.get(timeframe_id, f"unknown_{timeframe_id}")
        timeframes.append(timeframe)

    # Count cross-timeframe edges
    for edge_idx in range(total_edges):
        source_idx = edge_index[0, edge_idx].item()
        target_idx = edge_index[1, edge_idx].item()

        if source_idx < len(timeframes) and target_idx < len(timeframes):
            source_tf = timeframes[source_idx]
            target_tf = timeframes[target_idx]

            if source_tf != target_tf and source_tf != "unknown" and target_tf != "unknown":
                scale_edges += 1

    print("\n⚖️ Scale Edge Analysis:")
    print(
        f"  Cross-timeframe edges: {scale_edges}/{total_edges} ({scale_edges/total_edges*100:.1f}%)"
    )

    # Initialize TGAT discovery
    print("\n🏛️ Running TGAT Archaeological Discovery...")
    tgat_discovery = IRONFORGEDiscovery()

    # Test learn_session with full tensor pipeline
    learn_result = tgat_discovery.learn_session(X, edge_index, edge_times, metadata, edge_attr)

    patterns = learn_result.get("patterns", [])
    pattern_types = {}

    for pattern in patterns:
        pattern_type = pattern.get("type", "unknown")
        pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1

    print(f"  🎯 Patterns discovered: {len(patterns)}")
    print("  📊 Pattern breakdown:")
    for pattern_type, count in pattern_types.items():
        print(f"    {pattern_type}: {count}")

    # Success criteria for complete pipeline
    success_criteria = [
        ("Multiple timeframes", len(timeframe_counts) > 1),
        ("Valid price relativity", sum(price_level_counts.values()) > 0),
        ("Non-zero scale edges", scale_edges > 0),
        ("37D node features", X.shape[1] == 37),
        ("17D edge features", edge_attr.shape[1] == 17),
        ("Pattern discovery", len(patterns) > 0),
        ("Scale edge percentage", scale_edges / total_edges > 0.1),  # >10% scale edges
    ]

    print("\n✅ Archaeological Pipeline Validation:")
    all_passed = True
    for criterion, passed in success_criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {criterion}")
        if not passed:
            all_passed = False

    # Final assessment
    if all_passed:
        print("\n🎉 COMPLETE ARCHAEOLOGICAL PIPELINE: SUCCESS!")
        print("✅ Phase 3 Step 3: Scale edge validation COMPLETE")
        print("✅ Level 1 → HTF → TGAT pipeline fully operational")
        print("✅ Cross-timeframe hierarchical relationships working")
        print("✅ 37D+17D sophisticated feature processing intact")
        print("✅ No fallbacks, no compromises - full capability achieved")
    else:
        print("\n⚠️ Archaeological Pipeline: PARTIAL SUCCESS")
        print("🔧 Some criteria not met, investigate remaining issues")

    return {
        "nodes": X.shape[0],
        "edges": total_edges,
        "scale_edges": scale_edges,
        "scale_edge_percentage": scale_edges / total_edges * 100,
        "timeframes": list(timeframe_counts.keys()),
        "patterns_discovered": len(patterns),
        "pattern_types": pattern_types,
        "all_criteria_passed": all_passed,
        "node_feature_dims": X.shape[1],
        "edge_feature_dims": edge_attr.shape[1],
    }


if __name__ == "__main__":
    try:
        result = test_complete_archaeological_pipeline()
        print("\n📊 Final Pipeline Results:")
        print(f"  🏗️ Graph: {result['nodes']} nodes, {result['edges']} edges")
        print(f"  ⚖️ Scale edges: {result['scale_edges']} ({result['scale_edge_percentage']:.1f}%)")
        print(f"  ⏰ Timeframes: {len(result['timeframes'])}")
        print(f"  🎯 Patterns: {result['patterns_discovered']}")
        print(
            f"  📊 Features: {result['node_feature_dims']}D nodes, {result['edge_feature_dims']}D edges"
        )
        print(f"  ✅ Success: {result['all_criteria_passed']}")

    except Exception as e:
        print(f"❌ Archaeological pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
