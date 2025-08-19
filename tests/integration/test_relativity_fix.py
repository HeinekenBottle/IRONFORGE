#!/usr/bin/env python3
"""
Test Price Relativity Fix
"""
import json

from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery


def test_relativity_fix():
    print("🧪 Testing Price Relativity Fix")

    # Test session
    test_session = (
        "/Users/jack/IRONPULSE/data/sessions/htf_relativity/NY_PM_Lvl-1_2025_08_04_htf_rel.json"
    )

    with open(test_session) as f:
        session_data = json.load(f)

    # Build graph
    builder = EnhancedGraphBuilder()
    graph = builder.build_rich_graph(session_data, session_file_path=test_session)

    # Convert to TGAT format
    X, edge_index, edge_times, metadata, edge_attr = builder.to_tgat_format(graph)

    print(f"✅ X shape: {X.shape}")
    print(f"✅ Edge index shape: {edge_index.shape}")
    print(f"✅ Edge times shape: {edge_times.shape}")
    print(f"✅ Edge attr shape: {edge_attr.shape}")

    # Test TGAT
    discovery_engine = IRONFORGEDiscovery()

    try:
        result = discovery_engine.model(X, edge_index, edge_times, edge_attr)
        print(f"✅ TGAT output shape: {result.shape}")
        print("🎯 SUCCESS: Price relativity features working!")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    test_relativity_fix()
