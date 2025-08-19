#!/usr/bin/env python3
"""
Graph Builder Diagnostic
========================
Debug the graph building process to understand 'node_features' error.
"""

import json
import sys
from pathlib import Path

# Add IRONFORGE to path
ironforge_root = Path(__file__).parent
sys.path.append(str(ironforge_root))

from learning.enhanced_graph_builder import EnhancedGraphBuilder


def diagnose_graph_building():
    """Diagnose what's happening in graph building process."""
    print("🔍 GRAPH BUILDER DIAGNOSTIC")
    print("=" * 50)
    
    # Initialize graph builder
    try:
        builder = EnhancedGraphBuilder()
        print("✅ Graph builder initialized successfully")
    except Exception as e:
        print(f"❌ Graph builder initialization failed: {str(e)}")
        return
    
    # Load a test session
    test_session = ironforge_root / "enhanced_sessions_with_relativity" / "enhanced_rel_NY_PM_Lvl-1_2025_07_29.json"
    
    if not test_session.exists():
        print(f"❌ Test session file not found: {test_session}")
        return
    
    try:
        with open(test_session) as f:
            session_data = json.load(f)
        print(f"✅ Session data loaded: {len(session_data.get('price_movements', []))} movements")
    except Exception as e:
        print(f"❌ Failed to load session data: {str(e)}")
        return
    
    # Test graph building
    print("\n🔧 Testing graph building process...")
    try:
        graph = builder.build_rich_graph(session_data)
        print("✅ Graph built successfully")
        print(f"   Graph type: {type(graph)}")
        print(f"   Graph attributes: {dir(graph) if hasattr(graph, '__dict__') else 'No __dict__'}")
        
        # Check what's in the graph
        if hasattr(graph, 'node_features'):
            print(f"   ✅ node_features present: shape {graph.node_features.shape if hasattr(graph.node_features, 'shape') else type(graph.node_features)}")
        else:
            print("   ❌ node_features missing!")
            print(f"   Available attributes: {list(vars(graph).keys()) if hasattr(graph, '__dict__') else 'No attributes'}")
        
        if hasattr(graph, 'edge_features'):
            print(f"   ✅ edge_features present: shape {graph.edge_features.shape if hasattr(graph.edge_features, 'shape') else type(graph.edge_features)}")
        else:
            print("   ❌ edge_features missing!")
            
        if hasattr(graph, 'num_nodes'):
            print(f"   Nodes: {graph.num_nodes}")
        if hasattr(graph, 'num_edges'):
            print(f"   Edges: {graph.num_edges}")
            
    except Exception as e:
        print(f"❌ Graph building failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Test if we can create a minimal working graph
    print("\n🧪 Testing minimal graph creation...")
    try:
        # Create minimal session data
        minimal_session = {
            'price_movements': [
                {
                    'timestamp': '13:30:00',
                    'price_level': 23506.0,
                    'movement_type': 'open',
                    'normalized_price': 0.5,
                    'pct_from_open': 0.0,
                    'range_position': 0.5
                },
                {
                    'timestamp': '14:00:00', 
                    'price_level': 23520.0,
                    'movement_type': 'regular',
                    'normalized_price': 0.6,
                    'pct_from_open': 0.1,
                    'range_position': 0.6
                }
            ],
            'session_metadata': {'session_type': 'test'}
        }
        
        minimal_graph = builder.build_rich_graph(minimal_session)
        print("✅ Minimal graph built successfully")
        print(f"   Minimal graph type: {type(minimal_graph)}")
        
        if hasattr(minimal_graph, 'node_features'):
            print(f"   ✅ Minimal node_features present: {minimal_graph.node_features.shape if hasattr(minimal_graph.node_features, 'shape') else type(minimal_graph.node_features)}")
        else:
            print("   ❌ Minimal node_features missing!")
            
    except Exception as e:
        print(f"❌ Minimal graph building failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose_graph_building()