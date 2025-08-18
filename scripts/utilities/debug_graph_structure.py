#!/usr/bin/env python3
"""
Debug Graph Structure
====================
Debug the actual graph structure returned by enhanced graph builder.
"""

import json
from pathlib import Path

from learning.enhanced_graph_builder import EnhancedGraphBuilder


def debug_graph_structure():
    """Debug the actual graph structure."""
    print("🔬 IRONFORGE Graph Structure Debug")
    print("=" * 50)
    
    # Load a price-relativity enhanced HTF session
    htf_file = "/Users/jack/IRONPULSE/data/sessions/htf_relativity/NY_AM_Lvl-1_2025_07_30_htf_regenerated_rel.json"
    print(f"📊 Testing with: {Path(htf_file).name}")
    
    with open(htf_file, 'r') as f:
        htf_data = json.load(f)
    
    # Initialize enhanced graph builder
    print("🔧 Initializing Enhanced Graph Builder...")
    graph_builder = EnhancedGraphBuilder()
    
    # Build enhanced graph with HTF data
    print("🏗️ Building enhanced graph...")
    graph_data = graph_builder.build_rich_graph(htf_data)
    
    print("\n📊 Graph Data Structure:")
    print(f"  Type: {type(graph_data)}")
    print(f"  Keys: {list(graph_data.keys()) if isinstance(graph_data, dict) else 'Not a dict'}")
    
    if 'nodes' in graph_data:
        nodes = graph_data['nodes']
        print("\n📍 Nodes:")
        print(f"  Type: {type(nodes)}")
        print(f"  Length: {len(nodes) if hasattr(nodes, '__len__') else 'No length'}")
        
        # Try to get first few nodes safely
        try:
            if hasattr(nodes, '__iter__') and not isinstance(nodes, str):
                node_list = list(nodes)[:3] if hasattr(nodes, '__iter__') else []
            else:
                node_list = []
        except:
            node_list = []
            
        for i, node in enumerate(node_list):
            print(f"  Node {i}:")
            print(f"    Type: {type(node)}")
            if isinstance(node, dict):
                print(f"    Keys: {list(node.keys())}")
                for key, value in list(node.items())[:5]:  # First 5 keys
                    print(f"    {key}: {type(value)} = {str(value)[:50]}...")
            elif hasattr(node, '__dict__'):
                print(f"    Attributes: {list(node.__dict__.keys())}")
            print()
    
    if 'edges' in graph_data:
        edges = graph_data['edges']
        print("\n🔗 Edges:")
        print(f"  Type: {type(edges)}")
        print(f"  Length: {len(edges) if hasattr(edges, '__len__') else 'No length'}")
        
        # Try to get first few edges safely
        try:
            if hasattr(edges, '__iter__') and not isinstance(edges, str):
                edge_list = list(edges)[:3] if hasattr(edges, '__iter__') else []
            else:
                edge_list = []
        except:
            edge_list = []
            
        for i, edge in enumerate(edge_list):
            print(f"  Edge {i}:")
            print(f"    Type: {type(edge)}")
            if isinstance(edge, dict):
                print(f"    Keys: {list(edge.keys())}")
                for key, value in list(edge.items())[:5]:  # First 5 keys
                    print(f"    {key}: {type(value)} = {str(value)[:50]}...")
            elif hasattr(edge, '__dict__'):
                print(f"    Attributes: {list(edge.__dict__.keys())}")
            print()
    
    # Test TGAT conversion
    print("\n🧠 Testing TGAT Conversion...")
    try:
        X, edge_index, edge_times, metadata, edge_attr = graph_builder.to_tgat_format(graph_data)
        print("  ✅ TGAT conversion successful:")
        print(f"    X (node features): {X.shape}")
        print(f"    edge_index: {edge_index.shape}")
        print(f"    edge_attr (edge features): {edge_attr.shape}")
    except Exception as e:
        print(f"  ❌ TGAT conversion failed: {e}")

if __name__ == "__main__":
    try:
        debug_graph_structure()
        
    except Exception as e:
        print(f"❌ Graph structure debug failed: {e}")
        import traceback
        traceback.print_exc()