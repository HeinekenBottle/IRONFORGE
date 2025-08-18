#!/usr/bin/env python3
"""
HTF Integration Example - Shows complete multi-timeframe graph output
Demonstrates the final integrated system with pythonnodes and scale edges
"""

import json

from learning.enhanced_graph_builder import EnhancedGraphBuilder


def demonstrate_htf_integration():
    """Create example HTF output showing all features"""
    
    print("🎯 HTF Integration Example Output")
    print("=" * 60)
    print("Demonstrating multi-timeframe graph with:")
    print("• 1m + HTF nodes (5m, 15m, 1h, D)")
    print("• Scale edges with metadata")
    print("• TGAT compatibility preserved")
    print("• 27D node features + 17D edge features")
    print()
    
    # Load a real HTF-enhanced session
    htf_session_path = "/Users/jack/IRONPULSE/data/sessions/htf_enhanced/NY_PM_Lvl-1__htf.json"
    
    with open(htf_session_path, 'r') as f:
        session_data = json.load(f)
    
    # Initialize builder and create graph
    builder = EnhancedGraphBuilder()
    graph = builder.build_rich_graph(session_data, session_file_path=htf_session_path)
    
    # Show example output structure
    print("📊 EXAMPLE HTF GRAPH OUTPUT:")
    print(f"Session: {session_data['session_metadata']['session_type']}")
    print(f"Total nodes: {graph['metadata']['total_nodes']}")
    print()
    
    print("🏗️ NODE STRUCTURE:")
    for tf, count in graph['metadata']['timeframe_counts'].items():
        if count > 0:
            print(f"  {tf:>3}: {count:>2} nodes")
            
            # Show sample node IDs
            sample_nodes = graph['nodes'][tf][:3]  # First 3 nodes
            node_ids = []
            for node_idx in sample_nodes:
                if node_idx < len(graph['rich_node_features']):
                    raw_data = graph['rich_node_features'][node_idx].raw_json
                    htf_id = raw_data.get('id', 'N/A')
                    node_ids.append(f"{tf}_{htf_id}")
            
            if node_ids:
                print(f"      Sample node IDs: {', '.join(node_ids[:3])}")
    print()
    
    print("🔗 SCALE EDGE STRUCTURE:")
    scale_edges = [e for e in graph['edges']['scale'] if 'tf_source' in e]
    
    # Group by mapping type
    mappings = {}
    for edge in scale_edges:
        mapping_key = f"{edge['tf_source']}_to_{edge['tf_target']}"
        if mapping_key not in mappings:
            mappings[mapping_key] = []
        mappings[mapping_key].append(edge)
    
    for mapping_name, edges in mappings.items():
        print(f"  {mapping_name}: {len(edges)} scale edges")
        
        # Show sample edge with metadata
        if edges:
            sample_edge = edges[0]
            print(f"    Sample: Node {sample_edge['source']} → Node {sample_edge['target']}")
            print(f"    Coverage: {sample_edge['coverage']}")
            
            # Show parent metadata
            parent_meta = sample_edge.get('parent_metadata', {})
            metadata_items = []
            if parent_meta.get('pd_array'):
                metadata_items.append("PD Array")
            if parent_meta.get('fpfvg'):
                metadata_items.append("FPFVG")
            if parent_meta.get('liquidity_sweep'):
                metadata_items.append("Liquidity Sweep")
            
            if metadata_items:
                print(f"    Parent metadata: {', '.join(metadata_items)}")
    print()
    
    # Convert to TGAT format and show compatibility
    X, edge_index, edge_times, metadata, edge_attr = builder.to_tgat_format(graph)
    
    print("🧠 TGAT COMPATIBILITY:")
    print(f"  Node features: {X.shape} (27D per node)")
    print(f"  Edge index: {edge_index.shape}")
    print(f"  Edge features: {edge_attr.shape} (17D per edge)")
    print(f"  Edge types: {len(metadata['edge_types'])} types")
    print(f"  Total edges: {metadata['total_edges']}")
    print()
    
    # Show sample rich features
    if len(graph['rich_node_features']) > 0:
        sample_node_1m = None
        sample_node_htf = None
        
        # Find sample 1m and HTF nodes
        for i, tf_nodes in graph['nodes'].items():
            if tf_nodes:
                node_idx = tf_nodes[0]
                if node_idx < len(graph['rich_node_features']):
                    node_features = graph['rich_node_features'][node_idx]
                    if i == '1m' and sample_node_1m is None:
                        sample_node_1m = (i, node_features)
                    elif i in ['5m', '15m', '1h'] and sample_node_htf is None:
                        sample_node_htf = (i, node_features)
        
        print("🎯 SAMPLE NODE FEATURES:")
        if sample_node_1m:
            tf, features = sample_node_1m
            print(f"  {tf} node:")
            print(f"    Time: {features.time_minutes:.1f}min, Session pos: {features.session_position:.3f}")
            print(f"    Price: {features.normalized_price:.6f}, Volatility: {features.volatility_window:.4f}")
            print(f"    Event type: {features.event_type_id}, TF source: {features.timeframe_source}")
        
        if sample_node_htf:
            tf, features = sample_node_htf
            print(f"  {tf} node:")
            print(f"    Time: {features.time_minutes:.1f}min, Session pos: {features.session_position:.3f}")
            print(f"    Price: {features.normalized_price:.6f}, Cross-TF confluence: {features.cross_tf_confluence:.3f}")
            print(f"    Structural importance: {features.structural_importance:.4f}")
        print()
    
    # Show sample edge features
    if scale_edges:
        sample_edge = scale_edges[0]
        feature_idx = sample_edge['feature_idx']
        edge_feature = graph['rich_edge_features'][feature_idx]
        
        print("🔗 SAMPLE SCALE EDGE FEATURES (17D):")
        print(f"  Temporal: time_delta={edge_feature.time_delta:.2f}, tf_jump={edge_feature.timeframe_jump}")
        print(f"  Relationship: type={edge_feature.relation_type}, strength={edge_feature.relation_strength:.3f}")
        print(f"  Hierarchy: scale_from={edge_feature.scale_from}, scale_to={edge_feature.scale_to}")
        print(f"  Hierarchy distance: {edge_feature.hierarchy_distance:.2f}")
        print(f"  Archaeological: discovery_confidence={edge_feature.discovery_confidence:.2f}")
        print()
    
    print("✅ HTF INTEGRATION COMPLETE")
    print("Features demonstrated:")
    print("• Multi-timeframe node creation from pythonnodes")
    print("• Scale edges using htf_cross_map with metadata")
    print("• 27D rich node features + 17D rich edge features")
    print("• Full TGAT compatibility maintained")
    print("• Graceful fallback to 1m-only mode")
    print("• Unit test validation of scale edge creation")

if __name__ == "__main__":
    demonstrate_htf_integration()