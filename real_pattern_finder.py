#!/usr/bin/env python3
"""
Real Pattern Finder - Finds ACTUAL cross-timeframe patterns in IRONFORGE graphs
No embeddings, no similarities, just real market relationships
"""

import glob
import json
import pickle
import os
import re

def extract_price_from_node_feature(feature_str):
    """Extract price from RichNodeFeature string representation"""
    # Look for raw_json price_level
    match = re.search(r"'price_level': ([0-9.]+)", feature_str)
    if match:
        return float(match.group(1))
    return 0.0

def find_real_patterns():
    """
    Find ONE specific real pattern across all sessions:
    - 1m node near 23,000 level  
    - Has scale edge to 15m/1h/5m node
    - Parent node has PD Array or other structure
    """
    patterns = []
    
    # Load all session graph pickle files
    graph_files = glob.glob('/Users/jack/IRONPULSE/IRONFORGE/preservation/full_graph_store/*.pkl')
    
    for graph_file in graph_files:
        session_name = os.path.basename(graph_file).replace('.pkl', '')
        
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Get scale edges
            scale_edges = graph_data.get('edges', {}).get('scale', [])
            
            # Process each scale edge
            for edge in scale_edges:
                if edge.get('tf_source') == '1m':  # 1m source node
                    source_idx = edge.get('source', 0)
                    target_idx = edge.get('target', 0)
                    
                    # Get node features
                    rich_features = graph_data.get('rich_node_features', [])
                    if source_idx < len(rich_features):
                        source_feature = rich_features[source_idx]
                        price_1m = extract_price_from_node_feature(str(source_feature))
                        
                        # Check if near significant levels (23,000 area - broader range)
                        if 22800 <= price_1m <= 23500:
                            # Check for parent metadata (PD Array, FPFVG, liquidity)
                            parent_metadata = edge.get('parent_metadata', {})
                            
                            if (parent_metadata.get('pd_array') or 
                                parent_metadata.get('fpfvg') or
                                parent_metadata.get('liquidity_sweep') or
                                edge.get('coverage', 0) > 1):  # Multi-timeframe coverage
                                
                                # Extract timestamp if available
                                timestamp_match = re.search(r"'timestamp': '([^']+)'", str(source_feature))
                                timestamp = timestamp_match.group(1) if timestamp_match else 'unknown'
                                
                                pattern = {
                                    'session': session_name.split('_graph_')[0],  # Clean session name
                                    '1m_node_idx': source_idx,
                                    'target_node_idx': target_idx,
                                    'tf_target': edge.get('tf_target', 'unknown'),
                                    'price': price_1m,
                                    'timestamp': timestamp,
                                    'coverage': edge.get('coverage', 1),
                                    'parent_metadata': parent_metadata
                                }
                                patterns.append(pattern)
                                print(f"✓ Found real pattern in {pattern['session']}: {price_1m:.0f} @ {timestamp} -> {edge.get('tf_target')}")
        
        except Exception as e:
            print(f"Error processing {graph_file}: {e}")
            continue
    
    # Save real patterns found from actual graph data
    output = {
        'total_patterns': len(patterns),
        'patterns': patterns,
        'sessions_analyzed': len(graph_files),
        'status': 'OPERATIONAL',
        'note': 'Real cross-timeframe patterns found from properly loaded graph data'
    }
    
    with open('/Users/jack/IRONPULSE/IRONFORGE/real_patterns.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n🎯 OPERATIONAL RESULTS:")
    print(f"Sessions processed: {len(graph_files)}")
    print(f"Real patterns found: {len(patterns)}")
    print(f"Status: OPERATIONAL - Loading actual graph data")
    
    if patterns:
        print(f"\n📋 REAL CROSS-TIMEFRAME PATTERNS:")
        for i, pattern in enumerate(patterns):
            print(f"  {i+1}. {pattern['session']}: {pattern['price']:.0f} @ {pattern['timestamp']} -> {pattern['tf_target']} (coverage: {pattern['coverage']})")
    else:
        print("\n📋 No cross-timeframe patterns found in current data")
    
    return patterns

if __name__ == "__main__":
    find_real_patterns()