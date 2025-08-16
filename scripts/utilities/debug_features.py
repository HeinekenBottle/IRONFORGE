#!/usr/bin/env python3
"""
Debug Price Relativity Feature Dimensions
"""
import json
import torch
from learning.enhanced_graph_builder import EnhancedGraphBuilder

def debug_feature_dimensions():
    print("🔍 Debugging IRONFORGE Price Relativity Feature Dimensions")
    
    # Test with a relativity-enhanced session
    test_session = "/Users/jack/IRONPULSE/data/sessions/htf_relativity/NY_PM_Lvl-1_2025_08_04_htf_rel.json"
    
    # Load session data
    with open(test_session, 'r') as f:
        session_data = json.load(f)
    
    print(f"📁 Loaded: {test_session}")
    
    # Check if relativity features exist
    price_movements = session_data.get('price_movements', [])
    if price_movements:
        first_movement = price_movements[0]
        print(f"✅ Price movement relativity features:")
        for key in ['normalized_price', 'pct_from_open', 'pct_from_high', 'pct_from_low', 
                   'price_to_HTF_ratio', 'time_since_session_open', 'normalized_time']:
            if key in first_movement:
                print(f"   {key}: {first_movement[key]}")
            else:
                print(f"   ❌ MISSING: {key}")
    
    # Check pythonnodes for relativity features
    pythonnodes = session_data.get('pythonnodes', {})
    if pythonnodes:
        for tf_name, nodes in pythonnodes.items():
            if nodes:
                first_node = nodes[0]
                print(f"\n✅ {tf_name} node relativity features:")
                for key in ['normalized_price', 'pct_from_open', 'pct_from_high', 'pct_from_low',
                           'price_to_HTF_ratio', 'time_since_session_open', 'normalized_time']:
                    if key in first_node:
                        print(f"   {key}: {first_node[key]}")
                    else:
                        print(f"   ❌ MISSING: {key}")
                break
    
    # Build graph and check dimensions
    print(f"\n🏗️ Building graph with EnhancedGraphBuilder...")
    builder = EnhancedGraphBuilder()
    
    try:
        graph = builder.build_rich_graph(session_data, session_file_path=test_session)
        
        # Check rich node features
        rich_features = graph.get('rich_node_features', [])
        if rich_features:
            print(f"✅ Found {len(rich_features)} rich node features")
            
            # Convert first feature to tensor
            first_feature = rich_features[0]
            feature_tensor = first_feature.to_tensor()
            print(f"📐 Feature tensor shape: {feature_tensor.shape}")
            print(f"📐 Expected: torch.Size([34]) for price relativity")
            
            if feature_tensor.shape[0] == 34:
                print("✅ CORRECT: 34D features with price relativity!")
            elif feature_tensor.shape[0] == 27:
                print("❌ PROBLEM: Still using 27D features (missing relativity)")
            else:
                print(f"⚠️ UNEXPECTED: {feature_tensor.shape[0]}D features")
                
            # Show feature breakdown
            print(f"\n📊 Feature composition:")
            print(f"   Temporal (9): {feature_tensor[:9]}")
            print(f"   Price Relativity (7): {feature_tensor[9:16]}")
            print(f"   Legacy Price (3): {feature_tensor[16:19]}")
            print(f"   Market State (7): {feature_tensor[19:26]}")
            print(f"   Event Structure (8): {feature_tensor[26:34]}")
        else:
            print("❌ No rich node features found")
            
    except Exception as e:
        print(f"❌ Error building graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_dimensions()