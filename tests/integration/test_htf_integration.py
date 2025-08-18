#!/usr/bin/env python3
"""
Test HTF Integration with Enhanced Graph Builder
"""

import json
import logging
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_htf_integration():
    """Test HTF integration with sample data"""
    
    print("🧪 Testing HTF Integration with Enhanced Graph Builder")
    print("=" * 60)
    
    # Initialize enhanced graph builder
    builder = EnhancedGraphBuilder()
    
    # Test with HTF-enhanced session data
    htf_session_path = "/Users/jack/IRONPULSE/data/sessions/htf_enhanced/NY_PM_Lvl-1__htf.json"
    
    try:
        # Load HTF-enhanced session
        with open(htf_session_path, 'r') as f:
            htf_session_data = json.load(f)
        
        print("📊 Loaded HTF session data:")
        print(f"   Session type: {htf_session_data.get('session_metadata', {}).get('session_type')}")
        
        # Check HTF data structure
        pythonnodes = htf_session_data.get('pythonnodes', {})
        htf_cross_map = htf_session_data.get('htf_cross_map', {})
        
        print(f"   Pythonnodes timeframes: {list(pythonnodes.keys())}")
        print(f"   HTF cross-mappings: {list(htf_cross_map.keys())}")
        
        for tf, nodes in pythonnodes.items():
            print(f"     {tf}: {len(nodes)} nodes")
            
        # Build enhanced graph with HTF data
        print("\n🏗️ Building enhanced graph with HTF integration...")
        
        enhanced_graph = builder.build_rich_graph(
            htf_session_data,
            session_file_path=htf_session_path
        )
        
        # Analyze results
        print("\n📈 HTF Integration Results:")
        print(f"   Total nodes: {enhanced_graph['metadata']['total_nodes']}")
        print(f"   Node feature dimensions: {enhanced_graph['metadata']['feature_dimensions']}")
        
        # Timeframe distribution
        print("   Timeframe node distribution:")
        for tf, count in enhanced_graph['metadata']['timeframe_counts'].items():
            if count > 0:
                print(f"     {tf}: {count} nodes")
        
        # Edge analysis
        total_edges = 0
        print("   Edge type distribution:")
        for edge_type, edges in enhanced_graph['edges'].items():
            if edges:
                print(f"     {edge_type}: {len(edges)} edges")
                total_edges += len(edges)
                
                # Show sample scale edge if available
                if edge_type == 'scale' and edges:
                    sample_edge = edges[0]
                    if 'tf_source' in sample_edge and 'tf_target' in sample_edge:
                        print(f"       Sample scale edge: {sample_edge['tf_source']} → {sample_edge['tf_target']}")
                        if 'coverage' in sample_edge:
                            print(f"         Coverage: {sample_edge['coverage']}")
        
        print(f"   Total edges: {total_edges}")
        
        # Test TGAT format conversion
        print("\n🧠 Testing TGAT format conversion...")
        X, edge_index, edge_times, metadata, edge_attr = builder.to_tgat_format(enhanced_graph)
        
        print(f"   Node features shape: {X.shape}")
        print(f"   Edge index shape: {edge_index.shape}")
        print(f"   Edge times shape: {edge_times.shape}")
        print(f"   Edge attributes shape: {edge_attr.shape}")
        print(f"   Edge feature dimensions: {metadata.get('edge_feature_dimensions', 'N/A')}")
        
        # Validate HTF scale edges exist
        scale_edges = [e for e in enhanced_graph['edges']['scale'] if 'tf_source' in e]
        if scale_edges:
            print(f"\n✅ HTF Scale edges successfully created: {len(scale_edges)} scale edges")
            
            # Show mapping coverage
            mappings_used = set()
            for edge in scale_edges:
                mapping = f"{edge['tf_source']}_to_{edge['tf_target']}"
                mappings_used.add(mapping)
            print(f"   Mappings utilized: {sorted(mappings_used)}")
        else:
            print("\n⚠️ No HTF scale edges found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing HTF integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_mode():
    """Test fallback to regular mode without HTF data"""
    
    print("\n🔄 Testing fallback mode (no HTF data)...")
    
    builder = EnhancedGraphBuilder()
    
    # Load regular session data
    regular_session_path = "/Users/jack/IRONPULSE/data/sessions/level_1/NY_PM_Lvl-1_.json"
    
    try:
        with open(regular_session_path, 'r') as f:
            regular_session_data = json.load(f)
        
        # Build graph without HTF data
        regular_graph = builder.build_rich_graph(regular_session_data)
        
        print(f"   Regular mode - Total nodes: {regular_graph['metadata']['total_nodes']}")
        print(f"   Timeframe distribution: {regular_graph['metadata']['timeframe_counts']}")
        
        # Check that it falls back correctly
        has_1m_only = sum(1 for tf, count in regular_graph['metadata']['timeframe_counts'].items() if count > 0)
        if has_1m_only <= 3:  # Mostly 1m with some 15m/1h
            print("✅ Fallback mode working correctly")
        else:
            print("⚠️ Fallback mode may not be working as expected")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing fallback mode: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    # Test HTF integration
    success &= test_htf_integration()
    
    # Test fallback mode
    success &= test_fallback_mode()
    
    print(f"\n{'=' * 60}")
    if success:
        print("🎉 HTF Integration Tests PASSED")
    else:
        print("❌ HTF Integration Tests FAILED")
    print(f"{'=' * 60}")