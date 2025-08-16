#!/usr/bin/env python3
"""
Test IRONFORGE orchestrator with 37D temporal cycle features
Innovation Architect validation
"""

import sys
import os
import json
from pathlib import Path
sys.path.append('/Users/jack/IRONPULSE/IRONFORGE')

from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

def test_orchestrator_37d():
    """Test that orchestrator works with 37D temporal cycle features"""
    
    print("🎯 Testing IRONFORGE Orchestrator with 37D Features")
    print("=" * 55)
    
    # Load one real session file for testing
    data_dir = Path("/Users/jack/IRONPULSE/data/sessions/htf_relativity")
    session_files = list(data_dir.glob("*.json"))[:3]  # Test with first 3 files
    
    if not session_files:
        print("❌ No session files found in htf_relativity directory")
        return False
    
    print(f"📁 Testing with {len(session_files)} session files")
    
    # Initialize components
    builder = EnhancedGraphBuilder()
    discovery = IRONFORGEDiscovery(node_features=37)  # 37D features
    
    total_patterns = 0
    
    for i, session_file in enumerate(session_files):
        print(f"\n🔍 Processing {session_file.name}...")
        
        try:
            # Load session data
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Build enhanced graph
            graph = builder.build_rich_graph(session_data, session_file_path=str(session_file))
            print(f"   ✅ Graph built: {graph['metadata']['total_nodes']} nodes")
            print(f"   ✅ Feature dimensions: {graph['metadata']['feature_dimensions']}")
            
            # Convert to TGAT format
            X, edge_index, edge_times, metadata, edge_attr = builder.to_tgat_format(graph)
            print(f"   ✅ TGAT format: {X.shape}")
            
            # Ensure 37D features
            if X.shape[1] != 37:
                print(f"   ❌ Expected 37D features, got {X.shape[1]}D")
                return False
            
            # Run pattern discovery
            if X.shape[0] >= 3:  # Need minimum nodes for pattern extraction
                result = discovery.learn_session(X, edge_index, edge_times, metadata, edge_attr)
                patterns = result['patterns']
                
                print(f"   ✅ Discovered {len(patterns)} patterns")
                
                # Count temporal cycle patterns specifically  
                cycle_patterns = [p for p in patterns if 'cycle' in p.get('type', '')]
                if cycle_patterns:
                    print(f"   🔄 Temporal cycle patterns: {len(cycle_patterns)}")
                    for pattern in cycle_patterns[:2]:  # Show first 2
                        print(f"      - {pattern['type']}: {pattern['description']}")
                
                total_patterns += len(patterns)
            else:
                print(f"   ⚠️ Skipping pattern discovery (only {X.shape[0]} nodes)")
                
        except Exception as e:
            print(f"   ❌ Error processing {session_file.name}: {e}")
            return False
    
    print(f"\n🏆 SUCCESS: Processed {len(session_files)} sessions")
    print(f"🔍 Total patterns discovered: {total_patterns}")
    print(f"✅ IRONFORGE orchestrator works with 37D temporal cycle features!")
    
    return True

if __name__ == "__main__":
    success = test_orchestrator_37d()
    if not success:
        sys.exit(1)