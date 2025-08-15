#!/usr/bin/env python3
"""
Debug TGAT Initialization for Price Relativity
"""
import torch
from learning.tgat_discovery import IRONFORGEDiscovery

def debug_tgat_initialization():
    print("🔍 Debugging TGAT Initialization")
    
    # Test with default initialization (should be 34 now)
    print("🧪 Testing default IRONFORGEDiscovery initialization...")
    discovery_engine = IRONFORGEDiscovery()
    
    print(f"📐 TGAT model input dimensions:")
    print(f"   in_channels: {discovery_engine.model.in_channels}")
    print(f"   attention_dim: {discovery_engine.model.attention_dim}")
    print(f"   input_projection weight shape: {discovery_engine.model.input_projection.weight.shape}")
    
    # Test with 34D sample data
    print(f"\n🧪 Testing 34D sample data...")
    sample_features = torch.randn(5, 34)  # 5 nodes, 34 features each
    sample_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    sample_edge_times = torch.randn(4, 1)
    
    try:
        print(f"✅ Input shape: {sample_features.shape}")
        result = discovery_engine.model(sample_features, sample_edge_index, sample_edge_times)
        print(f"✅ Output shape: {result.shape}")
        print("✅ SUCCESS: TGAT accepts 34D price relativity features!")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("❌ TGAT initialization issue with 34D features")
        
        # Check the exact dimensions
        print(f"\n📊 Dimension analysis:")
        print(f"   Model expects: {discovery_engine.model.in_channels}D")
        print(f"   Data provides: 34D")
        print(f"   Projection layer: {discovery_engine.model.input_projection}")

if __name__ == "__main__":
    debug_tgat_initialization()