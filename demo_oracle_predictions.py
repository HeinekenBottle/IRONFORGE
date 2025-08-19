#!/usr/bin/env python3
"""
Demo script for Oracle Temporal Non-locality predictions
Shows how the oracle predicts session ranges from early events
"""

import sys
import torch
import networkx as nx
from pathlib import Path

# Add IRONFORGE to path
sys.path.insert(0, str(Path(__file__).parent))

from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
from ironforge.learning.enhanced_graph_builder import RichNodeFeature, RichEdgeFeature
from ironforge.sdk.app_config import OracleCfg


def create_demo_session_graph(n_events=24, session_range=(23000, 23200)):
    """Create a demo session graph simulating real market events"""
    print(f"📊 Creating demo session with {n_events} events...")
    print(f"📈 Simulated session range: {session_range[0]} - {session_range[1]}")
    
    graph = nx.Graph()
    session_low, session_high = session_range
    session_center = (session_high + session_low) / 2
    session_half_range = (session_high - session_low) / 2
    
    # Simulate price action throughout session
    prices = []
    for i in range(n_events):
        # Create realistic price progression
        progress = i / (n_events - 1)  # 0 to 1
        
        # Early events cluster around eventual center with some noise
        if progress < 0.3:  # Early events (first 30%)
            price = session_center + torch.randn(1).item() * (session_half_range * 0.3)
        else:
            # Later events explore the full range
            price = session_low + progress * (session_high - session_low) + torch.randn(1).item() * 10
        
        prices.append(price)
    
    # Build graph with 45D node features
    for i, price in enumerate(prices):
        node_feature = RichNodeFeature()
        
        # Set semantic event flags based on price action
        if i > 0 and abs(price - prices[i-1]) > 20:
            node_feature.set_semantic_event("fvg_redelivery_flag", 1.0)
            
        if i > 2 and price > max(prices[i-3:i]):
            node_feature.set_semantic_event("expansion_phase_flag", 1.0)
            
        if i == 0 or price == min(prices[:i+1]):
            node_feature.set_semantic_event("reversal_flag", 0.8)
            
        # Traditional features include price-based metrics
        traditional_features = torch.tensor([
            price / 23000.0,  # Normalized price
            (price - session_center) / session_half_range,  # Distance from center
            i / n_events,  # Time progression
            *torch.randn(34).tolist()  # Other technical indicators
        ], dtype=torch.float32)
        
        node_feature.set_traditional_features(traditional_features)
        graph.add_node(i, feature=node_feature.features)
    
    # Add temporal edges
    for i in range(n_events - 1):
        edge_feature = RichEdgeFeature()
        price_diff = abs(prices[i+1] - prices[i])
        
        edge_feature.set_semantic_relationship("semantic_event_link", min(price_diff / 50.0, 1.0))
        edge_feature.features[3:] = torch.randn(17) * 0.1
        
        graph.add_edge(i, i+1, 
                      feature=edge_feature.features,
                      temporal_distance=1.0)
    
    return graph, (session_low, session_high)


def demonstrate_oracle_predictions():
    """Demonstrate oracle prediction capabilities"""
    print("🔮 Oracle Temporal Non-locality Demonstration")
    print("=" * 70)
    
    # Initialize oracle system
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()
    
    oracle_cfg = OracleCfg()
    print(f"⚙️  Oracle Configuration:")
    print(f"   Enabled: {oracle_cfg.enabled}")
    print(f"   Early percentage: {oracle_cfg.early_pct}")
    print()
    
    # Create demo session
    graph, true_range = create_demo_session_graph(n_events=20)
    true_low, true_high = true_range
    
    print(f"🎯 TRUE SESSION RANGE (for comparison):")
    print(f"   Low: {true_low:.2f}")
    print(f"   High: {true_high:.2f}")
    print(f"   Center: {(true_low + true_high) / 2:.2f}")
    print(f"   Half-range: {(true_high - true_low) / 2:.2f}")
    print()
    
    # Test different early percentages
    early_percentages = [0.10, 0.20, 0.30, 0.50]
    
    print("🔮 ORACLE PREDICTIONS:")
    print("-" * 70)
    
    for early_pct in early_percentages:
        with torch.no_grad():
            oracle_result = discovery_engine.predict_session_range(
                graph, early_batch_pct=early_pct
            )
        
        pred_low = oracle_result["pred_low"]
        pred_high = oracle_result["pred_high"]
        confidence = oracle_result["confidence"]
        n_events = oracle_result["n_events"]
        
        # Calculate accuracy metrics
        range_error = abs((pred_high - pred_low) - (true_high - true_low))
        center_error = abs(oracle_result["center"] - (true_high + true_low) / 2)
        
        print(f"📊 Using {early_pct:.0%} of events ({n_events} events):")
        print(f"   Predicted: {pred_low:.2f} - {pred_high:.2f}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Range error: {range_error:.2f} points")
        print(f"   Center error: {center_error:.2f} points")
        print(f"   Notes: {oracle_result['notes']}")
        print()
    
    print("=" * 70)
    print("✨ Temporal Non-locality Theory:")
    print("   Early events contain forward-looking information about")
    print("   the eventual session range through attention mechanisms.")
    print()
    print("💡 Key Insights:")
    print("   - Oracle uses attention-weighted pooling of early embeddings")
    print("   - Regression head maps 44D embeddings to (center, half_range)")
    print("   - Confidence reflects attention concentration")
    print("   - Performance improves with more early events")


if __name__ == "__main__":
    try:
        demonstrate_oracle_predictions()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()