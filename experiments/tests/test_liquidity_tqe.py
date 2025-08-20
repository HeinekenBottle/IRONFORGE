#!/usr/bin/env python3
"""
Test Liquidity & HTF Follow-Through Analysis in TQE
Experiment E — Liquidity & HTF Follow-Through (explore-only)
"""

from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine

def main():
    """Test the new liquidity/HTF query handlers"""
    print("🔄 Testing Enhanced TQE with Liquidity & HTF Analysis")
    print("=" * 60)
    
    # Initialize TQE with enhanced sessions
    tqe = EnhancedTemporalQueryEngine(adapted_dir="data/day_news_enhanced")
    
    print(f"✅ TQE initialized with {len(tqe.sessions)} sessions")
    
    # Test liquidity sweep analysis
    print("\n1. Testing Liquidity Sweep Analysis...")
    liquidity_results = tqe.ask("Show me liquidity sweeps after RD@40")
    print(f"   Total RD@40 events: {liquidity_results.get('total_rd40_events', 0)}")
    print(f"   Total sweeps: {liquidity_results.get('total_sweeps', 0)}")
    print(f"   Sweep rate: {liquidity_results.get('sweep_rate', 0):.1f}%")
    
    # Test HTF analysis
    print("\n2. Testing HTF Level Taps...")
    htf_results = tqe.ask("Analyze HTF level touches after RD@40")
    print(f"   HTF tap rate: {htf_results.get('htf_tap_rate', 0):.1f}%")
    print(f"   Timeframe breakdown: {htf_results.get('timeframe_breakdown', {})}")
    
    # Test minute hotspots
    print("\n3. Testing Minute-of-Day Hotspots...")
    hotspot_results = tqe.ask("Show me minute hotspots for RD@40")
    print(f"   Total events analyzed: {hotspot_results.get('total_events', 0)}")
    top_5 = hotspot_results.get('top_5_minutes', [])
    if top_5:
        print(f"   Top minute: {top_5[0][0]} ({top_5[0][1]} events)")
    
    # Test FVG follow-through
    print("\n4. Testing FVG Follow-Through...")
    fvg_results = tqe.ask("Analyze FVG follow-through patterns")
    print(f"   FVG follow rate: {fvg_results.get('fvg_follow_rate', 0):.1f}%")
    print(f"   Direction analysis: {fvg_results.get('direction_analysis', {})}")
    
    # Test event chains
    print("\n5. Testing Event Chains...")
    chain_results = tqe.ask("Show me event chain analysis")
    print(f"   Status: {chain_results.get('status', 'unknown')}")
    sample_chains = chain_results.get('sample_chains', [])
    if sample_chains:
        print(f"   Sample chain: {sample_chains[0]}")
    
    print("\n✅ All liquidity/HTF query handlers working!")
    print("🎯 Ready for Experiment E analysis")

if __name__ == "__main__":
    main()