#!/usr/bin/env python3
"""
Demo: Advanced Experiment Set E1/E2/E3 Integration
Demonstrates sophisticated post-RD@40% path analysis with statistical rigor
"""

import sys
import traceback
from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine

def demo_e1_cont_analysis():
    """Demonstrate E1 CONT path analysis: RD@40 → 60% → 80% timing"""
    print("\n🎯 E1 CONT PATH ANALYSIS")
    print("=" * 60)
    print("📋 Definition: RD@40 → 60% within ≤45m, and 80% within ≤90m")
    print("🎲 Expected: f8_q ≥ P90, positive slope, TheoryB_Δt ≤ 30m, gap_age ≤ 2")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test E1-specific queries
        test_queries = [
            "Analyze E1 CONT paths with 60% and 80% progression timing",
            "Show me CONT patterns that reach 60% within 45 minutes",
            "What are the timing statistics for E1 continuation paths?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            result = engine.ask(query)
            print_e_analysis_results(result, result_type="E1_CONT")
            
    except Exception as e:
        print(f"❌ E1 Analysis Error: {e}")
        traceback.print_exc()

def demo_e2_mr_analysis():
    """Demonstrate E2 MR path analysis: RD@40 → mid with branching"""
    print("\n🔄 E2 MR PATH ANALYSIS")
    print("=" * 60)
    print("📋 Definition: RD@40 → mid (50-60%) within ≤60m; branches: second_rd or failure")
    print("🎲 Expected: f50 = mean-revert OR high news_impact, f8_slope ≤ 0, no H1 breakout")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test E2-specific queries
        test_queries = [
            "Analyze E2 MR paths with second RD and failure branching",
            "Show me mean revert patterns after RD@40 events", 
            "What are the branch probabilities for E2 MR paths?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            result = engine.ask(query)
            print_e_analysis_results(result, result_type="E2_MR")
            
    except Exception as e:
        print(f"❌ E2 Analysis Error: {e}")
        traceback.print_exc()

def demo_e3_accel_analysis():
    """Demonstrate E3 ACCEL path analysis: RD@40 + H1 breakout → fast 80%"""
    print("\n🚀 E3 ACCEL PATH ANALYSIS")
    print("=" * 60)
    print("📋 Definition: RD@40 + H1 breakout → 80% within ≤60m, pullback ≤0.25·ATR(M5)")
    print("🎲 Expected: H1_breakout_flag & dir-aligned, f8_q ≥ P95, TheoryB_Δt ≤ 30m")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test E3-specific queries
        test_queries = [
            "Analyze E3 ACCEL paths with H1 breakout confirmation",
            "Show me acceleration patterns with H1 alignment",
            "What are the performance metrics for E3 ACCEL paths?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            result = engine.ask(query)
            print_e_analysis_results(result, result_type="E3_ACCEL")
            
    except Exception as e:
        print(f"❌ E3 Analysis Error: {e}")
        traceback.print_exc()

def demo_pattern_switch_diagnostics():
    """Demonstrate pattern-switch diagnostics and regime analysis"""
    print("\n🔄 PATTERN-SWITCH DIAGNOSTICS")
    print("=" * 60)
    print("📋 Analysis: Regime flips, news proximity, H1 confirmation, gap context, micro momentum")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test pattern switch queries
        test_queries = [
            "Analyze pattern switches and regime transitions",
            "Show me regime flips from CONT to MR",
            "What causes path selection changes?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            result = engine.ask(query)
            print_switch_diagnostics(result)
            
    except Exception as e:
        print(f"❌ Pattern Switch Analysis Error: {e}")
        traceback.print_exc()

def demo_trigger_analysis():
    """Demonstrate RD-40-FT trigger condition analysis"""
    print("\n🎯 RD-40-FT TRIGGER ANALYSIS")
    print("=" * 60)
    print("📋 Triggers: RD-40-FT-CONT (≥85% precision), RD-40-FT-MR (alert mode), RD-40-FT-ACCEL (execution-grade)")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test trigger condition queries
        test_queries = [
            "Analyze RD-40-FT trigger conditions with precision gates",
            "Show me trigger candidates for CONT MR ACCEL",
            "What are the precision rates for execution triggers?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            result = engine.ask(query)
            print_trigger_analysis(result)
            
    except Exception as e:
        print(f"❌ Trigger Analysis Error: {e}")
        traceback.print_exc()

def demo_comparative_analysis():
    """Demonstrate comparative analysis across all E1/E2/E3 paths"""
    print("\n📊 COMPARATIVE E1/E2/E3 ANALYSIS")
    print("=" * 60)
    print("📋 Cross-path comparison: success rates, timing, confidence, feature importance")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Run all three analyses
        print("🔄 Running comprehensive E1/E2/E3 analysis...")
        
        e1_result = engine.ask("Analyze E1 CONT paths")
        e2_result = engine.ask("Analyze E2 MR paths") 
        e3_result = engine.ask("Analyze E3 ACCEL paths")
        
        # Comparative summary
        print(f"\n📈 COMPARATIVE SUMMARY")
        print("=" * 50)
        
        # Extract key metrics
        e1_count = e1_result.get("total_e1_cont", 0)
        e2_count = e2_result.get("total_e2_mr", 0)
        e3_count = e3_result.get("total_e3_accel", 0)
        total_rd40 = e1_result.get("total_rd40_events", 0)
        
        print(f"🎯 Total RD@40% Events Analyzed: {total_rd40}")
        print(f"📊 Path Distribution:")
        print(f"   E1 CONT: {e1_count} ({e1_count/total_rd40:.1%} of RD@40%)" if total_rd40 > 0 else "   E1 CONT: 0")
        print(f"   E2 MR:   {e2_count} ({e2_count/total_rd40:.1%} of RD@40%)" if total_rd40 > 0 else "   E2 MR: 0")  
        print(f"   E3 ACCEL:{e3_count} ({e3_count/total_rd40:.1%} of RD@40%)" if total_rd40 > 0 else "   E3 ACCEL: 0")
        
        # Timing comparison
        print(f"\n⏱️ Timing Analysis:")
        if e1_result.get("timing_analysis"):
            timing = e1_result["timing_analysis"]
            print(f"   E1 CONT: {timing.get('avg_time_to_60', 'N/A')}min→60%, {timing.get('avg_time_to_80', 'N/A')}min→80%")
        if e3_result.get("performance_metrics"):
            perf = e3_result["performance_metrics"]
            print(f"   E3 ACCEL: {perf.get('avg_time_to_80', 'N/A')}min→80% (fast track)")
        
        # Success rates
        print(f"\n🎯 Success Rates:")
        e1_success = e1_result.get("timing_analysis", {}).get("success_rate", 0)
        e2_success = e2_count / total_rd40 if total_rd40 > 0 else 0
        e3_success = e3_count / total_rd40 if total_rd40 > 0 else 0
        
        print(f"   E1 CONT: {e1_success:.1%}")
        print(f"   E2 MR:   {e2_success:.1%}")
        print(f"   E3 ACCEL:{e3_success:.1%}")
        
        # Identify dominant path
        path_counts = {"E1_CONT": e1_count, "E2_MR": e2_count, "E3_ACCEL": e3_count}
        dominant_path = max(path_counts, key=path_counts.get)
        print(f"\n👑 Dominant Path: {dominant_path} ({path_counts[dominant_path]} events)")
        
    except Exception as e:
        print(f"❌ Comparative Analysis Error: {e}")
        traceback.print_exc()

def print_e_analysis_results(result: dict, result_type: str = "E_PATH"):
    """Print E1/E2/E3 analysis results in formatted way"""
    if not result:
        print("  ❌ No results returned")
        return
    
    query_type = result.get("query_type", "unknown")
    print(f"  📋 Analysis Type: {query_type}")
    
    # Print key counts
    total_rd40 = result.get("total_rd40_events", 0)
    print(f"  🎯 Total RD@40% Events: {total_rd40}")
    
    if "e1_cont" in query_type:
        cont_events = result.get("total_e1_cont", 0)
        print(f"  📊 E1 CONT Events: {cont_events}")
        
        # Timing analysis
        if result.get("timing_analysis"):
            timing = result["timing_analysis"]
            print(f"  ⏱️ Timing Statistics:")
            print(f"     Average 60%: {timing.get('avg_time_to_60', 'N/A')}min")
            print(f"     Average 80%: {timing.get('avg_time_to_80', 'N/A')}min")
            print(f"     Success Rate: {timing.get('success_rate', 0):.1%}")
            print(f"     Drawdown Risk: {timing.get('avg_drawdown_risk', 0):.3f}")
    
    elif "e2_mr" in query_type:
        mr_events = result.get("total_e2_mr", 0)
        print(f"  📊 E2 MR Events: {mr_events}")
        
        # Branch analysis
        if result.get("branch_analysis"):
            branches = result["branch_analysis"]
            print(f"  🌿 Branch Analysis:")
            print(f"     Second RD Rate: {branches.get('second_rd_rate', 0):.1%}")
            print(f"     Failure Rate: {branches.get('failure_rate', 0):.1%}")
            print(f"     Unknown Rate: {branches.get('unknown_rate', 0):.1%}")
    
    elif "e3_accel" in query_type:
        accel_events = result.get("total_e3_accel", 0)
        print(f"  📊 E3 ACCEL Events: {accel_events}")
        
        # H1 analysis
        if result.get("h1_analysis"):
            h1 = result["h1_analysis"]
            print(f"  🎯 H1 Analysis:")
            print(f"     Breakout Detection Rate: {h1.get('breakout_detection_rate', 0):.1%}")
            print(f"     Direction Alignment Rate: {h1.get('direction_alignment_rate', 0):.1%}")
            print(f"     ACCEL Conversion Rate: {h1.get('accel_conversion_rate', 0):.1%}")
        
        # Performance metrics
        if result.get("performance_metrics"):
            perf = result["performance_metrics"]
            print(f"  ⚡ Performance Metrics:")
            print(f"     Average Time to 80%: {perf.get('avg_time_to_80', 'N/A')}min")
            print(f"     Average Pullback Depth: {perf.get('avg_pullback_depth', 0):.3f}")
            print(f"     Continuation Probability: {perf.get('avg_continuation_prob', 0):.1%}")
    
    # Print insights
    insights = result.get("insights", [])
    if insights:
        print(f"  💡 Key Insights:")
        for insight in insights[:3]:  # Show first 3 insights
            print(f"     • {insight}")
    
    print()  # Add spacing

def print_switch_diagnostics(result: dict):
    """Print pattern switch diagnostics results"""
    if not result:
        print("  ❌ No switch diagnostics returned")
        return
    
    print(f"  📋 Analysis Type: {result.get('query_type', 'unknown')}")
    
    # Regime analysis
    if result.get("regime_analysis"):
        regime = result["regime_analysis"]
        transitions = regime.get("regime_transitions", {})
        rates = regime.get("transition_rates", {})
        
        print(f"  🔄 Regime Transitions:")
        print(f"     CONT→MR: {transitions.get('cont_to_mr', 0)} ({rates.get('cont_to_mr_rate', 0):.1%})")
        print(f"     MR→CONT: {transitions.get('mr_to_cont', 0)} ({rates.get('mr_to_cont_rate', 0):.1%})")
        print(f"     Stable: {transitions.get('stable', 0)} ({rates.get('stability_rate', 0):.1%})")
        
        news = regime.get("news_effects", {})
        print(f"  📰 News Effects:")
        print(f"     High Impact Events: {news.get('high_impact_events', 0)}")
        print(f"     Suppression Events: {news.get('suppression_events', 0)}")
        
        h1 = regime.get("h1_confirmations", {})
        print(f"  📈 H1 Confirmations:")
        print(f"     Positive: {h1.get('positive_confirmations', 0)}/{h1.get('total_breakouts', 0)}")
    
    # Print insights
    insights = result.get("insights", [])
    if insights:
        print(f"  💡 Key Insights:")
        for insight in insights[:3]:
            print(f"     • {insight}")
    
    print()

def print_trigger_analysis(result: dict):
    """Print trigger analysis results"""
    if not result:
        print("  ❌ No trigger analysis returned")
        return
    
    print(f"  📋 Analysis Type: {result.get('query_type', 'unknown')}")
    
    # Trigger statistics
    if result.get("trigger_analysis"):
        triggers = result["trigger_analysis"]
        
        print(f"  🎯 Trigger Statistics:")
        for trigger_name, stats in triggers.items():
            candidates = stats.get("candidates", 0)
            high_conf = stats.get("high_confidence", 0)
            precision = stats.get("precision_rate", 0)
            print(f"     {trigger_name}: {high_conf}/{candidates} ({precision:.1%} precision)")
    
    # Precision gates
    if result.get("precision_gates"):
        gates = result["precision_gates"]
        print(f"  🚪 Precision Gates:")
        for gate_name, gate_config in gates.items():
            min_conf = gate_config.get("min_confidence", "N/A")
            features = gate_config.get("required_features", [])
            print(f"     {gate_name}: ≥{min_conf} confidence")
            if features:
                print(f"        Features: {', '.join(features[:2])}")  # Show first 2 features
    
    # Print insights
    insights = result.get("insights", [])
    if insights:
        print(f"  💡 Key Insights:")
        for insight in insights[:3]:
            print(f"     • {insight}")
    
    print()

def main():
    """Main demonstration function"""
    print("🚀 IRONFORGE: Advanced Experiment Set E1/E2/E3 Demonstration")
    print("🎯 Sophisticated Post-RD@40% Path Analysis with Statistical Rigor")
    print("=" * 80)
    
    try:
        # Run all demonstration modules
        demo_e1_cont_analysis()
        demo_e2_mr_analysis()
        demo_e3_accel_analysis()
        demo_pattern_switch_diagnostics()
        demo_trigger_analysis()
        demo_comparative_analysis()
        
        print("\n✅ Advanced Experiment Set E1/E2/E3 Demonstration Complete!")
        print("🎯 The Enhanced Temporal Query Engine now supports:")
        print("   • E1 CONT: 60%→80% progression with timing constraints")
        print("   • E2 MR: Mid-reversion with second_rd/failure branching")
        print("   • E3 ACCEL: H1 breakout confirmation with fast 80% reach")
        print("   • Pattern-switch diagnostics with regime analysis")
        print("   • RD-40-FT trigger conditions with precision gates")
        print("   • Advanced feature derivation (f8_q, f8_slope_sign, gap_age)")
        print("   • Statistical rigor with confidence scoring and KPIs")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are available:")
        print("   - enhanced_temporal_query_engine.py")
        print("   - experiment_e_analyzer.py")
        print("   - session_time_manager.py") 
        print("   - archaeological_zone_calculator.py")
        print("   - sklearn (for isotonic regression)")
        print("   - Data in data/adapted/ directory")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()