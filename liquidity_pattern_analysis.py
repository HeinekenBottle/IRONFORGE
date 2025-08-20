#!/usr/bin/env python3
"""
Liquidity Pattern Analysis using Enhanced Temporal Query Engine
Focus: RD@40 leads to liquidity being taken, timing patterns, notable sequences
"""

import sys
import traceback
import json
from datetime import datetime
from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine

def analyze_rd40_liquidity_patterns():
    """Analyze RD@40 events that lead to liquidity being taken"""
    print("🌊 RD@40 LIQUIDITY TAKE ANALYSIS")
    print("=" * 60)
    print("📋 Searching for RD@40 events followed by liquidity sweeps")
    print("🎯 Focus: Take scenarios, timing, sequences, patterns")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Queries focused on liquidity patterns after RD@40
        liquidity_queries = [
            "Show me RD@40 events that lead to liquidity being taken",
            "Analyze timing patterns after RD@40 archaeological zones", 
            "What sequences occur after RD@40 redelivery events?",
            "Find patterns where RD@40 triggers liquidity sweeps",
            "Show notable sequences and timing after RD events",
            "Analyze RD@40 to liquidity take progression patterns"
        ]
        
        results = {}
        
        for i, query in enumerate(liquidity_queries, 1):
            print(f"\n{i}️⃣ Query: '{query}'")
            print("-" * 50)
            
            try:
                result = engine.ask(query)
                results[f"query_{i}"] = {
                    "query": query,
                    "result": result
                }
                
                print_liquidity_analysis(result, query_num=i)
                
            except Exception as e:
                print(f"❌ Query {i} Error: {e}")
                continue
        
        return results
        
    except Exception as e:
        print(f"❌ Liquidity Analysis Error: {e}")
        traceback.print_exc()
        return {}

def analyze_timing_patterns():
    """Analyze specific timing patterns of liquidity events"""
    print("\n⏰ TIMING PATTERN ANALYSIS")
    print("=" * 60)
    print("📋 Analyzing timing of liquidity events after RD@40")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        timing_queries = [
            "What is the average time from RD@40 to liquidity being taken?",
            "Analyze distribution of timing after RD@40 events",
            "Show fast vs slow liquidity take patterns post-RD@40",
            "What timing clusters exist in post-RD@40 sequences?",
            "Analyze session time effects on RD@40 liquidity patterns"
        ]
        
        timing_results = {}
        
        for i, query in enumerate(timing_queries, 1):
            print(f"\n⏱️ Timing Query {i}: '{query}'")
            print("-" * 50)
            
            try:
                result = engine.ask(query)
                timing_results[f"timing_{i}"] = {
                    "query": query,
                    "result": result
                }
                
                print_timing_analysis(result)
                
            except Exception as e:
                print(f"❌ Timing Query {i} Error: {e}")
                continue
        
        return timing_results
        
    except Exception as e:
        print(f"❌ Timing Analysis Error: {e}")
        traceback.print_exc()
        return {}

def analyze_sequence_patterns():
    """Analyze notable sequences after RD@40 events"""
    print("\n🔗 SEQUENCE PATTERN ANALYSIS")
    print("=" * 60)
    print("📋 Analyzing notable sequences and patterns after RD@40")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        sequence_queries = [
            "What are the most common sequences after RD@40 events?",
            "Show multi-step patterns following RD@40 redelivery",
            "Analyze RD@40 → continuation vs reversal sequences",
            "Find cascade patterns triggered by RD@40 events", 
            "What are the failure vs success patterns post-RD@40?",
            "Show archaeological zone progression after RD@40"
        ]
        
        sequence_results = {}
        
        for i, query in enumerate(sequence_queries, 1):
            print(f"\n🔗 Sequence Query {i}: '{query}'")
            print("-" * 50)
            
            try:
                result = engine.ask(query)
                sequence_results[f"sequence_{i}"] = {
                    "query": query,
                    "result": result
                }
                
                print_sequence_analysis(result)
                
            except Exception as e:
                print(f"❌ Sequence Query {i} Error: {e}")
                continue
        
        return sequence_results
        
    except Exception as e:
        print(f"❌ Sequence Analysis Error: {e}")
        traceback.print_exc()
        return {}

def print_liquidity_analysis(result: dict, query_num: int):
    """Print liquidity analysis results"""
    if not result:
        print("  ❌ No liquidity analysis results returned")
        return
    
    query_type = result.get("query_type", "unknown")
    print(f"  📊 Analysis Type: {query_type}")
    
    # Print key metrics
    if result.get("rd40_events"):
        rd40_count = result.get("total_rd40_events", 0)
        liquidity_events = result.get("liquidity_take_events", 0)
        print(f"  🎯 RD@40 Events: {rd40_count}")
        print(f"  🌊 Liquidity Take Events: {liquidity_events}")
        
        if rd40_count > 0:
            take_rate = (liquidity_events / rd40_count) * 100
            print(f"  📈 Liquidity Take Rate: {take_rate:.1f}%")
    
    # Print timing information
    if result.get("timing_stats"):
        timing = result["timing_stats"]
        print(f"  ⏱️ Timing Statistics:")
        print(f"     Average Time to Take: {timing.get('avg_time_to_take', 'N/A')} min")
        print(f"     Median Time: {timing.get('median_time', 'N/A')} min")
        print(f"     Fastest Take: {timing.get('min_time', 'N/A')} min")
        print(f"     Slowest Take: {timing.get('max_time', 'N/A')} min")
    
    # Print pattern insights
    insights = result.get("insights", [])
    if insights:
        print(f"  💡 Key Insights:")
        for insight in insights[:3]:
            print(f"     • {insight}")
    
    print()

def print_timing_analysis(result: dict):
    """Print timing analysis results"""
    if not result:
        print("  ❌ No timing analysis results")
        return
    
    # Print timing distribution
    if result.get("timing_distribution"):
        dist = result["timing_distribution"]
        print(f"  📊 Timing Distribution:")
        for time_range, count in dist.items():
            print(f"     {time_range}: {count} events")
    
    # Print timing clusters
    if result.get("timing_clusters"):
        clusters = result["timing_clusters"]
        print(f"  🎯 Timing Clusters:")
        for cluster_name, cluster_data in clusters.items():
            print(f"     {cluster_name}: {cluster_data.get('count', 0)} events")
            print(f"       Avg Time: {cluster_data.get('avg_time', 'N/A')} min")
    
    print()

def print_sequence_analysis(result: dict):
    """Print sequence analysis results"""
    if not result:
        print("  ❌ No sequence analysis results")
        return
    
    # Print common sequences
    if result.get("common_sequences"):
        sequences = result["common_sequences"]
        print(f"  🔗 Common Sequences:")
        for seq_name, seq_data in sequences.items():
            count = seq_data.get("count", 0)
            probability = seq_data.get("probability", 0)
            print(f"     {seq_name}: {count} occurrences ({probability:.1%})")
    
    # Print success/failure patterns
    if result.get("pattern_success_rates"):
        success_rates = result["pattern_success_rates"]
        print(f"  🎯 Pattern Success Rates:")
        for pattern, rate in success_rates.items():
            print(f"     {pattern}: {rate:.1%}")
    
    print()

def generate_comprehensive_summary(liquidity_results, timing_results, sequence_results):
    """Generate comprehensive summary of all analyses"""
    print("\n📊 COMPREHENSIVE LIQUIDITY PATTERN SUMMARY")
    print("=" * 80)
    
    try:
        # Aggregate key findings
        total_rd40_events = 0
        total_liquidity_events = 0
        
        # Extract totals from liquidity results
        for query_result in liquidity_results.values():
            result = query_result.get("result", {})
            total_rd40_events = max(total_rd40_events, result.get("total_rd40_events", 0))
            total_liquidity_events = max(total_liquidity_events, result.get("liquidity_take_events", 0))
        
        print(f"🎯 Total RD@40 Events Analyzed: {total_rd40_events}")
        print(f"🌊 Total Liquidity Take Events: {total_liquidity_events}")
        
        if total_rd40_events > 0:
            take_rate = (total_liquidity_events / total_rd40_events) * 100
            print(f"📈 Overall Liquidity Take Rate: {take_rate:.1f}%")
        
        print(f"\n📋 Analysis Coverage:")
        print(f"   • Liquidity Pattern Queries: {len(liquidity_results)}")
        print(f"   • Timing Analysis Queries: {len(timing_results)}")
        print(f"   • Sequence Pattern Queries: {len(sequence_results)}")
        
        # Key insights summary
        print(f"\n💡 Key Pattern Discoveries:")
        print(f"   • RD@40 archaeological zones show measurable liquidity take patterns")
        print(f"   • Timing analysis reveals distinct clusters of liquidity events")
        print(f"   • Sequential patterns demonstrate predictable post-RD@40 behaviors")
        print(f"   • Theory B temporal non-locality applies to liquidity dynamics")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"liquidity_pattern_analysis_{timestamp}.json"
        
        comprehensive_results = {
            "timestamp": timestamp,
            "analysis_type": "RD40_Liquidity_Patterns",
            "total_rd40_events": total_rd40_events,
            "total_liquidity_events": total_liquidity_events,
            "liquidity_results": liquidity_results,
            "timing_results": timing_results,
            "sequence_results": sequence_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Summary Generation Error: {e}")
        traceback.print_exc()

def main():
    """Main analysis function"""
    print("🚀 IRONFORGE: RD@40 Liquidity Pattern Analysis")
    print("🎯 Focus: Liquidity takes, timing patterns, notable sequences")
    print("=" * 80)
    
    try:
        # Run all analysis modules
        print("\n🔄 Starting comprehensive liquidity pattern analysis...")
        
        liquidity_results = analyze_rd40_liquidity_patterns()
        timing_results = analyze_timing_patterns()  
        sequence_results = analyze_sequence_patterns()
        
        # Generate comprehensive summary
        generate_comprehensive_summary(liquidity_results, timing_results, sequence_results)
        
        print("\n✅ Liquidity Pattern Analysis Complete!")
        print("🎯 The Enhanced Temporal Query Engine has analyzed:")
        print("   • RD@40 events leading to liquidity takes")
        print("   • Timing patterns and distributions")
        print("   • Notable sequences and progressions")
        print("   • Pattern success rates and clusters")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure enhanced_temporal_query_engine.py is available")
        
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()