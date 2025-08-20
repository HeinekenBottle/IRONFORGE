#!/usr/bin/env python3
"""
Demo: Experiment Set E Integration with Enhanced Temporal Query Engine
Demonstrates post-RD@40% sequence analysis (CONT/MR/ACCEL path prediction)
"""

import sys
import traceback
from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine

def demo_rd40_detection():
    """Demonstrate RD@40% event detection capabilities"""
    print("🎯 EXPERIMENT SET E DEMONSTRATION")
    print("=" * 60)
    print("🔍 Post-RD@40% Sequence Analysis (CONT/MR/ACCEL Path Prediction)")
    print("=" * 60)
    
    try:
        # Initialize the Enhanced Temporal Query Engine
        print("\n📡 Initializing Enhanced Temporal Query Engine...")
        engine = EnhancedTemporalQueryEngine()
        
        # Display session summary
        sessions = engine.list_sessions()
        print(f"✅ Loaded {len(sessions)} sessions with price relativity calculations")
        
        if len(sessions) == 0:
            print("❌ No sessions found. Please check data/adapted/ directory.")
            return
        
        # Show sample sessions
        print(f"\n📊 Sample Sessions (first 5):")
        for session in sessions[:5]:
            print(f"  • {session}")
        
        print(f"\n🎯 Testing Experiment Set E Queries...")
        print("-" * 50)
        
        # Test 1: Basic RD@40% analysis
        print("\n1️⃣ Basic RD@40% Sequence Analysis")
        print("Query: 'What happens after RD@40% events?'")
        
        result1 = engine.ask("What happens after RD@40% events?")
        print_results(result1, max_items=3)
        
        # Test 2: Path probability analysis  
        print("\n2️⃣ Path Probability Analysis")
        print("Query: 'Show path probabilities for CONT MR ACCEL'")
        
        result2 = engine.ask("Show path probabilities for CONT MR ACCEL")
        print_results(result2, max_items=5)
        
        # Test 3: Post-RD sequence analysis
        print("\n3️⃣ Post-RD Sequence Analysis")
        print("Query: 'Analyze post-rd sequences after redelivery@40%'")
        
        result3 = engine.ask("Analyze post-rd sequences after redelivery@40%")
        print_results(result3, max_items=3)
        
        # Summary statistics
        print(f"\n📈 EXPERIMENT SET E SUMMARY")
        print("=" * 50)
        
        total_rd40_events = result1.get('total_rd40_events', 0)
        path_probs = result2.get('path_statistics', {})
        
        print(f"🎯 Total RD@40% Events Detected: {total_rd40_events}")
        
        if path_probs:
            print(f"📊 Path Probability Distribution:")
            for path, stats in path_probs.items():
                prob = stats.get('probability', 0)
                count = stats.get('count', 0) 
                conf = stats.get('avg_confidence', 0)
                print(f"   {path}: {prob:.1%} (n={count}, conf={conf:.2f})")
        
        # Test specific session analysis
        if len(sessions) > 0:
            print(f"\n🔬 Detailed Session Analysis")
            sample_session = sessions[0].split()[0]  # Extract session ID
            print(f"Analyzing session: {sample_session}")
            
            session_info = engine.session_info(sample_session)
            if 'archaeological_zones' in session_info:
                zones = session_info['archaeological_zones'].get('zones', {})
                print(f"Archaeological Zones:")
                for zone_pct, zone_data in zones.items():
                    level = zone_data.get('level', 0)
                    destiny = " ⭐" if zone_data.get('is_dimensional_destiny', False) else ""
                    print(f"  {zone_pct}: {level:.2f}{destiny}")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("🔍 Traceback:")
        traceback.print_exc()

def print_results(result: dict, max_items: int = 5):
    """Print query results in a formatted way"""
    if not result:
        print("  ❌ No results returned")
        return
    
    query_type = result.get('query_type', 'unknown')
    print(f"  📋 Query Type: {query_type}")
    
    # Print key metrics
    for key, value in result.items():
        if key in ['total_sessions', 'total_rd40_events', 'total_events']:
            print(f"  📊 {key.replace('_', ' ').title()}: {value}")
    
    # Print path probabilities
    if 'path_probabilities' in result:
        print(f"  🎯 Path Probabilities:")
        for path, prob in result['path_probabilities'].items():
            print(f"     {path}: {prob:.1%}")
    
    # Print path statistics
    if 'path_statistics' in result:
        print(f"  📈 Path Statistics:")
        for path, stats in result['path_statistics'].items():
            count = stats.get('count', 0)
            prob = stats.get('probability', 0)
            conf = stats.get('avg_confidence', 0)
            print(f"     {path}: {prob:.1%} (n={count}, confidence={conf:.2f})")
    
    # Print insights
    if 'insights' in result and result['insights']:
        print(f"  💡 Key Insights:")
        for insight in result['insights'][:max_items]:
            print(f"     • {insight}")
    
    # Print Wilson confidence intervals
    if 'wilson_confidence_intervals' in result:
        print(f"  📊 95% Confidence Intervals:")
        for path, ci in result['wilson_confidence_intervals'].items():
            lower = ci.get('lower', 0)
            upper = ci.get('upper', 0)
            if lower > 0 or upper > 0:
                print(f"     {path}: [{lower:.3f}, {upper:.3f}]")
    
    # Print RD@40% events (sample)
    if 'rd40_events' in result and result['rd40_events']:
        events = result['rd40_events'][:max_items]
        print(f"  🎯 Sample RD@40% Events ({len(events)} of {len(result['rd40_events'])}):")
        for event in events:
            session_id = event.get('session_id', 'Unknown')
            price = event.get('price', 0)
            strength = event.get('strength', 0)
            print(f"     • {session_id}: Price={price:.2f}, Strength={strength:.3f}")
    
    # Print path classifications (sample)
    if 'path_classifications' in result and result['path_classifications']:
        classifications = list(result['path_classifications'].items())[:max_items]
        print(f"  🛤️ Sample Path Classifications:")
        for event_key, classification in classifications:
            path = classification.get('path', 'UNKNOWN')
            confidence = classification.get('confidence', 0)
            print(f"     • {event_key}: {path} (confidence={confidence:.2f})")
    
    print()  # Add spacing

def demo_interactive_queries():
    """Demonstrate interactive query capabilities"""
    print("\n🤖 Interactive Query Demo")
    print("=" * 40)
    
    engine = EnhancedTemporalQueryEngine()
    
    # Predefined queries to test
    test_queries = [
        "Show me RD@40% events in NY sessions",
        "What is the path probability after redelivery@40%?",
        "Analyze post-RD sequences with high confidence",
        "Find archaeological zone patterns at 40%"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}️⃣ Query: '{query}'")
        try:
            result = engine.ask(query)
            print_results(result, max_items=2)
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Main demonstration function"""
    print("🚀 IRONFORGE: Experiment Set E Demonstration")
    print("🎯 Post-RD@40% Sequence Analysis Integration")
    print()
    
    try:
        # Core demonstration
        demo_rd40_detection()
        
        # Interactive queries
        demo_interactive_queries()
        
        print("\n✅ Experiment Set E Demonstration Complete!")
        print("🎯 The Enhanced Temporal Query Engine now supports:")
        print("   • RD@40% event detection")
        print("   • CONT/MR/ACCEL path classification")  
        print("   • Path probability analysis with Wilson CIs")
        print("   • Feature extraction for f8_q, HTF features f47-f50")
        print("   • Natural language queries for sequence analysis")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are available:")
        print("   - enhanced_temporal_query_engine.py")
        print("   - session_time_manager.py") 
        print("   - archaeological_zone_calculator.py")
        print("   - Data in data/adapted/ directory")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()