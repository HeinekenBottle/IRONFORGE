#!/usr/bin/env python3
"""
Demonstration of the IRONFORGE Temporal Query Engine
Shows programmatic usage without interactive CLI issues
"""

from temporal_query_engine import TemporalQueryEngine
import pandas as pd

def demonstrate_temporal_queries():
    """Demonstrate the key capabilities of the Temporal Query Engine"""
    
    print("🚀 IRONFORGE Temporal Query Engine Demo")
    print("=" * 50)
    
    # Initialize the engine
    print("\n1. Initializing Engine...")
    engine = TemporalQueryEngine()
    
    print(f"\n2. System Overview:")
    print(f"   Sessions loaded: {len(engine.sessions)}")
    print(f"   NetworkX graphs: {len(engine.graphs)}")
    
    # Show available sessions
    sessions = list(engine.sessions.keys())[:5]  # First 5
    print(f"\n3. Sample Sessions:")
    for i, session_id in enumerate(sessions, 1):
        nodes = engine.sessions[session_id]
        print(f"   {i}. {session_id}: {len(nodes)} events")
    
    # Demonstrate programmatic queries
    print(f"\n4. Example Query Analysis:")
    
    # Simulate a temporal sequence query
    print(f"\n   🤔 Query: 'What happens after high liquidity events?'")
    
    # Analyze high f8 events across sessions
    high_f8_outcomes = []
    
    for session_id, nodes in list(engine.sessions.items())[:10]:  # First 10 sessions
        if 'f8' in nodes.columns and len(nodes) > 20:
            # Find 95th percentile f8 events
            f8_threshold = nodes['f8'].quantile(0.95)
            high_events = nodes[nodes['f8'] > f8_threshold]
            
            for idx, event in high_events.iterrows():
                event_pos = nodes.index.get_loc(idx)
                
                # Look at next 10 events
                if event_pos < len(nodes) - 10:
                    future_events = nodes.iloc[event_pos+1:event_pos+11]
                    
                    # Simple outcome classification
                    price_change = future_events['price'].iloc[-1] - event['price']
                    price_range = future_events['price'].max() - future_events['price'].min()
                    
                    outcome = "expansion" if price_range > 20 else "consolidation"
                    high_f8_outcomes.append({
                        'session': session_id,
                        'outcome': outcome,
                        'price_change': price_change,
                        'range': price_range
                    })
    
    # Analyze results
    if high_f8_outcomes:
        df_outcomes = pd.DataFrame(high_f8_outcomes)
        
        print(f"   📊 Analysis Results:")
        print(f"     Total high f8 events: {len(df_outcomes)}")
        
        outcome_probs = df_outcomes['outcome'].value_counts(normalize=True)
        for outcome, prob in outcome_probs.items():
            print(f"     {outcome}: {prob:.1%}")
        
        avg_range = df_outcomes['range'].mean()
        print(f"     Average subsequent range: {avg_range:.1f} points")
    
    # Demonstrate session info capability
    print(f"\n5. Session Deep Dive Example:")
    sample_session = sessions[0]
    nodes = engine.sessions[sample_session]
    
    print(f"   Session: {sample_session}")
    print(f"   Duration: {(nodes['t'].max() - nodes['t'].min()) / (60*1000):.0f} minutes")
    print(f"   Price range: {nodes['price'].max() - nodes['price'].min():.1f} points")
    
    if 'f8' in nodes.columns:
        f8_stats = {
            'mean': nodes['f8'].mean(),
            'q90': nodes['f8'].quantile(0.90),
            'q95': nodes['f8'].quantile(0.95),
            'max': nodes['f8'].max()
        }
        print(f"   f8 liquidity stats:")
        for stat, value in f8_stats.items():
            print(f"     {stat}: {value:.0f}")
    
    print(f"\n6. Query Processing Architecture:")
    print(f"   ✅ Natural language parsing")
    print(f"   ✅ Temporal sequence analysis")  
    print(f"   ✅ Opening pattern detection")
    print(f"   ✅ Probability calculations")
    print(f"   ✅ NetworkX graph operations")
    
    print(f"\n🎯 Key Capabilities:")
    print(f"   • Ask questions in natural language")
    print(f"   • Analyze temporal cause-effect relationships")
    print(f"   • Calculate probabilistic outcomes")
    print(f"   • Search for specific patterns")
    print(f"   • Deep-dive into session characteristics")
    
    print(f"\n💡 Usage Patterns:")
    print(f"   • Programmatic: Import TemporalQueryEngine class")
    print(f"   • Jupyter: Use engine.ask('your question') method")
    print(f"   • Direct analysis: Access engine.sessions and engine.graphs")
    print(f"   • Interactive: Run temporal_query_engine.py in terminal")
    
    return engine

if __name__ == "__main__":
    engine = demonstrate_temporal_queries()
    print(f"\n✅ Demo complete! Engine ready for use.")