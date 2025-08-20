#!/usr/bin/env python3
"""
Demo: Day & News Query System Integration
Test the new TQE query handlers for day and news analysis
"""

from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine

def demo_day_news_queries():
    """Demonstrate the new day and news query capabilities"""
    print("🚀 DEMO: Day & News Query System")
    print("🎯 Testing new TQE query handlers")
    print("=" * 60)
    
    try:
        engine = EnhancedTemporalQueryEngine()
        
        # Test queries for day analysis
        day_queries = [
            "Analyze RD@40 patterns by day profile",
            "Show me RD@40 events by day of week"
        ]
        
        print("\n📅 TESTING DAY PROFILE QUERIES:")
        for i, query in enumerate(day_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 40)
            
            try:
                result = engine.ask(query)
                print(f"   Query Type: {result.get('query_type', 'unknown')}")
                print(f"   Total Sessions: {result.get('total_sessions', 0)}")
                print(f"   Total RD@40 Events: {result.get('total_rd40_events', 0)}")
                
                if 'day_analysis' in result:
                    print(f"   Days Analyzed: {len(result['day_analysis'])}")
                
                # Show first few insights
                insights = result.get('insights', [])
                if insights:
                    print(f"   Key Insights:")
                    for insight in insights[:3]:
                        print(f"     • {insight}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test queries for news analysis
        news_queries = [
            "Analyze RD@40 patterns by news proximity", 
            "Show me news impact on RD@40 events"
        ]
        
        print("\n📰 TESTING NEWS IMPACT QUERIES:")
        for i, query in enumerate(news_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 40)
            
            try:
                result = engine.ask(query)
                print(f"   Query Type: {result.get('query_type', 'unknown')}")
                print(f"   Events with News Context: {result.get('total_events_with_news_context', 0)}")
                
                if 'news_analysis' in result:
                    print(f"   News Buckets Analyzed: {len(result['news_analysis'])}")
                
                # Show insights
                insights = result.get('insights', [])
                if insights:
                    print(f"   Key Insights:")
                    for insight in insights[:3]:
                        print(f"     • {insight}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test matrix analysis
        matrix_queries = [
            "Create day news matrix for RD@40 analysis"
        ]
        
        print("\n🗓️📰 TESTING MATRIX QUERIES:")
        for i, query in enumerate(matrix_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 40)
            
            try:
                result = engine.ask(query)
                print(f"   Query Type: {result.get('query_type', 'unknown')}")
                
                if 'matrix_data' in result:
                    matrix_cells = sum(len(day_data) for day_data in result['matrix_data'].values())
                    print(f"   Matrix Cells: {matrix_cells}")
                
                # Show insights
                insights = result.get('insights', [])
                if insights:
                    print(f"   Key Insights:")
                    for insight in insights[:3]:
                        print(f"     • {insight}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test f8 interactions
        f8_queries = [
            "Analyze f8 interactions with day and news"
        ]
        
        print("\n📊 TESTING F8 INTERACTION QUERIES:")
        for i, query in enumerate(f8_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 40)
            
            try:
                result = engine.ask(query)
                print(f"   Query Type: {result.get('query_type', 'unknown')}")
                
                # Show insights
                insights = result.get('insights', [])
                if insights:
                    print(f"   Key Insights:")
                    for insight in insights[:3]:
                        print(f"     • {insight}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n✅ Day & News Query System Demo Complete!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    demo_day_news_queries()