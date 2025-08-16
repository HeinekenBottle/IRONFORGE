#!/usr/bin/env python3
"""
Test Task 3: Session Context Preservation - Simple Test
"""

import sys
sys.path.insert(0, '/Users/jack/IRONPULSE/IRONFORGE')

def test_task3():
    print("Testing Task 3: Session Context Preservation")
    
    try:
        from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
        
        builder = EnhancedGraphBuilder()
        
        # Simple test data
        test_data = {
            "session_metadata": {
                "session_type": "ny_am",
                "session_date": "2025-08-14",
                "session_start": "09:30:00",
                "session_end": "12:00:00",
                "session_duration": 150
            },
            "price_movements": [
                {
                    "timestamp": "09:30:00",
                    "price_level": 23500.0,
                    "movement_type": "open"
                }
            ]
        }
        
        # Test build_rich_graph returns tuple
        result = builder.build_rich_graph(test_data)
        
        if isinstance(result, tuple) and len(result) == 2:
            graph, metadata = result
            print("✅ build_rich_graph returns (graph, metadata) tuple")
            
            # Check metadata has required fields
            required = ['session_name', 'session_start_time', 'anchor_timeframe']
            missing = [f for f in required if f not in metadata]
            
            if not missing:
                print("✅ All required metadata fields present")
                print(f"  Session name: {metadata['session_name']}")
                print(f"  Start time: {metadata['session_start_time']}")
                print(f"  Anchor timeframe: {metadata['anchor_timeframe']}")
                print("🎉 TASK 3 SUCCESS!")
                return True
            else:
                print(f"❌ Missing fields: {missing}")
                return False
        else:
            print(f"❌ Expected tuple, got {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_task3()
    exit(0 if success else 1)