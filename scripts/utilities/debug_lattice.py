#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from analysis.enhanced_session_adapter import EnhancedSessionAdapter
from analysis.timeframe_lattice_mapper import TimeframeLatticeMapper

# Test with just one small session
adapter = EnhancedSessionAdapter()
mapper = TimeframeLatticeMapper()

# Load one session
with open('/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_NY_PM_Lvl-1_2025_08_05.json') as f:
    session_data = json.load(f)

events = session_data['events'][:5]  # Test with 5 events

print("Testing full lattice node creation:")
try:
    coordinates = mapper._create_event_coordinates(events)
    print(f"✅ Created {len(coordinates)} coordinates")
    
    # Now test node creation
    mapper._create_lattice_nodes(events, coordinates)
    print(f"✅ Created {len(mapper.lattice_nodes)} lattice nodes")
    for node_id, node in mapper.lattice_nodes.items():
        print(f"  {node_id}: {node.event_count} events")
        
except Exception as e:
    print(f"ERROR in node creation: {e}")
    import traceback
    traceback.print_exc()