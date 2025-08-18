#!/usr/bin/env python3
"""
PM Lattice Test - Process only PM sessions for faster validation
"""

import glob
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from analysis.enhanced_session_adapter import EnhancedSessionAdapter
from analysis.timeframe_lattice_mapper import TimeframeLatticeMapper


def main():
    adapter = EnhancedSessionAdapter()
    mapper = TimeframeLatticeMapper(
        grid_resolution=50,  # Reduced resolution for speed
        min_node_events=1,   # Lower threshold for testing
        hot_zone_threshold=0.7  # Lower threshold for testing
    )
    
    # Find only PM session files
    pm_patterns = [
        "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_*PM*.json",
        "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_NYPM*.json"
    ]
    
    all_events = []
    
    for pattern in pm_patterns:
        files = glob.glob(pattern)
        print(f"Pattern {pattern}: {len(files)} files")
        
        for file_path in files:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            
            events = session_data.get('events', [])
            all_events.extend(events)
            print(f"  {Path(file_path).name}: {len(events)} events")
    
    print(f"\nTotal PM events: {len(all_events)}")
    
    print("\nMapping to lattice...")
    start_time = time.time()
    lattice_dataset = mapper.map_events_to_lattice(all_events)
    end_time = time.time()
    
    print(f"✅ Lattice created in {end_time - start_time:.2f}s")
    print(f"   Nodes: {len(lattice_dataset.nodes)}")
    print(f"   Connections: {len(lattice_dataset.connections)}")
    print(f"   Hot zones: {len(lattice_dataset.hot_zones)}")
    
    # Export
    output_path = "/Users/jack/IRONFORGE/deliverables/lattice_dataset/pm_only_lattice_dataset.json"
    mapper.export_lattice_dataset(lattice_dataset, output_path)
    print(f"Exported to: {output_path}")

if __name__ == "__main__":
    import time
    main()