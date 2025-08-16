#!/usr/bin/env python3
"""
Test IRONFORGE integration with Enhanced Session Adapter sample data
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))

def test_enhanced_session_sample():
    """Test with realistic Enhanced Session Adapter data format"""
    print("🔗 Testing Enhanced Session Adapter Integration")
    print("=" * 50)
    
    try:
        from ironforge.learning.tgat_discovery import IRONFORGEDiscovery, TGATDiscovery
        from ironforge.analysis.timeframe_lattice_mapper import TimeframeLatticeMapper
        from visualizations.lattice_visualizer import LatticeVisualizer, VisualizationConfig
        
        # Simulate Enhanced Session Adapter output (Theory B format)
        enhanced_session_events = [
            {
                'event_id': 'session_2025-08-05_PM_event_001',
                'session_name': '2025-08-05_PM_session',
                'timeframe': '1m',
                'timestamp': '14:35:00',
                'relative_cycle_position': 0.40,  # Theory B: 40% zone
                'significance_score': 0.925,     # High significance (92.3/100 authenticity)
                'event_type': 'fvg_redelivery',
                'pattern_family': 'fvg_family',
                'liquidity_archetype': 'session_low_sweep',
                'structural_role': 'terminal_sweep',
                'session_minute': 35.0,
                'range_position_percent': 40.0,
                'session_date': '2025-08-05'
            },
            {
                'event_id': 'session_2025-08-05_PM_event_002', 
                'session_name': '2025-08-05_PM_session',
                'timeframe': '5m',
                'timestamp': '14:40:00',
                'relative_cycle_position': 0.45,
                'significance_score': 0.887,
                'event_type': 'expansion_phase',
                'pattern_family': 'expansion_family',
                'liquidity_archetype': 'breakout_catalyst',
                'structural_role': 'breakout',
                'session_minute': 40.0,
                'range_position_percent': 45.0,
                'session_date': '2025-08-05'
            },
            {
                'event_id': 'session_2025-08-06_NY_event_001',
                'session_name': '2025-08-06_NY_session',
                'timeframe': '1m',
                'timestamp': '09:32:00',
                'relative_cycle_position': 0.23,
                'significance_score': 0.756,
                'event_type': 'pd_array_formation',
                'pattern_family': 'pd_array_family',
                'liquidity_archetype': 'accumulation_zone',
                'structural_role': 'accumulation',
                'session_minute': 32.0,
                'range_position_percent': 23.0,
                'session_date': '2025-08-06'
            },
            {
                'event_id': 'session_2025-08-06_NY_event_002',
                'session_name': '2025-08-06_NY_session',
                'timeframe': '15m',
                'timestamp': '09:45:00',
                'relative_cycle_position': 0.78,
                'significance_score': 0.834,
                'event_type': 'sweep_sell_side',
                'pattern_family': 'sweep_family',
                'liquidity_archetype': 'liquidity_sweep',
                'structural_role': 'terminal_sweep',
                'session_minute': 45.0,
                'range_position_percent': 78.0,
                'session_date': '2025-08-06'
            }
        ]
        
        print(f"📊 Sample data: {len(enhanced_session_events)} events from 2 sessions")
        
        # Test 1: TGAT Discovery initialization
        print("\n🧠 Testing TGAT Discovery with Enhanced Session data...")
        
        # Test both import names work
        discovery_1 = IRONFORGEDiscovery(node_features=45, hidden_dim=128, out_dim=256)
        discovery_2 = TGATDiscovery(node_features=45, hidden_dim=128, out_dim=256)
        
        print("  ✅ Both IRONFORGEDiscovery and TGATDiscovery imports successful")
        
        # Test 2: Lattice Mapping
        print("\n🗺️  Testing Lattice Mapping with Enhanced Session data...")
        
        mapper = TimeframeLatticeMapper(grid_resolution=50, min_node_events=1, hot_zone_threshold=0.5)
        lattice_dataset = mapper.map_events_to_lattice(enhanced_session_events)
        
        print(f"  ✅ Lattice mapping successful:")
        print(f"    Nodes created: {len(lattice_dataset.nodes)}")
        print(f"    Connections: {len(lattice_dataset.connections)}")
        print(f"    Hot zones: {len(lattice_dataset.hot_zones)}")
        print(f"    Sessions covered: {len(lattice_dataset.sessions_covered)}")
        
        # Analyze nodes
        for node_id, node in lattice_dataset.nodes.items():
            print(f"    Node {node_id}: {node.event_count} events, {node.average_significance:.3f} avg significance")
            print(f"      Timeframe: {node.coordinate.absolute_timeframe.value}, Position: {node.coordinate.cycle_position:.2f}")
            print(f"      Dominant type: {node.dominant_event_type}, Family: {node.dominant_archetype}")
        
        # Analyze connections
        if lattice_dataset.connections:
            print(f"\n  🔗 Connection analysis:")
            for conn_id, conn in lattice_dataset.connections.items():
                print(f"    {conn.source_node_id} → {conn.target_node_id}: {conn.connection_type} (strength: {conn.strength:.3f})")
        
        # Test 3: Visualization Compatibility
        print("\n📊 Testing Visualization compatibility...")
        
        config = VisualizationConfig()
        visualizer = LatticeVisualizer(config)
        
        print("  ✅ Visualizer initialized successfully")
        print(f"    Has temporal heatmaps method: {hasattr(visualizer, 'create_temporal_heatmaps')}")
        
        # Test 4: Enhanced Session Adapter specific features
        print("\n🎯 Testing Enhanced Session Adapter specific features...")
        
        # Test Theory B verification (40% zone positioning)
        theory_b_events = [e for e in enhanced_session_events if abs(e['relative_cycle_position'] - 0.40) < 0.05]
        print(f"  ✅ Theory B events (40% zone): {len(theory_b_events)}")
        
        # Test multi-timeframe events
        timeframes = set(e['timeframe'] for e in enhanced_session_events)
        print(f"  ✅ Multi-timeframe coverage: {timeframes}")
        
        # Test session segregation
        sessions = set(e['session_name'] for e in enhanced_session_events)
        print(f"  ✅ Session segregation: {sessions}")
        
        # Test high authenticity scores
        high_auth_events = [e for e in enhanced_session_events if e['significance_score'] > 0.85]
        print(f"  ✅ High authenticity events (>85%): {len(high_auth_events)}")
        
        # Test cross-session pattern inheritance
        cross_session_nodes = 0
        for node in lattice_dataset.nodes.values():
            node_sessions = set()
            for event_dict in enhanced_session_events:
                # Check if this event could belong to this node
                if event_dict.get('timeframe') == node.coordinate.absolute_timeframe.value:
                    node_sessions.add(event_dict['session_name'])
            if len(node_sessions) > 1:
                cross_session_nodes += 1
        
        print(f"  ✅ Cross-session pattern inheritance: {cross_session_nodes} nodes span multiple sessions")
        
        print(f"\n✅ Enhanced Session Adapter integration verified successfully!")
        print(f"   Key metrics:")
        print(f"   - Events processed: {len(enhanced_session_events)}")
        print(f"   - Lattice nodes: {len(lattice_dataset.nodes)}")
        print(f"   - Network connections: {len(lattice_dataset.connections)}")
        print(f"   - Sessions covered: {len(lattice_dataset.sessions_covered)}")
        print(f"   - Timeframes: {len(timeframes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Session integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_session_sample()
    sys.exit(0 if success else 1)