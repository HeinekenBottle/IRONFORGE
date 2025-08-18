#!/usr/bin/env python3
"""
Test script to verify IRONFORGE integration fixes
"""

import sys
import traceback
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))

def test_tgat_import():
    """Test TGAT Discovery import with both class names"""
    print("🧠 Testing TGAT Discovery import...")
    
    try:
        # Test original import
        from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
        print("  ✅ IRONFORGEDiscovery import successful")
        
        # Test alias import
        from ironforge.learning.tgat_discovery import TGATDiscovery
        print("  ✅ TGATDiscovery alias import successful")
        
        # Test that they're the same
        assert IRONFORGEDiscovery is TGATDiscovery
        print("  ✅ Alias verification successful")
        
        # Test initialization
        discovery = IRONFORGEDiscovery(node_features=45, hidden_dim=128, out_dim=256)
        print("  ✅ IRONFORGEDiscovery initialization successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ TGAT import error: {e}")
        traceback.print_exc()
        return False

def test_lattice_mapper():
    """Test lattice mapper with both data formats"""
    print("\n🗺️  Testing Lattice Mapper...")
    
    try:
        from ironforge.analysis.broad_spectrum_archaeology import (
            ArchaeologicalEvent,
            EventType,
            HTFConfluenceStatus,
            LiquidityArchetype,
            RangeLevel,
            SessionPhase,
        )
        from ironforge.analysis.timeframe_lattice_mapper import (
            TimeframeLatticeMapper,
            TimeframeType,
        )
        
        # Initialize mapper
        mapper = TimeframeLatticeMapper(grid_resolution=20, min_node_events=1)
        print("  ✅ TimeframeLatticeMapper initialization successful")
        
        # Test with ArchaeologicalEvent objects
        mock_event = ArchaeologicalEvent(
            event_id="test_event_1",
            session_name="test_session",
            session_date="2025-08-15",
            timestamp="09:30:00",
            timeframe=TimeframeType.MINUTE_1,
            event_type=EventType.FVG_REDELIVERY,
            event_subtype="test",
            range_level=RangeLevel.SWEEP_ACCELERATION,
            liquidity_archetype=LiquidityArchetype.SESSION_LOW_SWEEP,
            session_phase=SessionPhase.OPENING,
            session_minute=30.0,
            relative_cycle_position=0.5,
            absolute_time_signature="test_sig",
            magnitude=1.0,
            duration_minutes=5.0,
            velocity_signature=0.5,
            significance_score=0.8,
            htf_confluence=HTFConfluenceStatus.ABSENT,
            htf_regime="test_regime",
            cross_session_inheritance=0.5,
            historical_matches=[],
            pattern_family="test_family",
            recurrence_rate=0.3,
            enhanced_features={},
            range_position_percent=50.0,
            structural_role="test_role",
            discovery_metadata={},
            confidence_score=0.8
        )
        
        # Test coordinate creation with objects
        coords = mapper._create_event_coordinates([mock_event])
        print("  ✅ Object-based coordinate creation successful")
        
        # Test with dictionary-based events
        dict_event = {
            'event_id': 'test_dict_event_1',
            'session_name': 'test_session_dict',
            'timeframe': '1m',
            'relative_cycle_position': 0.6,
            'significance_score': 0.7,
            'event_type': 'fvg_redelivery',
            'pattern_family': 'fvg_family',
            'liquidity_archetype': 'sweep_terminus',
            'structural_role': 'terminal_sweep'
        }
        
        # Test coordinate creation with dicts
        dict_coords = mapper._create_event_coordinates([dict_event])
        print("  ✅ Dictionary-based coordinate creation successful")
        
        # Test lattice mapping
        lattice_dataset = mapper.map_events_to_lattice([mock_event])
        print("  ✅ Lattice dataset creation successful")
        print(f"    Created {len(lattice_dataset.nodes)} nodes")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Lattice mapper error: {e}")
        traceback.print_exc()
        return False

def test_visualizer():
    """Test visualizer imports and initialization"""
    print("\n📊 Testing Visualizer...")
    
    try:
        from visualizations.lattice_visualizer import LatticeVisualizer, VisualizationConfig
        
        # Initialize visualizer
        config = VisualizationConfig()
        visualizer = LatticeVisualizer(config)
        print("  ✅ LatticeVisualizer initialization successful")
        
        # Check that methods exist with correct names
        assert hasattr(visualizer, 'create_temporal_heatmaps')
        print("  ✅ create_temporal_heatmaps method exists")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualizer error: {e}")
        traceback.print_exc()
        return False

def test_enhanced_session_integration():
    """Test integration with Enhanced Session Adapter data format"""
    print("\n🔗 Testing Enhanced Session Integration...")
    
    try:
        # Simulate Enhanced Session Adapter output format
        mock_enhanced_events = [
            {
                'event_id': 'enhanced_event_1',
                'session_name': '2025-08-05_PM_session',
                'timeframe': '1m',
                'timestamp': '14:35:00',
                'relative_cycle_position': 0.4,
                'significance_score': 0.82,
                'event_type': 'fvg_redelivery',
                'pattern_family': 'fvg_family',
                'liquidity_archetype': 'sweep_terminus',
                'structural_role': 'terminal_sweep',
                'session_minute': 35.0
            },
            {
                'event_id': 'enhanced_event_2', 
                'session_name': '2025-08-05_PM_session',
                'timeframe': '5m',
                'timestamp': '14:40:00',
                'relative_cycle_position': 0.45,
                'significance_score': 0.75,
                'event_type': 'expansion_phase',
                'pattern_family': 'expansion_family',
                'liquidity_archetype': 'breakout_catalyst',
                'structural_role': 'breakout',
                'session_minute': 40.0
            }
        ]
        
        from ironforge.analysis.timeframe_lattice_mapper import TimeframeLatticeMapper
        
        # Test lattice mapping with Enhanced Session format
        mapper = TimeframeLatticeMapper(grid_resolution=10, min_node_events=1)
        lattice_dataset = mapper.map_events_to_lattice(mock_enhanced_events)
        
        print("  ✅ Enhanced Session format processed successfully")
        print(f"    Created {len(lattice_dataset.nodes)} nodes from {len(mock_enhanced_events)} events")
        print(f"    Detected {len(lattice_dataset.hot_zones)} hot zones")
        
        # Verify nodes have expected properties
        for node_id, node in lattice_dataset.nodes.items():
            print(f"    Node {node_id}: {node.event_count} events, significance {node.average_significance:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced Session integration error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("🔧 IRONFORGE Integration Fixes Verification")
    print("=" * 50)
    
    tests = [
        test_tgat_import,
        test_lattice_mapper,
        test_visualizer,
        test_enhanced_session_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All integration fixes verified successfully!")
        return True
    else:
        print("❌ Some tests failed - check output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)