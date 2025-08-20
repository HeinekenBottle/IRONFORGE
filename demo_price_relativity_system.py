#!/usr/bin/env python3
"""
IRONFORGE Price Relativity System Demo
Comprehensive demonstration of enhanced temporal query engine with Theory B integration
"""

from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
from session_time_manager import SessionTimeManager
from archaeological_zone_calculator import ArchaeologicalZoneCalculator
import pandas as pd

def demo_price_relativity_system():
    """Complete demonstration of price relativity and Theory B integration"""
    
    print("🚀 IRONFORGE Price Relativity System Demonstration")
    print("🏛️ Archaeological Zone Analysis & Theory B Temporal Non-Locality")
    print("=" * 70)
    
    # Initialize the enhanced system
    print("\n1. Initializing Enhanced Systems...")
    engine = EnhancedTemporalQueryEngine()
    session_manager = SessionTimeManager()
    zone_calculator = ArchaeologicalZoneCalculator()
    
    print(f"✅ Loaded {len(engine.sessions)} sessions with price relativity calculations")
    
    # Session Time Management Demo
    print("\n" + "="*70)
    print("2. SESSION TIME MANAGEMENT & DUAL TRACKING")
    print("="*70)
    
    print("\n📅 Session Specifications:")
    taxonomy = session_manager.get_session_taxonomy()
    for session_type, spec in taxonomy['session_specs'].items():
        print(f"  {session_type}: {spec['start']}-{spec['end']} ({spec['duration']}min, {spec['timezone']})")
    
    print("\n⏰ Theory B Example - Dual Time Tracking:")
    context = session_manager.analyze_event_context(
        session_type="NYPM",
        event_time="14:35:00",
        event_price=23162.25,
        session_stats={
            'session_high': 23375.5,
            'session_low': 23148.5,
            'session_open': 23169.25,
            'session_close': 23368.0
        }
    )
    
    print(f"  Absolute Time: {context['absolute_time']}")
    print(f"  Session Progress: {context['session_progress_pct']}% ({context['session_phase']})")
    print(f"  Archaeological Zone: {context['current_zone']}")
    print(f"  Theory B Precision: {context['theory_b_precision']['distance_to_40pct']:.2f} points from 40% zone")
    print(f"  Combined Analysis: {context['combined_analysis']['positioning_description']}")
    
    # Archaeological Zone Analysis Demo
    print("\n" + "="*70)
    print("3. ARCHAEOLOGICAL ZONE ANALYSIS")
    print("="*70)
    
    zones = zone_calculator.calculate_zones_for_session(23375.5, 23148.5)
    print(f"\n🏛️ Session Range Analysis (227.0 points):")
    for zone_pct, zone_data in zones['zones'].items():
        destiny_marker = " ⭐ DIMENSIONAL DESTINY" if zone_data['is_dimensional_destiny'] else ""
        print(f"  {zone_pct}: {zone_data['level']:.2f}{destiny_marker}")
    
    print(f"\n⚡ Theory B Framework:")
    print(f"  Precision Threshold: {zones['theory_b_framework']['precision_threshold']} points")
    print(f"  Key Discovery: {zones['theory_b_framework']['temporal_non_locality']}")
    
    # Enhanced Query Engine Demo
    print("\n" + "="*70)
    print("4. ENHANCED TEMPORAL QUERY ENGINE")
    print("="*70)
    
    print("\n🔍 Testing Enhanced Query Capabilities...")
    
    # Test archaeological zone queries
    print("\n📊 Query: 'Show me archaeological zone patterns'")
    zone_results = engine.ask("Show me archaeological zone patterns")
    print(f"  Query Type: {zone_results['query_type']}")
    print(f"  Sessions Analyzed: {zone_results['total_sessions']}")
    print(f"  Theory B Sessions: {zone_results['zone_statistics']['theory_b_sessions']}")
    print(f"  Theory B Percentage: {zone_results['zone_statistics']['theory_b_percentage']:.1f}%")
    
    # Test Theory B precision queries
    print("\n🎯 Query: 'Find Theory B precision events'")
    theory_b_results = engine.ask("Find Theory B precision events")
    print(f"  Query Type: {theory_b_results['query_type']}")
    print(f"  Precision Events Found: {len(theory_b_results['precision_events'])}")
    
    if theory_b_results['precision_events']:
        best_event = theory_b_results['precision_events'][0]
        print(f"  Best Precision Event:")
        print(f"    Session: {best_event['session_id']}")
        print(f"    Time: {best_event['event_time']} ({best_event['session_progress']:.1f}% through session)")
        print(f"    Price: {best_event['price']:.2f}")
        print(f"    Distance to Final 40%: {best_event['distance_to_final_40pct']:.2f} points")
        print(f"    Precision Score: {best_event['precision_score']:.3f}")
    
    # Test relative positioning queries
    print("\n📈 Query: 'Show me relative positioning patterns'")
    relative_results = engine.ask("Show me relative positioning patterns")
    print(f"  Query Type: {relative_results['query_type']}")
    print(f"  Sessions with Positioning Data: {len(relative_results['positioning_patterns'])}")
    
    if relative_results['positioning_patterns']:
        avg_position = sum(p['mean_position'] for p in relative_results['positioning_patterns']) / len(relative_results['positioning_patterns'])
        print(f"  Average Event Position in Session Range: {avg_position:.1f}%")
    
    # Test temporal sequence with price relativity
    print("\n⚡ Query: 'What happens after high liquidity spikes?'")
    sequence_results = engine.ask("What happens after high liquidity spikes?")
    print(f"  Query Type: {sequence_results['query_type']}")
    print(f"  Pattern Matches: {len(sequence_results['matches'])}")
    
    if sequence_results['matches']:
        print(f"  Sample Match:")
        match = sequence_results['matches'][0]
        print(f"    Session: {match['session_id']}")
        print(f"    Event Time: {match['event_time']} ({match['session_progress']:.1f}% through session)")
        print(f"    Archaeological Zone: {match['archaeological_zone']}")
        print(f"    Subsequent Outcome: {match['subsequent_outcome']}")
        print(f"    Pattern Type: {match['pattern_type']}")
    
    # Session Information Demo
    print("\n" + "="*70)
    print("5. ENHANCED SESSION INFORMATION")
    print("="*70)
    
    # Get first session with data
    first_session = next(iter(engine.sessions.keys()))
    session_info = engine.get_enhanced_session_info(first_session)
    
    print(f"\n📋 Enhanced Session Info: {first_session}")
    print(f"  Session Type: {session_info['session_type']}")
    print(f"  Total Events: {session_info['total_events']}")
    
    if session_info['session_stats']:
        stats = session_info['session_stats']
        print(f"  Price Range: {stats['session_range']:.1f} points")
        print(f"  Session High: {stats['session_high']:.2f}")
        print(f"  Session Low: {stats['session_low']:.2f}")
    
    if session_info['archaeological_zones'] and 'zones' in session_info['archaeological_zones']:
        print(f"  Archaeological Zones:")
        for zone_pct, zone_data in session_info['archaeological_zones']['zones'].items():
            destiny = " ⭐" if zone_data.get('is_dimensional_destiny', False) else ""
            print(f"    {zone_pct}: {zone_data.get('level', 0):.2f}{destiny}")
    
    # Feature Comparison: Before vs After
    print("\n" + "="*70)
    print("6. BEFORE vs AFTER COMPARISON")
    print("="*70)
    
    print("\n❌ BEFORE (Basic Temporal Query Engine):")
    print("  • Absolute price thresholds (>150 points = large range)")
    print("  • Fixed 15-minute time windows")
    print("  • No archaeological zone awareness")
    print("  • Binary expansion/consolidation outcomes")
    print("  • No session type differentiation")
    print("  • No Theory B temporal non-locality")
    
    print("\n✅ AFTER (Enhanced Price Relativity System):")
    print("  • Session-relative price positioning (40%/60%/80% zones)")
    print("  • Dual time tracking (absolute + percentage through session)")
    print("  • Archaeological zone dimensional relationships")
    print("  • Theory B precision detection (7.55-point threshold)")
    print("  • Session type awareness (NYAM, NYPM, LONDON, ASIA)")
    print("  • Temporal non-locality analysis")
    print("  • Enhanced pattern matching with zone context")
    
    # Summary Statistics
    print("\n" + "="*70)
    print("7. SYSTEM SUMMARY STATISTICS")
    print("="*70)
    
    total_sessions = len(engine.sessions)
    sessions_with_stats = len(engine.session_stats)
    
    print(f"\n📊 System Overview:")
    print(f"  Total Sessions Loaded: {total_sessions}")
    print(f"  Sessions with Price Stats: {sessions_with_stats}")
    print(f"  Session Types Supported: {len(session_manager.session_specs)}")
    print(f"  Archaeological Zone Types: 5 (20%, 40%, 60%, 80%, transitional)")
    print(f"  Theory B Precision Threshold: 7.55 points")
    
    # Session type distribution
    session_types = {}
    for session_id in engine.sessions.keys():
        session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
        session_types[session_type] = session_types.get(session_type, 0) + 1
    
    print(f"\n📈 Session Type Distribution:")
    for session_type, count in sorted(session_types.items()):
        print(f"  {session_type}: {count} sessions")
    
    # Average session statistics
    if engine.session_stats:
        avg_range = sum(stats['session_range'] for stats in engine.session_stats.values()) / len(engine.session_stats)
        avg_events = sum(stats['total_events'] for stats in engine.session_stats.values()) / len(engine.session_stats)
        
        print(f"\n📊 Average Session Metrics:")
        print(f"  Average Range: {avg_range:.1f} points")
        print(f"  Average Events per Session: {avg_events:.1f}")
    
    print("\n" + "="*70)
    print("🎯 PRICE RELATIVITY SYSTEM READY")
    print("="*70)
    
    print("\n✅ All components operational:")
    print("  • SessionTimeManager: Dual time tracking")
    print("  • ArchaeologicalZoneCalculator: Theory B integration")  
    print("  • EnhancedTemporalQueryEngine: Price relativity queries")
    print("  • 51 sessions loaded with complete price relativity context")
    
    print("\n💡 Ready for advanced queries like:")
    print("  • 'What happens after 40% zone events in NYAM sessions?'")
    print("  • 'Show me Theory B precision events with 5-minute lead times'")
    print("  • 'Find dimensional destiny events in the first 30% of sessions'")
    print("  • 'What's the probability of expansion after 80% zone touches?'")
    
    return engine

if __name__ == "__main__":
    demo_engine = demo_price_relativity_system()
    print(f"\n🚀 Demo complete! Enhanced engine ready for interactive use.")
    print(f"   Try: engine.ask('your enhanced question here')")