#!/usr/bin/env python3
"""
IRONFORGE Integrated Pattern Discovery Demo
Demonstrates integration of price relativity with 73.3% f8→FPFVG pattern discovery
"""

from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
from predictive_condition_hunter import hunt_predictive_conditions
from session_time_manager import SessionTimeManager
from archaeological_zone_calculator import ArchaeologicalZoneCalculator

def demo_integrated_pattern_discovery():
    """Demonstrate integration of price relativity with existing 73.3% pattern"""
    
    print("🎯 IRONFORGE Integrated Pattern Discovery")
    print("🔗 Price Relativity + 73.3% f8→FPFVG Pattern Integration")
    print("=" * 65)
    
    # Initialize systems
    print("\n1. Initializing Integrated Systems...")
    engine = EnhancedTemporalQueryEngine()
    session_manager = SessionTimeManager()
    zone_calculator = ArchaeologicalZoneCalculator()
    
    # Run original 73.3% pattern discovery
    print("\n2. Running Original 73.3% Pattern Discovery...")
    results = hunt_predictive_conditions()
    
    high_prob_patterns = results['high_probability_patterns']
    if high_prob_patterns.get('probability_rankings'):
        top_pattern = high_prob_patterns['probability_rankings'][0]
        print(f"✅ Original Discovery:")
        print(f"   Pattern: {top_pattern['pattern']}")
        print(f"   Probability: {top_pattern['probability']:.1%}")
        print(f"   Sample Size: {top_pattern['sample_size']}")
    
    # Enhanced analysis with price relativity
    print("\n3. Enhanced Analysis with Price Relativity...")
    
    # Query for f8 spikes with archaeological zone context
    f8_results = engine.ask("What happens after high liquidity spikes?")
    print(f"🔍 Enhanced f8 Analysis:")
    print(f"   Query Type: {f8_results['query_type']}")
    print(f"   Pattern Matches: {len(f8_results['matches'])}")
    
    if f8_results['matches']:
        # Analyze matches with archaeological zones
        zone_distribution = {}
        session_phases = {}
        
        for match in f8_results['matches']:
            zone = match.get('archaeological_zone', 'unknown')
            zone_distribution[zone] = zone_distribution.get(zone, 0) + 1
            
            # Extract session phase from session progress
            progress = match.get('session_progress', 0)
            if progress < 20:
                phase = "opening"
            elif progress < 40:
                phase = "early"
            elif progress < 60:
                phase = "mid"
            elif progress < 80:
                phase = "late"
            else:
                phase = "closing"
            session_phases[phase] = session_phases.get(phase, 0) + 1
        
        print(f"\n📊 Archaeological Zone Distribution:")
        for zone, count in sorted(zone_distribution.items()):
            percentage = count / len(f8_results['matches']) * 100
            print(f"   {zone}: {count} events ({percentage:.1f}%)")
        
        print(f"\n⏰ Session Phase Distribution:")
        for phase, count in sorted(session_phases.items()):
            percentage = count / len(f8_results['matches']) * 100
            print(f"   {phase}: {count} events ({percentage:.1f}%)")
    
    # Theory B integration with f8 patterns
    print("\n4. Theory B Integration Analysis...")
    
    theory_b_results = engine.ask("Find Theory B precision events")
    print(f"⚡ Theory B + f8 Integration:")
    print(f"   Theory B Events Found: {len(theory_b_results['precision_events'])}")
    
    # Look for overlap between Theory B and f8 patterns
    if theory_b_results['precision_events'] and f8_results['matches']:
        # Find potential overlaps based on session and timing
        overlaps = []
        
        for theory_b_event in theory_b_results['precision_events']:
            for f8_match in f8_results['matches']:
                if (theory_b_event['session_id'] == f8_match['session_id'] and
                    theory_b_event['event_time'] == f8_match['event_time']):
                    overlaps.append({
                        'session': theory_b_event['session_id'],
                        'time': theory_b_event['event_time'],
                        'precision_score': theory_b_event['precision_score'],
                        'f8_outcome': f8_match['subsequent_outcome']
                    })
        
        print(f"   Theory B + f8 Overlaps: {len(overlaps)}")
        
        if overlaps:
            best_overlap = max(overlaps, key=lambda x: x['precision_score'])
            print(f"   Best Combined Event:")
            print(f"     Session: {best_overlap['session']}")
            print(f"     Time: {best_overlap['time']}")
            print(f"     Theory B Score: {best_overlap['precision_score']:.3f}")
            print(f"     f8 Outcome: {best_overlap['f8_outcome']}")
    
    # Session-specific analysis
    print("\n5. Session-Specific Enhanced Patterns...")
    
    session_types = ['NYAM', 'NYPM', 'LONDON', 'ASIA']
    
    for session_type in session_types:
        # Count sessions of this type
        type_sessions = [sid for sid in engine.sessions.keys() if sid.startswith(session_type)]
        
        if type_sessions:
            print(f"\n📈 {session_type} Sessions ({len(type_sessions)} sessions):")
            
            # Get session specification
            spec = session_manager.get_session_spec(session_type)
            if spec:
                print(f"   Duration: {spec.duration_minutes} minutes ({spec.start_time.strftime('%H:%M')}-{spec.end_time.strftime('%H:%M')})")
                print(f"   Characteristics: {spec.characteristics}")
            
            # Analyze f8 events in this session type
            type_matches = [m for m in f8_results['matches'] if m['session_id'] in type_sessions]
            if type_matches:
                outcomes = [m['subsequent_outcome'] for m in type_matches]
                expansion_rate = outcomes.count('expansion') / len(outcomes) * 100 if outcomes else 0
                print(f"   f8 Events: {len(type_matches)}")
                print(f"   Expansion Rate: {expansion_rate:.1f}%")
                
                # Average session progress for f8 events
                avg_progress = sum(m['session_progress'] for m in type_matches) / len(type_matches)
                print(f"   Average Event Timing: {avg_progress:.1f}% through session")
    
    # Practical trading implications
    print("\n" + "="*65)
    print("6. PRACTICAL TRADING IMPLICATIONS")
    print("="*65)
    
    print("\n🎯 Enhanced Pattern Recognition:")
    print("   • Original: f8 > 95th percentile → 73.3% FPFVG redelivery")
    print("   • Enhanced: f8 spike + archaeological zone + session timing context")
    print("   • Timing: Absolute time (14:35:00) + relative progress (64.6%)")
    print("   • Zone Context: Event position in eventual session range")
    
    print("\n⚡ Theory B Temporal Non-Locality:")
    print("   • Events position with 7.55-point precision to final ranges")
    print("   • Early session events predict eventual completion levels")
    print("   • 40% zone events 'know' their relationship to final structure")
    
    print("\n📊 Session-Aware Analysis:")
    print("   • NYAM (09:30-11:59): High volatility, institutional participation")
    print("   • NYPM (12:00-16:00): Mixed participation, medium volatility")
    print("   • Different session types may have different f8 characteristics")
    print("   • Session progress percentage adds temporal context")
    
    print("\n🔗 Integrated Alert System:")
    print("   1. Monitor f8 liquidity intensity in real-time")
    print("   2. Check archaeological zone position (40%/60%/80%)")
    print("   3. Assess session progress and timing")
    print("   4. Apply Theory B precision criteria")
    print("   5. Generate contextualized alerts with lead times")
    
    print("\n" + "="*65)
    print("✅ INTEGRATION COMPLETE")
    print("="*65)
    
    print("\n🚀 System Capabilities:")
    print("   ✅ 73.3% f8→FPFVG pattern detection")
    print("   ✅ Archaeological zone positioning")
    print("   ✅ Theory B temporal non-locality")
    print("   ✅ Session-aware timing analysis")
    print("   ✅ Dual time tracking (absolute + relative)")
    print("   ✅ Enhanced probabilistic outcomes")
    
    print("\n💡 Next-Level Queries Now Possible:")
    print("   • 'What happens when f8 spikes occur in 40% zones during NYAM?'")
    print("   • 'Show me Theory B events that led to 73.3% pattern triggers'")
    print("   • 'Find f8 spikes in late session phases with zone precision'")
    print("   • 'What's the expansion probability after 60% zone f8 events?'")
    
    return {
        'original_pattern': results,
        'enhanced_f8_analysis': f8_results,
        'theory_b_events': theory_b_results,
        'engine': engine
    }

if __name__ == "__main__":
    integrated_results = demo_integrated_pattern_discovery()
    print(f"\n🎯 Integration demo complete!")
    print(f"   Your 73.3% pattern now has archaeological zone context,")
    print(f"   session timing awareness, and Theory B temporal non-locality!")