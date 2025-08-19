#!/usr/bin/env python3
"""
📈 IRONFORGE Weekly→Daily Liquidity Sweep Cascade Lattice Execution
==================================================================

Macro-Level Cascade Pattern Discovery
Maps higher timeframe (Weekly→Daily) liquidity sweep cascade patterns across session networks.

Key Focus:
- Weekly HTF liquidity formation → Daily cascade propagation
- Cross-session sweep relationship mapping
- HTF influence transmission pattern analysis
- Macro-to-micro cascade timing validation

Usage:
    python run_weekly_daily_cascade_lattice.py
"""

import logging
import sys
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

from ironforge.analysis.weekly_daily_sweep_cascade_lattice import WeeklyDailySweepCascadeLattice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Execute Weekly→Daily Liquidity Sweep Cascade Lattice analysis"""
    
    print("📈 IRONFORGE Weekly→Daily Liquidity Sweep Cascade Lattice")
    print("=" * 70)
    print("Macro-Level Cascade Discovery: Weekly HTF → Daily session cascades")
    print("Focus: HTF influence transmission and cross-session propagation")
    print()
    
    try:
        # Initialize Weekly→Daily cascade lattice builder
        builder = WeeklyDailySweepCascadeLattice()
        
        # Build Weekly→Daily cascade lattice
        logger.info("Building Weekly→Daily Liquidity Sweep Cascade Lattice...")
        cascade_lattice = builder.build_weekly_daily_cascade_lattice()
        
        # Display results summary
        if 'error' not in cascade_lattice:
            print("✅ WEEKLY→DAILY CASCADE LATTICE COMPLETE")
            print("=" * 50)
            
            # Extract metadata
            metadata = cascade_lattice.get('lattice_metadata', {})
            print(f"📈 Focus: {metadata.get('focus', 'unknown')}")
            print(f"📊 Sessions Analyzed: {metadata.get('sessions_analyzed', 0)}")
            print(f"🕰️  HTF Timeframes: {', '.join(metadata.get('htf_timeframes', []))}")
            print(f"📅 Session Timeframes: {', '.join(metadata.get('session_timeframes', []))}")
            print(f"🔍 Detection Window: {metadata.get('cascade_detection_window_days', 0)} days")
            print()
            
            # Weekly HTF Liquidity Events Summary
            weekly_events = cascade_lattice.get('weekly_liquidity_events', [])
            print(f"🗓️  Weekly HTF Liquidity Events: {len(weekly_events)}")
            
            if weekly_events:
                # Event type distribution
                event_types = {}
                for event in weekly_events:
                    event_type = event.get('liquidity_type', 'unknown')
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                print("Weekly event type distribution:")
                for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {event_type.replace('_', ' ').title():>25}: {count:>3} events")
                print()
            
            # Daily Session Sweep Events Summary
            daily_sweeps = cascade_lattice.get('daily_sweep_events', [])
            print(f"📉 Daily Session Sweep Events: {len(daily_sweeps)}")
            
            if daily_sweeps:
                # Session distribution
                session_distribution = {}
                for sweep in daily_sweeps:
                    session_tf = sweep.get('session_timeframe', 'unknown')
                    session_distribution[session_tf] = session_distribution.get(session_tf, 0) + 1
                
                print("Session sweep distribution:")
                for session_tf, count in sorted(session_distribution.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {session_tf:>10}: {count:>3} sweeps")
                print()
            
            # Cascade Relationships Analysis
            cascade_relationships = cascade_lattice.get('cascade_relationships', [])
            print(f"🔗 Weekly→Daily Cascade Relationships: {len(cascade_relationships)}")
            
            if cascade_relationships:
                # Cascade strength distribution
                strength_categories = {'very_strong': 0, 'strong': 0, 'moderate': 0, 'weak': 0}
                for cascade in cascade_relationships:
                    strength = cascade.get('cascade_strength', 0)
                    if strength >= 0.8:
                        strength_categories['very_strong'] += 1
                    elif strength >= 0.6:
                        strength_categories['strong'] += 1
                    elif strength >= 0.4:
                        strength_categories['moderate'] += 1
                    else:
                        strength_categories['weak'] += 1
                
                print("Cascade strength distribution:")
                for category, count in strength_categories.items():
                    if count > 0:
                        print(f"  {category.replace('_', ' ').title():>15}: {count:>3} cascades")
                
                # Average cascade strength
                if cascade_relationships:
                    avg_strength = sum(c.get('cascade_strength', 0) for c in cascade_relationships) / len(cascade_relationships)
                    print(f"Average cascade strength: {avg_strength:.3f}")
                print()
            
            # Cascade Timing Analysis
            timing_analysis = cascade_lattice.get('cascade_timing_analysis', {})
            if timing_analysis:
                print("⏰ CASCADE TIMING ANALYSIS")
                print("-" * 26)
                
                total_analyzed = timing_analysis.get('total_cascades_analyzed', 0)
                avg_delay = timing_analysis.get('average_transmission_delay', 0)
                
                print(f"Total cascades analyzed: {total_analyzed}")
                print(f"Average transmission delay: {avg_delay:.1f} hours")
                
                # Timing distribution
                timing_dist = timing_analysis.get('timing_distribution', {})
                if timing_dist:
                    print("Timing category distribution:")
                    for category, count in timing_dist.items():
                        print(f"  {category.replace('_', ' ').title():>15}: {count:>3} cascades")
                
                # Timing precision
                precision_analysis = timing_analysis.get('timing_precision_analysis', {})
                if precision_analysis:
                    print("Timing precision analysis:")
                    for precision, count in precision_analysis.items():
                        print(f"  {precision.title():>10}: {count:>3} cascades")
                print()
            
            # Cross-Session Sweep Propagation Analysis
            propagation = cascade_lattice.get('sweep_propagation_analysis', {})
            if propagation:
                print("🌐 CROSS-SESSION SWEEP PROPAGATION")
                print("-" * 34)
                
                session_dist = propagation.get('session_sweep_distribution', {})
                if session_dist:
                    print("Session sweep distribution:")
                    for session, count in sorted(session_dist.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {session:>8}: {count:>3} sweeps")
                
                # Propagation chains
                propagation_chains = propagation.get('cross_session_propagation_chains', [])
                print(f"Cross-session propagation chains: {len(propagation_chains)}")
                
                if propagation_chains:
                    print("Top propagation chains:")
                    for i, chain in enumerate(propagation_chains[:3], 1):
                        price = chain.get('price_level', 0)
                        sweep_count = chain.get('sweep_count', 0)
                        sessions = chain.get('sessions_involved', [])
                        strength = chain.get('chain_strength', 0)
                        
                        print(f"  {i}. Price {price:.1f} - {sweep_count} sweeps across {len(sessions)} sessions (strength: {strength})")
                print()
            
            # HTF Transmission Analysis
            htf_transmission = cascade_lattice.get('htf_transmission_analysis', {})
            if htf_transmission:
                print("📡 HTF INFLUENCE TRANSMISSION")
                print("-" * 29)
                
                total_weekly = htf_transmission.get('total_weekly_events', 0)
                print(f"Total Weekly events analyzed: {total_weekly}")
                
                # Transmission efficiency by type
                efficiency_by_type = htf_transmission.get('transmission_efficiency_by_type', {})
                if efficiency_by_type:
                    print("Transmission efficiency by Weekly event type:")
                    for event_type, data in efficiency_by_type.items():
                        avg_eff = data.get('average_efficiency', 0)
                        count = data.get('event_count', 0)
                        print(f"  {event_type.replace('_', ' ').title():>25}: {avg_eff:.3f} ({count} events)")
                print()
            
            # Cascade Network Topology
            topology = cascade_lattice.get('cascade_network_topology', {})
            if topology:
                print("🕸️  CASCADE NETWORK TOPOLOGY")
                print("-" * 28)
                
                total_relationships = topology.get('total_cascade_relationships', 0)
                network_efficiency = topology.get('network_efficiency', 0)
                
                print(f"Total cascade relationships: {total_relationships}")
                print(f"Network efficiency: {network_efficiency:.3f}")
                
                # Strength distribution
                strength_dist = topology.get('cascade_strength_distribution', {})
                if strength_dist:
                    print("Network strength distribution:")
                    for category, count in strength_dist.items():
                        if count > 0:
                            print(f"  {category.replace('_', ' ').title():>15}: {count:>3} relationships")
                print()
            
            # Predictive Patterns
            predictive = cascade_lattice.get('predictive_patterns', {})
            if predictive:
                print("🔮 MACRO CASCADE PREDICTIVE PATTERNS")
                print("-" * 36)
                
                prediction_rules = predictive.get('cascade_prediction_rules', [])
                if prediction_rules:
                    print("Cascade prediction rules identified:")
                    for i, rule in enumerate(prediction_rules, 1):
                        rule_type = rule.get('rule_type', 'unknown')
                        pattern_count = rule.get('pattern_count', 0)
                        avg_strength = rule.get('average_strength', 0)
                        confidence = rule.get('prediction_confidence', 'unknown')
                        
                        print(f"  {i}. {rule_type.replace('_', ' ').title()}")
                        print(f"     Patterns: {pattern_count}, Strength: {avg_strength:.3f}, Confidence: {confidence}")
                print()
            
            # Discovery Insights
            insights = cascade_lattice.get('discovery_insights', {})
            if insights:
                print("🔍 DISCOVERY INSIGHTS")
                print("-" * 20)
                
                # Cascade mapping summary
                mapping_summary = insights.get('cascade_mapping_summary', {})
                if mapping_summary:
                    detection_success = mapping_summary.get('cascade_detection_success', 'unknown')
                    cascade_ratio = mapping_summary.get('cascade_ratio', 0)
                    mapped_cascades = mapping_summary.get('mapped_cascades', 0)
                    
                    print(f"Cascade detection success: {detection_success}")
                    print(f"Cascade mapping ratio: {cascade_ratio:.3f}")
                    print(f"Total mapped cascades: {mapped_cascades}")
                
                # HTF transmission assessment
                htf_assessment = insights.get('htf_transmission_assessment', {})
                if htf_assessment:
                    influence_strength = htf_assessment.get('htf_influence_strength', 'unknown')
                    print(f"HTF influence strength: {influence_strength}")
                
                # Cross-session propagation insights
                propagation_insights = insights.get('cross_session_propagation_insights', {})
                if propagation_insights:
                    propagation_strength = propagation_insights.get('propagation_strength', 'unknown')
                    chains_detected = propagation_insights.get('propagation_chains_detected', 0)
                    print(f"Cross-session propagation: {propagation_strength}")
                    print(f"Propagation chains detected: {chains_detected}")
                print()
                
                # Recommendations
                recommendations = insights.get('discovery_recommendations', [])
                if recommendations:
                    print("🚀 DISCOVERY RECOMMENDATIONS")
                    print("-" * 28)
                    for i, rec in enumerate(recommendations, 1):
                        priority = rec.get('priority', 'unknown')
                        rec_type = rec.get('type', 'unknown')
                        description = rec.get('description', 'No description')
                        action = rec.get('action', 'No action specified')
                        
                        print(f"  {i}. [{priority}] {description}")
                        print(f"     Type: {rec_type}")
                        print(f"     Action: {action}")
                    print()
            
            print("📈 Weekly→Daily Liquidity Sweep Cascade Lattice analysis complete")
            print("Macro cascade patterns mapped for HTF→Session transmission analysis")
            print("Comprehensive lattice framework now complete: Global → Terrain → Specialized → FVG → Cascades")
            
        else:
            print("❌ WEEKLY→DAILY CASCADE LATTICE BUILD FAILED")
            print(f"Error: {cascade_lattice.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Weekly→Daily cascade lattice execution failed: {e}")
        print(f"❌ Execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)