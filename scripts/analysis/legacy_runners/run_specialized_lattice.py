#!/usr/bin/env python3
"""
🧩 IRONFORGE Specialized Lattice Execution
==========================================

STEP 3: Builds specialized lattice views based on terrain analysis findings.
Priority 1: NY PM Archaeological Belt with Theory B validation.

Usage:
    python run_specialized_lattice.py
"""

import logging
import sys
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

from ironforge.analysis.specialized_lattice_builder import SpecializedLatticeBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Execute specialized lattice building for archaeological deep dive"""
    
    print("🧩 IRONFORGE Specialized Lattice Construction")
    print("=" * 55)
    print("STEP 3: Archaeological Deep Dive")
    print("Priority 1: NY PM Belt (14:35-38) with Theory B validation")
    print()
    
    try:
        # Initialize specialized lattice builder
        builder = SpecializedLatticeBuilder()
        
        # Build NY PM Archaeological Belt lattice
        logger.info("Building NY PM Archaeological Belt specialized lattice...")
        belt_lattice = builder.build_ny_pm_archaeological_belt()
        
        # Display results summary
        if 'error' not in belt_lattice:
            print("✅ NY PM ARCHAEOLOGICAL BELT LATTICE COMPLETE")
            print("=" * 50)
            
            # Extract metadata
            metadata = belt_lattice.get('lattice_metadata', {})
            print(f"🏛️  Focus Timeframe: {metadata.get('focus_timeframe', 'unknown')}")
            print(f"🔍 Resolution: {metadata.get('resolution', 'unknown')}")
            print(f"📊 Sessions Analyzed: {metadata.get('sessions_analyzed', 0)}")
            print(f"⚗️  Theory B Validation: {metadata.get('theory_b_validation', False)}")
            print()
            
            # Belt events summary
            belt_events = belt_lattice.get('belt_events', [])
            print(f"🎯 Total Belt Events: {len(belt_events)}")
            
            # Event type distribution
            event_types = {}
            for event in belt_events:
                event_type = event.get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            if event_types:
                print("📊 Event Type Distribution:")
                for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {event_type:>20}: {count:>3} events")
                print()
            
            # Theory B analysis results
            theory_b = belt_lattice.get('theory_b_analysis', {})
            if theory_b:
                print("🔬 THEORY B ANALYSIS RESULTS")
                print("-" * 30)
                
                total_sessions = theory_b.get('total_sessions', 0)
                total_events = theory_b.get('total_belt_events', 0)
                print(f"Sessions with Theory B data: {total_sessions}")
                print(f"Belt events analyzed: {total_events}")
                
                # Dimensional level preferences
                level_prefs = theory_b.get('dimensional_level_preferences', {})
                if level_prefs:
                    print("Dimensional level targeting:")
                    total_targeted = sum(level_prefs.values())
                    for level, count in sorted(level_prefs.items(), key=lambda x: x[1], reverse=True):
                        pct = (count / total_targeted * 100) if total_targeted > 0 else 0
                        level_clean = level.replace('_', ' ').title()
                        print(f"  {level_clean:>12}: {count:>3} events ({pct:>5.1f}%)")
                
                # Precision statistics
                precision = theory_b.get('precision_statistics', {})
                if precision:
                    print("Dimensional precision:")
                    avg_precision = precision.get('avg_dimensional_precision', 1.0)
                    best_precision = precision.get('best_dimensional_precision', 1.0)
                    avg_40_distance = precision.get('avg_distance_to_40_percent', 999)
                    best_40_distance = precision.get('best_distance_to_40_percent', 999)
                    
                    print(f"  Average precision: {avg_precision:.3f}")
                    print(f"  Best precision: {best_precision:.3f}")
                    print(f"  Avg distance to 40%: {avg_40_distance:.1f} points")
                    print(f"  Best distance to 40%: {best_40_distance:.1f} points")
                
                # Theory B scores
                scores = theory_b.get('theory_b_scores', {})
                if scores:
                    print("Theory B validation scores:")
                    avg_score = scores.get('avg_score', 0)
                    max_score = scores.get('max_score', 0)
                    high_score_events = scores.get('high_score_events', 0)
                    total_scored = scores.get('total_events', 0)
                    high_score_pct = (high_score_events / total_scored * 100) if total_scored > 0 else 0
                    
                    print(f"  Average Theory B score: {avg_score:.3f}")
                    print(f"  Maximum score achieved: {max_score:.3f}")
                    print(f"  High-score events (>0.8): {high_score_events} ({high_score_pct:.1f}%)")
                print()
            
            # Dimensional relationships
            dimensional = belt_lattice.get('dimensional_relationships', {})
            if dimensional:
                print("🌐 DIMENSIONAL RELATIONSHIPS")
                print("-" * 28)
                
                temporal = dimensional.get('temporal_patterns', {})
                if temporal:
                    most_active = temporal.get('most_active_minute', 'unknown')
                    print(f"Most active minute: 14:{most_active}")
                
                price_clustering = dimensional.get('price_clustering', {})
                if price_clustering:
                    price_range = price_clustering.get('price_range', 0)
                    unique_levels = price_clustering.get('unique_price_levels', 0)
                    print(f"Price range span: {price_range:.1f} points")
                    print(f"Unique price levels: {unique_levels}")
                print()
            
            # Archaeological zones
            zones = belt_lattice.get('archaeological_zones', {})
            if zones:
                zone_id = zones.get('zone_identification', {})
                zone_chars = zones.get('zone_characteristics', {})
                
                if zone_chars:
                    print("🏛️  ARCHAEOLOGICAL ZONES")
                    print("-" * 24)
                    for zone, characteristics in zone_chars.items():
                        event_count = characteristics.get('event_count', 0)
                        sessions = characteristics.get('sessions_involved', 0)
                        persistent = characteristics.get('cross_session_persistence', False)
                        avg_sig = characteristics.get('avg_significance', 0)
                        
                        zone_clean = zone.replace('_', ' ').title()
                        persistence_indicator = " ✓" if persistent else ""
                        print(f"  {zone_clean}{persistence_indicator}")
                        print(f"    Events: {event_count}, Sessions: {sessions}")
                        print(f"    Avg significance: {avg_sig:.3f}")
                    print()
            
            # Discovery insights
            insights = belt_lattice.get('discovery_insights', {})
            if insights:
                print("🔍 DISCOVERY INSIGHTS")
                print("-" * 20)
                
                # Theory B validation
                theory_b_val = insights.get('theory_b_validation', {})
                if theory_b_val:
                    status = theory_b_val.get('validation_status', 'unknown')
                    precision = theory_b_val.get('avg_dimensional_precision', 1.0)
                    best_distance = theory_b_val.get('best_40_percent_distance', 999)
                    
                    print(f"Theory B validation: {status}")
                    print(f"Dimensional precision: {precision:.3f}")
                    print(f"Best 40% zone distance: {best_distance:.1f} points")
                
                # Temporal non-locality
                temporal_nl = insights.get('temporal_non_locality', {})
                if temporal_nl:
                    evidence = temporal_nl.get('evidence_strength', 'unknown')
                    clustering = temporal_nl.get('clustering_detected', False)
                    
                    print(f"Temporal non-locality evidence: {evidence}")
                    print(f"Price clustering detected: {clustering}")
                
                # Archaeological patterns
                arch_patterns = insights.get('archaeological_patterns', {})
                if arch_patterns:
                    persistent = arch_patterns.get('persistent_zones_detected', 0)
                    reproducible = arch_patterns.get('cross_session_reproducibility', False)
                    
                    print(f"Persistent archaeological zones: {persistent}")
                    print(f"Cross-session reproducibility: {reproducible}")
                print()
                
                # Recommendations
                recommendations = insights.get('discovery_recommendations', [])
                if recommendations:
                    print("🚀 DISCOVERY RECOMMENDATIONS")
                    print("-" * 28)
                    for i, rec in enumerate(recommendations, 1):
                        priority = rec.get('priority', 'unknown')
                        rec_type = rec.get('type', 'unknown')
                        desc = rec.get('description', 'No description')
                        action = rec.get('action', 'No action specified')
                        
                        print(f"  {i}. [{priority}] {desc}")
                        print(f"     Type: {rec_type}")
                        print(f"     Action: {action}")
                    print()
            
            # Belt statistics
            stats = belt_lattice.get('belt_statistics', {})
            if stats:
                coverage = stats.get('belt_coverage', {})
                if coverage:
                    sessions_with_events = coverage.get('sessions_with_belt_events', 0)
                    avg_events = coverage.get('avg_events_per_session', 0)
                    
                    print("📈 BELT STATISTICS")
                    print("-" * 17)
                    print(f"Sessions with belt events: {sessions_with_events}")
                    print(f"Average events per session: {avg_events:.1f}")
                    print()
            
            print("🏛️  NY PM Archaeological Belt analysis complete")
            print("Ready for FPFVG redelivery network analysis")
            
        else:
            print("❌ NY PM BELT LATTICE BUILD FAILED")
            print(f"Error: {belt_lattice.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Specialized lattice execution failed: {e}")
        print(f"❌ Execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)