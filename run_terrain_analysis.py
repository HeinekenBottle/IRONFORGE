#!/usr/bin/env python3
"""
🔍 IRONFORGE Terrain Analysis Execution
======================================

STEP 2: Analyzes the global lattice terrain to identify hot zones and cascade patterns.
Based on the successful global lattice build results.

Usage:
    python run_terrain_analysis.py
"""

import logging
import sys
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

from ironforge.analysis.lattice_terrain_analyzer import LatticeTerrainAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Execute terrain analysis for hot zones and cascades"""
    
    print("🔍 IRONFORGE Lattice Terrain Analysis")
    print("=" * 50)
    print("STEP 2: Hot Zone & Cascade Identification")
    print("Based on: 57 sessions, 249 hot zones, 10,568 cascades, 12,546 bridges")
    print()
    
    try:
        # Initialize terrain analyzer
        analyzer = LatticeTerrainAnalyzer()
        
        # Execute terrain analysis
        logger.info("Starting terrain analysis for hot zones and cascades...")
        terrain_analysis = analyzer.analyze_terrain_from_log()
        
        # Display results summary
        if 'error' not in terrain_analysis:
            print("✅ TERRAIN ANALYSIS COMPLETE")
            print("=" * 35)
            
            # Global metrics
            metrics = terrain_analysis.get('global_metrics', {})
            print(f"📊 Sessions Analyzed: {metrics.get('sessions_processed', 0)}")
            print(f"🔥 Hot Zones: {metrics.get('hot_zones_identified', 0)}")
            print(f"📈 Vertical Cascades: {metrics.get('vertical_cascades', 0):,}")
            print(f"🌉 Bridge Nodes: {metrics.get('bridge_nodes', 0):,}")
            print()
            
            # Hot zone analysis
            hot_zone_analysis = terrain_analysis.get('hot_zone_analysis', {})
            if hot_zone_analysis:
                print("🔥 HOT ZONE DISTRIBUTION")
                print("-" * 24)
                expected = hot_zone_analysis.get('expected_characteristics', {})
                for category, count in expected.items():
                    category_clean = category.replace('_', ' ').title()
                    print(f"  {category_clean:>25}: {count:>3}")
                print()
                
                # Hottest areas
                hottest = hot_zone_analysis.get('predicted_hottest_areas', {})
                print("🏆 PREDICTED HOTTEST AREAS")
                print("-" * 27)
                for area_id, details in hottest.items():
                    area_name = area_id.replace('_', ' ').title()
                    enrichment = details.get('enrichment_category', 'unknown')
                    timeframes = ', '.join(details.get('timeframes', []))
                    print(f"  {area_name}")
                    print(f"    Enrichment: {enrichment}")
                    print(f"    Timeframes: {timeframes}")
                    print(f"    Significance: {details.get('archaeological_significance', 'unknown')}")
                    print()
            
            # Cascade analysis
            cascade_analysis = terrain_analysis.get('cascade_analysis', {})
            if cascade_analysis:
                print("📈 VERTICAL CASCADE PATTERNS")
                print("-" * 28)
                patterns = cascade_analysis.get('cascade_patterns', {})
                total_cascades = sum(patterns.values())
                for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        pct = (count / total_cascades * 100) if total_cascades > 0 else 0
                        print(f"  {pattern:>20}: {count:>4} ({pct:>5.1f}%)")
                print()
                
                # Strongest cascade types
                strongest = cascade_analysis.get('strongest_cascade_types', {})
                print("💪 STRONGEST CASCADE TYPES")
                print("-" * 26)
                for cascade_type, details in strongest.items():
                    cascade_name = cascade_type.replace('_', ' ').title()
                    pattern = details.get('pattern', 'unknown')
                    count = details.get('occurrences', 0)
                    characteristic = details.get('characteristic', 'unknown')
                    print(f"  {cascade_name}")
                    print(f"    Pattern: {pattern}")
                    print(f"    Count: {count}")
                    print(f"    Type: {characteristic}")
                    print()
            
            # Bridge node analysis
            bridge_analysis = terrain_analysis.get('bridge_node_analysis', {})
            if bridge_analysis:
                print("🌉 BRIDGE NODE DISTRIBUTION")
                print("-" * 27)
                bridge_types = bridge_analysis.get('bridge_types', {})
                total_bridges = sum(bridge_types.values())
                for bridge_type, count in sorted(bridge_types.items(), key=lambda x: x[1], reverse=True):
                    bridge_name = bridge_type.replace('_', ' ').title()
                    pct = (count / total_bridges * 100) if total_bridges > 0 else 0
                    print(f"  {bridge_name:>25}: {count:>4} ({pct:>5.1f}%)")
                print()
            
            # Candidate areas
            candidates = terrain_analysis.get('candidate_areas', [])
            if candidates:
                print("🎯 CANDIDATE AREAS FOR DEEPER ANALYSIS")
                print("-" * 38)
                for i, candidate in enumerate(candidates, 1):
                    area_id = candidate.get('area_id', 'unknown')
                    priority = candidate.get('priority', 'unknown')
                    description = candidate.get('description', 'No description')
                    timeframes = ', '.join(candidate.get('timeframes', []))
                    
                    area_name = area_id.replace('_', ' ').title()
                    print(f"  {i}. [{priority}] {area_name}")
                    print(f"     {description}")
                    print(f"     Timeframes: {timeframes}")
                    
                    characteristics = candidate.get('key_characteristics', [])
                    if characteristics:
                        print("     Key characteristics:")
                        for char in characteristics[:2]:  # Show top 2
                            print(f"       - {char}")
                    
                    discovery_potential = candidate.get('discovery_potential', 'unknown')
                    print(f"     Discovery potential: {discovery_potential}")
                    print()
            
            # Discovery priorities
            priorities = terrain_analysis.get('discovery_priorities', [])
            if priorities:
                print("🚀 DISCOVERY PRIORITIES")
                print("-" * 20)
                for priority in priorities:
                    rank = priority.get('priority_rank', 0)
                    focus = priority.get('focus_area', 'unknown')
                    rationale = priority.get('rationale', 'unknown')
                    
                    print(f"  {rank}. {focus}")
                    print(f"     Rationale: {rationale}")
                    
                    actions = priority.get('immediate_actions', [])
                    if actions:
                        print("     Immediate actions:")
                        for action in actions[:2]:  # Show top 2
                            print(f"       - {action}")
                    print()
            
            print("🔍 Terrain analysis complete - ready for specialized lattice construction")
            print("Next step: Build specialized lattice views for candidate areas")
            
        else:
            print("❌ TERRAIN ANALYSIS FAILED")
            print(f"Error: {terrain_analysis.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Terrain analysis execution failed: {e}")
        print(f"❌ Execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)