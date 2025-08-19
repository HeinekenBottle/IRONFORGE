#!/usr/bin/env python3
"""
🌐 IRONFORGE Global Lattice Execution
====================================

Executes the comprehensive Monthly→1m global lattice build across all enhanced sessions.
This is STEP 1 of the discovery framework.

Usage:
    python run_global_lattice.py
"""

import logging
import sys
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

from ironforge.analysis.global_lattice_builder import GlobalLatticeBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Execute global lattice build"""
    
    print("🌐 IRONFORGE Global Lattice Build - Monthly→1m")
    print("=" * 60)
    print("Scope: All enhanced sessions")
    print("Purpose: Global terrain mapping for pattern discovery")
    print()
    
    try:
        # Initialize global lattice builder
        builder = GlobalLatticeBuilder()
        
        # Execute comprehensive lattice build
        logger.info("Starting global lattice construction...")
        global_lattice = builder.build_global_lattice()
        
        # Display results summary
        if 'error' not in global_lattice:
            print("✅ GLOBAL LATTICE BUILD COMPLETE")
            print("=" * 40)
            
            # Extract key metrics
            global_lattice.get('lattice_metadata', {})
            nodes = global_lattice.get('global_nodes', [])
            edges = global_lattice.get('global_edges', [])
            hot_zones = global_lattice.get('hot_zones', [])
            cascades = global_lattice.get('vertical_cascades', [])
            bridges = global_lattice.get('bridge_nodes', [])
            stats = global_lattice.get('statistics', {})
            
            print(f"📊 Sessions Processed: {stats.get('sessions_processed', 0)}")
            print(f"📊 Total Events: {stats.get('total_events', 0)}")
            print(f"🏗️  Global Nodes: {len(nodes)}")
            print(f"🔗 Global Edges: {len(edges)}")
            print(f"🔥 Hot Zones: {len(hot_zones)}")
            print(f"📈 Vertical Cascades: {len(cascades)}")
            print(f"🌉 Bridge Nodes: {len(bridges)}")
            print()
            
            # Display timeframe distribution
            tf_dist = stats.get('timeframe_distribution', {})
            print("⏰ TIMEFRAME DISTRIBUTION")
            print("-" * 25)
            for tf, count in tf_dist.items():
                if count > 0:
                    print(f"  {tf:>8}: {count:>4} events")
            print()
            
            # Display hot zone enrichment
            enrichment = global_lattice.get('enrichment_analysis', {})
            zones_by_enrichment = enrichment.get('hot_zones_by_enrichment', {})
            
            print("🔥 HOT ZONE ENRICHMENT")
            print("-" * 22)
            for category, zones in zones_by_enrichment.items():
                if zones:
                    print(f"  {category:>18}: {len(zones):>3} zones")
            print()
            
            # Display top hot zones
            sorted_zones = sorted(hot_zones, key=lambda x: x.get('enrichment_ratio', 0), reverse=True)
            top_zones = sorted_zones[:5]
            
            if top_zones:
                print("🏆 TOP HOT ZONES (by enrichment)")
                print("-" * 35)
                for i, zone in enumerate(top_zones, 1):
                    tf = zone.get('timeframe', 'unknown')
                    price = zone.get('price_level', 0)
                    enrichment = zone.get('enrichment_ratio', 0)
                    events = zone.get('event_count', 0)
                    print(f"  {i}. {tf} @ {price:.1f} - {enrichment:.1f}x enrichment ({events} events)")
                print()
            
            # Display cascade patterns
            cascade_patterns = enrichment.get('cascade_patterns', {})
            if cascade_patterns:
                print("📈 VERTICAL CASCADE PATTERNS")
                print("-" * 28)
                for pattern, count in sorted(cascade_patterns.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        print(f"  {pattern:>20}: {count:>3} cascades")
                print()
            
            # Display discovery recommendations
            recommendations = global_lattice.get('discovery_recommendations', [])
            if recommendations:
                print("🎯 DISCOVERY RECOMMENDATIONS")
                print("-" * 28)
                for i, rec in enumerate(recommendations, 1):
                    priority = rec.get('priority', 'unknown')
                    desc = rec.get('description', 'No description')
                    rec_type = rec.get('type', 'unknown')
                    print(f"  {i}. [{priority.upper()}] {desc}")
                    print(f"     Type: {rec_type}")
                print()
            
            print("🌐 Global lattice data saved to discoveries/ directory")
            print("Ready for STEP 2: Hot zone identification and cascade analysis")
            
        else:
            print("❌ GLOBAL LATTICE BUILD FAILED")
            print(f"Error: {global_lattice.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Global lattice execution failed: {e}")
        print(f"❌ Execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)