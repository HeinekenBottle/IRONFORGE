#!/usr/bin/env python3
"""
Extract and display key findings from global lattice results
"""

import json
import sys
from pathlib import Path


def extract_lattice_summary(lattice_file):
    """Extract key summary information from lattice file"""
    
    try:
        with open(lattice_file) as f:
            lattice = json.load(f)
        
        print("🌐 GLOBAL LATTICE ANALYSIS RESULTS")
        print("=" * 50)
        
        # Extract metadata
        metadata = lattice.get('lattice_metadata', {})
        print(f"Build Time: {metadata.get('build_timestamp', 'unknown')}")
        print(f"Scope: {metadata.get('scope', 'unknown')}")
        print(f"Timeframes: {', '.join(metadata.get('timeframes', []))}")
        print()
        
        # Extract key metrics
        nodes = lattice.get('global_nodes', [])
        edges = lattice.get('global_edges', [])
        hot_zones = lattice.get('hot_zones', [])
        cascades = lattice.get('vertical_cascades', [])
        bridges = lattice.get('bridge_nodes', [])
        stats = lattice.get('statistics', {})
        
        print("📊 LATTICE STRUCTURE METRICS")
        print("-" * 30)
        print(f"Sessions Processed: {stats.get('sessions_processed', 0)}")
        print(f"Total Events: {stats.get('total_events', 0)}")
        print(f"Global Nodes: {len(nodes):,}")
        print(f"Global Edges: {len(edges):,}")
        print(f"Hot Zones: {len(hot_zones):,}")
        print(f"Vertical Cascades: {len(cascades):,}")
        print(f"Bridge Nodes: {len(bridges):,}")
        print()
        
        # Timeframe distribution
        tf_dist = stats.get('timeframe_distribution', {})
        print("⏰ TIMEFRAME EVENT DISTRIBUTION")
        print("-" * 32)
        total_events = sum(tf_dist.values())
        for tf, count in tf_dist.items():
            if count > 0:
                pct = (count / total_events * 100) if total_events > 0 else 0
                print(f"  {tf:>8}: {count:>5} events ({pct:>5.1f}%)")
        print()
        
        # Hot zone enrichment analysis
        enrichment = lattice.get('enrichment_analysis', {})
        zones_by_enrichment = enrichment.get('hot_zones_by_enrichment', {})
        
        print("🔥 HOT ZONE ENRICHMENT CATEGORIES")
        print("-" * 34)
        for category, zones in zones_by_enrichment.items():
            if zones:
                print(f"  {category:>20}: {len(zones):>3} zones")
        print()
        
        # Top enriched hot zones
        extreme_zones = zones_by_enrichment.get('extreme_enrichment', [])
        high_zones = zones_by_enrichment.get('high_enrichment', [])
        
        top_zones = sorted(extreme_zones + high_zones, 
                          key=lambda x: x.get('enrichment_ratio', 0), reverse=True)[:10]
        
        if top_zones:
            print("🏆 TOP ENRICHED HOT ZONES")
            print("-" * 25)
            for i, zone in enumerate(top_zones, 1):
                tf = zone.get('timeframe', 'unknown')
                price = zone.get('price_level', 0)
                enrichment = zone.get('enrichment_ratio', 0)
                events = zone.get('event_count', 0)
                zone_type = zone.get('zone_type', 'unknown')
                sessions = len(zone.get('sessions_involved', []))
                print(f"  {i:>2}. {tf:>7} @ {price:>8.1f} - {enrichment:>4.1f}x enrichment")
                print(f"      {events} events, {sessions} sessions, {zone_type}")
            print()
        
        # Cascade analysis
        cascade_patterns = enrichment.get('cascade_patterns', {})
        if cascade_patterns:
            print("📈 VERTICAL CASCADE PATTERNS")
            print("-" * 28)
            total_cascades = sum(cascade_patterns.values())
            for pattern, count in sorted(cascade_patterns.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    pct = (count / total_cascades * 100) if total_cascades > 0 else 0
                    print(f"  {pattern:>25}: {count:>4} cascades ({pct:>5.1f}%)")
            print()
        
        # Bridge node analysis
        if bridges:
            bridge_types = {}
            for bridge in bridges:
                bridge_type = bridge.get('bridge_type', 'unknown')
                bridge_types[bridge_type] = bridge_types.get(bridge_type, 0) + 1
            
            print("🌉 BRIDGE NODE TYPES")
            print("-" * 19)
            for bridge_type, count in sorted(bridge_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {bridge_type:>20}: {count:>4} nodes")
            print()
        
        # Discovery recommendations
        recommendations = lattice.get('discovery_recommendations', [])
        if recommendations:
            print("🎯 DISCOVERY RECOMMENDATIONS")
            print("-" * 28)
            for i, rec in enumerate(recommendations, 1):
                priority = rec.get('priority', 'unknown')
                desc = rec.get('description', 'No description')
                rec_type = rec.get('type', 'unknown')
                print(f"  {i}. [{priority.upper()}] {desc}")
                print(f"     Type: {rec_type}")
                
                # Show specific details for hot zone investigations
                if rec_type == 'hot_zone_investigation':
                    zones = rec.get('zones', [])
                    if zones:
                        print("     Top zones:")
                        for j, zone in enumerate(zones[:3], 1):
                            tf = zone.get('timeframe', 'unknown')
                            price = zone.get('price_level', 0)
                            enrichment = zone.get('enrichment_ratio', 0)
                            print(f"       {j}. {tf} @ {price:.1f} ({enrichment:.1f}x)")
                print()
        
        print("🌐 Global terrain map complete - ready for specialized analysis")
        
    except Exception as e:
        print(f"Error extracting lattice summary: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Find the latest global lattice file
    discoveries_path = Path("data/discoveries")
    lattice_files = list(discoveries_path.glob("global_lattice_monthly_to_1m_*.json"))
    
    if not lattice_files:
        print("No global lattice files found")
        sys.exit(1)
    
    # Get the most recent file
    latest_file = sorted(lattice_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Analyzing: {latest_file}")
    print()
    
    exit_code = extract_lattice_summary(latest_file)
    sys.exit(exit_code)