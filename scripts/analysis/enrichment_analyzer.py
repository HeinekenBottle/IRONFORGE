#!/usr/bin/env python3
"""
IRONFORGE Enrichment Analyzer
=============================

Tests the hypothesis: Do events in the 14:35-38pm PM belt disproportionately 
map to specific lattice zones (like the 40% dimensional anchor) compared to 
baseline PM minutes?

This statistical analysis determines whether the 14:35-38pm window is special
before building specialized tools for it.

Author: IRONFORGE Archaeological Discovery System
Date: August 16, 2025
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

class EnrichmentAnalyzer:
    """Analyzes statistical enrichment of PM belt events in lattice zones"""
    
    def __init__(self, lattice_dataset_path: str = "/Users/jack/IRONFORGE/deliverables/lattice_dataset/pm_only_lattice_dataset.json"):
        """Initialize enrichment analyzer"""
        self.lattice_dataset_path = Path(lattice_dataset_path)
        self.target_window = ("14:35:00", "14:36:00", "14:37:00", "14:38:00")
        self.pm_sessions = []  # Will store PM session events
        self.lattice_dataset = None
        
        print("🔬 IRONFORGE ENRICHMENT ANALYZER")
        print("=" * 60)
        print(f"Target window: {self.target_window}")
        print(f"Lattice dataset: {self.lattice_dataset_path}")
    
    def run_enrichment_analysis(self):
        """Run complete enrichment analysis"""
        
        # Step 1: Load lattice dataset
        if not self._load_lattice_dataset():
            return None
        
        # Step 2: Extract PM session events
        pm_events = self._extract_pm_session_events()
        if not pm_events:
            print("❌ No PM session events found")
            return None
        
        # Step 3: Analyze PM belt vs baseline distribution
        enrichment_results = self._analyze_pm_belt_enrichment(pm_events)
        
        # Step 4: Statistical significance testing
        statistical_results = self._perform_statistical_tests(enrichment_results)
        
        # Step 5: Generate comprehensive report
        self._generate_enrichment_report(enrichment_results, statistical_results)
        
        return {
            'enrichment_results': enrichment_results,
            'statistical_results': statistical_results,
            'pm_events_count': len(pm_events)
        }
    
    def _load_lattice_dataset(self) -> bool:
        """Load the global lattice dataset"""
        
        try:
            if not self.lattice_dataset_path.exists():
                print(f"❌ Lattice dataset not found at {self.lattice_dataset_path}")
                return False
            
            with open(self.lattice_dataset_path, 'r') as f:
                self.lattice_dataset = json.load(f)
            
            total_events = self.lattice_dataset['metadata']['total_events_mapped']
            nodes_count = len(self.lattice_dataset['nodes'])
            hot_zones_count = len(self.lattice_dataset['hot_zones'])
            
            print("✅ Loaded lattice dataset:")
            print(f"   Total events: {total_events}")
            print(f"   Lattice nodes: {nodes_count}")
            print(f"   Hot zones: {hot_zones_count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load lattice dataset: {e}")
            return False
    
    def _extract_pm_session_events(self) -> List[Dict]:
        """Extract all PM session events from enhanced sessions"""
        
        print("\n📊 Extracting PM session events...")
        
        # Find all PM session files
        pm_session_patterns = [
            "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_*PM*.json",
            "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_NYPM*.json"
        ]
        
        pm_events = []
        import glob
        
        for pattern in pm_session_patterns:
            files = glob.glob(pattern)
            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        session_data = json.load(f)
                    
                    events = session_data.get('events', [])
                    for event in events:
                        # Add session context
                        event['session_file'] = Path(file_path).name
                        pm_events.append(event)
                        
                except Exception as e:
                    print(f"⚠️ Failed to load {file_path}: {e}")
        
        print(f"✅ Extracted {len(pm_events)} PM session events")
        return pm_events
    
    def _analyze_pm_belt_enrichment(self, pm_events: List[Dict]) -> Dict:
        """Analyze PM belt enrichment vs baseline PM minutes"""
        
        print("\n🎯 Analyzing PM belt enrichment...")
        
        # Categorize events by time window
        pm_belt_events = []
        baseline_pm_events = []
        
        for event in pm_events:
            timestamp = event.get('timestamp', '00:00:00')
            
            if timestamp in self.target_window:
                pm_belt_events.append(event)
            else:
                baseline_pm_events.append(event)
        
        print(f"   PM belt events (14:35-38): {len(pm_belt_events)}")
        print(f"   Baseline PM events: {len(baseline_pm_events)}")
        
        # Analyze lattice zone distribution
        pm_belt_zones = self._map_events_to_zones(pm_belt_events)
        baseline_zones = self._map_events_to_zones(baseline_pm_events)
        
        # Calculate enrichment ratios
        enrichment_analysis = self._calculate_enrichment_ratios(pm_belt_zones, baseline_zones)
        
        return {
            'pm_belt_events_count': len(pm_belt_events),
            'baseline_events_count': len(baseline_pm_events),
            'pm_belt_zone_distribution': pm_belt_zones,
            'baseline_zone_distribution': baseline_zones,
            'enrichment_ratios': enrichment_analysis
        }
    
    def _map_events_to_zones(self, events: List[Dict]) -> Dict:
        """Map events to their lattice zones"""
        
        zone_counts = defaultdict(int)
        cycle_position_buckets = defaultdict(int)
        
        for event in events:
            # Calculate which lattice zone this event would map to
            relative_position = event.get('relative_cycle_position', 
                                        event.get('range_position', 0.5))
            
            # Quantize to lattice grid (same as TimeframeLatticeMapper)
            grid_resolution = 100
            position_bucket = int(relative_position * grid_resolution) / grid_resolution
            
            # Create zone identifiers
            zone_key = f"zone_{position_bucket:.3f}"
            zone_counts[zone_key] += 1
            
            # Special attention to theory B zones
            if 0.35 <= position_bucket <= 0.45:  # ~40% zone
                cycle_position_buckets['40_percent_zone'] += 1
            elif 0.15 <= position_bucket <= 0.25:  # ~20% zone
                cycle_position_buckets['20_percent_zone'] += 1
            elif 0.75 <= position_bucket <= 0.85:  # ~80% zone
                cycle_position_buckets['80_percent_zone'] += 1
            else:
                cycle_position_buckets['other_zones'] += 1
        
        return {
            'zone_counts': dict(zone_counts),
            'theory_b_zones': dict(cycle_position_buckets),
            'total_events': len(events)
        }
    
    def _calculate_enrichment_ratios(self, pm_belt_zones: Dict, baseline_zones: Dict) -> Dict:
        """Calculate enrichment ratios for each zone"""
        
        enrichment_ratios = {}
        
        # Calculate for Theory B zones
        for zone_name in ['40_percent_zone', '20_percent_zone', '80_percent_zone', 'other_zones']:
            belt_count = pm_belt_zones['theory_b_zones'].get(zone_name, 0)
            baseline_count = baseline_zones['theory_b_zones'].get(zone_name, 0)
            
            belt_total = pm_belt_zones['total_events']
            baseline_total = baseline_zones['total_events']
            
            if belt_total > 0 and baseline_total > 0:
                belt_freq = belt_count / belt_total
                baseline_freq = baseline_count / baseline_total
                
                enrichment_ratio = belt_freq / baseline_freq if baseline_freq > 0 else float('inf')
                
                enrichment_ratios[zone_name] = {
                    'belt_count': belt_count,
                    'belt_frequency': belt_freq,
                    'baseline_count': baseline_count,
                    'baseline_frequency': baseline_freq,
                    'enrichment_ratio': enrichment_ratio,
                    'enriched': enrichment_ratio > 1.5  # >50% enrichment threshold
                }
        
        return enrichment_ratios
    
    def _perform_statistical_tests(self, enrichment_results: Dict) -> Dict:
        """Perform statistical significance tests"""
        
        print("\n📈 Performing statistical tests...")
        
        statistical_results = {}
        
        for zone_name, zone_data in enrichment_results['enrichment_ratios'].items():
            belt_count = zone_data['belt_count']
            baseline_count = zone_data['baseline_count']
            belt_total = enrichment_results['pm_belt_events_count']
            baseline_total = enrichment_results['baseline_events_count']
            
            # Chi-square test for independence
            if belt_total > 0 and baseline_total > 0:
                # Create contingency table
                observed = np.array([
                    [belt_count, belt_total - belt_count],
                    [baseline_count, baseline_total - baseline_count]
                ])
                
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
                    
                    statistical_results[zone_name] = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'degrees_of_freedom': dof,
                        'significant': p_value < 0.05,
                        'highly_significant': p_value < 0.01
                    }
                    
                except Exception as e:
                    print(f"⚠️ Statistical test failed for {zone_name}: {e}")
                    statistical_results[zone_name] = {'error': str(e)}
        
        return statistical_results
    
    def _generate_enrichment_report(self, enrichment_results: Dict, statistical_results: Dict):
        """Generate comprehensive enrichment analysis report"""
        
        print("\n" + "🎯 ENRICHMENT ANALYSIS RESULTS" + "\n" + "=" * 60)
        
        # Summary statistics
        belt_count = enrichment_results['pm_belt_events_count']
        baseline_count = enrichment_results['baseline_events_count']
        
        print("📊 Event Distribution:")
        print(f"   PM Belt (14:35-38): {belt_count} events")
        print(f"   Baseline PM: {baseline_count} events")
        print(f"   Belt proportion: {belt_count/(belt_count+baseline_count)*100:.1f}%")
        
        # Zone enrichment analysis
        print("\n🔍 Zone Enrichment Analysis:")
        
        significant_zones = []
        for zone_name, zone_data in enrichment_results['enrichment_ratios'].items():
            ratio = zone_data['enrichment_ratio']
            enriched = zone_data['enriched']
            
            stat_data = statistical_results.get(zone_name, {})
            p_value = stat_data.get('p_value', 1.0)
            significant = stat_data.get('significant', False)
            
            status = "🔥 ENRICHED" if enriched and significant else "📊 Normal"
            
            print(f"   {zone_name.replace('_', ' ').title()}:")
            print(f"     Enrichment ratio: {ratio:.2f}x")
            print(f"     Belt frequency: {zone_data['belt_frequency']*100:.1f}%")
            print(f"     Baseline frequency: {zone_data['baseline_frequency']*100:.1f}%")
            print(f"     Statistical significance: p={p_value:.4f}")
            print(f"     Status: {status}")
            
            if enriched and significant:
                significant_zones.append(zone_name)
        
        # Key findings
        print("\n🎯 Key Findings:")
        
        if significant_zones:
            print("   ✅ SIGNIFICANT ENRICHMENT DETECTED")
            print(f"   📍 Enriched zones: {', '.join(significant_zones)}")
            
            # Check for 40% zone specifically
            forty_percent = enrichment_results['enrichment_ratios'].get('40_percent_zone', {})
            if forty_percent.get('enriched', False):
                ratio = forty_percent['enrichment_ratio']
                print(f"   🎯 Theory B Validation: 40% zone shows {ratio:.2f}x enrichment")
                print("   📈 This supports dimensional destiny hypothesis")
            
            print("\n💡 Recommendation: Proceed to bridge node analysis")
            print("   The PM belt shows statistical significance - build focused tools")
            
        else:
            print("   ❌ NO SIGNIFICANT ENRICHMENT")
            print("   📊 PM belt appears similar to baseline PM minutes")
            print("   💡 Recommendation: Focus on other time windows or patterns")
        
        # Export results
        self._export_enrichment_results({
            'enrichment_results': enrichment_results,
            'statistical_results': statistical_results,
            'summary': {
                'significant_zones': significant_zones,
                'forty_percent_enriched': 'forty_percent_zone' in significant_zones,
                'recommendation': 'proceed' if significant_zones else 'investigate_other_windows'
            }
        })
    
    def _export_enrichment_results(self, results: Dict):
        """Export enrichment analysis results"""
        
        output_path = Path("/Users/jack/IRONFORGE/deliverables/enrichment_analysis_results.json")
        
        try:
            results['analysis_timestamp'] = datetime.now().isoformat()
            results['target_window'] = self.target_window
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📁 Results exported: {output_path}")
            
        except Exception as e:
            print(f"⚠️ Export failed: {e}")

def main():
    """Run enrichment analysis"""
    
    analyzer = EnrichmentAnalyzer()
    results = analyzer.run_enrichment_analysis()
    
    if results:
        print("\n🎉 Enrichment analysis complete!")
    else:
        print("\n❌ Enrichment analysis failed")

if __name__ == "__main__":
    main()