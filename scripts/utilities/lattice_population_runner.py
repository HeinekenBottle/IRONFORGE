#!/usr/bin/env python3
"""
IRONFORGE Lattice Population Runner
==================================

Processes all enhanced session events through the fixed TimeframeLatticeMapper
to generate a comprehensive multi-timeframe lattice for global pattern analysis.

This establishes the baseline archaeological landscape before focusing on 
specific time windows (like 14:35-38pm PM belt).

Author: IRONFORGE Archaeological Discovery System
Date: August 16, 2025
"""

import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from analysis.enhanced_session_adapter import EnhancedSessionAdapter
from analysis.timeframe_lattice_mapper import TimeframeLatticeMapper


class LatticePopulationRunner:
    """Populates the complete IRONFORGE lattice with all available enhanced sessions"""
    
    def __init__(self, output_dir: str = "/Users/jack/IRONFORGE/deliverables/lattice_dataset"):
        """Initialize lattice population runner"""
        self.adapter = EnhancedSessionAdapter()
        self.mapper = TimeframeLatticeMapper(
            grid_resolution=100,
            min_node_events=2,  # Standard production threshold
            hot_zone_threshold=0.8  # Standard production threshold
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_stats = {
            'start_time': datetime.now().isoformat(),
            'sessions_processed': 0,
            'sessions_failed': 0,
            'total_events_mapped': 0,
            'processing_times': [],
            'failed_sessions': []
        }
        
        print("🏛️ IRONFORGE LATTICE POPULATION RUNNER")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Initialization timestamp: {self.run_stats['start_time']}")
    
    def run_complete_population(self):
        """Run complete lattice population with all enhanced sessions"""
        
        # Find all enhanced session files
        enhanced_session_files = self._discover_enhanced_sessions()
        
        if not enhanced_session_files:
            print("❌ No enhanced session files found")
            return None
        
        print(f"\n📊 Discovered {len(enhanced_session_files)} enhanced session files")
        
        # Process all sessions
        all_events = []
        for session_file in enhanced_session_files:
            session_events = self._process_session_file(session_file)
            if session_events:
                all_events.extend(session_events)
        
        if not all_events:
            print("❌ No events collected from any sessions")
            return None
        
        # Generate complete lattice
        print(f"\n🗺️ Generating complete lattice with {len(all_events)} total events")
        lattice_dataset = self._generate_complete_lattice(all_events)
        
        # Export results
        self._export_results(lattice_dataset)
        
        # Display final statistics
        self._display_final_statistics(lattice_dataset)
        
        return lattice_dataset
    
    def _discover_enhanced_sessions(self) -> list[Path]:
        """Discover all enhanced session files"""
        
        search_patterns = [
            "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_*.json",
            "/Users/jack/IRONFORGE/enhanced_sessions_with_relativity/enhanced_rel_*.json"
        ]
        
        found_files = []
        for pattern in search_patterns:
            files = glob.glob(pattern)
            found_files.extend([Path(f) for f in files])
        
        # Remove duplicates and sort
        unique_files = list(set(found_files))
        unique_files.sort()
        
        print("📁 Search patterns used:")
        for pattern in search_patterns:
            print(f"   {pattern}")
        
        return unique_files
    
    def _process_session_file(self, session_file: Path) -> list[dict]:
        """Process a single enhanced session file"""
        
        session_start = time.time()
        
        try:
            print(f"📄 Processing: {session_file.name}")
            
            # Load session data
            with open(session_file) as f:
                session_data = json.load(f)
            
            # Extract events
            if 'events' in session_data:
                # Already adapted format
                events = session_data['events']
                print(f"   ✅ Loaded {len(events)} pre-adapted events")
            else:
                # Need to adapt through Enhanced Session Adapter
                adapted = self.adapter.adapt_enhanced_session(session_data)
                events = adapted['events']
                print(f"   ✅ Adapted {len(events)} events")
            
            # Track processing time
            processing_time = time.time() - session_start
            self.run_stats['processing_times'].append(processing_time)
            self.run_stats['sessions_processed'] += 1
            self.run_stats['total_events_mapped'] += len(events)
            
            print(f"   ⏱️ Processing time: {processing_time:.3f}s")
            
            return events
            
        except Exception as e:
            processing_time = time.time() - session_start
            self.run_stats['sessions_failed'] += 1
            self.run_stats['failed_sessions'].append({
                'file': str(session_file),
                'error': str(e),
                'processing_time': processing_time
            })
            
            print(f"   ❌ Failed: {e}")
            return []
    
    def _generate_complete_lattice(self, all_events: list[dict]):
        """Generate the complete lattice from all events"""
        
        lattice_start = time.time()
        
        try:
            # Map all events to lattice
            lattice_dataset = self.mapper.map_events_to_lattice(all_events)
            
            lattice_time = time.time() - lattice_start
            
            print(f"✅ Lattice generation completed in {lattice_time:.3f}s")
            print(f"   Nodes created: {len(lattice_dataset.nodes)}")
            print(f"   Connections created: {len(lattice_dataset.connections)}")
            print(f"   Hot zones detected: {len(lattice_dataset.hot_zones)}")
            
            return lattice_dataset
            
        except Exception as e:
            print(f"❌ Lattice generation failed: {e}")
            return None
    
    def _export_results(self, lattice_dataset):
        """Export lattice dataset and run statistics"""
        
        if not lattice_dataset:
            return
        
        try:
            # Export main lattice dataset
            lattice_file = self.output_dir / "global_lattice_dataset.json"
            export_path = self.mapper.export_lattice_dataset(lattice_dataset, str(lattice_file))
            print(f"📁 Lattice dataset exported: {export_path}")
            
            # Export run statistics
            self.run_stats['end_time'] = datetime.now().isoformat()
            self.run_stats['total_processing_time'] = sum(self.run_stats['processing_times'])
            self.run_stats['average_processing_time'] = (
                self.run_stats['total_processing_time'] / len(self.run_stats['processing_times'])
                if self.run_stats['processing_times'] else 0
            )
            
            stats_file = self.output_dir / "population_run_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.run_stats, f, indent=2)
            print(f"📁 Run statistics exported: {stats_file}")
            
        except Exception as e:
            print(f"⚠️ Export failed: {e}")
    
    def _display_final_statistics(self, lattice_dataset):
        """Display comprehensive final statistics"""
        
        print("\n" + "🎯 LATTICE POPULATION RESULTS" + "\n" + "=" * 60)
        
        # Session processing stats
        total_sessions = self.run_stats['sessions_processed'] + self.run_stats['sessions_failed']
        success_rate = (self.run_stats['sessions_processed'] / total_sessions * 100) if total_sessions > 0 else 0
        
        print("📊 Session Processing:")
        print(f"   Sessions processed: {self.run_stats['sessions_processed']}")
        print(f"   Sessions failed: {self.run_stats['sessions_failed']}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Total events mapped: {self.run_stats['total_events_mapped']}")
        
        if self.run_stats['processing_times']:
            total_time = sum(self.run_stats['processing_times'])
            avg_time = total_time / len(self.run_stats['processing_times'])
            print(f"   Total processing time: {total_time:.3f}s")
            print(f"   Average time per session: {avg_time:.3f}s")
        
        # Lattice statistics
        if lattice_dataset:
            print("\n🗺️ Lattice Structure:")
            print(f"   Total nodes: {len(lattice_dataset.nodes)}")
            print(f"   Total connections: {len(lattice_dataset.connections)}")
            print(f"   Hot zones detected: {len(lattice_dataset.hot_zones)}")
            
            # Timeframe distribution
            timeframe_dist = {}
            for node in lattice_dataset.nodes.values():
                tf = node.coordinate.absolute_timeframe
                timeframe_dist[tf] = timeframe_dist.get(tf, 0) + 1
            
            print("\n📈 Timeframe Distribution:")
            for tf, count in sorted(timeframe_dist.items()):
                print(f"   {tf}: {count} nodes")
            
            # Hot zone analysis
            if lattice_dataset.hot_zones:
                print("\n🔥 Top Hot Zones:")
                sorted_zones = sorted(
                    lattice_dataset.hot_zones.items(), 
                    key=lambda x: x[1].total_events, 
                    reverse=True
                )
                for zone_id, zone in sorted_zones[:5]:
                    coord = f"{zone.coordinate.absolute_timeframe}@{zone.coordinate.cycle_position:.1%}"
                    print(f"   {zone_id}: {zone.total_events} events at {coord}")
        
        # Performance validation
        if self.run_stats['processing_times']:
            max_time = max(self.run_stats['processing_times'])
            ironforge_standard = 5.0  # <5s standard
            
            print("\n⚡ Performance Analysis:")
            print(f"   Max processing time: {max_time:.3f}s")
            if max_time < ironforge_standard:
                print(f"   ✅ Within IRONFORGE <{ironforge_standard}s standard")
            else:
                print(f"   ⚠️ Exceeds {ironforge_standard}s standard")
        
        print("\n✅ GLOBAL LATTICE POPULATION COMPLETE")
        print("   Ready for hot zone analysis and PM belt overlay")

def main():
    """Run the complete lattice population"""
    
    runner = LatticePopulationRunner()
    lattice_dataset = runner.run_complete_population()
    
    if lattice_dataset:
        print("\n🎉 Success! Global lattice populated with comprehensive archaeological data")
        print(f"📁 Results available in: {runner.output_dir}")
    else:
        print("\n❌ Population failed - check error messages above")

if __name__ == "__main__":
    main()