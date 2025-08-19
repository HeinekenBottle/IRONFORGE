#!/usr/bin/env python3
"""
IRONFORGE Bridge Node Mapper
============================

Maps cascade pathways from HTF events → PM belt events to identify:
1. Whether PM belt events are terminal nodes or relay points
2. Which HTF drivers (weekly/daily) consistently precede PM belt activity
3. Temporal patterns and lag relationships in the cascade structure

This reveals the structural "why" behind PM belt enrichment patterns.

Author: IRONFORGE Archaeological Discovery System  
Date: August 16, 2025
"""

import glob
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

class BridgeNodeMapper:
    """Maps HTF → PM cascade pathways and bridge node relationships"""
    
    def __init__(self):
        """Initialize bridge node mapper"""
        self.pm_belt_window = ("14:35:00", "14:36:00", "14:37:00", "14:38:00")
        self.all_sessions = {}  # Will store all session data by type
        self.cascade_patterns = []
        self.bridge_relationships = {}
        
        print("🌉 IRONFORGE BRIDGE NODE MAPPER")
        print("=" * 60)
        print(f"PM Belt target: {self.pm_belt_window}")
        print("Analyzing HTF → PM cascade structures...")
    
    def run_bridge_analysis(self):
        """Run complete bridge node analysis"""
        
        # Step 1: Load all enhanced sessions by type
        self._load_all_sessions()
        
        # Step 2: Extract PM belt events for analysis
        pm_belt_events = self._extract_pm_belt_events()
        
        if not pm_belt_events:
            print("❌ No PM belt events found")
            return None
        
        # Step 3: Map temporal cascade relationships
        cascade_patterns = self._map_cascade_patterns(pm_belt_events)
        
        # Step 4: Identify bridge node characteristics
        bridge_analysis = self._analyze_bridge_nodes(cascade_patterns)
        
        # Step 5: Generate cascade intelligence report
        self._generate_bridge_report(pm_belt_events, cascade_patterns, bridge_analysis)
        
        return {
            'pm_belt_events': pm_belt_events,
            'cascade_patterns': cascade_patterns,
            'bridge_analysis': bridge_analysis
        }
    
    def _load_all_sessions(self):
        """Load all enhanced session data organized by session type"""
        
        print("\n📊 Loading all enhanced sessions...")
        
        session_patterns = {
            'PREMARKET': "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_PREMARKET*.json",
            'ASIA': "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_ASIA*.json", 
            'LONDON': "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_LONDON*.json",
            'NY_AM': "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_NY_AM*.json",
            'LUNCH': "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_LUNCH*.json",
            'NY_PM': "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_*PM*.json",
            'MIDNIGHT': "/Users/jack/IRONFORGE/adapted_enhanced_sessions/adapted_enhanced_rel_MIDNIGHT*.json"
        }
        
        for session_type, pattern in session_patterns.items():
            self.all_sessions[session_type] = []
            files = glob.glob(pattern)
            
            for file_path in files:
                try:
                    with open(file_path) as f:
                        session_data = json.load(f)
                    
                    # Add metadata
                    session_data['session_type'] = session_type
                    session_data['file_name'] = Path(file_path).name
                    session_data['date'] = self._extract_date_from_filename(Path(file_path).name)
                    
                    self.all_sessions[session_type].append(session_data)
                    
                except Exception as e:
                    print(f"⚠️ Failed to load {file_path}: {e}")
            
            print(f"   {session_type}: {len(self.all_sessions[session_type])} sessions")
    
    def _extract_date_from_filename(self, filename: str) -> str | None:
        """Extract date from session filename"""
        try:
            # Look for pattern YYYY_MM_DD in filename
            import re
            match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
            if match:
                return match.group(1).replace('_', '-')
        except:
            pass
        return None
    
    def _extract_pm_belt_events(self) -> list[dict]:
        """Extract PM belt events with session context"""
        
        print("\n🎯 Extracting PM belt events...")
        
        pm_belt_events = []
        
        for session in self.all_sessions.get('NY_PM', []):
            events = session.get('events', [])
            session_date = session.get('date')
            
            for event in events:
                timestamp = event.get('timestamp', '00:00:00')
                
                if timestamp in self.pm_belt_window:
                    # Enrich with session context
                    enriched_event = event.copy()
                    enriched_event['session_context'] = {
                        'session_type': 'NY_PM',
                        'session_date': session_date,
                        'file_name': session.get('file_name'),
                        'session_events_count': len(events)
                    }
                    pm_belt_events.append(enriched_event)
        
        print(f"✅ Found {len(pm_belt_events)} PM belt events across {len(self.all_sessions.get('NY_PM', []))} PM sessions")
        
        # Group by date for analysis
        by_date = defaultdict(list)
        for event in pm_belt_events:
            date = event['session_context']['session_date']
            if date:
                by_date[date].append(event)
        
        print(f"   Distributed across {len(by_date)} trading days")
        for date, events in sorted(by_date.items()):
            print(f"     {date}: {len(events)} events")
        
        return pm_belt_events
    
    def _map_cascade_patterns(self, pm_belt_events: list[dict]) -> list[dict]:
        """Map temporal cascade patterns leading to PM belt events"""
        
        print("\n🌊 Mapping cascade patterns...")
        
        cascade_patterns = []
        
        for pm_event in pm_belt_events:
            session_date = pm_event['session_context']['session_date']
            if not session_date:
                continue
            
            # Find all events on the same trading day that could be precursors
            precursor_events = self._find_precursor_events(session_date, pm_event)
            
            # Build cascade pattern
            cascade_pattern = {
                'pm_belt_event': pm_event,
                'trading_date': session_date,
                'precursor_events': precursor_events,
                'cascade_structure': self._analyze_cascade_structure(precursor_events, pm_event)
            }
            
            cascade_patterns.append(cascade_pattern)
        
        print(f"✅ Mapped {len(cascade_patterns)} cascade patterns")
        
        return cascade_patterns
    
    def _find_precursor_events(self, target_date: str, pm_event: dict) -> dict[str, list[dict]]:
        """Find all events on target date that could precede the PM belt event"""
        
        precursor_events = {
            'PREMARKET': [],
            'ASIA': [],
            'LONDON': [], 
            'NY_AM': [],
            'LUNCH': []
        }
        
        # Define session precedence (sessions that come before NY_PM)
        precedence_sessions = ['PREMARKET', 'ASIA', 'LONDON', 'NY_AM', 'LUNCH']
        
        for session_type in precedence_sessions:
            sessions = self.all_sessions.get(session_type, [])
            
            for session in sessions:
                if session.get('date') == target_date:
                    events = session.get('events', [])
                    
                    for event in events:
                        # Add session context to event
                        enriched_event = event.copy()
                        enriched_event['session_context'] = {
                            'session_type': session_type,
                            'session_date': target_date,
                            'file_name': session.get('file_name')
                        }
                        precursor_events[session_type].append(enriched_event)
        
        return precursor_events
    
    def _analyze_cascade_structure(self, precursor_events: dict, pm_event: dict) -> dict:
        """Analyze the cascade structure for this specific PM event"""
        
        structure = {
            'total_precursor_events': 0,
            'session_participation': {},
            'key_patterns': [],
            'dimensional_relationships': [],
            'temporal_sequence': []
        }
        
        # Count events by session
        for session_type, events in precursor_events.items():
            count = len(events)
            structure['total_precursor_events'] += count
            structure['session_participation'][session_type] = count
        
        # Identify key patterns in precursor events
        all_precursors = []
        for events in precursor_events.values():
            all_precursors.extend(events)
        
        # Analyze event types leading to PM belt
        event_types = [event.get('type', 'unknown') for event in all_precursors]
        type_counts = Counter(event_types)
        structure['key_patterns'] = dict(type_counts.most_common(5))
        
        # Look for dimensional relationships (Theory B zones)
        pm_position = pm_event.get('range_position', 0.5)
        structure['pm_belt_position'] = pm_position
        
        # Check if PM event is in Theory B zones
        if 0.35 <= pm_position <= 0.45:
            structure['theory_b_zone'] = '40_percent'
        elif 0.15 <= pm_position <= 0.25:
            structure['theory_b_zone'] = '20_percent'
        elif 0.75 <= pm_position <= 0.85:
            structure['theory_b_zone'] = '80_percent'
        else:
            structure['theory_b_zone'] = 'other'
        
        # Analyze precursor positions for patterns
        precursor_positions = []
        for event in all_precursors:
            pos = event.get('range_position')
            if pos is not None:
                precursor_positions.append({
                    'position': pos,
                    'event_type': event.get('type'),
                    'session': event.get('session_context', {}).get('session_type')
                })
        
        structure['precursor_positions'] = precursor_positions
        
        # Create temporal sequence
        temporal_events = []
        for event in all_precursors:
            temporal_events.append({
                'timestamp': event.get('timestamp', '00:00:00'),
                'session': event.get('session_context', {}).get('session_type'),
                'event_type': event.get('type'),
                'position': event.get('range_position')
            })
        
        # Sort by session order then timestamp
        session_order = {'PREMARKET': 1, 'ASIA': 2, 'LONDON': 3, 'NY_AM': 4, 'LUNCH': 5}
        temporal_events.sort(key=lambda x: (session_order.get(x['session'], 99), x['timestamp']))
        structure['temporal_sequence'] = temporal_events
        
        return structure
    
    def _analyze_bridge_nodes(self, cascade_patterns: list[dict]) -> dict:
        """Analyze bridge node characteristics across all cascades"""
        
        print("\n🌉 Analyzing bridge node characteristics...")
        
        analysis = {
            'total_cascades': len(cascade_patterns),
            'session_bridge_frequency': defaultdict(int),
            'common_precursor_patterns': defaultdict(int),
            'theory_b_relationships': defaultdict(list),
            'cascade_depth_distribution': defaultdict(int),
            'terminal_vs_relay_classification': {},
            'key_findings': []
        }
        
        for pattern in cascade_patterns:
            structure = pattern['cascade_structure']
            
            # Track session participation as potential bridges
            for session, count in structure['session_participation'].items():
                if count > 0:
                    analysis['session_bridge_frequency'][session] += 1
            
            # Track cascade depth
            total_precursors = structure['total_precursor_events']
            analysis['cascade_depth_distribution'][total_precursors] += 1
            
            # Track Theory B relationships
            theory_b_zone = structure.get('theory_b_zone', 'other')
            analysis['theory_b_relationships'][theory_b_zone].append({
                'precursor_count': total_precursors,
                'key_patterns': structure['key_patterns'],
                'pm_position': structure['pm_belt_position']
            })
            
            # Track common precursor patterns
            for event_type, count in structure['key_patterns'].items():
                analysis['common_precursor_patterns'][event_type] += count
        
        # Classify terminal vs relay behavior
        total_patterns = len(cascade_patterns)
        bridge_sessions = analysis['session_bridge_frequency']
        
        for session, frequency in bridge_sessions.items():
            participation_rate = frequency / total_patterns
            analysis['terminal_vs_relay_classification'][session] = {
                'frequency': frequency,
                'participation_rate': participation_rate,
                'classification': 'frequent_bridge' if participation_rate > 0.5 else 'occasional_bridge'
            }
        
        # Generate key findings
        self._extract_key_findings(analysis)
        
        return analysis
    
    def _extract_key_findings(self, analysis: dict):
        """Extract key strategic findings from bridge analysis"""
        
        findings = []
        
        # Most frequent bridge sessions
        bridge_freq = analysis['session_bridge_frequency']
        if bridge_freq:
            top_bridge = max(bridge_freq.items(), key=lambda x: x[1])
            findings.append(f"Primary bridge session: {top_bridge[0]} (appears in {top_bridge[1]}/{analysis['total_cascades']} cascades)")
        
        # Theory B zone distribution
        theory_b = analysis['theory_b_relationships']
        forty_percent_count = len(theory_b.get('40_percent', []))
        if forty_percent_count > 0:
            findings.append(f"Theory B validation: {forty_percent_count} PM belt events in 40% zone")
        
        # Common precursor patterns
        precursor_patterns = analysis['common_precursor_patterns']
        if precursor_patterns:
            top_pattern = max(precursor_patterns.items(), key=lambda x: x[1])
            findings.append(f"Dominant precursor pattern: {top_pattern[0]} (appears {top_pattern[1]} times)")
        
        # Cascade depth analysis
        depth_dist = analysis['cascade_depth_distribution']
        if depth_dist:
            avg_depth = sum(depth * count for depth, count in depth_dist.items()) / sum(depth_dist.values())
            findings.append(f"Average cascade depth: {avg_depth:.1f} precursor events per PM belt event")
        
        analysis['key_findings'] = findings
    
    def _generate_bridge_report(self, pm_belt_events: list[dict], cascade_patterns: list[dict], bridge_analysis: dict):
        """Generate comprehensive bridge node analysis report"""
        
        print("\n" + "🌉 BRIDGE NODE ANALYSIS RESULTS" + "\n" + "=" * 60)
        
        # Summary statistics
        print("📊 Cascade Analysis Summary:")
        print(f"   PM belt events analyzed: {len(pm_belt_events)}")
        print(f"   Complete cascade patterns: {len(cascade_patterns)}")
        print(f"   Trading days covered: {len({p['trading_date'] for p in cascade_patterns if p['trading_date']})}")
        
        # Bridge session frequency
        print("\n🌉 Bridge Session Analysis:")
        bridge_freq = bridge_analysis['session_bridge_frequency']
        total_cascades = bridge_analysis['total_cascades']
        
        for session, frequency in sorted(bridge_freq.items(), key=lambda x: x[1], reverse=True):
            participation_rate = frequency / total_cascades * 100
            classification = bridge_analysis['terminal_vs_relay_classification'][session]['classification']
            print(f"   {session}: {frequency}/{total_cascades} cascades ({participation_rate:.1f}%) - {classification}")
        
        # Cascade depth distribution
        print("\n📏 Cascade Depth Distribution:")
        depth_dist = bridge_analysis['cascade_depth_distribution']
        for depth, count in sorted(depth_dist.items()):
            percentage = count / total_cascades * 100
            print(f"   {depth} precursor events: {count} cascades ({percentage:.1f}%)")
        
        # Theory B zone analysis
        print("\n🎯 Theory B Zone Distribution:")
        theory_b = bridge_analysis['theory_b_relationships']
        for zone, events in theory_b.items():
            count = len(events)
            percentage = count / total_cascades * 100
            if count > 0:
                avg_precursors = np.mean([e['precursor_count'] for e in events])
                print(f"   {zone.replace('_', ' ').title()}: {count} events ({percentage:.1f}%) - avg {avg_precursors:.1f} precursors")
        
        # Common precursor patterns
        print("\n🔄 Common Precursor Patterns:")
        precursor_patterns = bridge_analysis['common_precursor_patterns']
        for pattern, count in sorted(precursor_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {pattern}: appears {count} times across cascades")
        
        # Key findings
        print("\n🎯 Key Strategic Findings:")
        for i, finding in enumerate(bridge_analysis['key_findings'], 1):
            print(f"   {i}. {finding}")
        
        # Tactical intelligence
        print("\n💡 Tactical Intelligence:")
        
        # Check for consistent bridge patterns
        if bridge_freq:
            primary_bridge = max(bridge_freq.items(), key=lambda x: x[1])
            if primary_bridge[1] / total_cascades > 0.6:
                print(f"   🎯 HIGH CONSISTENCY: {primary_bridge[0]} appears in {primary_bridge[1]}/{total_cascades} cascades")
                print(f"      → Monitor {primary_bridge[0]} session for PM belt setup conditions")
        
        # Check for Theory B concentration
        forty_percent_events = theory_b.get('40_percent', [])
        if len(forty_percent_events) > 0:
            concentration = len(forty_percent_events) / total_cascades
            if concentration > 0.3:
                print(f"   🎯 THEORY B CONCENTRATION: {len(forty_percent_events)}/{total_cascades} PM belt events in 40% zone")
                print(f"      → 40% zone shows {concentration:.1%} hit rate - dimensional destiny confirmed")
        
        # Check cascade depth for predictive timing
        if depth_dist:
            high_depth_cascades = sum(count for depth, count in depth_dist.items() if depth >= 5)
            if high_depth_cascades > 0:
                depth_percentage = high_depth_cascades / total_cascades
                print(f"   ⚡ CASCADE DEPTH: {high_depth_cascades}/{total_cascades} cascades have 5+ precursors")
                print(f"      → Complex cascades = {depth_percentage:.1%} - look for multi-session setups")
        
        # Identify if PM belt is terminal or relay
        next_session_events = self._count_post_pm_events(cascade_patterns)
        if next_session_events:
            avg_follow_through = np.mean(list(next_session_events.values()))
            if avg_follow_through < 1.0:
                print(f"   🏁 TERMINAL NODE: PM belt appears to be cascade endpoint (avg {avg_follow_through:.1f} follow-through events)")
                print("      → PM belt = decision point, not continuation")
            else:
                print(f"   🔄 RELAY NODE: PM belt shows follow-through (avg {avg_follow_through:.1f} subsequent events)")
                print("      → PM belt = transition point in larger structure")
        
        # Export results
        self._export_bridge_results({
            'pm_belt_events': pm_belt_events,
            'cascade_patterns': cascade_patterns,
            'bridge_analysis': bridge_analysis
        })
    
    def _count_post_pm_events(self, cascade_patterns: list[dict]) -> dict:
        """Count events that occur after PM belt events (to determine terminal vs relay)"""
        
        post_pm_counts = {}
        
        for pattern in cascade_patterns:
            trading_date = pattern['trading_date']
            if not trading_date:
                continue
            
            # Look for MIDNIGHT session events on same day (come after PM)
            midnight_sessions = self.all_sessions.get('MIDNIGHT', [])
            
            for session in midnight_sessions:
                if session.get('date') == trading_date:
                    midnight_events = len(session.get('events', []))
                    post_pm_counts[trading_date] = midnight_events
                    break
            else:
                post_pm_counts[trading_date] = 0
        
        return post_pm_counts
    
    def _export_bridge_results(self, results: dict):
        """Export bridge analysis results"""
        
        output_path = Path("/Users/jack/IRONFORGE/deliverables/bridge_analysis_results.json")
        
        try:
            # Convert numpy types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, defaultdict):
                    return dict(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            results = convert_numpy_types(results)
            results['analysis_timestamp'] = datetime.now().isoformat()
            results['pm_belt_window'] = self.pm_belt_window
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n📁 Bridge analysis results exported: {output_path}")
            
        except Exception as e:
            print(f"⚠️ Export failed: {e}")

def main():
    """Run bridge node analysis"""
    
    mapper = BridgeNodeMapper()
    results = mapper.run_bridge_analysis()
    
    if results:
        print("\n🎉 Bridge node analysis complete!")
        print("   Discovered cascade intelligence for PM belt trading")
    else:
        print("\n❌ Bridge node analysis failed")

if __name__ == "__main__":
    main()