#!/usr/bin/env python3
"""
IRONFORGE Multi-Timeframe Convergence Analyzer
==============================================

Identifies nodes where 1m, 5m, 15m, and 1h+ timeframes converge into significant patterns.
Analyzes HTF contamination, cross-session inheritance, energy accumulation, and cascade events.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

@dataclass
class ConvergenceNode:
    """Multi-timeframe convergence point"""
    node_id: str
    session_date: str
    session_type: str
    convergence_timestamp: str
    convergence_strength: float
    
    # HTF Analysis
    htf_carryover_strength: float
    cross_session_inheritance: float
    phase_transitions: int
    
    # Timeframe Alignment
    timeframe_1m_events: List[Dict]
    timeframe_5m_events: List[Dict]
    timeframe_15m_events: List[Dict]  
    timeframe_1h_events: List[Dict]
    
    # Energy Dynamics
    energy_accumulation_rate: Optional[float]
    total_accumulated_energy: Optional[float]
    momentum_strength: Optional[float]
    cascade_events: List[Dict]
    
    # Cross-Session Interactions
    fpfvg_interactions: Dict[str, int]
    historical_influences: List[str]
    structural_integrity: float

@dataclass
class ConvergenceAnalysis:
    """Analysis results for convergence patterns"""
    total_nodes_analyzed: int
    significant_convergences: List[ConvergenceNode]
    convergence_statistics: Dict[str, float]
    pattern_insights: Dict[str, any]

class MultiTimeframeConvergenceAnalyzer:
    """
    Analyzes multi-timeframe convergence patterns in IRONFORGE sessions
    """
    
    def __init__(self, sessions_path: str = None):
        self.logger = logging.getLogger('convergence_analyzer')
        
        if sessions_path is None:
            sessions_path = '/Users/jack/IRONFORGE/enhanced_sessions_with_relativity'
        
        self.sessions_path = Path(sessions_path)
        self.session_files = list(self.sessions_path.glob('*.json'))
        
        # Convergence thresholds
        self.min_htf_carryover = 0.7  # 70% minimum for significance
        self.min_cross_session_inheritance = 0.6  # 60% minimum
        self.min_phase_transitions = 15  # Minimum phase transitions for complexity
        
        print(f"🔗 Multi-Timeframe Convergence Analyzer initialized")
        print(f"  Sessions to analyze: {len(self.session_files)}")
        print(f"  Significance thresholds: HTF={self.min_htf_carryover}, Inheritance={self.min_cross_session_inheritance}")
    
    def analyze_all_convergences(self) -> ConvergenceAnalysis:
        """Analyze all sessions for multi-timeframe convergences"""
        
        print(f"\n🔍 Scanning {len(self.session_files)} sessions for convergence patterns...")
        
        significant_nodes = []
        all_nodes = []
        
        for session_file in self.session_files:
            try:
                nodes = self._analyze_session_convergences(session_file)
                all_nodes.extend(nodes)
                
                # Filter for significant convergences
                significant = [node for node in nodes if self._is_convergence_significant(node)]
                significant_nodes.extend(significant)
                
                if significant:
                    print(f"  {session_file.name}: {len(significant)} significant convergences found")
            
            except Exception as e:
                self.logger.error(f"Error analyzing {session_file}: {e}")
                continue
        
        # Generate analysis statistics
        convergence_statistics = self._calculate_convergence_statistics(all_nodes, significant_nodes)
        pattern_insights = self._generate_pattern_insights(significant_nodes)
        
        return ConvergenceAnalysis(
            total_nodes_analyzed=len(all_nodes),
            significant_convergences=significant_nodes,
            convergence_statistics=convergence_statistics,
            pattern_insights=pattern_insights
        )
    
    def _analyze_session_convergences(self, session_file: Path) -> List[ConvergenceNode]:
        """Analyze single session for convergence nodes"""
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Extract session metadata
        metadata = session_data.get('session_metadata', {})
        session_date = metadata.get('session_date', 'unknown')
        session_type = metadata.get('session_type', 'unknown')
        
        # Extract contamination analysis
        contamination = session_data.get('contamination_analysis', {})
        htf_data = contamination.get('htf_contamination', {})
        inheritance_data = contamination.get('cross_session_inheritance', {})
        
        # HTF metrics
        htf_carryover_strength = htf_data.get('htf_carryover_strength', 0.0)
        cross_session_inheritance = htf_data.get('cross_session_inheritance', 0.0)
        phase_transitions = contamination.get('phase_transitions', 0)
        
        # Skip if not interesting
        if htf_carryover_strength < 0.3:  # Very low threshold for discovery
            return []
        
        nodes = []
        
        # Look for specific convergence events
        convergence_events = self._extract_convergence_events(session_data)
        
        for event_time, event_data in convergence_events.items():
            # Extract timeframe-specific events around this time
            tf_1m_events = self._extract_timeframe_events(session_data, event_time, '1m')
            tf_5m_events = self._extract_timeframe_events(session_data, event_time, '5m')
            tf_15m_events = self._extract_timeframe_events(session_data, event_time, '15m')
            tf_1h_events = self._extract_timeframe_events(session_data, event_time, '1h')
            
            # Extract energy dynamics
            energy_rate = None
            total_energy = None
            momentum = None
            
            if 'temporal_flow_analysis' in session_data:
                flow_data = session_data['temporal_flow_analysis']
                if 'energy_accumulation' in flow_data:
                    energy_acc = flow_data['energy_accumulation']
                    energy_rate = energy_acc.get('energy_rate')
                    total_energy = energy_acc.get('total_accumulated')
                if 'temporal_momentum_strength' in flow_data:
                    momentum = flow_data['temporal_momentum_strength']
            
            # Extract cascade events
            cascade_events = session_data.get('cascade_events', [])
            
            # Extract FPFVG interactions
            fpfvg_interactions = {}
            if isinstance(inheritance_data, dict):
                for key, value in inheritance_data.items():
                    if 'fpfvg_interactions' in key:
                        fpfvg_interactions[key] = value
            
            # Historical influences
            historical_influences = []
            if isinstance(htf_data, dict):
                for key, value in htf_data.items():
                    if value is True and 'influence' in key:
                        historical_influences.append(key)
            
            node = ConvergenceNode(
                node_id=f"{session_date}_{session_type}_{event_time}",
                session_date=session_date,
                session_type=session_type.upper(),
                convergence_timestamp=event_time,
                convergence_strength=self._calculate_convergence_strength(
                    htf_carryover_strength, cross_session_inheritance, phase_transitions,
                    len(tf_1m_events), len(tf_5m_events), len(tf_15m_events), len(tf_1h_events)
                ),
                htf_carryover_strength=htf_carryover_strength,
                cross_session_inheritance=cross_session_inheritance,
                phase_transitions=phase_transitions,
                timeframe_1m_events=tf_1m_events,
                timeframe_5m_events=tf_5m_events,
                timeframe_15m_events=tf_15m_events,
                timeframe_1h_events=tf_1h_events,
                energy_accumulation_rate=energy_rate,
                total_accumulated_energy=total_energy,
                momentum_strength=momentum,
                cascade_events=cascade_events,
                fpfvg_interactions=fpfvg_interactions,
                historical_influences=historical_influences,
                structural_integrity=contamination.get('structural_integrity', 0.0)
            )
            
            nodes.append(node)
        
        return nodes
    
    def _extract_convergence_events(self, session_data: Dict) -> Dict[str, Dict]:
        """Extract specific convergence event times"""
        
        convergence_events = {}
        
        # Look for cascade events
        if 'cascade_events' in session_data:
            for event in session_data['cascade_events']:
                timestamp = event.get('timestamp', 'unknown')
                convergence_events[timestamp] = event
        
        # Look for phase transitions
        if 'phase_transitions' in session_data:
            for transition in session_data['phase_transitions']:
                timestamp = transition.get('transition_timestamp', 'unknown')
                convergence_events[timestamp] = transition
        
        # Look for FPFVG interactions
        fpfvg_data = session_data.get('session_fpfvg', {})
        if 'fpfvg_formation' in fpfvg_data:
            formation = fpfvg_data['fpfvg_formation']
            if 'interactions' in formation:
                for interaction in formation['interactions']:
                    timestamp = interaction.get('interaction_time', 'unknown')
                    convergence_events[timestamp] = interaction
        
        # Look for price movements with high activity
        price_movements = session_data.get('price_movements', [])
        for movement in price_movements:
            timestamp = movement.get('timestamp', 'unknown')
            # High activity = significant price momentum or range position change
            momentum = abs(movement.get('price_momentum', 0))
            if momentum > 0.5:  # Significant momentum threshold
                convergence_events[timestamp] = movement
        
        return convergence_events
    
    def _extract_timeframe_events(self, session_data: Dict, target_time: str, timeframe: str) -> List[Dict]:
        """Extract events for specific timeframe around target time"""
        
        events = []
        
        # Convert target time to minutes for comparison
        try:
            if ':' in target_time:
                time_parts = target_time.split(':')
                target_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
            else:
                return []
        except:
            return []
        
        # Define timeframe windows
        windows = {
            '1m': 1,
            '5m': 5, 
            '15m': 15,
            '1h': 60
        }
        
        window_size = windows.get(timeframe, 1)
        
        # Extract events within timeframe window
        price_movements = session_data.get('price_movements', [])
        
        for movement in price_movements:
            try:
                timestamp = movement.get('timestamp', '00:00')
                if ':' in timestamp:
                    time_parts = timestamp.split(':')
                    event_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
                    
                    # Check if within timeframe window
                    if abs(event_minutes - target_minutes) <= window_size:
                        events.append(movement)
            except:
                continue
        
        return events
    
    def _calculate_convergence_strength(self, htf_carryover: float, inheritance: float,
                                      phase_transitions: int, tf_1m_count: int, 
                                      tf_5m_count: int, tf_15m_count: int, tf_1h_count: int) -> float:
        """Calculate overall convergence strength score"""
        
        # Base HTF strength (40% weight)
        htf_component = (htf_carryover * 0.6 + inheritance * 0.4) * 0.4
        
        # Phase transition complexity (20% weight)
        phase_component = min(phase_transitions / 30.0, 1.0) * 0.2  # Normalize to 30 max
        
        # Multi-timeframe activity (40% weight)
        total_events = tf_1m_count + tf_5m_count + tf_15m_count + tf_1h_count
        activity_component = min(total_events / 20.0, 1.0) * 0.2  # Normalize to 20 events
        
        # Timeframe balance (events across all timeframes)
        active_timeframes = sum(1 for count in [tf_1m_count, tf_5m_count, tf_15m_count, tf_1h_count] if count > 0)
        balance_component = (active_timeframes / 4.0) * 0.2
        
        return htf_component + phase_component + activity_component + balance_component
    
    def _is_convergence_significant(self, node: ConvergenceNode) -> bool:
        """Determine if convergence node is significant"""
        
        return (node.htf_carryover_strength >= self.min_htf_carryover and
                node.cross_session_inheritance >= self.min_cross_session_inheritance and
                node.phase_transitions >= self.min_phase_transitions)
    
    def _calculate_convergence_statistics(self, all_nodes: List[ConvergenceNode], 
                                        significant_nodes: List[ConvergenceNode]) -> Dict[str, float]:
        """Calculate convergence statistics"""
        
        if not all_nodes:
            return {}
        
        stats = {
            'total_nodes': len(all_nodes),
            'significant_nodes': len(significant_nodes),
            'significance_rate': len(significant_nodes) / len(all_nodes) if all_nodes else 0,
            'avg_convergence_strength': np.mean([node.convergence_strength for node in all_nodes]),
            'avg_htf_carryover': np.mean([node.htf_carryover_strength for node in all_nodes]),
            'avg_cross_session_inheritance': np.mean([node.cross_session_inheritance for node in all_nodes]),
            'avg_phase_transitions': np.mean([node.phase_transitions for node in all_nodes]),
            'max_convergence_strength': max([node.convergence_strength for node in all_nodes]),
            'max_htf_carryover': max([node.htf_carryover_strength for node in all_nodes]),
            'max_phase_transitions': max([node.phase_transitions for node in all_nodes])
        }
        
        return stats
    
    def _generate_pattern_insights(self, significant_nodes: List[ConvergenceNode]) -> Dict[str, any]:
        """Generate insights from convergence patterns"""
        
        if not significant_nodes:
            return {}
        
        insights = {}
        
        # Session type analysis
        session_types = [node.session_type for node in significant_nodes]
        insights['most_convergent_session_types'] = dict(Counter(session_types))
        
        # Timeframe activity patterns
        total_1m = sum(len(node.timeframe_1m_events) for node in significant_nodes)
        total_5m = sum(len(node.timeframe_5m_events) for node in significant_nodes)
        total_15m = sum(len(node.timeframe_15m_events) for node in significant_nodes)
        total_1h = sum(len(node.timeframe_1h_events) for node in significant_nodes)
        
        insights['timeframe_activity_distribution'] = {
            '1m_events': total_1m,
            '5m_events': total_5m,
            '15m_events': total_15m,
            '1h_events': total_1h
        }
        
        # Energy accumulation patterns
        energy_nodes = [node for node in significant_nodes if node.energy_accumulation_rate is not None]
        if energy_nodes:
            insights['energy_patterns'] = {
                'nodes_with_energy': len(energy_nodes),
                'avg_energy_rate': np.mean([node.energy_accumulation_rate for node in energy_nodes]),
                'max_accumulated_energy': max([node.total_accumulated_energy or 0 for node in energy_nodes])
            }
        
        # Historical influence patterns
        all_influences = []
        for node in significant_nodes:
            all_influences.extend(node.historical_influences)
        
        insights['historical_influence_frequency'] = dict(Counter(all_influences))
        
        # FPFVG interaction patterns
        fpfvg_totals = defaultdict(int)
        for node in significant_nodes:
            for interaction_type, count in node.fpfvg_interactions.items():
                fpfvg_totals[interaction_type] += count
        
        insights['fpfvg_interaction_totals'] = dict(fpfvg_totals)
        
        return insights
    
    def generate_convergence_report(self, analysis: ConvergenceAnalysis) -> Dict:
        """Generate comprehensive convergence analysis report"""
        
        # Top convergence nodes
        top_nodes = sorted(analysis.significant_convergences, 
                          key=lambda x: x.convergence_strength, reverse=True)[:10]
        
        report = {
            'analysis_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_sessions_analyzed': len(self.session_files),
                'convergence_nodes_found': analysis.total_nodes_analyzed,
                'significant_convergences': len(analysis.significant_convergences)
            },
            
            'convergence_statistics': analysis.convergence_statistics,
            'pattern_insights': analysis.pattern_insights,
            
            'top_convergence_nodes': [
                {
                    'node_id': node.node_id,
                    'session_date': node.session_date,
                    'session_type': node.session_type,
                    'convergence_timestamp': node.convergence_timestamp,
                    'convergence_strength': node.convergence_strength,
                    'htf_carryover_strength': node.htf_carryover_strength,
                    'cross_session_inheritance': node.cross_session_inheritance,
                    'phase_transitions': node.phase_transitions,
                    'timeframe_activity': {
                        '1m_events': len(node.timeframe_1m_events),
                        '5m_events': len(node.timeframe_5m_events),
                        '15m_events': len(node.timeframe_15m_events),
                        '1h_events': len(node.timeframe_1h_events)
                    },
                    'energy_metrics': {
                        'accumulation_rate': node.energy_accumulation_rate,
                        'total_accumulated': node.total_accumulated_energy,
                        'momentum_strength': node.momentum_strength
                    },
                    'cascade_events': len(node.cascade_events),
                    'fpfvg_interactions': node.fpfvg_interactions,
                    'historical_influences': node.historical_influences,
                    'structural_integrity': node.structural_integrity
                }
                for node in top_nodes
            ],
            
            'detailed_convergences': [
                {
                    'node_id': node.node_id,
                    'convergence_details': {
                        'session_info': {
                            'date': node.session_date,
                            'type': node.session_type,
                            'timestamp': node.convergence_timestamp
                        },
                        'strength_metrics': {
                            'overall_strength': node.convergence_strength,
                            'htf_carryover': node.htf_carryover_strength,
                            'inheritance': node.cross_session_inheritance,
                            'complexity': node.phase_transitions
                        },
                        'timeframe_convergence': {
                            '1m_activity': len(node.timeframe_1m_events),
                            '5m_activity': len(node.timeframe_5m_events),
                            '15m_activity': len(node.timeframe_15m_events),
                            '1h_activity': len(node.timeframe_1h_events),
                            'total_events': (len(node.timeframe_1m_events) + len(node.timeframe_5m_events) + 
                                          len(node.timeframe_15m_events) + len(node.timeframe_1h_events))
                        },
                        'energy_dynamics': {
                            'has_energy_data': node.energy_accumulation_rate is not None,
                            'energy_rate': node.energy_accumulation_rate,
                            'total_energy': node.total_accumulated_energy,
                            'momentum': node.momentum_strength
                        },
                        'cross_session_data': {
                            'fpfvg_interactions': node.fpfvg_interactions,
                            'historical_influences': node.historical_influences,
                            'cascade_events_count': len(node.cascade_events)
                        }
                    }
                }
                for node in analysis.significant_convergences
            ]
        }
        
        return report
    
    def save_convergence_analysis(self, output_path: str = None) -> str:
        """Execute convergence analysis and save results"""
        
        if output_path is None:
            output_path = '/Users/jack/IRONFORGE/analysis/multi_timeframe_convergence_analysis.json'
        
        print(f"\n🔗 Executing Multi-Timeframe Convergence Analysis...")
        
        # Execute analysis
        analysis = self.analyze_all_convergences()
        
        # Generate report
        report = self.generate_convergence_report(analysis)
        
        # Save results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Convergence Analysis saved to: {output_path}")
        
        # Summary
        print(f"\n📊 Convergence Analysis Summary:")
        print(f"  Total convergence nodes: {analysis.total_nodes_analyzed}")
        print(f"  Significant convergences: {len(analysis.significant_convergences)}")
        print(f"  Highest convergence strength: {analysis.convergence_statistics.get('max_convergence_strength', 0):.3f}")
        print(f"  Maximum HTF carryover: {analysis.convergence_statistics.get('max_htf_carryover', 0):.3f}")
        
        return output_path

if __name__ == "__main__":
    print("🔗 IRONFORGE Multi-Timeframe Convergence Analyzer")
    print("=" * 70)
    
    analyzer = MultiTimeframeConvergenceAnalyzer()
    output_file = analyzer.save_convergence_analysis()
    
    print(f"\n✅ Multi-Timeframe Convergence Analysis complete!")
    print(f"📊 Results saved to: {output_file}")