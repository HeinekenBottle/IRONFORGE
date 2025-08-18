#!/usr/bin/env python3
"""
IRONFORGE Pattern Monitor
=========================

Real-time pattern monitoring and alerting system.
Monitors for specific patterns and generates alerts.

Usage:
    python3 pattern_monitor.py [--watch-dir DIR] [--alert-threshold N]
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

from learning.enhanced_graph_builder import EnhancedGraphBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternMonitor:
    """Real-time pattern monitoring system"""
    
    def __init__(self, watch_dir: str = "enhanced_sessions_with_relativity", alert_threshold: int = 20):
        self.watch_dir = watch_dir
        self.alert_threshold = alert_threshold
        self.builder = EnhancedGraphBuilder()
        self.processed_files = set()
        self.pattern_history = []
        
    def monitor_patterns(self) -> None:
        """Monitor for new sessions and analyze patterns"""
        
        print("🔍 IRONFORGE Pattern Monitor Started")
        print(f"📁 Watching: {self.watch_dir}")
        print(f"🚨 Alert threshold: {self.alert_threshold} events")
        print("=" * 50)
        
        # Initial scan
        self._scan_directory()
        
        print("✅ Initial scan complete. Monitoring for new sessions...")
        print("Press Ctrl+C to stop monitoring")
        print()
        
        try:
            while True:
                self._scan_directory()
                time.sleep(10)  # Check every 10 seconds
        except KeyboardInterrupt:
            print("\n🛑 Pattern monitoring stopped")
            self._generate_summary()
    
    def _scan_directory(self) -> None:
        """Scan directory for new session files"""
        
        if not os.path.exists(self.watch_dir):
            logger.warning(f"Watch directory {self.watch_dir} does not exist")
            return
        
        session_files = [f for f in os.listdir(self.watch_dir) if f.endswith('.json')]
        new_files = [f for f in session_files if f not in self.processed_files]
        
        for session_file in new_files:
            self._analyze_new_session(session_file)
            self.processed_files.add(session_file)
    
    def _analyze_new_session(self, session_file: str) -> None:
        """Analyze a new session file"""
        
        try:
            print(f"🔍 Analyzing new session: {session_file}")
            
            # Load session data
            with open(os.path.join(self.watch_dir, session_file), 'r') as f:
                session_data = json.load(f)
            
            # Build graph and analyze
            start_time = time.time()
            graph, metadata = self.builder.build_rich_graph(session_data)
            processing_time = time.time() - start_time
            
            # Extract patterns
            patterns = self._extract_patterns(graph, metadata)
            patterns['processing_time'] = processing_time
            patterns['session_file'] = session_file
            patterns['timestamp'] = datetime.now().isoformat()
            
            # Store in history
            self.pattern_history.append(patterns)
            
            # Check for alerts
            self._check_alerts(patterns)
            
            # Display results
            self._display_session_results(patterns)
            
        except Exception as e:
            logger.error(f"Error analyzing {session_file}: {str(e)}")
    
    def _extract_patterns(self, graph: dict, metadata: dict) -> dict:
        """Extract key patterns from session"""
        
        nodes = graph['rich_node_features']
        if not nodes:
            return {'error': 'No nodes found'}
        
        # Count semantic events
        fvg_events = sum(1 for node in nodes if node.fvg_redelivery_flag > 0)
        expansion_events = sum(1 for node in nodes if node.expansion_phase_flag > 0)
        consolidation_events = sum(1 for node in nodes if node.consolidation_flag > 0)
        liquidity_sweeps = sum(1 for node in nodes if node.liq_sweep_flag > 0)
        pd_array_interactions = sum(1 for node in nodes if node.pd_array_interaction_flag > 0)
        
        # Session phases
        open_phase = sum(1 for node in nodes if node.phase_open > 0.5)
        mid_phase = sum(1 for node in nodes if node.phase_mid > 0.5)
        close_phase = sum(1 for node in nodes if node.phase_close > 0.5)
        
        # Price analysis
        prices = [node.normalized_price for node in nodes]
        price_changes = [node.pct_from_open for node in nodes]
        
        return {
            'session_name': metadata.get('session_name', 'unknown'),
            'session_date': metadata.get('session_date', 'unknown'),
            'session_quality': metadata.get('session_quality', 'unknown'),
            'total_nodes': len(nodes),
            'semantic_events': {
                'fvg_redelivery': fvg_events,
                'expansion_phase': expansion_events,
                'consolidation': consolidation_events,
                'liquidity_sweeps': liquidity_sweeps,
                'pd_array_interactions': pd_array_interactions,
                'total_semantic': fvg_events + expansion_events + consolidation_events + liquidity_sweeps + pd_array_interactions
            },
            'session_phases': {
                'open': open_phase,
                'mid': mid_phase,
                'close': close_phase
            },
            'price_analysis': {
                'price_range': max(prices) - min(prices) if prices else 0,
                'max_price_change': max(price_changes) if price_changes else 0,
                'min_price_change': min(price_changes) if price_changes else 0
            },
            'discovery_rates': {
                'fvg_rate': (fvg_events / len(nodes)) * 100,
                'expansion_rate': (expansion_events / len(nodes)) * 100,
                'semantic_rate': ((fvg_events + expansion_events + consolidation_events) / len(nodes)) * 100
            }
        }
    
    def _check_alerts(self, patterns: dict) -> None:
        """Check for pattern-based alerts"""
        
        alerts = []
        semantic_events = patterns.get('semantic_events', {})
        
        # High FVG activity alert
        if semantic_events.get('fvg_redelivery', 0) >= self.alert_threshold:
            alerts.append(f"🔥 HIGH FVG ACTIVITY: {semantic_events['fvg_redelivery']} events detected!")
        
        # High expansion activity alert
        if semantic_events.get('expansion_phase', 0) >= self.alert_threshold:
            alerts.append(f"📈 HIGH EXPANSION ACTIVITY: {semantic_events['expansion_phase']} events detected!")
        
        # High semantic discovery rate alert
        if patterns.get('discovery_rates', {}).get('semantic_rate', 0) >= 50:
            alerts.append(f"🎯 HIGH SEMANTIC DISCOVERY: {patterns['discovery_rates']['semantic_rate']:.1f}% discovery rate!")
        
        # Excellent session quality alert
        if patterns.get('session_quality') == 'excellent':
            alerts.append(f"⭐ EXCELLENT SESSION QUALITY: {patterns['session_name']} session!")
        
        # Display alerts
        if alerts:
            print("🚨 PATTERN ALERTS:")
            for alert in alerts:
                print(f"   {alert}")
            print()
    
    def _display_session_results(self, patterns: dict) -> None:
        """Display session analysis results"""
        
        print(f"📊 Session: {patterns['session_name']} ({patterns['session_date']})")
        print(f"   Quality: {patterns['session_quality']} | Nodes: {patterns['total_nodes']} | Time: {patterns['processing_time']:.2f}s")
        
        semantic = patterns['semantic_events']
        print(f"   🔥 FVG: {semantic['fvg_redelivery']} | 📈 Expansion: {semantic['expansion_phase']} | 📊 Consolidation: {semantic['consolidation']}")
        
        rates = patterns['discovery_rates']
        print(f"   Discovery Rates: FVG {rates['fvg_rate']:.1f}% | Expansion {rates['expansion_rate']:.1f}% | Overall {rates['semantic_rate']:.1f}%")
        print()
    
    def _generate_summary(self) -> None:
        """Generate monitoring summary"""
        
        if not self.pattern_history:
            print("No sessions analyzed during monitoring period")
            return
        
        print("\n📊 MONITORING SUMMARY")
        print("=" * 50)
        
        total_sessions = len(self.pattern_history)
        total_fvg = sum(p['semantic_events']['fvg_redelivery'] for p in self.pattern_history)
        total_expansion = sum(p['semantic_events']['expansion_phase'] for p in self.pattern_history)
        total_nodes = sum(p['total_nodes'] for p in self.pattern_history)
        
        print(f"Sessions Monitored: {total_sessions}")
        print(f"Total Nodes: {total_nodes:,}")
        print(f"Total FVG Events: {total_fvg:,}")
        print(f"Total Expansion Events: {total_expansion:,}")
        print(f"Average Processing Time: {sum(p['processing_time'] for p in self.pattern_history) / total_sessions:.2f}s")
        print()
        
        # Top sessions
        if total_sessions > 0:
            top_fvg = max(self.pattern_history, key=lambda x: x['semantic_events']['fvg_redelivery'])
            top_expansion = max(self.pattern_history, key=lambda x: x['semantic_events']['expansion_phase'])
            
            print(f"🏆 Most Active FVG Session: {top_fvg['session_name']} ({top_fvg['semantic_events']['fvg_redelivery']} events)")
            print(f"🏆 Most Active Expansion Session: {top_expansion['session_name']} ({top_expansion['semantic_events']['expansion_phase']} events)")
        
        # Save monitoring log
        with open(f"pattern_monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(self.pattern_history, f, indent=2, default=str)
        
        print(f"\n💾 Monitoring log saved to pattern_monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

def main():
    parser = argparse.ArgumentParser(description='IRONFORGE Pattern Monitor')
    parser.add_argument('--watch-dir', default='enhanced_sessions_with_relativity', 
                       help='Directory to monitor for session files')
    parser.add_argument('--alert-threshold', type=int, default=20,
                       help='Alert threshold for semantic events')
    
    args = parser.parse_args()
    
    monitor = PatternMonitor(watch_dir=args.watch_dir, alert_threshold=args.alert_threshold)
    monitor.monitor_patterns()

if __name__ == "__main__":
    main()
