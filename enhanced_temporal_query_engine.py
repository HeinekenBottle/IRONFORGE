#!/usr/bin/env python3
"""
IRONFORGE Enhanced Temporal Query Engine - With Price Relativity
Interactive system for querying temporal patterns with archaeological zone analysis
Integrates Theory B temporal non-locality and session-aware price relativity
"""
import pandas as pd
import networkx as nx
import numpy as np
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re

# Import our price relativity components
from session_time_manager import SessionTimeManager
from archaeological_zone_calculator import ArchaeologicalZoneCalculator

class EnhancedTemporalQueryEngine:
    """Interactive temporal pattern query system with price relativity and Theory B integration"""
    
    def __init__(self, shard_dir: str = "data/shards/NQ_M5"):
        self.shard_dir = shard_dir
        self.sessions = {}
        self.graphs = {}
        self.metadata = {}
        self.session_stats = {}  # Store session high/low/open/close for each session
        
        # Initialize price relativity components
        self.session_manager = SessionTimeManager()
        self.zone_calculator = ArchaeologicalZoneCalculator()
        
        print("🔍 Initializing Enhanced Temporal Query Engine with Price Relativity...")
        self._load_all_sessions()
        
    def _load_all_sessions(self):
        """Load all available sessions into memory with price relativity calculations"""
        shard_paths = sorted(glob.glob(f"{self.shard_dir}/shard_*"))
        print(f"📊 Loading {len(shard_paths)} sessions...")
        
        for shard_path in shard_paths:
            session_id = Path(shard_path).name.replace('shard_', '')
            try:
                # Load nodes and edges
                nodes = pd.read_parquet(f"{shard_path}/nodes.parquet")
                edges = pd.read_parquet(f"{shard_path}/edges.parquet")
                
                # Create NetworkX graph
                G = nx.from_pandas_edgelist(
                    edges, 
                    source='src', 
                    target='dst',
                    edge_attr=True,
                    create_using=nx.DiGraph()
                )
                
                # Add node attributes
                node_attrs = {row['node_id']: row.to_dict() for _, row in nodes.iterrows()}
                nx.set_node_attributes(G, node_attrs)
                
                self.sessions[session_id] = nodes
                self.graphs[session_id] = G
                
                # Calculate and store session statistics for price relativity
                if 'price' in nodes.columns and len(nodes) > 0:
                    self.session_stats[session_id] = {
                        'session_high': nodes['price'].max(),
                        'session_low': nodes['price'].min(),
                        'session_open': nodes['price'].iloc[0],
                        'session_close': nodes['price'].iloc[-1],
                        'session_range': nodes['price'].max() - nodes['price'].min(),
                        'total_events': len(nodes)
                    }
                
                # Store metadata
                try:
                    with open(f"{shard_path}/meta.json", 'r') as f:
                        import json
                        self.metadata[session_id] = json.load(f)
                except:
                    self.metadata[session_id] = {}
                    
            except Exception as e:
                print(f"⚠️  Failed to load {session_id}: {e}")
                
        print(f"✅ Loaded {len(self.sessions)} sessions with price relativity calculations")
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a temporal question and get probabilistic answers with price relativity"""
        print(f"\n🤔 Query: {question}")
        
        # Enhanced parsing for price relativity queries
        if "after" in question.lower() and "what happens" in question.lower():
            return self._analyze_temporal_sequence(question)
        elif "when" in question.lower() and ("starts with" in question.lower() or "begins with" in question.lower()):
            return self._analyze_opening_patterns(question)
        elif "zone" in question.lower() or "archaeological" in question.lower():
            return self._analyze_archaeological_zones(question)
        elif "theory b" in question.lower() or "temporal non-locality" in question.lower():
            return self._analyze_theory_b_patterns(question)
        elif "relative" in question.lower() or "percentage" in question.lower():
            return self._analyze_relative_positioning(question)
        elif "show me" in question.lower() or "find" in question.lower():
            return self._search_patterns(question)
        elif "probability" in question.lower() or "likely" in question.lower():
            return self._calculate_probabilities(question)
        else:
            return self._general_analysis(question)
    
    def _analyze_temporal_sequence(self, question: str) -> Dict[str, Any]:
        """Analyze what happens after specific events with price relativity"""
        results = {
            "query_type": "temporal_sequence_with_relativity",
            "total_sessions": len(self.sessions),
            "matches": [],
            "probabilities": {},
            "insights": [],
            "price_relativity_analysis": []
        }
        
        # Extract time window if mentioned
        time_match = re.search(r'(\d+)\s*minutes?', question)
        time_window = int(time_match.group(1)) if time_match else 15
        
        # Enhanced event pattern recognition with archaeological zones
        event_patterns = {
            "40% zone": "archaeological_40pct",
            "60% zone": "archaeological_60pct", 
            "80% zone": "archaeological_80pct",
            "high liquidity": "f8_spike",
            "expansion": "expansion",
            "retracement": "retracement", 
            "reversal": "reversal",
            "theory b": "dimensional_destiny"
        }
        
        # Process each session with price relativity analysis
        for session_id, nodes in self.sessions.items():
            if len(nodes) < 10:  # Skip sessions with too few events
                continue
                
            session_stats = self.session_stats.get(session_id, {})
            if not session_stats:
                continue
                
            # Extract session type from session_id (e.g., "NYAM_2025-08-06")
            session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
            
            # Analyze each event with price relativity
            for idx, (_, event) in enumerate(nodes.iterrows()):
                if idx >= len(nodes) - 5:  # Need future events for analysis
                    continue
                    
                # Get event context with both temporal and price relativity
                event_context = self._get_enhanced_event_context(
                    event, session_type, session_stats, nodes, idx
                )
                
                # Check if this event matches query patterns
                for pattern_name, pattern_key in event_patterns.items():
                    if pattern_name.lower() in question.lower():
                        match_result = self._check_pattern_match(
                            event_context, pattern_key, nodes, idx, time_window
                        )
                        
                        if match_result['matches']:
                            results['matches'].append({
                                'session_id': session_id,
                                'event_time': event_context['absolute_time'],
                                'session_progress': event_context['session_progress_pct'],
                                'archaeological_zone': event_context['zone_analysis'].get('current_zone', 'unknown'),
                                'theory_b_precision': event_context['zone_analysis'].get('theory_b_precision', {}),
                                'subsequent_outcome': match_result['outcome'],
                                'pattern_type': pattern_name
                            })
        
        # Calculate enhanced probabilities with price relativity
        results['probabilities'] = self._calculate_enhanced_probabilities(results['matches'])
        results['insights'] = self._generate_relativity_insights(results['matches'])
        
        return results
    
    def _get_enhanced_event_context(self, event, session_type: str, session_stats: Dict[str, float], 
                                   nodes: pd.DataFrame, event_idx: int) -> Dict[str, Any]:
        """Get complete event context with temporal and price relativity"""
        
        # Get timestamp (try different column names)
        event_time = "12:00:00"  # Default
        if 'timestamp' in event:
            event_time = event['timestamp']
        elif 't' in event and pd.notna(event['t']):
            # Convert milliseconds to time if needed
            try:
                dt = pd.to_datetime(event['t'], unit='ms')
                event_time = dt.strftime('%H:%M:%S')
            except:
                pass
        
        # Get temporal context
        temporal_context = self.session_manager.calculate_session_progress(session_type, event_time)
        
        # Get archaeological zone analysis
        if 'price' in event:
            zone_analysis = self.zone_calculator.analyze_event_positioning(
                event['price'], event_time, session_type, session_stats
            )
        else:
            zone_analysis = {'current_zone': 'unknown', 'zone_classification': {}}
        
        return {
            'absolute_time': event_time,
            'session_progress_pct': temporal_context.get('session_progress_pct', 0),
            'session_phase': temporal_context.get('session_phase', 'unknown'),
            'zone_analysis': zone_analysis,
            'event_price': event.get('price', 0),
            'session_type': session_type,
            'event_index': event_idx,
            'total_session_events': len(nodes)
        }
    
    def _check_pattern_match(self, event_context: Dict[str, Any], pattern_key: str, 
                           nodes: pd.DataFrame, event_idx: int, time_window: int) -> Dict[str, Any]:
        """Check if event matches specified pattern with enhanced criteria"""
        
        matches = False
        outcome = "unknown"
        
        if pattern_key == "archaeological_40pct":
            # Check if event is in or near 40% archaeological zone
            zone = event_context['zone_analysis'].get('dimensional_relationship', '')
            matches = 'dimensional_destiny_40pct' in zone or 'structural_support_20pct' in zone
            
        elif pattern_key == "archaeological_60pct":
            # Check for 60% zone events
            zone = event_context['zone_analysis'].get('dimensional_relationship', '')
            matches = 'resistance_confluence_60pct' in zone
            
        elif pattern_key == "archaeological_80pct":
            # Check for 80% zone events
            zone = event_context['zone_analysis'].get('dimensional_relationship', '')
            matches = 'momentum_threshold_80pct' in zone
            
        elif pattern_key == "f8_spike":
            # Check for high f8 liquidity intensity
            if 'f8' in nodes.columns:
                event_f8 = nodes.iloc[event_idx].get('f8', 0)
                f8_threshold = nodes['f8'].quantile(0.95)
                matches = event_f8 > f8_threshold
                
        elif pattern_key == "dimensional_destiny":
            # Check for Theory B criteria
            theory_b = event_context['zone_analysis'].get('theory_b_analysis', {})
            matches = theory_b.get('meets_theory_b_precision', False)
        
        # Analyze outcome if match found
        if matches and event_idx < len(nodes) - 5:
            future_events = nodes.iloc[event_idx+1:event_idx+6]
            if len(future_events) > 0 and 'price' in future_events.columns:
                price_change = future_events['price'].max() - future_events['price'].min()
                outcome = "expansion" if price_change > 15 else "consolidation"
        
        return {'matches': matches, 'outcome': outcome}
    
    def _calculate_enhanced_probabilities(self, matches: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate probabilities with price relativity context"""
        if not matches:
            return {}
            
        total = len(matches)
        probabilities = {}
        
        # Outcome probabilities
        outcomes = [m['subsequent_outcome'] for m in matches]
        for outcome in set(outcomes):
            probabilities[f"{outcome}_probability"] = outcomes.count(outcome) / total
            
        # Zone-based probabilities
        zones = [m['archaeological_zone'] for m in matches]
        for zone in set(zones):
            zone_matches = [m for m in matches if m['archaeological_zone'] == zone]
            probabilities[f"{zone}_frequency"] = len(zone_matches) / total
            
        return probabilities
    
    def _generate_relativity_insights(self, matches: List[Dict[str, Any]]) -> List[str]:
        """Generate insights based on price relativity analysis"""
        insights = []
        
        if not matches:
            return ["No matches found for the specified pattern"]
            
        # Temporal insights
        session_progress = [m['session_progress'] for m in matches if 'session_progress' in m]
        if session_progress:
            avg_progress = np.mean(session_progress)
            insights.append(f"Events typically occur {avg_progress:.1f}% through the session")
        
        # Zone insights
        zones = [m['archaeological_zone'] for m in matches if 'archaeological_zone' in m]
        if zones:
            most_common_zone = max(set(zones), key=zones.count) if zones else "unknown"
            insights.append(f"Most common archaeological zone: {most_common_zone}")
        
        # Theory B insights
        theory_b_events = [m for m in matches if m.get('theory_b_precision', {}).get('meets_precision', False)]
        if theory_b_events:
            insights.append(f"Found {len(theory_b_events)} Theory B precision events")
            
        return insights
    
    def _analyze_archaeological_zones(self, question: str) -> Dict[str, Any]:
        """Analyze archaeological zone patterns and Theory B events"""
        results = {
            "query_type": "archaeological_zone_analysis",
            "total_sessions": len(self.sessions),
            "zone_events": [],
            "theory_b_candidates": [],
            "zone_statistics": {}
        }
        
        # Process each session for archaeological zone analysis
        for session_id, nodes in self.sessions.items():
            session_stats = self.session_stats.get(session_id, {})
            if not session_stats or len(nodes) < 5:
                continue
                
            session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
            
            # Generate zone report for this session
            session_data = {
                'session_stats': session_stats,
                'events': nodes.to_dict('records'),
                'session_type': session_type
            }
            
            zone_report = self.zone_calculator.generate_zone_report(session_data)
            
            if zone_report.get('temporal_non_locality_confirmed', False):
                results['theory_b_candidates'].append({
                    'session_id': session_id,
                    'session_type': session_type,
                    'theory_b_events': zone_report['theory_b_candidates'],
                    'precision_score': zone_report.get('average_precision', 0)
                })
        
        # Calculate zone statistics
        results['zone_statistics'] = {
            'total_sessions_analyzed': len([s for s in self.sessions.keys() if s in self.session_stats]),
            'theory_b_sessions': len(results['theory_b_candidates']),
            'theory_b_percentage': len(results['theory_b_candidates']) / len(self.sessions) * 100 if self.sessions else 0
        }
        
        return results
    
    def _analyze_theory_b_patterns(self, question: str) -> Dict[str, Any]:
        """Analyze Theory B temporal non-locality patterns"""
        results = {
            "query_type": "theory_b_analysis",
            "total_sessions": len(self.sessions),
            "precision_events": [],
            "temporal_non_locality_evidence": []
        }
        
        precision_threshold = 7.55  # Your empirical discovery
        
        for session_id, nodes in self.sessions.items():
            session_stats = self.session_stats.get(session_id, {})
            if not session_stats:
                continue
                
            session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
            
            # Look for Theory B precision events
            for idx, (_, event) in enumerate(nodes.iterrows()):
                if 'price' not in event or pd.isna(event['price']):
                    continue
                    
                event_time = event.get('timestamp', '12:00:00')
                
                analysis = self.zone_calculator.analyze_event_positioning(
                    event['price'], event_time, session_type, session_stats
                )
                
                theory_b = analysis.get('theory_b_analysis', {})
                if theory_b.get('meets_theory_b_precision', False):
                    temporal_context = analysis.get('temporal_context', {})
                    results['precision_events'].append({
                        'session_id': session_id,
                        'event_time': event_time,
                        'price': event['price'],
                        'distance_to_final_40pct': theory_b['distance_to_final_40pct'],
                        'precision_score': theory_b['precision_score'],
                        'session_progress': temporal_context.get('session_progress_pct', 0)
                    })
        
        # Sort by precision score
        results['precision_events'].sort(key=lambda x: x['precision_score'], reverse=True)
        
        return results
    
    def _analyze_opening_patterns(self, question: str) -> Dict[str, Any]:
        """Analyze session opening patterns with price relativity"""
        results = {
            "query_type": "opening_patterns_with_relativity",
            "total_sessions": len(self.sessions),
            "pattern_analysis": [],
            "session_comparisons": []
        }
        
        # Analyze opening patterns with archaeological zone context
        for session_id, nodes in self.sessions.items():
            if len(nodes) < 20:
                continue
            
            session_stats = self.session_stats.get(session_id, {})
            if not session_stats:
                continue
                
            session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
            
            # Take first 20% of events
            early_count = max(5, len(nodes) // 5)
            early_nodes = nodes.iloc[:early_count]
            
            # Analyze opening with price relativity
            opening_analysis = self._analyze_session_opening(
                early_nodes, session_type, session_stats, session_id
            )
            
            results["pattern_analysis"].append(opening_analysis)
        
        # Generate comparative insights
        results["comparative_insights"] = self._generate_opening_insights(results["pattern_analysis"])
        
        return results
        
    def _analyze_session_opening(self, early_nodes: pd.DataFrame, session_type: str, 
                               session_stats: Dict[str, float], session_id: str) -> Dict[str, Any]:
        """Analyze session opening with complete price relativity context"""
        
        if len(early_nodes) == 0 or 'price' not in early_nodes.columns:
            return {'session_id': session_id, 'error': 'insufficient_data'}
        
        price_start = early_nodes['price'].iloc[0]
        price_early_end = early_nodes['price'].iloc[-1]
        price_change = price_early_end - price_start
        early_range = early_nodes['price'].max() - early_nodes['price'].min()
        
        # Calculate opening zones using final session stats
        opening_zones = self.zone_calculator.calculate_zones_for_session(
            session_stats['session_high'], session_stats['session_low']
        )
        
        # Classify opening pattern with archaeological context
        opening_pattern = self._classify_opening_pattern(price_change, early_range)
        
        # Analyze where opening events occurred relative to final zones
        zone_distribution = self._analyze_opening_zone_distribution(
            early_nodes, session_type, session_stats
        )
        
        return {
            'session_id': session_id,
            'session_type': session_type,
            'opening_pattern': opening_pattern,
            'price_change': price_change,
            'early_range': early_range,
            'final_session_range': session_stats['session_range'],
            'opening_efficiency': early_range / session_stats['session_range'] if session_stats['session_range'] > 0 else 0,
            'zone_distribution': zone_distribution,
            'archaeological_context': opening_zones
        }
    
    def _classify_opening_pattern(self, price_change: float, early_range: float) -> str:
        """Classify opening pattern with enhanced criteria"""
        if abs(price_change) < 5 and early_range < 10:
            return "tight_consolidation"
        elif abs(price_change) < 10 and early_range < 20:
            return "normal_consolidation"
        elif price_change > 15:
            return "strong_bullish_expansion"
        elif price_change > 5:
            return "bullish_expansion"
        elif price_change < -15:
            return "strong_bearish_expansion"
        elif price_change < -5:
            return "bearish_expansion"
        else:
            return "mixed_action"
    
    def _analyze_opening_zone_distribution(self, early_nodes: pd.DataFrame, session_type: str, 
                                         session_stats: Dict[str, float]) -> Dict[str, int]:
        """Analyze which archaeological zones were active during opening"""
        zone_counts = {}
        
        for _, event in early_nodes.iterrows():
            if 'price' not in event or pd.isna(event['price']):
                continue
                
            # Get event timestamp
            event_time = "09:30:00"  # Default opening time
            if 'timestamp' in event:
                event_time = event['timestamp']
                
            zone_analysis = self.zone_calculator.analyze_event_positioning(
                event['price'], event_time, session_type, session_stats
            )
            
            zone_type = zone_analysis.get('dimensional_relationship', 'unknown')
            zone_counts[zone_type] = zone_counts.get(zone_type, 0) + 1
        
        return zone_counts
    
    def _generate_opening_insights(self, pattern_analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from opening pattern analysis"""
        insights = []
        
        if not pattern_analyses:
            return ["No opening patterns analyzed"]
        
        # Pattern frequency analysis
        patterns = [p['opening_pattern'] for p in pattern_analyses if 'opening_pattern' in p]
        if patterns:
            most_common = max(set(patterns), key=patterns.count)
            insights.append(f"Most common opening pattern: {most_common} ({patterns.count(most_common)} sessions)")
        
        # Efficiency analysis
        efficiencies = [p['opening_efficiency'] for p in pattern_analyses if 'opening_efficiency' in p and p['opening_efficiency'] > 0]
        if efficiencies:
            avg_efficiency = np.mean(efficiencies)
            insights.append(f"Average opening efficiency (early range/final range): {avg_efficiency:.1%}")
        
        return insights
    
    def _analyze_relative_positioning(self, question: str) -> Dict[str, Any]:
        """Analyze relative positioning patterns"""
        results = {
            "query_type": "relative_positioning_analysis",
            "total_sessions": len(self.sessions),
            "positioning_patterns": [],
            "session_comparisons": []
        }
        
        for session_id, nodes in self.sessions.items():
            session_stats = self.session_stats.get(session_id, {})
            if not session_stats:
                continue
                
            # Calculate relative positioning for all events
            positioning_data = []
            for _, event in nodes.iterrows():
                if 'price' in event and pd.notna(event['price']):
                    price_pct = ((event['price'] - session_stats['session_low']) / 
                               session_stats['session_range'] * 100) if session_stats['session_range'] > 0 else 0
                    positioning_data.append(price_pct)
            
            if positioning_data:
                results['positioning_patterns'].append({
                    'session_id': session_id,
                    'mean_position': np.mean(positioning_data),
                    'position_range': np.max(positioning_data) - np.min(positioning_data),
                    'position_std': np.std(positioning_data)
                })
        
        return results
    
    def _search_patterns(self, question: str) -> Dict[str, Any]:
        """Enhanced pattern search with archaeological zones"""
        results = {
            "query_type": "enhanced_pattern_search",
            "matches": [],
            "total_sessions_searched": len(self.sessions)
        }
        
        # Enhanced pattern matching with archaeological zones
        for session_id, nodes in self.sessions.items():
            session_stats = self.session_stats.get(session_id, {})
            if not session_stats:
                continue
                
            session_range = session_stats['session_range']
            
            # Archaeological zone-aware search
            if "large range" in question.lower() and session_range > 150:
                results["matches"].append({
                    "session": session_id,
                    "range": session_range,
                    "events": len(nodes),
                    "quality": "high" if session_range > 200 else "medium",
                    "archaeological_zones": self.zone_calculator.calculate_zones_for_session(
                        session_stats['session_high'], session_stats['session_low']
                    )
                })
            elif "theory b" in question.lower():
                # Search for Theory B precision sessions
                session_data = {
                    'session_stats': session_stats,
                    'events': nodes.to_dict('records'),
                    'session_type': session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
                }
                
                zone_report = self.zone_calculator.generate_zone_report(session_data)
                if zone_report.get('temporal_non_locality_confirmed', False):
                    results["matches"].append({
                        "session": session_id,
                        "theory_b_events": zone_report['theory_b_candidates'],
                        "precision_score": zone_report.get('average_precision', 0),
                        "quality": "theory_b_confirmed"
                    })
        
        return results
    
    def _calculate_probabilities(self, question: str) -> Dict[str, Any]:
        """Calculate probabilities with enhanced analysis"""
        return {
            "query_type": "enhanced_probability_calculation",
            "note": "Probability calculation based on historical patterns with archaeological zone context",
            "total_samples": len(self.sessions),
            "available_analyses": [
                "temporal_sequence_probabilities",
                "archaeological_zone_probabilities", 
                "theory_b_precision_probabilities",
                "session_outcome_probabilities"
            ]
        }
    
    def _general_analysis(self, question: str) -> Dict[str, Any]:
        """General analysis with price relativity features"""
        return {
            "query_type": "general_analysis_enhanced",
            "suggestion": "Try questions like: 'What happens after a 40% zone event?', 'Show me Theory B precision events', 'What happens after high liquidity spikes?', 'When session starts with expansion in the 60% zone?'",
            "available_sessions": list(self.sessions.keys())[:5],
            "price_relativity_features": {
                "temporal_analysis": "Session progress percentages and timing",
                "archaeological_zones": "40%, 60%, 80% dimensional relationships", 
                "theory_b_detection": "7.55-point precision temporal non-locality",
                "session_types": list(self.session_manager.session_specs.keys())
            }
        }
    
    def get_enhanced_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get complete session information with price relativity analysis"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
            
        nodes = self.sessions[session_id]
        session_stats = self.session_stats.get(session_id, {})
        session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
        
        # Get session specification
        session_spec = self.session_manager.get_session_spec(session_type)
        
        # Calculate archaeological zones
        if session_stats:
            zone_framework = self.zone_calculator.calculate_zones_for_session(
                session_stats['session_high'], session_stats['session_low']
            )
        else:
            zone_framework = {"error": "No session stats available"}
        
        return {
            "session_id": session_id,
            "session_type": session_type,
            "session_spec": session_spec.__dict__ if session_spec else None,
            "session_stats": session_stats,
            "total_events": len(nodes),
            "archaeological_zones": zone_framework,
            "available_features": list(nodes.columns) if len(nodes) > 0 else []
        }
    
    def list_sessions(self) -> List[str]:
        """List all available sessions with type information"""
        session_list = []
        for session_id in self.sessions.keys():
            session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
            session_stats = self.session_stats.get(session_id, {})
            range_info = f" ({session_stats.get('session_range', 0):.1f} pts)" if session_stats else ""
            session_list.append(f"{session_id} [{session_type}]{range_info}")
        return session_list
    
    def session_info(self, session_id: str) -> Dict[str, Any]:
        """Get enhanced information about a specific session"""
        return self.get_enhanced_session_info(session_id)

# Interactive CLI wrapper
def run_enhanced_interactive_query():
    """Run interactive query session with price relativity"""
    print("🚀 IRONFORGE Enhanced Temporal Query Engine")
    print("🏛️ Price Relativity & Archaeological Zone Analysis")
    print("=" * 60)
    
    engine = EnhancedTemporalQueryEngine()
    
    print("\n💡 Example questions (Enhanced with Price Relativity):")
    print("• What happens after a 40% zone event?")
    print("• Show me Theory B precision events")
    print("• What happens after high liquidity spikes?")
    print("• When session starts with expansion, what's the final range?")
    print("• Find archaeological zone patterns")
    print("• Show me temporal non-locality events")
    print("\n📋 Commands: 'list', 'info <session>', 'help', 'quit'")
    print("\n🏛️ Price Relativity Features:")
    print("• Archaeological Zone Analysis (40%, 60%, 80%)")
    print("• Theory B Temporal Non-Locality Detection")
    print("• Session Progress Percentages")
    print("• Dual Time Tracking (Absolute + Relative)")
    
    while True:
        try:
            question = input("\n🤔 Ask your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'list':
                sessions = engine.list_sessions()
                print(f"\n📊 Available sessions ({len(sessions)}):")
                for session in sessions[:10]:  # Show first 10
                    print(f"  • {session}")
                if len(sessions) > 10:
                    print(f"  ... and {len(sessions) - 10} more")
            elif question.lower().startswith('info '):
                session_id = question[5:].strip()
                info = engine.session_info(session_id)
                print(f"\n📋 Enhanced Session Info:")
                for key, value in info.items():
                    if key == 'archaeological_zones' and isinstance(value, dict):
                        print(f"  {key}:")
                        zones = value.get('zones', {})
                        for zone_pct, zone_data in zones.items():
                            destiny = " ⭐" if zone_data.get('is_dimensional_destiny', False) else ""
                            print(f"    {zone_pct}: {zone_data.get('level', 0):.2f}{destiny}")
                    else:
                        print(f"  {key}: {value}")
            elif question.lower() == 'help':
                print("\n💡 Ask enhanced temporal questions with price relativity!")
                print("Examples:")
                print("• 'What happens after X?' - Temporal sequence analysis with zones")
                print("• 'Show me archaeological zone events' - Zone pattern analysis")
                print("• 'Find Theory B precision events' - Temporal non-locality detection")
                print("• 'When session starts with Y?' - Opening pattern analysis")
                print("• 'Show me relative positioning patterns' - Price relativity analysis")
                print("\n🏛️ Archaeological Zones: 40% (dimensional destiny), 60% (resistance confluence), 80% (momentum threshold)")
                print("⚡ Theory B: Events positioning with 7.55-point precision to eventual completion")
            else:
                result = engine.ask(question)
                print(f"\n🎯 Results:")
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 5:
                        print(f"  {key}: {len(value)} items (showing first 5)")
                        for item in value[:5]:
                            print(f"    • {item}")
                    else:
                        print(f"  {key}: {value}")
                        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_enhanced_interactive_query()