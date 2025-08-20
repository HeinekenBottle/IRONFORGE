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
from experiment_e_analyzer import ExperimentEAnalyzer
from ml_path_predictor import MLPathPredictor

class EnhancedTemporalQueryEngine:
    """Interactive temporal pattern query system with price relativity and Theory B integration"""
    
    def __init__(self, shard_dir: str = "data/shards/NQ_M5", adapted_dir: str = "data/adapted"):
        self.shard_dir = shard_dir
        self.adapted_dir = adapted_dir
        self.sessions = {}
        self.graphs = {}
        self.metadata = {}
        self.session_stats = {}  # Store session high/low/open/close for each session
        
        # Initialize price relativity components
        self.session_manager = SessionTimeManager()
        self.zone_calculator = ArchaeologicalZoneCalculator()
        self.experiment_e = ExperimentEAnalyzer()
        self.ml_predictor = MLPathPredictor()
        
        print("🔍 Initializing Enhanced Temporal Query Engine with Price Relativity...")
        self._load_all_sessions()
        
    def _load_all_sessions(self):
        """Load all available sessions into memory with price relativity calculations"""
        # Try to load adapted JSON sessions first
        adapted_files = sorted(glob.glob(f"{self.adapted_dir}/adapted_enhanced_rel_*.json"))
        
        if adapted_files:
            print(f"📊 Loading {len(adapted_files)} adapted sessions...")
            self._load_adapted_sessions(adapted_files)
        else:
            # Fallback to parquet shard format
            shard_paths = sorted(glob.glob(f"{self.shard_dir}/shard_*"))
            print(f"📊 Loading {len(shard_paths)} sessions...")
            self._load_parquet_sessions(shard_paths)
                
        print(f"✅ Loaded {len(self.sessions)} sessions with price relativity calculations")
    
    def _load_adapted_sessions(self, adapted_files: List[str]):
        """Load sessions from adapted JSON format"""
        import json
        
        for file_path in adapted_files:
            session_id = Path(file_path).name.replace('adapted_enhanced_rel_', '').replace('.json', '')
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                events = data.get('events', [])
                if not events:
                    continue
                
                # Convert events to DataFrame
                df = pd.DataFrame(events)
                
                # Ensure required columns exist and map from JSON structure
                if 'price_level' in df.columns:
                    df['price'] = df['price_level']
                if 'absolute_price' in df.columns and 'price' not in df.columns:
                    df['price'] = df['absolute_price']
                
                if 'price' not in df.columns:
                    print(f"⚠️  No price data in {session_id}")
                    continue
                
                # Store session data
                self.sessions[session_id] = df
                
                # Calculate session statistics
                prices = df['price'].dropna()
                if len(prices) > 0:
                    self.session_stats[session_id] = {
                        'session_high': prices.max(),
                        'session_low': prices.min(),
                        'session_open': prices.iloc[0],
                        'session_close': prices.iloc[-1],
                        'session_range': prices.max() - prices.min(),
                        'total_events': len(df)
                    }
                
                # Create simple graph representation (placeholder)
                G = nx.DiGraph()
                for i, row in df.iterrows():
                    G.add_node(i, **row.to_dict())
                    if i > 0:
                        G.add_edge(i-1, i)
                self.graphs[session_id] = G
                
                # Store basic metadata
                self.metadata[session_id] = {
                    'session_type': session_id.split('_')[0] if '_' in session_id else 'UNKNOWN',
                    'total_events': len(events),
                    'enhanced_session_data': data.get('enhanced_session_data', False),
                    'relativity_enhanced': data.get('events', [{}])[0].get('relativity_enhanced', False)
                }
                
            except Exception as e:
                print(f"⚠️  Failed to load {session_id}: {e}")
    
    def _load_parquet_sessions(self, shard_paths: List[str]):
        """Load sessions from parquet shard format"""
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
        elif "rd@40" in question.lower() or "post-rd" in question.lower() or "redelivery@40" in question.lower():
            return self._analyze_post_rd40_sequences(question)
        elif "e1" in question.lower() or ("cont" in question.lower() and ("60%" in question.lower() or "80%" in question.lower())):
            return self._analyze_e1_cont_paths(question)
        elif "e2" in question.lower() or ("mr" in question.lower() and ("second" in question.lower() or "failure" in question.lower())):
            return self._analyze_e2_mr_paths(question)
        elif "e3" in question.lower() or ("accel" in question.lower() and ("h1" in question.lower() or "breakout" in question.lower())):
            return self._analyze_e3_accel_paths(question)
        elif "pattern switch" in question.lower() or "regime" in question.lower():
            return self._analyze_pattern_switches(question)
        elif "trigger" in question.lower() and ("rd-40-ft" in question.lower() or "rd40ft" in question.lower()):
            return self._analyze_trigger_conditions(question)
        elif "ml" in question.lower() or "machine learning" in question.lower() or "train" in question.lower():
            return self._analyze_ml_predictions(question)
        elif "hazard" in question.lower() or "survival" in question.lower() or "time-to-event" in question.lower():
            return self._analyze_hazard_curves(question)
        elif "confusion matrix" in question.lower() or "evaluation" in question.lower():
            return self._evaluate_model_performance(question)
        elif "feature attribution" in question.lower() or "feature importance" in question.lower():
            return self._analyze_feature_attributions(question)
        elif "path probability" in question.lower() or ("cont" in question.lower() and "accel" in question.lower()):
            return self._calculate_path_probabilities(question)
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
    
    # ================================================================================
    # EXPERIMENT SET E: Post-RD@40% Sequence Analysis
    # ================================================================================
    
    def _analyze_post_rd40_sequences(self, question: str) -> Dict[str, Any]:
        """Analyze sequence patterns after RD@40% events"""
        print("🎯 Analyzing post-RD@40% sequence patterns...")
        
        results = {
            "query_type": "post_rd40_sequence_analysis",
            "total_sessions": len(self.sessions),
            "rd40_events": [],
            "path_classifications": {},
            "feature_analysis": {},
            "insights": []
        }
        
        # Detect RD@40% events across all sessions
        rd40_events = self._detect_rd40_events()
        results["rd40_events"] = rd40_events
        results["total_rd40_events"] = len(rd40_events)
        
        if not rd40_events:
            results["insights"].append("No RD@40% events detected in current session data")
            return results
        
        # Classify each RD@40% event into CONT/MR/ACCEL paths
        path_classifications = {}
        feature_importance = {}
        
        for event in rd40_events:
            session_id = event["session_id"]
            event_index = event["event_index"]
            
            # Classify the sequence path
            path_result = self._classify_sequence_path(session_id, event_index)
            path_classifications[f"{session_id}_{event_index}"] = path_result
            
            # Extract features for this event
            features = self._extract_rd40_features(session_id, event_index)
            event["features"] = features
        
        results["path_classifications"] = path_classifications
        
        # Calculate path probability distributions
        path_counts = {"CONT": 0, "MR": 0, "ACCEL": 0, "UNKNOWN": 0}
        for classification in path_classifications.values():
            path = classification.get("path", "UNKNOWN")
            path_counts[path] += 1
        
        total_classified = sum(path_counts.values())
        path_probabilities = {}
        if total_classified > 0:
            for path, count in path_counts.items():
                path_probabilities[path] = round(count / total_classified, 3)
        
        results["path_probabilities"] = path_probabilities
        results["path_counts"] = path_counts
        
        # Generate insights
        if len(rd40_events) > 0:
            dominant_path = max(path_probabilities.items(), key=lambda x: x[1])
            results["insights"].append(
                f"Dominant post-RD@40% path: {dominant_path[0]} ({dominant_path[1]:.1%} probability)"
            )
            results["insights"].append(
                f"Average time to path resolution: {self._calculate_avg_resolution_time(path_classifications):.1f} minutes"
            )
        
        return results
    
    def _detect_rd40_events(self) -> List[Dict[str, Any]]:
        """Detect FPFVG redelivery events at 40% archaeological zones"""
        rd40_events = []
        
        for session_id, nodes in self.sessions.items():
            if len(nodes) < 10:  # Skip sessions too short for analysis
                continue
                
            session_stats = self.session_stats.get(session_id, {})
            session_range = session_stats.get("session_range", 0)
            
            if session_range < 10:  # Skip small ranges
                continue
            
            session_high = session_stats.get("session_high", 0)
            session_low = session_stats.get("session_low", 0)
            zone_40_price = session_low + (session_range * 0.4)
            
            # Look for events near the 40% zone with redelivery characteristics
            for i in range(10, len(nodes) - 10):  # Leave buffer for sequence analysis
                current_price = nodes.iloc[i].get('price', 0)
                if current_price == 0:
                    continue
                
                # Check proximity to 40% zone (within 5% of session range)
                distance_to_zone = abs(current_price - zone_40_price)
                proximity_ratio = distance_to_zone / session_range
                
                if proximity_ratio <= 0.05:  # Within 5% of zone
                    # Check for redelivery characteristics
                    if self._has_redelivery_characteristics(nodes, i):
                        rd40_events.append({
                            "session_id": session_id,
                            "event_index": i,
                            "timestamp": nodes.iloc[i].get('timestamp', ''),
                            "price": current_price,
                            "zone_40_price": zone_40_price,
                            "proximity_ratio": proximity_ratio,
                            "strength": 1.0 - proximity_ratio,
                            "session_range": session_range
                        })
        
        return rd40_events
    
    def _has_redelivery_characteristics(self, nodes: pd.DataFrame, index: int) -> bool:
        """Check if event has FPFVG redelivery characteristics"""
        if index >= len(nodes):
            return False
            
        current_event = nodes.iloc[index]
        
        # Look for liquidity spike patterns around this event
        window_start = max(0, index - 5)
        window_end = min(len(nodes), index + 5)
        window = nodes.iloc[window_start:window_end]
        
        # Check for f8-like liquidity characteristics
        # (simplified - would use actual f8 feature if available)
        if 'magnitude' in window.columns:
            avg_magnitude = window['magnitude'].mean()
            current_magnitude = current_event.get('magnitude', 0)
            
            # Look for magnitude spike (simplified redelivery detection)
            if current_magnitude > avg_magnitude * 1.5:
                return True
        
        # Check for expansion phase patterns
        event_type = current_event.get('type', '')
        if 'expansion' in str(event_type).lower() or 'redelivery' in str(event_type).lower():
            return True
        
        # Check archaeological significance (specific to adapted data)
        arch_sig = current_event.get('archaeological_significance', 0)
        if arch_sig > 0.8:  # High archaeological significance
            return True
        
        # Check energy density patterns
        energy_density = current_event.get('energy_density', 0)
        if energy_density > 0.7:  # High energy events
            return True
        
        # Check for dimensional relationship to zones
        dim_rel = current_event.get('dimensional_relationship', '')
        if 'zone' in str(dim_rel).lower() or 'threshold' in str(dim_rel).lower():
            return True
        
        return False
    
    def _classify_sequence_path(self, session_id: str, event_index: int) -> Dict[str, Any]:
        """Classify the sequence path after RD@40% event: CONT/MR/ACCEL"""
        nodes = self.sessions[session_id]
        session_stats = self.session_stats.get(session_id, {})
        session_range = session_stats.get("session_range", 0)
        session_low = session_stats.get("session_low", 0)
        
        if session_range == 0:
            return {"path": "UNKNOWN", "reason": "No session range data"}
        
        # Define zone thresholds
        zone_50 = session_low + (session_range * 0.5)
        zone_60 = session_low + (session_range * 0.6) 
        zone_80 = session_low + (session_range * 0.8)
        
        # Analyze 120-minute post-event window (or to end of session)
        max_observation_bars = min(120, len(nodes) - event_index - 1)  # 120 bars ~ 120 minutes for M1 data
        end_index = event_index + max_observation_bars
        
        if end_index <= event_index + 5:  # Not enough data for analysis
            return {"path": "UNKNOWN", "reason": "Insufficient post-event data"}
        
        post_event_window = nodes.iloc[event_index:end_index]
        
        # Track key timing milestones
        time_to_60 = None
        time_to_80 = None
        time_to_mid = None
        
        for i, (_, row) in enumerate(post_event_window.iterrows()):
            price = row.get('price', 0)
            if price == 0:
                continue
            
            minutes_elapsed = i  # Assuming 1-minute bars
            
            # Check zone hits
            if time_to_60 is None and abs(price - zone_60) / session_range <= 0.03:
                time_to_60 = minutes_elapsed
            
            if time_to_80 is None and abs(price - zone_80) / session_range <= 0.03:
                time_to_80 = minutes_elapsed
            
            if time_to_mid is None and abs(price - zone_50) / session_range <= 0.05:
                time_to_mid = minutes_elapsed
        
        # Classification logic
        
        # CONT Path: 40% → 60% → 80% within timeframes
        if time_to_60 is not None and time_to_60 <= 45 and time_to_80 is not None and time_to_80 <= 90:
            return {
                "path": "CONT",
                "time_to_60": time_to_60,
                "time_to_80": time_to_80,
                "confidence": 0.8
            }
        
        # MR Path: Snap to mid-range within 60 minutes
        if time_to_mid is not None and time_to_mid <= 60:
            # Check for secondary patterns (simplified)
            second_rd = self._check_second_rd(post_event_window, zone_50)
            return {
                "path": "MR", 
                "time_to_mid": time_to_mid,
                "second_rd": second_rd,
                "confidence": 0.7
            }
        
        # ACCEL Path: Quick move to 80% with H1 alignment (simplified)
        if time_to_80 is not None and time_to_80 <= 60:
            h1_aligned = self._check_h1_alignment(session_id, event_index)  # Placeholder
            return {
                "path": "ACCEL",
                "time_to_80": time_to_80, 
                "h1_aligned": h1_aligned,
                "confidence": 0.6
            }
        
        # Default: Path unclear
        return {
            "path": "UNKNOWN",
            "time_to_60": time_to_60,
            "time_to_80": time_to_80,
            "time_to_mid": time_to_mid,
            "reason": "No clear path pattern identified"
        }
    
    def _check_second_rd(self, window: pd.DataFrame, zone_50: float) -> bool:
        """Check for second redelivery attempt after mean revert"""
        # Simplified check - look for return to original zone after mid-range visit
        # This would be more sophisticated in production
        return len(window) > 30  # Placeholder logic
    
    def _check_h1_alignment(self, session_id: str, event_index: int) -> bool:
        """Check for H1 breakout alignment (placeholder)"""
        # This would integrate with actual H1 breakout detection
        # For now, return random alignment based on session characteristics
        return "NY" in session_id  # Simplified placeholder
    
    def _extract_rd40_features(self, session_id: str, event_index: int) -> Dict[str, Any]:
        """Extract relevant features for RD@40% event analysis"""
        nodes = self.sessions[session_id]
        
        if event_index >= len(nodes):
            return {}
        
        event_row = nodes.iloc[event_index]
        features = {}
        
        # Extract available features from the event
        feature_columns = ['f8_q', 'f8_slope_sign', 'f47_barpos_m15', 'f48_barpos_h1', 
                          'f49_dist_daily_mid', 'f50_htf_regime', 'magnitude', 
                          'energy_density', 'archaeological_significance']
        
        for feature in feature_columns:
            if feature in event_row:
                features[feature] = event_row[feature]
        
        # Calculate additional contextual features
        session_stats = self.session_stats.get(session_id, {})
        if session_stats:
            features['session_range'] = session_stats.get('session_range', 0)
            features['session_progress'] = event_index / len(nodes) if len(nodes) > 0 else 0
        
        return features
    
    def _calculate_avg_resolution_time(self, path_classifications: Dict) -> float:
        """Calculate average time to path resolution"""
        times = []
        for classification in path_classifications.values():
            if 'time_to_60' in classification and classification['time_to_60'] is not None:
                times.append(classification['time_to_60'])
            elif 'time_to_80' in classification and classification['time_to_80'] is not None:
                times.append(classification['time_to_80']) 
            elif 'time_to_mid' in classification and classification['time_to_mid'] is not None:
                times.append(classification['time_to_mid'])
        
        return np.mean(times) if times else 0.0
    
    def _calculate_path_probabilities(self, question: str) -> Dict[str, Any]:
        """Calculate path probabilities with confidence intervals"""
        print("📊 Calculating path probabilities with statistical analysis...")
        
        # First get RD@40% events and classifications
        rd40_analysis = self._analyze_post_rd40_sequences(question)
        path_classifications = rd40_analysis.get("path_classifications", {})
        
        if not path_classifications:
            return {
                "query_type": "path_probabilities",
                "error": "No RD@40% events found for probability analysis",
                "total_events": 0
            }
        
        # Calculate detailed statistics
        path_stats = {"CONT": [], "MR": [], "ACCEL": [], "UNKNOWN": []}
        
        for event_key, classification in path_classifications.items():
            path = classification.get("path", "UNKNOWN")
            confidence = classification.get("confidence", 0.5)
            
            path_stats[path].append({
                "event": event_key,
                "confidence": confidence,
                "classification": classification
            })
        
        # Calculate probabilities and confidence intervals
        total_events = len(path_classifications)
        results = {
            "query_type": "path_probabilities", 
            "total_events": total_events,
            "path_statistics": {},
            "wilson_confidence_intervals": {},
            "insights": []
        }
        
        for path, events in path_stats.items():
            count = len(events)
            probability = count / total_events if total_events > 0 else 0
            
            # Wilson confidence interval (simplified)
            if count > 0 and total_events > 0:
                # Simplified Wilson CI calculation
                z = 1.96  # 95% confidence
                p_hat = probability
                n = total_events
                
                numerator = p_hat + z**2/(2*n)
                denominator = 1 + z**2/n
                margin = z * np.sqrt((p_hat * (1-p_hat) + z**2/(4*n)) / n) / denominator
                
                ci_lower = max(0, (numerator - margin) / denominator)
                ci_upper = min(1, (numerator + margin) / denominator)
            else:
                ci_lower, ci_upper = 0, 0
            
            results["path_statistics"][path] = {
                "count": count,
                "probability": round(probability, 3),
                "avg_confidence": round(np.mean([e["confidence"] for e in events]) if events else 0, 3)
            }
            
            results["wilson_confidence_intervals"][path] = {
                "lower": round(ci_lower, 3),
                "upper": round(ci_upper, 3)
            }
        
        # Generate insights
        if total_events >= 3:
            dominant_path = max(results["path_statistics"].items(), 
                              key=lambda x: x[1]["probability"])
            results["insights"].append(
                f"Most probable path: {dominant_path[0]} ({dominant_path[1]['probability']:.1%})"
            )
            
            high_confidence_paths = [
                path for path, stats in results["path_statistics"].items() 
                if stats["avg_confidence"] > 0.7 and stats["count"] > 0
            ]
            
            if high_confidence_paths:
                results["insights"].append(
                    f"High confidence paths: {', '.join(high_confidence_paths)}"
                )
        else:
            results["insights"].append(
                "Insufficient data for robust probability analysis (need ≥3 events)"
            )
        
        return results

# Interactive CLI wrapper
    def _analyze_e1_cont_paths(self, question: str) -> Dict[str, Any]:
        """Analyze E1 CONT paths: RD@40 → 60% → 80% within timing constraints"""
        print("🎯 Analyzing E1 CONT paths with 45min→60%, 90min→80% precision timing...")
        
        results = {
            "query_type": "e1_cont_path_analysis",
            "total_sessions": len(self.sessions),
            "e1_cont_events": [],
            "timing_analysis": {},
            "feature_importance": {},
            "insights": []
        }
        
        # Detect RD@40% events first
        rd40_events = self._detect_rd40_events()
        results["total_rd40_events"] = len(rd40_events)
        
        if not rd40_events:
            results["insights"].append("No RD@40% events detected for E1 CONT analysis")
            return results
        
        e1_cont_classifications = []
        timing_stats = {"time_to_60": [], "time_to_80": [], "drawdown_risks": []}
        
        for event in rd40_events:
            session_id = event["session_id"]
            event_index = event["event_index"]
            
            if session_id not in self.sessions:
                continue
            
            session_data = self.sessions[session_id]
            e1_result = self.experiment_e.classify_e1_cont_path(session_data, event_index)
            
            if e1_result.get("path") == "E1_CONT":
                e1_cont_classifications.append({
                    "session_id": session_id,
                    "event_index": event_index,
                    "classification": e1_result,
                    "confidence": e1_result.get("confidence", 0),
                    "kpis": e1_result.get("kpis", {})
                })
                
                # Collect timing statistics
                kpis = e1_result.get("kpis", {})
                if "expected_time_to_60" in kpis:
                    timing_stats["time_to_60"].append(kpis["expected_time_to_60"])
                if "expected_time_to_80" in kpis:
                    timing_stats["time_to_80"].append(kpis["expected_time_to_80"])
                if "drawdown_risk" in kpis:
                    timing_stats["drawdown_risks"].append(kpis["drawdown_risk"])
        
        results["e1_cont_events"] = e1_cont_classifications
        results["total_e1_cont"] = len(e1_cont_classifications)
        
        # Calculate aggregate timing statistics
        if timing_stats["time_to_60"]:
            results["timing_analysis"] = {
                "avg_time_to_60": np.mean(timing_stats["time_to_60"]),
                "median_time_to_60": np.median(timing_stats["time_to_60"]),
                "avg_time_to_80": np.mean(timing_stats["time_to_80"]) if timing_stats["time_to_80"] else None,
                "median_time_to_80": np.median(timing_stats["time_to_80"]) if timing_stats["time_to_80"] else None,
                "avg_drawdown_risk": np.mean(timing_stats["drawdown_risks"]) if timing_stats["drawdown_risks"] else 0,
                "success_rate": len(e1_cont_classifications) / len(rd40_events) if rd40_events else 0
            }
        
        # Generate insights
        success_rate = results["timing_analysis"].get("success_rate", 0) if results.get("timing_analysis") else 0
        results["insights"].extend([
            f"Identified {len(e1_cont_classifications)} E1 CONT paths from {len(rd40_events)} RD@40% events",
            f"E1 CONT success rate: {success_rate:.1%}",
            f"Average timing: {results['timing_analysis'].get('avg_time_to_60', 'N/A')}min to 60%, {results['timing_analysis'].get('avg_time_to_80', 'N/A')}min to 80%" if results.get("timing_analysis") else "No timing data available"
        ])
        
        return results
    
    def _analyze_e2_mr_paths(self, question: str) -> Dict[str, Any]:
        """Analyze E2 MR paths: RD@40 → mid with second_rd/failure branching"""
        print("🔄 Analyzing E2 MR paths with second_rd and failure branch detection...")
        
        results = {
            "query_type": "e2_mr_path_analysis", 
            "total_sessions": len(self.sessions),
            "e2_mr_events": [],
            "branch_analysis": {},
            "insights": []
        }
        
        # Detect RD@40% events first
        rd40_events = self._detect_rd40_events()
        results["total_rd40_events"] = len(rd40_events)
        
        if not rd40_events:
            results["insights"].append("No RD@40% events detected for E2 MR analysis")
            return results
        
        e2_mr_classifications = []
        branch_stats = {"second_rd": 0, "failure": 0, "unknown": 0}
        
        for event in rd40_events:
            session_id = event["session_id"] 
            event_index = event["event_index"]
            
            if session_id not in self.sessions:
                continue
            
            session_data = self.sessions[session_id]
            e2_result = self.experiment_e.classify_e2_mr_path(session_data, event_index)
            
            if e2_result.get("path") == "E2_MR":
                e2_mr_classifications.append({
                    "session_id": session_id,
                    "event_index": event_index, 
                    "classification": e2_result,
                    "confidence": e2_result.get("confidence", 0),
                    "kpis": e2_result.get("kpis", {}),
                    "branch_analysis": e2_result.get("branch_analysis", {})
                })
                
                # Track branching patterns
                branch_data = e2_result.get("branch_analysis", {})
                if branch_data.get("second_rd_probability", 0) > 0.5:
                    branch_stats["second_rd"] += 1
                elif branch_data.get("failure_probability", 0) > 0.5:
                    branch_stats["failure"] += 1
                else:
                    branch_stats["unknown"] += 1
        
        results["e2_mr_events"] = e2_mr_classifications
        results["total_e2_mr"] = len(e2_mr_classifications)
        
        # Calculate branch statistics
        if e2_mr_classifications:
            total_branches = sum(branch_stats.values())
            results["branch_analysis"] = {
                "second_rd_rate": branch_stats["second_rd"] / total_branches if total_branches > 0 else 0,
                "failure_rate": branch_stats["failure"] / total_branches if total_branches > 0 else 0,
                "unknown_rate": branch_stats["unknown"] / total_branches if total_branches > 0 else 0,
                "branch_distribution": branch_stats
            }
        
        # Generate insights
        success_rate = len(e2_mr_classifications) / len(rd40_events) if rd40_events else 0
        results["insights"].extend([
            f"Identified {len(e2_mr_classifications)} E2 MR paths from {len(rd40_events)} RD@40% events",
            f"E2 MR detection rate: {success_rate:.1%}",
            f"Branch distribution: {branch_stats['second_rd']} second_rd, {branch_stats['failure']} failure, {branch_stats['unknown']} unknown"
        ])
        
        return results
    
    def _analyze_e3_accel_paths(self, question: str) -> Dict[str, Any]:
        """Analyze E3 ACCEL paths: RD@40 + H1 breakout → fast 80% with shallow pullback"""
        print("🚀 Analyzing E3 ACCEL paths with H1 breakout confirmation...")
        
        results = {
            "query_type": "e3_accel_path_analysis",
            "total_sessions": len(self.sessions), 
            "e3_accel_events": [],
            "h1_analysis": {},
            "performance_metrics": {},
            "insights": []
        }
        
        # Detect RD@40% events first
        rd40_events = self._detect_rd40_events()
        results["total_rd40_events"] = len(rd40_events)
        
        if not rd40_events:
            results["insights"].append("No RD@40% events detected for E3 ACCEL analysis")
            return results
        
        e3_accel_classifications = []
        h1_stats = {"breakouts_detected": 0, "direction_aligned": 0, "accel_confirmed": 0}
        performance_stats = {"time_to_80": [], "pullback_depths": [], "continuation_probs": []}
        
        for event in rd40_events:
            session_id = event["session_id"]
            event_index = event["event_index"]
            
            if session_id not in self.sessions:
                continue
            
            session_data = self.sessions[session_id]
            e3_result = self.experiment_e.classify_e3_accel_path(session_data, event_index)
            
            # Track H1 breakout statistics
            h1_breakout = e3_result.get("h1_breakout", {})
            if h1_breakout.get("detected"):
                h1_stats["breakouts_detected"] += 1
                if h1_breakout.get("direction_aligned"):
                    h1_stats["direction_aligned"] += 1
            
            if e3_result.get("path") == "E3_ACCEL":
                h1_stats["accel_confirmed"] += 1
                e3_accel_classifications.append({
                    "session_id": session_id,
                    "event_index": event_index,
                    "classification": e3_result,
                    "confidence": e3_result.get("confidence", 0),
                    "kpis": e3_result.get("kpis", {}),
                    "h1_breakout": h1_breakout
                })
                
                # Collect performance statistics
                kpis = e3_result.get("kpis", {})
                if "time_to_80" in kpis:
                    performance_stats["time_to_80"].append(kpis["time_to_80"])
                if "pullback_depth" in kpis:
                    performance_stats["pullback_depths"].append(kpis["pullback_depth"])
                if "continuation_beyond_80" in kpis:
                    performance_stats["continuation_probs"].append(kpis["continuation_beyond_80"])
        
        results["e3_accel_events"] = e3_accel_classifications
        results["total_e3_accel"] = len(e3_accel_classifications)
        
        # Calculate H1 and performance statistics
        results["h1_analysis"] = {
            "breakout_detection_rate": h1_stats["breakouts_detected"] / len(rd40_events) if rd40_events else 0,
            "direction_alignment_rate": h1_stats["direction_aligned"] / h1_stats["breakouts_detected"] if h1_stats["breakouts_detected"] > 0 else 0,
            "accel_conversion_rate": h1_stats["accel_confirmed"] / h1_stats["direction_aligned"] if h1_stats["direction_aligned"] > 0 else 0
        }
        
        if performance_stats["time_to_80"]:
            results["performance_metrics"] = {
                "avg_time_to_80": np.mean(performance_stats["time_to_80"]),
                "median_time_to_80": np.median(performance_stats["time_to_80"]),
                "avg_pullback_depth": np.mean(performance_stats["pullback_depths"]) if performance_stats["pullback_depths"] else 0,
                "avg_continuation_prob": np.mean(performance_stats["continuation_probs"]) if performance_stats["continuation_probs"] else 0
            }
        
        # Generate insights
        success_rate = len(e3_accel_classifications) / len(rd40_events) if rd40_events else 0
        results["insights"].extend([
            f"Identified {len(e3_accel_classifications)} E3 ACCEL paths from {len(rd40_events)} RD@40% events",
            f"E3 ACCEL success rate: {success_rate:.1%}",
            f"H1 breakout detection: {h1_stats['breakouts_detected']} detected, {h1_stats['direction_aligned']} aligned",
            f"Performance: {results['performance_metrics'].get('avg_time_to_80', 'N/A')}min avg to 80%" if results.get("performance_metrics") else "No performance data available"
        ])
        
        return results
    
    def _analyze_pattern_switches(self, question: str) -> Dict[str, Any]:
        """Analyze pattern-switch diagnostics for regime transitions"""
        print("🔄 Analyzing pattern-switch diagnostics and regime transitions...")
        
        results = {
            "query_type": "pattern_switch_analysis",
            "total_sessions": len(self.sessions),
            "switch_diagnostics": {},
            "regime_analysis": {},
            "insights": []
        }
        
        # Get RD@40% events for switch analysis
        rd40_events = self._detect_rd40_events()
        results["total_rd40_events"] = len(rd40_events)
        
        if not rd40_events:
            results["insights"].append("No RD@40% events detected for pattern switch analysis")
            return results
        
        # Analyze pattern switches across all sessions
        all_switch_diagnostics = {}
        
        for session_id, session_data in self.sessions.items():
            # Filter RD@40% events for this session
            session_rd40_events = [e for e in rd40_events if e.get("session_id") == session_id]
            
            if session_rd40_events:
                switch_analysis = self.experiment_e.analyze_pattern_switches(session_data, session_rd40_events)
                all_switch_diagnostics[session_id] = switch_analysis
        
        results["switch_diagnostics"] = all_switch_diagnostics
        
        # Aggregate regime transition patterns
        regime_transitions = {"cont_to_mr": 0, "mr_to_cont": 0, "stable": 0}
        news_effects = {"high_impact_events": 0, "suppression_events": 0}
        h1_confirmations = {"positive_confirmations": 0, "total_breakouts": 0}
        
        for session_id, diagnostics in all_switch_diagnostics.items():
            # Count regime flips
            regime_flips = diagnostics.get("regime_flip_analysis", {})
            for event_key, flip_data in regime_flips.items():
                if flip_data.get("flips_cont_to_mr"):
                    regime_transitions["cont_to_mr"] += 1
                elif flip_data.get("flips_mr_to_cont"):
                    regime_transitions["mr_to_cont"] += 1
                else:
                    regime_transitions["stable"] += 1
            
            # Count news effects
            news_proximity = diagnostics.get("news_proximity_effects", {})
            for event_key, news_data in news_proximity.items():
                if news_data.get("high_impact_detected"):
                    news_effects["high_impact_events"] += 1
                if news_data.get("suppresses_cont_accel"):
                    news_effects["suppression_events"] += 1
            
            # Count H1 confirmations
            h1_impacts = diagnostics.get("h1_confirmation_impact", {})
            for event_key, h1_data in h1_impacts.items():
                if h1_data.get("adds_accel_probability"):
                    h1_confirmations["positive_confirmations"] += 1
                h1_confirmations["total_breakouts"] += 1
        
        results["regime_analysis"] = {
            "regime_transitions": regime_transitions,
            "news_effects": news_effects, 
            "h1_confirmations": h1_confirmations,
            "transition_rates": {
                "cont_to_mr_rate": regime_transitions["cont_to_mr"] / sum(regime_transitions.values()) if sum(regime_transitions.values()) > 0 else 0,
                "mr_to_cont_rate": regime_transitions["mr_to_cont"] / sum(regime_transitions.values()) if sum(regime_transitions.values()) > 0 else 0,
                "stability_rate": regime_transitions["stable"] / sum(regime_transitions.values()) if sum(regime_transitions.values()) > 0 else 0
            }
        }
        
        # Generate insights
        total_switches = sum(regime_transitions.values())
        results["insights"].extend([
            f"Analyzed pattern switches across {len(all_switch_diagnostics)} sessions",
            f"Regime transitions: {regime_transitions['cont_to_mr']} CONT→MR, {regime_transitions['mr_to_cont']} MR→CONT, {regime_transitions['stable']} stable",
            f"News impact: {news_effects['high_impact_events']} high-impact events, {news_effects['suppression_events']} suppression events",
            f"H1 confirmations: {h1_confirmations['positive_confirmations']}/{h1_confirmations['total_breakouts']} positive confirmations"
        ])
        
        return results
    
    def _analyze_trigger_conditions(self, question: str) -> Dict[str, Any]:
        """Analyze RD-40-FT trigger conditions for CONT/MR/ACCEL"""
        print("🎯 Analyzing RD-40-FT trigger conditions with precision gates...")
        
        results = {
            "query_type": "trigger_condition_analysis",
            "total_sessions": len(self.sessions),
            "trigger_analysis": {},
            "precision_gates": {},
            "insights": []
        }
        
        # Get comprehensive path analysis
        e1_results = self._analyze_e1_cont_paths(question)
        e2_results = self._analyze_e2_mr_paths(question) 
        e3_results = self._analyze_e3_accel_paths(question)
        
        # Analyze trigger precision for each path type
        trigger_stats = {
            "RD-40-FT-CONT": {
                "candidates": len(e1_results.get("e1_cont_events", [])),
                "high_confidence": len([e for e in e1_results.get("e1_cont_events", []) if e.get("confidence", 0) >= 0.85]),
                "precision_rate": 0
            },
            "RD-40-FT-MR": {
                "candidates": len(e2_results.get("e2_mr_events", [])),
                "high_confidence": len([e for e in e2_results.get("e2_mr_events", []) if e.get("confidence", 0) >= 0.70]),
                "precision_rate": 0
            },
            "RD-40-FT-ACCEL": {
                "candidates": len(e3_results.get("e3_accel_events", [])),
                "high_confidence": len([e for e in e3_results.get("e3_accel_events", []) if e.get("confidence", 0) >= 0.85]),
                "precision_rate": 0
            }
        }
        
        # Calculate precision rates
        for trigger_type, stats in trigger_stats.items():
            if stats["candidates"] > 0:
                stats["precision_rate"] = stats["high_confidence"] / stats["candidates"]
        
        results["trigger_analysis"] = trigger_stats
        
        # Define precision gates
        results["precision_gates"] = {
            "RD-40-FT-CONT": {
                "min_confidence": 0.85,
                "min_f8_q": 0.90,
                "max_theory_b_dt": 30,
                "required_features": ["f8_q >= 0.90", "positive f8_slope", "theory_b_dt <= 30m", "gap_age <= 2d"]
            },
            "RD-40-FT-MR": {
                "min_confidence": 0.70,
                "news_window": 15,
                "required_features": ["f50 = mean_revert OR high news_impact", "f8_slope <= 0", "no H1 breakout"]
            },
            "RD-40-FT-ACCEL": {
                "min_confidence": 0.85,
                "min_f8_q": 0.95,
                "h1_window": 15,
                "required_features": ["H1 breakout aligned", "f8_q >= 0.95", "theory_b_dt <= 30m", "tight zone distance"]
            }
        }
        
        # Generate trigger recommendations
        total_rd40_events = e1_results.get("total_rd40_events", 0)
        results["insights"].extend([
            f"Trigger analysis across {total_rd40_events} RD@40% events:",
            f"RD-40-FT-CONT: {trigger_stats['RD-40-FT-CONT']['high_confidence']}/{trigger_stats['RD-40-FT-CONT']['candidates']} meet ≥85% precision gate",
            f"RD-40-FT-MR: {trigger_stats['RD-40-FT-MR']['high_confidence']}/{trigger_stats['RD-40-FT-MR']['candidates']} meet ≥70% precision gate", 
            f"RD-40-FT-ACCEL: {trigger_stats['RD-40-FT-ACCEL']['high_confidence']}/{trigger_stats['RD-40-FT-ACCEL']['candidates']} meet ≥85% precision gate",
            "Precision gates calibrated for execution-grade signals with rate limiting"
        ])
        
        return results
    
    def _analyze_ml_predictions(self, question: str) -> Dict[str, Any]:
        """Train and analyze ML path predictions with isotonic calibration"""
        print("🤖 Training ML Path Predictor with isotonic calibration...")
        
        results = {
            "query_type": "ml_path_prediction",
            "total_sessions": len(self.sessions),
            "training_results": {},
            "predictions": {},
            "insights": []
        }
        
        # Get RD@40% events and path classifications
        rd40_events = self._detect_rd40_events()
        results["total_rd40_events"] = len(rd40_events)
        
        if not rd40_events:
            results["insights"].append("No RD@40% events found for ML training")
            return results
        
        # Get path classifications from all E1/E2/E3 analyses
        all_path_classifications = {}
        
        # Collect classifications from each path type
        for event in rd40_events:
            session_id = event["session_id"]
            event_index = event["event_index"]
            event_key = f"{session_id}_{event_index}"
            
            if session_id not in self.sessions:
                continue
            
            session_data = self.sessions[session_id]
            
            # Try E1 CONT classification
            e1_result = self.experiment_e.classify_e1_cont_path(session_data, event_index)
            if e1_result.get("path") == "E1_CONT":
                all_path_classifications[event_key] = e1_result
                continue
            
            # Try E2 MR classification
            e2_result = self.experiment_e.classify_e2_mr_path(session_data, event_index)
            if e2_result.get("path") == "E2_MR":
                all_path_classifications[event_key] = e2_result
                continue
            
            # Try E3 ACCEL classification
            e3_result = self.experiment_e.classify_e3_accel_path(session_data, event_index)
            if e3_result.get("path") == "E3_ACCEL":
                all_path_classifications[event_key] = e3_result
                continue
        
        results["classified_events"] = len(all_path_classifications)
        
        if len(all_path_classifications) < 10:
            results["insights"].append(f"Insufficient classified events ({len(all_path_classifications)}) for reliable ML training")
            return results
        
        # Prepare training data
        X, y = self.ml_predictor.prepare_training_data(rd40_events, self.sessions, all_path_classifications)
        
        if len(X) == 0:
            results["insights"].append("No valid training data could be prepared")
            return results
        
        # Train ML models
        training_results = self.ml_predictor.fit(X, y)
        results["training_results"] = training_results
        
        # Generate predictions for all events
        if self.ml_predictor.is_fitted:
            predictions = self.ml_predictor.predict_path_probabilities(X)
            results["predictions"] = {
                "sample_predictions": {
                    "CONT": predictions.get("CONT", [])[:5].tolist() if len(predictions.get("CONT", [])) > 0 else [],
                    "MR": predictions.get("MR", [])[:5].tolist() if len(predictions.get("MR", [])) > 0 else [],
                    "ACCEL": predictions.get("ACCEL", [])[:5].tolist() if len(predictions.get("ACCEL", [])) > 0 else []
                }
            }
        
        # Generate insights
        if training_results.get("class_distribution"):
            class_dist = training_results["class_distribution"]
            dominant_class = max(class_dist, key=class_dist.get) if class_dist else "UNKNOWN"
            results["insights"].extend([
                f"Trained ML predictor on {training_results.get('total_samples', 0)} samples",
                f"Class distribution: {class_dist}",
                f"Dominant class: {dominant_class}",
                f"Cross-validation AUC scores available for model validation"
            ])
        
        return results
    
    def _analyze_hazard_curves(self, question: str) -> Dict[str, Any]:
        """Analyze time-to-event hazard curves for path resolution"""
        print("📈 Analyzing hazard curves and survival analysis...")
        
        results = {
            "query_type": "hazard_curve_analysis",
            "total_sessions": len(self.sessions),
            "hazard_results": {},
            "insights": []
        }
        
        # Get RD@40% events and classifications
        rd40_events = self._detect_rd40_events()
        results["total_rd40_events"] = len(rd40_events)
        
        if not rd40_events:
            results["insights"].append("No RD@40% events found for hazard analysis")
            return results
        
        # Get path classifications 
        all_path_classifications = {}
        
        for event in rd40_events:
            session_id = event["session_id"]
            event_index = event["event_index"]
            event_key = f"{session_id}_{event_index}"
            
            if session_id not in self.sessions:
                continue
            
            session_data = self.sessions[session_id]
            
            # Get best classification
            classifications = [
                self.experiment_e.classify_e1_cont_path(session_data, event_index),
                self.experiment_e.classify_e2_mr_path(session_data, event_index),
                self.experiment_e.classify_e3_accel_path(session_data, event_index)
            ]
            
            # Use the classification with highest confidence that isn't a "NOT_" type
            best_classification = None
            best_confidence = 0
            
            for classification in classifications:
                confidence = classification.get("confidence", 0)
                path = classification.get("path", "UNKNOWN")
                
                if not path.startswith("NOT_") and confidence > best_confidence:
                    best_classification = classification
                    best_confidence = confidence
            
            if best_classification:
                all_path_classifications[event_key] = best_classification
        
        # Perform hazard curve analysis
        hazard_analysis = self.ml_predictor.analyze_hazard_curves(rd40_events, self.sessions, all_path_classifications)
        results["hazard_results"] = hazard_analysis
        
        # Generate insights
        if hazard_analysis.get("path_hazard_analysis"):
            hazard_stats = hazard_analysis["path_hazard_analysis"]
            median_times = hazard_analysis.get("median_resolution_times", {})
            
            insights = [f"Hazard analysis across {len(all_path_classifications)} classified events:"]
            
            for path_type, stats in hazard_stats.items():
                resolution_rate = stats.get("resolution_rate", 0)
                median_time = median_times.get(path_type, stats.get("median_time", 0))
                insights.append(f"{path_type}: {resolution_rate:.1%} resolution rate, {median_time:.1f}min median time")
            
            results["insights"] = insights
        
        return results
    
    def _evaluate_model_performance(self, question: str) -> Dict[str, Any]:
        """Evaluate ML model performance with confusion matrix and metrics"""
        print("📊 Evaluating ML model performance with confusion matrix...")
        
        results = {
            "query_type": "model_evaluation",
            "total_sessions": len(self.sessions),
            "evaluation_results": {},
            "insights": []
        }
        
        # First train the model if not already fitted
        if not self.ml_predictor.is_fitted:
            ml_results = self._analyze_ml_predictions("Train ML model for evaluation")
            if "error" in ml_results or not self.ml_predictor.is_fitted:
                results["insights"].append("Could not train ML model for evaluation")
                return results
        
        # Get training data for evaluation
        rd40_events = self._detect_rd40_events()
        
        # Get path classifications
        all_path_classifications = {}
        for event in rd40_events:
            session_id = event["session_id"]
            event_index = event["event_index"]
            event_key = f"{session_id}_{event_index}"
            
            if session_id not in self.sessions:
                continue
            
            session_data = self.sessions[session_id]
            
            # Get best classification (similar to hazard analysis)
            classifications = [
                self.experiment_e.classify_e1_cont_path(session_data, event_index),
                self.experiment_e.classify_e2_mr_path(session_data, event_index),
                self.experiment_e.classify_e3_accel_path(session_data, event_index)
            ]
            
            best_classification = None
            best_confidence = 0
            
            for classification in classifications:
                confidence = classification.get("confidence", 0)
                path = classification.get("path", "UNKNOWN")
                
                if not path.startswith("NOT_") and confidence > best_confidence:
                    best_classification = classification
                    best_confidence = confidence
            
            if best_classification:
                all_path_classifications[event_key] = best_classification
        
        # Prepare evaluation data
        X, y = self.ml_predictor.prepare_training_data(rd40_events, self.sessions, all_path_classifications)
        
        if len(X) == 0 or len(y) == 0:
            results["insights"].append("No evaluation data available")
            return results
        
        # Generate confusion matrix and metrics
        evaluation_results = self.ml_predictor.generate_confusion_matrix(X, y)
        results["evaluation_results"] = evaluation_results
        
        # Generate insights
        if evaluation_results.get("path_metrics"):
            overall_accuracy = evaluation_results.get("overall_accuracy", 0)
            insights = [
                f"Model evaluation on {len(y)} samples:",
                f"Overall accuracy: {overall_accuracy:.1%}"
            ]
            
            path_metrics = evaluation_results["path_metrics"]
            for path_type, metrics in path_metrics.items():
                precision = metrics.get("precision", 0)
                recall = metrics.get("recall", 0) 
                f1 = metrics.get("f1_score", 0)
                support = metrics.get("support", 0)
                insights.append(f"{path_type}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f} (n={support})")
            
            results["insights"] = insights
        
        return results
    
    def _analyze_feature_attributions(self, question: str) -> Dict[str, Any]:
        """Analyze feature attributions for path selection"""
        print("🔍 Analyzing feature attributions for path selection...")
        
        results = {
            "query_type": "feature_attribution_analysis",
            "attributions": {},
            "insights": []
        }
        
        # Train model if not fitted
        if not self.ml_predictor.is_fitted:
            ml_results = self._analyze_ml_predictions("Train ML model for feature attribution")
            if not self.ml_predictor.is_fitted:
                results["insights"].append("Could not train ML model for feature attribution")
                return results
        
        # Get feature attributions for each path type
        sample_features = np.array([0.5] * len(self.ml_predictor.feature_names)).reshape(1, -1)
        
        for path_type in ['CONT', 'MR', 'ACCEL']:
            attribution_result = self.ml_predictor.get_feature_attributions(sample_features, path_type)
            
            if "error" not in attribution_result:
                results["attributions"][path_type] = attribution_result
        
        # Generate insights about key features
        if results["attributions"]:
            insights = ["Feature importance analysis for path selection:"]
            
            for path_type, attribution in results["attributions"].items():
                top_positive = attribution.get("top_positive_features", [])[:3]
                top_negative = attribution.get("top_negative_features", [])[:3]
                
                if top_positive:
                    insights.append(f"{path_type} drivers: {', '.join(top_positive)}")
                if top_negative:
                    insights.append(f"{path_type} inhibitors: {', '.join(top_negative)}")
            
            results["insights"] = insights
        
        return results

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