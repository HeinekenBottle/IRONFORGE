#!/usr/bin/env python3
"""
IRONFORGE Enhanced Temporal Query Engine - Refactored
Interactive system for querying temporal patterns with archaeological zone analysis
Integrates Theory B temporal non-locality and session-aware price relativity

This is the main interface that maintains backward compatibility while using
the refactored modular architecture.
"""
from typing import Any

import numpy as np
import pandas as pd

from .price_relativity import PriceRelativityEngine
from .query_core import TemporalQueryCore
from .session_manager import SessionDataManager
from .visualization import VisualizationManager


class EnhancedTemporalQueryEngine:
    """
    Interactive temporal pattern query system with price relativity and Theory B integration
    
    This class maintains the original interface while delegating functionality to specialized modules.
    All original method signatures and behavior are preserved for backward compatibility.
    """
    
    def __init__(self, shard_dir: str = "data/shards/NQ_M5", adapted_dir: str = "data/adapted"):
        """Initialize the Enhanced Temporal Query Engine with modular architecture"""
        self.shard_dir = shard_dir
        self.adapted_dir = adapted_dir
        
        print("🔍 Initializing Enhanced Temporal Query Engine with Price Relativity...")
        
        # Initialize core modules
        self.session_manager = SessionDataManager(shard_dir, adapted_dir)
        self.price_engine = PriceRelativityEngine()
        self.query_core = TemporalQueryCore(self.session_manager, self.price_engine)
        self.visualization = VisualizationManager()
        
        # Load all sessions
        self.session_manager.load_all_sessions()
        
        # Expose session data for backward compatibility
        self.sessions = self.session_manager.sessions
        self.graphs = self.session_manager.graphs
        self.metadata = self.session_manager.metadata
        self.session_stats = self.session_manager.session_stats
        
    def ask(self, question: str) -> dict[str, Any]:
        """
        Ask a temporal question and get probabilistic answers with price relativity
        
        This is the main query interface that maintains full backward compatibility.
        """
        return self.query_core.ask(question)
        
    def get_enhanced_session_info(self, session_id: str) -> dict[str, Any]:
        """Get complete session information with price relativity analysis"""
        return self.session_manager.get_enhanced_session_info(session_id)
        
    def list_sessions(self) -> list[str]:
        """List all available sessions with type information"""
        return self.session_manager.list_sessions()
        
    # ================================================================================
    # ARCHAEOLOGICAL ZONE ANALYSIS METHODS
    # ================================================================================
    
    def _analyze_archaeological_zones(self, question: str) -> dict[str, Any]:
        """Analyze archaeological zone patterns and Theory B events"""
        return self.price_engine.analyze_archaeological_zones(
            question, self.sessions, self.session_stats
        )
        
    def _analyze_theory_b_patterns(self, question: str) -> dict[str, Any]:
        """Analyze Theory B temporal non-locality patterns"""
        return self.price_engine.analyze_theory_b_patterns(
            question, self.sessions, self.session_stats
        )
        
    # ================================================================================
    # EXPERIMENT SET E: Post-RD@40% Sequence Analysis
    # ================================================================================
    
    def _analyze_post_rd40_sequences(self, question: str) -> dict[str, Any]:
        """Analyze sequence patterns after RD@40% events"""
        return self.price_engine.analyze_post_rd40_sequences(
            question, self.sessions, self.session_stats
        )
        
    def _detect_rd40_events(self) -> list[dict[str, Any]]:
        """Detect FPFVG redelivery events at 40% archaeological zones"""
        return self.price_engine._detect_rd40_events(self.sessions, self.session_stats)
        
    def _classify_sequence_path(self, session_id: str, event_index: int) -> dict[str, Any]:
        """Classify the sequence path after RD@40% event: CONT/MR/ACCEL"""
        if session_id not in self.sessions or session_id not in self.session_stats:
            return {"error": f"Session {session_id} not found"}
            
        nodes_df = self.sessions[session_id]
        stats = self.session_stats[session_id]
        
        return self.price_engine._classify_sequence_path(nodes_df, stats, event_index)
        
    # ================================================================================
    # TEMPORAL SEQUENCE ANALYSIS METHODS
    # ================================================================================
    
    def _analyze_temporal_sequence(self, question: str) -> dict[str, Any]:
        """Analyze what happens after specific events with price relativity"""
        return self.query_core._analyze_temporal_sequence(question)
        
    def _analyze_opening_patterns(self, question: str) -> dict[str, Any]:
        """Analyze session opening patterns with price relativity"""
        return self.query_core._analyze_opening_patterns(question)
        
    def _get_enhanced_event_context(self, event, session_type: str, session_stats: dict[str, float],
                                   nodes: pd.DataFrame, event_idx: int) -> dict[str, Any]:
        """Get complete event context with temporal and price relativity"""
        return self.query_core._get_enhanced_event_context(
            event, session_type, session_stats, nodes, event_idx
        )
        
    def _check_pattern_match(self, event_context: dict[str, Any], pattern_key: str,
                           nodes: pd.DataFrame, event_idx: int, time_window: int) -> dict[str, Any]:
        """Check if event matches specified pattern with enhanced criteria"""
        return self.query_core._check_pattern_match(
            event_context, pattern_key, nodes, event_idx, time_window
        )
        
    # ================================================================================
    # PLACEHOLDER METHODS FOR ADDITIONAL ANALYSIS
    # ================================================================================
    
    def _analyze_relative_positioning(self, question: str) -> dict[str, Any]:
        """Analyze relative positioning patterns"""
        results = {
            "query_type": "relative_positioning",
            "total_sessions": len(self.sessions),
            "positioning_analysis": {},
            "insights": []
        }

        # Analyze positioning patterns across sessions
        for session_id, nodes_df in self.sessions.items():
            if session_id not in self.session_stats:
                continue

            stats = self.session_stats[session_id]
            session_type = self._determine_session_type(session_id)

            # Calculate relative positioning metrics
            if 'price' in nodes_df.columns and len(nodes_df) > 0:
                session_range = stats.get('range', 0)
                session_low = stats.get('low', 0)

                if session_range > 0:
                    # Calculate position distribution
                    relative_positions = ((nodes_df['price'] - session_low) / session_range * 100)

                    positioning_metrics = {
                        "session_type": session_type,
                        "avg_position": relative_positions.mean(),
                        "position_std": relative_positions.std(),
                        "upper_quartile_events": (relative_positions > 75).sum(),
                        "lower_quartile_events": (relative_positions < 25).sum(),
                        "mid_range_events": ((relative_positions >= 25) & (relative_positions <= 75)).sum()
                    }

                    results["positioning_analysis"][session_id] = positioning_metrics

        # Generate insights
        if results["positioning_analysis"]:
            avg_positions = [metrics["avg_position"] for metrics in results["positioning_analysis"].values()]
            results["insights"] = [
                f"Analyzed positioning patterns across {len(results['positioning_analysis'])} sessions",
                f"Average relative position: {np.mean(avg_positions):.1f}%",
                f"Position variability: {np.std(avg_positions):.1f}%"
            ]
        else:
            results["insights"] = ["No positioning data available for analysis"]

        return results
        
    def _search_patterns(self, question: str) -> dict[str, Any]:
        """Enhanced pattern search with archaeological zones"""
        results = {
            "query_type": "pattern_search",
            "total_sessions": len(self.sessions),
            "pattern_matches": [],
            "search_criteria": self._extract_search_criteria(question),
            "insights": []
        }

        search_criteria = results["search_criteria"]

        # Search for patterns across sessions
        for session_id, nodes_df in self.sessions.items():
            if session_id not in self.session_stats:
                continue

            stats = self.session_stats[session_id]
            session_type = self._determine_session_type(session_id)

            # Apply search criteria
            matches = self._find_pattern_matches(nodes_df, stats, search_criteria)

            if matches:
                results["pattern_matches"].extend([
                    {
                        "session_id": session_id,
                        "session_type": session_type,
                        "match": match
                    } for match in matches
                ])

        # Generate insights
        total_matches = len(results["pattern_matches"])
        if total_matches > 0:
            session_count = len({match["session_id"] for match in results["pattern_matches"]})
            results["insights"] = [
                f"Found {total_matches} pattern matches across {session_count} sessions",
                f"Search criteria: {search_criteria}",
                f"Average matches per session: {total_matches / session_count:.1f}"
            ]
        else:
            results["insights"] = [
                f"No patterns found matching criteria: {search_criteria}",
                "Consider broadening search parameters"
            ]

        return results

    def _extract_search_criteria(self, question: str) -> dict[str, Any]:
        """Extract search criteria from question"""
        criteria = {
            "pattern_type": "general",
            "zone_filter": None,
            "time_filter": None,
            "volume_filter": None
        }

        question_lower = question.lower()

        # Extract pattern type
        if "liquidity" in question_lower:
            criteria["pattern_type"] = "liquidity"
        elif "gap" in question_lower or "fvg" in question_lower:
            criteria["pattern_type"] = "gap"
        elif "sweep" in question_lower:
            criteria["pattern_type"] = "sweep"
        elif "precision" in question_lower:
            criteria["pattern_type"] = "precision"

        # Extract zone filter
        import re
        zone_match = re.search(r'(\d+)%', question)
        if zone_match:
            criteria["zone_filter"] = int(zone_match.group(1))

        return criteria

    def _find_pattern_matches(self, nodes_df: pd.DataFrame, stats: dict[str, float],
                            criteria: dict[str, Any]) -> list[dict[str, Any]]:
        """Find pattern matches in session data"""
        matches = []

        if 'price' not in nodes_df.columns or len(nodes_df) == 0:
            return matches

        session_range = stats.get('range', 0)
        session_low = stats.get('low', 0)

        if session_range == 0:
            return matches

        # Calculate relative positions
        relative_positions = ((nodes_df['price'] - session_low) / session_range * 100)

        # Apply zone filter if specified
        if criteria.get("zone_filter"):
            zone_pct = criteria["zone_filter"]
            tolerance = 5  # 5% tolerance
            zone_mask = (relative_positions >= zone_pct - tolerance) & (relative_positions <= zone_pct + tolerance)
            candidate_events = nodes_df[zone_mask]
        else:
            candidate_events = nodes_df

        # Find pattern-specific matches
        pattern_type = criteria.get("pattern_type", "general")

        for idx, event in candidate_events.iterrows():
            match_score = self._calculate_pattern_score(event, pattern_type, nodes_df, idx)

            if match_score > 0.5:  # Threshold for pattern match
                matches.append({
                    "event_index": idx,
                    "price": event.get('price', 0),
                    "relative_position": relative_positions.iloc[idx] if idx < len(relative_positions) else 0,
                    "pattern_score": match_score,
                    "pattern_type": pattern_type
                })

        return matches

    def _calculate_pattern_score(self, event: pd.Series, pattern_type: str,
                               nodes_df: pd.DataFrame, event_idx: int) -> float:
        """Calculate pattern match score for an event"""
        score = 0.0

        # Base score for having price data
        if 'price' in event.index and pd.notna(event['price']):
            score += 0.3

        # Pattern-specific scoring
        if pattern_type == "liquidity":
            if 'volume' in event.index and pd.notna(event['volume']):
                # Higher volume indicates potential liquidity event
                volume_percentile = (event['volume'] > nodes_df['volume']).mean() if 'volume' in nodes_df.columns else 0.5
                score += volume_percentile * 0.4

        elif pattern_type == "precision":
            # Look for events with specific characteristics
            if 'energy_density' in event.index and pd.notna(event['energy_density']):
                score += event['energy_density'] * 0.4
            else:
                # Use price precision as proxy
                price_precision = 1.0 - (abs(event['price'] % 1.0) if event['price'] % 1.0 != 0 else 0.1)
                score += price_precision * 0.2

        elif pattern_type == "gap":
            # Look for price gaps (simplified)
            if event_idx > 0 and event_idx < len(nodes_df) - 1:
                prev_price = nodes_df.iloc[event_idx - 1].get('price', event['price'])
                next_price = nodes_df.iloc[event_idx + 1].get('price', event['price'])

                # Check for gap pattern
                if abs(event['price'] - prev_price) > abs(next_price - event['price']):
                    score += 0.3

        # Default scoring for general patterns
        if pattern_type == "general":
            score += 0.5  # Base score for any event

        return min(1.0, score)
        
    def _analyze_liquidity_sweeps(self, question: str) -> dict[str, Any]:
        """Analyze liquidity sweep patterns"""
        results = {
            "query_type": "liquidity_sweeps",
            "total_sessions": len(self.sessions),
            "sweep_events": [],
            "sweep_statistics": {},
            "insights": []
        }

        # Analyze liquidity sweeps across sessions
        for session_id, nodes_df in self.sessions.items():
            if session_id not in self.session_stats:
                continue

            stats = self.session_stats[session_id]
            session_type = self._determine_session_type(session_id)

            # Detect liquidity sweep events
            sweep_events = self._detect_liquidity_sweeps(nodes_df, stats)

            for sweep in sweep_events:
                sweep["session_id"] = session_id
                sweep["session_type"] = session_type
                results["sweep_events"].append(sweep)

        # Calculate sweep statistics
        if results["sweep_events"]:
            results["sweep_statistics"] = self._calculate_sweep_statistics(results["sweep_events"])

            # Generate insights
            total_sweeps = len(results["sweep_events"])
            session_count = len({sweep["session_id"] for sweep in results["sweep_events"]})

            results["insights"] = [
                f"Detected {total_sweeps} liquidity sweep events across {session_count} sessions",
                f"Average sweeps per session: {total_sweeps / session_count:.1f}",
                f"Most common sweep type: {results['sweep_statistics'].get('most_common_type', 'N/A')}"
            ]
        else:
            results["insights"] = ["No liquidity sweep events detected in the analyzed sessions"]

        return results

    def _detect_liquidity_sweeps(self, nodes_df: pd.DataFrame, stats: dict[str, float]) -> list[dict[str, Any]]:
        """Detect liquidity sweep events in session data"""
        sweeps = []

        if 'price' not in nodes_df.columns or len(nodes_df) < 3:
            return sweeps

        session_high = stats.get('high', 0)
        session_low = stats.get('low', 0)
        session_range = session_high - session_low

        if session_range == 0:
            return sweeps

        # Look for sweep patterns
        for i in range(1, len(nodes_df) - 1):
            current_price = nodes_df.iloc[i]['price']
            prev_price = nodes_df.iloc[i-1]['price']
            next_price = nodes_df.iloc[i+1]['price']

            # Calculate relative position
            relative_pos = ((current_price - session_low) / session_range) * 100

            # Detect high liquidity sweep (price spikes above 90% then retraces)
            if (relative_pos > 90 and
                current_price > prev_price and
                next_price < current_price):

                sweep_strength = self._calculate_sweep_strength(nodes_df, i, "high")

                sweeps.append({
                    "event_index": i,
                    "sweep_type": "high_liquidity_sweep",
                    "price": current_price,
                    "relative_position": relative_pos,
                    "sweep_strength": sweep_strength,
                    "retracement": ((current_price - next_price) / current_price) * 100
                })

            # Detect low liquidity sweep (price drops below 10% then recovers)
            elif (relative_pos < 10 and
                  current_price < prev_price and
                  next_price > current_price):

                sweep_strength = self._calculate_sweep_strength(nodes_df, i, "low")

                sweeps.append({
                    "event_index": i,
                    "sweep_type": "low_liquidity_sweep",
                    "price": current_price,
                    "relative_position": relative_pos,
                    "sweep_strength": sweep_strength,
                    "recovery": ((next_price - current_price) / current_price) * 100
                })

        return sweeps

    def _calculate_sweep_strength(self, nodes_df: pd.DataFrame, event_idx: int, sweep_type: str) -> float:
        """Calculate the strength of a liquidity sweep"""
        strength = 0.0

        # Volume-based strength (if available)
        if 'volume' in nodes_df.columns:
            event_volume = nodes_df.iloc[event_idx].get('volume', 0)
            avg_volume = nodes_df['volume'].mean()
            if avg_volume > 0:
                volume_ratio = event_volume / avg_volume
                strength += min(volume_ratio / 3.0, 0.5)  # Cap at 0.5

        # Price movement strength
        if event_idx > 0 and event_idx < len(nodes_df) - 1:
            current_price = nodes_df.iloc[event_idx]['price']
            prev_price = nodes_df.iloc[event_idx - 1]['price']
            next_price = nodes_df.iloc[event_idx + 1]['price']

            if sweep_type == "high":
                price_move = (current_price - prev_price) / prev_price if prev_price > 0 else 0
                retracement = (current_price - next_price) / current_price if current_price > 0 else 0
            else:  # low sweep
                price_move = (prev_price - current_price) / prev_price if prev_price > 0 else 0
                retracement = (next_price - current_price) / current_price if current_price > 0 else 0

            strength += min(abs(price_move) * 10, 0.3)  # Price movement component
            strength += min(abs(retracement) * 10, 0.2)  # Retracement component

        return min(strength, 1.0)

    def _calculate_sweep_statistics(self, sweep_events: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate statistics for liquidity sweep events"""
        if not sweep_events:
            return {}

        # Count sweep types
        sweep_types = {}
        strengths = []
        positions = []

        for sweep in sweep_events:
            sweep_type = sweep.get("sweep_type", "unknown")
            sweep_types[sweep_type] = sweep_types.get(sweep_type, 0) + 1
            strengths.append(sweep.get("sweep_strength", 0))
            positions.append(sweep.get("relative_position", 0))

        most_common_type = max(sweep_types, key=sweep_types.get) if sweep_types else "none"

        return {
            "sweep_type_distribution": sweep_types,
            "most_common_type": most_common_type,
            "average_strength": np.mean(strengths) if strengths else 0,
            "average_position": np.mean(positions) if positions else 0,
            "strength_std": np.std(strengths) if strengths else 0
        }
        
    def _analyze_htf_taps(self, question: str) -> dict[str, Any]:
        """Analyze higher timeframe tap patterns"""
        return {
            "query_type": "htf_taps",
            "message": "HTF tap analysis - implementation delegated to visualization module",
            "total_sessions": len(self.sessions),
            "insights": ["HTF tap analysis available through visualization module"]
        }
        
    def _analyze_fvg_follow_through(self, question: str) -> dict[str, Any]:
        """Analyze fair value gap follow-through patterns"""
        return {
            "query_type": "fvg_follow_through",
            "message": "FVG follow-through analysis - implementation delegated to visualization module",
            "total_sessions": len(self.sessions),
            "insights": ["FVG follow-through analysis available through visualization module"]
        }
        
    def _analyze_event_chains(self, question: str) -> dict[str, Any]:
        """Analyze event chain patterns"""
        return {
            "query_type": "event_chains",
            "message": "Event chain analysis - implementation delegated to visualization module",
            "total_sessions": len(self.sessions),
            "insights": ["Event chain analysis available through visualization module"]
        }
        
    def _analyze_minute_hotspots(self, question: str) -> dict[str, Any]:
        """Analyze minute-level hotspot patterns"""
        return {
            "query_type": "minute_hotspots",
            "message": "Minute hotspot analysis - implementation delegated to visualization module",
            "total_sessions": len(self.sessions),
            "insights": ["Minute hotspot analysis available through visualization module"]
        }
        
    def _analyze_rd40_day_news_matrix(self, question: str) -> dict[str, Any]:
        """Analyze RD@40% day news impact matrix"""
        return {
            "query_type": "rd40_news_matrix",
            "message": "RD@40% news matrix analysis - implementation delegated to visualization module",
            "total_sessions": len(self.sessions),
            "insights": ["RD@40% news matrix analysis available through visualization module"]
        }
        
    def _analyze_f8_interactions(self, question: str) -> dict[str, Any]:
        """Analyze F8 feature interactions"""
        return {
            "query_type": "f8_interactions",
            "message": "F8 interaction analysis - implementation delegated to visualization module",
            "total_sessions": len(self.sessions),
            "insights": ["F8 interaction analysis available through visualization module"]
        }
        
    def _analyze_ml_predictions(self, question: str) -> dict[str, Any]:
        """Analyze machine learning predictions"""
        return {
            "query_type": "ml_predictions",
            "message": "ML prediction analysis - implementation delegated to specialized ML modules",
            "total_sessions": len(self.sessions),
            "insights": ["ML prediction analysis available through specialized modules"]
        }
        
    def _general_temporal_analysis(self, question: str) -> dict[str, Any]:
        """General temporal analysis for unrecognized queries"""
        return {
            "query_type": "general_temporal",
            "message": f"General temporal analysis for: {question}",
            "total_sessions": len(self.sessions),
            "available_methods": [
                "Archaeological zone analysis",
                "Theory B pattern detection", 
                "RD@40% sequence analysis",
                "Temporal sequence analysis",
                "Opening pattern analysis"
            ],
            "insights": [
                "Use specific keywords like 'zone', 'theory b', 'rd40', 'after', 'when' for targeted analysis",
                f"Analyzed {len(self.sessions)} sessions with modular architecture"
            ]
        }
        
    # ================================================================================
    # VISUALIZATION METHODS
    # ================================================================================
    
    def display_results(self, results: dict[str, Any]) -> None:
        """Display query results using the visualization manager"""
        self.visualization.display_query_results(results)
        
    def plot_results(self, results: dict[str, Any], save_path: str | None = None) -> None:
        """Plot query results using the visualization manager"""
        query_type = results.get("query_type", "")
        
        if "temporal_sequence" in query_type:
            self.visualization.plot_temporal_sequence(results, save_path)
        elif "archaeological" in query_type:
            self.visualization.plot_archaeological_zones(results, save_path)
        else:
            print(f"Plotting not yet implemented for query type: {query_type}")
            
    # ================================================================================
    # UTILITY METHODS
    # ================================================================================
    
    def validate_session(self, session_id: str) -> dict[str, Any]:
        """Validate session data quality and completeness"""
        return self.session_manager.validate_session_data(session_id)
        
    def get_session_statistics(self) -> dict[str, Any]:
        """Get overall statistics about loaded sessions"""
        total_sessions = len(self.sessions)
        total_events = sum(len(df) for df in self.sessions.values())
        
        session_types = {}
        for session_id in self.sessions:
            session_type = self._determine_session_type(session_id)
            session_types[session_type] = session_types.get(session_type, 0) + 1
            
        return {
            "total_sessions": total_sessions,
            "total_events": total_events,
            "session_types": session_types,
            "average_events_per_session": total_events / total_sessions if total_sessions > 0 else 0,
            "data_sources": {
                "shard_dir": self.shard_dir,
                "adapted_dir": self.adapted_dir
            }
        }
        
    def _determine_session_type(self, session_id: str) -> str:
        """Determine session type from session ID"""
        return self.session_manager._determine_session_type(session_id)
