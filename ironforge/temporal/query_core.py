#!/usr/bin/env python3
"""
IRONFORGE Temporal Query Core
Core temporal querying logic and pattern matching for archaeological discovery
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re

from .session_manager import SessionDataManager
from .price_relativity import PriceRelativityEngine
from ml_path_predictor import MLPathPredictor
from liquidity_htf_analyzer import LiquidityHTFAnalyzer


class TemporalQueryCore:
    """Core temporal analysis and pattern matching engine"""
    
    def __init__(self, session_manager: SessionDataManager, price_engine: PriceRelativityEngine):
        self.session_manager = session_manager
        self.price_engine = price_engine
        self.ml_predictor = MLPathPredictor()
        self.liquidity_analyzer = LiquidityHTFAnalyzer()
        
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a temporal question and get probabilistic answers with price relativity"""
        print(f"\n🤔 Query: {question}")
        
        # Enhanced parsing for price relativity queries
        if "after" in question.lower() and "what happens" in question.lower():
            return self._analyze_temporal_sequence(question)
        elif "when" in question.lower() and ("starts with" in question.lower() or "begins with" in question.lower()):
            return self._analyze_opening_patterns(question)
        elif "zone" in question.lower() or "archaeological" in question.lower():
            return self.price_engine.analyze_archaeological_zones(
                question, self.session_manager.sessions, self.session_manager.session_stats
            )
        elif "theory b" in question.lower() or "temporal non-locality" in question.lower():
            return self.price_engine.analyze_theory_b_patterns(
                question, self.session_manager.sessions, self.session_manager.session_stats
            )
        elif "rd@40" in question.lower() or "rd40" in question.lower():
            return self.price_engine.analyze_post_rd40_sequences(
                question, self.session_manager.sessions, self.session_manager.session_stats
            )
        elif "relative positioning" in question.lower() or "positioning" in question.lower():
            return self._analyze_relative_positioning(question)
        elif "pattern" in question.lower() or "search" in question.lower():
            return self._search_patterns(question)
        elif "liquidity" in question.lower() and "sweep" in question.lower():
            return self._analyze_liquidity_sweeps(question)
        elif "htf" in question.lower() and ("tap" in question.lower() or "touch" in question.lower()):
            return self._analyze_htf_taps(question)
        elif "fvg" in question.lower() and "follow" in question.lower():
            return self._analyze_fvg_follow_through(question)
        elif "event chain" in question.lower() or "sequence" in question.lower():
            return self._analyze_event_chains(question)
        elif "minute" in question.lower() and "hotspot" in question.lower():
            return self._analyze_minute_hotspots(question)
        elif "news" in question.lower() and ("impact" in question.lower() or "matrix" in question.lower()):
            return self._analyze_rd40_day_news_matrix(question)
        elif "f8" in question.lower() and "interaction" in question.lower():
            return self._analyze_f8_interactions(question)
        elif "ml" in question.lower() or "prediction" in question.lower():
            return self._analyze_ml_predictions(question)
        else:
            return self._general_temporal_analysis(question)
            
    def _analyze_temporal_sequence(self, question: str) -> Dict[str, Any]:
        """Analyze what happens after specific events with price relativity"""
        results = {
            "query_type": "temporal_sequence_with_relativity",
            "total_sessions": len(self.session_manager.sessions),
            "matches": [],
            "probabilities": {},
            "insights": [],
            "price_relativity_analysis": []
        }
        
        # Extract time window from question
        time_window = self._extract_time_window(question)
        pattern_key = self._extract_pattern_key(question)
        
        for session_id, nodes_df in self.session_manager.sessions.items():
            if session_id not in self.session_manager.session_stats:
                continue
                
            session_stats = self.session_manager.session_stats[session_id]
            session_type = self._determine_session_type(session_id)
            
            # Find matching events
            for idx, event in nodes_df.iterrows():
                # Get enhanced event context with price relativity
                event_context = self._get_enhanced_event_context(
                    event, session_type, session_stats, nodes_df, idx
                )
                
                # Check if this event matches the pattern
                pattern_match = self._check_pattern_match(
                    event_context, pattern_key, nodes_df, idx, time_window
                )
                
                if pattern_match["matches"]:
                    results["matches"].append({
                        "session_id": session_id,
                        "event_index": idx,
                        "event_context": event_context,
                        "pattern_match": pattern_match,
                        "subsequent_events": self._analyze_subsequent_events(
                            nodes_df, idx, time_window, session_stats
                        )
                    })
                    
        # Calculate probabilities and generate insights
        results["probabilities"] = self._calculate_sequence_probabilities(results["matches"])
        results["insights"] = self._generate_sequence_insights(results)
        
        return results
        
    def _analyze_opening_patterns(self, question: str) -> Dict[str, Any]:
        """Analyze session opening patterns with price relativity"""
        results = {
            "query_type": "opening_patterns_with_relativity",
            "total_sessions": len(self.session_manager.sessions),
            "opening_analysis": {},
            "pattern_distribution": {},
            "insights": []
        }
        
        # Extract opening criteria from question
        opening_criteria = self._extract_opening_criteria(question)
        
        for session_id, nodes_df in self.session_manager.sessions.items():
            if session_id not in self.session_manager.session_stats:
                continue
                
            session_stats = self.session_manager.session_stats[session_id]
            session_type = self._determine_session_type(session_id)
            
            # Analyze opening pattern
            opening_analysis = self._analyze_session_opening(
                nodes_df, session_stats, session_type, opening_criteria
            )
            
            results["opening_analysis"][session_id] = opening_analysis
            
        # Calculate pattern distribution
        results["pattern_distribution"] = self._calculate_opening_distribution(results["opening_analysis"])
        results["insights"] = self._generate_opening_insights(results)
        
        return results
        
    def _get_enhanced_event_context(self, event: pd.Series, session_type: str, 
                                  session_stats: Dict[str, float], nodes: pd.DataFrame, 
                                  event_idx: int) -> Dict[str, Any]:
        """Get complete event context with temporal and price relativity"""
        context = {
            "session_type": session_type,
            "event_index": event_idx,
            "price": event.get('price', 0),
            "timestamp": event.get('timestamp', ''),
            "session_stats": session_stats
        }
        
        # Add price relativity calculations
        if session_stats.get('range', 0) > 0:
            session_range = session_stats['range']
            session_low = session_stats['low']
            
            # Calculate archaeological zone percentage
            zone_percentage = ((event.get('price', 0) - session_low) / session_range) * 100
            context["archaeological_zone_pct"] = zone_percentage
            
            # Determine which zone this event is in
            if zone_percentage <= 25:
                context["archaeological_zone"] = "Lower_25"
            elif zone_percentage <= 50:
                context["archaeological_zone"] = "Mid_Lower_50"
            elif zone_percentage <= 75:
                context["archaeological_zone"] = "Mid_Upper_75"
            else:
                context["archaeological_zone"] = "Upper_100"
                
        # Add temporal context
        context["temporal_position"] = event_idx / len(nodes) if len(nodes) > 0 else 0
        
        # Add liquidity and energy features if available
        for feature in ['liquidity_score', 'energy_density', 'volume', 'price_momentum']:
            if feature in event.index:
                context[feature] = event[feature]
                
        return context
        
    def _check_pattern_match(self, event_context: Dict[str, Any], pattern_key: str,
                           nodes: pd.DataFrame, event_idx: int, time_window: int) -> Dict[str, Any]:
        """Check if event matches specified pattern with enhanced criteria"""
        match_result = {
            "matches": False,
            "confidence": 0.0,
            "criteria_met": [],
            "criteria_failed": []
        }
        
        # Parse pattern criteria
        criteria = self._parse_pattern_criteria(pattern_key)
        
        # Check each criterion
        for criterion in criteria:
            if self._evaluate_criterion(event_context, criterion, nodes, event_idx):
                match_result["criteria_met"].append(criterion)
            else:
                match_result["criteria_failed"].append(criterion)
                
        # Calculate match confidence
        total_criteria = len(criteria)
        met_criteria = len(match_result["criteria_met"])
        
        if total_criteria > 0:
            match_result["confidence"] = met_criteria / total_criteria
            match_result["matches"] = match_result["confidence"] >= 0.7  # 70% threshold
            
        return match_result
        
    def _analyze_subsequent_events(self, nodes: pd.DataFrame, start_idx: int, 
                                 time_window: int, session_stats: Dict[str, float]) -> Dict[str, Any]:
        """Analyze events that occur after the matched pattern"""
        subsequent_analysis = {
            "event_count": 0,
            "price_movement": {},
            "zone_transitions": [],
            "significant_events": []
        }
        
        # Look ahead within time window
        end_idx = min(start_idx + time_window, len(nodes))
        subsequent_events = nodes.iloc[start_idx+1:end_idx]
        
        if len(subsequent_events) == 0:
            return subsequent_analysis
            
        subsequent_analysis["event_count"] = len(subsequent_events)
        
        # Analyze price movement
        start_price = nodes.iloc[start_idx].get('price', 0)
        end_price = subsequent_events.iloc[-1].get('price', 0)
        
        subsequent_analysis["price_movement"] = {
            "start_price": start_price,
            "end_price": end_price,
            "price_change": end_price - start_price,
            "price_change_pct": ((end_price - start_price) / start_price * 100) if start_price > 0 else 0,
            "max_price": subsequent_events['price'].max() if 'price' in subsequent_events.columns else 0,
            "min_price": subsequent_events['price'].min() if 'price' in subsequent_events.columns else 0
        }
        
        # Analyze zone transitions
        if session_stats.get('range', 0) > 0:
            session_range = session_stats['range']
            session_low = session_stats['low']
            
            for idx, event in subsequent_events.iterrows():
                zone_pct = ((event.get('price', 0) - session_low) / session_range) * 100
                subsequent_analysis["zone_transitions"].append({
                    "event_index": idx,
                    "zone_percentage": zone_pct,
                    "price": event.get('price', 0)
                })
                
        return subsequent_analysis

    def _extract_time_window(self, question: str) -> int:
        """Extract time window from question (default to 20 events)"""
        time_match = re.search(r'(\d+)\s*(minute|event|step)', question.lower())
        if time_match:
            return int(time_match.group(1))
        return 20  # Default window

    def _extract_pattern_key(self, question: str) -> str:
        """Extract pattern key from question"""
        # Look for specific patterns mentioned in the question
        patterns = {
            "liquidity sweep": "liquidity_sweep",
            "fvg": "fair_value_gap",
            "htf tap": "htf_tap",
            "redelivery": "redelivery",
            "precision": "precision_event"
        }

        question_lower = question.lower()
        for pattern_name, pattern_key in patterns.items():
            if pattern_name in question_lower:
                return pattern_key

        return "general_pattern"

    def _extract_opening_criteria(self, question: str) -> Dict[str, Any]:
        """Extract opening criteria from question"""
        criteria = {
            "pattern_type": "any",
            "price_threshold": None,
            "time_window": 30  # First 30 events
        }

        # Extract specific opening patterns
        if "gap" in question.lower():
            criteria["pattern_type"] = "gap"
        elif "sweep" in question.lower():
            criteria["pattern_type"] = "sweep"
        elif "precision" in question.lower():
            criteria["pattern_type"] = "precision"

        return criteria

    def _determine_session_type(self, session_id: str) -> str:
        """Determine session type from session ID"""
        session_id_lower = session_id.lower()

        if 'ny_am' in session_id_lower or 'nyam' in session_id_lower:
            return 'NY_AM'
        elif 'ny_pm' in session_id_lower or 'nypm' in session_id_lower:
            return 'NY_PM'
        elif 'london' in session_id_lower:
            return 'LONDON'
        elif 'asia' in session_id_lower:
            return 'ASIA'
        else:
            return 'UNKNOWN'

    def _parse_pattern_criteria(self, pattern_key: str) -> List[Dict[str, Any]]:
        """Parse pattern criteria based on pattern key"""
        criteria_map = {
            "liquidity_sweep": [
                {"type": "liquidity_score", "threshold": 0.7, "operator": ">="},
                {"type": "volume", "threshold": "high", "operator": ">="}
            ],
            "fair_value_gap": [
                {"type": "price_gap", "threshold": 0.02, "operator": ">="},
                {"type": "volume", "threshold": "low", "operator": "<="}
            ],
            "htf_tap": [
                {"type": "archaeological_zone_pct", "threshold": [75, 100], "operator": "between"},
                {"type": "energy_density", "threshold": 0.6, "operator": ">="}
            ],
            "redelivery": [
                {"type": "archaeological_zone_pct", "threshold": [35, 45], "operator": "between"},
                {"type": "liquidity_score", "threshold": 0.6, "operator": ">="}
            ],
            "precision_event": [
                {"type": "energy_density", "threshold": 0.8, "operator": ">="},
                {"type": "archaeological_zone_pct", "threshold": [38, 42], "operator": "between"}
            ]
        }

        return criteria_map.get(pattern_key, [])

    def _evaluate_criterion(self, event_context: Dict[str, Any], criterion: Dict[str, Any],
                          nodes: pd.DataFrame, event_idx: int) -> bool:
        """Evaluate a single pattern criterion"""
        criterion_type = criterion["type"]
        threshold = criterion["threshold"]
        operator = criterion["operator"]

        # Get the value to evaluate
        if criterion_type in event_context:
            value = event_context[criterion_type]
        else:
            return False

        # Apply the operator
        if operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "between":
            return threshold[0] <= value <= threshold[1]
        elif operator == ">" and threshold == "high":
            # For relative thresholds like "high volume"
            if criterion_type == "volume" and 'volume' in nodes.columns:
                return value > nodes['volume'].quantile(0.8)
        elif operator == "<=" and threshold == "low":
            if criterion_type == "volume" and 'volume' in nodes.columns:
                return value <= nodes['volume'].quantile(0.2)

        return False

    def _calculate_sequence_probabilities(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate probabilities from sequence matches"""
        if not matches:
            return {}

        probabilities = {}

        # Analyze subsequent events
        outcome_counts = {}
        for match in matches:
            subsequent_events = match.get("subsequent_events", {})
            price_movement = subsequent_events.get("price_movement", {})

            # Classify outcome
            price_change_pct = price_movement.get("price_change_pct", 0)
            if price_change_pct > 2:
                outcome = "significant_up"
            elif price_change_pct < -2:
                outcome = "significant_down"
            else:
                outcome = "sideways"

            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        # Calculate probabilities
        total_matches = len(matches)
        for outcome, count in outcome_counts.items():
            probabilities[outcome] = {
                "probability": count / total_matches,
                "count": count,
                "total": total_matches
            }

        return probabilities

    def _generate_sequence_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from sequence analysis"""
        insights = []

        matches = results.get("matches", [])
        probabilities = results.get("probabilities", {})

        if matches:
            insights.append(f"Found {len(matches)} matching temporal sequences")

            # Analyze confidence distribution
            confidences = []
            for match in matches:
                pattern_match = match.get("pattern_match", {})
                confidence = pattern_match.get("confidence", 0)
                confidences.append(confidence)

            if confidences:
                avg_confidence = np.mean(confidences)
                insights.append(f"Average pattern match confidence: {avg_confidence:.1%}")

        if probabilities:
            # Find most likely outcome
            best_outcome = None
            best_prob = 0
            for outcome, prob_data in probabilities.items():
                prob = prob_data.get("probability", 0)
                if prob > best_prob:
                    best_prob = prob
                    best_outcome = outcome

            if best_outcome:
                insights.append(f"Most likely outcome: {best_outcome} ({best_prob:.1%} probability)")

        return insights

    def _analyze_session_opening(self, nodes_df: pd.DataFrame, session_stats: Dict[str, float],
                               session_type: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze session opening pattern"""
        opening_analysis = {
            "session_type": session_type,
            "pattern_type": "unknown",
            "opening_events": 0,
            "price_movement": {},
            "zone_analysis": {}
        }

        time_window = criteria.get("time_window", 30)
        opening_events = nodes_df.head(time_window)

        if len(opening_events) == 0:
            return opening_analysis

        opening_analysis["opening_events"] = len(opening_events)

        # Analyze price movement in opening
        if 'price' in opening_events.columns:
            first_price = opening_events.iloc[0]['price']
            last_price = opening_events.iloc[-1]['price']

            opening_analysis["price_movement"] = {
                "first_price": first_price,
                "last_price": last_price,
                "price_change": last_price - first_price,
                "price_change_pct": ((last_price - first_price) / first_price * 100) if first_price > 0 else 0,
                "max_price": opening_events['price'].max(),
                "min_price": opening_events['price'].min()
            }

        # Classify opening pattern
        pattern_type = criteria.get("pattern_type", "any")
        if pattern_type == "gap":
            # Check for gap pattern
            if 'price' in opening_events.columns and len(opening_events) > 1:
                price_diff = abs(opening_events.iloc[1]['price'] - opening_events.iloc[0]['price'])
                session_range = session_stats.get('range', 0)
                if session_range > 0 and price_diff > session_range * 0.01:
                    opening_analysis["pattern_type"] = "gap"

        return opening_analysis

    def _calculate_opening_distribution(self, opening_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate opening pattern distribution"""
        distribution = {}

        pattern_counts = {}
        session_type_counts = {}

        for session_id, analysis in opening_analysis.items():
            pattern_type = analysis.get("pattern_type", "unknown")
            session_type = analysis.get("session_type", "UNKNOWN")

            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            session_type_counts[session_type] = session_type_counts.get(session_type, 0) + 1

        total_sessions = len(opening_analysis)

        # Calculate percentages
        for pattern, count in pattern_counts.items():
            distribution[pattern] = {
                "count": count,
                "percentage": (count / total_sessions * 100) if total_sessions > 0 else 0
            }

        distribution["session_types"] = session_type_counts

        return distribution

    def _generate_opening_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from opening pattern analysis"""
        insights = []

        opening_analysis = results.get("opening_analysis", {})
        pattern_distribution = results.get("pattern_distribution", {})

        if opening_analysis:
            insights.append(f"Analyzed opening patterns for {len(opening_analysis)} sessions")

        if pattern_distribution:
            # Find most common pattern
            pattern_counts = {}
            for pattern, data in pattern_distribution.items():
                if isinstance(data, dict) and "count" in data:
                    pattern_counts[pattern] = data["count"]

            if pattern_counts:
                most_common = max(pattern_counts, key=pattern_counts.get)
                count = pattern_counts[most_common]
                total = sum(pattern_counts.values())
                percentage = (count / total * 100) if total > 0 else 0
                insights.append(f"Most common opening pattern: {most_common} ({percentage:.1f}%)")

        return insights
