#!/usr/bin/env python3
"""
Core Analysis Functions for Predictive Condition Hunter
Implements the actual pattern detection and probability calculations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import itertools

class ConditionAnalyzerCore:
    """Core analysis functions for finding high-probability patterns"""
    
    def __init__(self, sessions: Dict[str, pd.DataFrame], top_features: List[str]):
        self.sessions = sessions
        self.top_features = top_features
        self.feature_stats = self._calculate_feature_statistics()
        
    def _calculate_feature_statistics(self) -> Dict[str, Dict]:
        """Calculate statistical baselines for all features across sessions"""
        stats = {}
        
        for feature in self.top_features:
            feature_values = []
            for session_id, nodes in self.sessions.items():
                if feature in nodes.columns:
                    feature_values.extend(nodes[feature].values)
            
            if feature_values:
                stats[feature] = {
                    "mean": np.mean(feature_values),
                    "std": np.std(feature_values),
                    "q25": np.percentile(feature_values, 25),
                    "q75": np.percentile(feature_values, 75),
                    "q85": np.percentile(feature_values, 85),
                    "q90": np.percentile(feature_values, 90),
                    "q95": np.percentile(feature_values, 95)
                }
        
        return stats
    
    def analyze_single_feature_patterns(self, feature: str, target_outcomes: List[str]) -> Dict[str, Any]:
        """Analyze patterns for a single feature with probability calculations"""
        
        if feature not in self.feature_stats:
            return {"sample_size": 0, "patterns": [], "probabilities": {}}
        
        feature_data = self.feature_stats[feature]
        patterns = []
        outcome_counts = defaultdict(lambda: defaultdict(int))
        
        # Define significance thresholds
        thresholds = {
            "high": feature_data["q90"],
            "very_high": feature_data["q95"],
            "low": feature_data["q25"],
            "normal": (feature_data["q25"], feature_data["q75"])
        }
        
        # Analyze each session
        for session_id, nodes in self.sessions.items():
            if feature not in nodes.columns or len(nodes) < 15:
                continue
                
            # Find significant feature events
            for i in range(10, len(nodes) - 10):
                feature_value = nodes.iloc[i][feature]
                
                # Classify feature level
                feature_level = self._classify_feature_level(feature_value, thresholds)
                
                if feature_level in ["high", "very_high", "low"]:  # Significant levels only
                    # Analyze outcome in next 5-15 minutes
                    outcome = self._analyze_future_outcome(
                        nodes, i, target_outcomes, 
                        min_lead_time=300000,  # 5 minutes
                        max_lead_time=900000   # 15 minutes
                    )
                    
                    if outcome["occurred"]:
                        patterns.append({
                            "session_id": session_id,
                            "feature": feature,
                            "feature_level": feature_level,
                            "feature_value": feature_value,
                            "outcome": outcome["outcome"],
                            "lead_time": outcome["lead_time"],
                            "confidence": outcome.get("confidence", 0.5)
                        })
                        
                        # Count for probability calculation
                        outcome_counts[feature_level][outcome["outcome"]] += 1
                        outcome_counts[feature_level]["total"] += 1
        
        # Calculate probabilities
        probabilities = {}
        for level, outcomes in outcome_counts.items():
            if outcomes["total"] >= 5:  # Minimum sample size
                level_probs = {}
                for outcome in target_outcomes:
                    if outcome in outcomes:
                        level_probs[outcome] = outcomes[outcome] / outcomes["total"]
                    else:
                        level_probs[outcome] = 0.0
                probabilities[level] = level_probs
        
        return {
            "sample_size": len(patterns),
            "patterns": patterns,
            "probabilities": probabilities,
            "feature_stats": feature_data
        }
    
    def analyze_feature_pair_patterns(self, f1: str, f2: str, target_outcomes: List[str]) -> Dict[str, Any]:
        """Analyze patterns for feature pairs"""
        
        if f1 not in self.feature_stats or f2 not in self.feature_stats:
            return {"sample_size": 0, "patterns": [], "probabilities": {}}
        
        patterns = []
        outcome_counts = defaultdict(lambda: defaultdict(int))
        
        # Get thresholds for both features
        f1_thresholds = {
            "high": self.feature_stats[f1]["q90"],
            "very_high": self.feature_stats[f1]["q95"]
        }
        f2_thresholds = {
            "high": self.feature_stats[f2]["q90"],
            "very_high": self.feature_stats[f2]["q95"]
        }
        
        # Analyze combinations
        for session_id, nodes in self.sessions.items():
            if (f1 not in nodes.columns or f2 not in nodes.columns or len(nodes) < 15):
                continue
                
            for i in range(10, len(nodes) - 10):
                f1_value = nodes.iloc[i][f1]
                f2_value = nodes.iloc[i][f2]
                
                # Check for significant combinations
                f1_sig = f1_value > f1_thresholds["high"]
                f2_sig = f2_value > f2_thresholds["high"]
                f1_very_sig = f1_value > f1_thresholds["very_high"]
                f2_very_sig = f2_value > f2_thresholds["very_high"]
                
                combination_type = None
                if f1_very_sig and f2_very_sig:
                    combination_type = "both_very_high"
                elif f1_sig and f2_sig:
                    combination_type = "both_high"
                elif f1_very_sig and f2_sig:
                    combination_type = f"{f1}_very_high_{f2}_high"
                elif f1_sig and f2_very_sig:
                    combination_type = f"{f1}_high_{f2}_very_high"
                
                if combination_type:
                    # Analyze outcome
                    outcome = self._analyze_future_outcome(
                        nodes, i, target_outcomes,
                        min_lead_time=180000,  # 3 minutes
                        max_lead_time=600000   # 10 minutes
                    )
                    
                    if outcome["occurred"]:
                        patterns.append({
                            "session_id": session_id,
                            "feature_pair": f"{f1}+{f2}",
                            "combination_type": combination_type,
                            "f1_value": f1_value,
                            "f2_value": f2_value,
                            "outcome": outcome["outcome"],
                            "lead_time": outcome["lead_time"],
                            "confidence": outcome.get("confidence", 0.5)
                        })
                        
                        outcome_counts[combination_type][outcome["outcome"]] += 1
                        outcome_counts[combination_type]["total"] += 1
        
        # Calculate probabilities
        probabilities = {}
        for combo_type, outcomes in outcome_counts.items():
            if outcomes["total"] >= 5:
                combo_probs = {}
                for outcome in target_outcomes:
                    combo_probs[outcome] = outcomes.get(outcome, 0) / outcomes["total"]
                probabilities[combo_type] = combo_probs
        
        return {
            "sample_size": len(patterns),
            "patterns": patterns,
            "probabilities": probabilities,
            "combination_types": list(outcome_counts.keys())
        }
    
    def analyze_complex_combinations(self, feature: str, zone_type: str, timing: str, 
                                   target_outcomes: List[str]) -> Dict[str, Any]:
        """Analyze complex feature+zone+timing combinations"""
        
        patterns = []
        outcome_counts = defaultdict(lambda: defaultdict(int))
        
        for session_id, nodes in self.sessions.items():
            if feature not in nodes.columns or len(nodes) < 20:
                continue
                
            # Define session phases
            session_length = len(nodes)
            timing_ranges = {
                "early_session": (0, int(session_length * 0.3)),
                "mid_session": (int(session_length * 0.3), int(session_length * 0.7)),
                "late_session": (int(session_length * 0.7), session_length)
            }
            
            if timing not in timing_ranges:
                continue
                
            start_idx, end_idx = timing_ranges[timing]
            
            # Calculate archaeological zones for this session
            session_high = nodes['price'].max()
            session_low = nodes['price'].min()
            session_range = session_high - session_low
            
            if session_range < 15:  # Skip very small ranges
                continue
                
            zone_prices = {
                "40%": session_low + (session_range * 0.4),
                "60%": session_low + (session_range * 0.6),
                "80%": session_low + (session_range * 0.8)
            }
            
            if zone_type not in zone_prices:
                continue
                
            zone_price = zone_prices[zone_type]
            
            # Look for feature+zone+timing combinations
            for i in range(max(10, start_idx), min(end_idx, len(nodes) - 10)):
                feature_value = nodes.iloc[i][feature]
                current_price = nodes.iloc[i]['price']
                
                # Check feature significance
                if feature in self.feature_stats:
                    feature_threshold = self.feature_stats[feature]["q85"]  # 85th percentile
                    feature_significant = feature_value > feature_threshold
                else:
                    continue
                    
                # Check zone proximity
                zone_distance = abs(current_price - zone_price)
                zone_proximity = zone_distance / session_range
                zone_near = zone_proximity < 0.08  # Within 8% of zone
                
                if feature_significant and zone_near:
                    # Found a complex pattern
                    combination_key = f"{feature}+{zone_type}_zone+{timing}"
                    
                    # Analyze outcome
                    outcome = self._analyze_future_outcome(
                        nodes, i, target_outcomes,
                        min_lead_time=120000,  # 2 minutes
                        max_lead_time=900000   # 15 minutes
                    )
                    
                    if outcome["occurred"]:
                        patterns.append({
                            "session_id": session_id,
                            "combination": combination_key,
                            "feature_value": feature_value,
                            "zone_proximity": zone_proximity,
                            "session_phase": timing,
                            "outcome": outcome["outcome"],
                            "lead_time": outcome["lead_time"],
                            "confidence": outcome.get("confidence", 0.5)
                        })
                        
                        outcome_counts[combination_key][outcome["outcome"]] += 1
                        outcome_counts[combination_key]["total"] += 1
        
        # Calculate probabilities
        probabilities = {}
        for combo_key, outcomes in outcome_counts.items():
            if outcomes["total"] >= 3:  # Lower threshold for complex patterns
                combo_probs = {}
                for outcome in target_outcomes:
                    combo_probs[outcome] = outcomes.get(outcome, 0) / outcomes["total"]
                probabilities[combo_key] = combo_probs
        
        return {
            "sample_size": len(patterns),
            "patterns": patterns,
            "probabilities": probabilities,
            "combinations": list(outcome_counts.keys())
        }
    
    def calculate_event_sequence_probabilities(self, event_sequences: List[Dict], 
                                             target_outcomes: List[str]) -> Dict[str, Any]:
        """Calculate probabilities for primary→secondary event sequences"""
        
        sequence_outcomes = defaultdict(lambda: defaultdict(int))
        
        for sequence in event_sequences:
            primary_type = sequence["primary_event"]["type"]
            secondary_type = sequence["secondary_event"]["type"]
            outcome = sequence["outcome"]["outcome"]
            timing_window = sequence["timing_window"]
            
            sequence_key = f"{primary_type}→{secondary_type}({timing_window})"
            
            sequence_outcomes[sequence_key][outcome] += 1
            sequence_outcomes[sequence_key]["total"] += 1
        
        # Calculate probabilities for sequences with sufficient data
        probabilities = {}
        for seq_key, outcomes in sequence_outcomes.items():
            if outcomes["total"] >= 5:  # Minimum sample size
                seq_probs = {
                    "total_occurrences": outcomes["total"],
                    "outcome_probabilities": {}
                }
                
                for outcome in target_outcomes:
                    probability = outcomes.get(outcome, 0) / outcomes["total"]
                    seq_probs["outcome_probabilities"][outcome] = probability
                
                # Calculate overall sequence probability
                best_outcome = max(seq_probs["outcome_probabilities"], 
                                 key=seq_probs["outcome_probabilities"].get)
                seq_probs["probability"] = seq_probs["outcome_probabilities"][best_outcome]
                seq_probs["best_outcome"] = best_outcome
                seq_probs["sample_size"] = outcomes["total"]
                
                probabilities[seq_key] = seq_probs
        
        return probabilities
    
    def test_hybrid_patterns(self, feature_clusters: Dict, primary_secondary: Dict, 
                           target_outcomes: List[str]) -> Dict[str, Any]:
        """Test hybrid patterns combining features and events"""
        
        hybrid_patterns = {}
        
        # Get promising feature patterns
        promising_features = []
        for cluster_type, clusters in feature_clusters.items():
            for cluster_name, cluster_data in clusters.items():
                if cluster_data.get("sample_size", 0) >= 5:
                    for outcome, prob in cluster_data.get("probabilities", {}).items():
                        if isinstance(prob, dict):
                            for sub_outcome, sub_prob in prob.items():
                                if sub_prob > 0.6:  # 60% threshold for feature patterns
                                    promising_features.append({
                                        "cluster_name": cluster_name,
                                        "outcome": sub_outcome,
                                        "probability": sub_prob
                                    })
        
        # Get promising event sequences
        event_probabilities = self.calculate_event_sequence_probabilities(
            primary_secondary.get("event_sequences", []), target_outcomes
        )
        
        promising_events = []
        for seq_key, seq_data in event_probabilities.items():
            if seq_data["probability"] > 0.6:  # 60% threshold
                promising_events.append({
                    "sequence": seq_key,
                    "outcome": seq_data["best_outcome"],
                    "probability": seq_data["probability"]
                })
        
        # Combine promising features with promising events
        for feature_pattern in promising_features[:5]:  # Top 5 features
            for event_pattern in promising_events[:3]:  # Top 3 events
                if feature_pattern["outcome"] == event_pattern["outcome"]:
                    # Same outcome - potential reinforcement
                    hybrid_key = f"{feature_pattern['cluster_name']}+{event_pattern['sequence']}"
                    
                    # Calculate combined probability (simplified model)
                    # P(A and B) ≈ P(A) * P(B) if independent, but we'll use a reinforcement model
                    base_prob = max(feature_pattern["probability"], event_pattern["probability"])
                    reinforcement = min(feature_pattern["probability"], event_pattern["probability"]) * 0.3
                    combined_prob = min(base_prob + reinforcement, 0.95)  # Cap at 95%
                    
                    hybrid_patterns[hybrid_key] = {
                        "feature_component": feature_pattern,
                        "event_component": event_pattern,
                        "outcome": feature_pattern["outcome"],
                        "probability": combined_prob,
                        "sample_size": 0,  # Would need to calculate from actual data
                        "pattern_type": "reinforcement"
                    }
        
        return hybrid_patterns
    
    def _classify_feature_level(self, value: float, thresholds: Dict[str, float]) -> str:
        """Classify feature value into significance levels"""
        if value > thresholds["very_high"]:
            return "very_high"
        elif value > thresholds["high"]:
            return "high"
        elif value < thresholds["low"]:
            return "low"
        else:
            return "normal"
    
    def _analyze_future_outcome(self, nodes: pd.DataFrame, event_index: int, 
                              target_outcomes: List[str], min_lead_time: int = 120000,
                              max_lead_time: int = 900000) -> Dict[str, Any]:
        """Analyze what outcome occurs in the future after an event"""
        
        event_time = nodes.iloc[event_index]['t']
        future_start_time = event_time + min_lead_time
        future_end_time = event_time + max_lead_time
        
        # Get future time window
        future_nodes = nodes[
            (nodes['t'] >= future_start_time) & 
            (nodes['t'] <= future_end_time) &
            (nodes.index > event_index)
        ]
        
        if len(future_nodes) == 0:
            return {"occurred": False, "outcome": None}
        
        # Test each outcome with confidence scoring
        outcome_scores = {}
        for outcome in target_outcomes:
            score = self._calculate_outcome_confidence(nodes, event_index, future_nodes, outcome)
            if score > 0:
                outcome_scores[outcome] = score
        
        if outcome_scores:
            # Return the highest confidence outcome
            best_outcome = max(outcome_scores, key=outcome_scores.get)
            confidence = outcome_scores[best_outcome]
            
            # Calculate lead time to first indication of outcome
            first_indication_time = self._find_first_outcome_indication(
                nodes, event_index, future_nodes, best_outcome
            )
            
            return {
                "occurred": True,
                "outcome": best_outcome,
                "confidence": confidence,
                "lead_time": first_indication_time - event_time
            }
        
        return {"occurred": False, "outcome": None}
    
    def _calculate_outcome_confidence(self, nodes: pd.DataFrame, event_index: int,
                                    future_nodes: pd.DataFrame, outcome: str) -> float:
        """Calculate confidence score for a specific outcome"""
        
        if outcome == "expansion":
            return self._confidence_expansion(nodes, event_index, future_nodes)
        elif outcome == "retracement":
            return self._confidence_retracement(nodes, event_index, future_nodes)
        elif outcome == "reversal":
            return self._confidence_reversal(nodes, event_index, future_nodes)
        elif outcome == "consolidation":
            return self._confidence_consolidation(nodes, event_index, future_nodes)
        elif outcome == "fpfvg_redelivery":
            return self._confidence_fpfvg_redelivery(nodes, event_index, future_nodes)
        
        return 0.0
    
    def _confidence_expansion(self, nodes: pd.DataFrame, event_index: int,
                            future_nodes: pd.DataFrame) -> float:
        """Calculate confidence for expansion outcome"""
        if len(future_nodes) == 0:
            return 0.0
            
        future_range = future_nodes['price'].max() - future_nodes['price'].min()
        recent_range = nodes.iloc[max(0, event_index-10):event_index]['price'].max() - \
                      nodes.iloc[max(0, event_index-10):event_index]['price'].min()
        
        if recent_range <= 0:
            return 0.0
            
        expansion_ratio = future_range / recent_range
        
        if expansion_ratio > 2.0:
            return 0.9  # High confidence
        elif expansion_ratio > 1.5:
            return 0.7  # Medium-high confidence
        elif expansion_ratio > 1.2:
            return 0.5  # Medium confidence
        else:
            return 0.0
    
    def _confidence_retracement(self, nodes: pd.DataFrame, event_index: int,
                              future_nodes: pd.DataFrame) -> float:
        """Calculate confidence for retracement outcome"""
        if len(future_nodes) == 0 or event_index < 5:
            return 0.0
            
        # Determine trend
        event_price = nodes.iloc[event_index]['price']
        trend_start_price = nodes.iloc[event_index-5]['price']
        trend_move = abs(event_price - trend_start_price)
        
        if trend_move < 10:  # Need significant trend
            return 0.0
            
        trend_direction = "up" if event_price > trend_start_price else "down"
        
        # Calculate retracement
        if trend_direction == "up":
            min_future_price = future_nodes['price'].min()
            retracement_amount = event_price - min_future_price
        else:
            max_future_price = future_nodes['price'].max()
            retracement_amount = max_future_price - event_price
        
        retracement_ratio = retracement_amount / trend_move
        
        if retracement_ratio > 0.6:
            return 0.8  # High confidence
        elif retracement_ratio > 0.4:
            return 0.6  # Medium-high confidence
        elif retracement_ratio > 0.2:
            return 0.4  # Medium confidence
        else:
            return 0.0
    
    def _confidence_reversal(self, nodes: pd.DataFrame, event_index: int,
                           future_nodes: pd.DataFrame) -> float:
        """Calculate confidence for reversal outcome"""
        if len(future_nodes) == 0 or event_index < 5:
            return 0.0
            
        event_price = nodes.iloc[event_index]['price']
        trend_start_price = nodes.iloc[event_index-5]['price']
        
        if abs(event_price - trend_start_price) < 15:  # Need significant trend
            return 0.0
            
        trend_direction = "up" if event_price > trend_start_price else "down"
        future_end_price = future_nodes['price'].iloc[-1]
        
        # Calculate reversal strength
        if trend_direction == "up":
            reversal_amount = event_price - future_end_price
            reversal_occurred = future_end_price < event_price * 0.98
        else:
            reversal_amount = future_end_price - event_price
            reversal_occurred = future_end_price > event_price * 1.02
        
        if not reversal_occurred:
            return 0.0
            
        reversal_strength = abs(reversal_amount) / abs(event_price - trend_start_price)
        
        if reversal_strength > 0.8:
            return 0.9
        elif reversal_strength > 0.5:
            return 0.7
        elif reversal_strength > 0.2:
            return 0.5
        else:
            return 0.0
    
    def _confidence_consolidation(self, nodes: pd.DataFrame, event_index: int,
                                future_nodes: pd.DataFrame) -> float:
        """Calculate confidence for consolidation outcome"""
        if len(future_nodes) == 0:
            return 0.0
            
        future_range = future_nodes['price'].max() - future_nodes['price'].min()
        recent_volatility = nodes.iloc[max(0, event_index-10):event_index]['price'].std()
        
        if recent_volatility <= 0:
            return 0.0
            
        consolidation_ratio = future_range / recent_volatility
        
        if consolidation_ratio < 0.3:
            return 0.8
        elif consolidation_ratio < 0.5:
            return 0.6
        elif consolidation_ratio < 0.8:
            return 0.4
        else:
            return 0.0
    
    def _confidence_fpfvg_redelivery(self, nodes: pd.DataFrame, event_index: int,
                                   future_nodes: pd.DataFrame) -> float:
        """Calculate confidence for FPFVG redelivery outcome"""
        if len(future_nodes) == 0:
            return 0.0
            
        event_price = nodes.iloc[event_index]['price']
        
        # Look for price returning to event area with varying tolerances
        for tolerance, confidence in [(5, 0.9), (10, 0.7), (15, 0.5), (20, 0.3)]:
            redelivery_events = future_nodes[
                abs(future_nodes['price'] - event_price) <= tolerance
            ]
            
            if len(redelivery_events) > 0:
                return confidence
        
        return 0.0
    
    def _find_first_outcome_indication(self, nodes: pd.DataFrame, event_index: int,
                                     future_nodes: pd.DataFrame, outcome: str) -> int:
        """Find the timestamp of first indication of the outcome"""
        if len(future_nodes) == 0:
            return nodes.iloc[event_index]['t']
        
        # For simplicity, return the timestamp of the first future node
        # In practice, this could be more sophisticated
        return future_nodes.iloc[0]['t']