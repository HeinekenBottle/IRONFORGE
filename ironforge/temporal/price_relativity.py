#!/usr/bin/env python3
"""
IRONFORGE Price Relativity Engine
Archaeological zone calculations and Theory B temporal non-locality analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

class PriceRelativityEngine:
    """Handles archaeological zone calculations and Theory B temporal non-locality analysis"""
    
    def __init__(self):
        # Archaeological zone calculations handled internally
        pass
        
    def analyze_archaeological_zones(self, question: str, sessions: Dict[str, pd.DataFrame], 
                                   session_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze archaeological zone patterns and Theory B events"""
        results = {
            "query_type": "archaeological_zones",
            "total_sessions": len(sessions),
            "zone_analysis": {},
            "theory_b_events": [],
            "insights": []
        }
        
        # Extract zone criteria from question
        zone_criteria = self._parse_zone_criteria(question)
        
        for session_id, nodes_df in sessions.items():
            if session_id not in session_stats:
                continue
                
            stats = session_stats[session_id]
            session_type = self._determine_session_type(session_id)
            
            # Calculate archaeological zones for this session
            zone_analysis = self._calculate_session_zones(nodes_df, stats, session_type)
            
            # Detect Theory B events
            theory_b_events = self._detect_theory_b_events(nodes_df, stats, zone_analysis)
            
            results["zone_analysis"][session_id] = zone_analysis
            results["theory_b_events"].extend(theory_b_events)
            
        # Generate insights
        results["insights"] = self._generate_zone_insights(results)
        
        return results
        
    def analyze_theory_b_patterns(self, question: str, sessions: Dict[str, pd.DataFrame],
                                session_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze Theory B temporal non-locality patterns"""
        results = {
            "query_type": "theory_b_patterns",
            "total_sessions": len(sessions),
            "precision_events": [],
            "non_locality_patterns": [],
            "temporal_correlations": {},
            "insights": []
        }
        
        # Extract Theory B criteria from question
        theory_b_criteria = self._parse_theory_b_criteria(question)
        
        for session_id, nodes_df in sessions.items():
            if session_id not in session_stats:
                continue
                
            stats = session_stats[session_id]
            
            # Detect precision events (Theory B markers)
            precision_events = self._detect_precision_events(nodes_df, stats, theory_b_criteria)
            
            # Analyze temporal non-locality patterns
            non_locality_patterns = self._analyze_non_locality_patterns(nodes_df, precision_events)
            
            results["precision_events"].extend(precision_events)
            results["non_locality_patterns"].extend(non_locality_patterns)
            
        # Calculate temporal correlations
        results["temporal_correlations"] = self._calculate_temporal_correlations(results["precision_events"])
        
        # Generate insights
        results["insights"] = self._generate_theory_b_insights(results)
        
        return results
        
    def analyze_post_rd40_sequences(self, question: str, sessions: Dict[str, pd.DataFrame],
                                  session_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze sequence patterns after RD@40% events"""
        results = {
            "query_type": "post_rd40_sequences",
            "total_sessions": len(sessions),
            "rd40_events": [],
            "sequence_paths": {
                "CONT": [],  # Continuation paths
                "MR": [],    # Mean reversion paths
                "ACCEL": []  # Acceleration paths
            },
            "path_probabilities": {},
            "insights": []
        }
        
        # Detect RD@40% events across all sessions
        rd40_events = self._detect_rd40_events(sessions, session_stats)
        results["rd40_events"] = rd40_events
        
        # Classify sequence paths for each RD@40% event
        for event in rd40_events:
            session_id = event["session_id"]
            event_index = event["event_index"]
            
            if session_id in sessions:
                path_classification = self._classify_sequence_path(
                    sessions[session_id], session_stats[session_id], event_index
                )
                
                path_type = path_classification["path_type"]
                if path_type in results["sequence_paths"]:
                    results["sequence_paths"][path_type].append({
                        "session_id": session_id,
                        "event_index": event_index,
                        "classification": path_classification
                    })
                    
        # Calculate path probabilities
        results["path_probabilities"] = self._calculate_path_probabilities(results["sequence_paths"])
        
        # Generate insights
        results["insights"] = self._generate_rd40_insights(results)
        
        return results
        
    def _calculate_session_zones(self, nodes_df: pd.DataFrame, stats: Dict[str, float], 
                               session_type: str) -> Dict[str, Any]:
        """Calculate archaeological zones for a session"""
        if 'price' not in nodes_df.columns:
            return {}
            
        session_high = stats.get('high', 0)
        session_low = stats.get('low', 0)
        session_range = session_high - session_low
        
        if session_range == 0:
            return {}
            
        zone_analysis = {
            "session_range": session_range,
            "zone_boundaries": {},
            "zone_events": {},
            "zone_statistics": {}
        }
        
        # Calculate zone boundaries (20%, 40%, 60%, 80%)
        for zone_pct in [20, 40, 60, 80]:
            zone_price = session_low + (session_range * zone_pct / 100)
            zone_analysis["zone_boundaries"][f"{zone_pct}%"] = zone_price
            
        # Classify events by zone
        for zone_pct in [20, 40, 60, 80]:
            zone_price = zone_analysis["zone_boundaries"][f"{zone_pct}%"]
            tolerance = session_range * 0.02  # 2% tolerance
            
            # Find events near this zone
            zone_events = nodes_df[
                (nodes_df['price'] >= zone_price - tolerance) &
                (nodes_df['price'] <= zone_price + tolerance)
            ]
            
            zone_analysis["zone_events"][f"{zone_pct}%"] = len(zone_events)
            
        return zone_analysis
        
    def _detect_theory_b_events(self, nodes_df: pd.DataFrame, stats: Dict[str, float],
                              zone_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect Theory B temporal non-locality events"""
        theory_b_events = []
        
        if 'price' not in nodes_df.columns or 'timestamp' in nodes_df.columns:
            return theory_b_events
            
        # Look for precision events at archaeological zones
        for zone_pct in [40, 60, 80]:  # Focus on key zones
            zone_price = zone_analysis.get("zone_boundaries", {}).get(f"{zone_pct}%")
            if zone_price is None:
                continue
                
            session_range = stats.get('range', 0)
            tolerance = session_range * 0.01  # 1% tolerance for precision
            
            # Find precision events at this zone
            precision_events = nodes_df[
                (nodes_df['price'] >= zone_price - tolerance) &
                (nodes_df['price'] <= zone_price + tolerance)
            ]
            
            for idx, event in precision_events.iterrows():
                # Check for Theory B characteristics
                if self._has_theory_b_characteristics(event, nodes_df, idx):
                    theory_b_events.append({
                        "event_index": idx,
                        "zone_percentage": zone_pct,
                        "price": event['price'],
                        "zone_price": zone_price,
                        "precision_score": self._calculate_precision_score(event, zone_price, tolerance),
                        "characteristics": self._extract_theory_b_characteristics(event, nodes_df, idx)
                    })
                    
        return theory_b_events
        
    def _detect_rd40_events(self, sessions: Dict[str, pd.DataFrame],
                          session_stats: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Detect FPFVG redelivery events at 40% archaeological zones"""
        rd40_events = []
        
        for session_id, nodes_df in sessions.items():
            if session_id not in session_stats:
                continue
                
            stats = session_stats[session_id]
            session_range = stats.get('range', 0)
            
            if session_range == 0:
                continue
                
            # Calculate 40% zone
            zone_40_price = stats['low'] + (session_range * 0.4)
            tolerance = session_range * 0.02  # 2% tolerance
            
            # Find events near 40% zone
            zone_events = nodes_df[
                (nodes_df['price'] >= zone_40_price - tolerance) &
                (nodes_df['price'] <= zone_40_price + tolerance)
            ]
            
            for idx, event in zone_events.iterrows():
                # Check for redelivery characteristics
                if self._has_redelivery_characteristics(nodes_df, idx):
                    rd40_events.append({
                        "session_id": session_id,
                        "event_index": idx,
                        "price": event['price'],
                        "zone_price": zone_40_price,
                        "redelivery_score": self._calculate_redelivery_score(event, nodes_df, idx),
                        "features": self._extract_rd40_features(nodes_df, idx)
                    })
                    
        return rd40_events
        
    def _classify_sequence_path(self, nodes_df: pd.DataFrame, stats: Dict[str, float],
                              event_index: int) -> Dict[str, Any]:
        """Classify the sequence path after RD@40% event: CONT/MR/ACCEL"""
        classification = {
            "path_type": "UNKNOWN",
            "confidence": 0.0,
            "timing_analysis": {},
            "features": {}
        }
        
        # Look ahead from the RD@40% event
        future_events = nodes_df.iloc[event_index:event_index+50]  # Look ahead 50 events
        
        if len(future_events) < 10:
            return classification
            
        session_range = stats.get('range', 0)
        session_low = stats.get('low', 0)
        
        # Calculate target zones
        zone_60 = session_low + (session_range * 0.6)
        zone_80 = session_low + (session_range * 0.8)
        mid_range = session_low + (session_range * 0.5)
        
        # Analyze price movement patterns
        max_price_reached = future_events['price'].max()
        time_to_max = future_events['price'].idxmax() - event_index
        
        # Classification logic
        if max_price_reached >= zone_80:
            if time_to_max <= 20:  # Quick move to 80%
                classification["path_type"] = "ACCEL"
                classification["confidence"] = 0.8
            else:
                classification["path_type"] = "CONT"
                classification["confidence"] = 0.7
        elif max_price_reached >= zone_60:
            classification["path_type"] = "CONT"
            classification["confidence"] = 0.6
        else:
            # Check for mean reversion
            if max_price_reached <= mid_range:
                classification["path_type"] = "MR"
                classification["confidence"] = 0.7
                
        classification["timing_analysis"] = {
            "max_price_reached": max_price_reached,
            "time_to_max": time_to_max,
            "zone_60_reached": max_price_reached >= zone_60,
            "zone_80_reached": max_price_reached >= zone_80
        }
        
        return classification

    def _parse_zone_criteria(self, question: str) -> Dict[str, Any]:
        """Parse archaeological zone criteria from question"""
        criteria = {
            "zones": [],
            "time_window": None,
            "precision_threshold": 0.02
        }

        # Extract zone percentages
        import re
        zone_matches = re.findall(r'(\d+)%', question)
        criteria["zones"] = [int(z) for z in zone_matches if int(z) in [20, 40, 60, 80]]

        # Extract time window
        if "within" in question.lower():
            time_match = re.search(r'within (\d+) (minute|hour|day)', question.lower())
            if time_match:
                value, unit = time_match.groups()
                criteria["time_window"] = (int(value), unit)

        return criteria

    def _parse_theory_b_criteria(self, question: str) -> Dict[str, Any]:
        """Parse Theory B criteria from question"""
        criteria = {
            "precision_threshold": 0.01,
            "temporal_window": 30,  # minutes
            "non_locality_threshold": 0.05
        }

        # Extract precision requirements
        if "precision" in question.lower():
            criteria["precision_threshold"] = 0.005  # Higher precision

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

    def _has_theory_b_characteristics(self, event: pd.Series, nodes_df: pd.DataFrame,
                                    event_idx: int) -> bool:
        """Check if event has Theory B temporal non-locality characteristics"""
        # Look for precision timing and energy characteristics
        if 'energy_density' in event.index and event['energy_density'] > 0.8:
            return True

        # Check for liquidity characteristics
        if 'liquidity_score' in event.index and event['liquidity_score'] > 0.7:
            return True

        # Check for temporal clustering
        nearby_events = nodes_df.iloc[max(0, event_idx-5):event_idx+5]
        if len(nearby_events) > 8:  # High event density
            return True

        return False

    def _calculate_precision_score(self, event: pd.Series, zone_price: float,
                                 tolerance: float) -> float:
        """Calculate precision score for Theory B event"""
        price_diff = abs(event['price'] - zone_price)
        precision_score = 1.0 - (price_diff / tolerance)
        return max(0.0, min(1.0, precision_score))

    def _extract_theory_b_characteristics(self, event: pd.Series, nodes_df: pd.DataFrame,
                                        event_idx: int) -> Dict[str, Any]:
        """Extract Theory B characteristics from event"""
        characteristics = {
            "energy_density": event.get('energy_density', 0.0),
            "liquidity_score": event.get('liquidity_score', 0.0),
            "temporal_clustering": 0.0,
            "price_precision": 0.0
        }

        # Calculate temporal clustering
        nearby_events = nodes_df.iloc[max(0, event_idx-10):event_idx+10]
        characteristics["temporal_clustering"] = len(nearby_events) / 20.0

        return characteristics

    def _has_redelivery_characteristics(self, nodes_df: pd.DataFrame, event_idx: int) -> bool:
        """Check if event has FPFVG redelivery characteristics"""
        if event_idx >= len(nodes_df):
            return False

        event = nodes_df.iloc[event_idx]

        # Check for liquidity and energy features
        if 'liquidity_score' in event.index and event['liquidity_score'] > 0.6:
            if 'energy_density' in event.index and event['energy_density'] > 0.5:
                return True

        # Check for volume characteristics
        if 'volume' in event.index and event['volume'] > nodes_df['volume'].quantile(0.8):
            return True

        return False

    def _calculate_redelivery_score(self, event: pd.Series, nodes_df: pd.DataFrame,
                                  event_idx: int) -> float:
        """Calculate redelivery score for RD@40% event"""
        score = 0.0

        # Liquidity component
        if 'liquidity_score' in event.index:
            score += event['liquidity_score'] * 0.4

        # Energy component
        if 'energy_density' in event.index:
            score += event['energy_density'] * 0.3

        # Volume component
        if 'volume' in event.index:
            volume_percentile = (event['volume'] > nodes_df['volume']).mean()
            score += volume_percentile * 0.3

        return min(1.0, score)

    def _extract_rd40_features(self, nodes_df: pd.DataFrame, event_idx: int) -> Dict[str, Any]:
        """Extract relevant features for RD@40% event analysis"""
        if event_idx >= len(nodes_df):
            return {}

        event = nodes_df.iloc[event_idx]
        features = {}

        # Extract available features
        feature_columns = ['f8_q', 'f8_slope_sign', 'f47_barpos_m15', 'liquidity_score',
                          'energy_density', 'volume', 'price_momentum']

        for col in feature_columns:
            if col in event.index:
                features[col] = event[col]

        return features

    def _calculate_path_probabilities(self, sequence_paths: Dict[str, List]) -> Dict[str, Any]:
        """Calculate path probabilities with confidence intervals"""
        total_events = sum(len(paths) for paths in sequence_paths.values())

        if total_events == 0:
            return {}

        probabilities = {}
        for path_type, paths in sequence_paths.items():
            count = len(paths)
            probability = count / total_events

            # Calculate Wilson confidence interval
            if count > 0:
                z = 1.96  # 95% confidence
                p = probability
                n = total_events

                denominator = 1 + z**2/n
                center = (p + z**2/(2*n)) / denominator
                margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator

                probabilities[path_type] = {
                    "probability": probability,
                    "count": count,
                    "confidence_interval": [max(0, center - margin), min(1, center + margin)]
                }
            else:
                probabilities[path_type] = {
                    "probability": 0.0,
                    "count": 0,
                    "confidence_interval": [0.0, 0.0]
                }

        return probabilities

    def _detect_precision_events(self, nodes_df: pd.DataFrame, stats: Dict[str, float],
                               criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect precision events based on Theory B criteria"""
        precision_events = []

        if 'price' not in nodes_df.columns:
            return precision_events

        precision_threshold = criteria.get("precision_threshold", 0.01)
        session_range = stats.get('range', 0)

        if session_range == 0:
            return precision_events

        # Look for high-precision events
        for idx, event in nodes_df.iterrows():
            # Check energy density
            energy_density = event.get('energy_density', 0)
            if energy_density > 0.8:
                precision_events.append({
                    "event_index": idx,
                    "energy_density": energy_density,
                    "precision_score": energy_density,
                    "characteristics": self._extract_theory_b_characteristics(event, nodes_df, idx)
                })

        return precision_events

    def _analyze_non_locality_patterns(self, nodes_df: pd.DataFrame,
                                     precision_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze temporal non-locality patterns around precision events"""
        non_locality_patterns = []

        for precision_event in precision_events:
            event_idx = precision_event["event_index"]

            # Look for non-local correlations
            pattern = {
                "precision_event_index": event_idx,
                "temporal_correlations": [],
                "non_locality_score": 0.0
            }

            # Check for events at similar price levels but different times
            if event_idx < len(nodes_df):
                event_price = nodes_df.iloc[event_idx].get('price', 0)

                # Look for price echoes
                for idx, other_event in nodes_df.iterrows():
                    if abs(idx - event_idx) > 10:  # Non-local in time
                        price_diff = abs(other_event.get('price', 0) - event_price)
                        if price_diff < event_price * 0.01:  # Similar price
                            pattern["temporal_correlations"].append({
                                "event_index": idx,
                                "time_separation": abs(idx - event_idx),
                                "price_similarity": 1.0 - (price_diff / event_price)
                            })

            pattern["non_locality_score"] = len(pattern["temporal_correlations"]) / 10.0
            non_locality_patterns.append(pattern)

        return non_locality_patterns

    def _calculate_temporal_correlations(self, precision_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate temporal correlations between precision events"""
        if len(precision_events) < 2:
            return {}

        correlations = {
            "event_count": len(precision_events),
            "average_separation": 0.0,
            "correlation_strength": 0.0
        }

        # Calculate average separation between events
        separations = []
        for i in range(len(precision_events) - 1):
            sep = precision_events[i+1]["event_index"] - precision_events[i]["event_index"]
            separations.append(sep)

        if separations:
            correlations["average_separation"] = np.mean(separations)
            correlations["separation_std"] = np.std(separations)

        return correlations

    def _generate_zone_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from archaeological zone analysis"""
        insights = []

        zone_analysis = results.get("zone_analysis", {})
        theory_b_events = results.get("theory_b_events", [])

        if zone_analysis:
            # Analyze zone activity
            total_events_by_zone = {}
            for session_id, analysis in zone_analysis.items():
                zone_events = analysis.get("zone_events", {})
                for zone, count in zone_events.items():
                    total_events_by_zone[zone] = total_events_by_zone.get(zone, 0) + count

            if total_events_by_zone:
                most_active_zone = max(total_events_by_zone, key=total_events_by_zone.get)
                insights.append(f"Most active archaeological zone: {most_active_zone} with {total_events_by_zone[most_active_zone]} events")

        if theory_b_events:
            avg_precision = np.mean([event.get("precision_score", 0) for event in theory_b_events])
            insights.append(f"Theory B events show average precision score of {avg_precision:.3f}")

            # Zone distribution of Theory B events
            zone_counts = {}
            for event in theory_b_events:
                zone = f"{event.get('zone_percentage', 0)}%"
                zone_counts[zone] = zone_counts.get(zone, 0) + 1

            if zone_counts:
                preferred_zone = max(zone_counts, key=zone_counts.get)
                insights.append(f"Theory B events prefer {preferred_zone} zone ({zone_counts[preferred_zone]} events)")

        return insights

    def _generate_theory_b_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from Theory B pattern analysis"""
        insights = []

        precision_events = results.get("precision_events", [])
        non_locality_patterns = results.get("non_locality_patterns", [])
        temporal_correlations = results.get("temporal_correlations", {})

        if precision_events:
            insights.append(f"Detected {len(precision_events)} precision events with Theory B characteristics")

            # Analyze precision distribution
            precision_scores = [event.get("precision_score", 0) for event in precision_events]
            if precision_scores:
                avg_precision = np.mean(precision_scores)
                max_precision = max(precision_scores)
                insights.append(f"Precision scores range from {min(precision_scores):.3f} to {max_precision:.3f} (avg: {avg_precision:.3f})")

        if non_locality_patterns:
            total_correlations = sum(len(pattern.get("temporal_correlations", [])) for pattern in non_locality_patterns)
            insights.append(f"Found {total_correlations} temporal non-locality correlations across {len(non_locality_patterns)} patterns")

        if temporal_correlations:
            avg_sep = temporal_correlations.get("average_separation", 0)
            if avg_sep > 0:
                insights.append(f"Average temporal separation between precision events: {avg_sep:.1f} time units")

        return insights

    def _generate_rd40_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from RD@40% sequence analysis"""
        insights = []

        rd40_events = results.get("rd40_events", [])
        sequence_paths = results.get("sequence_paths", {})
        path_probabilities = results.get("path_probabilities", {})

        if rd40_events:
            insights.append(f"Identified {len(rd40_events)} RD@40% redelivery events")

            # Analyze redelivery scores
            redelivery_scores = [event.get("redelivery_score", 0) for event in rd40_events]
            if redelivery_scores:
                avg_score = np.mean(redelivery_scores)
                insights.append(f"Average redelivery score: {avg_score:.3f}")

        if sequence_paths:
            total_paths = sum(len(paths) for paths in sequence_paths.values())
            if total_paths > 0:
                insights.append(f"Classified {total_paths} sequence paths after RD@40% events")

                # Find dominant path type
                path_counts = {path_type: len(paths) for path_type, paths in sequence_paths.items()}
                if path_counts:
                    dominant_path = max(path_counts, key=path_counts.get)
                    dominant_percentage = (path_counts[dominant_path] / total_paths) * 100
                    insights.append(f"Dominant path type: {dominant_path} ({dominant_percentage:.1f}% of sequences)")

        if path_probabilities:
            # Highlight most probable path
            best_path = None
            best_prob = 0
            for path_type, prob_data in path_probabilities.items():
                prob = prob_data.get("probability", 0)
                if prob > best_prob:
                    best_prob = prob
                    best_path = path_type

            if best_path:
                ci = path_probabilities[best_path].get("confidence_interval", [0, 0])
                insights.append(f"Highest probability path: {best_path} ({best_prob:.1%}, 95% CI: {ci[0]:.1%}-{ci[1]:.1%})")

        return insights
