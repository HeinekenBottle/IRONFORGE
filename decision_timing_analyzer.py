#!/usr/bin/env python3
"""
Decision Timing Intelligence System for IRONFORGE
Identifies patterns that predict subsequent market behavior for optimal decision timing
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from temporal_query_engine import TemporalQueryEngine
from enhanced_fpfvg_analyzer import EnhancedFPFVGAnalyzer
import warnings
warnings.filterwarnings('ignore')

class DecisionTimingAnalyzer:
    """Analyze timing signals and their effects on subsequent market behavior"""
    
    def __init__(self):
        print("🎯 Initializing Decision Timing Intelligence System...")
        self.engine = TemporalQueryEngine()
        self.fpfvg_analyzer = EnhancedFPFVGAnalyzer()
        self.timing_signals = self._identify_timing_signal_types()
        print(f"✅ Loaded {len(self.engine.sessions)} sessions for timing analysis")
        
    def _identify_timing_signal_types(self) -> Dict[str, Dict]:
        """Identify categories of timing signals to detect"""
        return {
            "feature_spikes": {
                "description": "Sudden increases in feature values (especially f8, f9)",
                "detection_method": "statistical_outlier",
                "lead_time": "immediate_to_5min",
                "affects": ["consolidation", "expansion", "liquidity_taken"]
            },
            "price_exhaustion": {
                "description": "Price moving to extremes with decreasing momentum",
                "detection_method": "momentum_divergence",
                "lead_time": "2_to_10min", 
                "affects": ["retracement", "reversal", "fpfvg_redelivery"]
            },
            "volume_anomalies": {
                "description": "Unusual f8 intensity patterns",
                "detection_method": "f8_analysis",
                "lead_time": "1_to_15min",
                "affects": ["expansion", "liquidity_taken", "fpfvg_redelivery"]
            },
            "session_position": {
                "description": "Timing within session development (early/mid/late)",
                "detection_method": "temporal_position",
                "lead_time": "session_level",
                "affects": ["consolidation", "expansion", "retracement"]
            },
            "archaeological_zones": {
                "description": "Approach to or interaction with 40%, 60%, 80% zones",
                "detection_method": "zone_proximity",
                "lead_time": "immediate_to_session",
                "affects": ["reversal", "fpfvg_redelivery", "liquidity_taken"]
            },
            "cross_session_setup": {
                "description": "Previous session patterns affecting current timing",
                "detection_method": "session_dependency",
                "lead_time": "session_to_next_session",
                "affects": ["fpfvg_redelivery", "expansion", "consolidation"]
            }
        }
    
    def analyze_timing_signals_and_effects(self) -> Dict[str, Any]:
        """Comprehensive analysis of timing signals and their subsequent effects"""
        print("\n🔍 Analyzing timing signals across all sessions...")
        
        results = {
            "signal_analysis": {},
            "effect_predictions": {},
            "timing_optimization": {},
            "practical_insights": []
        }
        
        # Analyze each signal type
        for signal_type, signal_info in self.timing_signals.items():
            print(f"\n📊 Analyzing {signal_type} signals...")
            signal_results = self._analyze_signal_type(signal_type, signal_info)
            results["signal_analysis"][signal_type] = signal_results
        
        # Generate effect predictions
        results["effect_predictions"] = self._generate_effect_predictions(results["signal_analysis"])
        
        # Create timing optimization guidelines
        results["timing_optimization"] = self._create_timing_guidelines(results)
        
        # Generate practical insights
        results["practical_insights"] = self._generate_practical_insights(results)
        
        return results
    
    def _analyze_signal_type(self, signal_type: str, signal_info: Dict) -> Dict[str, Any]:
        """Analyze a specific type of timing signal"""
        signal_results = {
            "total_signals_detected": 0,
            "signal_instances": [],
            "effect_correlations": {},
            "timing_stats": {}
        }
        
        for session_id, nodes in self.engine.sessions.items():
            if len(nodes) < 10:  # Skip very short sessions
                continue
                
            # Detect signals based on type
            detected_signals = self._detect_signals_in_session(nodes, session_id, signal_type)
            
            for signal in detected_signals:
                # Analyze effects of this signal
                effects = self._analyze_signal_effects(nodes, signal, signal_info["affects"])
                
                signal_instance = {
                    "session_id": session_id,
                    "signal_time": signal["time"],
                    "signal_strength": signal["strength"],
                    "signal_details": signal,
                    "observed_effects": effects
                }
                
                signal_results["signal_instances"].append(signal_instance)
                signal_results["total_signals_detected"] += 1
        
        # Calculate effect correlations
        signal_results["effect_correlations"] = self._calculate_effect_correlations(
            signal_results["signal_instances"], signal_info["affects"]
        )
        
        return signal_results
    
    def _detect_signals_in_session(self, nodes: pd.DataFrame, session_id: str, signal_type: str) -> List[Dict]:
        """Detect specific signal type within a session"""
        signals = []
        
        if signal_type == "feature_spikes":
            signals.extend(self._detect_feature_spikes(nodes))
        elif signal_type == "price_exhaustion":
            signals.extend(self._detect_price_exhaustion(nodes))
        elif signal_type == "volume_anomalies":
            signals.extend(self._detect_volume_anomalies(nodes))
        elif signal_type == "session_position":
            signals.extend(self._detect_session_position_signals(nodes))
        elif signal_type == "archaeological_zones":
            signals.extend(self._detect_archaeological_zones(nodes))
        elif signal_type == "cross_session_setup":
            signals.extend(self._detect_cross_session_signals(nodes, session_id))
        
        return signals
    
    def _detect_feature_spikes(self, nodes: pd.DataFrame) -> List[Dict]:
        """Detect sudden spikes in feature values"""
        signals = []
        
        # Focus on most important features
        important_features = ['f8', 'f9', 'f4', 'f1', 'f3']
        
        for feature in important_features:
            if feature not in nodes.columns:
                continue
                
            feature_values = nodes[feature]
            if feature_values.var() == 0:
                continue
                
            # Detect spikes (values > 2 standard deviations above mean)
            mean_val = feature_values.mean()
            std_val = feature_values.std()
            spike_threshold = mean_val + (2 * std_val)
            
            spike_indices = nodes[feature_values > spike_threshold].index
            
            for idx in spike_indices:
                if idx < len(nodes) - 5:  # Ensure we have future data
                    signals.append({
                        "type": "feature_spike",
                        "time": nodes.iloc[idx]['t'],
                        "feature": feature,
                        "value": feature_values.iloc[idx],
                        "strength": (feature_values.iloc[idx] - mean_val) / std_val,
                        "index": idx
                    })
        
        return signals
    
    def _detect_price_exhaustion(self, nodes: pd.DataFrame) -> List[Dict]:
        """Detect price exhaustion patterns"""
        signals = []
        
        if len(nodes) < 15:
            return signals
            
        # Calculate momentum (price change rate)
        price_changes = nodes['price'].diff()
        momentum = price_changes.rolling(window=5).mean()
        
        # Detect exhaustion: large price moves with decreasing momentum
        for i in range(10, len(nodes) - 5):
            current_price_move = abs(price_changes.iloc[i])
            current_momentum = abs(momentum.iloc[i])
            recent_momentum = abs(momentum.iloc[i-5:i].mean())
            
            # Exhaustion: big move but momentum decreasing
            if (current_price_move > price_changes.std() and 
                current_momentum < recent_momentum * 0.7):
                
                signals.append({
                    "type": "price_exhaustion",
                    "time": nodes.iloc[i]['t'],
                    "price": nodes.iloc[i]['price'],
                    "momentum_decline": (recent_momentum - current_momentum) / recent_momentum,
                    "strength": current_price_move / price_changes.std(),
                    "index": i
                })
        
        return signals
    
    def _detect_volume_anomalies(self, nodes: pd.DataFrame) -> List[Dict]:
        """Detect f8 volume anomalies"""
        signals = []
        
        if 'f8' not in nodes.columns or nodes['f8'].var() == 0:
            return signals
            
        f8_values = nodes['f8']
        
        # Rolling mean and std for anomaly detection
        rolling_mean = f8_values.rolling(window=10, min_periods=3).mean()
        rolling_std = f8_values.rolling(window=10, min_periods=3).std()
        
        # Detect anomalies
        for i in range(10, len(nodes) - 5):
            current_f8 = f8_values.iloc[i]
            expected_f8 = rolling_mean.iloc[i]
            f8_std = rolling_std.iloc[i]
            
            if f8_std > 0:
                z_score = abs(current_f8 - expected_f8) / f8_std
                
                if z_score > 2.5:  # Significant anomaly
                    signals.append({
                        "type": "volume_anomaly",
                        "time": nodes.iloc[i]['t'],
                        "f8_value": current_f8,
                        "z_score": z_score,
                        "anomaly_direction": "high" if current_f8 > expected_f8 else "low",
                        "strength": z_score,
                        "index": i
                    })
        
        return signals
    
    def _detect_session_position_signals(self, nodes: pd.DataFrame) -> List[Dict]:
        """Detect signals based on position within session"""
        signals = []
        
        session_length = len(nodes)
        
        # Define session phases
        early_phase = int(session_length * 0.2)  # First 20%
        late_phase = int(session_length * 0.8)   # Last 20%
        
        # Early session high activity
        if session_length > 20:
            early_activity = len(nodes[:early_phase])
            late_activity = len(nodes[late_phase:])
            
            if early_activity > session_length * 0.3:  # High early activity
                signals.append({
                    "type": "early_high_activity",
                    "time": nodes.iloc[early_phase]['t'],
                    "activity_ratio": early_activity / session_length,
                    "strength": early_activity / (session_length * 0.2),
                    "index": early_phase
                })
            
            if late_activity > session_length * 0.15:  # High late activity
                signals.append({
                    "type": "late_high_activity", 
                    "time": nodes.iloc[late_phase]['t'],
                    "activity_ratio": late_activity / session_length,
                    "strength": late_activity / (session_length * 0.2),
                    "index": late_phase
                })
        
        return signals
    
    def _detect_archaeological_zones(self, nodes: pd.DataFrame) -> List[Dict]:
        """Detect approach to or interaction with archaeological zones"""
        signals = []
        
        if len(nodes) < 10:
            return signals
            
        session_high = nodes['price'].max()
        session_low = nodes['price'].min()
        session_range = session_high - session_low
        
        if session_range < 10:  # Skip very small ranges
            return signals
        
        # Define archaeological zones
        zone_40 = session_low + (session_range * 0.4)
        zone_60 = session_low + (session_range * 0.6)
        zone_80 = session_low + (session_range * 0.8)
        
        # Detect approaches to zones
        for i in range(5, len(nodes) - 5):
            current_price = nodes.iloc[i]['price']
            
            # Check proximity to each zone
            for zone_level, zone_price in [("40%", zone_40), ("60%", zone_60), ("80%", zone_80)]:
                distance = abs(current_price - zone_price)
                proximity_ratio = distance / session_range
                
                if proximity_ratio < 0.03:  # Within 3% of zone
                    signals.append({
                        "type": "archaeological_zone_approach",
                        "time": nodes.iloc[i]['t'],
                        "zone_level": zone_level,
                        "zone_price": zone_price,
                        "current_price": current_price,
                        "proximity_ratio": proximity_ratio,
                        "strength": 1.0 - proximity_ratio,
                        "index": i
                    })
        
        return signals
    
    def _detect_cross_session_signals(self, nodes: pd.DataFrame, session_id: str) -> List[Dict]:
        """Detect cross-session setup signals"""
        signals = []
        
        # Find previous session
        session_sequence = self.fpfvg_analyzer.base_analyzer.session_sequence
        current_session_info = None
        
        for session_info in session_sequence:
            if session_info['session_id'] == session_id:
                current_session_info = session_info
                break
        
        if not current_session_info:
            return signals
            
        current_index = session_sequence.index(current_session_info)
        
        if current_index > 0:
            prev_session_info = session_sequence[current_index - 1]
            prev_session_id = prev_session_info['session_id']
            
            if prev_session_id in self.engine.sessions:
                prev_nodes = self.engine.sessions[prev_session_id]
                
                # Analyze previous session characteristics
                prev_range = prev_nodes['price'].max() - prev_nodes['price'].min()
                prev_f8_intensity = prev_nodes['f8'].std() if 'f8' in prev_nodes.columns else 0
                
                # Signal if previous session had high range or f8 intensity
                if prev_range > 100 or prev_f8_intensity > 1000:
                    signals.append({
                        "type": "cross_session_setup",
                        "time": nodes.iloc[0]['t'],  # Beginning of current session
                        "prev_session": prev_session_id,
                        "prev_range": prev_range,
                        "prev_f8_intensity": prev_f8_intensity,
                        "strength": min((prev_range / 100) + (prev_f8_intensity / 1000), 5.0),
                        "index": 0
                    })
        
        return signals
    
    def _analyze_signal_effects(self, nodes: pd.DataFrame, signal: Dict, possible_effects: List[str]) -> Dict[str, Any]:
        """Analyze the effects of a detected signal on subsequent market behavior"""
        effects = {}
        
        signal_index = signal["index"]
        signal_time = signal["time"]
        
        # Look ahead 15 minutes (900,000 ms) or to end of session
        future_time_limit = signal_time + 900000  # 15 minutes
        future_nodes = nodes[
            (nodes['t'] > signal_time) & 
            (nodes['t'] <= future_time_limit) &
            (nodes.index > signal_index)
        ]
        
        if len(future_nodes) == 0:
            return effects
        
        # Analyze each possible effect
        if "consolidation" in possible_effects:
            effects["consolidation"] = self._measure_consolidation(nodes, signal_index, future_nodes)
            
        if "expansion" in possible_effects:
            effects["expansion"] = self._measure_expansion(nodes, signal_index, future_nodes)
            
        if "retracement" in possible_effects:
            effects["retracement"] = self._measure_retracement(nodes, signal_index, future_nodes)
            
        if "reversal" in possible_effects:
            effects["reversal"] = self._measure_reversal(nodes, signal_index, future_nodes)
            
        if "liquidity_taken" in possible_effects:
            effects["liquidity_taken"] = self._measure_liquidity_consumption(future_nodes)
            
        if "fpfvg_redelivery" in possible_effects:
            effects["fpfvg_redelivery"] = self._measure_fpfvg_redelivery(nodes, signal_index, future_nodes)
        
        return effects
    
    def _measure_consolidation(self, nodes: pd.DataFrame, signal_index: int, future_nodes: pd.DataFrame) -> Dict[str, float]:
        """Measure consolidation behavior after signal"""
        if len(future_nodes) == 0:
            return {"occurred": False, "strength": 0.0}
            
        signal_price = nodes.iloc[signal_index]['price']
        future_range = future_nodes['price'].max() - future_nodes['price'].min()
        
        # Consolidation = small range relative to recent volatility
        recent_volatility = nodes.iloc[max(0, signal_index-10):signal_index]['price'].std()
        
        if recent_volatility > 0:
            consolidation_ratio = future_range / recent_volatility
            consolidation_occurred = consolidation_ratio < 0.5
            
            return {
                "occurred": consolidation_occurred,
                "strength": max(0, 1.0 - consolidation_ratio) if consolidation_occurred else 0.0,
                "range": future_range,
                "ratio": consolidation_ratio
            }
        
        return {"occurred": False, "strength": 0.0}
    
    def _measure_expansion(self, nodes: pd.DataFrame, signal_index: int, future_nodes: pd.DataFrame) -> Dict[str, float]:
        """Measure expansion behavior after signal"""
        if len(future_nodes) == 0:
            return {"occurred": False, "strength": 0.0}
            
        signal_price = nodes.iloc[signal_index]['price']
        future_range = future_nodes['price'].max() - future_nodes['price'].min()
        
        # Expansion = large range relative to recent action
        recent_range = nodes.iloc[max(0, signal_index-10):signal_index]['price'].max() - \
                      nodes.iloc[max(0, signal_index-10):signal_index]['price'].min()
        
        if recent_range > 0:
            expansion_ratio = future_range / recent_range
            expansion_occurred = expansion_ratio > 1.5
            
            return {
                "occurred": expansion_occurred,
                "strength": min(expansion_ratio / 2.0, 2.0) if expansion_occurred else 0.0,
                "range": future_range,
                "ratio": expansion_ratio
            }
        
        return {"occurred": False, "strength": 0.0}
    
    def _measure_retracement(self, nodes: pd.DataFrame, signal_index: int, future_nodes: pd.DataFrame) -> Dict[str, float]:
        """Measure retracement behavior after signal"""
        if len(future_nodes) == 0:
            return {"occurred": False, "strength": 0.0}
            
        signal_price = nodes.iloc[signal_index]['price']
        
        # Determine recent trend direction
        if signal_index >= 5:
            trend_start_price = nodes.iloc[signal_index-5]['price']
            trend_direction = "up" if signal_price > trend_start_price else "down"
            
            # Look for retracement (movement against trend)
            if trend_direction == "up":
                retracement_level = min(future_nodes['price'])
                retracement_amount = signal_price - retracement_level
            else:
                retracement_level = max(future_nodes['price'])
                retracement_amount = retracement_level - signal_price
            
            if retracement_amount > 0:
                recent_move = abs(signal_price - trend_start_price)
                retracement_ratio = retracement_amount / max(recent_move, 1)
                
                retracement_occurred = retracement_ratio > 0.3  # 30% retracement
                
                return {
                    "occurred": retracement_occurred,
                    "strength": min(retracement_ratio, 1.0),
                    "amount": retracement_amount,
                    "ratio": retracement_ratio
                }
        
        return {"occurred": False, "strength": 0.0}
    
    def _measure_reversal(self, nodes: pd.DataFrame, signal_index: int, future_nodes: pd.DataFrame) -> Dict[str, float]:
        """Measure reversal behavior after signal"""
        if len(future_nodes) == 0 or signal_index < 5:
            return {"occurred": False, "strength": 0.0}
            
        # Determine trend before signal
        pre_signal_nodes = nodes.iloc[signal_index-5:signal_index]
        signal_price = nodes.iloc[signal_index]['price']
        
        if len(pre_signal_nodes) < 3:
            return {"occurred": False, "strength": 0.0}
            
        pre_trend_direction = "up" if pre_signal_nodes['price'].iloc[-1] > pre_signal_nodes['price'].iloc[0] else "down"
        
        # Check if future movement reverses the trend
        future_end_price = future_nodes['price'].iloc[-1]
        
        if pre_trend_direction == "up":
            reversal_occurred = future_end_price < signal_price * 0.98  # 2% reversal threshold
            reversal_strength = (signal_price - future_end_price) / signal_price
        else:
            reversal_occurred = future_end_price > signal_price * 1.02  # 2% reversal threshold
            reversal_strength = (future_end_price - signal_price) / signal_price
        
        return {
            "occurred": reversal_occurred,
            "strength": min(abs(reversal_strength) * 10, 1.0) if reversal_occurred else 0.0,
            "direction": "bullish" if pre_trend_direction == "down" else "bearish"
        }
    
    def _measure_liquidity_consumption(self, future_nodes: pd.DataFrame) -> Dict[str, float]:
        """Measure liquidity consumption after signal"""
        if len(future_nodes) == 0 or 'f8' not in future_nodes.columns:
            return {"occurred": False, "strength": 0.0}
            
        f8_activity = future_nodes['f8'].sum()
        event_density = len(future_nodes)
        
        # High liquidity consumption = high f8 + high event density
        liquidity_score = min((f8_activity / 1000) + (event_density / 20), 2.0)
        
        return {
            "occurred": liquidity_score > 0.5,
            "strength": min(liquidity_score / 2.0, 1.0),
            "f8_total": f8_activity,
            "event_count": event_density
        }
    
    def _measure_fpfvg_redelivery(self, nodes: pd.DataFrame, signal_index: int, future_nodes: pd.DataFrame) -> Dict[str, float]:
        """Measure FPFVG redelivery after signal"""
        if len(future_nodes) == 0:
            return {"occurred": False, "strength": 0.0}
            
        signal_price = nodes.iloc[signal_index]['price']
        
        # Look for price returning to signal area (gap redelivery)
        redelivery_tolerance = 10  # 10 point tolerance
        
        redelivery_events = future_nodes[
            abs(future_nodes['price'] - signal_price) <= redelivery_tolerance
        ]
        
        if len(redelivery_events) > 0:
            first_redelivery = redelivery_events.iloc[0]
            accuracy = redelivery_tolerance - abs(first_redelivery['price'] - signal_price)
            
            return {
                "occurred": True,
                "strength": accuracy / redelivery_tolerance,
                "redelivery_price": first_redelivery['price'],
                "accuracy": accuracy
            }
        
        return {"occurred": False, "strength": 0.0}
    
    def _calculate_effect_correlations(self, signal_instances: List[Dict], possible_effects: List[str]) -> Dict[str, float]:
        """Calculate correlation between signals and effects"""
        correlations = {}
        
        for effect in possible_effects:
            effect_occurrences = []
            signal_strengths = []
            
            for instance in signal_instances:
                if effect in instance["observed_effects"]:
                    effect_data = instance["observed_effects"][effect]
                    effect_occurrences.append(1.0 if effect_data.get("occurred", False) else 0.0)
                    signal_strengths.append(instance["signal_strength"])
            
            if len(effect_occurrences) > 5:  # Minimum sample size
                correlation = np.corrcoef(signal_strengths, effect_occurrences)[0, 1]
                correlations[effect] = correlation if not np.isnan(correlation) else 0.0
            else:
                correlations[effect] = 0.0
        
        return correlations
    
    def _generate_effect_predictions(self, signal_analysis: Dict) -> Dict[str, Any]:
        """Generate predictive insights about signal effects"""
        predictions = {}
        
        for signal_type, analysis in signal_analysis.items():
            if analysis["total_signals_detected"] == 0:
                continue
                
            correlations = analysis["effect_correlations"]
            
            # Find strongest correlations
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            predictions[signal_type] = {
                "strongest_effect": sorted_correlations[0] if sorted_correlations else None,
                "reliable_predictions": [item for item in sorted_correlations if abs(item[1]) > 0.3],
                "sample_size": analysis["total_signals_detected"]
            }
        
        return predictions
    
    def _create_timing_guidelines(self, results: Dict) -> Dict[str, Any]:
        """Create practical timing guidelines based on analysis"""
        guidelines = {
            "immediate_signals": [],    # 0-2 minutes
            "short_term_signals": [],   # 2-10 minutes  
            "session_signals": [],      # Session level
            "cross_session_signals": [] # Next session
        }
        
        for signal_type, predictions in results["effect_predictions"].items():
            if predictions["sample_size"] < 5:  # Skip low sample sizes
                continue
                
            signal_info = self.timing_signals[signal_type]
            lead_time = signal_info["lead_time"]
            
            if predictions["strongest_effect"]:
                effect, correlation = predictions["strongest_effect"]
                
                guideline = {
                    "signal": signal_type,
                    "watch_for": effect,
                    "correlation": correlation,
                    "confidence": "high" if abs(correlation) > 0.5 else "medium"
                }
                
                if "immediate" in lead_time:
                    guidelines["immediate_signals"].append(guideline)
                elif "2_to_10min" in lead_time or "1_to_15min" in lead_time:
                    guidelines["short_term_signals"].append(guideline)
                elif "session" in lead_time:
                    guidelines["session_signals"].append(guideline)
                else:
                    guidelines["cross_session_signals"].append(guideline)
        
        return guidelines
    
    def _generate_practical_insights(self, results: Dict) -> List[str]:
        """Generate practical trading insights"""
        insights = []
        
        # High-confidence patterns
        for signal_type, predictions in results["effect_predictions"].items():
            if predictions["strongest_effect"] and abs(predictions["strongest_effect"][1]) > 0.5:
                effect, correlation = predictions["strongest_effect"]
                direction = "increases" if correlation > 0 else "decreases"
                
                insights.append(
                    f"{signal_type.replace('_', ' ').title()} signals {direction} "
                    f"probability of {effect.replace('_', ' ')} by {abs(correlation):.1%}"
                )
        
        # Sample size warnings
        low_sample_signals = [s for s, p in results["effect_predictions"].items() 
                            if p["sample_size"] < 10]
        if low_sample_signals:
            insights.append(
                f"Low sample size for: {', '.join(low_sample_signals)} - need more data for confidence"
            )
        
        return insights

def analyze_decision_timing():
    """Run comprehensive decision timing analysis"""
    print("🎯 IRONFORGE Decision Timing Intelligence Analysis")
    print("=" * 60)
    
    analyzer = DecisionTimingAnalyzer()
    results = analyzer.analyze_timing_signals_and_effects()
    
    print("\n📊 TIMING SIGNAL ANALYSIS RESULTS")
    print("=" * 40)
    
    # Summary statistics
    total_signals = sum(analysis["total_signals_detected"] 
                       for analysis in results["signal_analysis"].values())
    
    print(f"Total signals detected: {total_signals}")
    print(f"Signal types analyzed: {len(results['signal_analysis'])}")
    
    # Effect predictions
    print(f"\n🎯 STRONGEST SIGNAL-EFFECT CORRELATIONS:")
    for signal_type, predictions in results["effect_predictions"].items():
        if predictions["strongest_effect"] and abs(predictions["strongest_effect"][1]) > 0.3:
            effect, correlation = predictions["strongest_effect"]
            print(f"• {signal_type}: {effect} (r={correlation:.3f}, n={predictions['sample_size']})")
    
    # Timing guidelines
    print(f"\n⏰ TIMING GUIDELINES:")
    guidelines = results["timing_optimization"]
    
    if guidelines["immediate_signals"]:
        print(f"Immediate (0-2min) signals:")
        for guide in guidelines["immediate_signals"]:
            print(f"  • {guide['signal']} → {guide['watch_for']} ({guide['confidence']})")
    
    if guidelines["short_term_signals"]:
        print(f"Short-term (2-15min) signals:")
        for guide in guidelines["short_term_signals"]:
            print(f"  • {guide['signal']} → {guide['watch_for']} ({guide['confidence']})")
    
    # Practical insights
    print(f"\n💡 PRACTICAL INSIGHTS:")
    for i, insight in enumerate(results["practical_insights"], 1):
        print(f"{i}. {insight}")
    
    return results

if __name__ == "__main__":
    results = analyze_decision_timing()