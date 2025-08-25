#!/usr/bin/env python3
"""
IRONFORGE Daily Temporal Non-Locality Research Framework
H-Daily1 HYPOTHESIS: Inter-Day Dimensional Architecture Theory

SCALING DISCOVERY: Session-level temporal non-locality (7.55-point precision, 
18-minute lead time) scaled to daily timeframes with previous day 40% zone 
reactions predicting current day high/low positioning.

CRITICAL DISTINCTION:
- Session-level: Events position for future completion within same session
- Daily-level: Previous day reactions contain predictive information for next day extremes
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

@dataclass
class DailyReactionSignature:
    """Signature structure for daily 40% zone reactions"""
    reaction_date: str
    reaction_timestamp: str
    previous_day_40pct_level: float
    reaction_price: float
    reaction_type: str  # "touch", "break_above", "break_below", "reject"
    
    # Timing characteristics (scaled from 3-13 second session signatures)
    reaction_duration_minutes: int  # Daily equivalent: 15-60 minutes
    approach_velocity: float  # Points per minute approaching zone
    departure_velocity: float  # Points per minute departing zone
    
    # Prediction targets (next day)
    predicted_extreme: str  # "high" or "low"
    prediction_confidence: float  # 0.0-1.0 based on signature strength
    
    # Validation data (filled after next day completes)
    actual_high: Optional[float] = None
    actual_low: Optional[float] = None
    prediction_accuracy: Optional[float] = None
    prediction_success: Optional[bool] = None

class DailyTemporalNonLocalityHypothesis:
    """
    H-Daily1 HYPOTHESIS: Inter-Day Dimensional Architecture Theory
    
    CORE MECHANISM:
    Previous day 40% zone reactions exhibit temporal non-locality across day boundaries,
    containing predictive information about next day high/low positioning through
    inter-day dimensional architecture relationships.
    """
    
    def __init__(self):
        self.hypothesis_id = "H-Daily1"
        self.hypothesis_name = "Inter-Day Dimensional Architecture Theory"
        self.session_reference_precision = 7.55  # Reference from Theory B
        self.session_reference_lead_time = 18    # Minutes from Theory B
        
        # Daily scaling factors (session -> daily conversion)
        self.daily_scaling_factors = {
            "temporal_lead_multiplier": 80,      # 18 minutes -> ~24 hours
            "precision_scaling": 0.3,           # Expect ~30% of session precision
            "signature_duration_multiplier": 4,  # 3-13 seconds -> 15-60 minutes
            "cascade_probability_scaling": 0.75  # 87.5% -> ~65% for daily
        }
    
    def get_daily_hypothesis_statement(self) -> Dict[str, Any]:
        """Get the complete H-Daily1 hypothesis statement with scaled mechanisms"""
        return {
            "hypothesis_id": "H-Daily1",
            "title": "Inter-Day Dimensional Architecture Theory",
            "parent_discovery": "H1 Session-Level Temporal Non-Locality (Theory B)",
            
            "core_mechanism": {
                "primary_principle": "Inter-Day Temporal Non-Locality in Archaeological Zone Reactions",
                "scaled_mathematical_foundation": {
                    "expected_daily_precision": self.session_reference_precision * self.daily_scaling_factors["precision_scaling"],
                    "expected_lead_time_hours": self.session_reference_lead_time * self.daily_scaling_factors["temporal_lead_multiplier"] / 60,
                    "signature_duration_range": "15-60 minutes (scaled from 3-13 seconds)",
                    "cascade_probability": f"{87.5 * self.daily_scaling_factors['cascade_probability_scaling']:.1f}%"
                }
            },
            
            "reaction_signature_classification": self._define_daily_reaction_signatures(),
            "directional_bias_theory": self._define_directional_bias_framework(),
            "daily_testable_predictions": self._generate_daily_testable_predictions()
        }
    
    def _define_daily_reaction_signatures(self) -> Dict[str, Any]:
        """Define classification system for daily 40% zone reaction signatures"""
        return {
            "signature_types": {
                "Bullish_Extreme_Predictor": {
                    "reaction_pattern": "Break above previous day 40% with sustained momentum",
                    "duration": "30-60 minutes sustained above zone",
                    "velocity_profile": "High approach velocity, sustained departure velocity",
                    "prediction_target": "Next day high"
                },
                
                "Bearish_Extreme_Predictor": {
                    "reaction_pattern": "Break below previous day 40% with momentum continuation",
                    "duration": "30-60 minutes sustained below zone", 
                    "velocity_profile": "High approach velocity, accelerating departure velocity",
                    "prediction_target": "Next day low"
                },
                
                "Reversal_Touch_Signature": {
                    "reaction_pattern": "Precise touch of 40% level with immediate rejection",
                    "duration": "15-30 minutes touch and reversal",
                    "velocity_profile": "Moderate approach, high departure velocity opposite direction",
                    "prediction_target": "Opposite extreme"
                }
            }
        }
    
    def _define_directional_bias_framework(self) -> Dict[str, Any]:
        """Define how to determine if reaction predicts high vs low"""
        return {
            "directional_bias_rules": {
                "High_Prediction_Indicators": [
                    "Break above previous day 40% with momentum",
                    "Touch from below with sharp rejection upward", 
                    "Late-session strength above 40% zone"
                ],
                
                "Low_Prediction_Indicators": [
                    "Break below previous day 40% with momentum",
                    "Touch from above with sharp rejection downward",
                    "Late-session weakness below 40% zone"
                ]
            }
        }
    
    def _generate_daily_testable_predictions(self) -> List[Dict[str, Any]]:
        """Generate specific testable predictions for daily temporal non-locality"""
        return [
            {
                "prediction_id": "PD1.1", 
                "statement": "Previous day 40% zone reactions will predict next day extremes with >65% accuracy",
                "success_criteria": "Prediction accuracy >65% (scaled from 87.5% session cascade probability)"
            },
            {
                "prediction_id": "PD1.2",
                "statement": "Daily precision will be ~30% of session-level precision (scaled Theory B)",
                "success_criteria": "Average precision ~2.27 points (scaled from 7.55-point session precision)"
            },
            {
                "prediction_id": "PD1.3",
                "statement": "Late-session reactions will show higher prediction accuracy than early-session",
                "success_criteria": "Last 2 hours of session show >20% higher accuracy than first 2 hours"
            }
        ]

# TODO(human): Implement daily reaction signature detector
def detect_daily_reaction_signatures(previous_day_data: Dict[str, Any], 
                                   current_day_data: Dict[str, Any]) -> List[DailyReactionSignature]:
    """
    Detect and classify daily reaction signatures at previous day 40% zones
    
    Context: I've established the theoretical framework for daily temporal non-locality 
    with reaction signature classification and directional bias theory. The signature
    detector needs to identify 15-60 minute reaction patterns at previous day 40% levels
    and classify them as bullish/bearish extreme predictors.
    
    Your Task: Implement the core signature detection logic that identifies when current
    day price action reacts significantly to previous day 40% archaeological zones.
    Look for TODO(human) in the function body.
    
    Guidance: Analyze current day events for interactions with previous day 40% level,
    measure reaction duration and velocity profiles, classify signatures based on the
    framework (Bullish_Extreme_Predictor, Bearish_Extreme_Predictor, Reversal_Touch_Signature),
    and assign directional bias and confidence scores. Consider timing factors - late-session
    reactions should have higher prediction confidence.
    """
    
    """
    Detect daily reaction signatures at previous day 40% archaeological zones
    """
    
    # Extract previous day data
    prev_day_high = previous_day_data.get('high', 0)
    prev_day_low = previous_day_data.get('low', 0)
    prev_day_range = prev_day_high - prev_day_low
    
    if prev_day_range <= 0:
        return []
    
    # Calculate previous day 40% level
    prev_40pct_level = prev_day_low + (prev_day_range * 0.4)
    
    # Extract current day events
    current_events = current_day_data.get('events', [])
    if not current_events:
        return []
    
    detected_signatures = []
    interaction_threshold = 3.0  # Points - within 2-3 points as specified
    
    # Find interactions with previous day 40% level
    interaction_events = []
    for event in current_events:
        price = event.get('price', 0)
        timestamp = event.get('timestamp', '')
        
        distance_to_40pct = abs(price - prev_40pct_level)
        
        if distance_to_40pct <= interaction_threshold:
            interaction_events.append({
                'price': price,
                'timestamp': timestamp,
                'distance': distance_to_40pct,
                'above_40pct': price > prev_40pct_level
            })
    
    if not interaction_events:
        return []
    
    # Group interactions into reaction sequences, including continuation events
    reaction_sequences = _group_interactions_into_extended_sequences(interaction_events, current_events, prev_40pct_level)
    
    # Process sequences for signature detection
    
    # Classify each reaction sequence
    for sequence in reaction_sequences:
        if len(sequence) < 2:  # Need at least approach and departure
            continue
            
        # Calculate reaction characteristics
        start_event = sequence[0]
        end_event = sequence[-1]
        reaction_duration = _calculate_duration_minutes(start_event['timestamp'], end_event['timestamp'])
        
        # Filter for 15-60 minute duration range
        if not (15 <= reaction_duration <= 60):
            continue
        
        # Calculate velocity profiles
        approach_velocity = _calculate_approach_velocity(sequence, prev_40pct_level)
        departure_velocity = _calculate_departure_velocity(sequence, prev_40pct_level)
        
        # Classify reaction type based on velocity and price behavior
        reaction_type, predicted_extreme = _classify_reaction_signature(
            sequence, prev_40pct_level, approach_velocity, departure_velocity
        )
        
        # Calculate confidence based on timing, duration, and velocity
        confidence = _calculate_reaction_confidence(
            reaction_duration, approach_velocity, departure_velocity, 
            start_event['timestamp'], reaction_type
        )
        
        # Create reaction signature
        signature = DailyReactionSignature(
            reaction_date=current_day_data.get('date', ''),
            reaction_timestamp=start_event['timestamp'],
            previous_day_40pct_level=prev_40pct_level,
            reaction_price=start_event['price'],
            reaction_type=reaction_type,
            reaction_duration_minutes=reaction_duration,
            approach_velocity=approach_velocity,
            departure_velocity=departure_velocity,
            predicted_extreme=predicted_extreme,
            prediction_confidence=confidence
        )
        
        detected_signatures.append(signature)
    
    return detected_signatures

def _group_interactions_into_extended_sequences(interaction_events: List[Dict], all_events: List[Dict], target_level: float) -> List[List[Dict]]:
    """Group interaction events into extended sequences including continuation events"""
    if not interaction_events:
        return []
    
    # Sort interaction events by timestamp
    sorted_interactions = sorted(interaction_events, key=lambda x: x['timestamp'])
    sequences = []
    
    for interaction in sorted_interactions:
        # Find all events within 60 minutes of this interaction
        interaction_time = interaction['timestamp']
        extended_sequence = []
        
        for event in all_events:
            event_time = event['timestamp']
            time_diff = _calculate_duration_minutes(interaction_time, event_time)
            
            # Include events within 60 minutes after interaction (both directions)
            if abs(time_diff) <= 60:
                extended_sequence.append({
                    'price': event['price'],
                    'timestamp': event['timestamp'],
                    'distance': abs(event['price'] - target_level),
                    'above_40pct': event['price'] > target_level
                })
        
        # Sort by timestamp and filter for minimum duration
        extended_sequence.sort(key=lambda x: x['timestamp'])
        if len(extended_sequence) >= 2:
            duration = _calculate_duration_minutes(extended_sequence[0]['timestamp'], extended_sequence[-1]['timestamp'])
            if 15 <= duration <= 60:  # Only keep sequences in our target duration range
                sequences.append(extended_sequence)
    
    # Remove duplicate sequences
    unique_sequences = []
    for seq in sequences:
        # Check if this sequence is already included
        is_duplicate = False
        for existing in unique_sequences:
            if (seq[0]['timestamp'] == existing[0]['timestamp'] and 
                seq[-1]['timestamp'] == existing[-1]['timestamp']):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_sequences.append(seq)
    
    return unique_sequences

def _group_interactions_into_sequences(interaction_events: List[Dict]) -> List[List[Dict]]:
    """Group interaction events into coherent reaction sequences"""
    if not interaction_events:
        return []
    
    # Sort by timestamp
    sorted_events = sorted(interaction_events, key=lambda x: x['timestamp'])
    
    sequences = []
    current_sequence = [sorted_events[0]]
    
    for i in range(1, len(sorted_events)):
        current_event = sorted_events[i]
        prev_event = sorted_events[i-1]
        
        # If events are within 10 minutes, consider them part of same sequence
        time_gap = _calculate_duration_minutes(prev_event['timestamp'], current_event['timestamp'])
        
        if time_gap <= 10:  # 10 minute max gap within sequence
            current_sequence.append(current_event)
        else:
            # Start new sequence
            if len(current_sequence) >= 2:
                sequences.append(current_sequence)
            current_sequence = [current_event]
    
    # Add final sequence if it has enough events
    if len(current_sequence) >= 2:
        sequences.append(current_sequence)
    
    return sequences

def _calculate_duration_minutes(start_time: str, end_time: str) -> int:
    """Calculate duration between timestamps in minutes"""
    try:
        # Simple time parsing - assumes HH:MM:SS format
        start_parts = start_time.split(':')
        end_parts = end_time.split(':')
        
        start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
        end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
        
        duration = end_minutes - start_minutes
        if duration < 0:  # Handle day boundary
            duration += 24 * 60
            
        return duration
    except:
        return 30  # Default to 30 minutes if parsing fails

def _calculate_approach_velocity(sequence: List[Dict], target_level: float) -> float:
    """Calculate approach velocity (points per minute) toward 40% level"""
    if len(sequence) < 2:
        return 0.0
    
    # Find closest point to target level
    closest_event = min(sequence, key=lambda x: x['distance'])
    first_event = sequence[0]
    
    distance_change = first_event['distance'] - closest_event['distance']
    time_duration = _calculate_duration_minutes(first_event['timestamp'], closest_event['timestamp'])
    
    return distance_change / max(time_duration, 1)  # Avoid division by zero

def _calculate_departure_velocity(sequence: List[Dict], target_level: float) -> float:
    """Calculate departure velocity (points per minute) away from 40% level"""
    if len(sequence) < 2:
        return 0.0
    
    # Find closest point and final point
    closest_event = min(sequence, key=lambda x: x['distance'])
    final_event = sequence[-1]
    
    distance_change = final_event['distance'] - closest_event['distance']
    time_duration = _calculate_duration_minutes(closest_event['timestamp'], final_event['timestamp'])
    
    return distance_change / max(time_duration, 1)  # Avoid division by zero

def _classify_reaction_signature(sequence: List[Dict], target_level: float, 
                               approach_vel: float, departure_vel: float) -> Tuple[str, str]:
    """Classify reaction signature and predict extreme direction"""
    
    closest_event = min(sequence, key=lambda x: x['distance'])
    final_event = sequence[-1]
    
    # Check if reaction sustained above/below level
    sustained_above = all(event['above_40pct'] for event in sequence[-3:]) if len(sequence) >= 3 else final_event['above_40pct']
    sustained_below = all(not event['above_40pct'] for event in sequence[-3:]) if len(sequence) >= 3 else not final_event['above_40pct']
    
    # High approach and departure velocity = momentum break
    if approach_vel > 2.0 and departure_vel > 1.5:
        if sustained_above:
            return "break_above", "high"  # Bullish Extreme Predictor
        elif sustained_below:
            return "break_below", "low"   # Bearish Extreme Predictor
    
    # Low distance with high departure opposite direction = reversal touch
    if closest_event['distance'] <= 1.5 and departure_vel > 2.0:
        if final_event['above_40pct'] and not closest_event['above_40pct']:
            return "touch_and_reject_up", "high"  # Reversal Touch - touch from below, reject up
        elif not final_event['above_40pct'] and closest_event['above_40pct']:
            return "touch_and_reject_down", "low"  # Reversal Touch - touch from above, reject down
    
    # Default classification
    if final_event['above_40pct']:
        return "approach_above", "high"
    else:
        return "approach_below", "low"

def _calculate_reaction_confidence(duration: int, approach_vel: float, departure_vel: float, 
                                 timestamp: str, reaction_type: str) -> float:
    """Calculate prediction confidence based on reaction characteristics"""
    confidence = 0.5  # Base confidence
    
    # Duration factor: longer reactions = higher confidence
    if duration >= 45:
        confidence += 0.2
    elif duration >= 30:
        confidence += 0.1
    
    # Velocity factor: higher velocity differential = stronger signal
    velocity_factor = min(0.2, (approach_vel + departure_vel) / 10)
    confidence += velocity_factor
    
    # Reaction type factor
    type_bonuses = {
        "break_above": 0.15,
        "break_below": 0.15, 
        "touch_and_reject_up": 0.2,
        "touch_and_reject_down": 0.2
    }
    confidence += type_bonuses.get(reaction_type, 0.05)
    
    # Late-session timing bonus (as specified)
    try:
        hour = int(timestamp.split(':')[0])
        if 15 <= hour <= 17:  # Last 2 hours of typical session
            confidence += 0.15  # Late-session reactions get higher confidence
    except:
        pass
    
    return min(1.0, confidence)  # Cap at 1.0

def validate_daily_hypothesis_framework():
    """Validation framework for H-Daily1 hypothesis testing"""
    print("🔬 H-DAILY1 HYPOTHESIS VALIDATION FRAMEWORK")
    print("=" * 70)
    
    daily_hypothesis = DailyTemporalNonLocalityHypothesis()
    h_daily_statement = daily_hypothesis.get_daily_hypothesis_statement()
    
    print("📋 Daily Hypothesis Summary:")
    print(f"   ID: {h_daily_statement['hypothesis_id']}")
    print(f"   Title: {h_daily_statement['title']}")
    print(f"   Parent: {h_daily_statement['parent_discovery']}")
    print(f"   Core Principle: {h_daily_statement['core_mechanism']['primary_principle']}")
    
    print("\n📊 Scaled Mathematical Foundation:")
    math_foundation = h_daily_statement['core_mechanism']['scaled_mathematical_foundation']
    for key, value in math_foundation.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\n🎯 Daily Testable Predictions:")
    for i, prediction in enumerate(h_daily_statement['daily_testable_predictions'], 1):
        print(f"   PD1.{i}: {prediction['statement']}")
        print(f"           Success: {prediction['success_criteria']}")
    
    print("\n🔍 Reaction Signature Classification:")
    signatures = h_daily_statement['reaction_signature_classification']['signature_types']
    for sig_name, sig_data in list(signatures.items())[:2]:  # Show first 2
        print(f"   {sig_name.replace('_', ' ')}: {sig_data['prediction_target']} prediction")
        print(f"      Duration: {sig_data['duration']}")
    
    print("\n💡 Directional Bias Framework:")
    bias_rules = h_daily_statement['directional_bias_theory']['directional_bias_rules']
    print(f"   High Prediction: {len(bias_rules['High_Prediction_Indicators'])} indicators")
    print(f"   Low Prediction: {len(bias_rules['Low_Prediction_Indicators'])} indicators")
    
    print("\n🔬 Next Implementation Steps:")
    print("   1. Implement daily reaction signature detector")
    print("   2. Build previous day 40% level calculation system")
    print("   3. Create reaction duration and velocity measurement tools")
    print("   4. Develop directional bias classification engine") 
    print("   5. Build daily prediction validation framework")
    
    return h_daily_statement

def demo_daily_reaction_detection():
    """Demonstrate daily reaction signature detection with sample data"""
    print("🎯 DAILY REACTION SIGNATURE DETECTION DEMO")
    print("=" * 60)
    
    # Sample previous day data
    previous_day = {
        'date': '2025-08-05',
        'high': 23250.0,
        'low': 23100.0,
        'range': 150.0
    }
    
    # Calculate 40% level: 23100 + (150 * 0.4) = 23160
    prev_40pct = previous_day['low'] + ((previous_day['high'] - previous_day['low']) * 0.4)
    
    print(f"📊 Previous Day: {previous_day['date']}")
    print(f"   Range: {previous_day['high']:.0f} - {previous_day['low']:.0f} = {previous_day['high'] - previous_day['low']:.0f} pts")
    print(f"   40% Level: {prev_40pct:.2f}")
    
    # Sample current day with reaction signatures
    current_day = {
        'date': '2025-08-06',
        'events': [
            # Bullish Extreme Predictor: Break above with momentum (30-60 min sustained)
            {'price': 23158.0, 'timestamp': '14:30:00'},  # Approach from below
            {'price': 23162.5, 'timestamp': '14:35:00'},  # Break above 40% level
            {'price': 23168.0, 'timestamp': '14:45:00'},  # Sustained above
            {'price': 23175.0, 'timestamp': '14:55:00'},  # Continued momentum
            {'price': 23180.0, 'timestamp': '15:00:00'},  # Strong departure
            
            # Reversal Touch Signature: Touch and immediate rejection (15-30 min)
            {'price': 23165.0, 'timestamp': '15:30:00'},  # Approach from above  
            {'price': 23161.5, 'timestamp': '15:35:00'},  # Precise touch of 40% level
            {'price': 23155.0, 'timestamp': '15:42:00'},  # Sharp rejection downward
            {'price': 23148.0, 'timestamp': '15:50:00'},  # Continued departure down
        ]
    }
    
    print(f"\n📈 Current Day: {current_day['date']} - {len(current_day['events'])} events")
    
    # Debug: Show interaction detection
    print(f"\n🔍 Detecting interactions within 3.0 points of {prev_40pct:.2f}:")
    for event in current_day['events']:
        distance = abs(event['price'] - prev_40pct)
        print(f"   {event['timestamp']}: {event['price']:.2f} -> {distance:.2f} pts {'✓' if distance <= 3.0 else '✗'}")
    
    # Detect reaction signatures
    signatures = detect_daily_reaction_signatures(previous_day, current_day)
    
    print(f"\n🔍 Detection Results: {len(signatures)} signatures detected")
    
    for i, signature in enumerate(signatures, 1):
        print(f"\n   Signature {i}:")
        print(f"      Type: {signature.reaction_type}")
        print(f"      Time: {signature.reaction_timestamp}")
        print(f"      Price: {signature.reaction_price:.2f} (vs 40% level {signature.previous_day_40pct_level:.2f})")
        print(f"      Duration: {signature.reaction_duration_minutes} minutes")
        print(f"      Velocities: Approach {signature.approach_velocity:.2f}, Departure {signature.departure_velocity:.2f} pts/min")
        print(f"      Prediction: Next day {signature.predicted_extreme} (confidence: {signature.prediction_confidence:.1%})")
        
        # Classify according to our framework
        if signature.reaction_type in ["break_above", "break_below"]:
            framework_type = "Bullish Extreme Predictor" if signature.predicted_extreme == "high" else "Bearish Extreme Predictor"
        elif "touch_and_reject" in signature.reaction_type:
            framework_type = "Reversal Touch Signature"
        else:
            framework_type = "Approach Signature"
            
        print(f"      Framework: {framework_type}")
    
    # Validate against our scaling expectations
    print(f"\n📏 Validation Against H-Daily1 Scaling:")
    if signatures:
        avg_confidence = sum(s.prediction_confidence for s in signatures) / len(signatures)
        print(f"   Average Confidence: {avg_confidence:.1%} (target: >65%)")
        
        duration_range = [s.reaction_duration_minutes for s in signatures]
        print(f"   Duration Range: {min(duration_range)}-{max(duration_range)} min (expected: 15-60 min)")
        
        high_confidence_sigs = [s for s in signatures if s.prediction_confidence > 0.75]
        print(f"   High Confidence Signatures: {len(high_confidence_sigs)}/{len(signatures)} ({len(high_confidence_sigs)/len(signatures):.1%})")
    
    return signatures

if __name__ == "__main__":
    print("🧠 H-DAILY1 HYPOTHESIS: Inter-Day Dimensional Architecture Theory")
    print("=" * 70)
    
    # Run theoretical validation
    validate_daily_hypothesis_framework()
    
    print("\n" + "=" * 70)
    
    # Run detection demo
    demo_signatures = demo_daily_reaction_detection()