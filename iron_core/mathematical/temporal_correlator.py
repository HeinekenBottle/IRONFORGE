"""
Temporal Correlator - Extracted from cascade classifier for modular integration
Handles prediction-validation correlation and sequence analysis
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class CorrelationResult:
    """Result from temporal correlation analysis"""
    prediction_time: str
    closest_event_time: str | None
    time_error_minutes: float
    correlation_strength: float
    validation_status: str
    contextual_matches: list[dict[str, Any]]

class TemporalCorrelationEngine:
    """Engine for correlating predictions with validation data across sequences"""
    
    def __init__(self):
        self.correlation_threshold = 0.7
        self.temporal_window_minutes = 15
        self.correlation_history = []
        
    def correlate_prediction_validation(self, prediction_time: str, actual_events: list[Any], 
                                      prediction_context: dict | None = None) -> CorrelationResult:
        """
        Correlate a prediction time with actual validation events
        
        Args:
            prediction_time: Time when prediction was made (HH:MM:SS format)
            actual_events: List of actual cascade events that occurred
            prediction_context: Optional context about the prediction
            
        Returns:
            CorrelationResult with correlation analysis
        """
        if not actual_events:
            return CorrelationResult(
                prediction_time=prediction_time,
                closest_event_time=None,
                time_error_minutes=float('inf'),
                correlation_strength=0.0,
                validation_status="no_events",
                contextual_matches=[]
            )
        
        # Find closest event in time
        closest_event = None
        min_time_diff = float('inf')
        
        for event in actual_events:
            event_time = getattr(event, 'timestamp', str(event))
            time_diff = self._calculate_time_difference(prediction_time, event_time)
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_event = event
        
        # Calculate correlation strength
        correlation_strength = self._calculate_correlation_strength(
            min_time_diff, closest_event, prediction_context
        )
        
        # Determine validation status
        validation_status = self._determine_validation_status(
            min_time_diff, correlation_strength
        )
        
        # Find contextual matches
        contextual_matches = self._find_contextual_matches(
            prediction_context, actual_events
        )
        
        result = CorrelationResult(
            prediction_time=prediction_time,
            closest_event_time=getattr(closest_event, 'timestamp', str(closest_event)) if closest_event else None,
            time_error_minutes=min_time_diff,
            correlation_strength=correlation_strength,
            validation_status=validation_status,
            contextual_matches=contextual_matches
        )
        
        # Store in history
        self.correlation_history.append(result)
        
        return result
    
    def _calculate_time_difference(self, time1: str, time2: str) -> float:
        """Calculate time difference in minutes between two timestamps"""
        try:
            # Handle HH:MM or HH:MM:SS format
            if len(time1.split(':')) == 2:
                time1 += ':00'
            if len(time2.split(':')) == 2:
                time2 += ':00'
                
            dt1 = datetime.strptime(time1, '%H:%M:%S')
            dt2 = datetime.strptime(time2, '%H:%M:%S')
            
            diff = abs((dt2 - dt1).total_seconds() / 60)
            
            # Handle day rollover
            if diff > 12 * 60:  # More than 12 hours suggests day rollover
                diff = 24 * 60 - diff
                
            return diff
        except:
            return float('inf')
    
    def _calculate_correlation_strength(self, time_diff: float, event: Any, 
                                      prediction_context: dict | None) -> float:
        """Calculate correlation strength based on temporal and contextual factors"""
        if time_diff == float('inf'):
            return 0.0
        
        # Temporal correlation (exponential decay)
        temporal_strength = np.exp(-time_diff / self.temporal_window_minutes)
        
        # Contextual correlation
        contextual_strength = 0.5  # Base contextual strength
        
        if prediction_context and hasattr(event, 'event_context'):
            # Simple keyword matching for context
            prediction_keywords = set(prediction_context.get('keywords', []))
            event_keywords = set(str(event.event_context).lower().split())
            
            if prediction_keywords and event_keywords:
                overlap = len(prediction_keywords.intersection(event_keywords))
                contextual_strength = min(1.0, overlap / len(prediction_keywords))
        
        # Combined correlation strength
        return (temporal_strength * 0.7 + contextual_strength * 0.3)
    
    def _determine_validation_status(self, time_diff: float, correlation_strength: float) -> str:
        """Determine validation status based on correlation metrics"""
        if time_diff == float('inf'):
            return "no_correlation"
        
        if correlation_strength >= self.correlation_threshold:
            if time_diff <= 5:
                return "strong_validation"
            elif time_diff <= 15:
                return "good_validation"
            else:
                return "weak_validation"
        else:
            if time_diff <= 30:
                return "temporal_match_only"
            else:
                return "poor_correlation"
    
    def _find_contextual_matches(self, prediction_context: dict | None, 
                               actual_events: list[Any]) -> list[dict[str, Any]]:
        """Find events that match prediction context"""
        if not prediction_context:
            return []
        
        matches = []
        prediction_keywords = set(prediction_context.get('keywords', []))
        
        for event in actual_events:
            event_context = getattr(event, 'event_context', '')
            event_keywords = set(str(event_context).lower().split())
            
            overlap = prediction_keywords.intersection(event_keywords)
            if overlap:
                matches.append({
                    'event': event,
                    'matching_keywords': list(overlap),
                    'match_score': len(overlap) / len(prediction_keywords) if prediction_keywords else 0
                })
        
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)
    
    def get_correlation_statistics(self) -> dict[str, Any]:
        """Get statistics from correlation history"""
        if not self.correlation_history:
            return {"total_correlations": 0}
        
        correlations = self.correlation_history
        
        return {
            "total_correlations": len(correlations),
            "average_correlation_strength": np.mean([c.correlation_strength for c in correlations]),
            "average_time_error": np.mean([c.time_error_minutes for c in correlations]),
            "validation_status_distribution": self._get_status_distribution(correlations),
            "strong_validations": len([c for c in correlations if c.validation_status == "strong_validation"]),
            "correlation_success_rate": len([c for c in correlations if c.correlation_strength >= self.correlation_threshold]) / len(correlations)
        }
    
    def _get_status_distribution(self, correlations: list[CorrelationResult]) -> dict[str, int]:
        """Get distribution of validation statuses"""
        distribution = {}
        for correlation in correlations:
            status = correlation.validation_status
            distribution[status] = distribution.get(status, 0) + 1
        return distribution

class SequencePatternAnalyzer:
    """Analyzer for detecting patterns in cascade sequences"""
    
    def __init__(self):
        self.known_patterns = {
            'escalation': ['primer', 'standard', 'major'],
            'deescalation': ['major', 'standard', 'primer'],
            'double_tap': ['standard', 'standard'],
            'triple_cascade': ['primer', 'standard', 'major'],
            'reversal': ['major', 'primer']
        }
        
    def analyze_sequence_pattern(self, sequence_events: list[Any]) -> dict[str, Any]:
        """Analyze a sequence for known patterns"""
        if len(sequence_events) < 2:
            return {"pattern": "insufficient_data", "confidence": 0.0}
        
        # Extract cascade types
        cascade_types = []
        for event in sequence_events:
            if hasattr(event, 'cascade_type'):
                cascade_types.append(event.cascade_type.value.lower())
            else:
                cascade_types.append('unknown')
        
        # Check against known patterns
        best_match = None
        best_confidence = 0.0
        
        for pattern_name, pattern_sequence in self.known_patterns.items():
            confidence = self._calculate_pattern_match(cascade_types, pattern_sequence)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = pattern_name
        
        return {
            "pattern": best_match or "unknown",
            "confidence": best_confidence,
            "sequence_types": cascade_types,
            "sequence_length": len(cascade_types)
        }
    
    def _calculate_pattern_match(self, actual_sequence: list[str], 
                               pattern_sequence: list[str]) -> float:
        """Calculate how well an actual sequence matches a known pattern"""
        if len(actual_sequence) != len(pattern_sequence):
            # Partial matching for different lengths
            min_len = min(len(actual_sequence), len(pattern_sequence))
            actual_partial = actual_sequence[:min_len]
            pattern_partial = pattern_sequence[:min_len]
            
            matches = sum(1 for a, p in zip(actual_partial, pattern_partial, strict=False) if a == p)
            return (matches / min_len) * 0.8  # Penalty for length mismatch
        else:
            # Exact length matching
            matches = sum(1 for a, p in zip(actual_sequence, pattern_sequence, strict=False) if a == p)
            return matches / len(pattern_sequence)
    
    def detect_emerging_patterns(self, all_sequences: list[list[Any]]) -> dict[str, Any]:
        """Detect emerging patterns across multiple sequences"""
        if not all_sequences:
            return {"emerging_patterns": []}
        
        # Extract all sequence patterns
        sequence_patterns = []
        for sequence in all_sequences:
            pattern_analysis = self.analyze_sequence_pattern(sequence)
            if pattern_analysis["confidence"] > 0.5:
                sequence_patterns.append(pattern_analysis["sequence_types"])
        
        # Find common subsequences
        pattern_frequency = {}
        for pattern in sequence_patterns:
            pattern_str = " -> ".join(pattern)
            pattern_frequency[pattern_str] = pattern_frequency.get(pattern_str, 0) + 1
        
        # Identify emerging patterns (frequency > 1, not in known patterns)
        emerging = []
        for pattern_str, frequency in pattern_frequency.items():
            if frequency > 1:
                pattern_list = pattern_str.split(" -> ")
                if not any(pattern_list == list(known) for known in self.known_patterns.values()):
                    emerging.append({
                        "pattern": pattern_list,
                        "frequency": frequency,
                        "pattern_string": pattern_str
                    })
        
        return {
            "emerging_patterns": sorted(emerging, key=lambda x: x["frequency"], reverse=True),
            "total_sequences_analyzed": len(all_sequences),
            "pattern_coverage": len(sequence_patterns) / len(all_sequences) if all_sequences else 0
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    from dataclasses import dataclass
    from enum import Enum
    
    class CascadeType(Enum):
        PRIMER = "primer"
        STANDARD = "standard"
        MAJOR = "major"
    
    @dataclass
    class MockEvent:
        timestamp: str
        cascade_type: CascadeType
        event_context: str
    
    # Test correlation engine
    correlation_engine = TemporalCorrelationEngine()
    
    sample_events = [
        MockEvent("10:15:30", CascadeType.PRIMER, "initial setup"),
        MockEvent("10:18:45", CascadeType.STANDARD, "main cascade"),
        MockEvent("10:22:15", CascadeType.MAJOR, "major breakout")
    ]
    
    # Test prediction correlation
    prediction_time = "10:17:00"
    prediction_context = {"keywords": ["cascade", "breakout"]}
    
    result = correlation_engine.correlate_prediction_validation(
        prediction_time, sample_events, prediction_context
    )
    
    print("Temporal Correlation Test Results:")
    print(f"Prediction Time: {result.prediction_time}")
    print(f"Closest Event: {result.closest_event_time}")
    print(f"Time Error: ±{result.time_error_minutes:.1f} minutes")
    print(f"Correlation Strength: {result.correlation_strength:.2f}")
    print(f"Validation Status: {result.validation_status}")
    
    # Test pattern analyzer
    pattern_analyzer = SequencePatternAnalyzer()
    pattern_result = pattern_analyzer.analyze_sequence_pattern(sample_events)
    
    print("\nSequence Pattern Analysis:")
    print(f"Pattern: {pattern_result['pattern']}")
    print(f"Confidence: {pattern_result['confidence']:.2f}")
    print(f"Sequence: {' -> '.join(pattern_result['sequence_types'])}")


# HTF Master Controller Implementation

@dataclass  
class HTFEvent:
    """Higher Timeframe event"""
    event_type: str
    timestamp: str
    magnitude: float
    session_context: str
    influence_strength: float

@dataclass
class HTFIntensity:
    """HTF intensity calculation result"""
    current_intensity: float
    baseline: float
    excitation_sum: float
    decay_factor: float
    activation_threshold: float
    activated: bool

class HTFMasterController:
    """
    Higher Timeframe Master Controller
    
    Implements fractal cascade architecture where HTF events serve as master
    controllers for session-level prediction activation.
    
    Mathematical Foundation: λ_HTF(t) = μ_h + Σ α_h · exp(-β_h (t - t_j)) · magnitude
    """
    
    def __init__(self, activation_threshold: float = 0.5):
        # HTF Hawkes parameters (validated)
        self.mu_h = 0.02      # Baseline intensity
        self.alpha_h = 35.51  # Excitation strength  
        self.beta_h = 0.00442 # Decay rate (16.7-hour half-life)
        
        self.activation_threshold = activation_threshold
        self.htf_events: list[HTFEvent] = []
        self.intensity_history: list[HTFIntensity] = []
        
    def add_htf_event(self, event_type: str, timestamp: str, magnitude: float = 1.0, 
                      session_context: str = "unknown") -> None:
        """Add HTF event to the system"""
        
        # Calculate influence strength based on event type
        event_multipliers = {
            'session_high': 2.3,
            'session_low': 2.2,
            'friday_close': 2.0,
            'monday_open': 1.8,
            'liquidity_sweep': 1.5
        }
        
        influence = event_multipliers.get(event_type, 1.0) * magnitude
        
        event = HTFEvent(
            event_type=event_type,
            timestamp=timestamp,
            magnitude=magnitude,
            session_context=session_context,
            influence_strength=influence
        )
        
        self.htf_events.append(event)
        
    def calculate_htf_intensity(self, current_time: str) -> HTFIntensity:
        """
        Calculate current HTF intensity using Hawkes process
        
        Formula: λ_HTF(t) = μ_h + Σ α_h · exp(-β_h (t - t_j)) · magnitude
        """
        
        # Convert time to minutes for calculation
        current_minutes = self._time_to_minutes(current_time)
        
        # Calculate excitation sum from all HTF events
        excitation_sum = 0.0
        
        for event in self.htf_events:
            event_minutes = self._time_to_minutes(event.timestamp)
            time_diff = current_minutes - event_minutes
            
            if time_diff > 0:  # Only consider past events
                excitation = self.alpha_h * np.exp(-self.beta_h * time_diff) * event.influence_strength
                excitation_sum += excitation
        
        # Calculate total intensity
        current_intensity = self.mu_h + excitation_sum
        
        # Check activation
        activated = current_intensity > self.activation_threshold
        
        intensity = HTFIntensity(
            current_intensity=current_intensity,
            baseline=self.mu_h,
            excitation_sum=excitation_sum,
            decay_factor=self.beta_h,
            activation_threshold=self.activation_threshold,
            activated=activated
        )
        
        self.intensity_history.append(intensity)
        return intensity
        
    def generate_activation_signal(self, htf_intensity: HTFIntensity) -> dict[str, Any]:
        """Generate activation signal for subordinate session processors"""
        
        if not htf_intensity.activated:
            return {"activated": False}
            
        # Calculate enhancement factors
        intensity_ratio = htf_intensity.current_intensity / self.activation_threshold
        
        return {
            "activated": True,
            "baseline_boost": intensity_ratio,
            "decay_gamma": 0.143,  # Calibrated gamma for session processes
            "confidence_boost": min(1.5, intensity_ratio ** 0.5),
            "htf_intensity": htf_intensity.current_intensity,
            "excitation_strength": htf_intensity.excitation_sum
        }
        
    def _time_to_minutes(self, time_str: str) -> float:
        """Convert HH:MM:SS time to minutes"""
        try:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2]) if len(parts[2:]) else 0
            return hours * 60 + minutes + seconds / 60.0
        except (ValueError, IndexError):
            return 0.0
