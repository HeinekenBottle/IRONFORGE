"""
Cascade Classification and Sequential Analysis System
Advanced system to detect, classify, and correlate cascade types across temporal sequences
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class CascadeType(Enum):
    PRIMER = "primer_cascade"
    STANDARD = "standard_cascade" 
    MAJOR = "major_cascade"
    MICRO = "micro"  # Keep for compatibility
    UNCLASSIFIED = "unclassified"

@dataclass
class CascadeEvent:
    """Individual cascade event with full context"""
    timestamp: str
    price_level: float
    cascade_type: CascadeType
    magnitude: float
    event_context: str
    prediction_time: Optional[str] = None
    validation_status: Optional[str] = None
    sequence_id: Optional[str] = None

@dataclass
class CascadeSequence:
    """Sequence of related cascade events"""
    sequence_id: str
    events: List[CascadeEvent]
    start_time: str
    end_time: str
    total_magnitude: float
    sequence_type: str
    correlation_score: float = 0.0
    
class CascadeClassifier:
    """
    Advanced cascade classification system with sequential analysis
    
    Classifies individual cascade events and analyzes temporal sequences
    for pattern recognition and correlation analysis.
    """
    
    def __init__(self):
        self.classification_thresholds = {
            CascadeType.MICRO: (0.0, 0.15),
            CascadeType.PRIMER: (0.15, 0.35),
            CascadeType.STANDARD: (0.35, 0.65),
            CascadeType.MAJOR: (0.65, 1.0)
        }
        
        self.temporal_correlation_window = 300  # 5 minutes in seconds
        self.sequence_gap_threshold = 600  # 10 minutes max gap between events
        
    def classify_cascade(self, magnitude: float, context: Dict[str, Any] = None) -> CascadeType:
        """
        Classify a cascade event based on magnitude and context
        
        Args:
            magnitude: Cascade magnitude (0.0 to 1.0)
            context: Additional context for classification
            
        Returns:
            CascadeType: Classified cascade type
        """
        if not (0.0 <= magnitude <= 1.0):
            return CascadeType.UNCLASSIFIED
            
        for cascade_type, (min_thresh, max_thresh) in self.classification_thresholds.items():
            if min_thresh <= magnitude < max_thresh:
                return cascade_type
                
        return CascadeType.UNCLASSIFIED
    
    def create_cascade_event(self, timestamp: str, price_level: float, 
                           magnitude: float, event_context: str = "") -> CascadeEvent:
        """Create a classified cascade event"""
        cascade_type = self.classify_cascade(magnitude)
        
        return CascadeEvent(
            timestamp=timestamp,
            price_level=price_level,
            cascade_type=cascade_type,
            magnitude=magnitude,
            event_context=event_context,
            sequence_id=None  # Will be assigned during sequence analysis
        )
    
    def analyze_temporal_sequences(self, events: List[CascadeEvent]) -> List[CascadeSequence]:
        """
        Analyze temporal sequences of cascade events
        
        Groups events into sequences based on temporal proximity and
        calculates correlation scores.
        """
        if not events:
            return []
            
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: self._parse_timestamp(e.timestamp))
        
        sequences = []
        current_sequence = []
        sequence_counter = 0
        
        for event in sorted_events:
            if not current_sequence:
                current_sequence = [event]
            else:
                # Check if event belongs to current sequence
                last_event_time = self._parse_timestamp(current_sequence[-1].timestamp)
                current_event_time = self._parse_timestamp(event.timestamp)
                
                time_gap = (current_event_time - last_event_time).total_seconds()
                
                if time_gap <= self.sequence_gap_threshold:
                    current_sequence.append(event)
                else:
                    # Finalize current sequence and start new one
                    if len(current_sequence) >= 2:  # Only sequences with 2+ events
                        sequence = self._create_sequence(current_sequence, sequence_counter)
                        sequences.append(sequence)
                        sequence_counter += 1
                    
                    current_sequence = [event]
        
        # Handle final sequence
        if len(current_sequence) >= 2:
            sequence = self._create_sequence(current_sequence, sequence_counter)
            sequences.append(sequence)
            
        return sequences
    
    def _create_sequence(self, events: List[CascadeEvent], sequence_id: int) -> CascadeSequence:
        """Create a CascadeSequence from a list of events"""
        sequence_id_str = f"seq_{sequence_id:03d}"
        
        # Assign sequence ID to all events
        for event in events:
            event.sequence_id = sequence_id_str
            
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        total_magnitude = sum(event.magnitude for event in events)
        
        # Determine sequence type based on cascade types
        cascade_types = [event.cascade_type for event in events]
        sequence_type = self._determine_sequence_type(cascade_types)
        
        # Calculate correlation score
        correlation_score = self._calculate_correlation_score(events)
        
        return CascadeSequence(
            sequence_id=sequence_id_str,
            events=events,
            start_time=start_time,
            end_time=end_time,
            total_magnitude=total_magnitude,
            sequence_type=sequence_type,
            correlation_score=correlation_score
        )
    
    def _determine_sequence_type(self, cascade_types: List[CascadeType]) -> str:
        """Determine sequence type based on constituent cascade types"""
        type_counts = {}
        for cascade_type in cascade_types:
            type_counts[cascade_type] = type_counts.get(cascade_type, 0) + 1
            
        # Determine dominant type
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        if len(set(cascade_types)) == 1:
            return f"homogeneous_{dominant_type.value}"
        else:
            return f"mixed_{dominant_type.value}_dominant"
    
    def _calculate_correlation_score(self, events: List[CascadeEvent]) -> float:
        """Calculate correlation score for a sequence of events"""
        if len(events) < 2:
            return 0.0
            
        # Simple correlation based on magnitude progression
        magnitudes = [event.magnitude for event in events]
        
        # Calculate trend consistency
        differences = [magnitudes[i+1] - magnitudes[i] for i in range(len(magnitudes)-1)]
        
        if not differences:
            return 0.0
            
        # Correlation score based on trend consistency
        positive_trends = sum(1 for d in differences if d > 0)
        negative_trends = sum(1 for d in differences if d < 0)
        
        trend_consistency = abs(positive_trends - negative_trends) / len(differences)
        
        # Normalize to 0-1 range
        return min(trend_consistency, 1.0)
    
    def _parse_timestamp(self, timestamp: str) -> datetime:
        """Parse timestamp string to datetime object"""
        try:
            # Try multiple timestamp formats
            formats = [
                "%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f"
            ]
            
            for fmt in formats:
                try:
                    if len(timestamp.split()) == 1 and ":" in timestamp:
                        # Time only, assume today's date
                        today = datetime.now().date()
                        time_obj = datetime.strptime(timestamp, "%H:%M:%S").time()
                        return datetime.combine(today, time_obj)
                    else:
                        return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
                    
            # Fallback: assume it's a simple time format
            return datetime.strptime(timestamp, "%H:%M:%S")
            
        except Exception as e:
            print(f"Warning: Could not parse timestamp '{timestamp}': {e}")
            return datetime.now()
    
    def get_classification_summary(self, events: List[CascadeEvent]) -> Dict[str, Any]:
        """Get summary statistics for classified events"""
        if not events:
            return {"total_events": 0}
            
        type_counts = {}
        magnitude_stats = []
        
        for event in events:
            type_counts[event.cascade_type] = type_counts.get(event.cascade_type, 0) + 1
            magnitude_stats.append(event.magnitude)
            
        return {
            "total_events": len(events),
            "type_distribution": {t.value: count for t, count in type_counts.items()},
            "magnitude_stats": {
                "mean": np.mean(magnitude_stats),
                "std": np.std(magnitude_stats),
                "min": np.min(magnitude_stats),
                "max": np.max(magnitude_stats)
            }
        }
    
    def export_analysis(self, sequences: List[CascadeSequence], 
                       filename: str = None) -> Dict[str, Any]:
        """Export sequence analysis results"""
        analysis_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_sequences": len(sequences),
            "sequences": []
        }
        
        for sequence in sequences:
            sequence_data = {
                "sequence_id": sequence.sequence_id,
                "sequence_type": sequence.sequence_type,
                "start_time": sequence.start_time,
                "end_time": sequence.end_time,
                "total_magnitude": sequence.total_magnitude,
                "correlation_score": sequence.correlation_score,
                "event_count": len(sequence.events),
                "events": [
                    {
                        "timestamp": event.timestamp,
                        "price_level": event.price_level,
                        "cascade_type": event.cascade_type.value,
                        "magnitude": event.magnitude,
                        "event_context": event.event_context
                    }
                    for event in sequence.events
                ]
            }
            analysis_data["sequences"].append(sequence_data)
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(analysis_data, f, indent=2)
                
        return analysis_data

# Example usage and testing
if __name__ == "__main__":
    classifier = CascadeClassifier()
    
    # Example cascade events
    sample_events = [
        classifier.create_cascade_event("09:30:00", 18450.0, 0.25, "Initial expansion"),
        classifier.create_cascade_event("09:32:15", 18465.0, 0.45, "Major breakout"),
        classifier.create_cascade_event("09:35:30", 18480.0, 0.35, "Continuation"),
        classifier.create_cascade_event("10:15:00", 18420.0, 0.20, "Separate event"),
    ]
    
    # Analyze sequences
    sequences = classifier.analyze_temporal_sequences(sample_events)
    
    # Print results
    print("Cascade Classification Analysis")
    print("=" * 40)
    
    summary = classifier.get_classification_summary(sample_events)
    print(f"Total Events: {summary['total_events']}")
    print(f"Type Distribution: {summary['type_distribution']}")
    
    print(f"\nSequences Found: {len(sequences)}")
    for seq in sequences:
        print(f"  {seq.sequence_id}: {seq.sequence_type} "
              f"({len(seq.events)} events, correlation: {seq.correlation_score:.2f})")
