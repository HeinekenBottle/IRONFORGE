"""
IRONFORGE Pattern Graduation Pipeline
Validates discovered patterns and bridges to production
"""
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class PatternGraduation:
    """
    Tests discovered patterns against 87% baseline
    Promotes proven patterns to production
    """
    
    def __init__(self, baseline_accuracy: float = 0.87):
        self.baseline_accuracy = baseline_accuracy
        self.validated_patterns = []
        
    def validate_pattern(self, pattern: Dict, historical_data: List[Dict]) -> Dict:
        """
        Backtest pattern against historical sessions
        
        Returns:
            Validation results with improvement metrics
        """
        pattern_hits = 0
        pattern_accuracy = []
        
        for session in historical_data:
            if self._pattern_matches(pattern, session):
                pattern_hits += 1
                accuracy = self._measure_accuracy(pattern, session)
                pattern_accuracy.append(accuracy)
        
        if len(pattern_accuracy) == 0:
            return {
                'pattern': pattern,
                'status': 'NO_MATCHES',
                'improvement': 0.0
            }
        
        avg_accuracy = np.mean(pattern_accuracy)
        improvement = avg_accuracy - self.baseline_accuracy
        
        result = {
            'pattern': pattern,
            'appearances': pattern_hits,
            'accuracy': avg_accuracy,
            'improvement': improvement,
            'confidence': np.std(pattern_accuracy) if len(pattern_accuracy) > 1 else 0,
            'status': 'VALIDATED' if improvement > 0.02 else 'INSUFFICIENT'
        }
        
        if result['status'] == 'VALIDATED':
            self.validated_patterns.append(result)
            
        return result
    
    def _pattern_matches(self, pattern: Dict, session: Dict) -> bool:
        """Check if pattern appears in session"""
        pattern_type = pattern.get('type')
        
        if pattern_type == 'high_energy_cluster':
            energy = session.get('metadata', {}).get('energy_state', {})
            return any(v > 0.8 for v in energy.values() if isinstance(v, (int, float)))
            
        elif pattern_type == 'potential_cascade':
            movements = session.get('nodes', [])
            if len(movements) < 4:
                return False
            for i in range(len(movements) - 3):
                types = [m.get('type') for m in movements[i:i+4]]
                if len(set(types)) == 1:
                    return True
                    
        return False
    
    def _measure_accuracy(self, pattern: Dict, session: Dict) -> float:
        """Measure pattern's predictive accuracy"""
        # Placeholder: actual implementation would measure real prediction accuracy
        # For now, return random accuracy for testing
        return 0.85 + np.random.random() * 0.1
    
    def convert_to_simple_feature(self, pattern: Dict) -> Dict:
        """Convert validated pattern to simple feature for production"""
        return {
            'name': f"pattern_{pattern['type']}",
            'type': 'discovered',
            'extractor': self._create_extractor(pattern),
            'improvement': pattern.get('improvement', 0),
            'confidence': pattern.get('confidence', 0)
        }
    
    def _create_extractor(self, pattern: Dict):
        """Create feature extraction function for pattern"""
        def extractor(session_data):
            # Simple counting based on pattern type
            if pattern['type'] == 'high_energy_cluster':
                return 1.0 if self._pattern_matches(pattern, session_data) else 0.0
            return 0.0
        return extractor
    
    def save_validated_patterns(self, path: str):
        """Save validated patterns for production use"""
        output_path = os.path.join(path, 'validated_patterns.json')
        with open(output_path, 'w') as f:
            json.dump(self.validated_patterns, f, indent=2)
        print(f"Saved {len(self.validated_patterns)} validated patterns")
