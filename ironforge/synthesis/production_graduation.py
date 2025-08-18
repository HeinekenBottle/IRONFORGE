"""
Production Graduation System
Bridge validated patterns to production features
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from config import get_config
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import get_config

logger = logging.getLogger(__name__)

class ProductionGraduation:
    """
    Production feature export for graduated patterns
    Converts validated archaeological discoveries into production-ready features
    
    NOTE: This class maintains NO STATE between sessions to ensure complete session independence.
    Each export is completely isolated and cannot be contaminated by previous sessions.
    """
    
    def __init__(self, output_path: Path | None = None):
        if output_path is None:
            config = get_config()
            preservation_path = config.get_preservation_path()
            self.output_path = Path(preservation_path) / "production_features.json"
        else:
            self.output_path = output_path
        # REMOVED: self.production_features = [] to ensure session independence
        logger.info(f"Production Graduation initialized, output: {self.output_path}")
    
    def export_graduated_patterns(self, graduation_results: dict[str, Any]) -> dict[str, Any]:
        """
        Export graduated patterns as production features
        
        Args:
            graduation_results: Results from PatternGraduation
            
        Returns:
            Production export results
        """
        try:
            session_name = graduation_results.get('session_name', 'unknown')
            production_ready = graduation_results.get('production_ready', False)
            
            if not production_ready:
                logger.warning(f"Session {session_name} not production ready, skipping export")
                return {
                    'status': 'SKIPPED',
                    'reason': 'Not production ready',
                    'session_name': session_name
                }
            
            # Convert to production features
            production_features = self._convert_to_production_features(graduation_results)
            
            # Validate feature format
            validation_result = self._validate_production_features(production_features)
            
            if not validation_result['valid']:
                logger.error(f"Production feature validation failed: {validation_result['errors']}")
                return {
                    'status': 'VALIDATION_FAILED',
                    'errors': validation_result['errors'],
                    'session_name': session_name
                }
            
            # REMOVED: production_features.append() to ensure session independence
            # Each session export is completely isolated
            
            # Export to file
            export_result = self._export_to_file(production_features)
            
            logger.info(f"Production export successful for {session_name}")
            return {
                'status': 'SUCCESS',
                'session_name': session_name,
                'feature_count': len(production_features.get('features', [])),
                'export_path': str(self.output_path),
                'export_result': export_result
            }
            
        except Exception as e:
            logger.error(f"Production export failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'session_name': graduation_results.get('session_name', 'unknown')
            }
    
    def _convert_to_production_features(self, graduation_results: dict[str, Any]) -> dict[str, Any]:
        """Convert graduation results to production feature format"""
        
        session_name = graduation_results.get('session_name', 'unknown')
        graduation_score = graduation_results.get('graduation_score', 0.0)
        validation_metrics = graduation_results.get('validation_metrics', {})
        graduated_patterns = graduation_results.get('graduated_patterns', {})
        
        # Create production feature structure
        production_features = {
            'session_name': session_name,
            'graduation_timestamp': datetime.now().isoformat(),
            'graduation_score': graduation_score,
            'feature_version': '1.0.0',
            'feature_type': 'archaeological_patterns',
            'features': [],
            'metadata': {
                'validation_metrics': validation_metrics,
                'baseline_threshold': graduation_results.get('baseline_threshold', 0.87),
                'production_ready': True
            }
        }
        
        # Convert each graduated pattern to production feature
        for metric_name, metric_value in graduated_patterns.items():
            if isinstance(metric_value, (int, float)) and metric_value > 0:
                feature = {
                    'feature_id': f"{session_name}_{metric_name}_{int(datetime.now().timestamp())}",
                    'feature_name': metric_name,
                    'feature_value': float(metric_value),
                    'feature_confidence': min(1.0, metric_value / 0.87),  # Normalize by baseline
                    'archaeological_significance': metric_value,
                    'session_context': session_name,
                    'pattern_type': self._classify_pattern_type(metric_name),
                    'production_metadata': {
                        'graduated_at': datetime.now().isoformat(),
                        'graduation_score': graduation_score,
                        'baseline_exceeded': metric_value >= 0.87
                    }
                }
                production_features['features'].append(feature)
        
        return production_features
    
    def _classify_pattern_type(self, metric_name: str) -> str:
        """Classify pattern type from metric name"""
        
        classification_map = {
            'pattern_confidence': 'confidence_based',
            'significance_score': 'archaeological_significance',
            'attention_coherence': 'attention_pattern',
            'pattern_consistency': 'consistency_based',
            'archaeological_value': 'archaeological_discovery'
        }
        
        return classification_map.get(metric_name, 'unknown_pattern')
    
    def _validate_production_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Validate production feature format and content"""
        
        errors = []
        warnings = []
        
        # Required fields validation
        required_fields = ['session_name', 'graduation_timestamp', 'graduation_score', 'features']
        for field in required_fields:
            if field not in features:
                errors.append(f"Missing required field: {field}")
        
        # Feature validation
        feature_list = features.get('features', [])
        if not feature_list:
            warnings.append("No features exported")
        
        for i, feature in enumerate(feature_list):
            feature_errors = self._validate_single_feature(feature, i)
            errors.extend(feature_errors)
        
        # Graduation score validation
        graduation_score = features.get('graduation_score', 0.0)
        if graduation_score < 0.87:
            errors.append(f"Graduation score {graduation_score} below 87% threshold")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_single_feature(self, feature: dict[str, Any], index: int) -> list[str]:
        """Validate individual production feature"""
        
        errors = []
        required_feature_fields = [
            'feature_id', 'feature_name', 'feature_value', 
            'feature_confidence', 'archaeological_significance'
        ]
        
        for field in required_feature_fields:
            if field not in feature:
                errors.append(f"Feature {index}: Missing required field {field}")
        
        # Value validation
        feature_value = feature.get('feature_value', 0.0)
        if not isinstance(feature_value, (int, float)):
            errors.append(f"Feature {index}: feature_value must be numeric")
        elif feature_value <= 0:
            errors.append(f"Feature {index}: feature_value must be positive")
        
        # Confidence validation
        confidence = feature.get('feature_confidence', 0.0)
        if not isinstance(confidence, (int, float)):
            errors.append(f"Feature {index}: feature_confidence must be numeric")
        elif not (0.0 <= confidence <= 1.0):
            errors.append(f"Feature {index}: feature_confidence must be between 0 and 1")
        
        return errors
    
    def _export_to_file(self, production_features: dict[str, Any]) -> dict[str, Any]:
        """Export production features to file"""
        
        try:
            # Ensure output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing features if file exists
            existing_features = []
            if self.output_path.exists():
                try:
                    with open(self.output_path) as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            existing_features = existing_data
                        elif isinstance(existing_data, dict) and 'features' in existing_data:
                            existing_features = [existing_data]
                except Exception as e:
                    logger.warning(f"Could not load existing features: {e}")
            
            # Append new features
            existing_features.append(production_features)
            
            # Write back to file
            with open(self.output_path, 'w') as f:
                json.dump(existing_features, f, indent=2, default=str)
            
            return {
                'success': True,
                'file_path': str(self.output_path),
                'total_sessions': len(existing_features)
            }
            
        except Exception as e:
            logger.error(f"Failed to export to file: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_production_summary(self) -> dict[str, Any]:
        """Get summary of production system configuration
        
        NOTE: No historical data is maintained to ensure session independence.
        Use external logging/storage if cross-session analytics are needed.
        """
        
        return {
            'output_path': str(self.output_path),
            'session_independence': True,
            'note': 'No cross-session state maintained - each export is isolated'
        }