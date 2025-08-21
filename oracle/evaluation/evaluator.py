#!/usr/bin/env python3
"""
Oracle Model Evaluator

Computes comprehensive metrics (MAE, RMSE, MAPE) on test data and saves metrics.json.
Provides detailed analysis of Oracle performance across different scenarios.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

from ..core import (
    ORACLE_MODEL_FILES, TGAT_EMBEDDING_DIM, ORACLE_OUTPUT_DIM,
    OracleModelError, EvaluationError, create_model_error,
    handle_oracle_errors
)
from ..models import TrainingManifest, OraclePrediction

logger = logging.getLogger(__name__)


class OracleEvaluator:
    """Evaluate trained Oracle model performance"""
    
    def __init__(self, model_dir: Path):
        """
        Initialize Oracle evaluator
        
        Args:
            model_dir: Directory containing trained model files
        """
        self.model_dir = Path(model_dir)
        self.discovery_engine = None
        self.scaler = None
        self.training_manifest = None
        
        # Load trained model
        self._load_trained_model()
        
    @handle_oracle_errors(OracleModelError, "MODEL_LOADING_ERROR")
    def _load_trained_model(self):
        """Load trained Oracle model and scaler"""
        weights_path = self.model_dir / ORACLE_MODEL_FILES['weights']
        scaler_path = self.model_dir / ORACLE_MODEL_FILES['scaler']
        manifest_path = self.model_dir / ORACLE_MODEL_FILES['manifest']
        
        # Validate required files exist
        missing_files = []
        for file_type, filename in ORACLE_MODEL_FILES.items():
            file_path = self.model_dir / filename
            if not file_path.exists():
                missing_files.append(f"{file_type}: {filename}")
        
        if missing_files:
            raise create_model_error(
                "loading",
                str(self.model_dir),
                f"Missing required files: {', '.join(missing_files)}"
            )
        
        try:
            # Load TGAT discovery engine
            self.discovery_engine = IRONFORGEDiscovery()
            self.discovery_engine.eval()
            
            # Load trained range head weights
            range_head_weights = torch.load(weights_path, map_location='cpu')
            self.discovery_engine.range_head.load_state_dict(range_head_weights)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load training manifest
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
                self.training_manifest = TrainingManifest.from_dict(manifest_data)
            
            logger.info(f"Oracle model loaded from {self.model_dir}")
            
        except Exception as e:
            raise create_model_error(
                "loading",
                str(self.model_dir),
                f"Failed to load model components: {e}"
            )
    
    @handle_oracle_errors(EvaluationError, "PREDICTION_ERROR")
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Make predictions using loaded Oracle model
        
        Args:
            embeddings: TGAT embeddings (N, 44)
            
        Returns:
            Predictions array (N, 2) with [center, half_range]
        """
        if self.discovery_engine is None or self.scaler is None:
            raise EvaluationError("Model not properly loaded")
        
        # Validate embedding dimensions
        if embeddings.shape[1] != TGAT_EMBEDDING_DIM:
            raise EvaluationError(
                f"Invalid embedding dimension: {embeddings.shape[1]} (expected {TGAT_EMBEDDING_DIM})"
            )
        
        # Normalize embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        
        # Convert to tensor
        embeddings_tensor = torch.FloatTensor(embeddings_scaled)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.discovery_engine.range_head(embeddings_tensor)
            
        return predictions.numpy()
    
    @handle_oracle_errors(EvaluationError, "EVALUATION_ERROR")
    def evaluate_test_data(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance on test dataset
        
        Args:
            test_data: Test dataset with embeddings and targets
            
        Returns:
            Dict with comprehensive evaluation metrics
        """
        logger.info(f"Evaluating Oracle model on {len(test_data)} test samples")
        
        # Extract embeddings and targets
        embedding_cols = [f'pooled_{i:02d}' for i in range(TGAT_EMBEDDING_DIM)]
        target_cols = ['target_center', 'target_half_range']
        
        # Validate required columns
        missing_cols = []
        for col in embedding_cols + target_cols:
            if col not in test_data.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise EvaluationError(f"Missing required columns: {missing_cols}")
        
        # Extract data
        embeddings = test_data[embedding_cols].values
        targets = test_data[target_cols].values
        
        # Make predictions
        predictions = self.predict(embeddings)
        
        # Calculate metrics
        metrics = self._calculate_metrics(targets, predictions)
        
        # Add metadata
        metrics['test_samples'] = len(test_data)
        metrics['model_version'] = self.training_manifest.version if self.training_manifest else "unknown"
        metrics['evaluation_timestamp'] = pd.Timestamp.now().isoformat()
        
        # Session-level analysis if session info available
        if 'session_id' in test_data.columns:
            session_metrics = self._calculate_session_metrics(test_data, predictions, targets)
            metrics['session_analysis'] = session_metrics
        
        logger.info(f"Evaluation complete - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        # Split targets and predictions
        target_center, target_half_range = targets[:, 0], targets[:, 1]
        pred_center, pred_half_range = predictions[:, 0], predictions[:, 1]
        
        # Overall metrics
        mae_center = mean_absolute_error(target_center, pred_center)
        mae_half_range = mean_absolute_error(target_half_range, pred_half_range)
        mae_overall = (mae_center + mae_half_range) / 2
        
        rmse_center = np.sqrt(mean_squared_error(target_center, pred_center))
        rmse_half_range = np.sqrt(mean_squared_error(target_half_range, pred_half_range))
        rmse_overall = (rmse_center + rmse_half_range) / 2
        
        # MAPE (Mean Absolute Percentage Error) - handle division by zero
        def safe_mape(y_true, y_pred):
            mask = y_true != 0
            if not mask.any():
                return 0.0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        mape_center = safe_mape(target_center, pred_center)
        mape_half_range = safe_mape(target_half_range, pred_half_range)
        
        # Range-based metrics
        target_range = 2 * target_half_range
        pred_range = 2 * pred_half_range
        mae_range = mean_absolute_error(target_range, pred_range)
        rmse_range = np.sqrt(mean_squared_error(target_range, pred_range))
        mape_range = safe_mape(target_range, pred_range)
        
        # Correlation metrics
        corr_center = np.corrcoef(target_center, pred_center)[0, 1]
        corr_half_range = np.corrcoef(target_half_range, pred_half_range)[0, 1]
        corr_range = np.corrcoef(target_range, pred_range)[0, 1]
        
        return {
            # Overall metrics
            'mae': mae_overall,
            'rmse': rmse_overall,
            
            # Center metrics
            'mae_center': mae_center,
            'rmse_center': rmse_center,
            'mape_center': mape_center,
            'correlation_center': corr_center,
            
            # Half-range metrics
            'mae_half_range': mae_half_range,
            'rmse_half_range': rmse_half_range,
            'mape_half_range': mape_half_range,
            'correlation_half_range': corr_half_range,
            
            # Range metrics
            'mae_range': mae_range,
            'rmse_range': rmse_range,
            'mape_range': mape_range,
            'correlation_range': corr_range
        }
    
    def _calculate_session_metrics(self, test_data: pd.DataFrame, 
                                 predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Calculate per-session evaluation metrics"""
        session_metrics = {}
        
        for session_id in test_data['session_id'].unique():
            session_mask = test_data['session_id'] == session_id
            session_targets = targets[session_mask]
            session_predictions = predictions[session_mask]
            
            if len(session_targets) > 0:
                session_metrics[session_id] = self._calculate_metrics(
                    session_targets, session_predictions
                )
                session_metrics[session_id]['sample_count'] = len(session_targets)
        
        # Calculate summary statistics across sessions
        if session_metrics:
            mae_values = [metrics['mae'] for metrics in session_metrics.values()]
            rmse_values = [metrics['rmse'] for metrics in session_metrics.values()]
            
            summary = {
                'session_count': len(session_metrics),
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values),
                'mae_min': np.min(mae_values),
                'mae_max': np.max(mae_values),
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values),
                'rmse_min': np.min(rmse_values),
                'rmse_max': np.max(rmse_values)
            }
            
            return {
                'per_session': session_metrics,
                'summary': summary
            }
        
        return {}
    
    @handle_oracle_errors(EvaluationError, "METRICS_SAVE_ERROR")
    def save_metrics(self, metrics: Dict[str, Any], output_path: Optional[Path] = None) -> None:
        """
        Save evaluation metrics to JSON file
        
        Args:
            metrics: Evaluation metrics dictionary
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = self.model_dir / ORACLE_MODEL_FILES['metrics']
        
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info(f"Evaluation metrics saved to {output_path}")
            
        except Exception as e:
            raise EvaluationError(f"Failed to save metrics: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.training_manifest is None:
            return {'error': 'Training manifest not available'}
        
        return {
            'version': self.training_manifest.version,
            'training_date': self.training_manifest.training_date.isoformat(),
            'training_sessions': len(self.training_manifest.training_sessions),
            'validation_sessions': len(self.training_manifest.validation_sessions),
            'total_training_pairs': self.training_manifest.total_training_pairs,
            'total_validation_pairs': self.training_manifest.total_validation_pairs,
            'early_pct': self.training_manifest.early_pct,
            'hyperparameters': self.training_manifest.hyperparameters,
            'training_metrics': self.training_manifest.training_metrics,
            'validation_metrics': self.training_manifest.validation_metrics
        }
