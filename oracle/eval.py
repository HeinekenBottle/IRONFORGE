"""
Oracle Model Evaluator

Computes comprehensive metrics (MAE, RMSE, MAPE) on test data and saves metrics.json.
Provides detailed analysis of Oracle performance across different scenarios.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

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
        
        # Load trained model
        self._load_trained_model()
        
    def _load_trained_model(self):
        """Load trained Oracle model and scaler"""
        weights_path = self.model_dir / "weights.pt"
        scaler_path = self.model_dir / "scaler.pkl"
        manifest_path = self.model_dir / "training_manifest.json"
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        # Load manifest for validation
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            logger.warning("Training manifest not found")
            self.manifest = {}
        
        # Initialize discovery engine
        self.discovery_engine = IRONFORGEDiscovery()
        
        # Load trained weights
        state_dict = torch.load(weights_path, map_location='cpu')
        self.discovery_engine.range_head.load_state_dict(state_dict)
        self.discovery_engine.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Loaded trained Oracle model from {self.model_dir}")
    
    def load_test_data(self, training_pairs_file: Path, 
                      test_size: float = 0.2, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """Load and split test data"""
        logger.info(f"Loading test data from {training_pairs_file}")
        
        df = pd.read_parquet(training_pairs_file)
        
        # Same split as training to get consistent test set
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Extract features and targets
        embedding_cols = [f"pooled_{i:02d}" for i in range(44)]
        X_test = torch.tensor(test_df[embedding_cols].values, dtype=torch.float32)
        y_test = torch.tensor(test_df[["target_center", "target_half_range"]].values, dtype=torch.float32)
        
        return X_test, y_test, test_df
    
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions on batch of embeddings"""
        self.discovery_engine.eval()
        
        with torch.no_grad():
            # Get scaled predictions
            pred_scaled = self.discovery_engine.range_head(X)
            
            # Denormalize predictions
            pred_scaled_np = pred_scaled.numpy()
            pred_denorm = self.scaler.inverse_transform(pred_scaled_np)
            
            return torch.tensor(pred_denorm, dtype=torch.float32)
    
    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        
        # Separate center and half_range
        true_center = y_true_np[:, 0]
        pred_center = y_pred_np[:, 0]
        true_half_range = y_true_np[:, 1]
        pred_half_range = np.abs(y_pred_np[:, 1])  # Ensure positive
        
        # Center metrics
        center_mae = mean_absolute_error(true_center, pred_center)
        center_rmse = np.sqrt(mean_squared_error(true_center, pred_center))
        center_mape = np.mean(np.abs((true_center - pred_center) / np.maximum(np.abs(true_center), 1e-6))) * 100
        
        # Half_range metrics
        range_mae = mean_absolute_error(true_half_range, pred_half_range)
        range_rmse = np.sqrt(mean_squared_error(true_half_range, pred_half_range))
        range_mape = np.mean(np.abs((true_half_range - pred_half_range) / np.maximum(true_half_range, 1e-6))) * 100
        
        # Combined range metrics (high - low)
        true_range = true_half_range * 2
        pred_range = pred_half_range * 2
        
        full_range_mae = mean_absolute_error(true_range, pred_range)
        full_range_rmse = np.sqrt(mean_squared_error(true_range, pred_range))
        full_range_mape = np.mean(np.abs((true_range - pred_range) / np.maximum(true_range, 1e-6))) * 100
        
        # Directional accuracy (for range size)
        range_direction_correct = np.mean((pred_half_range > true_half_range.mean()) == 
                                        (true_half_range > true_half_range.mean()))
        
        return {
            "center_mae": float(center_mae),
            "center_rmse": float(center_rmse), 
            "center_mape": float(center_mape),
            "range_mae": float(range_mae),
            "range_rmse": float(range_rmse),
            "range_mape": float(range_mape),
            "full_range_mae": float(full_range_mae),
            "full_range_rmse": float(full_range_rmse),
            "full_range_mape": float(full_range_mape),
            "range_direction_accuracy": float(range_direction_correct),
            "n_samples": len(y_true)
        }
    
    def compute_detailed_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
                               test_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute detailed metrics with breakdowns by symbol, timeframe, etc."""
        base_metrics = self.compute_metrics(y_true, y_pred)
        
        detailed_metrics = {
            "overall": base_metrics,
            "by_symbol": {},
            "by_timeframe": {},
            "by_htf_mode": {},
            "by_confidence_quartile": {},
            "prediction_quality": {}
        }
        
        # Add predictions to test dataframe for analysis
        y_pred_np = y_pred.numpy()
        test_df_copy = test_df.copy()
        test_df_copy["pred_center"] = y_pred_np[:, 0]
        test_df_copy["pred_half_range"] = np.abs(y_pred_np[:, 1])
        
        # By symbol
        for symbol in test_df_copy["symbol"].unique():
            mask = test_df_copy["symbol"] == symbol
            if mask.sum() > 5:  # Only compute if enough samples
                symbol_metrics = self.compute_metrics(y_true[mask], y_pred[mask])
                detailed_metrics["by_symbol"][symbol] = symbol_metrics
        
        # By timeframe
        for tf in test_df_copy["tf"].unique():
            mask = test_df_copy["tf"] == tf
            if mask.sum() > 5:
                tf_metrics = self.compute_metrics(y_true[mask], y_pred[mask])
                detailed_metrics["by_timeframe"][tf] = tf_metrics
        
        # By HTF mode
        for htf_mode in test_df_copy["htf_mode"].unique():
            mask = test_df_copy["htf_mode"] == htf_mode
            if mask.sum() > 5:
                htf_metrics = self.compute_metrics(y_true[mask], y_pred[mask])
                detailed_metrics["by_htf_mode"][htf_mode] = htf_metrics
        
        # By confidence quartile (handle duplicate values)
        try:
            confidence_quartiles = pd.qcut(test_df_copy["attention_confidence"], 
                                         q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates='drop')
        except ValueError:
            # If all values are the same, skip quartile analysis
            confidence_quartiles = pd.Series(['Q1'] * len(test_df_copy), index=test_df_copy.index)
        
        for quartile in ["Q1", "Q2", "Q3", "Q4"]:
            mask = confidence_quartiles == quartile
            if mask.sum() > 5:
                conf_metrics = self.compute_metrics(y_true[mask], y_pred[mask])
                detailed_metrics["by_confidence_quartile"][quartile] = conf_metrics
        
        # Prediction quality analysis
        pred_errors = {
            "center_errors": np.abs(y_true.numpy()[:, 0] - y_pred_np[:, 0]).tolist(),
            "range_errors": np.abs(y_true.numpy()[:, 1] - np.abs(y_pred_np[:, 1])).tolist()
        }
        
        detailed_metrics["prediction_quality"] = {
            "center_error_percentiles": {
                "p50": float(np.percentile(pred_errors["center_errors"], 50)),
                "p75": float(np.percentile(pred_errors["center_errors"], 75)),
                "p90": float(np.percentile(pred_errors["center_errors"], 90)),
                "p95": float(np.percentile(pred_errors["center_errors"], 95))
            },
            "range_error_percentiles": {
                "p50": float(np.percentile(pred_errors["range_errors"], 50)),
                "p75": float(np.percentile(pred_errors["range_errors"], 75)),
                "p90": float(np.percentile(pred_errors["range_errors"], 90)),
                "p95": float(np.percentile(pred_errors["range_errors"], 95))
            }
        }
        
        return detailed_metrics
    
    def evaluate_model(self, training_pairs_file: Path, 
                      test_size: float = 0.2) -> Dict[str, Any]:
        """Run complete model evaluation with enhanced error handling"""
        logger.info("Starting Oracle model evaluation")
        
        try:
            # Load test data
            logger.info(f"Loading test data from {training_pairs_file}")
            X_test, y_test, test_df = self.load_test_data(training_pairs_file, test_size)
            
            # Validate test data dimensions
            if X_test.shape[1] != 44:
                raise ValueError(f"Test data dimension mismatch: expected 44D features, got {X_test.shape[1]}D")
            
            if y_test.shape[1] != 2:
                raise ValueError(f"Test target dimension mismatch: expected 2D targets, got {y_test.shape[1]}D")
            
            logger.info(f"✅ Test data loaded: {len(test_df)} samples, {X_test.shape[1]}D features")
            
            # Make predictions with validation
            logger.info("Making predictions on test set")
            y_pred = self.predict_batch(X_test)
            
            # Validate predictions
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                raise RuntimeError("Invalid predictions detected (NaN or Inf values)")
            
            logger.info(f"✅ Predictions completed: shape {y_pred.shape}")
            
            # Compute detailed metrics
            logger.info("Computing evaluation metrics")
            detailed_metrics = self.compute_detailed_metrics(y_true=y_test, y_pred=y_pred, test_df=test_df)
            
            # Validate computed metrics
            overall_metrics = detailed_metrics.get('overall', {})
            critical_metrics = ['center_mae', 'center_rmse', 'center_mape']
            
            for metric in critical_metrics:
                value = overall_metrics.get(metric)
                if value is None or np.isnan(value) or np.isinf(value):
                    raise RuntimeError(f"Invalid metric computed: {metric} = {value}")
            
            logger.info("✅ Metrics validation passed")
            
            # Add evaluation metadata
            evaluation_info = {
                "model_dir": str(self.model_dir),
                "training_data": str(training_pairs_file),
                "test_samples": len(test_df),
                "evaluation_timestamp": pd.Timestamp.now().isoformat(),
                "test_data_summary": {
                    "symbols": test_df["symbol"].value_counts().to_dict(),
                    "timeframes": test_df["tf"].value_counts().to_dict(),
                    "htf_modes": test_df["htf_mode"].value_counts().to_dict(),
                    "date_range": {
                        "start": test_df["session_date"].min(),
                        "end": test_df["session_date"].max()
                    }
                },
                "model_validation": {
                    "discovery_engine_loaded": self.discovery_engine is not None,
                    "scaler_loaded": self.scaler is not None,
                    "feature_dimensions": X_test.shape[1],
                    "target_dimensions": y_test.shape[1]
                }
            }
            
            logger.info("✅ Oracle model evaluation completed successfully")
            
            return {
                "evaluation_info": evaluation_info,
                "metrics": detailed_metrics,
                "model_manifest": self.manifest
            }
            
        except Exception as e:
            logger.error(f"❌ Oracle model evaluation failed: {e}")
            import traceback
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            
            # Re-raise the exception to ensure CLI fails with non-zero exit
            raise RuntimeError(f"Oracle evaluation failed: {e}") from e
    
    def save_metrics(self, evaluation_results: Dict[str, Any], output_path: Path = None):
        """Save evaluation metrics to JSON file"""
        if output_path is None:
            output_path = self.model_dir / "metrics.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation metrics saved to: {output_path}")
    
    def print_summary(self, evaluation_results: Dict[str, Any]):
        """Print evaluation summary"""
        overall_metrics = evaluation_results["metrics"]["overall"]
        
        print(f"\n📊 Oracle Model Evaluation Summary")
        print(f"{'='*50}")
        print(f"Test samples: {overall_metrics['n_samples']:,}")
        print(f"\n🎯 Center Prediction:")
        print(f"  MAE:  {overall_metrics['center_mae']:.2f}")
        print(f"  RMSE: {overall_metrics['center_rmse']:.2f}")
        print(f"  MAPE: {overall_metrics['center_mape']:.1f}%")
        print(f"\n📏 Range Prediction (half_range):")
        print(f"  MAE:  {overall_metrics['range_mae']:.2f}")
        print(f"  RMSE: {overall_metrics['range_rmse']:.2f}")
        print(f"  MAPE: {overall_metrics['range_mape']:.1f}%")
        print(f"\n📐 Full Range Prediction (high-low):")
        print(f"  MAE:  {overall_metrics['full_range_mae']:.2f}")
        print(f"  RMSE: {overall_metrics['full_range_rmse']:.2f}")
        print(f"  MAPE: {overall_metrics['full_range_mape']:.1f}%")
        print(f"\n🔄 Direction Accuracy: {overall_metrics['range_direction_accuracy']:.1%}")
        
        # Performance assessment
        range_accuracy = 100 - overall_metrics['range_mape']
        if range_accuracy > 80:
            assessment = "🎯 Excellent"
        elif range_accuracy > 70:
            assessment = "✅ Good"
        elif range_accuracy > 60:
            assessment = "⚠️  Fair"
        else:
            assessment = "❌ Poor"
        
        print(f"\n{assessment} ({range_accuracy:.1f}% range accuracy)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained Oracle model")
    parser.add_argument("--model-dir", required=True, help="Trained model directory")
    parser.add_argument("--training-data", required=True, help="Training pairs parquet file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--output", help="Output metrics file (default: model_dir/metrics.json)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Initialize evaluator
    evaluator = OracleEvaluator(model_dir=Path(args.model_dir))
    
    # Run evaluation
    results = evaluator.evaluate_model(
        training_pairs_file=Path(args.training_data),
        test_size=args.test_size
    )
    
    # Save metrics
    output_path = Path(args.output) if args.output else None
    evaluator.save_metrics(results, output_path)
    
    # Print summary
    evaluator.print_summary(results)
    
    return 0


if __name__ == "__main__":
    exit(main())