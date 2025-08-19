"""
Oracle Range Head Trainer

Trains only the range_head (44→32→2) while freezing TGAT layers.
Uses StandardScaler normalization, early stopping, and saves:
- models/oracle/v1.0.2/weights.pt
- models/oracle/v1.0.2/scaler.pkl  
- models/oracle/v1.0.2/training_manifest.json
"""

import json
import logging
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

logger = logging.getLogger(__name__)


class OracleTrainer:
    """Train Oracle range prediction head"""
    
    def __init__(self, model_dir: Path = Path("models/oracle/v1.0.2")):
        """
        Initialize Oracle trainer
        
        Args:
            model_dir: Directory to save model weights and metadata
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TGAT discovery engine (range_head will be trained)
        self.discovery_engine = IRONFORGEDiscovery()
        
        # Freeze TGAT layers (only train range_head)
        self._freeze_tgat_layers()
        
        # Training components
        self.scaler = StandardScaler()
        self.training_history = []
        self.best_model_state = None
        
        logger.info(f"Oracle trainer initialized, model dir: {model_dir}")
    
    def _freeze_tgat_layers(self):
        """Freeze all TGAT layers, only train range_head"""
        # Freeze attention layers
        for layer in self.discovery_engine.attention_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Freeze other components
        for param in self.discovery_engine.edge_processor.parameters():
            param.requires_grad = False
        for param in self.discovery_engine.pattern_classifier.parameters():
            param.requires_grad = False
        for param in self.discovery_engine.significance_scorer.parameters():
            param.requires_grad = False
            
        # Ensure range_head is trainable
        for param in self.discovery_engine.range_head.parameters():
            param.requires_grad = True
            
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.discovery_engine.parameters())
        trainable_params = sum(p.numel() for p in self.discovery_engine.parameters() if p.requires_grad)
        
        logger.info(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
        logger.info(f"Training only range_head: {trainable_params} parameters")
    
    def load_training_data(self, training_pairs_file: Path) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """Load training data from parquet file"""
        logger.info(f"Loading training data from {training_pairs_file}")
        
        df = pd.read_parquet(training_pairs_file)
        logger.info(f"Loaded {len(df)} training pairs")
        
        # Extract pooled embeddings (columns pooled_00 to pooled_43)
        embedding_cols = [f"pooled_{i:02d}" for i in range(44)]
        X = torch.tensor(df[embedding_cols].values, dtype=torch.float32)
        
        # Extract targets (center, half_range)
        y = torch.tensor(df[["target_center", "target_half_range"]].values, dtype=torch.float32)
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target center range: {y[:, 0].min():.1f} to {y[:, 0].max():.1f}")
        logger.info(f"Target half_range mean: {y[:, 1].mean():.1f}")
        
        return X, y, df
    
    def prepare_data_splits(self, X: torch.Tensor, y: torch.Tensor, 
                           test_size: float = 0.2, val_size: float = 0.2,
                           random_state: int = 42) -> Dict[str, torch.Tensor]:
        """Prepare train/val/test splits with normalization"""
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train/val split from train set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        logger.info(f"Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # Normalize targets using training set statistics
        y_train_np = y_train.numpy()
        self.scaler.fit(y_train_np)
        
        # Apply normalization
        y_train_scaled = torch.tensor(self.scaler.transform(y_train_np), dtype=torch.float32)
        y_val_scaled = torch.tensor(self.scaler.transform(y_val.numpy()), dtype=torch.float32)
        y_test_scaled = torch.tensor(self.scaler.transform(y_test.numpy()), dtype=torch.float32)
        
        logger.info(f"Target normalization applied (StandardScaler)")
        
        return {
            "X_train": X_train, "y_train": y_train_scaled,
            "X_val": X_val, "y_val": y_val_scaled, 
            "X_test": X_test, "y_test": y_test_scaled,
            "y_train_raw": y_train, "y_val_raw": y_val, "y_test_raw": y_test
        }
    
    def train_model(self, data_splits: Dict[str, torch.Tensor],
                   epochs: int = 100, learning_rate: float = 0.001,
                   batch_size: int = 32, patience: int = 15) -> Dict[str, Any]:
        """Train the range_head with early stopping"""
        logger.info(f"Starting training: {epochs} epochs, lr={learning_rate}, batch_size={batch_size}")
        
        # Prepare data loaders
        train_dataset = torch.utils.data.TensorDataset(data_splits["X_train"], data_splits["y_train"])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(data_splits["X_val"], data_splits["y_val"])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        range_head = self.discovery_engine.range_head
        range_head.train()
        
        optimizer = optim.Adam(range_head.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            range_head.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = range_head(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(range_head.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            range_head.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    pred = range_head(batch_X)
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model state
                self.best_model_state = range_head.state_dict().copy()
            else:
                patience_counter += 1
            
            # Logging
            if (epoch + 1) % 10 == 0 or patience_counter >= patience:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, "
                           f"Val: {avg_val_loss:.4f}, LR: {current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            range_head.load_state_dict(self.best_model_state)
        
        training_metrics = {
            "epochs_trained": epoch + 1,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "train_loss_history": train_losses,
            "val_loss_history": val_losses
        }
        
        self.training_history.append(training_metrics)
        logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
        
        return training_metrics
    
    def save_model(self, training_metadata: Dict[str, Any], git_sha: str = None) -> Dict[str, str]:
        """Save trained model, scaler, and training manifest"""
        logger.info(f"Saving model to {self.model_dir}")
        
        # Get git SHA if not provided
        if git_sha is None:
            try:
                result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                     capture_output=True, text=True, cwd=Path.cwd())
                git_sha = result.stdout.strip() if result.returncode == 0 else "unknown"
            except:
                git_sha = "unknown"
        
        # Save model weights
        weights_path = self.model_dir / "weights.pt"
        torch.save(self.discovery_engine.range_head.state_dict(), weights_path)
        
        # Save scaler
        scaler_path = self.model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Create training manifest
        manifest = {
            "version": "v1.0.2",
            "timestamp": datetime.now().isoformat(),
            "git_sha": git_sha,
            "model_architecture": {
                "input_dim": 44,
                "hidden_dim": 32,
                "output_dim": 2,
                "layers": ["Linear(44->32)", "ReLU", "Linear(32->2)"]
            },
            "training_config": training_metadata,
            "data_statistics": {
                "scaler_mean": self.scaler.mean_.tolist(),
                "scaler_scale": self.scaler.scale_.tolist()
            },
            "files": {
                "weights": "weights.pt",
                "scaler": "scaler.pkl",
                "manifest": "training_manifest.json"
            },
            "model_validation": {
                "embedding_dim": 44,
                "expected_input_shape": [44],
                "expected_output_shape": [2]
            }
        }
        
        # Save manifest
        manifest_path = self.model_dir / "training_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        file_paths = {
            "weights": str(weights_path),
            "scaler": str(scaler_path),
            "manifest": str(manifest_path)
        }
        
        logger.info(f"Model saved successfully:")
        for name, path in file_paths.items():
            logger.info(f"  {name}: {path}")
        
        return file_paths
    
    def validate_model_dims(self, expected_dims: Dict[str, Any]) -> bool:
        """Validate model dimensions match expected values"""
        range_head = self.discovery_engine.range_head
        
        # Check input dimension
        first_layer = range_head[0]  # First Linear layer
        if hasattr(first_layer, 'in_features'):
            actual_input_dim = first_layer.in_features
            expected_input_dim = expected_dims.get('input_dim', 44)
            
            if actual_input_dim != expected_input_dim:
                logger.error(f"Input dim mismatch: expected {expected_input_dim}, got {actual_input_dim}")
                return False
        
        # Check output dimension
        last_layer = range_head[-1]  # Last Linear layer
        if hasattr(last_layer, 'out_features'):
            actual_output_dim = last_layer.out_features
            expected_output_dim = expected_dims.get('output_dim', 2)
            
            if actual_output_dim != expected_output_dim:
                logger.error(f"Output dim mismatch: expected {expected_output_dim}, got {actual_output_dim}")
                return False
        
        logger.info("Model dimensions validated successfully")
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Oracle range prediction head")
    parser.add_argument("--training-data", required=True, help="Training pairs parquet file")
    parser.add_argument("--model-dir", default="models/oracle/v1.0.2", help="Model save directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation set fraction")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Initialize trainer
    trainer = OracleTrainer(model_dir=Path(args.model_dir))
    
    # Load and prepare data
    X, y, df = trainer.load_training_data(Path(args.training_data))
    data_splits = trainer.prepare_data_splits(X, y, test_size=args.test_size, val_size=args.val_size)
    
    # Train model
    training_metrics = trainer.train_model(
        data_splits,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patience=args.patience
    )
    
    # Prepare training metadata
    training_metadata = {
        "data_file": str(args.training_data),
        "training_samples": len(data_splits["X_train"]),
        "validation_samples": len(data_splits["X_val"]),
        "test_samples": len(data_splits["X_test"]),
        "symbols": df["symbol"].unique().tolist(),
        "timeframes": df["tf"].unique().tolist(),
        "htf_modes": df["htf_mode"].unique().tolist(),
        "date_range": {
            "start": df["session_date"].min(),
            "end": df["session_date"].max()
        },
        "early_pct": df["early_pct"].iloc[0],
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "test_size": args.test_size,
            "val_size": args.val_size
        },
        "metrics": training_metrics
    }
    
    # Save model
    file_paths = trainer.save_model(training_metadata)
    
    print(f"✅ Oracle training completed")
    print(f"📊 Best validation loss: {training_metrics['best_val_loss']:.4f}")
    print(f"📁 Model saved to: {args.model_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())