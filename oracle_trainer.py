"""
Oracle Temporal Non-locality Training System

Calibrate Oracle predictions by training the regression head using historical session data.
Transforms from cold-start mode (0.5 confidence, NaN predictions) to calibrated mode 
with accurate range predictions based on early 20% session events.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from simple_data_loader import SimpleDataLoader

logger = logging.getLogger(__name__)


class OracleTrainer:
    """
    Train Oracle temporal non-locality system for accurate session range predictions
    """
    
    def __init__(self, model_save_dir: Path = Path("oracle_weights")):
        """
        Initialize Oracle trainer
        
        Args:
            model_save_dir: Directory to save/load calibrated weights
        """
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.discovery_engine = IRONFORGEDiscovery()
        self.graph_builder = EnhancedGraphBuilder()
        self.data_loader = SimpleDataLoader()
        
        # Calibration metrics
        self.scaler = StandardScaler()
        self.training_history = []
        
        logger.info(f"Oracle Trainer initialized with weights dir: {model_save_dir}")
    
    def prepare_training_data(self, symbol: str = "NQ_M5", num_sessions: int = 100) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        """
        Prepare training data from historical sessions
        
        Args:
            symbol: Symbol to train on (default NQ_M5)
            num_sessions: Number of recent sessions to use for training
            
        Returns:
            Tuple of (early_embeddings, target_ranges) lists
        """
        logger.info(f"Preparing training data from {num_sessions} recent sessions of {symbol}")
        
        # Load recent session data
        shard_data = self.data_loader.load_recent_sessions(
            symbol=symbol,
            limit=num_sessions,
            enhanced_format=True
        )
        
        early_embeddings = []
        target_ranges = []
        session_metadata = []
        
        for session_data in shard_data:
            try:
                # Build full session graph
                graph = self.graph_builder.build_session_graph(session_data)
                nodes = list(graph.nodes())
                
                if len(nodes) < 5:  # Need minimum events for training
                    continue
                
                # Extract early 20% events and get embedding
                early_pct = 0.20
                k = max(1, int(len(nodes) * early_pct))
                early_nodes = nodes[:k]
                early_subgraph = graph.subgraph(early_nodes).copy()
                
                # Get early embedding via forward pass
                results = self.discovery_engine.forward(early_subgraph, return_attn=True)
                embeddings = results["node_embeddings"]  # [k, 44]
                attention_weights = results["attention_weights"]
                
                # Attention-weighted pooling (same as predict_session_range)
                if attention_weights is not None and attention_weights.size(0) > 0:
                    attn_scores = attention_weights.sum(dim=0)
                    attn_weights = torch.softmax(attn_scores, dim=0)
                    pooled_embedding = (attn_weights.unsqueeze(-1) * embeddings).sum(dim=0)
                else:
                    pooled_embedding = embeddings.mean(dim=0)
                
                # Extract actual session range from full graph
                session_prices = []
                for node_id in nodes:
                    node_data = graph.nodes[node_id]
                    if 'price' in node_data:
                        session_prices.append(node_data['price'])
                    elif 'feature' in node_data and len(node_data['feature']) > 10:
                        # Extract price from feature vector (assumed to be in position 10-12)
                        session_prices.append(node_data['feature'][10])  # price_close proxy
                
                if len(session_prices) < 3:
                    continue
                    
                actual_high = max(session_prices)
                actual_low = min(session_prices)
                actual_center = (actual_high + actual_low) / 2
                actual_half_range = (actual_high - actual_low) / 2
                
                early_embeddings.append(pooled_embedding.detach())
                target_ranges.append((actual_center, actual_half_range))
                
                session_metadata.append({
                    'session_name': session_data.get('session_name', 'unknown'),
                    'n_events': len(nodes),
                    'early_events': k,
                    'actual_range': actual_high - actual_low
                })
                
            except Exception as e:
                logger.warning(f"Failed to process session {session_data.get('session_name', 'unknown')}: {e}")
                continue
        
        logger.info(f"Prepared {len(early_embeddings)} training samples from {symbol}")
        
        # Save training metadata
        training_info = {
            'symbol': symbol,
            'num_sessions': len(early_embeddings),
            'sessions': session_metadata
        }
        
        training_info_path = self.model_save_dir / f"training_info_{symbol}.json"
        with open(training_info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        return early_embeddings, target_ranges
    
    def train_regression_head(
        self, 
        early_embeddings: List[torch.Tensor], 
        target_ranges: List[Tuple[float, float]],
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train the Oracle regression head using prepared data
        
        Args:
            early_embeddings: List of pooled early embeddings [44]
            target_ranges: List of (center, half_range) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            
        Returns:
            Dictionary of final training metrics
        """
        logger.info(f"Training Oracle regression head on {len(early_embeddings)} samples")
        
        # Convert to tensors
        X = torch.stack(early_embeddings)  # [N, 44]
        y = torch.tensor(target_ranges, dtype=torch.float32)  # [N, 2]
        
        # Normalize targets for stable training
        y_np = y.numpy()
        y_scaled = torch.tensor(self.scaler.fit_transform(y_np), dtype=torch.float32)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_scaled, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        range_head = self.discovery_engine.range_head
        range_head.train()
        
        optimizer = optim.Adam(range_head.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training phase
            range_head.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = range_head(batch_X)  # [batch, 2]
                loss = criterion(pred, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(range_head.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            
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
            
            avg_train_loss = train_loss / num_batches
            avg_val_loss = val_loss / val_batches
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self._save_calibrated_weights()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        final_metrics = {
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
        
        self.training_history.append(final_metrics)
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return final_metrics
    
    def _save_calibrated_weights(self) -> None:
        """Save calibrated Oracle weights"""
        weights_path = self.model_save_dir / "oracle_range_head.pth"
        scaler_path = self.model_save_dir / "target_scaler.pkl"
        
        # Save range_head weights
        torch.save(self.discovery_engine.range_head.state_dict(), weights_path)
        
        # Save scaler for target denormalization
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Calibrated weights saved to {weights_path}")
    
    def load_calibrated_weights(self) -> bool:
        """
        Load calibrated Oracle weights
        
        Returns:
            bool: True if weights loaded successfully
        """
        weights_path = self.model_save_dir / "oracle_range_head.pth"
        scaler_path = self.model_save_dir / "target_scaler.pkl"
        
        if not weights_path.exists():
            logger.warning(f"No calibrated weights found at {weights_path}")
            return False
        
        try:
            # Load range_head weights
            state_dict = torch.load(weights_path, map_location='cpu')
            self.discovery_engine.range_head.load_state_dict(state_dict)
            
            # Load scaler
            if scaler_path.exists():
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            logger.info(f"Calibrated weights loaded from {weights_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibrated weights: {e}")
            return False
    
    def evaluate_calibrated_oracle(self, test_symbol: str = "NQ_M5", num_test_sessions: int = 20) -> Dict[str, float]:
        """
        Evaluate calibrated Oracle performance on test data
        
        Args:
            test_symbol: Symbol to test on
            num_test_sessions: Number of test sessions
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating calibrated Oracle on {num_test_sessions} sessions of {test_symbol}")
        
        test_embeddings, test_targets = self.prepare_training_data(
            symbol=test_symbol, 
            num_sessions=num_test_sessions
        )
        
        if len(test_embeddings) == 0:
            logger.warning("No test data available")
            return {}
        
        self.discovery_engine.range_head.eval()
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for embedding, (actual_center, actual_half_range) in zip(test_embeddings, test_targets):
                # Get calibrated prediction
                pred = self.discovery_engine.range_head(embedding.unsqueeze(0))  # [1, 2]
                
                # Denormalize prediction
                pred_scaled = pred.numpy()
                pred_denorm = self.scaler.inverse_transform(pred_scaled)[0]
                
                pred_center, pred_half_range = pred_denorm[0], abs(pred_denorm[1])
                
                predictions.append((pred_center, pred_half_range))
                actuals.append((actual_center, actual_half_range))
        
        # Calculate metrics
        center_errors = [abs(p[0] - a[0]) for p, a in zip(predictions, actuals)]
        range_errors = [abs(p[1] - a[1]) for p, a in zip(predictions, actuals)]
        
        # Range percentage errors
        range_pct_errors = [
            abs(p[1] - a[1]) / max(a[1], 1e-6) * 100 
            for p, a in zip(predictions, actuals)
        ]
        
        metrics = {
            'n_test_samples': len(predictions),
            'center_mae': np.mean(center_errors),
            'center_rmse': np.sqrt(np.mean([e**2 for e in center_errors])),
            'range_mae': np.mean(range_errors),
            'range_rmse': np.sqrt(np.mean([e**2 for e in range_errors])),
            'range_mape': np.mean(range_pct_errors),
            'center_median_error': np.median(center_errors),
            'range_median_error': np.median(range_errors)
        }
        
        logger.info(f"Calibrated Oracle Evaluation Results:")
        logger.info(f"  Center MAE: {metrics['center_mae']:.2f}")
        logger.info(f"  Range MAE: {metrics['range_mae']:.2f}")  
        logger.info(f"  Range MAPE: {metrics['range_mape']:.1f}%")
        
        return metrics


def main():
    """Demonstrate Oracle calibration workflow"""
    trainer = OracleTrainer()
    
    # Step 1: Prepare training data from recent sessions
    print("🔄 Preparing training data from recent NQ_M5 sessions...")
    early_embeddings, target_ranges = trainer.prepare_training_data(
        symbol="NQ_M5",
        num_sessions=50
    )
    
    if len(early_embeddings) == 0:
        print("❌ No training data available. Check data loader and session availability.")
        return
    
    print(f"✅ Prepared {len(early_embeddings)} training samples")
    
    # Step 2: Train the regression head  
    print("\n🔄 Training Oracle regression head...")
    training_metrics = trainer.train_regression_head(
        early_embeddings=early_embeddings,
        target_ranges=target_ranges,
        epochs=100,
        learning_rate=0.001
    )
    
    print(f"✅ Training completed. Final validation loss: {training_metrics['final_val_loss']:.4f}")
    
    # Step 3: Evaluate calibrated performance
    print("\n🔄 Evaluating calibrated Oracle performance...")
    eval_metrics = trainer.evaluate_calibrated_oracle(
        test_symbol="NQ_M5",
        num_test_sessions=20
    )
    
    if eval_metrics:
        print(f"✅ Evaluation completed:")
        print(f"  📊 Range Accuracy (MAPE): {eval_metrics['range_mape']:.1f}%")
        print(f"  📊 Center Accuracy (MAE): {eval_metrics['center_mae']:.2f}")
    
    print(f"\n🎯 Calibrated weights saved to: {trainer.model_save_dir}")
    print("Oracle is now ready for calibrated predictions with improved accuracy!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()