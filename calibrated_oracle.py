"""
Calibrated Oracle Interface

Enhanced Oracle predictions using trained regression head weights.
Provides both cold-start and calibrated prediction modes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
import torch

from oracle_trainer import OracleTrainer
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

logger = logging.getLogger(__name__)


class CalibratedOracle:
    """
    Enhanced Oracle with calibrated prediction capabilities
    """
    
    def __init__(self, weights_dir: Path = Path("oracle_weights")):
        """
        Initialize Calibrated Oracle
        
        Args:
            weights_dir: Directory containing trained Oracle weights
        """
        self.weights_dir = Path(weights_dir)
        self.trainer = OracleTrainer(model_save_dir=self.weights_dir)
        self.discovery_engine = self.trainer.discovery_engine
        self.graph_builder = EnhancedGraphBuilder()
        
        # Load calibrated weights if available
        self.is_calibrated = self.trainer.load_calibrated_weights()
        
        if self.is_calibrated:
            logger.info("🎯 Calibrated Oracle mode - Using trained regression head")
        else:
            logger.info("🔧 Cold-start Oracle mode - Using default Xavier initialization")
    
    def predict_session_range(
        self, 
        session_data: Dict[str, Any], 
        early_batch_pct: float = 0.20,
        force_cold_start: bool = False
    ) -> Dict[str, Any]:
        """
        Predict session range with calibrated or cold-start mode
        
        Args:
            session_data: Session JSON data
            early_batch_pct: Percentage of early events to use (0, 0.5]
            force_cold_start: Force cold-start predictions even if calibrated
            
        Returns:
            Oracle prediction results with calibration metadata
        """
        try:
            # Build session graph
            graph = self.graph_builder.build_session_graph(session_data)
            
            # Get base Oracle prediction
            if force_cold_start or not self.is_calibrated:
                # Use original cold-start prediction
                prediction = self._cold_start_prediction(graph, early_batch_pct)
                prediction["mode"] = "cold_start"
                prediction["calibrated"] = False
            else:
                # Use calibrated prediction with denormalization
                prediction = self._calibrated_prediction(graph, early_batch_pct)
                prediction["mode"] = "calibrated"
                prediction["calibrated"] = True
            
            # Add session metadata
            prediction["session_name"] = session_data.get("session_name", "unknown")
            prediction["timestamp"] = session_data.get("timestamp", "")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Calibrated Oracle prediction failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "mode": "error",
                "calibrated": False,
                "session_name": session_data.get("session_name", "unknown")
            }
    
    def _cold_start_prediction(self, graph: nx.Graph, early_batch_pct: float) -> Dict[str, Any]:
        """Get cold-start Oracle prediction using original method"""
        # Use original TGAT prediction method
        prediction = self.discovery_engine.predict_session_range(graph, early_batch_pct)
        
        # Add cold-start specific metadata
        prediction["confidence_type"] = "attention_based"
        prediction["training_status"] = "untrained_weights"
        
        return prediction
    
    def _calibrated_prediction(self, graph: nx.Graph, early_batch_pct: float) -> Dict[str, Any]:
        """Get calibrated Oracle prediction with denormalized outputs"""
        nodes = list(graph.nodes())
        if len(nodes) < 3:
            return self._empty_result()
        
        num_events = len(nodes)
        k = max(1, int(num_events * early_batch_pct))
        
        # Extract early subgraph
        early_nodes = nodes[:k]
        early_subgraph = graph.subgraph(early_nodes).copy()
        
        # Get early embedding
        results = self.discovery_engine.forward(early_subgraph, return_attn=True)
        embeddings = results["node_embeddings"]  # [k, 44]
        attention_weights = results["attention_weights"]
        
        # Attention-weighted pooling
        if attention_weights is not None and attention_weights.size(0) > 0:
            attn_scores = attention_weights.sum(dim=0)
            attn_weights = torch.softmax(attn_scores, dim=0)
            pooled = (attn_weights.unsqueeze(-1) * embeddings).sum(dim=0)
            attention_confidence = float(attn_weights.max().item())
        else:
            pooled = embeddings.mean(dim=0)
            attention_confidence = 0.5
        
        # Get calibrated prediction
        self.discovery_engine.range_head.eval()
        with torch.no_grad():
            range_pred = self.discovery_engine.range_head(pooled.unsqueeze(0))  # [1, 2]
            
            # Denormalize using fitted scaler
            pred_scaled = range_pred.numpy()
            pred_denorm = self.trainer.scaler.inverse_transform(pred_scaled)[0]
            
            center, half_range = pred_denorm[0], abs(pred_denorm[1])
        
        pred_high = center + half_range
        pred_low = center - half_range
        
        # Extract phase sequences for breadcrumb analysis
        phase_counts = self.discovery_engine._extract_phase_sequences(early_subgraph)
        
        # Calculate calibrated confidence (combination of attention + training certainty)
        calibrated_confidence = min(0.95, attention_confidence * 1.2)  # Boost for calibration
        
        return {
            "pct_seen": round(k / num_events, 4),
            "n_events": int(k),
            "pred_high": float(pred_high),
            "pred_low": float(pred_low),
            "center": float(center),
            "half_range": float(half_range),
            "confidence": float(calibrated_confidence),
            "confidence_type": "calibrated_attention_weighted",
            "training_status": "calibrated_regression_head",
            "notes": f"calibrated prediction from {k} early events",
            "node_idx_used": list(early_nodes),
            **phase_counts
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for edge cases"""
        return {
            "pct_seen": 0.0,
            "n_events": 0,
            "pred_high": 0.0,
            "pred_low": 0.0,
            "center": 0.0,
            "half_range": 0.0,
            "confidence": 0.0,
            "confidence_type": "empty_session",
            "training_status": "no_prediction_possible",
            "notes": "empty session - no predictions available",
            "node_idx_used": [],
            "early_expansion_cnt": 0,
            "early_retracement_cnt": 0,
            "early_reversal_cnt": 0,
            "first_seq": ""
        }
    
    def compare_predictions(
        self, 
        session_data: Dict[str, Any], 
        early_batch_pct: float = 0.20
    ) -> Dict[str, Any]:
        """
        Compare cold-start vs calibrated predictions side-by-side
        
        Args:
            session_data: Session JSON data
            early_batch_pct: Percentage of early events to use
            
        Returns:
            Dictionary containing both prediction modes and comparison
        """
        if not self.is_calibrated:
            return {
                "error": "No calibrated weights available for comparison",
                "calibrated": False
            }
        
        try:
            # Get both predictions
            cold_start = self.predict_session_range(
                session_data, early_batch_pct, force_cold_start=True
            )
            calibrated = self.predict_session_range(
                session_data, early_batch_pct, force_cold_start=False
            )
            
            # Calculate differences
            range_diff = abs(calibrated["half_range"] - cold_start["half_range"])
            center_diff = abs(calibrated["center"] - cold_start["center"])
            confidence_diff = calibrated["confidence"] - cold_start["confidence"]
            
            return {
                "session_name": session_data.get("session_name", "unknown"),
                "cold_start": cold_start,
                "calibrated": calibrated,
                "comparison": {
                    "range_difference": float(range_diff),
                    "center_difference": float(center_diff),
                    "confidence_improvement": float(confidence_diff),
                    "prediction_mode_change": calibrated["mode"] != cold_start["mode"]
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction comparison failed: {e}")
            return {
                "error": str(e),
                "calibrated": True,
                "session_name": session_data.get("session_name", "unknown")
            }
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get detailed calibration status and model information"""
        status = {
            "is_calibrated": self.is_calibrated,
            "weights_directory": str(self.weights_dir),
            "mode": "calibrated" if self.is_calibrated else "cold_start"
        }
        
        # Check for weight files
        weights_path = self.weights_dir / "oracle_range_head.pth"
        scaler_path = self.weights_dir / "target_scaler.pkl"
        training_info_path = self.weights_dir / "training_info_NQ_M5.json"
        
        status["files"] = {
            "weights_file": weights_path.exists(),
            "scaler_file": scaler_path.exists(), 
            "training_info": training_info_path.exists()
        }
        
        # Load training info if available
        if training_info_path.exists():
            try:
                import json
                with open(training_info_path, 'r') as f:
                    training_info = json.load(f)
                status["training_metadata"] = training_info
            except Exception as e:
                status["training_metadata_error"] = str(e)
        
        return status


def main():
    """Demonstrate calibrated Oracle usage"""
    oracle = CalibratedOracle()
    
    print("🔮 Calibrated Oracle Status:")
    status = oracle.get_calibration_status()
    print(f"  Mode: {status['mode']}")
    print(f"  Calibrated: {status['is_calibrated']}")
    print(f"  Weights Dir: {status['weights_directory']}")
    
    if status.get("training_metadata"):
        metadata = status["training_metadata"]
        print(f"  Training Sessions: {metadata.get('num_sessions', 'unknown')}")
        print(f"  Symbol: {metadata.get('symbol', 'unknown')}")
    
    # Example prediction (would need real session data)
    print(f"\n🎯 Oracle ready for {'calibrated' if oracle.is_calibrated else 'cold-start'} predictions!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()