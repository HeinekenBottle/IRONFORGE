"""
Oracle Training Data Builder

Recomputes early embeddings from normalized sessions using the same pipeline as discovery.
Builds training_pairs.parquet with pooled 44D embeddings and target ranges.

No sidecar files required - uses EnhancedGraphBuilder and TGAT forward pass directly.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import torch
import numpy as np
from dataclasses import dataclass

from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

logger = logging.getLogger(__name__)


@dataclass
class TrainingPair:
    """Single training pair: early embedding -> target range"""
    symbol: str
    tf: str
    session_date: str
    htf_mode: str
    early_pct: float
    pooled_embedding: np.ndarray  # [44] pooled early embedding
    target_center: float
    target_half_range: float
    
    # Metadata
    source_session: str
    n_events_total: int
    n_events_early: int
    attention_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation"""
        result = {
            "symbol": self.symbol,
            "tf": self.tf,
            "session_date": self.session_date,
            "htf_mode": self.htf_mode,
            "early_pct": self.early_pct,
            "target_center": self.target_center,
            "target_half_range": self.target_half_range,
            "source_session": self.source_session,
            "n_events_total": self.n_events_total,
            "n_events_early": self.n_events_early,
            "attention_confidence": self.attention_confidence,
        }
        
        # Add pooled embedding as separate columns
        for i in range(44):
            result[f"pooled_{i:02d}"] = self.pooled_embedding[i]
            
        return result


class OracleDataBuilder:
    """Build training data for Oracle calibration"""
    
    def __init__(self, enhanced_dir: Path = Path("data/enhanced"), 
                 shard_dir: Path = Path("data/shards")):
        """
        Initialize Oracle data builder
        
        Args:
            enhanced_dir: Directory with enhanced JSON sessions
            shard_dir: Directory with parquet shards (backup data source)
        """
        self.enhanced_dir = Path(enhanced_dir)
        self.shard_dir = Path(shard_dir)
        
        # Initialize IRONFORGE components
        self.graph_builder = EnhancedGraphBuilder()
        self.discovery_engine = IRONFORGEDiscovery()
        self.discovery_engine.eval()  # Set to evaluation mode
        
        # Statistics
        self.processed_sessions = 0
        self.failed_sessions = 0
        self.training_pairs = []
        
        logger.info("Oracle data builder initialized")
    
    def load_normalized_sessions(self, sessions_file: Path) -> pd.DataFrame:
        """Load normalized sessions from parquet file"""
        logger.info(f"Loading normalized sessions from {sessions_file}")
        
        if not sessions_file.exists():
            raise FileNotFoundError(f"Normalized sessions file not found: {sessions_file}")
            
        df = pd.read_parquet(sessions_file)
        logger.info(f"Loaded {len(df)} normalized sessions")
        
        return df
    
    def find_session_data(self, session_info: pd.Series) -> Dict[str, Any]:
        """Find actual session data from enhanced or shard directories"""
        source_file = session_info["source_file"]
        symbol = session_info["symbol"]
        
        # Try enhanced directory first
        enhanced_candidates = [
            self.enhanced_dir / source_file,
            self.enhanced_dir / f"{source_file}.json",
            *self.enhanced_dir.glob(f"*{source_file}*"),
            *self.enhanced_dir.glob(f"**/*{source_file}*")
        ]
        
        for candidate in enhanced_candidates:
            if candidate.exists() and candidate.is_file():
                return self._load_session_file(candidate)
        
        # Try shard directory as fallback
        session_date = session_info["session_date"]
        shard_candidates = list(self.shard_dir.glob(f"{symbol}_*/*{session_date}*"))
        
        for shard_dir in shard_candidates:
            if shard_dir.is_dir() and (shard_dir / "meta.json").exists():
                return self._load_shard_session(shard_dir)
        
        raise FileNotFoundError(f"Could not find session data for {source_file}")
    
    def _load_session_file(self, file_path: Path) -> Dict[str, Any]:
        """Load session from JSON file"""
        import json
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Ensure it's a single session dict
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        elif isinstance(data, list):
            logger.warning(f"Multiple sessions in {file_path}, using first")
            data = data[0]
            
        return data
    
    def _load_shard_session(self, shard_dir: Path) -> Dict[str, Any]:
        """Load session from parquet shard"""
        import json
        
        # Load metadata
        with open(shard_dir / "meta.json", 'r') as f:
            metadata = json.load(f)
        
        # Load node and edge data
        nodes_df = pd.read_parquet(shard_dir / "nodes.parquet")
        
        # Convert to session format
        events = []
        for _, node in nodes_df.iterrows():
            # Extract basic event info from node features
            feature = node.get("feature", [])
            if len(feature) >= 15:  # Ensure we have price data
                events.append({
                    "index": len(events),
                    "price": feature[10],  # price_close is typically at index 10
                    "volume": feature[12] if len(feature) > 12 else 100.0,
                    "timestamp": node.get("timestamp", "2025-01-01T12:00:00"),
                    "feature": feature
                })
        
        return {
            "session_name": metadata.get("session_id", shard_dir.name),
            "timestamp": metadata.get("conversion_timestamp", "2025-01-01T12:00:00"),
            "events": events,
            "metadata": metadata,
            "symbol": metadata.get("symbol", "UNKNOWN")
        }
    
    def build_training_pair(self, session_info: pd.Series, early_pct: float = 0.20) -> TrainingPair:
        """Build single training pair from session"""
        try:
            # Load actual session data
            session_data = self.find_session_data(session_info)
            
            # Build graph using EnhancedGraphBuilder (same as discovery pipeline)
            graph = self.graph_builder.build_session_graph(session_data)
            nodes = list(graph.nodes())
            
            if len(nodes) < 5:
                raise ValueError(f"Session too small: {len(nodes)} events < 5 minimum")
            
            # Extract early subgraph
            k = max(1, int(len(nodes) * early_pct))
            early_nodes = nodes[:k]
            early_subgraph = graph.subgraph(early_nodes).copy()
            
            # Forward pass with attention (same as Oracle prediction)
            with torch.no_grad():
                results = self.discovery_engine.forward(early_subgraph, return_attn=True)
                
                embeddings = results["node_embeddings"]  # [k, 44]
                attention_weights = results["attention_weights"]  # [k, k] or None
            
            # Attention-weighted pooling (exactly as in predict_session_range)
            if attention_weights is not None and attention_weights.size(0) > 0:
                attn_scores = attention_weights.sum(dim=0)
                attn_weights = torch.softmax(attn_scores, dim=0)
                pooled_embedding = (attn_weights.unsqueeze(-1) * embeddings).sum(dim=0)  # [44]
                confidence = float(attn_weights.max().item())
            else:
                pooled_embedding = embeddings.mean(dim=0)  # [44]
                confidence = 0.5
            
            # Use ground truth targets from normalized session
            target_center = session_info["center"]
            target_half_range = session_info["half_range"]
            
            training_pair = TrainingPair(
                symbol=session_info["symbol"],
                tf=session_info["tf"],
                session_date=session_info["session_date"],
                htf_mode=session_info["htf_mode"],
                early_pct=early_pct,
                pooled_embedding=pooled_embedding.detach().numpy(),
                target_center=target_center,
                target_half_range=target_half_range,
                source_session=session_info["source_file"],
                n_events_total=len(nodes),
                n_events_early=k,
                attention_confidence=confidence
            )
            
            self.processed_sessions += 1
            return training_pair
            
        except Exception as e:
            logger.warning(f"Failed to build training pair for {session_info['source_file']}: {e}")
            self.failed_sessions += 1
            raise
    
    def build_training_pairs(self, sessions_df: pd.DataFrame, 
                           early_pct: float = 0.20,
                           max_sessions: int = None) -> pd.DataFrame:
        """Build training pairs from all sessions"""
        logger.info(f"Building training pairs from {len(sessions_df)} sessions (early_pct={early_pct})")
        
        # Limit sessions if requested
        if max_sessions:
            sessions_df = sessions_df.head(max_sessions)
            logger.info(f"Limited to first {len(sessions_df)} sessions")
        
        training_pairs = []
        
        for idx, session_info in sessions_df.iterrows():
            try:
                training_pair = self.build_training_pair(session_info, early_pct)
                training_pairs.append(training_pair)
                
                if len(training_pairs) % 10 == 0:
                    logger.info(f"Processed {len(training_pairs)} sessions...")
                    
            except Exception as e:
                logger.debug(f"Skipping session {idx}: {e}")
                continue
        
        logger.info(f"Built {len(training_pairs)} training pairs")
        logger.info(f"Success rate: {self.processed_sessions}/{self.processed_sessions + self.failed_sessions} "
                   f"({100 * self.processed_sessions / max(1, self.processed_sessions + self.failed_sessions):.1f}%)")
        
        if not training_pairs:
            logger.warning("No training pairs created")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([pair.to_dict() for pair in training_pairs])
        
        # Sort by session date
        df = df.sort_values("session_date")
        
        # Report statistics
        logger.info(f"Training pairs shape: {df.shape}")
        logger.info(f"Symbol distribution: {dict(df['symbol'].value_counts())}")
        logger.info(f"HTF mode distribution: {dict(df['htf_mode'].value_counts())}")
        logger.info(f"Average confidence: {df['attention_confidence'].mean():.3f}")
        
        return df
    
    def save_training_pairs(self, df: pd.DataFrame, output_path: Path):
        """Save training pairs to parquet file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Training pairs saved to: {output_path}")
        
        # Save metadata summary
        metadata = {
            "total_pairs": len(df),
            "symbols": df["symbol"].unique().tolist(),
            "timeframes": df["tf"].unique().tolist(),
            "htf_modes": df["htf_mode"].unique().tolist(),
            "date_range": {
                "start": df["session_date"].min(),
                "end": df["session_date"].max()
            },
            "early_pct": df["early_pct"].iloc[0] if not df.empty else None,
            "embedding_dim": 44,
            "avg_confidence": float(df["attention_confidence"].mean()),
            "processing_stats": {
                "processed": self.processed_sessions,
                "failed": self.failed_sessions
            }
        }
        
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to: {metadata_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Oracle training data from normalized sessions")
    parser.add_argument("--sessions", required=True, help="Normalized sessions parquet file")
    parser.add_argument("--output", required=True, help="Output training pairs parquet file")
    parser.add_argument("--early-pct", type=float, default=0.20, help="Early batch percentage")
    parser.add_argument("--max-sessions", type=int, help="Limit number of sessions to process")
    parser.add_argument("--enhanced-dir", default="data/enhanced", help="Enhanced sessions directory")
    parser.add_argument("--shard-dir", default="data/shards", help="Shard data directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Initialize builder
    builder = OracleDataBuilder(
        enhanced_dir=Path(args.enhanced_dir),
        shard_dir=Path(args.shard_dir)
    )
    
    # Load normalized sessions
    sessions_df = builder.load_normalized_sessions(Path(args.sessions))
    
    # Build training pairs
    training_df = builder.build_training_pairs(
        sessions_df, 
        early_pct=args.early_pct,
        max_sessions=args.max_sessions
    )
    
    # Save results
    if not training_df.empty:
        builder.save_training_pairs(training_df, Path(args.output))
        print(f"✅ Training pairs built: {len(training_df)} pairs")
        return 0
    else:
        print("❌ No training pairs created")
        return 1


if __name__ == "__main__":
    exit(main())