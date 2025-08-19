#!/usr/bin/env python3
"""
Oracle Pairs Builder - Convert normalized sessions to training embeddings

Parquet-native implementation that builds training pairs directly from
Parquet shard data without enhanced JSON dependencies.
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import networkx as nx

from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from oracle.session_mapping import SessionMapper, SessionMappingError


class OraclePairsBuilder:
    """Builds Oracle training pairs from Parquet shards using TGAT embeddings"""
    
    def __init__(self, data_dir: str = "data/shards", verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        
        # Initialize components
        self.discovery = IRONFORGEDiscovery()
        self.graph_builder = EnhancedGraphBuilder()
        self.session_mapper = SessionMapper(base_shard_dir=data_dir)
        
    def log(self, message: str) -> None:
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[PAIRS_BUILDER] {message}")
    
    def load_shard_data(self, shard_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict]]:
        """
        Load shard data from Parquet files
        
        Args:
            shard_path: Path to shard directory
            
        Returns:
            Tuple of (nodes_df, edges_df, metadata_dict)
        """
        if not shard_path.exists():
            self.log(f"Shard path does not exist: {shard_path}")
            return None, None, None
        
        # Load metadata
        meta_file = shard_path / "meta.json"
        metadata = {}
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                self.log(f"Loaded metadata: {meta_file}")
            except Exception as e:
                self.log(f"Failed to load metadata {meta_file}: {e}")
        
        # Load nodes
        nodes_file = shard_path / "nodes.parquet"
        nodes_df = None
        if nodes_file.exists():
            try:
                nodes_df = pd.read_parquet(nodes_file)
                self.log(f"Loaded {len(nodes_df)} nodes from {nodes_file}")
            except Exception as e:
                self.log(f"Failed to load nodes {nodes_file}: {e}")
                return None, None, metadata
        
        # Load edges
        edges_file = shard_path / "edges.parquet"
        edges_df = None
        if edges_file.exists():
            try:
                edges_df = pd.read_parquet(edges_file)
                self.log(f"Loaded {len(edges_df)} edges from {edges_file}")
            except Exception as e:
                self.log(f"Failed to load edges {edges_file}: {e}")
                return nodes_df, None, metadata
        
        return nodes_df, edges_df, metadata
    
    def extract_early_nodes(self, nodes_df: pd.DataFrame, early_pct: float = 0.20) -> pd.DataFrame:
        """
        Extract early nodes from Parquet node data
        
        Args:
            nodes_df: DataFrame with node data
            early_pct: Percentage of nodes to consider as early
            
        Returns:
            DataFrame with early nodes
        """
        if nodes_df is None or len(nodes_df) == 0:
            self.log("No nodes to extract from")
            return pd.DataFrame()
        
        # Sort by timestamp to ensure proper temporal order
        if 't' in nodes_df.columns:
            nodes_df = nodes_df.sort_values('t')
        elif 'timestamp' in nodes_df.columns:
            nodes_df = nodes_df.sort_values('timestamp')
        else:
            # If no timestamp column, use index order
            nodes_df = nodes_df.sort_index()
        
        # Calculate early cutoff
        n_early = max(1, int(len(nodes_df) * early_pct))
        early_nodes = nodes_df.head(n_early).copy()
        
        self.log(f"Extracted {len(early_nodes)} early nodes from {len(nodes_df)} total nodes")
        return early_nodes
    
    def assemble_node_features(self, nodes_df: pd.DataFrame) -> torch.Tensor:
        """
        Assemble node features from Parquet DataFrame into feature tensor
        
        Args:
            nodes_df: DataFrame with individual feature columns
            
        Returns:
            Feature tensor of shape [N, 45] or [N, 51] depending on HTF mode
        """
        n_nodes = len(nodes_df)
        available_cols = set(nodes_df.columns)
        
        # Detect feature dimensions based on available columns
        has_51d = any(f'f{i}' in available_cols for i in range(45, 51))
        feature_dim = 51 if has_51d else 45
        
        self.log(f"Assembling {feature_dim}D features for {n_nodes} nodes")
        
        # Initialize feature tensor
        features = torch.zeros(n_nodes, feature_dim, dtype=torch.float32)
        
        try:
            # Assemble features in strict column order
            for i in range(feature_dim):
                col_name = f'f{i}'
                if col_name in available_cols:
                    features[:, i] = torch.tensor(nodes_df[col_name].values, dtype=torch.float32)
                else:
                    self.log(f"Warning: Missing feature column {col_name}, using zeros")
            
            # Validation assertion
            assert features.shape[1] in (45, 51), f"FEATURE_DIMS_MISMATCH: Expected 45D or 51D, got {features.shape[1]}D"
            
            self.log(f"✅ Assembled {feature_dim}D features successfully")
            return features
            
        except Exception as e:
            # Enhanced error reporting
            missing_features = [f'f{i}' for i in range(feature_dim) if f'f{i}' not in available_cols]
            available_features = [col for col in available_cols if col.startswith('f') and col[1:].isdigit()]
            
            error_msg = f"Feature assembly failed: {e}\n"
            error_msg += f"Missing feature columns: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}\n"
            error_msg += f"Available feature columns: {available_features[:10]}{'...' if len(available_features) > 10 else ''}"
            
            self.log(error_msg)
            raise RuntimeError(error_msg)

    def assemble_edge_features(self, edges_df: pd.DataFrame) -> torch.Tensor:
        """
        Assemble edge features from Parquet DataFrame into feature tensor
        
        Args:
            edges_df: DataFrame with edge data
            
        Returns:
            Feature tensor of shape [E, 20]
        """
        n_edges = len(edges_df)
        if n_edges == 0:
            return torch.zeros(0, 20, dtype=torch.float32)
        
        available_cols = set(edges_df.columns)
        
        # Initialize 20D edge features
        features = torch.zeros(n_edges, 20, dtype=torch.float32)
        
        try:
            # Use weight column if available, otherwise default values
            if 'weight' in available_cols:
                features[:, 0] = torch.tensor(edges_df['weight'].values, dtype=torch.float32)
            else:
                features[:, 0] = 1.0  # Default edge weight
            
            # Add temporal distance if available
            if 'temporal_distance' in available_cols:
                features[:, 1] = torch.tensor(edges_df['temporal_distance'].values, dtype=torch.float32)
            
            self.log(f"✅ Assembled 20D edge features for {n_edges} edges")
            return features
            
        except Exception as e:
            error_msg = f"Edge feature assembly failed: {e}\n"
            error_msg += f"Available edge columns: {list(available_cols)}"
            self.log(error_msg)
            raise RuntimeError(error_msg)

    def build_networkx_graph(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Optional[nx.Graph]:
        """
        Build NetworkX graph from Parquet node and edge data
        
        Args:
            nodes_df: DataFrame with node data
            edges_df: DataFrame with edge data
            
        Returns:
            NetworkX graph or None if construction fails
        """
        if nodes_df is None or len(nodes_df) == 0:
            self.log("Cannot build graph: no nodes")
            return None
        
        if edges_df is None or len(edges_df) == 0:
            self.log("Cannot build graph: no edges")
            return None
        
        try:
            # Assemble features using the new adapter
            node_features = self.assemble_node_features(nodes_df)
            edge_features = self.assemble_edge_features(edges_df)
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes with assembled features
            for i, (_, node_row) in enumerate(nodes_df.iterrows()):
                node_id = node_row.get('node_id', node_row.name)
                G.add_node(node_id, feature=node_features[i])
            
            # Add edges with assembled features
            for i, (_, edge_row) in enumerate(edges_df.iterrows()):
                source = edge_row.get('source', edge_row.get('src'))
                target = edge_row.get('target', edge_row.get('dst'))
                
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, 
                             feature=edge_features[i],
                             temporal_distance=edge_row.get('temporal_distance', 1.0))
            
            self.log(f"Built NetworkX graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
            return G
            
        except Exception as e:
            self.log(f"Failed to build NetworkX graph: {e}")
            return None
    
    def compute_early_embedding(self, early_nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Compute 44D embedding from early nodes using TGAT
        
        Args:
            early_nodes_df: DataFrame with early node data
            edges_df: DataFrame with edge data (filtered to early nodes)
            
        Returns:
            44D embedding vector or None if computation fails
        """
        if early_nodes_df is None or len(early_nodes_df) < 2:
            self.log("Insufficient early nodes for embedding computation")
            return None
        
        try:
            # Build NetworkX graph from early nodes and relevant edges
            early_node_ids = set(early_nodes_df.get('node_id', early_nodes_df.index))
            
            # Filter edges to only include connections between early nodes
            if edges_df is not None and len(edges_df) > 0:
                early_edges_mask = (
                    edges_df.get('source', edges_df.get('src')).isin(early_node_ids) &
                    edges_df.get('target', edges_df.get('dst')).isin(early_node_ids)
                )
                early_edges_df = edges_df[early_edges_mask]
            else:
                early_edges_df = pd.DataFrame()  # Empty edges
            
            # Build NetworkX graph
            G = self.build_networkx_graph(early_nodes_df, early_edges_df)
            
            if G is None or G.number_of_nodes() == 0:
                self.log("Empty graph built from early nodes")
                return None
            
            # TGAT forward pass using NetworkX graph
            with torch.no_grad():
                result = self.discovery(G, return_attn=True)
                embeddings = result.get('node_embeddings')
                attention_weights = result.get('attention_weights')
            
            # Check if embeddings were computed successfully
            if embeddings is None:
                self.log("No embeddings returned from TGAT")
                return None
            
            # Attention-weighted pooling
            if attention_weights is not None:
                try:
                    # Handle different attention weight formats
                    if isinstance(attention_weights, list) and len(attention_weights) > 0:
                        # Use last layer attention weights
                        attn = attention_weights[-1]
                        if attn.dim() > 2:
                            attn = attn.mean(dim=1)  # Average across heads if multi-head
                    else:
                        attn = attention_weights
                        if attn.dim() > 2:
                            attn = attn.mean(dim=1)  # Average across heads if multi-head
                    
                    # Pool using attention weights
                    if attn.dim() == 2:
                        attn_weights = torch.softmax(attn.mean(dim=0), dim=0)  # Average over target dimension
                        pooled = torch.sum(embeddings * attn_weights.unsqueeze(1), dim=0)
                    else:
                        # Fallback to mean pooling if attention format is unexpected
                        pooled = embeddings.mean(dim=0)
                except Exception as e:
                    self.log(f"Attention pooling failed: {e}, using mean pooling")
                    pooled = embeddings.mean(dim=0)
            else:
                # Fallback to mean pooling
                pooled = embeddings.mean(dim=0)
            
            # Ensure 44D output
            if pooled.shape[0] != 44:
                self.log(f"Warning: embedding dimension {pooled.shape[0]} != 44, truncating/padding")
                if pooled.shape[0] > 44:
                    pooled = pooled[:44]
                else:
                    padding = torch.zeros(44 - pooled.shape[0])
                    pooled = torch.cat([pooled, padding])
            
            return pooled.numpy()
            
        except Exception as e:
            self.log(f"Failed to compute embedding: {e}")
            return None
    
    def build_training_pair(self, session_row: Dict, early_pct: float = 0.20) -> Optional[Dict]:
        """
        Build a single training pair from a normalized session row using Parquet data
        
        Args:
            session_row: Session information with symbol, tf, session_date, etc.
            early_pct: Percentage of nodes to use for early embedding
            
        Returns:
            Training pair dict or None if construction fails
        """
        session_id = session_row['session_id']
        session_date = session_row['session_date']
        
        self.log(f"Building training pair for {session_id}")
        
        try:
            # Parse session components
            parsed = self.session_mapper.parse_session_id(session_id)
            session_type = parsed['session_type']
            
            # Resolve shard path
            shard_path = self.session_mapper.resolve_shard_path(
                session_row['symbol'], session_row['tf'], 
                session_type, session_date
            )
            
            # Load shard data
            nodes_df, edges_df, metadata = self.load_shard_data(shard_path)
            
            if nodes_df is None:
                self.log(f"Skipping {session_id} - no node data")
                return None
            
            # Extract early nodes
            early_nodes_df = self.extract_early_nodes(nodes_df, early_pct)
            if early_nodes_df.empty:
                self.log(f"Skipping {session_id} - no early nodes")
                return None
            
            # Compute early embedding
            embedding = self.compute_early_embedding(early_nodes_df, edges_df)
            if embedding is None:
                self.log(f"Skipping {session_id} - failed to compute embedding")
                return None
            
            # Create training pair
            training_pair = {
                'symbol': session_row['symbol'],
                'tf': session_row['tf'],
                'session_date': session_row['session_date'],
                'htf_mode': session_row.get('htf_mode', 'disabled'),
                'early_pct': early_pct,
                'target_center': session_row['center'],
                'target_half_range': session_row['half_range'],
                'attention_confidence': 0.5 + (len(early_nodes_df) * 0.05)  # Confidence based on node count
            }
            
            # Add 44D embedding features
            for i in range(44):
                training_pair[f'pooled_{i:02d}'] = float(embedding[i])
            
            self.log(f"Built training pair for {session_id} with {len(early_nodes_df)} early nodes")
            return training_pair
            
        except SessionMappingError as e:
            self.log(f"Skipping {session_id} - session mapping error: {e}")
            return None
        except Exception as e:
            self.log(f"Skipping {session_id} - unexpected error: {e}")
            return None
    
    def build_training_pairs(
        self, 
        normalized_sessions_df: pd.DataFrame,
        early_pct: float = 0.20,
        max_sessions: Optional[int] = None
    ) -> pd.DataFrame:
        """Build training pairs from normalized sessions"""
        
        sessions_to_process = normalized_sessions_df.copy()
        
        if max_sessions:
            sessions_to_process = sessions_to_process.head(max_sessions)
            self.log(f"Limited to {max_sessions} sessions")
        
        training_pairs = []
        
        for idx, session_row in sessions_to_process.iterrows():
            try:
                training_pair = self.build_training_pair(session_row.to_dict(), early_pct)
                if training_pair:
                    training_pairs.append(training_pair)
                else:
                    self.log(f"Failed to build pair for session {idx}")
                    
            except Exception as e:
                self.log(f"Error processing session {idx}: {e}")
                continue
        
        if not training_pairs:
            self.log("No training pairs built")
            return pd.DataFrame()
        
        df = pd.DataFrame(training_pairs)
        self.log(f"Built {len(df)} training pairs from {len(sessions_to_process)} sessions")
        
        return df


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Build Oracle training pairs from normalized sessions")
    parser.add_argument("--sessions", required=True, help="Normalized sessions parquet file")
    parser.add_argument("--output", required=True, help="Output training pairs parquet file")
    parser.add_argument("--early-pct", type=float, default=0.20, help="Early batch percentage")
    parser.add_argument("--enhanced-dir", default="data/enhanced", help="Enhanced session data directory")
    parser.add_argument("--max-sessions", type=int, help="Limit number of sessions to process")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Load normalized sessions
    sessions_df = pd.read_parquet(args.sessions)
    print(f"📊 Loaded {len(sessions_df)} normalized sessions")
    
    # Initialize pairs builder
    builder = OraclePairsBuilder(args.enhanced_dir, args.verbose)
    
    # Build training pairs
    print(f"🧠 Building training pairs with {args.early_pct:.0%} early events...")
    training_pairs_df = builder.build_training_pairs(
        sessions_df, 
        early_pct=args.early_pct,
        max_sessions=args.max_sessions
    )
    
    if training_pairs_df.empty:
        print("❌ No training pairs built")
        return 1
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_pairs_df.to_parquet(output_path, index=False)
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'n_pairs': len(training_pairs_df),
        'early_pct': args.early_pct,
        'source_sessions': args.sessions,
        'enhanced_dir': args.enhanced_dir,
        'embedding_dim': 44,
        'target_dim': 2,
        'symbols': sorted(training_pairs_df['symbol'].unique().tolist()),
        'timeframes': sorted(training_pairs_df['tf'].unique().tolist()),
        'htf_modes': sorted(training_pairs_df['htf_mode'].unique().tolist()),
        'date_range': {
            'start': training_pairs_df['session_date'].min(),
            'end': training_pairs_df['session_date'].max()
        }
    }
    
    metadata_file = output_path.with_suffix('.json').with_name(f"{output_path.stem}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print(f"✅ Built {len(training_pairs_df)} training pairs")
    print(f"📁 Saved to: {output_path}")
    print(f"📋 Metadata: {metadata_file}")
    print(f"🎯 Target stats: center={training_pairs_df['target_center'].mean():.1f}, range={training_pairs_df['target_half_range'].mean()*2:.1f}")
    
    return 0


if __name__ == "__main__":
    exit(main())