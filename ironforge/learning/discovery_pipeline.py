from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch

from ironforge.data_engine.parquet_reader import read_nodes_edges
from ironforge.learning.feature_adapter import create_feature_adapter
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

logger = logging.getLogger(__name__)


def run_discovery(shard_paths: Iterable[str], out_dir: str, loader_cfg) -> list[str]:  # type: ignore[no-untyped-def]
    """Run discovery over a list of shard directories.

    Parameters
    ----------
    shard_paths: iterable of str
        Paths to shard directories containing ``nodes.parquet`` and ``edges.parquet``.
    out_dir: str
        Output directory where results will be stored.
    loader_cfg: LoaderCfg
        Configuration for loader. Contains HTF settings and other parameters.

    Returns
    -------
    list[str]
        List of pattern parquet paths produced for each shard.
    """
    # Create output directories per contract
    run_dir = Path(out_dir)
    embeddings_dir = run_dir / "embeddings"
    patterns_dir = run_dir / "patterns"

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.mkdir(parents=True, exist_ok=True)

    # Determine HTF mode from loader config
    htf_enabled = getattr(loader_cfg, 'htf_enabled', False)

    # Create feature adapter
    feature_adapter = create_feature_adapter(htf_enabled=htf_enabled)

    # Initialize discovery engine
    discovery_engine = IRONFORGEDiscovery()
    discovery_engine.eval()

    outputs: list[str] = []

    for shard_path in shard_paths:
        try:
            # Read shard data
            nodes_df, edges_df = read_nodes_edges(shard_path)

            # Convert to TGAT-compatible graph using feature adapter
            graph = feature_adapter.adapt_shard_to_graph(nodes_df, edges_df)

            # Validate graph features
            if not feature_adapter.validate_graph_features(graph):
                logger.error(f"Feature validation failed for shard: {shard_path}")
                continue

            # Generate session identifier for file naming
            shard_name = Path(shard_path).name
            session_id = shard_name.replace("shard_", "").replace("/", "_")
            safe_session_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_id)

            # Run TGAT discovery
            with torch.no_grad():
                results = discovery_engine.forward(graph, return_attn=True)

            # Save embeddings (per-session file) under embeddings/
            embeddings = results["node_embeddings"]
            embeddings_path = embeddings_dir / f"node_embeddings_{safe_session_id}.parquet"
            pd.DataFrame(embeddings.numpy()).to_parquet(embeddings_path, index=False)

            # Save attention weights if available
            if results.get("attention_weights") is not None:
                attention_weights = results["attention_weights"]
                attention_path = embeddings_dir / f"attention_weights_{safe_session_id}.parquet"
                pd.DataFrame(attention_weights.numpy()).to_parquet(attention_path, index=False)

            # Save patterns (per-session file) under patterns/
            pattern_scores = results["pattern_scores"]
            significance_scores = results["significance_scores"]

            patterns_df = pd.DataFrame({
                "session_id": [safe_session_id] * len(pattern_scores),
                "pattern_scores": pattern_scores.tolist(),
                "significance_scores": significance_scores.flatten().tolist(),
                "node_count": [len(graph.nodes)] * len(pattern_scores),
                "edge_count": [len(graph.edges)] * len(pattern_scores),
            })

            patterns_path = patterns_dir / f"patterns_{safe_session_id}.parquet"
            patterns_df.to_parquet(patterns_path, index=False)
            outputs.append(str(patterns_path))

            logger.info(f"Discovery completed for {shard_path}: {len(pattern_scores)} patterns discovered")

        except Exception as e:
            logger.error(f"Discovery failed for shard {shard_path}: {e}")
            continue

    logger.info(f"Discovery pipeline completed: {len(outputs)} sessions processed")
    return outputs
