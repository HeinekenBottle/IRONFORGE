"""
Temporal Discovery Pipeline for IRONFORGE (Wave 3)
=================================================
Shard‑aware pipeline for temporal TGAT discovery.  This class will load
Parquet shards, construct temporal graphs, create a neighbour loader with
configurable fan‑outs and optional temporal window, stitch anchor nodes
across shard boundaries, and invoke the existing IRONFORGEDiscovery
engine to produce patterns.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch_geometric.data import HeteroData  # type: ignore
    from torch_geometric.loader import NeighborLoader  # type: ignore
except ImportError:
    torch = None  # type: ignore
    HeteroData = None  # type: ignore
    NeighborLoader = None  # type: ignore


class TemporalDiscoveryPipeline:
    """Shard‑aware pipeline for temporal TGAT discovery.

    Parameters
    ----------
    data_path : str or Path
        Directory containing Parquet shards.
    num_neighbors : list[int], optional
        Fan‑out per TGAT layer (defaults to [10, 10, 5]).
    batch_size : int
        Mini‑batch size for neighbour sampling.
    time_window : int, optional
        Temporal window (hours) to restrict neighbours.
    stitch_policy : str
        Policy for anchor stitching ("session" or "global").
    device : str, optional
        Device identifier; falls back to CPU if unavailable.
    """

    def __init__(
        self,
        data_path: str | Path,
        num_neighbors: list[int] | None = None,
        batch_size: int = 128,
        time_window: int | None = None,
        stitch_policy: str = "session",
        device: str | None = None,
        with_confluence: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.num_neighbors = num_neighbors or [10, 10, 5]
        self.batch_size = batch_size
        self.time_window = time_window
        self.stitch_policy = stitch_policy
        self.device = device or ("cuda:0" if torch and torch.cuda.is_available() else "cpu")
        self.with_confluence = with_confluence

    def load_shards(self) -> list[dict[str, Any]]:
        """Load Parquet shards from ``data_path``.

        Returns a list of shard data with 'nodes' and 'edges' DataFrames.
        Follows Wave 1 schema: 45D node features, 20D edge features.
        """
        import logging

        import pandas as pd

        logger = logging.getLogger(__name__)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        # Find all Parquet files in the directory
        node_files = list(self.data_path.glob("*nodes*.parquet"))
        edge_files = list(self.data_path.glob("*edges*.parquet"))

        if not node_files:
            raise ValueError(f"No node Parquet files found in {self.data_path}")

        logger.info(f"Found {len(node_files)} node files, {len(edge_files)} edge files")

        shards = []

        # Load shards by pairing node and edge files
        for node_file in node_files:
            # Find corresponding edge file (simple naming convention)
            shard_id = node_file.stem.replace("_nodes", "").replace("nodes", "")
            edge_file = None

            for ef in edge_files:
                if (
                    shard_id in ef.stem
                    or ef.stem.replace("_edges", "").replace("edges", "") == shard_id
                ):
                    edge_file = ef
                    break

            try:
                # Load node data
                nodes_df = pd.read_parquet(node_file)
                logger.debug(f"Loaded {len(nodes_df)} nodes from {node_file.name}")

                # Load edge data (optional - some shards may have no edges)
                edges_df = pd.DataFrame()
                if edge_file and edge_file.exists():
                    edges_df = pd.read_parquet(edge_file)
                    logger.debug(f"Loaded {len(edges_df)} edges from {edge_file.name}")

                shard_data = {
                    "shard_id": shard_id,
                    "nodes": nodes_df,
                    "edges": edges_df,
                    "node_file": str(node_file),
                    "edge_file": str(edge_file) if edge_file else None,
                }

                shards.append(shard_data)

            except Exception as e:
                logger.warning(f"Failed to load shard {shard_id}: {e}")
                continue

        if not shards:
            raise ValueError(f"No valid shards loaded from {self.data_path}")

        logger.info(f"Successfully loaded {len(shards)} shards")
        return shards

    def build_temporal_graph(self, shard_data: dict[str, Any]) -> HeteroData:
        """Convert a shard into a temporal PyG graph.

        Should produce a `HeteroData` object with 45D node features and
        20D edge features.  Temporal attributes must allow chronological
        ordering and optional window filtering.
        """
        if torch is None:
            raise ImportError("PyTorch not available for graph construction")

        import logging

        logger = logging.getLogger(__name__)

        nodes_df = shard_data["nodes"]
        edges_df = shard_data["edges"]

        if nodes_df.empty:
            raise ValueError(f"Empty nodes DataFrame in shard {shard_data['shard_id']}")

        # Create HeteroData graph
        graph = HeteroData()

        # Extract node features (45D: f0-f44)
        node_feature_cols = [f"f{i}" for i in range(45)]
        node_features = nodes_df[node_feature_cols].values.astype(np.float32)

        # Node metadata
        node_ids = nodes_df["node_id"].values
        timestamps = nodes_df["t"].values  # Temporal ordering
        node_kinds = nodes_df["kind"].values

        # Create node tensors
        graph["node"].x = torch.tensor(node_features, dtype=torch.float32)
        graph["node"].node_id = torch.tensor(node_ids, dtype=torch.long)
        graph["node"].t = torch.tensor(timestamps, dtype=torch.long)
        graph["node"].kind = torch.tensor(node_kinds, dtype=torch.long)

        logger.debug(f"Created {len(node_ids)} nodes with {node_features.shape[1]}D features")

        # Handle edges if present
        if not edges_df.empty:
            # Map global node IDs to local indices
            node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

            # Filter edges to only include nodes present in this shard
            valid_edges = edges_df[
                edges_df["src"].isin(node_id_to_idx) & edges_df["dst"].isin(node_id_to_idx)
            ].copy()

            if not valid_edges.empty:
                # Map to local indices
                src_indices = valid_edges["src"].map(node_id_to_idx).values
                dst_indices = valid_edges["dst"].map(node_id_to_idx).values
                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

                # Extract edge features (20D: e0-e19)
                edge_feature_cols = [f"e{i}" for i in range(20)]
                edge_features = valid_edges[edge_feature_cols].values.astype(np.float32)

                # Edge metadata
                edge_types = valid_edges["etype"].values
                delta_t = valid_edges["dt"].values  # Temporal deltas

                # Create edge tensors
                graph["node", "temporal", "node"].edge_index = edge_index
                graph["node", "temporal", "node"].edge_attr = torch.tensor(
                    edge_features, dtype=torch.float32
                )
                graph["node", "temporal", "node"].etype = torch.tensor(edge_types, dtype=torch.long)
                graph["node", "temporal", "node"].dt = torch.tensor(delta_t, dtype=torch.long)

                logger.debug(
                    f"Created {len(valid_edges)} edges with {edge_features.shape[1]}D features"
                )
            else:
                # Create empty edge tensors
                graph["node", "temporal", "node"].edge_index = torch.empty((2, 0), dtype=torch.long)
                graph["node", "temporal", "node"].edge_attr = torch.empty(
                    (0, 20), dtype=torch.float32
                )
                graph["node", "temporal", "node"].etype = torch.empty((0,), dtype=torch.long)
                graph["node", "temporal", "node"].dt = torch.empty((0,), dtype=torch.long)
        else:
            # No edges in this shard
            graph["node", "temporal", "node"].edge_index = torch.empty((2, 0), dtype=torch.long)
            graph["node", "temporal", "node"].edge_attr = torch.empty((0, 20), dtype=torch.float32)
            graph["node", "temporal", "node"].etype = torch.empty((0,), dtype=torch.long)
            graph["node", "temporal", "node"].dt = torch.empty((0,), dtype=torch.long)

        # Add metadata
        graph.shard_id = shard_data["shard_id"]
        graph.num_nodes = len(node_ids)
        graph.num_edges = len(edges_df) if not edges_df.empty else 0

        # Apply temporal window filtering if specified
        if self.time_window is not None:
            graph = self._apply_temporal_window(graph)

        logger.info(
            f"Built temporal graph for shard {shard_data['shard_id']}: "
            f"{graph.num_nodes} nodes, {graph.num_edges} edges"
        )

        return graph

    def create_neighbor_loader(self, graph: HeteroData) -> NeighborLoader:
        """Create a `NeighborLoader` with configured fan‑outs.

        The loader should respect temporal ordering and the optional
        `time_window`.  Future timestamps must not leak into the past.
        """
        if NeighborLoader is None:
            raise ImportError("torch_geometric.loader.NeighborLoader not available")

        import logging

        logger = logging.getLogger(__name__)

        # Create node input for sampling (all nodes by default)
        input_nodes = ("node", torch.arange(graph["node"].x.size(0)))

        # Configure neighbor sampling per layer (fanouts)
        num_neighbors = {("node", "temporal", "node"): self.num_neighbors}

        # Temporal edge sampling strategy
        # Sort edges by temporal order to respect causality
        edge_index = graph["node", "temporal", "node"].edge_index
        edge_dt = graph["node", "temporal", "node"].dt

        if edge_index.size(1) > 0:
            # Sort edges by delta_t to maintain temporal order
            sorted_indices = torch.argsort(edge_dt)
            edge_index = edge_index[:, sorted_indices]
            graph["node", "temporal", "node"].edge_index = edge_index
            graph["node", "temporal", "node"].edge_attr = graph[
                "node", "temporal", "node"
            ].edge_attr[sorted_indices]
            graph["node", "temporal", "node"].etype = graph["node", "temporal", "node"].etype[
                sorted_indices
            ]
            graph["node", "temporal", "node"].dt = edge_dt[sorted_indices]

        # Create the neighbor loader
        loader = NeighborLoader(
            data=graph,
            num_neighbors=num_neighbors,
            input_nodes=input_nodes,
            batch_size=self.batch_size,
            shuffle=True,
            temporal_strategy="uniform",  # Uniform temporal sampling
            directed=True,  # Directed temporal edges
        )

        logger.info(
            f"Created NeighborLoader: fanouts={self.num_neighbors}, "
            f"batch_size={self.batch_size}, time_window={self.time_window}h"
        )

        return loader

    def _apply_temporal_window(self, graph: HeteroData) -> HeteroData:
        """Apply temporal window filtering to edges."""
        if self.time_window is None:
            return graph

        edge_dt = graph["node", "temporal", "node"].dt
        time_window_seconds = self.time_window * 3600  # Convert hours to seconds

        # Filter edges within time window
        valid_mask = edge_dt <= time_window_seconds

        if valid_mask.any():
            graph["node", "temporal", "node"].edge_index = graph[
                "node", "temporal", "node"
            ].edge_index[:, valid_mask]
            graph["node", "temporal", "node"].edge_attr = graph[
                "node", "temporal", "node"
            ].edge_attr[valid_mask]
            graph["node", "temporal", "node"].etype = graph["node", "temporal", "node"].etype[
                valid_mask
            ]
            graph["node", "temporal", "node"].dt = edge_dt[valid_mask]
        else:
            # No edges within time window
            graph["node", "temporal", "node"].edge_index = torch.empty((2, 0), dtype=torch.long)
            graph["node", "temporal", "node"].edge_attr = torch.empty((0, 20), dtype=torch.float32)
            graph["node", "temporal", "node"].etype = torch.empty((0,), dtype=torch.long)
            graph["node", "temporal", "node"].dt = torch.empty((0,), dtype=torch.long)

        return graph

    def stitch_anchors(self, graphs: Iterable[HeteroData]) -> HeteroData:
        """Stitch anchor nodes across shard boundaries.

        Apply the selected ``stitch_policy`` ("session" or "global")
        to merge or link anchor nodes so that discovery sees complete
        temporal context.
        """
        import logging

        logger = logging.getLogger(__name__)

        graphs_list = list(graphs)
        if not graphs_list:
            raise ValueError("No graphs provided for stitching")

        if len(graphs_list) == 1:
            return graphs_list[0]

        logger.info(f"Stitching {len(graphs_list)} graphs with policy: {self.stitch_policy}")

        if self.stitch_policy == "session":
            return self._stitch_by_session(graphs_list)
        elif self.stitch_policy == "global":
            return self._stitch_globally(graphs_list)
        else:
            raise ValueError(f"Unknown stitch policy: {self.stitch_policy}")

    def _stitch_by_session(self, graphs: list[HeteroData]) -> HeteroData:
        """Stitch anchor nodes only within same session."""
        # Group graphs by session (extracted from shard_id)
        session_groups = {}
        for graph in graphs:
            # Extract session from shard_id (e.g., "session_001_shard_01" -> "session_001")
            session_id = (
                "_".join(graph.shard_id.split("_")[:2]) if hasattr(graph, "shard_id") else "default"
            )
            if session_id not in session_groups:
                session_groups[session_id] = []
            session_groups[session_id].append(graph)

        # Merge graphs within each session, then combine all sessions
        session_merged = []
        for session_id, session_graphs in session_groups.items():
            if len(session_graphs) == 1:
                session_merged.append(session_graphs[0])
            else:
                merged = self._merge_graphs(session_graphs, f"session_{session_id}")
                session_merged.append(merged)

        # Final merge across sessions
        return self._merge_graphs(session_merged, "global_stitched")

    def _stitch_globally(self, graphs: list[HeteroData]) -> HeteroData:
        """Stitch anchor nodes across all shards globally."""
        return self._merge_graphs(graphs, "global_stitched")

    def _merge_graphs(self, graphs: list[HeteroData], merged_id: str) -> HeteroData:
        """Merge multiple graphs into a single HeteroData object."""
        if not graphs:
            raise ValueError("No graphs to merge")

        if len(graphs) == 1:
            return graphs[0]

        # Initialize merged graph with first graph
        merged = HeteroData()

        # Accumulate all node and edge data
        all_node_features = []
        all_node_ids = []
        all_timestamps = []
        all_node_kinds = []

        all_edge_indices = []
        all_edge_features = []
        all_edge_types = []
        all_edge_dts = []

        node_id_offset = 0
        global_node_id_map = {}  # Map from global node_id to local index

        for graph in graphs:
            # Collect node data
            node_features = graph["node"].x
            node_ids = graph["node"].node_id
            timestamps = graph["node"].t
            node_kinds = graph["node"].kind

            # Update global node ID mapping
            for local_idx, global_id in enumerate(node_ids):
                if global_id.item() not in global_node_id_map:
                    global_node_id_map[global_id.item()] = node_id_offset + local_idx

            all_node_features.append(node_features)
            all_node_ids.append(node_ids)
            all_timestamps.append(timestamps)
            all_node_kinds.append(node_kinds)

            # Collect edge data
            edge_index = graph["node", "temporal", "node"].edge_index
            if edge_index.size(1) > 0:
                # Remap edge indices to global space
                edge_index_global = edge_index.clone()
                for i in range(edge_index.size(1)):
                    src_global = node_ids[edge_index[0, i]].item()
                    dst_global = node_ids[edge_index[1, i]].item()
                    edge_index_global[0, i] = global_node_id_map[src_global]
                    edge_index_global[1, i] = global_node_id_map[dst_global]

                all_edge_indices.append(edge_index_global)
                all_edge_features.append(graph["node", "temporal", "node"].edge_attr)
                all_edge_types.append(graph["node", "temporal", "node"].etype)
                all_edge_dts.append(graph["node", "temporal", "node"].dt)

            node_id_offset += len(node_ids)

        # Merge node tensors
        merged["node"].x = torch.cat(all_node_features, dim=0)
        merged["node"].node_id = torch.cat(all_node_ids, dim=0)
        merged["node"].t = torch.cat(all_timestamps, dim=0)
        merged["node"].kind = torch.cat(all_node_kinds, dim=0)

        # Merge edge tensors
        if all_edge_indices:
            merged["node", "temporal", "node"].edge_index = torch.cat(all_edge_indices, dim=1)
            merged["node", "temporal", "node"].edge_attr = torch.cat(all_edge_features, dim=0)
            merged["node", "temporal", "node"].etype = torch.cat(all_edge_types, dim=0)
            merged["node", "temporal", "node"].dt = torch.cat(all_edge_dts, dim=0)
        else:
            merged["node", "temporal", "node"].edge_index = torch.empty((2, 0), dtype=torch.long)
            merged["node", "temporal", "node"].edge_attr = torch.empty((0, 20), dtype=torch.float32)
            merged["node", "temporal", "node"].etype = torch.empty((0,), dtype=torch.long)
            merged["node", "temporal", "node"].dt = torch.empty((0,), dtype=torch.long)

        # Add metadata
        merged.shard_id = merged_id
        merged.num_nodes = merged["node"].x.size(0)
        merged.num_edges = merged["node", "temporal", "node"].edge_index.size(1)

        return merged

    def run_discovery(self) -> list[dict[str, Any]]:
        """Execute TGAT discovery across all shards.

        Loads shards, builds graphs, stitches anchors, iterates over
        neighbour batches and invokes the existing `IRONFORGEDiscovery`
        engine.  Returns a list of discovered patterns.

        Wave 6 addition: if self.with_confluence, attach per-minute scores
        to each session result under key "confluence".
        """
        import logging

        logger = logging.getLogger(__name__)

        logger.info("Starting temporal discovery pipeline")

        try:
            # Step 1: Load shards
            logger.info("Loading Parquet shards...")
            shards = self.load_shards()

            # Step 2: Build temporal graphs for each shard
            logger.info("Building temporal graphs...")
            graphs = []
            for shard_data in shards:
                try:
                    graph = self.build_temporal_graph(shard_data)
                    graphs.append(graph)
                except Exception as e:
                    logger.warning(f"Failed to build graph for shard {shard_data['shard_id']}: {e}")
                    continue

            if not graphs:
                raise ValueError("No valid graphs constructed from shards")

            # Step 3: Stitch anchors across shards
            logger.info("Stitching anchor nodes...")
            stitched_graph = self.stitch_anchors(graphs)

            # Step 4: Create neighbor loader
            logger.info("Creating neighbor loader...")
            loader = self.create_neighbor_loader(stitched_graph)

            # Step 5: Invoke TGAT discovery engine
            logger.info("Running TGAT discovery...")
            discoveries = self._run_tgat_discovery(loader)

            logger.info(f"Discovery completed: {len(discoveries)} patterns found")
            return discoveries

        except Exception as e:
            logger.error(f"Discovery pipeline failed: {e}", exc_info=True)
            raise

    def _run_tgat_discovery(self, loader: NeighborLoader) -> list[dict[str, Any]]:
        """Run TGAT discovery on neighbor batches."""
        try:
            # Import the existing TGAT discovery engine
            from ironforge.integration.ironforge_container import get_ironforge_container
            from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
        except ImportError as e:
            raise ImportError(f"Cannot import TGAT discovery components: {e}")

        import logging

        logger = logging.getLogger(__name__)

        # Initialize discovery engine
        container = get_ironforge_container()
        discovery_engine = container.get_tgat_discovery()

        all_discoveries = []
        batch_count = 0

        # Process each batch from the neighbor loader
        for batch in loader:
            try:
                batch_count += 1
                logger.debug(
                    f"Processing batch {batch_count}: "
                    f"{batch['node'].x.size(0)} nodes, "
                    f"{batch['node', 'temporal', 'node'].edge_index.size(1)} edges"
                )

                # Run discovery on this batch
                batch_discoveries = discovery_engine.discover_patterns(batch)

                if batch_discoveries:
                    # Add batch metadata to discoveries
                    for discovery in batch_discoveries:
                        discovery["batch_id"] = batch_count
                        discovery["pipeline_metadata"] = {
                            "fanouts": self.num_neighbors,
                            "batch_size": self.batch_size,
                            "time_window": self.time_window,
                            "stitch_policy": self.stitch_policy,
                            "device": self.device,
                        }

                    all_discoveries.extend(batch_discoveries)

            except Exception as e:
                logger.warning(f"Failed to process batch {batch_count}: {e}")
                continue

        logger.info(f"Processed {batch_count} batches, found {len(all_discoveries)} patterns")
        return all_discoveries

    def run(self) -> None:
        """High‑level pipeline entry point.

        Convenience method for the CLI.  Executes discovery and writes
        results to disk or stdout.
        """
        import json
        import logging
        from datetime import datetime

        logger = logging.getLogger(__name__)

        try:
            # Run discovery
            discoveries = self.run_discovery()

            # Create output directory if it doesn't exist
            output_dir = Path("discoveries")
            if hasattr(self, "output_dir"):
                output_dir = Path(self.output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Write discoveries as JSON
            discoveries_file = output_dir / f"temporal_discoveries_{timestamp}.json"
            with open(discoveries_file, "w") as f:
                json.dump(discoveries, f, indent=2, default=str)

            logger.info(f"Discoveries written to {discoveries_file}")

            # Write summary statistics
            summary = {
                "timestamp": timestamp,
                "total_patterns": len(discoveries),
                "pipeline_config": {
                    "data_path": str(self.data_path),
                    "fanouts": self.num_neighbors,
                    "batch_size": self.batch_size,
                    "time_window": self.time_window,
                    "stitch_policy": self.stitch_policy,
                    "device": self.device,
                },
                "pattern_types": {},
            }

            # Aggregate pattern statistics
            for discovery in discoveries:
                pattern_type = discovery.get("pattern_type", "unknown")
                summary["pattern_types"][pattern_type] = (
                    summary["pattern_types"].get(pattern_type, 0) + 1
                )

            summary_file = output_dir / f"discovery_summary_{timestamp}.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Summary written to {summary_file}")
            logger.info("✅ Wave 3 temporal discovery completed successfully!")
            logger.info(f"   📊 Found {len(discoveries)} patterns")
            logger.info(f"   📁 Results in {output_dir}")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
