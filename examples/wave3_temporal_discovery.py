#!/usr/bin/env python3
"""
Wave 3 Temporal Discovery Example

This script demonstrates how to use the new shard-aware temporal discovery
pipeline introduced in Wave 3. It shows both programmatic usage and CLI usage.
"""

import logging
import tempfile
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_example_shards(output_dir: Path) -> None:
    """Create example Parquet shards for demonstration."""
    logger.info("Creating example Parquet shards...")

    # Create 3 sessions with 2 shards each
    sessions = [
        {"id": "NY_AM_20250815", "base_time": 1000, "nodes": 100},
        {"id": "LONDON_20250815", "base_time": 2000, "nodes": 120},
        {"id": "ASIA_20250816", "base_time": 3000, "nodes": 80},
    ]

    for session in sessions:
        for shard_idx in range(2):
            shard_id = f"{session['id']}_shard_{shard_idx:02d}"

            # Generate node data (45D features)
            num_nodes = session["nodes"] // 2 + shard_idx * 10
            node_ids = range(
                session["base_time"] + shard_idx * 200,
                session["base_time"] + shard_idx * 200 + num_nodes,
            )

            nodes_df = pd.DataFrame(
                {
                    "node_id": list(node_ids),
                    "t": [
                        session["base_time"] + i * 60 for i in range(num_nodes)
                    ],  # 1-minute intervals
                    "kind": [(i + shard_idx) % 3 + 1 for i in range(num_nodes)],
                    **{
                        f"f{i}": [(i * 0.1 + j * 0.01) % 1.0 for j in range(num_nodes)]
                        for i in range(45)
                    },
                }
            )

            # Generate edge data (20D features)
            num_edges = max(num_nodes - 5, 10)
            edges_df = pd.DataFrame(
                {
                    "src": list(node_ids)[:num_edges],
                    "dst": list(node_ids)[1 : num_edges + 1],
                    "etype": [(i + shard_idx) % 4 + 1 for i in range(num_edges)],
                    "dt": [60 + i * 30 for i in range(num_edges)],  # Temporal deltas
                    **{
                        f"e{i}": [(i * 0.05 + j * 0.02) % 1.0 for j in range(num_edges)]
                        for i in range(20)
                    },
                }
            )

            # Save to Parquet
            nodes_file = output_dir / f"{shard_id}_nodes.parquet"
            edges_file = output_dir / f"{shard_id}_edges.parquet"

            nodes_df.to_parquet(nodes_file, index=False)
            edges_df.to_parquet(edges_file, index=False)

            logger.info(f"Created shard {shard_id}: {num_nodes} nodes, {num_edges} edges")


def example_programmatic_usage():
    """Demonstrate programmatic usage of TemporalDiscoveryPipeline."""
    logger.info("=== Programmatic Usage Example ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir) / "shards"
        output_path = Path(temp_dir) / "discoveries"

        data_path.mkdir()
        output_path.mkdir()

        # Create example data
        create_example_shards(data_path)

        # Import and configure pipeline
        try:
            from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline

            pipeline = TemporalDiscoveryPipeline(
                data_path=data_path,
                num_neighbors=[8, 4, 2],  # 3-layer TGAT with decreasing fanouts
                batch_size=64,
                time_window=2,  # 2-hour temporal window
                stitch_policy="session",  # Stitch anchors within sessions
            )

            # Set output directory
            pipeline.output_dir = output_path

            logger.info("Pipeline configured successfully")
            logger.info(f"  Data path: {data_path}")
            logger.info(f"  Fanouts: {pipeline.num_neighbors}")
            logger.info(f"  Batch size: {pipeline.batch_size}")
            logger.info(f"  Time window: {pipeline.time_window}h")
            logger.info(f"  Stitch policy: {pipeline.stitch_policy}")

            # For demonstration, we'll just load and validate shards
            # (actual discovery requires TGAT components)
            shards = pipeline.load_shards()
            logger.info(f"Successfully loaded {len(shards)} shards")

            for shard in shards:
                logger.info(
                    f"  Shard {shard['shard_id']}: "
                    f"{len(shard['nodes'])} nodes, {len(shard['edges'])} edges"
                )

            # Note: Full pipeline execution would be:
            # discoveries = pipeline.run_discovery()
            # But this requires the TGAT discovery engine

        except ImportError as e:
            logger.warning(f"Could not import pipeline: {e}")
            logger.info("This is expected if PyTorch/torch_geometric are not installed")


def example_cli_usage():
    """Demonstrate CLI usage."""
    logger.info("=== CLI Usage Example ===")

    # Show example CLI commands
    examples = [
        {
            "description": "Basic temporal discovery",
            "command": [
                "python",
                "-m",
                "ironforge.sdk.cli",
                "discover-temporal",
                "--data-path",
                "/path/to/parquet/shards",
                "--output-dir",
                "/path/to/output",
            ],
        },
        {
            "description": "Advanced configuration with custom fanouts",
            "command": [
                "python",
                "-m",
                "ironforge.sdk.cli",
                "discover-temporal",
                "--data-path",
                "/data/shards",
                "--output-dir",
                "/results",
                "--fanouts",
                "10",
                "5",
                "3",
                "--batch-size",
                "128",
                "--time-window",
                "6",
                "--stitch-policy",
                "global",
            ],
        },
        {
            "description": "High-performance setup for large datasets",
            "command": [
                "python",
                "-m",
                "ironforge.sdk.cli",
                "discover-temporal",
                "--data-path",
                "/large/dataset/shards",
                "--fanouts",
                "20",
                "10",
                "5",
                "--batch-size",
                "256",
                "--stitch-policy",
                "session",
            ],
        },
    ]

    for example in examples:
        logger.info(f"\n{example['description']}:")
        logger.info(f"  {' '.join(example['command'])}")

    logger.info("\nCLI Arguments:")
    logger.info("  --data-path: Directory containing Parquet shards (required)")
    logger.info("  --output-dir: Output directory for discoveries (default: discoveries)")
    logger.info("  --fanouts: Neighbor fanouts per TGAT layer (default: 10 10 5)")
    logger.info("  --batch-size: Mini-batch size (default: 128)")
    logger.info("  --time-window: Temporal window in hours (default: unlimited)")
    logger.info("  --stitch-policy: Anchor stitching policy (session|global, default: session)")


def example_expected_outputs():
    """Show what outputs to expect from Wave 3 discovery."""
    logger.info("=== Expected Outputs ===")

    logger.info("\nOutput files generated:")
    logger.info("  temporal_discoveries_YYYYMMDD_HHMMSS.json - Discovered patterns")
    logger.info("  discovery_summary_YYYYMMDD_HHMMSS.json - Summary statistics")

    logger.info("\nExample discovery pattern:")
    example_pattern = {
        "pattern_type": "temporal_cascade",
        "confidence": 0.92,
        "batch_id": 3,
        "temporal_features": {
            "duration_seconds": 1800,
            "peak_intensity": 0.85,
            "node_sequence": ["node_1001", "node_1015", "node_1023"],
        },
        "spatial_features": {
            "anchor_zones": ["40%", "618%"],
            "session_span": "NY_AM_20250815",
            "cross_session": False,
        },
        "pipeline_metadata": {
            "fanouts": [8, 4, 2],
            "batch_size": 64,
            "time_window": 2,
            "stitch_policy": "session",
            "device": "cpu",
        },
    }

    import json

    logger.info(f"\n{json.dumps(example_pattern, indent=2)}")

    logger.info("\nExample summary statistics:")
    example_summary = {
        "timestamp": "20250815_143022",
        "total_patterns": 27,
        "pipeline_config": {
            "data_path": "/path/to/shards",
            "fanouts": [8, 4, 2],
            "batch_size": 64,
            "time_window": 2,
            "stitch_policy": "session",
        },
        "pattern_types": {
            "temporal_cascade": 12,
            "cross_session_anchor": 8,
            "zone_confluence": 5,
            "redelivery_chain": 2,
        },
    }

    logger.info(f"\n{json.dumps(example_summary, indent=2)}")


def main():
    """Run all examples."""
    logger.info("🚀 Wave 3 Temporal Discovery Examples")
    logger.info("=====================================")

    try:
        example_programmatic_usage()
        example_cli_usage()
        example_expected_outputs()

        logger.info("\n✅ Examples completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Prepare your Parquet shards using Wave 1 data engine")
        logger.info("2. Choose appropriate fanouts and batch size for your dataset")
        logger.info("3. Run discovery using either programmatic API or CLI")
        logger.info("4. Analyze discovered patterns in the output JSON files")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
