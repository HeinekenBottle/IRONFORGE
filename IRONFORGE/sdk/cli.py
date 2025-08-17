"""
IRONFORGE SDK Command‑Line Interface (Wave 3)
=============================================
Provides the ``discover‑temporal`` subcommand to run TGAT discovery on
Parquet shards.  The CLI parses arguments, configures the
`TemporalDiscoveryPipeline`, and executes the pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline
except ImportError:
    TemporalDiscoveryPipeline = None  # type: ignore


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="ironforge",
        description="IRONFORGE SDK command‑line interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # discover‑temporal subcommand
    discover_parser = subparsers.add_parser(
        "discover-temporal",
        help="Run temporal TGAT discovery across Parquet shards",
    )
    discover_parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Directory containing Parquet shards",
    )
    discover_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("discoveries"),
        help="Directory to write discovery results",
    )
    discover_parser.add_argument(
        "--fanouts",
        type=int,
        nargs="+",
        default=[10, 10, 5],
        help="Neighbour fan‑out per TGAT layer (e.g. 10 10 5)",
    )
    discover_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini‑batch size for neighbour sampling",
    )
    discover_parser.add_argument(
        "--time-window",
        type=int,
        default=None,
        help="Temporal window (hours) limiting neighbours",
    )
    discover_parser.add_argument(
        "--stitch-policy",
        type=str,
        default="session",
        choices=["session", "global"],
        help="Anchor stitching policy",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `ironforge` CLI."""
    args = _parse_args(argv or sys.argv[1:])
    if args.command == "discover-temporal":
        if TemporalDiscoveryPipeline is None:
            raise ImportError(
                "TemporalDiscoveryPipeline not available; install Wave 3 components."
            )
        pipeline = TemporalDiscoveryPipeline(
            data_path=args.data_path,
            num_neighbors=args.fanouts,
            batch_size=args.batch_size,
            time_window=args.time_window,
            stitch_policy=args.stitch_policy,
        )
        pipeline.output_dir = args.output_dir
        pipeline.run()
        return 0
    raise NotImplementedError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())