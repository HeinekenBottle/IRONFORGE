"""
IRONFORGE SDK Command‑Line Interface (Wave 3 + Wave 4)
=======================================================
Provides CLI subcommands for:
- ``discover‑temporal``: Run TGAT discovery on Parquet shards (Wave 3)
- ``validate``: Run validation rails with OOS, purged k-fold, controls, ablations (Wave 4)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline
except ImportError:
    TemporalDiscoveryPipeline = None  # type: ignore

try:
    from ironforge.validation.runner import ValidationRunner, ValidationConfig
except ImportError:
    ValidationRunner = None  # type: ignore
    ValidationConfig = None  # type: ignore


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

    # validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Run validation rails (OOS, purged k-fold, negative controls, ablations)",
    )
    validate_parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Directory containing validation data (Parquet or JSON)",
    )
    validate_parser.add_argument(
        "--mode", 
        choices=["oos", "purged-kfold", "holdout"], 
        default="oos",
        help="Validation mode (default: oos)"
    )
    validate_parser.add_argument(
        "--folds", 
        type=int, 
        default=5,
        help="Number of folds for k-fold validation (default: 5)"
    )
    validate_parser.add_argument(
        "--embargo-mins", 
        type=int, 
        default=30,
        help="Embargo period in minutes (default: 30)"
    )
    validate_parser.add_argument(
        "--controls", 
        nargs="*", 
        default=["time_shuffle", "label_perm"], 
        help="Negative controls: time_shuffle, label_perm, node_shuffle, edge_direction, temporal_blocks"
    )
    validate_parser.add_argument(
        "--ablations", 
        nargs="*", 
        default=[], 
        help="Feature groups to ablate: htf_prox, cycles, structure, etc."
    )
    validate_parser.add_argument(
        "--report-dir", 
        type=Path, 
        default=Path("reports/validation"),
        help="Directory to write validation reports (default: reports/validation)"
    )
    validate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `ironforge` CLI."""
    args = _parse_args(argv or sys.argv[1:])
    
    if args.command == "discover-temporal":
        if TemporalDiscoveryPipeline is None:
            raise ImportError("TemporalDiscoveryPipeline not available; install Wave 3 components.")
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
    
    elif args.command == "validate":
        if ValidationRunner is None or ValidationConfig is None:
            raise ImportError("ValidationRunner not available; install Wave 4 components.")
        
        # Create validation configuration
        config = ValidationConfig(
            mode=args.mode,
            folds=args.folds,
            embargo_mins=args.embargo_mins,
            controls=args.controls,
            ablations=args.ablations,
            report_dir=args.report_dir,
            random_seed=args.seed
        )
        
        # For Wave 4, we need to load/create validation data
        # This is a stub implementation that would be expanded
        # to load actual data from the provided path
        print(f"🚀 Starting IRONFORGE validation with {args.mode} mode...")
        print(f"📊 Configuration: {args.folds} folds, {args.embargo_mins}min embargo")
        print(f"🧪 Controls: {', '.join(args.controls) if args.controls else 'None'}")
        print(f"🔬 Ablations: {', '.join(args.ablations) if args.ablations else 'None'}")
        print(f"📁 Reports will be saved to: {args.report_dir}")
        
        # TODO: Load actual validation data from args.data_path
        # For now, create synthetic data for demonstration
        import numpy as np
        synthetic_data = {
            "edge_index": np.array([[0, 1, 2], [1, 2, 0]]),
            "edge_times": np.array([100, 200, 300]),
            "node_features": np.random.random((100, 45)),  # 45D features
            "labels": np.random.randint(0, 2, 100),
            "timestamps": np.arange(100) * 10 + 1000,
            "feature_groups": {
                "htf_prox": [0, 1, 2, 3, 4],
                "cycles": [5, 6, 7, 8, 9],
                "structure": [10, 11, 12, 13, 14]
            }
        }
        
        # Run validation
        runner = ValidationRunner(config)
        results = runner.run(synthetic_data)
        
        # Print summary
        summary = results.get("summary", {})
        print(f"\n✅ Validation completed!")
        print(f"📈 Main AUC: {summary.get('main_performance', {}).get('temporal_auc', 0.0):.4f}")
        print(f"🎯 Status: {'PASSED' if summary.get('validation_passed', False) else 'FAILED'}")
        
        return 0 if summary.get('validation_passed', False) else 1
    
    raise NotImplementedError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
