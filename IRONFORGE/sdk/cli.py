"""
IRONFORGE SDK Command‑Line Interface (Waves 3-6)
==================================================
Provides CLI subcommands for:
- ``discover‑temporal``: Run TGAT discovery on Parquet shards (Wave 3)
- ``validate``: Run validation rails with OOS, purged k-fold, controls, ablations (Wave 4)
- ``report``: Generate heatmaps and confluence strips from discovery results (Wave 5)
- ``motifs``: Scan discovery outputs for motif card matches (Wave 6)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

# Base directories that shards must reside within. By default this allows any
# absolute path but tests can monkeypatch this to tighten the restriction.
APPROVED_SHARDS_BASE_DIRS = [Path("/")]

try:
    from ironforge.learning.discovery_pipeline import TemporalDiscoveryPipeline
except ImportError:
    TemporalDiscoveryPipeline = None  # type: ignore

try:
    from ironforge.validation.runner import ValidationConfig, ValidationRunner
except ImportError:
    ValidationRunner = None  # type: ignore
    ValidationConfig = None  # type: ignore


def _resolve_shards_dir(c: SimpleNamespace, allowed_roots: list[Path] | None = None) -> Path:
    """Resolve and validate ``c.paths.shards_dir``.

    The path is resolved to an absolute path and checked to ensure it is within
    one of the ``allowed_roots``. A :class:`ValueError` is raised if the
    resolved path escapes the allowed base directories.
    """

    allowed_roots = [Path(p).resolve() for p in (allowed_roots or APPROVED_SHARDS_BASE_DIRS)]

    shards_path = Path(c.paths.shards_dir).resolve()

    if not any(root == shards_path or root in shards_path.parents for root in allowed_roots):
        raise ValueError(
            f"Shards directory {shards_path} escapes approved base directories: {allowed_roots}"
        )

    c.paths.shards_dir = shards_path
    return shards_path


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
    discover_parser.add_argument(
        "--with-confluence",
        action="store_true",
        help="Compute and attach Confluence Score (0..100) to outputs",
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
        help="Validation mode (default: oos)",
    )
    validate_parser.add_argument(
        "--folds", type=int, default=5, help="Number of folds for k-fold validation (default: 5)"
    )
    validate_parser.add_argument(
        "--embargo-mins", type=int, default=30, help="Embargo period in minutes (default: 30)"
    )
    validate_parser.add_argument(
        "--controls",
        nargs="*",
        default=["time_shuffle", "label_perm"],
        help="Negative controls: time_shuffle, label_perm, node_shuffle, edge_direction, temporal_blocks",
    )
    validate_parser.add_argument(
        "--ablations",
        nargs="*",
        default=[],
        help="Feature groups to ablate: htf_prox, cycles, structure, etc.",
    )
    validate_parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/validation"),
        help="Directory to write validation reports (default: reports/validation)",
    )
    validate_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )

    # report subcommand (Wave 5)
    report_parser = subparsers.add_parser(
        "report",
        help="Generate heatmaps and confluence strips from discovery results",
    )
    report_parser.add_argument(
        "--discovery-file",
        type=Path,
        required=True,
        help="JSON file containing discovery results",
    )
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write generated reports (default: reports)",
    )
    report_parser.add_argument(
        "--format",
        choices=["heatmap", "confluence", "both"],
        default="both",
        help="Report format to generate (default: both)",
    )
    report_parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)",
    )
    report_parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Image height in pixels (default: auto)",
    )

    # motifs subcommand (Wave 6)
    motifs_parser = subparsers.add_parser(
        "motifs",
        help="Scan discovery outputs for motif card matches",
    )
    motifs_parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Discovery JSON with per-session events and (optional) confluence",
    )
    motifs_parser.add_argument("--min-confluence", type=float, default=65.0)
    motifs_parser.add_argument("--top-k", type=int, default=3)
    motifs_parser.add_argument(
        "--preset", type=str, default="default", help="Card preset (default for now)"
    )

    # prepare-motifs-input subcommand
    prep_parser = subparsers.add_parser(
        "prepare-motifs-input",
        help="Convert discovery/validation JSONs into motifs scanner input",
    )
    prep_parser.add_argument("--discovery-json", type=Path, required=True)
    prep_parser.add_argument("--validation-json", type=Path)
    prep_parser.add_argument("--out", type=Path, required=True)

    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `ironforge` CLI."""
    args = _parse_args(argv or sys.argv[1:])

    if args.command == "discover-temporal":
        # Resolve and validate the shards directory before running the pipeline.
        c = SimpleNamespace(paths=SimpleNamespace(shards_dir=args.data_path))
        shards_dir = _resolve_shards_dir(c)

        if TemporalDiscoveryPipeline is None:
            raise ImportError("TemporalDiscoveryPipeline not available; install Wave 3 components.")
        pipeline = TemporalDiscoveryPipeline(
            data_path=shards_dir,
            num_neighbors=args.fanouts,
            batch_size=args.batch_size,
            time_window=args.time_window,
            stitch_policy=args.stitch_policy,
            with_confluence=bool(args.with_confluence),
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
            random_seed=args.seed,
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
                "structure": [10, 11, 12, 13, 14],
            },
        }

        # Run validation
        runner = ValidationRunner(config)
        results = runner.run(synthetic_data)

        # Print summary
        summary = results.get("summary", {})
        print("\n✅ Validation completed!")
        print(f"📈 Main AUC: {summary.get('main_performance', {}).get('temporal_auc', 0.0):.4f}")
        print(f"🎯 Status: {'PASSED' if summary.get('validation_passed', False) else 'FAILED'}")

        return 0 if summary.get("validation_passed", False) else 1

    elif args.command == "report":
        # Import Wave 5 components
        try:
            import json

            import numpy as np

            from ironforge.reporting import (
                ConfluenceStripSpec,
                TimelineHeatmapSpec,
                build_confluence_strip,
                build_session_heatmap,
            )
        except ImportError as e:
            raise ImportError(f"Wave 5 reporting components not available: {e}")

        print("🎨 Starting IRONFORGE report generation...")
        print(f"📊 Discovery file: {args.discovery_file}")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🖼️  Format: {args.format}")

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Load discovery results
        if not args.discovery_file.exists():
            raise FileNotFoundError(f"Discovery file not found: {args.discovery_file}")

        with open(args.discovery_file) as f:
            discovery_data = json.load(f)

        print(f"📈 Loaded discovery data with {len(discovery_data.get('patterns', []))} patterns")

        # Extract timing and scoring data from discoveries
        patterns = discovery_data.get("patterns", [])
        if not patterns:
            print("⚠️  No patterns found in discovery data")
            return 1

        # Prepare data for visualization
        minute_bins = []
        densities = []
        scores_0_100 = []
        marker_minutes = []

        for pattern in patterns:
            # Extract temporal features
            temporal_features = pattern.get("temporal_features", {})
            duration = temporal_features.get("duration_seconds", 300) / 60  # Convert to minutes
            intensity = temporal_features.get("peak_intensity", 0.5)

            # Extract confidence as score
            confidence = pattern.get("confidence", 0.5) * 100  # Convert to 0-100 scale

            # Use batch_id or pattern index as minute offset
            minute_offset = pattern.get("batch_id", len(minute_bins)) * 5  # 5-minute intervals

            minute_bins.append(minute_offset)
            densities.append(intensity * 3.0)  # Scale for visibility
            scores_0_100.append(confidence)

            # Add markers for high-confidence patterns
            if confidence > 80:
                marker_minutes.append(minute_offset)

        # Convert to numpy arrays
        minute_bins = np.array(minute_bins)
        densities = np.array(densities)
        scores_0_100 = np.array(scores_0_100)
        marker_minutes = np.array(marker_minutes) if marker_minutes else None

        # Generate reports based on format
        reports_generated = []

        if args.format in ["heatmap", "both"]:
            print("🔥 Generating session heatmap...")

            # Configure heatmap spec
            heatmap_height = args.height or 160
            heatmap_spec = TimelineHeatmapSpec(
                width=args.width, height=heatmap_height, pad=8, colormap="viridis"
            )

            # Generate heatmap
            heatmap = build_session_heatmap(minute_bins, densities, heatmap_spec)

            # Save heatmap
            heatmap_path = args.output_dir / f"session_heatmap_{args.discovery_file.stem}.png"
            heatmap.save(heatmap_path, "PNG")
            reports_generated.append(str(heatmap_path))
            print(f"  ✅ Heatmap saved: {heatmap_path}")

        if args.format in ["confluence", "both"]:
            print("🌊 Generating confluence strip...")

            # Configure confluence spec
            confluence_height = args.height or 54
            confluence_spec = ConfluenceStripSpec(
                width=args.width, height=confluence_height, pad=6, marker_radius=3
            )

            # Generate confluence strip
            confluence = build_confluence_strip(
                minute_bins, scores_0_100, marker_minutes, confluence_spec
            )

            # Save confluence strip
            confluence_path = args.output_dir / f"confluence_strip_{args.discovery_file.stem}.png"
            confluence.save(confluence_path, "PNG")
            reports_generated.append(str(confluence_path))
            print(f"  ✅ Confluence strip saved: {confluence_path}")

        # Generate summary report
        summary_data = {
            "discovery_file": str(args.discovery_file),
            "generation_timestamp": discovery_data.get("timestamp", "unknown"),
            "total_patterns": len(patterns),
            "reports_generated": reports_generated,
            "visualization_stats": {
                "time_range_minutes": (
                    int(minute_bins.max() - minute_bins.min()) if len(minute_bins) > 0 else 0
                ),
                "average_confidence": float(np.mean(scores_0_100)) if len(scores_0_100) > 0 else 0,
                "high_confidence_patterns": (
                    len(marker_minutes) if marker_minutes is not None else 0
                ),
                "peak_density": float(np.max(densities)) if len(densities) > 0 else 0,
            },
        }

        summary_path = args.output_dir / f"report_summary_{args.discovery_file.stem}.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print("\n✅ Report generation completed!")
        print(f"📊 Generated {len(reports_generated)} visualization(s)")
        print(f"📋 Summary saved: {summary_path}")
        print(
            f"📈 Processed {len(patterns)} patterns across {summary_data['visualization_stats']['time_range_minutes']} minutes"
        )

        return 0

    elif args.command == "motifs":
        try:
            from ironforge.motifs.scanner import run_cli_scan
        except Exception as e:
            raise SystemExit(f"Motif scanner unavailable: {e}")
        run_cli_scan(
            args.input_json,
            top_k=args.top_k,
            min_confluence=args.min_confluence,
            preset=args.preset,
        )
        return 0

    elif args.command == "prepare-motifs-input":
        try:
            from ironforge.scripts.prepare_motifs_input import main as prep_main
        except Exception as e:
            raise SystemExit(f"Adapter unavailable: {e}")
        # reuse its argparse by building argv
        argv = [
            "--discovery-json",
            str(args.discovery_json),
            "--out",
            str(args.out),
        ]
        if args.validation_json:
            argv.extend(["--validation-json", str(args.validation_json)])
        return prep_main(argv)

    raise NotImplementedError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
