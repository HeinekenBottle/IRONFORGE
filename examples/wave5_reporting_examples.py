#!/usr/bin/env python3
"""
Wave 5 Reporting Examples for IRONFORGE
========================================
Comprehensive examples demonstrating visualization and reporting capabilities
for temporal pattern discovery results.

Provides:
- Timeline heatmap generation from session data
- Confluence strip visualization with event markers
- CLI usage examples
- Integration with discovery pipeline outputs
- Batch report generation workflows
"""

import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_mock_discovery_data(n_patterns=25, session_name="NY_AM_20250817"):
    """Create realistic mock discovery data for examples."""
    logger.info(f"Creating mock discovery data with {n_patterns} patterns...")
    
    # Generate patterns with realistic structure
    patterns = []
    base_timestamp = datetime.now() - timedelta(hours=4)
    
    pattern_types = ["temporal_cascade", "cross_session_anchor", "zone_confluence", "fvg_redelivery", "bos_liquidity"]
    
    for i in range(n_patterns):
        # Realistic temporal features
        duration = np.random.uniform(300, 3600)  # 5 minutes to 1 hour
        intensity = np.random.beta(2, 5)  # Skewed toward lower intensities
        confidence = np.random.beta(3, 2)  # Skewed toward higher confidence
        
        # Pattern timing (realistic batch progression)
        batch_id = i // 3  # Group patterns into batches
        minute_offset = batch_id * 15 + np.random.uniform(-5, 5)  # 15-minute intervals with jitter
        
        pattern = {
            "pattern_id": f"{session_name}_pattern_{i:03d}",
            "pattern_type": np.random.choice(pattern_types),
            "confidence": float(confidence),
            "batch_id": batch_id,
            "temporal_features": {
                "duration_seconds": float(duration),
                "peak_intensity": float(intensity),
                "node_sequence": [f"node_{j}" for j in range(i, i + 3)],
                "minute_offset": float(minute_offset)
            },
            "spatial_features": {
                "anchor_zones": [f"{np.random.uniform(20, 80):.1f}%"],
                "session_span": session_name,
                "cross_session": np.random.choice([True, False], p=[0.2, 0.8])
            },
            "archaeological_significance": {
                "archaeological_value": np.random.choice(
                    ["high_archaeological_value", "medium_archaeological_value", "low_archaeological_value"],
                    p=[0.15, 0.35, 0.5]
                ),
                "permanence_score": float(np.random.uniform(0.3, 0.95))
            }
        }
        patterns.append(pattern)
    
    discovery_data = {
        "timestamp": base_timestamp.strftime("%Y%m%d_%H%M%S"),
        "session_name": session_name,
        "total_patterns": len(patterns),
        "patterns": patterns,
        "pipeline_metadata": {
            "fanouts": [10, 5, 3],
            "batch_size": 128,
            "time_window": 4,
            "stitch_policy": "session"
        }
    }
    
    logger.info(f"Generated {len(patterns)} patterns across {max([p['batch_id'] for p in patterns]) + 1} batches")
    return discovery_data


def example_1_basic_heatmap_generation():
    """Example 1: Basic timeline heatmap generation."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Timeline Heatmap Generation")
    logger.info("=" * 60)
    
    try:
        from ironforge.reporting import TimelineHeatmapSpec, build_session_heatmap
    except ImportError as e:
        logger.warning(f"Could not import Wave 5 components: {e}")
        logger.info("This is expected if PIL/Pillow is not installed")
        return None
    
    # Create synthetic session density data
    timeline_minutes = 240  # 4-hour session
    minute_bins = np.arange(0, timeline_minutes, 5)  # 5-minute bins
    
    # Create realistic density pattern (higher activity at open/close)
    densities = np.zeros_like(minute_bins, dtype=float)
    
    # Opening hour spike
    opening_spike = np.exp(-((minute_bins - 15) / 20) ** 2) * 3.0
    densities += opening_spike
    
    # Closing hour spike
    closing_spike = np.exp(-((minute_bins - (timeline_minutes - 30)) / 25) ** 2) * 2.5
    densities += closing_spike
    
    # Random activity throughout
    np.random.seed(42)
    noise = np.random.exponential(0.3, len(minute_bins))
    densities += noise
    
    logger.info(f"Generated density data: {len(minute_bins)} time bins")
    logger.info(f"Peak density: {densities.max():.2f}, Average: {densities.mean():.2f}")
    
    # Generate heatmap with default settings
    logger.info("Generating heatmap with default settings...")
    heatmap_default = build_session_heatmap(minute_bins, densities)
    
    # Generate heatmap with custom settings
    logger.info("Generating heatmap with custom settings...")
    custom_spec = TimelineHeatmapSpec(
        width=800,
        height=120,
        pad=4,
        colormap="plasma"  # Wave 5 keeps it simple
    )
    heatmap_custom = build_session_heatmap(minute_bins, densities, custom_spec)
    
    logger.info(f"Default heatmap size: {heatmap_default.size}")
    logger.info(f"Custom heatmap size: {heatmap_custom.size}")
    
    # Save to temporary files for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        default_path = temp_path / "heatmap_default.png"
        custom_path = temp_path / "heatmap_custom.png"
        
        heatmap_default.save(default_path, "PNG")
        heatmap_custom.save(custom_path, "PNG")
        
        logger.info(f"Saved heatmaps to temporary directory: {temp_dir}")
        logger.info(f"  Default: {default_path} ({default_path.stat().st_size} bytes)")
        logger.info(f"  Custom: {custom_path} ({custom_path.stat().st_size} bytes)")
    
    return {
        "default_heatmap": heatmap_default,
        "custom_heatmap": heatmap_custom,
        "timeline_minutes": timeline_minutes,
        "peak_density": float(densities.max())
    }


def example_2_confluence_strip_with_markers():
    """Example 2: Confluence strip generation with event markers."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Confluence Strip with Event Markers")
    logger.info("=" * 60)
    
    try:
        from ironforge.reporting import ConfluenceStripSpec, build_confluence_strip
    except ImportError as e:
        logger.warning(f"Could not import Wave 5 components: {e}")
        return None
    
    # Create confluence score timeline
    session_minutes = 300  # 5-hour session
    minute_bins = np.arange(0, session_minutes, 2)  # 2-minute resolution
    
    # Generate realistic confluence scores (0-100)
    np.random.seed(123)
    base_scores = np.random.uniform(20, 80, len(minute_bins))
    
    # Add some high-confluence zones
    high_zones = [60, 120, 180, 240]  # Minutes with high confluence
    for zone_minute in high_zones:
        np.abs(minute_bins - zone_minute).argmin()
        # Create gaussian peak around high-confluence zones
        influence = np.exp(-((minute_bins - zone_minute) / 15) ** 2)
        base_scores += influence * 30
    
    scores_0_100 = np.clip(base_scores, 0, 100)
    
    # Define event markers (significant pattern discoveries)
    marker_minutes = np.array([45, 85, 125, 165, 195, 225, 265])
    
    logger.info(f"Generated confluence timeline: {len(minute_bins)} points")
    logger.info(f"Score range: {scores_0_100.min():.1f} - {scores_0_100.max():.1f}")
    logger.info(f"Event markers: {len(marker_minutes)} events")
    
    # Generate confluence strip with markers
    logger.info("Generating confluence strip with event markers...")
    confluence_with_markers = build_confluence_strip(
        minute_bins, scores_0_100, marker_minutes
    )
    
    # Generate confluence strip without markers for comparison
    logger.info("Generating confluence strip without markers...")
    confluence_no_markers = build_confluence_strip(
        minute_bins, scores_0_100
    )
    
    # Generate with custom specification
    logger.info("Generating high-resolution confluence strip...")
    custom_spec = ConfluenceStripSpec(
        width=1280,
        height=80,
        pad=10,
        marker_radius=4
    )
    confluence_hires = build_confluence_strip(
        minute_bins, scores_0_100, marker_minutes, custom_spec
    )
    
    logger.info(f"Standard confluence strip: {confluence_with_markers.size}")
    logger.info(f"High-res confluence strip: {confluence_hires.size}")
    
    # Save demonstrations
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with_markers_path = temp_path / "confluence_with_markers.png"
        no_markers_path = temp_path / "confluence_no_markers.png"
        hires_path = temp_path / "confluence_hires.png"
        
        confluence_with_markers.save(with_markers_path, "PNG")
        confluence_no_markers.save(no_markers_path, "PNG")
        confluence_hires.save(hires_path, "PNG")
        
        logger.info(f"Saved confluence strips to: {temp_dir}")
        logger.info(f"  With markers: {with_markers_path.stat().st_size} bytes")
        logger.info(f"  Without markers: {no_markers_path.stat().st_size} bytes")
        logger.info(f"  High-res: {hires_path.stat().st_size} bytes")
    
    return {
        "confluence_with_markers": confluence_with_markers,
        "confluence_no_markers": confluence_no_markers,
        "confluence_hires": confluence_hires,
        "scores_range": (float(scores_0_100.min()), float(scores_0_100.max())),
        "marker_count": len(marker_minutes)
    }


def example_3_discovery_data_integration():
    """Example 3: Integration with discovery pipeline outputs."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Discovery Data Integration")
    logger.info("=" * 60)
    
    try:
        from ironforge.reporting import (
            ConfluenceStripSpec,
            TimelineHeatmapSpec,
            build_confluence_strip,
            build_session_heatmap,
        )
    except ImportError as e:
        logger.warning(f"Could not import Wave 5 components: {e}")
        return None
    
    # Create realistic discovery data
    discovery_data = create_mock_discovery_data(n_patterns=30, session_name="LONDON_20250817")
    patterns = discovery_data["patterns"]
    
    logger.info(f"Processing discovery data: {len(patterns)} patterns")
    
    # Extract visualization data from patterns
    minute_bins = []
    densities = []
    scores_0_100 = []
    marker_minutes = []
    
    for pattern in patterns:
        # Extract timing information
        temporal_features = pattern["temporal_features"]
        minute_offset = temporal_features.get("minute_offset", 0)
        intensity = temporal_features["peak_intensity"]
        
        # Extract confidence score
        confidence = pattern["confidence"] * 100  # Convert to 0-100 scale
        
        minute_bins.append(minute_offset)
        densities.append(intensity * 4.0)  # Scale for visibility
        scores_0_100.append(confidence)
        
        # Mark high-significance patterns
        arch_value = pattern["archaeological_significance"]["archaeological_value"]
        if arch_value == "high_archaeological_value" and confidence > 70:
            marker_minutes.append(minute_offset)
    
    # Convert to numpy arrays
    minute_bins = np.array(minute_bins)
    densities = np.array(densities)
    scores_0_100 = np.array(scores_0_100)
    marker_minutes = np.array(marker_minutes) if marker_minutes else None
    
    logger.info("Extracted visualization data:")
    logger.info(f"  Time range: {minute_bins.min():.1f} - {minute_bins.max():.1f} minutes")
    logger.info(f"  Density range: {densities.min():.2f} - {densities.max():.2f}")
    logger.info(f"  Confidence range: {scores_0_100.min():.1f} - {scores_0_100.max():.1f}")
    logger.info(f"  High-significance markers: {len(marker_minutes) if marker_minutes is not None else 0}")
    
    # Generate session heatmap from discovery data
    logger.info("Generating session heatmap from discovery patterns...")
    heatmap_spec = TimelineHeatmapSpec(width=1024, height=160, pad=8)
    session_heatmap = build_session_heatmap(minute_bins, densities, heatmap_spec)
    
    # Generate confluence strip from discovery data
    logger.info("Generating confluence strip from discovery patterns...")
    confluence_spec = ConfluenceStripSpec(width=1024, height=54, pad=6, marker_radius=3)
    confluence_strip = build_confluence_strip(
        minute_bins, scores_0_100, marker_minutes, confluence_spec
    )
    
    # Save integrated visualization
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        heatmap_path = temp_path / f"session_heatmap_{discovery_data['session_name']}.png"
        confluence_path = temp_path / f"confluence_strip_{discovery_data['session_name']}.png"
        discovery_path = temp_path / f"discovery_data_{discovery_data['session_name']}.json"
        
        session_heatmap.save(heatmap_path, "PNG")
        confluence_strip.save(confluence_path, "PNG")
        
        with open(discovery_path, 'w') as f:
            json.dump(discovery_data, f, indent=2)
        
        logger.info(f"Saved integrated visualization to: {temp_dir}")
        logger.info(f"  Heatmap: {heatmap_path.stat().st_size} bytes")
        logger.info(f"  Confluence: {confluence_path.stat().st_size} bytes")
        logger.info(f"  Discovery data: {discovery_path.stat().st_size} bytes")
    
    # Generate summary statistics
    pattern_types = {}
    for pattern in patterns:
        ptype = pattern["pattern_type"]
        pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
    
    summary_stats = {
        "total_patterns": len(patterns),
        "pattern_types": pattern_types,
        "time_span_minutes": float(minute_bins.max() - minute_bins.min()) if len(minute_bins) > 0 else 0,
        "average_confidence": float(np.mean(scores_0_100)) if len(scores_0_100) > 0 else 0,
        "high_significance_count": len(marker_minutes) if marker_minutes is not None else 0,
        "peak_density": float(np.max(densities)) if len(densities) > 0 else 0
    }
    
    logger.info("Discovery summary statistics:")
    for key, value in summary_stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"  {key}: {value}")
    
    return {
        "discovery_data": discovery_data,
        "session_heatmap": session_heatmap,
        "confluence_strip": confluence_strip,
        "summary_stats": summary_stats
    }


def example_4_cli_equivalent_usage():
    """Example 4: Programmatic equivalent of CLI usage."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: CLI Equivalent Usage")
    logger.info("=" * 60)
    
    # Show CLI command examples
    cli_examples = [
        {
            "description": "Basic heatmap and confluence generation",
            "command": [
                "python", "-m", "ironforge.sdk.cli", "report",
                "--discovery-file", "/path/to/discoveries_20250817_143022.json",
                "--output-dir", "./reports/wave5",
                "--format", "both"
            ]
        },
        {
            "description": "High-resolution heatmap only",
            "command": [
                "python", "-m", "ironforge.sdk.cli", "report",
                "--discovery-file", "/data/discoveries/ny_session.json",
                "--output-dir", "./reports/heatmaps",
                "--format", "heatmap",
                "--width", "1920",
                "--height", "200"
            ]
        },
        {
            "description": "Confluence strips for monitoring dashboard",
            "command": [
                "python", "-m", "ironforge.sdk.cli", "report",
                "--discovery-file", "/discoveries/latest.json",
                "--format", "confluence",
                "--width", "1280",
                "--height", "40"
            ]
        }
    ]
    
    logger.info("CLI Command Examples:")
    for i, example in enumerate(cli_examples, 1):
        logger.info(f"\n{i}. {example['description']}:")
        logger.info(f"   {' '.join(example['command'])}")
    
    logger.info("\nCLI Arguments Reference:")
    logger.info("  --discovery-file: JSON file containing discovery results (required)")
    logger.info("  --output-dir: Directory for generated reports (default: reports)")
    logger.info("  --format: heatmap, confluence, or both (default: both)")
    logger.info("  --width: Image width in pixels (default: 1024)")
    logger.info("  --height: Image height in pixels (default: auto)")
    
    # Demonstrate programmatic equivalent
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock discovery file
        discovery_data = create_mock_discovery_data(n_patterns=15)
        discovery_file = temp_path / "mock_discoveries.json"
        
        with open(discovery_file, 'w') as f:
            json.dump(discovery_data, f, indent=2)
        
        logger.info("\nProgrammatic Equivalent:")
        logger.info(f"Created mock discovery file: {discovery_file}")
        
        # This would be equivalent to CLI usage
        logger.info("# Equivalent to CLI command:")
        logger.info("# python -m ironforge.sdk.cli report \\")
        logger.info(f"#   --discovery-file {discovery_file} \\")
        logger.info(f"#   --output-dir {temp_path / 'reports'} \\")
        logger.info("#   --format both")
        
        # For demonstration, we'll show what the CLI would do
        try:
            from ironforge.reporting import (
                ConfluenceStripSpec,
                TimelineHeatmapSpec,
                build_confluence_strip,
                build_session_heatmap,
            )
            
            # Process the discovery file (same as CLI would do)
            patterns = discovery_data["patterns"]
            
            minute_bins = np.array([p.get("batch_id", i) * 10 for i, p in enumerate(patterns)])
            densities = np.array([p["temporal_features"]["peak_intensity"] * 3 for p in patterns])
            scores = np.array([p["confidence"] * 100 for p in patterns])
            
            # Generate reports
            output_dir = temp_path / "reports"
            output_dir.mkdir(exist_ok=True)
            
            heatmap = build_session_heatmap(minute_bins, densities)
            confluence = build_confluence_strip(minute_bins, scores)
            
            heatmap_path = output_dir / "session_heatmap_mock_discoveries.png"
            confluence_path = output_dir / "confluence_strip_mock_discoveries.png"
            
            heatmap.save(heatmap_path, "PNG")
            confluence.save(confluence_path, "PNG")
            
            logger.info("Generated reports:")
            logger.info(f"  Heatmap: {heatmap_path} ({heatmap_path.stat().st_size} bytes)")
            logger.info(f"  Confluence: {confluence_path} ({confluence_path.stat().st_size} bytes)")
            
        except ImportError:
            logger.info("Wave 5 components not available for demonstration")
    
    return cli_examples


def example_5_batch_report_generation():
    """Example 5: Batch processing multiple discovery files."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Batch Report Generation")
    logger.info("=" * 60)
    
    try:
        from ironforge.reporting import (
            ConfluenceStripSpec,
            TimelineHeatmapSpec,
            build_confluence_strip,
            build_session_heatmap,
        )
    except ImportError as e:
        logger.warning(f"Could not import Wave 5 components: {e}")
        return None
    
    # Simulate multiple trading sessions
    sessions = [
        {"name": "ASIA_20250817", "patterns": 18, "duration_hours": 6},
        {"name": "LONDON_20250817", "patterns": 25, "duration_hours": 8}, 
        {"name": "NY_AM_20250817", "patterns": 32, "duration_hours": 4},
        {"name": "NY_PM_20250817", "patterns": 28, "duration_hours": 4}
    ]
    
    logger.info("Processing batch report generation for multiple sessions:")
    
    batch_results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        batch_output = temp_path / "batch_reports"
        batch_output.mkdir()
        
        for session in sessions:
            logger.info(f"\nProcessing {session['name']}...")
            
            # Generate discovery data for this session
            discovery_data = create_mock_discovery_data(
                n_patterns=session["patterns"],
                session_name=session["name"]
            )
            
            # Save discovery file
            discovery_file = temp_path / f"{session['name']}_discoveries.json"
            with open(discovery_file, 'w') as f:
                json.dump(discovery_data, f, indent=2)
            
            # Extract visualization data
            patterns = discovery_data["patterns"]
            minute_bins = np.array([p["temporal_features"].get("minute_offset", i * 10) 
                                  for i, p in enumerate(patterns)])
            densities = np.array([p["temporal_features"]["peak_intensity"] * 2.5 
                                for p in patterns])
            scores = np.array([p["confidence"] * 100 for p in patterns])
            
            # Find high-significance markers
            markers = []
            for i, pattern in enumerate(patterns):
                if (pattern["archaeological_significance"]["archaeological_value"] == "high_archaeological_value" 
                    and pattern["confidence"] > 0.8):
                    markers.append(minute_bins[i])
            marker_array = np.array(markers) if markers else None
            
            # Generate session-specific reports
            session_output = batch_output / session["name"]
            session_output.mkdir()
            
            # Heatmap
            heatmap_spec = TimelineHeatmapSpec(width=1024, height=160)
            heatmap = build_session_heatmap(minute_bins, densities, heatmap_spec)
            heatmap_path = session_output / f"heatmap_{session['name']}.png"
            heatmap.save(heatmap_path, "PNG")
            
            # Confluence strip
            confluence_spec = ConfluenceStripSpec(width=1024, height=54)
            confluence = build_confluence_strip(minute_bins, scores, marker_array, confluence_spec)
            confluence_path = session_output / f"confluence_{session['name']}.png"
            confluence.save(confluence_path, "PNG")
            
            # Session summary
            session_stats = {
                "session_name": session["name"],
                "total_patterns": len(patterns),
                "time_span_minutes": float(minute_bins.max() - minute_bins.min()) if len(minute_bins) > 0 else 0,
                "average_confidence": float(np.mean(scores)) / 100,
                "peak_density": float(np.max(densities)),
                "high_significance_markers": len(markers),
                "reports_generated": [
                    str(heatmap_path.relative_to(temp_path)),
                    str(confluence_path.relative_to(temp_path))
                ]
            }
            
            summary_path = session_output / f"summary_{session['name']}.json"
            with open(summary_path, 'w') as f:
                json.dump(session_stats, f, indent=2)
            
            batch_results[session["name"]] = session_stats
            
            logger.info(f"  Generated {len(patterns)} patterns over {session_stats['time_span_minutes']:.1f} minutes")
            logger.info(f"  Average confidence: {session_stats['average_confidence']:.3f}")
            logger.info(f"  High-significance markers: {session_stats['high_significance_markers']}")
        
        # Generate batch summary
        batch_summary = {
            "batch_timestamp": datetime.now().isoformat(),
            "total_sessions": len(sessions),
            "total_patterns": sum(result["total_patterns"] for result in batch_results.values()),
            "sessions": batch_results
        }
        
        batch_summary_path = batch_output / "batch_summary.json"
        with open(batch_summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        logger.info("\n📊 Batch Processing Summary:")
        logger.info(f"  Sessions processed: {len(sessions)}")
        logger.info(f"  Total patterns: {batch_summary['total_patterns']}")
        logger.info(f"  Reports directory: {batch_output}")
        logger.info(f"  Batch summary: {batch_summary_path}")
        
        # Show directory structure
        logger.info("\nGenerated directory structure:")
        for session_dir in batch_output.iterdir():
            if session_dir.is_dir():
                logger.info(f"  {session_dir.name}/")
                for report_file in sorted(session_dir.iterdir()):
                    size = report_file.stat().st_size
                    logger.info(f"    {report_file.name} ({size} bytes)")
    
    return batch_results


def example_6_production_monitoring_integration():
    """Example 6: Production monitoring dashboard integration."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Production Monitoring Integration")
    logger.info("=" * 60)
    
    try:
        from ironforge.reporting import (
            ConfluenceStripSpec,
            TimelineHeatmapSpec,
            build_confluence_strip,
            build_session_heatmap,
        )
    except ImportError as e:
        logger.warning(f"Could not import Wave 5 components: {e}")
        return None
    
    def generate_monitoring_dashboard_reports(discovery_file_path, output_dir):
        """Generate standardized reports for monitoring dashboard."""
        logger.info("🖥️  Generating monitoring dashboard reports...")
        logger.info(f"  Source: {discovery_file_path}")
        logger.info(f"  Output: {output_dir}")
        
        # Load discovery data
        with open(discovery_file_path) as f:
            discovery_data = json.load(f)
        
        patterns = discovery_data.get("patterns", [])
        if not patterns:
            logger.warning("No patterns found in discovery data")
            return {}
        
        # Extract data for monitoring visualizations
        minute_bins = np.array([p.get("batch_id", i) * 12 for i, p in enumerate(patterns)])  # 12-minute intervals
        densities = np.array([p["temporal_features"]["peak_intensity"] for p in patterns])
        scores = np.array([p["confidence"] * 100 for p in patterns])
        
        # Generate standardized monitoring reports
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        reports = {}
        
        # 1. Compact heatmap for dashboard header
        compact_heatmap_spec = TimelineHeatmapSpec(width=800, height=60, pad=2)
        compact_heatmap = build_session_heatmap(minute_bins, densities, compact_heatmap_spec)
        compact_path = output_path / "dashboard_heatmap_compact.png"
        compact_heatmap.save(compact_path, "PNG")
        reports["compact_heatmap"] = str(compact_path)
        
        # 2. Full-width confluence strip for main display
        fullwidth_confluence_spec = ConfluenceStripSpec(width=1200, height=40, pad=4, marker_radius=2)
        # Only show markers for very high confidence patterns
        high_conf_markers = minute_bins[scores > 85] if np.any(scores > 85) else None
        fullwidth_confluence = build_confluence_strip(
            minute_bins, scores, high_conf_markers, fullwidth_confluence_spec
        )
        confluence_path = output_path / "dashboard_confluence_fullwidth.png"
        fullwidth_confluence.save(confluence_path, "PNG")
        reports["fullwidth_confluence"] = str(confluence_path)
        
        # 3. Thumbnail versions for overview widgets
        thumb_heatmap_spec = TimelineHeatmapSpec(width=200, height=30, pad=1)
        thumb_heatmap = build_session_heatmap(minute_bins, densities, thumb_heatmap_spec)
        thumb_heatmap_path = output_path / "thumbnail_heatmap.png"
        thumb_heatmap.save(thumb_heatmap_path, "PNG")
        reports["thumbnail_heatmap"] = str(thumb_heatmap_path)
        
        thumb_confluence_spec = ConfluenceStripSpec(width=200, height=20, pad=1, marker_radius=1)
        thumb_confluence = build_confluence_strip(minute_bins, scores, None, thumb_confluence_spec)  # No markers in thumb
        thumb_confluence_path = output_path / "thumbnail_confluence.png"
        thumb_confluence.save(thumb_confluence_path, "PNG")
        reports["thumbnail_confluence"] = str(thumb_confluence_path)
        
        # Generate monitoring metadata
        monitoring_metadata = {
            "timestamp": datetime.now().isoformat(),
            "session_name": discovery_data.get("session_name", "unknown"),
            "total_patterns": len(patterns),
            "monitoring_stats": {
                "avg_confidence": float(np.mean(scores)),
                "max_confidence": float(np.max(scores)),
                "high_confidence_count": int(np.sum(scores > 80)),
                "peak_density": float(np.max(densities)),
                "time_span_minutes": float(minute_bins.max() - minute_bins.min()) if len(minute_bins) > 0 else 0
            },
            "reports": reports,
            "dashboard_urls": {
                "compact_heatmap": "/reports/dashboard_heatmap_compact.png",
                "fullwidth_confluence": "/reports/dashboard_confluence_fullwidth.png",
                "thumbnail_heatmap": "/reports/thumbnail_heatmap.png",
                "thumbnail_confluence": "/reports/thumbnail_confluence.png"
            }
        }
        
        metadata_path = output_path / "monitoring_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(monitoring_metadata, f, indent=2)
        
        logger.info("  ✅ Generated monitoring reports:")
        for report_name, report_path in reports.items():
            size = Path(report_path).stat().st_size
            logger.info(f"    {report_name}: {size} bytes")
        
        return monitoring_metadata
    
    # Simulate production monitoring workflow
    logger.info("Simulating production monitoring workflow...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create realistic production discovery data
        production_discovery = create_mock_discovery_data(
            n_patterns=45, 
            session_name="PRODUCTION_LIVE"
        )
        
        # Add some production-specific metadata
        production_discovery.update({
            "deployment_version": "ironforge_v2.1.0",
            "data_source": "live_market_feed",
            "processing_latency_ms": 2347,
            "system_load": 0.67
        })
        
        discovery_file = temp_path / "production_live_discovery.json"
        with open(discovery_file, 'w') as f:
            json.dump(production_discovery, f, indent=2)
        
        # Generate monitoring dashboard reports
        monitoring_output = temp_path / "monitoring_reports"
        monitoring_metadata = generate_monitoring_dashboard_reports(discovery_file, monitoring_output)
        
        # Simulate dashboard integration
        logger.info("\n🖥️  Monitoring Dashboard Integration:")
        logger.info(f"  Session: {monitoring_metadata['session_name']}")
        logger.info(f"  Patterns: {monitoring_metadata['total_patterns']}")
        logger.info(f"  Avg Confidence: {monitoring_metadata['monitoring_stats']['avg_confidence']:.1f}%")
        logger.info(f"  High Confidence Patterns: {monitoring_metadata['monitoring_stats']['high_confidence_count']}")
        logger.info(f"  Peak Activity: {monitoring_metadata['monitoring_stats']['peak_density']:.2f}")
        
        logger.info("\n📊 Dashboard Assets:")
        for asset_name, asset_url in monitoring_metadata["dashboard_urls"].items():
            logger.info(f"  {asset_name}: {asset_url}")
        
        # Show file sizes for bandwidth planning
        logger.info("\n💾 Asset Sizes (for bandwidth planning):")
        total_size = 0
        for report_path in monitoring_metadata["reports"].values():
            size = Path(report_path).stat().st_size
            total_size += size
            logger.info(f"  {Path(report_path).name}: {size:,} bytes")
        logger.info(f"  Total: {total_size:,} bytes")
    
    return monitoring_metadata


def main():
    """Run all Wave 5 reporting examples."""
    logger.info("IRONFORGE Wave 5 Reporting Examples")
    logger.info("=" * 80)
    logger.info("Comprehensive demonstration of visualization and reporting capabilities")
    logger.info("for temporal pattern discovery results.")
    logger.info("=" * 80)
    
    # Track execution time
    import time
    start_time = time.time()
    
    # Run all examples
    examples = [
        example_1_basic_heatmap_generation,
        example_2_confluence_strip_with_markers,
        example_3_discovery_data_integration,
        example_4_cli_equivalent_usage,
        example_5_batch_report_generation,
        example_6_production_monitoring_integration
    ]
    
    example_results = {}
    
    for example_func in examples:
        try:
            logger.info(f"\n🚀 Starting {example_func.__name__}...")
            result = example_func()
            example_results[example_func.__name__] = result
            logger.info(f"✅ Completed {example_func.__name__}")
        except Exception as e:
            logger.error(f"❌ {example_func.__name__} failed: {e}")
            example_results[example_func.__name__] = None
    
    # Summary
    elapsed_time = time.time() - start_time
    successful_examples = sum(1 for result in example_results.values() if result is not None)
    
    logger.info("\n" + "=" * 80)
    logger.info("Wave 5 Reporting Examples Summary")
    logger.info("=" * 80)
    logger.info(f"Examples completed: {successful_examples}/{len(examples)}")
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    logger.info(f"Performance budget: {'✅ PASSED' if elapsed_time < 15.0 else '❌ EXCEEDED'} (<15s)")
    
    logger.info("\nExample Status:")
    for example_name, result in example_results.items():
        status = "✅ SUCCESS" if result is not None else "❌ FAILED"
        logger.info(f"  {example_name}: {status}")
    
    logger.info("\nWave 5 reporting system ready for production use! 🎨")
    logger.info("\nKey capabilities demonstrated:")
    logger.info("  📈 Timeline heatmap generation for session density visualization")
    logger.info("  🌊 Confluence strip generation with event markers")
    logger.info("  🔗 Integration with discovery pipeline outputs")
    logger.info("  🖥️  Production monitoring dashboard assets")
    logger.info("  📊 Batch processing workflows")
    logger.info("  🛠️  CLI command equivalents")
    
    return example_results


if __name__ == "__main__":
    # Run examples when script is executed directly
    results = main()
