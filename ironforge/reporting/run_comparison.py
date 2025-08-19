"""
Run Comparison System for IRONFORGE
Tracks KPIs across runs and generates comparison reports
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def append_run_to_index(
    run_dir: str | Path,
    zones_total: int = 0,
    mean_confidence: float = 0.0,
    avg_out_degree: float = 0.0,
    bursts_total: int = 0,
    runtime_s: float = 0.0,
    additional_kpis: dict[str, Any] = None
) -> str:
    """
    Append run KPIs to repository-level index
    
    Args:
        run_dir: Path to the run directory
        zones_total: Total number of zones processed
        mean_confidence: Average confluence score
        avg_out_degree: Average node out-degree
        bursts_total: Total number of event bursts
        runtime_s: Total runtime in seconds
        additional_kpis: Additional KPIs to include
        
    Returns:
        Path to the updated index.csv file
    """
    try:
        run_dir = Path(run_dir)
        run_name = run_dir.name
        
        # Find repository root (look for runs directory)
        repo_root = run_dir.parent
        while repo_root.name != "runs" and repo_root.parent != repo_root:
            repo_root = repo_root.parent
        
        if repo_root.name == "runs":
            repo_root = repo_root.parent
        
        index_path = repo_root / "runs" / "index.csv"
        
        # Create new record
        new_record = {
            "run_dir": run_name,
            "timestamp": time.time(),
            "zones_total": zones_total,
            "mean_confidence": mean_confidence,
            "avg_out_degree": avg_out_degree,
            "bursts_total": bursts_total,
            "runtime_s": runtime_s
        }
        
        # Add additional KPIs if provided
        if additional_kpis:
            new_record.update(additional_kpis)
        
        # Load existing index or create new
        if index_path.exists():
            try:
                index_df = pd.read_csv(index_path)
            except Exception as e:
                logger.warning(f"Failed to read existing index: {e}, creating new")
                index_df = pd.DataFrame()
        else:
            index_df = pd.DataFrame()
        
        # Append new record
        new_df = pd.DataFrame([new_record])
        index_df = pd.concat([index_df, new_df], ignore_index=True)
        
        # Keep only last 50 runs to prevent unlimited growth
        if len(index_df) > 50:
            index_df = index_df.tail(50)
        
        # Ensure directory exists and save
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(index_path, index=False)
        
        logger.info(f"Appended run {run_name} to index: {len(index_df)} total runs")
        return str(index_path)
        
    except Exception as e:
        logger.error(f"Failed to append run to index: {e}")
        return ""


def generate_run_comparison(runs_dir: str | Path = None, limit: int = 10) -> str:
    """
    Generate comparison HTML page showing KPI trends across recent runs
    
    Args:
        runs_dir: Path to runs directory (defaults to ./runs)
        limit: Maximum number of recent runs to show
        
    Returns:
        Path to generated compare.html file
    """
    try:
        runs_dir = Path("runs") if runs_dir is None else Path(runs_dir)
        
        index_path = runs_dir / "index.csv"
        compare_path = runs_dir / "compare.html"
        
        # Also check parent directory for index (in case runs_dir is a specific run directory)
        if not index_path.exists() and runs_dir.name != "runs":
            parent_index = runs_dir.parent / "index.csv"
            if parent_index.exists():
                index_path = parent_index
                compare_path = runs_dir.parent / "compare.html"
        
        if not index_path.exists():
            logger.warning("No index.csv found, creating minimal comparison page")
            minimal_html = """<!doctype html><meta charset="utf-8">
            <style>body{font:14px system-ui}</style>
            <h1>IRONFORGE Run Comparison</h1>
            <p>No run history available yet. Run some sessions to see trends here.</p>"""
            compare_path.write_text(minimal_html, encoding="utf-8")
            return str(compare_path)
        
        # Load run index
        index_df = pd.read_csv(index_path)
        
        # Get recent runs (limit to specified number)
        recent_runs = index_df.tail(limit)
        
        if len(recent_runs) == 0:
            logger.warning("No runs in index, creating minimal comparison page")
            minimal_html = """<!doctype html><meta charset="utf-8">
            <style>body{font:14px system-ui}</style>
            <h1>IRONFORGE Run Comparison</h1>
            <p>Index file exists but contains no runs.</p>"""
            compare_path.write_text(minimal_html, encoding="utf-8")
            return str(compare_path)
        
        # Generate sparkline data (simplified as text for now)
        kpi_summaries = []
        
        # Zones Total sparkline
        zones_values = recent_runs['zones_total'].tolist()
        zones_trend = "↗" if len(zones_values) > 1 and zones_values[-1] > zones_values[0] else "→"
        kpi_summaries.append(f"Zones: {zones_values[-1]:.0f} {zones_trend} (avg: {sum(zones_values)/len(zones_values):.1f})")
        
        # Mean Confidence sparkline
        conf_values = recent_runs['mean_confidence'].tolist()
        conf_trend = "↗" if len(conf_values) > 1 and conf_values[-1] > conf_values[0] else "→"
        kpi_summaries.append(f"Confidence: {conf_values[-1]:.3f} {conf_trend} (avg: {sum(conf_values)/len(conf_values):.3f})")
        
        # Runtime sparkline
        runtime_values = recent_runs['runtime_s'].tolist()
        runtime_trend = "↘" if len(runtime_values) > 1 and runtime_values[-1] < runtime_values[0] else "→"
        kpi_summaries.append(f"Runtime: {runtime_values[-1]:.1f}s {runtime_trend} (avg: {sum(runtime_values)/len(runtime_values):.1f}s)")
        
        # Create run table rows
        table_rows = []
        for _, run in recent_runs.iterrows():
            timestamp_str = pd.to_datetime(run['timestamp'], unit='s').strftime('%m-%d %H:%M')
            table_rows.append(f"""
                <tr>
                    <td>{run['run_dir']}</td>
                    <td>{timestamp_str}</td>
                    <td>{run['zones_total']:.0f}</td>
                    <td>{run['mean_confidence']:.3f}</td>
                    <td>{run.get('avg_out_degree', 0):.1f}</td>
                    <td>{run.get('bursts_total', 0):.0f}</td>
                    <td>{run['runtime_s']:.1f}s</td>
                </tr>
            """)
        
        # Generate HTML
        html = f"""<!doctype html><meta charset="utf-8">
        <style>
            body {{font:14px system-ui; margin:20px}}
            table {{border-collapse:collapse; width:100%}}
            td,th {{border:1px solid #ccc; padding:8px; text-align:left}}
            th {{background:#f5f5f5}}
            .kpi-summary {{background:#e3f2fd; padding:15px; margin:10px 0; border-radius:5px}}
            .kpi-item {{display:inline-block; margin:5px 15px 5px 0; padding:5px 10px; background:white; border-radius:3px}}
        </style>
        <h1>IRONFORGE — Run Comparison</h1>
        
        <div class="kpi-summary">
            <strong>Recent Trends ({len(recent_runs)} runs):</strong><br>
            {''.join(f'<span class="kpi-item">{kpi}</span>' for kpi in kpi_summaries)}
        </div>
        
        <h2>Recent Runs</h2>
        <table>
            <tr>
                <th>Run</th>
                <th>Time</th>
                <th>Zones</th>
                <th>Confidence</th>
                <th>Avg Degree</th>
                <th>Bursts</th>
                <th>Runtime</th>
            </tr>
            {''.join(table_rows)}
        </table>
        
        <p><small>Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        """
        
        compare_path.write_text(html, encoding="utf-8")
        logger.info(f"Generated comparison for {len(recent_runs)} runs -> {compare_path}")
        return str(compare_path)
        
    except Exception as e:
        logger.error(f"Failed to generate run comparison: {e}")
        return ""


def extract_run_kpis(run_dir: str | Path) -> dict[str, Any]:
    """
    Extract KPIs from a completed run directory
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary of extracted KPIs
    """
    try:
        run_dir = Path(run_dir)
        kpis = {
            "zones_total": 0,
            "mean_confidence": 0.0,
            "avg_out_degree": 0.0,
            "bursts_total": 0,
            "runtime_s": 0.0
        }
        
        # Extract from confluence stats
        confluence_stats_path = run_dir / "confluence" / "stats.json"
        if confluence_stats_path.exists():
            try:
                with open(confluence_stats_path) as f:
                    stats = json.load(f)
                kpis["mean_confidence"] = stats.get("mean", 0.0)
            except Exception as e:
                logger.debug(f"Failed to read confluence stats: {e}")
        
        # Extract from confluence scores
        confluence_scores_path = run_dir / "confluence" / "confluence_scores.parquet"
        if confluence_scores_path.exists():
            try:
                scores_df = pd.read_parquet(confluence_scores_path)
                kpis["zones_total"] = len(scores_df)
            except Exception as e:
                logger.debug(f"Failed to read confluence scores: {e}")
        
        # Extract from timing data
        timing_summary_path = run_dir / "aux" / "timing" / "summary.parquet"
        if timing_summary_path.exists():
            try:
                timing_df = pd.read_parquet(timing_summary_path)
                kpis["bursts_total"] = len(timing_df)
            except Exception as e:
                logger.debug(f"Failed to read timing summary: {e}")
        
        # Estimate runtime from directory timestamps (fallback)
        try:
            dir_stat = run_dir.stat()
            kpis["runtime_s"] = time.time() - dir_stat.st_mtime
        except Exception:
            kpis["runtime_s"] = 0.0
        
        logger.debug(f"Extracted KPIs from {run_dir.name}: {kpis}")
        return kpis
        
    except Exception as e:
        logger.error(f"Failed to extract KPIs from {run_dir}: {e}")
        return {
            "zones_total": 0,
            "mean_confidence": 0.0,
            "avg_out_degree": 0.0,
            "bursts_total": 0,
            "runtime_s": 0.0
        }


def auto_index_run(run_dir: str | Path) -> bool:
    """
    Automatically extract KPIs and append to index for a completed run
    
    Args:
        run_dir: Path to completed run directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        kpis = extract_run_kpis(run_dir)
        index_path = append_run_to_index(
            run_dir=run_dir,
            zones_total=kpis["zones_total"],
            mean_confidence=kpis["mean_confidence"],
            avg_out_degree=kpis["avg_out_degree"],
            bursts_total=kpis["bursts_total"],
            runtime_s=kpis["runtime_s"]
        )
        
        if index_path:
            # Also generate updated comparison
            compare_path = generate_run_comparison()
            logger.info(f"Auto-indexed run and updated comparison: {compare_path}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to auto-index run {run_dir}: {e}")
        return False