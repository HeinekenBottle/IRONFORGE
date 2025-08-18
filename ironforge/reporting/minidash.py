from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_aux_timing_data(run_dir: Path) -> dict[str, Any]:
    """Load auxiliary timing data for minidash overlay"""
    timing_dir = run_dir / "aux" / "timing"
    
    if not timing_dir.exists():
        return {"bursts": [], "node_annotations": {}}
    
    try:
        # Load burst summaries
        summary_path = timing_dir / "summary.parquet"
        bursts = []
        if summary_path.exists():
            summary_df = pd.read_parquet(summary_path)
            bursts = summary_df.to_dict('records')
        
        # Load node annotations  
        annotations_path = timing_dir / "node_annotations.parquet"
        node_annotations = {}
        if annotations_path.exists():
            annotations_df = pd.read_parquet(annotations_path)
            node_annotations = annotations_df.set_index('node_id').to_dict('index')
        
        return {
            "bursts": bursts,
            "node_annotations": node_annotations,
            "total_bursts": len(bursts),
            "total_annotated_nodes": len(node_annotations)
        }
    except Exception:
        return {"bursts": [], "node_annotations": {}}


def load_aux_trader_data(run_dir: Path) -> dict[str, Any]:
    """Load trader-relevant AUX data for minidash panels"""
    aux_dir = run_dir / "aux"
    result = {
        "trajectories": None,
        "phase_stats": None, 
        "chains": None
    }
    
    if not aux_dir.exists():
        return result
    
    try:
        # Load trajectories
        traj_path = aux_dir / "trajectories.parquet"
        if traj_path.exists():
            traj_df = pd.read_parquet(traj_path)
            result["trajectories"] = {
                "zones_count": len(traj_df),
                "populated_rate": traj_df[['fwd_ret_3b', 'fwd_ret_12b', 'fwd_ret_24b']].notna().any(axis=1).mean(),
                "hit_rates": {
                    "50_ticks": traj_df['hit_+50_12b'].mean() if 'hit_+50_12b' in traj_df.columns else 0,
                    "100_ticks": traj_df['hit_+100_12b'].mean() if 'hit_+100_12b' in traj_df.columns else 0,
                    "200_ticks": traj_df['hit_+200_12b'].mean() if 'hit_+200_12b' in traj_df.columns else 0
                },
                "avg_returns": {
                    "3b": traj_df['fwd_ret_3b'].mean() if 'fwd_ret_3b' in traj_df.columns else 0,
                    "12b": traj_df['fwd_ret_12b'].mean() if 'fwd_ret_12b' in traj_df.columns else 0,
                    "24b": traj_df['fwd_ret_24b'].mean() if 'fwd_ret_24b' in traj_df.columns else 0
                }
            }
        
        # Load phase stats
        phase_path = aux_dir / "phase_stats.json"
        if phase_path.exists():
            with open(phase_path) as f:
                phase_data = json.load(f)
            
            # Find best performing buckets
            best_buckets = []
            for bucket_name, bucket_data in phase_data.items():
                hit_100 = bucket_data.get('P_hit_+100_12b', 0)
                count = bucket_data.get('count', 0)
                if count >= 2:  # Only buckets with reasonable sample size
                    best_buckets.append((bucket_name, hit_100, count))
            
            best_buckets.sort(key=lambda x: x[1], reverse=True)
            
            result["phase_stats"] = {
                "total_buckets": len(phase_data),
                "valid_buckets": len([b for b in phase_data.values() if b.get('count', 0) >= 2]),
                "best_buckets": best_buckets[:3]  # Top 3
            }
        
        # Load chains
        chains_path = aux_dir / "chains.parquet"
        if chains_path.exists():
            chains_df = pd.read_parquet(chains_path)
            
            # Chain statistics
            valid_returns = chains_df['subsequent_ret_12b'].dropna() if 'subsequent_ret_12b' in chains_df.columns else pd.Series([])
            
            result["chains"] = {
                "total_chains": len(chains_df),
                "chain_types": chains_df['chain'].value_counts().to_dict() if 'chain' in chains_df.columns else {},
                "avg_span_bars": chains_df['span_bars'].mean() if 'span_bars' in chains_df.columns else 0,
                "avg_span_minutes": chains_df['span_minutes'].mean() if 'span_minutes' in chains_df.columns else 0,
                "subsequent_returns": {
                    "mean": valid_returns.mean() if len(valid_returns) > 0 else 0,
                    "std": valid_returns.std() if len(valid_returns) > 0 else 0,
                    "count": len(valid_returns)
                }
            }
        
    except Exception as e:
        # Graceful fallback on any errors
        pass
    
    return result


def build_minidash(
    activity: pd.DataFrame,
    confluence: pd.DataFrame,
    motifs: list[dict[str, Any]],
    out_html: str | Path,
    out_png: str | Path,
    width: int = 1200,
    height: int = 700,
    htf_regime_data: dict[str, Any] | None = None,
    run_dir: Path | None = None,
) -> tuple[Path, Path]:
    if activity.empty:
        activity = pd.DataFrame(
            {
                "ts": pd.date_range("2025-01-01", periods=10, freq="T"),
                "count": list(range(10)),
            }
        )
    if confluence.empty:
        confluence = pd.DataFrame(
            {
                "ts": activity["ts"],
                "score": [min(99, i * 10) for i in range(len(activity))],
            }
        )

    activity = activity.sort_values("ts")
    confluence = confluence.sort_values("ts")

    fig = plt.figure(figsize=(width / 100, height / 100))
    ax1 = fig.add_axes([0.08, 0.58, 0.9, 0.35])
    ax2 = fig.add_axes([0.08, 0.12, 0.9, 0.35])

    ax1.bar(range(len(activity)), activity["count"], align="center")
    ax1.set_title("Session Activity")
    ax1.set_xticks([])
    ax1.set_ylabel("Count")

    ax2.plot(range(len(confluence)), confluence["score"])
    ax2.set_ylim(0, 100)
    ax2.set_title("Confluence (0–100)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Score")

    out_png = Path(out_png)
    out_html = Path(out_html)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Check for archetype cards
    cards_index_path = Path(run_dir) / "motifs" / "cards_index.csv" if run_dir else None
    archetype_cards = {}
    
    if cards_index_path and cards_index_path.exists():
        try:
            cards_df = pd.read_csv(cards_index_path)
            for _, row in cards_df.iterrows():
                zone_id = row['zone_id']
                archetype_cards[zone_id] = {
                    'score': row.get('score', 0),
                    'chain_tag': row.get('chain_tag', 'none'),
                    'phase_bucket': row.get('phase_bucket', 'unknown')
                }
        except:
            pass
    
    # Build motifs table with archetype links
    rows = []
    for m in motifs:
        name = m.get('name', '')
        support = m.get('support', '')
        ppv = m.get('ppv', '')
        
        # Add archetype info if available
        archetype_info = ""
        if name in archetype_cards:
            card = archetype_cards[name]
            score = card['score']
            chain_tag = card['chain_tag']
            archetype_info = f" <small style='color:#666'>[Arch: {score:.3f}, {chain_tag}]</small>"
        
        rows.append(f"<tr><td>{name}{archetype_info}</td><td>{support}</td><td>{ppv}</td></tr>")
    
    rows = "".join(rows)
    
    # Load auxiliary timing data
    timing_data = load_aux_timing_data(Path(run_dir)) if run_dir else {"bursts": [], "node_annotations": {}}
    
    # Load trader-relevant AUX data
    trader_data = load_aux_trader_data(Path(run_dir)) if run_dir else {"trajectories": None, "phase_stats": None, "chains": None}
    
    # Load confluence scale information and health status
    scale_badge = ""
    health_badge = ""
    if run_dir:
        stats_path = Path(run_dir) / "confluence" / "stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                scale = stats.get("scale", "")
                threshold = stats.get("threshold", None)
                health_status = stats.get("health_status", "unknown")
                coverage = stats.get("coverage", 0.0)
                variance = stats.get("variance", 0.0)
                
                # Scale badge (blue)
                if scale:
                    scale_badge = f'<span style="background:#007bff;color:white;padding:2px 6px;border-radius:3px;margin-left:10px">Score scale: {scale}'
                    if threshold is not None:
                        scale_badge += f' • threshold={threshold}'
                    scale_badge += '</span>'
                
                # Health gate badge (green/red)
                if health_status == "pass":
                    health_color = "#28a745"  # green
                    health_text = f"Health: ✓ Pass (cov={coverage:.1%}, var={variance:.2e})"
                elif health_status == "fail":
                    health_color = "#dc3545"  # red
                    health_text = f"Health: ✗ Fail (cov={coverage:.1%}, var={variance:.2e})"
                else:
                    health_color = "#ffc107"  # yellow
                    health_text = f"Health: ? Unknown"
                
                health_badge = f'<span style="background:{health_color};color:white;padding:2px 6px;border-radius:3px;margin-left:10px">{health_text}</span>'
                
            except Exception:
                pass  # Ignore errors reading stats
    
    # HTF Regime Ribbon (minimal text badges)
    htf_ribbon = ""
    if htf_regime_data:
        regime_dist = htf_regime_data.get('regime_distribution', {})
        total_zones = htf_regime_data.get('total_zones', 0)
        theory_b_zones = htf_regime_data.get('theory_b_zones', 0)
        quality_score = htf_regime_data.get('quality_score', 0.0)
        
        # Create regime badges
        regime_badges = []
        regime_colors = {'consolidation': '#ffc107', 'transition': '#17a2b8', 'expansion': '#dc3545'}
        
        for regime, count in regime_dist.items():
            color = regime_colors.get(regime, '#6c757d')
            badge = f'<span style="background:{color};color:white;padding:2px 6px;border-radius:3px;margin:2px">{regime.title()}: {count}</span>'
            regime_badges.append(badge)
        
        htf_ribbon = f'''
        <div style="background:#f8f9fa;padding:10px;margin:10px 0;border-radius:5px;">
            <strong>HTF Context (v0.7.1):</strong> 
            {' '.join(regime_badges)}
            <br><small>
                Zones: {total_zones} | Theory B: {theory_b_zones} | Quality: {quality_score:.2f}
            </small>
        </div>'''
    
    # AUX Timing Panel
    timing_panel = ""
    if timing_data["bursts"]:
        burst_items = []
        for i, burst in enumerate(timing_data["bursts"][:5]):  # Show first 5 bursts
            start_ts = burst.get('burst_start_ts', 0)
            end_ts = burst.get('burst_end_ts', 0)
            events_count = burst.get('events_in_burst', 0)
            gap_mean = burst.get('gap_s_mean', 0)
            
            # Convert timestamps to readable format (simplified)
            duration_s = (end_ts - start_ts) / 1000 if end_ts > start_ts else 0
            
            burst_items.append(f'''
                <span style="background:#28a745;color:white;padding:2px 6px;border-radius:3px;margin:2px">
                    Burst {i}: {events_count} events, {duration_s:.1f}s duration, {gap_mean:.1f}s avg gap
                </span>
            ''')
        
        timing_panel = f'''
        <div style="background:#f8f9fa;padding:10px;margin:10px 0;border-radius:5px;">
            <strong>AUX: Timing Analysis</strong><br>
            <small>Total bursts: {timing_data["total_bursts"]} | Annotated nodes: {timing_data["total_annotated_nodes"]}</small><br>
            {''.join(burst_items)}
            <br><small><em>Burst = 3+ events within 2 minutes</em></small>
        </div>'''
    
    # AUX Trader Panels
    trajectories_panel = ""
    if trader_data["trajectories"]:
        traj = trader_data["trajectories"]
        hit_badges = []
        for target, rate in traj["hit_rates"].items():
            color = "#28a745" if rate > 0.3 else "#ffc107" if rate > 0.1 else "#6c757d"
            hit_badges.append(f'<span style="background:{color};color:white;padding:2px 6px;border-radius:3px;margin:2px">{target}: {rate:.1%}</span>')
        
        trajectories_panel = f'''
        <div style="background:#e7f3ff;padding:10px;margin:10px 0;border-radius:5px;">
            <strong>AUX: Post-Zone Trajectories</strong><br>
            <small>Zones: {traj["zones_count"]} | Populated: {traj["populated_rate"]:.1%}</small><br>
            <div>Hit Rates (12b): {''.join(hit_badges)}</div>
            <small>Avg Returns: 3b={traj["avg_returns"]["3b"]:.2f}% | 12b={traj["avg_returns"]["12b"]:.2f}% | 24b={traj["avg_returns"]["24b"]:.2f}%</small>
        </div>'''
    
    phase_panel = ""
    if trader_data["phase_stats"]:
        phase = trader_data["phase_stats"]
        bucket_badges = []
        for bucket_name, hit_rate, count in phase["best_buckets"]:
            color = "#28a745" if hit_rate > 0.3 else "#ffc107" if hit_rate > 0.1 else "#6c757d"
            bucket_badges.append(f'<span style="background:{color};color:white;padding:2px 6px;border-radius:3px;margin:2px">{bucket_name}: {hit_rate:.1%} (n={count})</span>')
        
        phase_panel = f'''
        <div style="background:#fff2e7;padding:10px;margin:10px 0;border-radius:5px;">
            <strong>AUX: HTF Phase Stratification</strong><br>
            <small>Buckets: {phase["total_buckets"]} total | {phase["valid_buckets"]} valid (n≥2)</small><br>
            <div>Best P(hit_+100_12b): {''.join(bucket_badges) if bucket_badges else '<em>No valid buckets</em>'}</div>
        </div>'''
    
    chains_panel = ""
    if trader_data["chains"]:
        chains = trader_data["chains"]
        chain_badges = []
        for chain_type, count in list(chains["chain_types"].items())[:3]:  # Top 3 chain types
            chain_badges.append(f'<span style="background:#17a2b8;color:white;padding:2px 6px;border-radius:3px;margin:2px">{chain_type}: {count}</span>')
        
        ret_stats = chains["subsequent_returns"]
        ret_color = "#28a745" if ret_stats["mean"] > 0 else "#dc3545"
        
        chains_panel = f'''
        <div style="background:#f0fff0;padding:10px;margin:10px 0;border-radius:5px;">
            <strong>AUX: Event Chains</strong><br>
            <small>Chains: {chains["total_chains"]} | Avg span: {chains["avg_span_bars"]:.1f} bars ({chains["avg_span_minutes"]:.0f} min)</small><br>
            <div>Types: {''.join(chain_badges) if chain_badges else '<em>No chains</em>'}</div>
            <div>Subsequent returns: <span style="background:{ret_color};color:white;padding:2px 6px;border-radius:3px;">μ={ret_stats["mean"]:.2f}% σ={ret_stats["std"]:.2f}% (n={ret_stats["count"]})</span></div>
        </div>'''
    
    html = f"""<!doctype html><meta charset="utf-8">
    <style>body{{font:14px system-ui}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:6px}}</style>
    <h1>IRONFORGE — Minimal Report{scale_badge}{health_badge}</h1>
    {htf_ribbon}
    {timing_panel}
    {trajectories_panel}
    {phase_panel}
    {chains_panel}
    <img src="{out_png.name}" alt="Confluence & Activity" />
    <h2>Motifs</h2><table><tr><th>Name</th><th>Support</th><th>PPV</th></tr>{rows}</table>"""
    out_html.write_text(html, encoding="utf-8")
    return out_html, out_png
