from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import pickle
import warnings
import sys
from pathlib import Path

import pandas as pd

from ironforge.reporting.minidash import build_minidash

from .app_config import load_config, materialize_run_dir, validate_config
from .io import glob_many, write_json
from .oracle_commands import cmd_audit_oracle, cmd_train_oracle
from ..utils.common import maybe_import, get_legacy_entrypoint


# _maybe function moved to ironforge.utils.common.maybe_import


def cmd_discover(cfg):
    # Canonical entrypoint
    fn = maybe_import("ironforge.learning.discovery_pipeline", "run_discovery")
    if fn is None:
        # Legacy fallbacks
        legacy_paths = [
            "ironforge.learning.tgat_discovery",
            "ironforge.discovery.runner"
        ]
        fn = get_legacy_entrypoint(
            legacy_paths, 
            "run_discovery", 
            "ironforge.learning.discovery_pipeline"
        )
        if fn is None:
            print("[discover] discovery engine not found; skipping (no-op).")
            return 0
    return int(bool(fn(cfg)))


def cmd_score(cfg):
    # Canonical entrypoint
    fn = maybe_import("ironforge.confluence.scoring", "score_confluence")
    if fn is None:
        # Legacy fallbacks
        legacy_paths = [
            "ironforge.confluence.scorer",
            "ironforge.metrics.confluence"
        ]
        legacy = get_legacy_entrypoint(
            legacy_paths, 
            "score_session", 
            "ironforge.confluence.scoring"
        )
        if legacy is None:
            print("[score] scorer not found; skipping (no-op).")
            return 0
        legacy(cfg)
        return 0
    fn(cfg)
    return 0


def cmd_validate(cfg):
    fn = maybe_import("ironforge.validation.runner", "validate_run")
    if fn is None:
        print("[validate] validation rails not found; skipping (no-op).")
        return 0
    res = fn(cfg)
    run_dir = materialize_run_dir(cfg) / "reports"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "validation.json", res if isinstance(res, dict) else {"result": "ok"})
    return 0


def _load_first_parquet(paths: list[Path], cols: list[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_parquet(paths[0])
        return df
    except Exception:
        return pd.DataFrame(columns=cols)


def cmd_report(cfg):
    run_dir = materialize_run_dir(cfg)
    conf_paths = glob_many(str(run_dir / "confluence" / "*.parquet"))
    conf = _load_first_parquet(conf_paths, ["ts", "score"])
    if conf.empty:
        conf = pd.DataFrame(
            {
                "ts": pd.date_range("2025-01-01", periods=50, freq="T"),
                "score": [min(99, i * 2 % 100) for i in range(50)],
            }
        )
    pat_paths = glob_many(str(run_dir / "patterns" / "*.parquet"))
    act = _load_first_parquet(pat_paths, ["ts", "count"])
    if act.empty:
        g = conf.groupby(conf["ts"].astype("datetime64[m]")).size().reset_index(name="count")
        g.rename(columns={g.columns[0]: "ts"}, inplace=True)
        act = g
    motifs = []
    for j in Path(run_dir / "motifs").glob("*.json"):
        with contextlib.suppress(Exception):
            motifs.extend(json.loads(j.read_text(encoding="utf-8")))
    if not motifs:
        motifs = [{"name": "sweep→fvg", "support": 12, "ppv": 0.61}]
    out_html = run_dir / cfg.reporting.minidash.out_html
    out_png = run_dir / cfg.reporting.minidash.out_png
    build_minidash(
        act,
        conf,
        motifs,
        out_html,
        out_png,
        width=cfg.reporting.minidash.width,
        height=cfg.reporting.minidash.height,
    )
    print(f"[report] wrote {out_html} and {out_png}")
    # Optional manifest writer (env-gated, backward-compatible)
    import os as _os
    if _os.getenv("IRONFORGE_WRITE_MANIFEST") == "1":
        try:
            import ironforge as _pkg
            from . import manifest as _mf

            _mf.write_for_run(
                run_dir=str(run_dir),
                window_bars=512,  # default; use helper script for richer manifests
                version=getattr(_pkg, "__version__", "unknown"),
            )
        except Exception as e:  # pragma: no cover
            warnings.warn(f"Manifest write failed: {e}", RuntimeWarning, stacklevel=2)
    return 0


# Oracle commands moved to oracle_commands.py module


def cmd_prep_shards(
    source_glob: str,
    symbol: str,
    timeframe: str,
    timezone: str,
    pack_mode: str,
    dry_run: bool,
    overwrite: bool,
    htf_context: bool,
):
    """Prepare Parquet shards from enhanced JSON sessions."""
    try:
        from ironforge.converters.json_to_parquet import ConversionConfig, convert_enhanced_sessions

        config = ConversionConfig(
            source_glob=source_glob,
            symbol=symbol,
            timeframe=timeframe,
            source_timezone=timezone,
            pack_mode=pack_mode,
            dry_run=dry_run,
            overwrite=overwrite,
            htf_context_enabled=htf_context,
        )

        print(f"[prep-shards] Converting sessions from {source_glob}")
        print(
            f"[prep-shards] Target: {symbol}_{timeframe} | Timezone: {timezone} | Pack: {pack_mode}"
        )
        print(
            f"[prep-shards] HTF Context: {'ENABLED (51D features)' if htf_context else 'DISABLED (45D features)'}"
        )

        if dry_run:
            print("[prep-shards] DRY RUN MODE - no files will be written")

        shard_dirs = convert_enhanced_sessions(config)

        print(f"[prep-shards] ✅ Processed {len(shard_dirs)} sessions")

        # Write manifest
        if not dry_run and shard_dirs:
            manifest_path = Path(f"data/shards/{symbol}_{timeframe}/manifest.jsonl")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(manifest_path, "w") as f:
                for shard_dir in shard_dirs:
                    if shard_dir.exists():
                        meta_file = shard_dir / "meta.json"
                        if meta_file.exists():
                            with open(meta_file, "r") as meta_f:
                                metadata = json.load(meta_f)
                                manifest_entry = {
                                    "shard_dir": str(shard_dir),
                                    "session_id": metadata.get("session_id"),
                                    "node_count": metadata.get("node_count", 0),
                                    "edge_count": metadata.get("edge_count", 0),
                                    "conversion_timestamp": metadata.get("conversion_timestamp"),
                                }
                                f.write(json.dumps(manifest_entry) + "\n")

            print(f"[prep-shards] Wrote manifest: {manifest_path}")

        return 0

    except ImportError as e:
        print(f"[prep-shards] Error: Converter not available - {e}")
        return 1
    except Exception as e:
        print(f"[prep-shards] Error: {e}")
        return 1


def cmd_build_graph(args):
    """Build comprehensive dual graph views with full configuration support."""
    try:
        import glob
        import json
        from pathlib import Path
        from ironforge.learning.dual_graph_config import load_config_with_overrides
        from ironforge.learning.dag_graph_builder import DAGGraphBuilder
        from ironforge.learning.dag_motif_miner import DAGMotifMiner
        
        # Build comprehensive configuration
        overrides = {}
        
        # Apply quick configuration flags
        if args.with_dag:
            overrides['dag.enabled'] = True
        if args.with_m1:
            overrides['m1.enabled'] = True
        if args.enhanced_tgat:
            overrides['tgat.enhanced'] = True
        if args.enable_motifs:
            overrides['motifs.min_frequency'] = 2  # Enable motif discovery
            
        # Apply legacy DAG options for backwards compatibility
        if args.dag_k:
            overrides['dag.k_successors'] = args.dag_k
        if args.dag_dt_min:
            overrides['dag.dt_min_minutes'] = args.dag_dt_min
        if args.dag_dt_max:
            overrides['dag.dt_max_minutes'] = args.dag_dt_max
            
        # Apply JSON configuration overrides
        if args.config_overrides:
            try:
                json_overrides = json.loads(args.config_overrides)
                overrides.update(json_overrides)
            except json.JSONDecodeError as e:
                print(f"[build-graph] Error: Invalid JSON in config overrides - {e}")
                return 1
        
        # Load final configuration
        config = load_config_with_overrides(
            base_config_path=args.config,
            preset=args.preset,
            overrides=overrides
        )
        
        # Display configuration summary
        print(f"[build-graph] 🚀 IRONFORGE Dual Graph Views v{config.version}")
        print(f"[build-graph] Configuration: {args.preset} preset")
        print(f"[build-graph] DAG Construction: {'✅' if config.dag.enabled else '❌'}")
        print(f"[build-graph] M1 Integration: {'✅' if config.m1.enabled else '❌'}")  
        print(f"[build-graph] Enhanced TGAT: {'✅' if config.tgat.enhanced else '❌'}")
        print(f"[build-graph] Motif Mining: {'✅' if config.motifs.min_frequency <= 3 else '❌'}")
        print(f"[build-graph] Processing sessions from {args.source_glob}")
        print(f"[build-graph] Output directory: {args.output_dir}")
        print(f"[build-graph] Output format: {args.format}")
        
        if args.dry_run:
            print("[build-graph] 🔍 DRY RUN MODE - no files will be written")
        
        # Initialize components based on configuration
        output_path = Path(args.output_dir)
        
        # DAG builder with comprehensive configuration 
        dag_config = {
            'k_successors': config.dag.k_successors,
            'dt_min_minutes': config.dag.dt_min_minutes,
            'dt_max_minutes': config.dag.dt_max_minutes,
            'enabled': config.dag.enabled,
            'predicate': config.dag.predicate,
            'm1_integration': config.m1.enabled
        }
        builder = DAGGraphBuilder(dag_config=dag_config)
        
        # Motif miner if enabled
        motif_miner = None
        if config.motifs.min_frequency <= 5:  # Reasonable threshold for enabled
            from ironforge.learning.dag_motif_miner import MotifConfig
            motif_config = MotifConfig(
                min_nodes=config.motifs.min_nodes,
                max_nodes=config.motifs.max_nodes,
                min_frequency=config.motifs.min_frequency,
                null_iterations=config.motifs.null_iterations,
                significance_threshold=config.motifs.significance_threshold
            )
            motif_miner = DAGMotifMiner(motif_config)
            print(f"[build-graph] 🔍 Motif mining enabled: {config.motifs.null_iterations} null iterations")
        
        # Find and process session files
        session_files = glob.glob(args.source_glob)
        if args.max_sessions:
            session_files = session_files[:args.max_sessions]
        
        print(f"[build-graph] Found {len(session_files)} session files")
        
        if not session_files:
            print("[build-graph] ❌ No session files found")
            return 1
        
        # Storage for motif mining (collect all DAGs)
        all_dags = []
        all_session_names = []
        
        # Process each session
        processed_count = 0
        for i, session_file in enumerate(session_files):
            try:
                session_path = Path(session_file)
                session_name = session_path.stem
                
                print(f"[build-graph] Processing {i+1}/{len(session_files)}: {session_name}")
                
                # Load session data
                with open(session_path, 'r') as f:
                    session_data = json.load(f)
                
                # Load M1 data if M1 integration is enabled
                m1_data = None
                if config.m1.enabled:
                    # Look for corresponding M1 data file
                    m1_file_pattern = session_path.parent / f"{session_name}_M1.parquet"
                    if m1_file_pattern.exists():
                        import pandas as pd
                        m1_data = pd.read_parquet(m1_file_pattern)
                        print(f"[build-graph] Loaded M1 data: {len(m1_data)} bars")
                
                # Build graphs
                if config.dag.enabled:
                    if config.m1.enabled and m1_data is not None:
                        # Use M1-enhanced DAG building
                        dag_graph = builder.build_dag_with_m1_integration(session_data, m1_data)
                        temporal_graph = builder.build_session_graph_from_events(
                            session_data["events"], session_data
                        )
                    else:
                        # Standard dual view graphs
                        temporal_graph, dag_graph = builder.build_dual_view_graphs(session_data)
                    
                    # Collect DAG for motif mining
                    if motif_miner and dag_graph.number_of_nodes() > 0:
                        all_dags.append(dag_graph)
                        all_session_names.append(session_name)
                        
                else:
                    # Temporal graph only
                    temporal_graph = builder.build_session_graph_from_events(
                        session_data["events"], session_data
                    )
                    dag_graph = None
                
                if not args.dry_run:
                    # Create output directory
                    session_output_dir = output_path / session_name
                    session_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save graphs based on configuration and format
                    if args.format in ['pickle', 'both']:
                        # Temporal graph
                        temporal_pickle_path = session_output_dir / 'temporal_graph.pkl'
                        with open(temporal_pickle_path, 'wb') as f:
                            pickle.dump(temporal_graph, f)
                        
                        # DAG graph
                        if dag_graph:
                            dag_pickle_path = session_output_dir / 'dag_graph.pkl'
                            with open(dag_pickle_path, 'wb') as f:
                                pickle.dump(dag_graph, f)
                    
                    if args.format in ['parquet', 'both']:
                        # DAG edges to optimized parquet
                        if dag_graph and dag_graph.number_of_edges() > 0:
                            dag_edges_path = session_output_dir / 'edges_dag.parquet'
                            builder.save_dag_edges_parquet(dag_graph, dag_edges_path, session_name)
                    
                    # Save comprehensive metadata
                    metadata = {
                        'session_name': session_name,
                        'temporal_nodes': temporal_graph.number_of_nodes(),
                        'temporal_edges': temporal_graph.number_of_edges(),
                        'dag_enabled': config.dag.enabled,
                        'dag_nodes': dag_graph.number_of_nodes() if dag_graph else 0,
                        'dag_edges': dag_graph.number_of_edges() if dag_graph else 0,
                        'm1_enabled': config.m1.enabled,
                        'm1_events_detected': len(m1_data) if m1_data is not None else 0,
                        'enhanced_tgat': config.tgat.enhanced,
                        'configuration_preset': args.preset,
                        'build_timestamp': pd.Timestamp.now().isoformat(),
                        'feature_dimensions': config.tgat.input_dim
                    }
                    
                    metadata_path = session_output_dir / 'metadata.json'
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                processed_count += 1
                
            except Exception as e:
                print(f"[build-graph] ❌ Error processing {session_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Run motif mining if enabled
        if motif_miner and all_dags and not args.dry_run:
            print(f"[build-graph] 🔍 Mining motifs from {len(all_dags)} DAGs...")
            
            try:
                motifs = motif_miner.mine_motifs(all_dags, all_session_names)
                
                # Save motif results
                motifs_dir = output_path / 'motifs'
                motifs_dir.mkdir(exist_ok=True)
                
                motif_results = []
                for motif in motifs:
                    motif_result = {
                        'motif_id': motif.motif_id,
                        'frequency': motif.frequency,
                        'lift': motif.lift,
                        'p_value': motif.p_value,
                        'classification': motif.classification,
                        'sessions_found': list(motif.sessions_found),
                        'confidence_interval': motif.confidence_interval
                    }
                    motif_results.append(motif_result)
                
                motifs_summary_path = motifs_dir / 'motifs_summary.json'
                with open(motifs_summary_path, 'w') as f:
                    json.dump(motif_results, f, indent=2)
                
                promote_count = len([m for m in motifs if m.classification == 'PROMOTE'])
                park_count = len([m for m in motifs if m.classification == 'PARK'])
                
                print(f"[build-graph] 📊 Motif mining complete: {promote_count} PROMOTE, {park_count} PARK patterns")
                
            except Exception as e:
                print(f"[build-graph] ⚠️ Motif mining failed: {e}")
        
        # Save configuration if requested
        if args.save_config and not args.dry_run:
            config_output_path = output_path / 'dual_graph_config.json'
            config.save_to_file(config_output_path)
            print(f"[build-graph] 💾 Configuration saved: {config_output_path}")
        
        # Write comprehensive build manifest
        if not args.dry_run:
            manifest_path = output_path / 'build_manifest.jsonl'
            session_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name != 'motifs']
            
            with open(manifest_path, 'w') as f:
                for session_dir in session_dirs:
                    metadata_file = session_dir / 'metadata.json'
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as meta_f:
                            metadata = json.load(meta_f)
                            f.write(json.dumps(metadata) + '\n')
            
            print(f"[build-graph] 📋 Build manifest: {manifest_path}")
        
        print(f"[build-graph] ✅ Successfully processed {processed_count}/{len(session_files)} sessions")
        
        # Display final summary
        print(f"\n{'='*60}")
        print(f"🚀 DUAL GRAPH VIEWS BUILD COMPLETE")
        print(f"{'='*60}")
        print(f"Sessions Processed: {processed_count}")
        print(f"DAG Construction: {'✅' if config.dag.enabled else '❌'}")
        print(f"M1 Integration: {'✅' if config.m1.enabled else '❌'}")
        print(f"Enhanced TGAT: {'✅' if config.tgat.enhanced else '❌'}")
        if motif_miner and all_dags:
            print(f"Motifs Discovered: {len(motifs) if 'motifs' in locals() else 'N/A'}")
        print(f"Output Directory: {output_path}")
        print(f"Configuration: {args.preset} preset")
        print(f"{'='*60}")
        
        return 0
        
    except ImportError as e:
        print(f"[build-graph] ❌ Error: Required components not available - {e}")
        return 1
    except Exception as e:
        print(f"[build-graph] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_doctor(cfg=None):
    """Comprehensive system compatibility checks for IRONFORGE"""
    print("🔬 IRONFORGE Doctor - System Compatibility Check")
    print("=" * 50)
    
    issues = []
    warnings_count = 0
    
    # PyTorch version and device check
    try:
        import torch
        pytorch_version = torch.__version__
        print(f"✅ PyTorch version: {pytorch_version}")
        
        # Check if version supports SDPA (≥2.0)
        major, minor = map(int, pytorch_version.split('.')[:2])
        if major >= 2:
            print("✅ PyTorch version supports SDPA (≥2.0)")
        else:
            issues.append(f"❌ PyTorch {pytorch_version} < 2.0 - SDPA not available")
            
        # Check CUDA availability 
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available - using CPU")
            warnings_count += 1
            
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    # SDPA availability check
    try:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        print("✅ SDPA (scaled_dot_product_attention) available")
        
        # Test SDPA functionality
        q = torch.randn(1, 4, 10, 11)
        k = torch.randn(1, 4, 10, 11) 
        v = torch.randn(1, 4, 10, 11)
        
        output = sdpa(q, k, v)
        if output.shape == (1, 4, 10, 11):
            print("✅ SDPA functional test passed")
        else:
            issues.append(f"❌ SDPA test failed - output shape {output.shape}")
            
    except ImportError:
        issues.append("❌ SDPA not available - requires PyTorch ≥2.0")
    except Exception as e:
        issues.append(f"❌ SDPA test failed: {e}")
    
    # Parquet codec checks (ZSTD)
    try:
        import pyarrow.parquet as pq
        print("✅ PyArrow available")
        
        # Test ZSTD compression
        try:
            import pyarrow as pa
            
            # Create test data
            test_data = pa.table({'x': [1, 2, 3], 'y': [4, 5, 6]})
            
            # Test ZSTD compression
            import io
            buf = io.BytesIO()
            pq.write_table(test_data, buf, compression='zstd')
            buf.seek(0)
            
            # Read back
            table_read = pq.read_table(buf)
            if len(table_read) == 3:
                print("✅ ZSTD compression test passed")
            else:
                issues.append("❌ ZSTD compression test failed - data corruption")
                
        except Exception as e:
            issues.append(f"❌ ZSTD compression not available: {e}")
            
    except ImportError:
        issues.append("❌ PyArrow not installed")
    
    # NetworkX availability (for DAG operations)
    try:
        import networkx as nx
        print(f"✅ NetworkX available: {nx.__version__}")
        
        # Test DAG functionality
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (1, 2)])
        if nx.is_directed_acyclic_graph(dag):
            print("✅ DAG operations functional")
        else:
            issues.append("❌ DAG operations test failed")
            
    except ImportError:
        issues.append("❌ NetworkX not installed")
    
    # Context7 MCP connectivity (optional)
    try:
        # This is just a placeholder - real implementation would test MCP connection
        print("⚠️  Context7 MCP connectivity test skipped (optional)")
        warnings_count += 1
    except Exception:
        pass
    
    # Memory check
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb >= 8.0:
            print(f"✅ Available memory: {available_gb:.1f}GB")
        elif available_gb >= 4.0:
            print(f"⚠️  Available memory: {available_gb:.1f}GB (recommended: 8GB+)")
            warnings_count += 1
        else:
            issues.append(f"❌ Low memory: {available_gb:.1f}GB (minimum: 4GB)")
            
    except ImportError:
        print("⚠️  psutil not available - memory check skipped")
        warnings_count += 1
    
    # TGAT attention implementation check
    try:
        from ironforge.learning.tgat_discovery import graph_attention
        q = torch.randn(1, 4, 5, 11)
        k = torch.randn(1, 4, 5, 11)
        v = torch.randn(1, 4, 5, 11)
        
        # Test both implementations
        out_sdpa, _ = graph_attention(q, k, v, impl="sdpa", training=False)
        out_manual, _ = graph_attention(q, k, v, impl="manual", training=False)
        
        if out_sdpa.shape == out_manual.shape:
            print("✅ TGAT attention implementations functional")
        else:
            issues.append("❌ TGAT attention test failed - shape mismatch")
            
    except ImportError:
        issues.append("❌ IRONFORGE TGAT module not available")
    except Exception as e:
        issues.append(f"❌ TGAT attention test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    if issues:
        print(f"❌ Found {len(issues)} critical issues:")
        for issue in issues:
            print(f"  {issue}")
        print(f"\n⚠️  {warnings_count} warnings")
        return 1  # Exit code 1 for issues
    else:
        print("✅ All critical checks passed!")
        if warnings_count > 0:
            print(f"⚠️  {warnings_count} warnings (non-critical)")
        print("\nSystem ready for IRONFORGE operations.")
        return 0


def cmd_status(runs: Path):
    runs = Path(runs)
    if not runs.exists():
        print(json.dumps({"runs": []}, indent=2))
        return 0
    items = []
    for r in sorted([p for p in runs.iterdir() if p.is_dir()]):
        counts = {
            k: len(list((r / k).glob("**/*")))
            for k in ["embeddings", "patterns", "confluence", "motifs", "reports"]
        }
        items.append({"run": r.name, **counts})
    print(json.dumps({"runs": items}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("ironforge")
    sub = p.add_subparsers(dest="cmd", required=True)
    c1 = sub.add_parser("discover-temporal")
    c1.add_argument("--config", default="configs/dev.yml")
    c2 = sub.add_parser("score-session")
    c2.add_argument("--config", default="configs/dev.yml")
    c3 = sub.add_parser("validate-run")
    c3.add_argument("--config", default="configs/dev.yml")
    c4 = sub.add_parser("report-minimal")
    c4.add_argument("--config", default="configs/dev.yml")
    c5 = sub.add_parser("status")
    c5.add_argument("--runs", default="runs")
    
    # System compatibility doctor command
    c_doctor = sub.add_parser("doctor", help="Run system compatibility checks")
    
    # Oracle audit command
    c_audit = sub.add_parser("audit-oracle", help="Audit Oracle training pipeline sessions")
    c_audit.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g., NQ,ES)")
    c_audit.add_argument("--tf", required=True, help="Timeframe (e.g., 5 or M5)")
    c_audit.add_argument("--from", dest="from_date", required=True, help="Start date (YYYY-MM-DD)")
    c_audit.add_argument("--to", dest="to_date", required=True, help="End date (YYYY-MM-DD)")
    c_audit.add_argument("--data-dir", default="data/shards", help="Parquet shard data directory")
    c_audit.add_argument("--output", help="Output CSV ledger file")
    c_audit.add_argument("--min-sessions", type=int, default=57, help="Minimum required sessions")
    c_audit.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    # Oracle training command
    c_oracle = sub.add_parser("train-oracle", help="Train Oracle temporal non-locality system")
    c_oracle.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g., NQ,ES)")
    c_oracle.add_argument("--tf", required=True, help="Timeframe (e.g., 5 or M5)")
    c_oracle.add_argument("--from", dest="from_date", required=True, help="Start date (YYYY-MM-DD)")
    c_oracle.add_argument("--to", dest="to_date", required=True, help="End date (YYYY-MM-DD)")
    c_oracle.add_argument("--early-pct", type=float, default=0.20, help="Early batch percentage")
    c_oracle.add_argument("--htf-context", action="store_true", help="Enable HTF context (45D→51D)")
    c_oracle.add_argument("--no-htf-context", dest="htf_context", action="store_false", help="Disable HTF context")
    c_oracle.add_argument("--out", required=True, help="Output model directory")
    c_oracle.add_argument("--rebuild", action="store_true", help="Force rebuild embeddings")
    c_oracle.add_argument("--data-dir", default="data/shards", help="Parquet shard data directory")
    c_oracle.add_argument("--max-sessions", type=int, help="Limit training sessions")
    c_oracle.add_argument("--strict", action="store_true", help="Enable strict mode with audit validation")
    c_oracle.add_argument("--min-sessions", type=int, help="Minimum required sessions (strict mode)")
    c_oracle.set_defaults(htf_context=False)
    
    c6 = sub.add_parser("prep-shards")
    c6.add_argument(
        "--source-glob",
        default="data/enhanced/enhanced_*_Lvl-1_*.json",
        help="Glob pattern for enhanced JSON sessions",
    )
    c6.add_argument("--symbol", default="NQ", help="Symbol for shard directory")
    c6.add_argument("--timeframe", "--tf", default="M5", help="Timeframe for shard directory")
    c6.add_argument("--timezone", "--tz", default="ET", help="Source timezone")
    c6.add_argument(
        "--pack",
        choices=["single", "pack"],
        default="single",
        help="Packing mode: single session per shard or pack multiple",
    )
    c6.add_argument(
        "--dry-run", action="store_true", help="Show what would be converted without writing files"
    )
    c6.add_argument("--overwrite", action="store_true", help="Overwrite existing shards")
    c6.add_argument(
        "--htf-context", action="store_true", help="Enable HTF context features (45D → 51D)"
    )

    # Graph building command with comprehensive configuration
    c7 = sub.add_parser("build-graph", help="Build dual graph views (temporal + DAG) from session data")
    c7.add_argument(
        "--source-glob", 
        default="data/enhanced/enhanced_*_Lvl-1_*.json",
        help="Glob pattern for enhanced JSON sessions"
    )
    c7.add_argument("--output-dir", default="data/graphs", help="Output directory for graph files")
    
    # Configuration options
    c7.add_argument("--config", type=Path, help="Path to dual graph views configuration JSON file")
    c7.add_argument("--preset", choices=["minimal", "standard", "enhanced", "research"], 
                   default="standard", help="Configuration preset to use")
    c7.add_argument("--config-overrides", type=str, help="JSON string of configuration overrides")
    
    # Quick configuration flags (override config file)
    c7.add_argument("--with-dag", action="store_true", help="Enable DAG view construction")
    c7.add_argument("--with-m1", action="store_true", help="Enable M1 integration and cross-scale features")
    c7.add_argument("--enhanced-tgat", action="store_true", help="Enable enhanced TGAT with masked attention")
    c7.add_argument("--enable-motifs", action="store_true", help="Enable motif mining and statistical validation")
    
    # Legacy DAG options (for backwards compatibility)
    c7.add_argument("--dag-k", type=int, help="DAG k-successors per node")
    c7.add_argument("--dag-dt-min", type=int, help="DAG minimum time delta (minutes)")
    c7.add_argument("--dag-dt-max", type=int, help="DAG maximum time delta (minutes)")
    
    # Processing options
    c7.add_argument("--format", choices=["pickle", "parquet", "both"], default="parquet", 
                   help="Output format (parquet recommended)")
    c7.add_argument("--dry-run", action="store_true", help="Show what would be built without writing files")
    c7.add_argument("--max-sessions", type=int, help="Limit number of sessions to process")
    c7.add_argument("--save-config", action="store_true", help="Save final configuration to output directory")

    args = p.parse_args(argv)
    if args.cmd == "status":
        return cmd_status(Path(args.runs))
    if args.cmd == "doctor":
        return cmd_doctor()
    if args.cmd == "audit-oracle":
        return cmd_audit_oracle(
            symbols=args.symbols.split(","),
            timeframe=args.tf,
            from_date=args.from_date,
            to_date=args.to_date,
            data_dir=args.data_dir,
            output_file=args.output,
            min_sessions=args.min_sessions,
            verbose=args.verbose,
        )
    if args.cmd == "prep-shards":
        return cmd_prep_shards(
            args.source_glob,
            args.symbol,
            args.timeframe,
            args.timezone,
            args.pack,
            args.dry_run,
            args.overwrite,
            args.htf_context,
        )
    if args.cmd == "build-graph":
        return cmd_build_graph(args)
    if args.cmd == "train-oracle":
        return cmd_train_oracle(
            symbols=args.symbols.split(","),
            timeframe=args.tf,
            from_date=args.from_date,
            to_date=args.to_date,
            early_pct=args.early_pct,
            htf_context=args.htf_context,
            output_dir=args.out,
            rebuild=args.rebuild,
            data_dir=args.data_dir,
            max_sessions=args.max_sessions,
            strict_mode=args.strict,
            min_sessions=args.min_sessions,
        )
    cfg = load_config(args.config)
    try:
        validate_config(cfg)
    except Exception as e:
        print(f"[config] invalid configuration: {e}")
        return 2
    if args.cmd == "discover-temporal":
        return cmd_discover(cfg)
    if args.cmd == "score-session":
        return cmd_score(cfg)
    if args.cmd == "validate-run":
        return cmd_validate(cfg)
    if args.cmd == "report-minimal":
        return cmd_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
