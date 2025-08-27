"""Oracle training and audit commands extracted from CLI for better separation of concerns."""

from __future__ import annotations

from pathlib import Path

from ..constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_SESSIONS,
    DEFAULT_PATIENCE,
    MIN_TRAINING_SESSIONS,
)


def cmd_audit_oracle(
    symbols: list[str],
    timeframe: str,
    from_date: str,
    to_date: str,
    data_dir: str,
    output_file: str = None,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    verbose: bool = False,
):
    """Audit Oracle training pipeline sessions"""
    from oracle.audit import OracleAuditor
    
    print("🔍 Oracle Training Pipeline Audit")
    print(f"{'='*50}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {timeframe}")
    print(f"Date range: {from_date} to {to_date}")
    print(f"Data source: {data_dir}")
    print(f"Minimum sessions required: {min_sessions}")
    
    # Use first symbol for now
    symbol = symbols[0]
    
    try:
        # Initialize auditor
        auditor = OracleAuditor(data_dir=data_dir, verbose=verbose)
        
        # Run comprehensive audit
        audit_summary = auditor.run_audit(symbol, timeframe, from_date, to_date)
        
        # Save ledger if requested
        if output_file:
            ledger_path = Path(output_file)
            auditor.save_audit_ledger(audit_summary, ledger_path)
            print(f"📋 Audit ledger saved: {output_file}")
        
        # Generate gap analysis
        gap_analysis = auditor.generate_gap_analysis(audit_summary, min_sessions)
        
        # Print detailed results
        print("\n📊 Audit Results")
        print(f"Sessions discovered: {audit_summary['sessions_discovered']}")
        print(f"Sessions processable: {audit_summary['audit_total']}")
        print(f"Success rate: {audit_summary['success_rate']:.1%}")
        
        # Error breakdown
        if any(count > 0 for code, count in audit_summary['error_breakdown'].items() if code != 'SUCCESS'):
            print("\n❌ Error Breakdown:")
            for error_code, count in audit_summary['error_breakdown'].items():
                if count > 0 and error_code != 'SUCCESS':
                    description = auditor.ERROR_CODES.get(error_code, 'Unknown error')
                    print(f"  {error_code}: {count} sessions - {description}")
        
        # Gap analysis
        if gap_analysis['gap_exists']:
            print(f"\n⚠️  Coverage Gap: {gap_analysis['missing_count']} sessions missing")
            print("🔧 Remediation Steps:")
            for step in gap_analysis['remediation_steps']:
                print(f"  {step}")
            
            if gap_analysis.get('expected_missing_paths'):
                print("\n📂 Expected Missing Shard Paths (examples):")
                for path in gap_analysis['expected_missing_paths'][:3]:
                    print(f"  {path}")
                if len(gap_analysis['expected_missing_paths']) > 3:
                    remaining = len(gap_analysis['expected_missing_paths']) - 3
                    print(f"  ... and {remaining} more")
        else:
            print(f"\n✅ Coverage Sufficient: {audit_summary['audit_total']}/{min_sessions} sessions")
        
        # Critical output for train-oracle integration
        print(f"\naudit_total: {audit_summary['audit_total']}")
        
        # Exit with appropriate code for script integration
        if audit_summary['audit_total'] >= min_sessions:
            return 0
        else:
            print(f"\n❌ Insufficient sessions: {audit_summary['audit_total']} < {min_sessions}")
            return 1
            
    except Exception as e:
        print(f"❌ Audit failed: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return 2


def cmd_train_oracle(
    symbols: list[str],
    timeframe: str,
    from_date: str,
    to_date: str,
    early_pct: float,
    htf_context: bool,
    output_dir: str,
    rebuild: bool,
    data_dir: str,
    max_sessions: int = None,
    strict_mode: bool = False,
    min_sessions: int = None,
):
    """Train Oracle temporal non-locality system"""
    from pathlib import Path
    
    # Normalize timeframe: accept both "5" and "M5", convert to numeric
    tf_numeric = timeframe
    tf_string = f"M{timeframe}"
    if timeframe.upper().startswith('M'):
        tf_numeric = timeframe[1:]
        tf_string = timeframe.upper()
    elif timeframe.isdigit():
        tf_numeric = timeframe
        tf_string = f"M{timeframe}"
    else:
        print(f"❌ Invalid timeframe: {timeframe}. Use '5' or 'M5'")
        return 1
    
    print("🚀 Starting Oracle Training Pipeline")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {tf_string} (numeric: {tf_numeric})")
    print(f"Date range: {from_date} to {to_date}")
    print(f"Early percentage: {early_pct}")
    print(f"HTF context: {'ENABLED (51D)' if htf_context else 'DISABLED (45D)'}") 
    print(f"Output: {output_dir}")
    print(f"Data source: {data_dir} (Parquet shards only)")
    if strict_mode:
        print(f"⚠️  STRICT MODE: Requiring {min_sessions or 'audit_total'} sessions")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 0: Audit preflight (mandatory for strict mode)
    if strict_mode:
        print("\n🔍 Step 0: Running audit preflight...")
        audit_result = cmd_audit_oracle(
            symbols, tf_string, from_date, to_date, 
            data_dir=data_dir, verbose=False, 
            min_sessions=min_sessions or 57  # Use passed min_sessions or default 57
        )
        
        if audit_result != 0:
            print(f"❌ Audit preflight failed with code {audit_result}")
            return audit_result
        
        # Extract audit_total from audit output
        from oracle.audit import OracleAuditor
        auditor = OracleAuditor(data_dir=data_dir, verbose=False)
        audit_summary = auditor.run_audit(symbols[0], tf_string, from_date, to_date)
        audit_total = audit_summary['audit_total']
        
        if min_sessions and audit_total < min_sessions:
            print(f"❌ Insufficient sessions: {audit_total} < {min_sessions} required")
            return 1
        
        print(f"✅ Audit passed: {audit_total} sessions available")
    else:
        pass
    
    # Enforce real Parquet shard processing only
    print("\n🔥 Real Data Processing Pipeline (Zero Compromises)")
    
    # Validate minimum session requirements upfront
    from pathlib import Path
    shard_base = Path(data_dir) / f"{symbols[0]}_{tf_string}"
    if not shard_base.exists():
        print(f"❌ No shard directory found: {shard_base}")
        print("💡 Run prep-shards first to convert enhanced sessions")
        return 1
        
    available_shards = [d for d in shard_base.iterdir() if d.is_dir() and d.name.startswith("shard_")]
    print(f"📊 Found {len(available_shards)} available shards")
    
    if len(available_shards) < MIN_TRAINING_SESSIONS:  # Minimum viable sessions
        print(f"❌ Insufficient shards: {len(available_shards)} < {MIN_TRAINING_SESSIONS} minimum")
        print(f"💡 Add more sessions to {shard_base}")
        return 1
    # Step 1: Normalize sessions with date filtering
    print("\n📊 Step 1: Normalizing Parquet shards...")
    normalized_file = output_path / "normalized_sessions.parquet"
    
    # Import normalizer directly instead of subprocess
    try:
        from oracle.data_normalizer import OracleDataNormalizer
        normalizer = OracleDataNormalizer(data_dir, verbose=True)
        
        # Normalize with strict filtering
        sessions_df = normalizer.normalize_symbol_timeframe_with_dates(
            symbol=symbols[0],
            timeframe=tf_string,
            from_date=from_date,
            to_date=to_date,
            min_quality="fair",
            min_events=10
        )
        
        if sessions_df.empty:
            print(f"❌ No sessions found in date range {from_date} to {to_date}")
            return 1
            
        if len(sessions_df) < MIN_TRAINING_SESSIONS:
            print(f"❌ Insufficient sessions: {len(sessions_df)} < {MIN_TRAINING_SESSIONS} required")
            print("💡 Expand date range or reduce quality threshold")
            return 1
            
        sessions_df.to_parquet(normalized_file, index=False)
        print(f"✅ Normalized {len(sessions_df)} sessions")
        
    except Exception as e:
        print(f"❌ Normalization error: {e}")
        return 1
    
    # Step 2: Build training pairs
    print("\n🧠 Step 2: Building training embeddings...")
    training_pairs_file = output_path / "training_pairs.parquet"
    
    try:
        from oracle.pairs_builder import OraclePairsBuilder
        builder = OraclePairsBuilder(data_dir=data_dir, verbose=True)
        
        # Load normalized sessions
        import pandas as pd
        sessions_df = pd.read_parquet(normalized_file)
        
        # Limit sessions if specified
        if max_sessions:
            sessions_df = sessions_df.head(max_sessions)
            print(f"📊 Limited to {max_sessions} sessions for training")
        
        # Build training pairs with strict validation
        training_pairs_df = builder.build_training_pairs(
            sessions_df,
            early_pct=early_pct,
            max_sessions=max_sessions
        )
        
        if training_pairs_df.empty:
            print("❌ No training pairs generated")
            print(f"💡 Check Parquet shard availability in {data_dir}")
            return 1
            
        if len(training_pairs_df) < MIN_TRAINING_SESSIONS:
            print(f"❌ Insufficient training pairs: {len(training_pairs_df)} < {MIN_TRAINING_SESSIONS} required")
            return 1
            
        training_pairs_df.to_parquet(training_pairs_file, index=False)
        print(f"✅ Built {len(training_pairs_df)} training pairs")
        
    except Exception as e:
        print(f"❌ Training pairs build error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Train Oracle model
    print("\n🎯 Step 3: Training Oracle range head...")
    model_dir = output_path
    
    try:
        from oracle.trainer import OracleTrainer
        trainer = OracleTrainer(model_dir=model_dir)
        
        # Load and validate training data
        X, y, metadata = trainer.load_training_data(training_pairs_file)
        data_splits = trainer.prepare_data_splits(X, y, test_size=0.2, val_size=0.2)
        
        print(f"📊 Training data: {len(X)} samples")
        print(f"📊 Train/Val/Test split: {len(data_splits['X_train'])}/{len(data_splits['X_val'])}/{len(data_splits['X_test'])}")
        
        # Train with early stopping
        training_results = trainer.train_model(
            data_splits=data_splits,
            epochs=DEFAULT_EPOCHS,
            learning_rate=DEFAULT_LEARNING_RATE,
            batch_size=DEFAULT_BATCH_SIZE,
            patience=DEFAULT_PATIENCE
        )
        
        # Save complete artifacts
        training_config = {
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "batch_size": DEFAULT_BATCH_SIZE,
            "patience": DEFAULT_PATIENCE,
            "early_pct": early_pct,
            "n_training_pairs": len(training_pairs_df),
            "date_range": f"{from_date} to {to_date}"
        }
        trainer.save_model(training_config)
        print(f"✅ Training completed: {training_results['epochs_trained']} epochs")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Evaluate model
    print("\n📈 Step 4: Evaluating trained model...")
    
    try:
        from oracle.eval import OracleEvaluator
        evaluator = OracleEvaluator(model_dir=model_dir)
        
        # Evaluate on test set
        metrics = evaluator.evaluate_model(training_pairs_file, test_size=0.2)
        
        # Validate metrics are reasonable
        import numpy as np
        overall_metrics = metrics.get('overall', metrics)
        center_mae = overall_metrics.get('center_mae', 0.0)
        center_rmse = overall_metrics.get('center_rmse', 0.0)
        center_mape = overall_metrics.get('center_mape', 0.0)
        
        if any(np.isnan(v) or np.isinf(v) for v in [center_mae, center_rmse, center_mape]):
            print(f"❌ Invalid metrics detected: MAE={center_mae}, RMSE={center_rmse}, MAPE={center_mape}")
            return 1
        
        # Save metrics
        metrics_file = model_dir / "metrics.json"
        evaluator.save_metrics(metrics, metrics_file)
        
        print(f"✅ Evaluation completed - MAE: {center_mae:.3f}, RMSE: {center_rmse:.3f}, MAPE: {center_mape:.1f}%")
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Runtime validation and final summary
    print("\n🔍 Step 5: Final validation and cleanup...")
    
    try:
        # Verify required artifacts exist
        required_files = ["weights.pt", "scaler.pkl", "training_manifest.json", "metrics.json"]
        missing_files = []
        for filename in required_files:
            if not (model_dir / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            print(f"❌ Missing artifacts: {missing_files}")
            return 1
            
        print("✅ All required artifacts present")
        
    except Exception as e:
        print(f"❌ Final validation error: {e}")
        return 1
    
    # Final summary
    print("\n🎉 Oracle Training Pipeline Completed!")
    print(f"{'='*60}")
    print(f"📁 Model saved to: {model_dir}")
    print(f"📊 Sessions processed: {len(sessions_df)} → {len(training_pairs_df)} training pairs")
    print(f"📊 Metrics: MAE={center_mae:.3f}, RMSE={center_rmse:.3f}, MAPE={center_mape:.1f}%")
    print("📋 Artifacts: weights.pt, scaler.pkl, training_manifest.json, metrics.json")
    print("\n🔮 Oracle is now ready for temporal non-locality predictions!")
    
    return 0