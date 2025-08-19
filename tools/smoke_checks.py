#!/usr/bin/env python3
"""
IRONFORGE Smoke Checks
=====================

Lightweight validation script that asserts core system functionality:
- Entrypoint imports succeed
- CLI help shows all 5 commands  
- Shard dims 51/20 detected on sample shard
- Latest run has confluence artifacts
- Attention data includes required fields or logs rank proxy
"""

import glob
import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path


def check_entrypoint_imports():
    """Test 1: Verify all entrypoints can be imported"""
    print("🔍 Test 1: Entrypoint Import Validation")
    print("-" * 50)
    
    entrypoints = [
        ("ironforge.learning.discovery_pipeline", "run_discovery"),
        ("ironforge.confluence.scoring", "score_confluence"), 
        ("ironforge.validation.runner", "validate_run"),
        ("ironforge.reporting.minidash", "build_minidash")
    ]
    
    failed = []
    for module_name, function_name in entrypoints:
        try:
            module = import_module(module_name)
            getattr(module, function_name)
            print(f"  ✅ {module_name}:{function_name}")
        except Exception as e:
            print(f"  ❌ {module_name}:{function_name} - {e}")
            failed.append(f"{module_name}:{function_name}")
    
    if failed:
        print(f"❌ Failed imports: {failed}")
        return False
    
    print("✅ All entrypoints imported successfully")
    return True


def check_cli_commands():
    """Test 2: Verify CLI shows all 5 commands"""
    print("\n🔍 Test 2: CLI Command Validation")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ironforge.sdk.cli", "--help"],
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode != 0:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
        
        help_text = result.stdout
        required_commands = [
            "discover-temporal",
            "score-session", 
            "validate-run",
            "report-minimal",
            "status"
        ]
        
        missing = []
        for cmd in required_commands:
            if cmd in help_text:
                print(f"  ✅ {cmd}")
            else:
                print(f"  ❌ {cmd}")
                missing.append(cmd)
        
        if missing:
            print(f"❌ Missing CLI commands: {missing}")
            return False
        
        print("✅ All CLI commands available")
        return True
        
    except Exception as e:
        print(f"❌ CLI check failed: {e}")
        return False


def check_shard_dimensions():
    """Test 3: Verify shard dimensions are 51/20"""
    print("\n🔍 Test 3: Shard Dimension Validation")
    print("-" * 50)
    
    try:
        import pyarrow.parquet as pq
        
        # Find sample shard
        shard_pattern = "data/shards/*/shard_*"
        shard_dirs = glob.glob(shard_pattern)
        
        if not shard_dirs:
            print("⚠️  No shards found - skipping dimension check")
            return True
        
        sample_shard = shard_dirs[0]
        print(f"  Checking: {sample_shard}")
        
        # Check nodes
        nodes_file = Path(sample_shard) / "nodes.parquet"
        if nodes_file.exists():
            nodes_table = pq.read_table(str(nodes_file))
            node_features = [col for col in nodes_table.column_names if col.startswith('f')]
            print(f"  Node features: {len(node_features)} (expected: 51)")
            
            if len(node_features) != 51:
                print(f"❌ Wrong node dimensions: {len(node_features)} != 51")
                return False
        else:
            print(f"❌ Missing nodes.parquet in {sample_shard}")
            return False
        
        # Check edges  
        edges_file = Path(sample_shard) / "edges.parquet"
        if edges_file.exists():
            edges_table = pq.read_table(str(edges_file))
            edge_features = [col for col in edges_table.column_names if col.startswith('e')]
            print(f"  Edge features: {len(edge_features)} (expected: 20)")
            
            if len(edge_features) != 20:
                print(f"❌ Wrong edge dimensions: {len(edge_features)} != 20")
                return False
        else:
            print(f"❌ Missing edges.parquet in {sample_shard}")
            return False
        
        print("✅ Shard dimensions correct (51D nodes, 20D edges)")
        return True
        
    except ImportError:
        print("⚠️  PyArrow not available - skipping shard check")
        return True
    except Exception as e:
        print(f"❌ Shard check failed: {e}")
        return False


def check_latest_run_artifacts():
    """Test 4: Verify latest run has required artifacts"""
    print("\n🔍 Test 4: Latest Run Artifact Validation")
    print("-" * 50)
    
    try:
        # Find latest run
        run_dirs = sorted([p for p in Path("runs").glob("20*") if p.is_dir()])
        
        if not run_dirs:
            print("⚠️  No runs found - skipping artifact check")
            return True
        
        latest_run = run_dirs[-1]
        print(f"  Checking: {latest_run}")
        
        # Check confluence scores
        scores_file = latest_run / "confluence" / "scores.parquet"
        if scores_file.exists():
            print("  ✅ confluence/scores.parquet")
            
            # Check scale
            try:
                import pandas as pd
                df = pd.read_parquet(scores_file)
                if 'confidence' in df.columns:
                    max_score = float(df['confidence'].max())
                    if max_score <= 1:
                        scale = "0-1"
                    elif max_score <= 100:
                        scale = "0-100"
                    else:
                        scale = "threshold"
                    print(f"  📊 Confluence scale: {scale} (max: {max_score:.2f})")
                else:
                    print("  ⚠️  No 'confidence' column in scores")
            except ImportError:
                print("  ⚠️  Pandas not available - skipping scale check")
        else:
            print("  ❌ Missing confluence/scores.parquet")
            return False
        
        # Check stats.json
        stats_file = latest_run / "confluence" / "stats.json"
        if stats_file.exists():
            print("  ✅ confluence/stats.json")
            
            try:
                stats = json.loads(stats_file.read_text())
                health = stats.get('health_status', 'unknown')
                print(f"  🏥 Health status: {health}")
            except Exception as e:
                print(f"  ⚠️  Could not parse stats.json: {e}")
        else:
            print("  ❌ Missing confluence/stats.json")
            return False
        
        print("✅ Latest run artifacts present")
        return True
        
    except Exception as e:
        print(f"❌ Run artifact check failed: {e}")
        return False


def check_attention_data():
    """Test 5: Verify attention data or log rank proxy"""
    print("\n🔍 Test 5: Attention Data Validation")
    print("-" * 50)
    
    try:
        # Find latest run
        run_dirs = sorted([p for p in Path("runs").glob("20*") if p.is_dir()])
        
        if not run_dirs:
            print("⚠️  No runs found - skipping attention check")
            return True
        
        latest_run = run_dirs[-1]
        attention_file = latest_run / "embeddings" / "attention_topk.parquet"
        
        if attention_file.exists():
            print("  ✅ embeddings/attention_topk.parquet found")
            
            try:
                import pandas as pd
                df = pd.read_parquet(attention_file)
                
                required_cols = ['edge_intent', 'weight', 'attn_rank']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"  ❌ Missing columns: {missing_cols}")
                    return False
                
                print("  ✅ Required columns present: edge_intent, weight, attn_rank")
                
                if 'true_dt_s' in df.columns:
                    non_null_dt = df['true_dt_s'].notna().sum()
                    total_rows = len(df)
                    print(f"  ⏱️  Real time deltas: {non_null_dt}/{total_rows} rows")
                    if non_null_dt > 0:
                        print("  🎯 Mode: real seconds")
                    else:
                        print("  📊 Mode: rank proxy")
                else:
                    print("  📊 Mode: rank proxy (no true_dt_s column)")
                
            except ImportError:
                print("  ⚠️  Pandas not available - basic file check only")
        else:
            print("  📊 No attention data - using rank proxy mode")
        
        print("✅ Attention analysis ready")
        return True
        
    except Exception as e:
        print(f"❌ Attention check failed: {e}")
        return False


def main():
    """Run all smoke checks"""
    print("🏛️ IRONFORGE SMOKE CHECKS")
    print("=" * 60)
    print("Validating core system functionality...")
    print()
    
    checks = [
        ("Entrypoint Imports", check_entrypoint_imports),
        ("CLI Commands", check_cli_commands),
        ("Shard Dimensions", check_shard_dimensions), 
        ("Run Artifacts", check_latest_run_artifacts),
        ("Attention Data", check_attention_data)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SMOKE CHECK RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All smoke checks PASSED!")
        print("💡 IRONFORGE core functionality verified")
        return 0
    else:
        print("⚠️  Some checks failed - investigate before proceeding")
        print("💡 Check logs and configuration")
        return 1


if __name__ == "__main__":
    sys.exit(main())
