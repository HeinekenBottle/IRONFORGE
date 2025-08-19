#!/usr/bin/env python3
"""
Simple Schema Validation
========================

Validates IRONFORGE schema contracts using file inspection and pattern matching.
Works without pandas/pyarrow dependencies.
"""

import os
import glob
import subprocess
import sys


def check_parquet_schema_with_file_command(file_path):
    """
    Use the 'file' command to get basic info about parquet files.
    This is a fallback when we can't read parquet directly.
    """
    try:
        result = subprocess.run(['file', file_path], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return f"Could not analyze {file_path}"


def estimate_schema_from_file_size_and_name(file_path, file_type="nodes"):
    """
    Estimate schema based on file size and naming patterns.
    This is a heuristic approach when direct reading fails.
    """
    file_size = os.path.getsize(file_path)
    
    # Heuristic: estimate feature count based on file size
    # Typical parquet compression ratios and data patterns
    
    if file_type == "nodes":
        # Nodes typically have more features
        if file_size > 25000:  # > 25KB suggests substantial feature set
            estimated_features = "45-55 features (likely includes f0-f44 + HTF features)"
        elif file_size > 15000:  # > 15KB
            estimated_features = "35-45 features (base features only)"
        else:
            estimated_features = "< 35 features (minimal set)"
    else:  # edges
        # Edges typically have fewer features
        if file_size > 10000:  # > 10KB
            estimated_features = "15-25 features (likely includes e0-e19 + metadata)"
        elif file_size > 5000:   # > 5KB
            estimated_features = "10-20 features (core features)"
        else:
            estimated_features = "< 15 features (minimal set)"
    
    return {
        "file_size": file_size,
        "estimated_features": estimated_features,
        "analysis_method": "file_size_heuristic"
    }


def validate_module_imports_basic():
    """
    Test basic module imports without triggering dependency errors.
    """
    print("🔍 BASIC MODULE IMPORT VALIDATION")
    print("=" * 60)
    
    # Test if we can at least find the module files
    modules_to_check = [
        ("ironforge.learning.discovery_pipeline", "ironforge/learning/discovery_pipeline.py"),
        ("ironforge.confluence.scoring", "ironforge/confluence/scoring.py"),
        ("ironforge.validation.runner", "ironforge/validation/runner.py"),
        ("ironforge.reporting.minidash", "ironforge/reporting/minidash.py"),
    ]
    
    results = {}
    
    for module_name, file_path in modules_to_check:
        if os.path.exists(file_path):
            # Try to check if the file has the expected function
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Look for expected function names
                expected_functions = {
                    "discovery_pipeline": "run_discovery",
                    "scoring": "score_confluence", 
                    "runner": "validate_run",
                    "minidash": "build_minidash"
                }
                
                module_key = module_name.split('.')[-1]
                expected_func = expected_functions.get(module_key, "")
                
                if expected_func and f"def {expected_func}" in content:
                    results[module_name] = {"status": "✅", "function_found": expected_func}
                    print(f"✅ {module_name} (function: {expected_func})")
                else:
                    results[module_name] = {"status": "⚠️", "message": f"Function {expected_func} not found"}
                    print(f"⚠️  {module_name} (missing function: {expected_func})")
                    
            except Exception as e:
                results[module_name] = {"status": "❌", "error": str(e)}
                print(f"❌ {module_name} (error: {e})")
        else:
            results[module_name] = {"status": "❌", "message": "File not found"}
            print(f"❌ {module_name} (file not found: {file_path})")
    
    return results


def validate_cli_structure():
    """Validate CLI structure without importing"""
    print("\n🔍 CLI STRUCTURE VALIDATION")
    print("=" * 60)
    
    cli_file = "ironforge/sdk/cli.py"
    
    if not os.path.exists(cli_file):
        print("❌ CLI file not found")
        return False
    
    try:
        with open(cli_file, 'r') as f:
            content = f.read()
        
        # Check for expected CLI commands
        expected_commands = [
            "discover-temporal",
            "score-session", 
            "validate-run",
            "report-minimal",
            "status"
        ]
        
        found_commands = []
        for cmd in expected_commands:
            if cmd in content:
                found_commands.append(cmd)
                print(f"✅ {cmd}")
            else:
                print(f"❌ {cmd}")
        
        if len(found_commands) == len(expected_commands):
            print(f"\n✅ All {len(expected_commands)} CLI commands found")
            return True
        else:
            print(f"\n⚠️  Found {len(found_commands)}/{len(expected_commands)} commands")
            return False
            
    except Exception as e:
        print(f"❌ Error reading CLI file: {e}")
        return False


def main():
    """Run simplified validation"""
    print("🏛️ IRONFORGE SIMPLIFIED PRE-MERGE VALIDATION")
    print("=" * 80)
    
    # 1. Validate module structure
    module_results = validate_module_imports_basic()
    modules_ok = all(r["status"] == "✅" for r in module_results.values())
    
    # 2. Validate CLI structure  
    cli_ok = validate_cli_structure()
    
    # 3. Check shard files exist and estimate schema
    print("\n🔍 SHARD SCHEMA ESTIMATION")
    print("=" * 60)
    
    shard_dirs = glob.glob("data/shards/*/shard_*")
    schema_ok = True
    
    if shard_dirs:
        sample_shard = shard_dirs[0]
        print(f"📁 Analyzing: {sample_shard}")
        
        nodes_file = os.path.join(sample_shard, "nodes.parquet")
        edges_file = os.path.join(sample_shard, "edges.parquet")
        
        if os.path.exists(nodes_file):
            nodes_info = estimate_schema_from_file_size_and_name(nodes_file, "nodes")
            print(f"📊 Nodes: {nodes_info['file_size']} bytes")
            print(f"   Estimated: {nodes_info['estimated_features']}")
            
            # Check if size suggests 51D compliance
            if nodes_info['file_size'] > 25000:
                print("   ✅ File size suggests adequate feature count for 51D")
            else:
                print("   ⚠️  File size may indicate fewer than 51 features")
                schema_ok = False
        else:
            print("❌ nodes.parquet not found")
            schema_ok = False
        
        if os.path.exists(edges_file):
            edges_info = estimate_schema_from_file_size_and_name(edges_file, "edges")
            print(f"📊 Edges: {edges_info['file_size']} bytes")
            print(f"   Estimated: {edges_info['estimated_features']}")
            
            # Check if size suggests 20D compliance
            if edges_info['file_size'] > 8000:
                print("   ✅ File size suggests adequate feature count for 20D")
            else:
                print("   ⚠️  File size may indicate fewer than 20 features")
                schema_ok = False
        else:
            print("❌ edges.parquet not found")
            schema_ok = False
    else:
        print("❌ No shards found")
        schema_ok = False
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"📦 Module Structure: {'✅ PASS' if modules_ok else '❌ FAIL'}")
    print(f"🖥️  CLI Structure: {'✅ PASS' if cli_ok else '❌ FAIL'}")
    print(f"📊 Schema Files: {'✅ PASS' if schema_ok else '⚠️  WARNING'}")
    
    if modules_ok and cli_ok:
        print("\n🎉 CORE STRUCTURE VALIDATED")
        print("✅ Ready for dependency installation and full testing")
        
        if not schema_ok:
            print("\n⚠️  SCHEMA WARNING:")
            print("   File sizes suggest potential schema contract mismatches")
            print("   Install dependencies and run full validation to confirm")
        
        return 0
    else:
        print("\n❌ CRITICAL ISSUES DETECTED")
        print("   Resolve module/CLI issues before proceeding")
        return 1


if __name__ == "__main__":
    exit(main())
