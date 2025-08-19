#!/usr/bin/env python3
"""
Schema Contract Validation (Dependency-Free)
============================================

Validates actual shard schema against documented contracts without requiring pandas/pyarrow.
Uses only Python standard library to check parquet file structure.
"""

import os
import glob
import struct
import json


def read_parquet_schema_basic(file_path):
    """
    Basic parquet schema reader using only standard library.
    Extracts column names from parquet metadata.
    """
    try:
        with open(file_path, 'rb') as f:
            # Read file size
            f.seek(0, 2)
            file_size = f.tell()
            
            # Read footer length (last 4 bytes)
            f.seek(file_size - 4)
            footer_length = struct.unpack('<I', f.read(4))[0]
            
            # Read footer
            f.seek(file_size - 4 - footer_length)
            footer_data = f.read(footer_length)
            
            # Look for column names in footer (basic text search)
            footer_str = footer_data.decode('utf-8', errors='ignore')
            
            # Extract potential column names (simple heuristic)
            columns = []
            
            # Look for feature patterns
            for i in range(100):  # Check f0-f99
                if f'"f{i}"' in footer_str or f'f{i}' in footer_str:
                    columns.append(f'f{i}')
            
            # Look for edge patterns  
            for i in range(50):  # Check e0-e49
                if f'"e{i}"' in footer_str or f'e{i}' in footer_str:
                    columns.append(f'e{i}')
            
            # Look for other common columns
            common_cols = ['timestamp', 'session_id', 'event_type', 'price', 'volume', 
                          'node_id', 'edge_id', 'source', 'target', 'intent']
            for col in common_cols:
                if f'"{col}"' in footer_str or col in footer_str:
                    if col not in columns:
                        columns.append(col)
            
            return sorted(columns)
            
    except Exception as e:
        return f"Error reading {file_path}: {e}"


def validate_schema_contracts():
    """Validate actual schema against documented contracts"""
    
    print("🔍 IRONFORGE Schema Contract Validation")
    print("=" * 60)
    
    # Find sample shard
    shard_pattern = "data/shards/*/shard_*"
    shard_dirs = glob.glob(shard_pattern)
    
    if not shard_dirs:
        print("❌ No shards found for validation")
        return False
    
    sample_shard = shard_dirs[0]
    print(f"📁 Validating shard: {sample_shard}")
    
    # Check nodes schema
    nodes_file = os.path.join(sample_shard, "nodes.parquet")
    edges_file = os.path.join(sample_shard, "edges.parquet")
    
    results = {
        "nodes_schema_valid": False,
        "edges_schema_valid": False,
        "nodes_features": [],
        "edges_features": [],
        "issues": []
    }
    
    if os.path.exists(nodes_file):
        print(f"\n📊 Analyzing nodes.parquet ({os.path.getsize(nodes_file)} bytes)")
        
        node_columns = read_parquet_schema_basic(nodes_file)
        if isinstance(node_columns, str):  # Error case
            print(f"❌ {node_columns}")
            results["issues"].append(f"Nodes schema read error: {node_columns}")
        else:
            # Extract feature columns
            node_features = [col for col in node_columns if col.startswith('f') and col[1:].isdigit()]
            results["nodes_features"] = sorted(node_features)
            
            print(f"🔢 Node feature columns found: {len(node_features)}")
            print(f"📋 Features: {', '.join(node_features[:10])}{'...' if len(node_features) > 10 else ''}")
            
            # Validate against contract (51D: f0..f50)
            expected_features = [f'f{i}' for i in range(51)]
            missing_features = [f for f in expected_features if f not in node_features]
            extra_features = [f for f in node_features if f not in expected_features]
            
            if len(node_features) == 51 and not missing_features:
                print("✅ Node schema matches contract (51D: f0..f50)")
                results["nodes_schema_valid"] = True
            else:
                print(f"❌ Node schema mismatch:")
                print(f"   Expected: 51 features (f0..f50)")
                print(f"   Found: {len(node_features)} features")
                if missing_features:
                    print(f"   Missing: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
                if extra_features:
                    print(f"   Extra: {extra_features[:5]}{'...' if len(extra_features) > 5 else ''}")
                
                results["issues"].append(f"Node schema: expected 51D (f0..f50), found {len(node_features)}D")
    else:
        print("❌ nodes.parquet not found")
        results["issues"].append("Missing nodes.parquet file")
    
    if os.path.exists(edges_file):
        print(f"\n📊 Analyzing edges.parquet ({os.path.getsize(edges_file)} bytes)")
        
        edge_columns = read_parquet_schema_basic(edges_file)
        if isinstance(edge_columns, str):  # Error case
            print(f"❌ {edge_columns}")
            results["issues"].append(f"Edges schema read error: {edge_columns}")
        else:
            # Extract feature columns
            edge_features = [col for col in edge_columns if col.startswith('e') and col[1:].isdigit()]
            results["edges_features"] = sorted(edge_features)
            
            print(f"🔢 Edge feature columns found: {len(edge_features)}")
            print(f"📋 Features: {', '.join(edge_features[:10])}{'...' if len(edge_features) > 10 else ''}")
            
            # Show all columns for debugging
            print(f"🔍 All edge columns: {', '.join(edge_columns)}")
            
            # Validate against contract (20D: e0..e19)
            expected_features = [f'e{i}' for i in range(20)]
            missing_features = [f for f in expected_features if f not in edge_features]
            extra_features = [f for f in edge_features if f not in expected_features]
            
            if len(edge_features) == 20 and not missing_features:
                print("✅ Edge schema matches contract (20D: e0..e19)")
                results["edges_schema_valid"] = True
            else:
                print(f"❌ Edge schema mismatch:")
                print(f"   Expected: 20 features (e0..e19)")
                print(f"   Found: {len(edge_features)} features")
                if missing_features:
                    print(f"   Missing: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
                if extra_features:
                    print(f"   Extra: {extra_features[:5]}{'...' if len(extra_features) > 5 else ''}")
                
                results["issues"].append(f"Edge schema: expected 20D (e0..e19), found {len(edge_features)}D")
    else:
        print("❌ edges.parquet not found")
        results["issues"].append("Missing edges.parquet file")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SCHEMA VALIDATION SUMMARY")
    print("=" * 60)
    
    if results["nodes_schema_valid"] and results["edges_schema_valid"]:
        print("🎉 All schema contracts VALIDATED")
        print("✅ Nodes: 51D (f0..f50) ✅ Edges: 20D (e0..e19)")
        return True
    else:
        print("⚠️  Schema contract violations detected:")
        for issue in results["issues"]:
            print(f"   • {issue}")
        
        print("\n💡 Resolution required:")
        if not results["nodes_schema_valid"]:
            print("   1. Update node schema to 51D (f0..f50) OR update documentation")
        if not results["edges_schema_valid"]:
            print("   2. Update edge schema to 20D (e0..e19) OR update documentation")
        
        return False


def check_module_structure():
    """Check if core modules exist"""
    print("\n🔍 MODULE STRUCTURE VALIDATION")
    print("=" * 60)
    
    required_modules = [
        "ironforge/__init__.py",
        "ironforge/learning/__init__.py", 
        "ironforge/learning/discovery_pipeline.py",
        "ironforge/confluence/__init__.py",
        "ironforge/confluence/scoring.py",
        "ironforge/validation/__init__.py",
        "ironforge/validation/runner.py",
        "ironforge/reporting/__init__.py",
        "ironforge/reporting/minidash.py",
        "ironforge/sdk/__init__.py",
        "ironforge/sdk/cli.py"
    ]
    
    missing_modules = []
    for module in required_modules:
        if os.path.exists(module):
            print(f"✅ {module}")
        else:
            print(f"❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing {len(missing_modules)} required modules")
        return False
    else:
        print("\n🎉 All required modules present")
        return True


def main():
    """Run all validations"""
    print("🏛️ IRONFORGE PRE-MERGE VALIDATION")
    print("=" * 80)
    
    schema_valid = validate_schema_contracts()
    modules_valid = check_module_structure()
    
    print("\n" + "=" * 80)
    print("🎯 VALIDATION RESULTS")
    print("=" * 80)
    
    if schema_valid and modules_valid:
        print("🎉 ALL VALIDATIONS PASSED")
        print("✅ Ready for merge")
        return 0
    else:
        print("⚠️  VALIDATION FAILURES DETECTED")
        print("❌ Resolve issues before merge")
        return 1


if __name__ == "__main__":
    exit(main())
