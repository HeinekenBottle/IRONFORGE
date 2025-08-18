#!/usr/bin/env python3
"""
Run All Analytics: Comprehensive validation of macro-micro relationship analytics
Executes all three analyses: HTF→trade horizons, cross-session influence, session prototypes
"""
import subprocess
import sys
from pathlib import Path
import json

def run_analysis(script_name: str, description: str):
    """Run an analysis script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: {description}")
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {description} took longer than 5 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def validate_outputs(run_path: Path):
    """Validate that all expected output files were created."""
    print(f"\n{'='*60}")
    print("Validating Output Files")
    print('='*60)
    
    expected_files = [
        ("aux/macro_micro_map.json", "Macro→Micro outcome mapping"),
        ("aux/xsession_candidates.parquet", "Cross-session influence candidates"),
        ("aux/xsession_analysis.json", "Cross-session analysis summary"),
        ("aux/session_prototypes.parquet", "Session prototype transitions"),
        ("aux/session_prototypes_analysis.json", "Session prototype analysis")
    ]
    
    validation_results = []
    
    for file_path, description in expected_files:
        full_path = run_path / file_path
        exists = full_path.exists()
        
        if exists:
            try:
                if file_path.endswith('.json'):
                    with open(full_path) as f:
                        data = json.load(f)
                    size = len(str(data))
                elif file_path.endswith('.parquet'):
                    import pandas as pd
                    df = pd.read_parquet(full_path)
                    size = len(df)
                else:
                    size = full_path.stat().st_size
                
                print(f"✅ {description}: {full_path} (size: {size})")
                validation_results.append(True)
                
            except Exception as e:
                print(f"⚠️  {description}: {full_path} exists but has read error: {e}")
                validation_results.append(False)
        else:
            print(f"❌ {description}: {full_path} NOT FOUND")
            validation_results.append(False)
    
    total_success = sum(validation_results)
    total_expected = len(expected_files)
    
    print(f"\nValidation Summary: {total_success}/{total_expected} files created successfully")
    
    return total_success == total_expected

def create_summary_report(run_path: Path):
    """Create a comprehensive summary report of all analytics."""
    print(f"\n{'='*60}")
    print("Creating Summary Report")
    print('='*60)
    
    summary_data = {
        "analytics_suite": "Macro-Micro Relationship Analytics",
        "run_path": str(run_path),
        "completed_analyses": [],
        "overall_status": "completed"
    }
    
    # Load results from each analysis
    try:
        # Macro-Micro Map
        macro_micro_path = run_path / "aux" / "macro_micro_map.json"
        if macro_micro_path.exists():
            with open(macro_micro_path) as f:
                macro_micro_data = json.load(f)
            
            summary_data["completed_analyses"].append({
                "name": "Macro→Micro Outcome Map",
                "description": "HTF conditions vs trade horizon outcomes",
                "status": "completed",
                "key_findings": {
                    "total_buckets": macro_micro_data.get("valid_buckets", 0),
                    "acceptance_passed": macro_micro_data.get("acceptance_criteria", {}).get("passes", False),
                    "top_hit_rate_bucket": macro_micro_data.get("top_hit_rate_buckets", [{}])[0].get("bucket_name", "N/A") if macro_micro_data.get("top_hit_rate_buckets") else "N/A"
                }
            })
        
        # Cross-Session Influence
        xsession_path = run_path / "aux" / "xsession_analysis.json"
        if xsession_path.exists():
            with open(xsession_path) as f:
                xsession_data = json.load(f)
            
            summary_data["completed_analyses"].append({
                "name": "Cross-Session Influence",
                "description": "Yesterday→today embedding similarity",
                "status": "completed",
                "key_findings": {
                    "total_pairs": xsession_data.get("total_pairs", 0),
                    "acceptance_passed": xsession_data.get("acceptance_criteria", {}).get("passes", False),
                    "influence_effect": xsession_data.get("influence_analysis", {}).get("influence_effect", "N/A")
                }
            })
        
        # Session Prototypes
        prototypes_path = run_path / "aux" / "session_prototypes_analysis.json"
        if prototypes_path.exists():
            with open(prototypes_path) as f:
                prototypes_data = json.load(f)
            
            summary_data["completed_analyses"].append({
                "name": "Session Prototypes",
                "description": "Macro fingerprints for next-session payoffs",
                "status": "completed",
                "key_findings": {
                    "total_sessions": prototypes_data.get("total_sessions", 0),
                    "session_transitions": prototypes_data.get("session_transitions", 0),
                    "correlation_computed": prototypes_data.get("acceptance_criteria", {}).get("correlation_computed", False),
                    "proto_sim_correlation": prototypes_data.get("correlation_analysis", {}).get("proto_sim_vs_next_hit_rate", {}).get("correlation", "N/A")
                }
            })
        
    except Exception as e:
        print(f"Warning: Could not load all analysis results: {e}")
    
    # Save summary report
    summary_path = run_path / "aux" / "analytics_suite_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"✅ Summary report saved: {summary_path}")
    
    # Display summary
    print(f"\n📊 ANALYTICS SUITE SUMMARY")
    print(f"Run Path: {run_path}")
    print(f"Completed Analyses: {len(summary_data['completed_analyses'])}/3")
    
    for analysis in summary_data["completed_analyses"]:
        print(f"\n  • {analysis['name']}")
        print(f"    {analysis['description']}")
        print(f"    Status: {analysis['status']}")
        findings = analysis["key_findings"]
        for key, value in findings.items():
            print(f"    {key}: {value}")
    
    return summary_data

def main():
    """Run comprehensive analytics suite validation."""
    print("🚀 IRONFORGE Macro-Micro Analytics Suite")
    print("Validating HTF→trade horizon relationship analytics")
    
    run_path = Path("runs/real-tgat-fixed-2025-08-18")
    
    if not run_path.exists():
        print(f"❌ Run path not found: {run_path}")
        return False
    
    # Run all three analyses
    analyses = [
        ("scripts/macro_micro_map.py", "Macro→Micro Outcome Map"),
        ("scripts/cross_session_influence.py", "Cross-Session Influence Analysis"),
        ("scripts/session_prototypes.py", "Session Prototypes Analysis")
    ]
    
    results = []
    for script, description in analyses:
        success = run_analysis(script, description)
        results.append(success)
    
    # Validate outputs
    validation_success = validate_outputs(run_path)
    
    # Create summary report
    summary_data = create_summary_report(run_path)
    
    # Final assessment
    total_success = sum(results)
    overall_success = total_success == len(analyses) and validation_success
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)
    print(f"Analysis Success: {total_success}/{len(analyses)}")
    print(f"Output Validation: {'✅' if validation_success else '❌'}")
    print(f"Overall Success: {'✅' if overall_success else '❌'}")
    
    if overall_success:
        print(f"\n🎯 SUCCESS: All macro-micro analytics completed successfully")
        print(f"📁 Results available in: {run_path}/aux/")
        print(f"📋 Summary report: {run_path}/aux/analytics_suite_summary.json")
    else:
        print(f"\n❌ Some analyses failed or outputs missing")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)