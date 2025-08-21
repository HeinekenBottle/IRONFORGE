#!/usr/bin/env python3
"""
IRONFORGE Unified Script Runner v1.1.0
======================================

Consolidated runner for all IRONFORGE analysis workflows.
Replaces the deprecated run_*.py scripts with a unified interface.

Usage:
    python scripts/unified_runner.py <workflow> [options]

Available workflows:
    - discovery: Run temporal pattern discovery
    - confluence: Score session confluence
    - validation: Validate run results
    - reporting: Generate minidash reports
    - oracle: Train or run Oracle predictions
    - analysis: Run various analysis workflows

Examples:
    python scripts/unified_runner.py discovery --config configs/dev.yml
    python scripts/unified_runner.py confluence --config configs/dev.yml
    python scripts/unified_runner.py oracle --train --symbols NQ --tf M5
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class UnifiedRunner:
    """Unified runner for IRONFORGE workflows."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        
    def run_discovery(self, config: Optional[str] = None, **kwargs) -> int:
        """Run temporal pattern discovery."""
        cmd = ["python", "-m", "ironforge.sdk.cli", "discover-temporal"]
        if config:
            cmd.extend(["--config", config])
        return subprocess.call(cmd, cwd=self.root_dir)
    
    def run_confluence(self, config: Optional[str] = None, **kwargs) -> int:
        """Run confluence scoring."""
        cmd = ["python", "-m", "ironforge.sdk.cli", "score-session"]
        if config:
            cmd.extend(["--config", config])
        return subprocess.call(cmd, cwd=self.root_dir)
    
    def run_validation(self, config: Optional[str] = None, **kwargs) -> int:
        """Run validation."""
        cmd = ["python", "-m", "ironforge.sdk.cli", "validate-run"]
        if config:
            cmd.extend(["--config", config])
        return subprocess.call(cmd, cwd=self.root_dir)
    
    def run_reporting(self, config: Optional[str] = None, **kwargs) -> int:
        """Generate reports."""
        cmd = ["python", "-m", "ironforge.sdk.cli", "report-minimal"]
        if config:
            cmd.extend(["--config", config])
        return subprocess.call(cmd, cwd=self.root_dir)
    
    def run_oracle(self, train: bool = False, symbols: Optional[str] = None, 
                   tf: Optional[str] = None, **kwargs) -> int:
        """Train or run Oracle predictions."""
        if train:
            cmd = ["ironforge", "train-oracle"]
            if symbols:
                cmd.extend(["--symbols", symbols])
            if tf:
                cmd.extend(["--tf", tf])
            cmd.extend(["--out", "models/oracle/v1.1.0"])
        else:
            cmd = ["ironforge", "discover-temporal", "--oracle-enabled"]
        return subprocess.call(cmd, cwd=self.root_dir)
    
    def run_analysis(self, analysis_type: str = "comprehensive", **kwargs) -> int:
        """Run analysis workflows."""
        analysis_scripts = {
            "comprehensive": "scripts/analysis/comprehensive_discovery_report.py",
            "patterns": "scripts/analysis/real_pattern_finder.py",
            "archaeology": "scripts/analysis/run_archaeology_demonstration.py",
            "validation": "scripts/analysis/phase5_archaeological_discovery_validation.py"
        }
        
        script_path = analysis_scripts.get(analysis_type)
        if not script_path or not (self.root_dir / script_path).exists():
            print(f"❌ Analysis type '{analysis_type}' not found or script missing")
            return 1
            
        cmd = ["python", script_path]
        return subprocess.call(cmd, cwd=self.root_dir)
    
    def run_full_pipeline(self, config: Optional[str] = None, **kwargs) -> int:
        """Run the complete IRONFORGE pipeline."""
        print("🚀 Running complete IRONFORGE pipeline...")
        
        steps = [
            ("Discovery", self.run_discovery),
            ("Confluence", self.run_confluence),
            ("Validation", self.run_validation),
            ("Reporting", self.run_reporting)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📊 Running {step_name}...")
            result = step_func(config=config)
            if result != 0:
                print(f"❌ {step_name} failed with exit code {result}")
                return result
            print(f"✅ {step_name} completed successfully")
        
        print("\n🎉 Complete pipeline finished successfully!")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="IRONFORGE Unified Script Runner v1.1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "workflow",
        choices=["discovery", "confluence", "validation", "reporting", 
                "oracle", "analysis", "pipeline"],
        help="Workflow to run"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train Oracle (for oracle workflow)"
    )
    
    parser.add_argument(
        "--symbols",
        help="Symbols for Oracle training"
    )
    
    parser.add_argument(
        "--tf",
        help="Timeframe for Oracle training"
    )
    
    parser.add_argument(
        "--analysis-type",
        default="comprehensive",
        choices=["comprehensive", "patterns", "archaeology", "validation"],
        help="Type of analysis to run"
    )
    
    args = parser.parse_args()
    
    runner = UnifiedRunner()
    
    # Map workflow to method
    workflow_map = {
        "discovery": runner.run_discovery,
        "confluence": runner.run_confluence,
        "validation": runner.run_validation,
        "reporting": runner.run_reporting,
        "oracle": runner.run_oracle,
        "analysis": runner.run_analysis,
        "pipeline": runner.run_full_pipeline
    }
    
    workflow_func = workflow_map[args.workflow]
    
    # Convert args to kwargs
    kwargs = {
        "config": args.config,
        "train": args.train,
        "symbols": args.symbols,
        "tf": args.tf,
        "analysis_type": args.analysis_type
    }
    
    try:
        result = workflow_func(**kwargs)
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⚠️  Workflow interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
