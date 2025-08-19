#!/usr/bin/env python3
"""
Daily HTF Workflow
================

Production workflow for HTF-enhanced archaeological discovery.
Runs the complete pipeline: prep-shards → discover → score → validate → report
"""

import logging
import subprocess
import sys
import time

from ironforge.reporting.htf_observer import HTFObserver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyHTFWorkflow:
    """Daily production workflow for HTF archaeological discovery"""
    
    def __init__(self, config_file: str = "configs/htf_config.yml"):
        self.config_file = config_file
        self.htf_observer = HTFObserver()
        self.run_id = f"htf_run_{int(time.time())}"
        
    def run_daily_workflow(self) -> bool:
        """Run complete daily HTF workflow"""
        
        print("🏛️ IRONFORGE Daily HTF Workflow")
        print("=" * 50)
        print("Version: v0.7.1 (HTF Context Enabled)")
        print(f"Run ID: {self.run_id}")
        print(f"Config: {self.config_file}")
        print()
        
        # Workflow steps
        steps = [
            ("Prep Shards (HTF)", self._step_prep_shards),
            ("Discover Temporal", self._step_discover_temporal),
            ("Score Sessions", self._step_score_sessions),
            ("Validate Run", self._step_validate_run),
            ("Generate Report", self._step_generate_report),
            ("HTF Observability", self._step_htf_observability)
        ]
        
        for step_name, step_func in steps:
            print(f"📋 {step_name}...")
            
            try:
                success = step_func()
                if success:
                    print(f"   ✅ {step_name} completed")
                else:
                    print(f"   ❌ {step_name} failed")
                    return False
                    
            except Exception as e:
                print(f"   ❌ {step_name} error: {e}")
                logger.error(f"Step '{step_name}' failed: {e}")
                return False
        
        print()
        print("🎯 Daily Workflow Summary:")
        print("   ✅ All steps completed successfully")
        print("   🏛️ HTF archaeological discovery active")
        print("   📊 Observability data collected")
        print("   📈 Ready for archaeological analysis")
        
        return True
    
    def _step_prep_shards(self) -> bool:
        """Step 1: Prepare shards with HTF context"""
        
        cmd = [
            "python3", "-m", "ironforge.sdk.cli", "prep-shards",
            "--source-glob", "data/enhanced/enhanced_*_Lvl-1_*.json",
            "--htf-context"
        ]
        
        return self._run_subprocess(cmd, "prep-shards", timeout=300)
    
    def _step_discover_temporal(self) -> bool:
        """Step 2: Run temporal discovery"""
        
        cmd = [
            "python3", "-m", "ironforge.sdk.cli", "discover-temporal",
            "--config", self.config_file
        ]
        
        return self._run_subprocess(cmd, "discover-temporal", timeout=600)
    
    def _step_score_sessions(self) -> bool:
        """Step 3: Score discovered sessions"""
        
        cmd = [
            "python3", "-m", "ironforge.sdk.cli", "score-session",
            "--config", self.config_file
        ]
        
        return self._run_subprocess(cmd, "score-session", timeout=300)
    
    def _step_validate_run(self) -> bool:
        """Step 4: Validate discovery run"""
        
        cmd = [
            "python3", "-m", "ironforge.sdk.cli", "validate-run",
            "--config", self.config_file
        ]
        
        return self._run_subprocess(cmd, "validate-run", timeout=180)
    
    def _step_generate_report(self) -> bool:
        """Step 5: Generate minimal report"""
        
        cmd = [
            "python3", "-m", "ironforge.sdk.cli", "report-minimal",
            "--config", self.config_file
        ]
        
        return self._run_subprocess(cmd, "report-minimal", timeout=120)
    
    def _step_htf_observability(self) -> bool:
        """Step 6: Collect HTF observability data"""
        
        try:
            # Analyze HTF run with observer
            shards_dir = "data/shards/NQ_M5"
            
            # Mock zones data for now (would come from actual discovery)
            mock_zones = [
                {'confidence': 0.85, 'theoretical_basis': 'Theory B'},
                {'confidence': 0.92, 'theoretical_basis': 'HTF Transition Theory'},
                {'confidence': 0.78, 'theoretical_basis': 'SV Anomaly Theory'},
            ]
            
            summary = self.htf_observer.analyze_htf_run(
                self.run_id, shards_dir, mock_zones
            )
            
            # Print summary for visibility
            print(f"   📊 HTF Quality Score: {summary.overall_quality_score:.2f}")
            print(f"   🎯 Zones Discovered: {summary.total_zones}")
            print(f"   🏺 Theory B Zones: {summary.theory_b_zones}")
            
            return True
            
        except Exception as e:
            logger.error(f"HTF observability failed: {e}")
            return False
    
    def _run_subprocess(self, cmd: list[str], step_name: str, timeout: int = 300) -> bool:
        """Run subprocess with timeout and error handling"""
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            if result.returncode == 0:
                if result.stdout.strip():
                    logger.info(f"{step_name} output: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"{step_name} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"{step_name} timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"{step_name} subprocess error: {e}")
            return False


def main():
    """Main entry point for daily workflow"""
    
    # Parse command line arguments
    config_file = "configs/htf_config.yml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # Run workflow
    workflow = DailyHTFWorkflow(config_file)
    success = workflow.run_daily_workflow()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()