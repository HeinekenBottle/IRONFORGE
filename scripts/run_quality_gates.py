#!/usr/bin/env python3
"""
IRONFORGE Quality Gates Runner
==============================

Comprehensive quality gate validation script for IRONFORGE.
Runs all contract tests, performance benchmarks, and validation checks.
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import argparse


class QualityGate:
    """Quality gate with validation."""
    
    def __init__(self, name: str, command: List[str], description: str, critical: bool = True):
        self.name = name
        self.command = command
        self.description = description
        self.critical = critical
        self.result = None
        self.duration = None
        self.output = None
        self.error = None
    
    def run(self) -> bool:
        """Run the quality gate."""
        print(f"\n{'='*60}")
        print(f"Running: {self.name}")
        print(f"Description: {self.description}")
        print(f"Command: {' '.join(self.command)}")
        print(f"Critical: {'Yes' if self.critical else 'No'}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            self.duration = time.time() - start_time
            self.output = result.stdout
            self.error = result.stderr
            self.result = result.returncode == 0
            
            if self.result:
                print(f"✅ PASSED ({self.duration:.2f}s)")
            else:
                print(f"❌ FAILED ({self.duration:.2f}s)")
                if self.error:
                    print(f"Error: {self.error}")
            
            return self.result
            
        except subprocess.TimeoutExpired:
            self.duration = time.time() - start_time
            self.result = False
            self.error = "Timeout"
            print(f"⏰ TIMEOUT ({self.duration:.2f}s)")
            return False
        
        except Exception as e:
            self.duration = time.time() - start_time
            self.result = False
            self.error = str(e)
            print(f"💥 ERROR ({self.duration:.2f}s): {e}")
            return False


def get_quality_gates() -> List[QualityGate]:
    """Get all quality gates."""
    return [
        # Contract Tests - Critical
        QualityGate(
            name="Golden Invariants Validation",
            command=["python", "-m", "pytest", "tests/contracts/test_golden_invariants.py", "-v"],
            description="Validate Golden Invariants: 6 event types, 4 edge intents, feature dimensions",
            critical=True
        ),
        
        QualityGate(
            name="HTF Compliance Tests",
            command=["python", "-m", "pytest", "tests/contracts/test_htf_compliance.py", "-v"],
            description="Validate HTF compliance: last-closed only, no intra-candle violations",
            critical=True
        ),
        
        QualityGate(
            name="Confluence Configuration Tests",
            command=["python", "-m", "pytest", "tests/contracts/test_confluence_config.py", "-v"],
            description="Validate confluence scoring configuration and DAG weighting feature flag",
            critical=True
        ),
        
        # Performance Tests - Critical
        QualityGate(
            name="Performance Budget Gates",
            command=["python", "-m", "pytest", "tests/performance/test_performance_gates.py", "-v", "-m", "not slow"],
            description="Validate performance gates: <3s session, <180s pipeline, <100MB memory",
            critical=True
        ),
        
        QualityGate(
            name="Import Performance Tests",
            command=["python", "-m", "pytest", "tests/performance/test_import_performance.py", "-v"],
            description="Validate import performance and lazy loading: <2s initialization",
            critical=True
        ),
        
        # Integration Tests - Important but not critical
        QualityGate(
            name="Reporting Pipeline Tests",
            command=["python", "-m", "pytest", "tests/integration/test_reporting.py", "-v"],
            description="Validate reporting pipeline: minidash generation, PNG export",
            critical=False
        ),
        
        QualityGate(
            name="Oracle Integration Tests",
            command=["python", "-m", "pytest", "tests/integration/test_oracle_integration.py", "-v"],
            description="Validate Oracle integration: 16-column schema, sidecar integration",
            critical=False
        ),
        
        # Code Quality - Critical
        QualityGate(
            name="Code Formatting (Black)",
            command=["black", "--check", "."],
            description="Validate code formatting with Black",
            critical=True
        ),
        
        QualityGate(
            name="Code Linting (Ruff)",
            command=["ruff", "check", "."],
            description="Validate code quality with Ruff linting",
            critical=True
        ),
        
        QualityGate(
            name="Type Checking (MyPy)",
            command=["mypy", "ironforge"],
            description="Validate type annotations with MyPy",
            critical=True
        ),
        
        # System Validation - Critical
        QualityGate(
            name="Package Build Test",
            command=["python", "-m", "build"],
            description="Validate package can be built successfully",
            critical=True
        ),
        
        QualityGate(
            name="Import Smoke Test",
            command=["python", "-c", "import ironforge; from ironforge.api import run_discovery; print('✅ Import successful')"],
            description="Validate package can be imported successfully",
            critical=True
        ),
    ]


def run_quality_gates(gates: List[QualityGate], fail_fast: bool = False) -> Dict[str, Any]:
    """Run all quality gates."""
    results = {
        'total': len(gates),
        'passed': 0,
        'failed': 0,
        'critical_failed': 0,
        'gates': [],
        'duration': 0,
    }
    
    start_time = time.time()
    
    for gate in gates:
        success = gate.run()
        
        gate_result = {
            'name': gate.name,
            'passed': success,
            'critical': gate.critical,
            'duration': gate.duration,
            'error': gate.error,
        }
        
        results['gates'].append(gate_result)
        
        if success:
            results['passed'] += 1
        else:
            results['failed'] += 1
            if gate.critical:
                results['critical_failed'] += 1
        
        # Fail fast on critical failures
        if fail_fast and not success and gate.critical:
            print(f"\n💥 FAIL FAST: Critical gate '{gate.name}' failed")
            break
    
    results['duration'] = time.time() - start_time
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print quality gates summary."""
    print(f"\n{'='*80}")
    print("IRONFORGE QUALITY GATES SUMMARY")
    print('='*80)
    
    print(f"Total Gates: {results['total']}")
    print(f"Passed: {results['passed']} ✅")
    print(f"Failed: {results['failed']} ❌")
    print(f"Critical Failed: {results['critical_failed']} 💥")
    print(f"Total Duration: {results['duration']:.2f}s")
    
    print(f"\n{'Gate Results:':<40} {'Status':<10} {'Duration':<10} {'Critical'}")
    print('-'*80)
    
    for gate in results['gates']:
        status = "✅ PASS" if gate['passed'] else "❌ FAIL"
        duration = f"{gate['duration']:.2f}s" if gate['duration'] else "N/A"
        critical = "🔴 YES" if gate['critical'] else "🟡 NO"
        
        print(f"{gate['name']:<40} {status:<10} {duration:<10} {critical}")
        
        if not gate['passed'] and gate['error']:
            print(f"  Error: {gate['error']}")
    
    print('='*80)
    
    if results['critical_failed'] > 0:
        print("💥 CRITICAL FAILURES DETECTED - PIPELINE MUST NOT PROCEED")
        return False
    elif results['failed'] > 0:
        print("⚠️  NON-CRITICAL FAILURES DETECTED - REVIEW RECOMMENDED")
        return True
    else:
        print("🎉 ALL QUALITY GATES PASSED - PIPELINE READY")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run IRONFORGE quality gates")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first critical failure")
    parser.add_argument("--contracts-only", action="store_true", help="Run only contract tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    
    args = parser.parse_args()
    
    # Get all gates
    all_gates = get_quality_gates()
    
    # Filter gates based on arguments
    if args.contracts_only:
        gates = [g for g in all_gates if "contract" in g.name.lower() or "invariant" in g.name.lower() or "htf" in g.name.lower() or "confluence" in g.name.lower()]
    elif args.performance_only:
        gates = [g for g in all_gates if "performance" in g.name.lower() or "import" in g.name.lower()]
    elif args.integration_only:
        gates = [g for g in all_gates if "integration" in g.name.lower() or "reporting" in g.name.lower() or "oracle" in g.name.lower()]
    else:
        gates = all_gates
    
    print(f"Running {len(gates)} quality gates...")
    
    # Run gates
    results = run_quality_gates(gates, fail_fast=args.fail_fast)
    
    # Print summary
    success = print_summary(results)
    
    # Exit with appropriate code
    if results['critical_failed'] > 0:
        sys.exit(1)  # Critical failure
    elif results['failed'] > 0:
        sys.exit(2)  # Non-critical failures
    else:
        sys.exit(0)  # All passed


if __name__ == "__main__":
    main()
