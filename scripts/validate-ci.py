#!/usr/bin/env python3
"""
Pre-commit validation script to ensure CI will pass.
Run this before pushing changes to avoid CI failures.

Usage:
    python scripts/validate-ci.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success/failure."""
    print(f"\n🔍 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed with exit code {e.returncode}")
        if e.stdout:
            print(f"   STDOUT: {e.stdout}")
        if e.stderr:
            print(f"   STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"   ❌ Command not found: {cmd[0]}")
        return False


def check_tool_availability():
    """Check if all required tools are available."""
    tools = [
        ("ruff", "Ruff linter"),
        ("black", "Black formatter"),  
        ("mypy", "MyPy type checker"),
        ("pytest", "PyTest test runner")
    ]
    
    print("🛠️  Checking tool availability...")
    all_available = True
    
    for tool, description in tools:
        if run_command([tool, "--version"], f"Checking {description}"):
            continue
        else:
            all_available = False
            
    return all_available


def main():
    """Main validation workflow."""
    print("🚀 IRONFORGE CI Validation")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    original_dir = Path.cwd()
    
    try:
        import os
        os.chdir(project_root)
        
        # Check tool availability
        if not check_tool_availability():
            print("\n❌ Some tools are missing. Run: pip install -r requirements-dev.txt")
            return 1
            
        # Run all checks
        checks = [
            (["ruff", "check", "ironforge", "tests"], "Running ruff linter"),
            (["black", "--check", "ironforge", "tests"], "Checking black formatting"),
            (["mypy", "ironforge"], "Running mypy type checking"),
            (["pytest", "-q", "--tb=short"], "Running tests")
        ]
        
        failed_checks = []
        
        for cmd, description in checks:
            if not run_command(cmd, description):
                failed_checks.append(description)
                
        # Summary
        print("\n" + "=" * 50)
        if failed_checks:
            print("❌ CI Validation FAILED")
            print("Failed checks:")
            for check in failed_checks:
                print(f"   • {check}")
            print("\nFix these issues before pushing to avoid CI failures.")
            return 1
        else:
            print("✅ CI Validation PASSED")
            print("All checks passed! Safe to push changes.")
            return 0
            
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        return 1
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    sys.exit(main())