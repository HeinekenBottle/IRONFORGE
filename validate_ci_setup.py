#!/usr/bin/env python3
"""
Local CI Validation Script
==========================
Test the same commands that CI will run to ensure local/CI parity.
"""

import subprocess  # nosec B404
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        )  # nosec B602
        if result.returncode == 0:
            print(f"   ✅ {description} - PASSED")
            return True
        else:
            print(f"   ❌ {description} - FAILED")
            if result.stdout:
                print(f"   STDOUT: {result.stdout[:200]}...")
            if result.stderr:
                print(f"   STDERR: {result.stderr[:200]}...")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏰ {description} - TIMEOUT (>60s)")
        return False
    except Exception as e:
        print(f"   💥 {description} - ERROR: {e}")
        return False


def main() -> int:
    print("🚀 IRONFORGE CI Validation")
    print("=" * 50)

    # Check we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Not in IRONFORGE root directory!")
        return 1

    # List of validation steps
    checks = [
        ("ruff check ironforge --quiet", "Ruff linting (production code)"),
        ("ruff check tests --quiet || true", "Ruff linting (tests - allowed to have warnings)"),
        ("black --check ironforge --quiet", "Black formatting (production code)"),
        (
            "black --check tests --quiet || true",
            "Black formatting (tests - allowed to need reformatting)",
        ),
        ("mypy ironforge --no-error-summary", "MyPy type checking"),
        ("python -c 'import ironforge; print(\"✅ Import test passed\")'", "Basic import test"),
    ]

    passed = 0
    total = len(checks)

    for cmd, desc in checks:
        if run_command(cmd, desc):
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All validation checks PASSED!")
        print("💡 Your code should pass CI lint-type job")
        return 0
    else:
        print("⚠️  Some checks failed - fix these before committing")
        print("💡 The new CI setup will show you exactly which step failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
