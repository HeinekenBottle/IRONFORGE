#!/usr/bin/env python3
"""
⚠️  DEPRECATED: run_fpfvg_network_analysis.py
=============================

This script has been moved to scripts/analysis/legacy_runners/

For new workflows, use the canonical CLI:
    python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
    python -m ironforge.sdk.cli score-session     --config configs/dev.yml
    python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml

To run the legacy script:
    python scripts/analysis/legacy_runners/run_fpfvg_network_analysis.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("⚠️  DEPRECATED: run_fpfvg_network_analysis.py")
    print("=" * 60)
    print("This script has been moved to scripts/analysis/legacy_runners/")
    print()
    print("Recommended workflow:")
    print("  python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml")
    print("  python -m ironforge.sdk.cli score-session     --config configs/dev.yml")
    print("  python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml")
    print()
    
    legacy_path = Path("scripts/analysis/legacy_runners/run_fpfvg_network_analysis.py")
    if legacy_path.exists():
        print(f"To run legacy script: python {legacy_path}")
        response = input("Run legacy script now? [y/N]: ").strip().lower()
        if response == 'y':
            return subprocess.call([sys.executable, str(legacy_path)])
    else:
        print("❌ Legacy script not found!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())