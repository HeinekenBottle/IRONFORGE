#!/usr/bin/env python3
"""
Modern version bumping script that works with pyproject.toml and __version__.py

Usage:
    python scripts/bump_version_modern.py 1.0.2
"""

import re
import sys
import pathlib
import subprocess
import tomllib
import tomli_w


def main():
    if len(sys.argv) != 2:
        print("usage: bump_version_modern.py <new-version>")
        sys.exit(1)
    
    new_version = sys.argv[1]
    
    # Update __version__.py
    version_file = pathlib.Path("ironforge/__version__.py")
    version_txt = version_file.read_text(encoding="utf-8")
    new_version_txt = re.sub(r'__version__\s*=\s*".*?"', f'__version__ = "{new_version}"', version_txt)
    
    if version_txt == new_version_txt:
        print(f"Version already at {new_version}")
        return
        
    # Update version tuple
    version_parts = [int(x) for x in new_version.split('.')]
    version_tuple = ', '.join(str(x) for x in version_parts)
    new_version_txt = re.sub(
        r'__version_info__\s*=\s*\([^)]+\)',
        f'__version_info__ = ({version_tuple})',
        new_version_txt
    )
    
    version_file.write_text(new_version_txt, encoding="utf-8")
    
    # Read and verify pyproject.toml uses dynamic versioning
    pyproject_file = pathlib.Path("pyproject.toml")
    with open(pyproject_file, "rb") as f:
        pyproject_data = tomllib.load(f)
    
    if "dynamic" not in pyproject_data.get("project", {}) or "version" not in pyproject_data["project"]["dynamic"]:
        print("Warning: pyproject.toml does not use dynamic versioning")
        print("Expected: [project] dynamic = ['version']")
    
    # Commit and tag
    subprocess.check_call(["git", "add", str(version_file)])
    subprocess.check_call(["git", "commit", "-m", f"chore: bump version to {new_version}"])
    subprocess.check_call(["git", "tag", f"v{new_version}"])
    
    print(f"✅ Bumped to {new_version}")
    print(f"📦 Ready to rebuild wheel with: python -m build")
    print(f"🚀 Push with: git push origin main --tags")


if __name__ == "__main__":
    main()