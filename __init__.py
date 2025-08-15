"""
IRONFORGE Package Initialization
===============================
Ensures proper iron_core package visibility for IRONFORGE components.
"""

import sys
import os

# Add parent directory to path to find iron_core package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)