#!/usr/bin/env python3
"""Test IRONFORGE imports after refactoring"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    print("Testing ironforge package import...")
    import ironforge
    print(f"✅ ironforge package imported successfully: {ironforge.__version__}")
    
    print("Testing integration module...")
    from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
    print("✅ integration module imported successfully")
    
    print("Testing learning module...")
    from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
    print("✅ learning module imported successfully")
    
    print("Testing analysis module...")
    from ironforge.analysis.timeframe_lattice_mapper import TimeframeLatticeMapper
    print("✅ analysis module (with improvements) imported successfully")
    
    print("\n🎉 All IRONFORGE refactored imports working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)