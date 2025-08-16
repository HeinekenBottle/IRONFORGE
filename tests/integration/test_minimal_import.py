#!/usr/bin/env python3
"""
Minimal import test to isolate timeout issues in Sprint 2 components
"""

import sys
import time
sys.path.append('.')

def test_import(module_name, class_name=None):
    """Test importing a specific module/class with timing"""
    start_time = time.time()
    try:
        print(f"Testing import: {module_name}" + (f".{class_name}" if class_name else ""))
        
        if class_name:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} imported in {time.time() - start_time:.2f}s")
        else:
            __import__(module_name)
            print(f"✅ {module_name} imported in {time.time() - start_time:.2f}s")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {module_name}" + (f".{class_name}" if class_name else "") + f" failed in {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Minimal Import Test for Sprint 2 Components")
    print("=" * 50)
    
    # Test imports one by one
    test_import("learning.enhanced_graph_builder", "EnhancedGraphBuilder")
    test_import("learning.tgat_discovery", "IRONFORGEDiscovery")  
    test_import("learning.regime_segmentation", "RegimeSegmentation")
    test_import("learning.precursor_detection", "EventPrecursorDetector")
    test_import("performance_monitor", "PerformanceMonitor")
    test_import("reporting.analyst_reports", "AnalystReports")
    
    print("\n✅ All imports completed!")