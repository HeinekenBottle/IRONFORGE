"""
Import Performance Tests
========================

Tests to ensure lazy loading and import performance meet <2s initialization gate.
"""

import time
import subprocess
import sys
import pytest
from pathlib import Path


class TestImportPerformance:
    """Test import performance and lazy loading."""
    
    def test_top_level_import_speed(self):
        """Test top-level import meets performance gate."""
        start_time = time.time()
        
        import ironforge
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # Should be very fast due to lazy loading
        assert import_time < 0.5, f"Top-level import took {import_time:.3f}s, expected <0.5s"
        print(f"✅ Top-level import: {import_time:.3f}s")
    
    def test_api_import_speed(self):
        """Test API import performance."""
        start_time = time.time()
        
        from ironforge.api import run_discovery, score_confluence, validate_run, build_minidash
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # API imports should be fast
        assert import_time < 1.0, f"API import took {import_time:.3f}s, expected <1.0s"
        print(f"✅ API import: {import_time:.3f}s")
    
    def test_container_initialization_speed(self):
        """Test container initialization performance."""
        start_time = time.time()
        
        from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
        container = initialize_ironforge_lazy_loading()
        
        end_time = time.time()
        init_time = end_time - start_time
        
        # Container initialization should be fast
        assert init_time < 2.0, f"Container init took {init_time:.3f}s, expected <2.0s"
        print(f"✅ Container initialization: {init_time:.3f}s")
    
    def test_lazy_component_access(self):
        """Test lazy component access performance."""
        from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
        
        container = initialize_ironforge_lazy_loading()
        
        # Test accessing components
        components = [
            'get_enhanced_graph_builder',
            'get_tgat_discovery',
        ]
        
        total_time = 0
        for component_name in components:
            if hasattr(container, component_name):
                start_time = time.time()
                component = getattr(container, component_name)()
                end_time = time.time()
                
                access_time = end_time - start_time
                total_time += access_time
                
                print(f"✅ {component_name}: {access_time:.3f}s")
        
        # Total component access should be reasonable
        assert total_time < 5.0, f"Total component access took {total_time:.3f}s, expected <5.0s"
    
    def test_cold_start_subprocess(self):
        """Test cold start performance in subprocess."""
        # Test basic import in clean subprocess
        start_time = time.time()
        
        result = subprocess.run([
            sys.executable, "-c", 
            "import ironforge; print('Import successful')"
        ], capture_output=True, text=True, timeout=10)
        
        end_time = time.time()
        cold_start_time = end_time - start_time
        
        assert result.returncode == 0, f"Cold start failed: {result.stderr}"
        assert "Import successful" in result.stdout
        
        # Cold start should meet the 2s gate
        assert cold_start_time < 2.0, f"Cold start took {cold_start_time:.3f}s, expected <2.0s"
        print(f"✅ Cold start: {cold_start_time:.3f}s")
    
    def test_api_usage_subprocess(self):
        """Test API usage performance in subprocess."""
        # Test API usage in clean subprocess
        start_time = time.time()
        
        result = subprocess.run([
            sys.executable, "-c", 
            """
from ironforge.api import load_config, initialize_ironforge_lazy_loading
container = initialize_ironforge_lazy_loading()
print('API usage successful')
"""
        ], capture_output=True, text=True, timeout=15)
        
        end_time = time.time()
        api_usage_time = end_time - start_time
        
        assert result.returncode == 0, f"API usage failed: {result.stderr}"
        assert "API usage successful" in result.stdout
        
        # API usage should be reasonable
        assert api_usage_time < 5.0, f"API usage took {api_usage_time:.3f}s, expected <5.0s"
        print(f"✅ API usage: {api_usage_time:.3f}s")


class TestImportOptimization:
    """Test import optimization and lazy loading effectiveness."""
    
    def test_heavy_imports_are_lazy(self):
        """Test that heavy imports are properly lazy-loaded."""
        # These should not be imported at top level
        heavy_modules = [
            'torch',
            'networkx', 
            'pandas',
            'numpy',
            'matplotlib',
            'plotly',
        ]
        
        # Import ironforge and check sys.modules
        import ironforge
        import sys
        
        # Count how many heavy modules were imported
        imported_heavy = [mod for mod in heavy_modules if mod in sys.modules]
        
        # Some heavy modules might be imported, but not all
        heavy_ratio = len(imported_heavy) / len(heavy_modules)
        
        print(f"Heavy modules imported: {imported_heavy}")
        print(f"Heavy import ratio: {heavy_ratio:.2f}")
        
        # Should have lazy loading effectiveness
        assert heavy_ratio < 0.8, f"Too many heavy modules imported: {heavy_ratio:.2f}"
    
    def test_module_import_order(self):
        """Test that modules are imported in optimal order."""
        import sys
        
        # Clear any existing ironforge modules
        to_remove = [mod for mod in sys.modules.keys() if mod.startswith('ironforge')]
        for mod in to_remove:
            del sys.modules[mod]
        
        start_time = time.time()
        
        # Import in recommended order
        import ironforge
        from ironforge.api import load_config
        from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
        
        end_time = time.time()
        ordered_import_time = end_time - start_time
        
        print(f"✅ Ordered imports: {ordered_import_time:.3f}s")
        
        # Should be fast due to proper ordering
        assert ordered_import_time < 1.0, f"Ordered imports took {ordered_import_time:.3f}s"


class TestMemoryEfficiency:
    """Test memory efficiency of imports."""
    
    def test_import_memory_usage(self):
        """Test memory usage of imports."""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Import ironforge
        import ironforge
        
        gc.collect()
        after_import_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        import_memory_cost = after_import_memory - baseline_memory
        
        print(f"Import memory cost: {import_memory_cost:.1f}MB")
        
        # Import should be memory-efficient
        assert import_memory_cost < 50.0, f"Import used {import_memory_cost:.1f}MB, expected <50MB"
    
    def test_container_memory_usage(self):
        """Test container initialization memory usage."""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Import first
        import ironforge
        from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
        
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize container
        container = initialize_ironforge_lazy_loading()
        
        gc.collect()
        after_container_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        container_memory_cost = after_container_memory - baseline_memory
        
        print(f"Container memory cost: {container_memory_cost:.1f}MB")
        
        # Container should be memory-efficient
        assert container_memory_cost < 30.0, f"Container used {container_memory_cost:.1f}MB, expected <30MB"


def test_import_performance_summary():
    """Print import performance summary."""
    print("\n" + "="*60)
    print("IRONFORGE Import Performance Summary")
    print("="*60)
    print("Performance Gates:")
    print("  - Top-level import: <0.5s")
    print("  - API import: <1.0s") 
    print("  - Container init: <2.0s")
    print("  - Cold start: <2.0s")
    print("  - Import memory: <50MB")
    print("  - Container memory: <30MB")
    print("="*60)
