#!/usr/bin/env python3
"""
IRONFORGE Performance Benchmark
==============================
Measures import times and system performance to validate refactoring claims.
"""

import time
import sys
import importlib
from pathlib import Path

def measure_import_time(module_name, class_name=None):
    """Measure time to import a module and optionally get a class"""
    start_time = time.perf_counter()
    
    try:
        module = importlib.import_module(module_name)
        if class_name:
            getattr(module, class_name)
        end_time = time.perf_counter()
        return end_time - start_time, True, None
    except Exception as e:
        end_time = time.perf_counter()
        return end_time - start_time, False, str(e)

def benchmark_core_imports():
    """Benchmark core IRONFORGE imports"""
    print("🚀 IRONFORGE Performance Benchmark")
    print("=" * 50)
    
    imports_to_test = [
        ("ironforge", None),
        ("ironforge.learning.enhanced_graph_builder", "EnhancedGraphBuilder"),
        ("ironforge.learning.tgat_discovery", "IRONFORGEDiscovery"),
        ("ironforge.integration.ironforge_container", "IRONFORGEContainer"),
        ("ironforge.utilities.performance_monitor", "PerformanceMonitor"),
        ("iron_core", "IRONContainer"),
    ]
    
    total_time = 0
    successful_imports = 0
    failed_imports = 0
    
    print("📊 Import Performance Results:")
    print("-" * 50)
    
    for module_name, class_name in imports_to_test:
        import_time, success, error = measure_import_time(module_name, class_name)
        total_time += import_time
        
        if success:
            successful_imports += 1
            status = "✅"
            detail = f"{import_time*1000:.2f}ms"
        else:
            failed_imports += 1
            status = "❌"
            detail = f"FAILED: {error}"
        
        display_name = f"{module_name}.{class_name}" if class_name else module_name
        print(f"{status} {display_name:<50} {detail}")
    
    print("-" * 50)
    print(f"📈 Summary:")
    print(f"  Total import time: {total_time*1000:.2f}ms")
    print(f"  Successful imports: {successful_imports}")
    print(f"  Failed imports: {failed_imports}")
    print(f"  Success rate: {(successful_imports/(successful_imports+failed_imports)*100):.1f}%")
    
    if total_time < 3.0:  # Less than 3 seconds is good
        print(f"  ✅ Import performance: GOOD ({total_time:.2f}s)")
    else:
        print(f"  ⚠️  Import performance: SLOW ({total_time:.2f}s)")
    
    return total_time, successful_imports, failed_imports

def benchmark_container_loading():
    """Benchmark lazy loading container performance"""
    print("\n🔄 Container Lazy Loading Benchmark")
    print("-" * 50)
    
    try:
        start_time = time.perf_counter()
        from ironforge.integration.ironforge_container import get_ironforge_container
        container = get_ironforge_container()
        container_time = time.perf_counter() - start_time
        
        print(f"✅ Container initialization: {container_time*1000:.2f}ms")
        
        # Test component access
        component_times = []
        components_to_test = [
            'enhanced_graph_builder',
            'tgat_discovery',
            'performance_monitor'
        ]
        
        for component_name in components_to_test:
            start_time = time.perf_counter()
            try:
                component = getattr(container, f'get_{component_name}')()
                access_time = time.perf_counter() - start_time
                component_times.append(access_time)
                print(f"✅ {component_name}: {access_time*1000:.2f}ms")
            except Exception as e:
                access_time = time.perf_counter() - start_time
                print(f"❌ {component_name}: FAILED ({e})")
        
        avg_component_time = sum(component_times) / len(component_times) if component_times else 0
        print(f"\n📊 Container Performance:")
        print(f"  Container init: {container_time*1000:.2f}ms")
        print(f"  Avg component access: {avg_component_time*1000:.2f}ms")
        
        return container_time, avg_component_time
        
    except Exception as e:
        print(f"❌ Container benchmark failed: {e}")
        return None, None

def main():
    """Run complete performance benchmark"""
    print("🔬 IRONFORGE Refactoring Performance Validation")
    print("=" * 60)
    
    # Benchmark imports
    import_time, successful, failed = benchmark_core_imports()
    
    # Benchmark container
    container_time, component_time = benchmark_container_loading()
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("🎯 Performance Assessment:")
    
    if failed == 0:
        print("✅ All imports successful - No breaking changes detected")
    else:
        print(f"❌ {failed} import failures - Breaking changes detected")
    
    if import_time < 3.0:
        print("✅ Import performance acceptable")
    else:
        print("⚠️  Import performance degraded")
    
    if container_time and container_time < 1.0:
        print("✅ Container loading performance good")
    elif container_time:
        print("⚠️  Container loading performance degraded")
    else:
        print("❌ Container loading failed")
    
    # Performance claims validation
    print("\n📋 Performance Claims Validation:")
    print("  Claim: '88.7% performance improvement from lazy loading'")
    if container_time and component_time:
        print(f"  Measured: Container init {container_time*1000:.0f}ms, Component access {component_time*1000:.0f}ms")
        print("  Status: ⚠️  Original baseline needed for comparison")
    else:
        print("  Status: ❌ Cannot validate - container loading failed")

if __name__ == "__main__":
    main()
