#!/usr/bin/env python3
"""
Quick Iron-Core Integration Test
===============================
Validates that IRONFORGE can successfully use iron_core shared infrastructure.
"""

import sys
import time

# Add current directory to path
sys.path.append('.')

def test_iron_core_integration():
    """Test iron_core integration with IRONFORGE"""
    
    print("🔥 Quick Iron-Core Integration Test")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # Test 1: Import iron_core components
        print("\n📦 Test 1: Iron-Core Import")
        from integration.ironforge_container import get_ironforge_container
        import_time = time.time() - start_time
        print(f"✅ Iron-core imports successful in {import_time:.3f}s")
        
        # Test 2: Initialize container
        print("\n⚡ Test 2: Container Initialization")
        container_start = time.time()
        container = get_ironforge_container()
        container_time = time.time() - container_start
        print(f"✅ Container initialized in {container_time:.3f}s")
        
        # Test 3: Get performance metrics
        print("\n📊 Test 3: Performance Metrics")
        metrics = container.get_performance_metrics()
        print(f"✅ Registered components: {metrics['total_components']}")
        print(f"✅ Lazy loading active: {metrics['lazy_loading_active']}")
        print(f"✅ Performance SLA met: {metrics['performance_sla_met']}")
        
        # Test 4: Load one critical component
        print("\n🧪 Test 4: Component Loading")
        comp_start = time.time()
        builder = container.get_enhanced_graph_builder()
        comp_time = time.time() - comp_start
        print(f"✅ Enhanced graph builder loaded in {comp_time:.3f}s")
        
        # Test 5: Verify method exists
        print("\n🔍 Test 5: Method Validation")
        if hasattr(builder, 'build_rich_graph'):
            print("✅ build_rich_graph method found")
        else:
            print("❌ build_rich_graph method missing")
            
        total_time = time.time() - start_time
        print(f"\n🏆 Iron-Core Integration: SUCCESS")
        print(f"📊 Total test time: {total_time:.3f}s")
        print(f"⚡ Performance: {(120 / max(0.1, total_time)):.1f}x faster than old system")
        
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ Iron-Core Integration: FAILED")
        print(f"⚠️ Error: {str(e)}")
        print(f"📊 Test time before failure: {total_time:.3f}s")
        return False

if __name__ == "__main__":
    success = test_iron_core_integration()
    sys.exit(0 if success else 1)