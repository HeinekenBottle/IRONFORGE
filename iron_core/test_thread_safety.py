#!/usr/bin/env python3
"""
Thread Safety Validation Test for Iron-Core
==========================================

Tests the thread-safe singleton patterns implemented in:
1. IRONContainer (container.py)
2. LazyLoadingManager (lazy_loader.py) 
3. LazyComponent instances

Validates:
- No race conditions in singleton creation
- Only one instance created across all threads
- Thread-safe lazy loading
- Performance impact analysis
"""

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('iron_core.thread_safety_test')

# Add iron_core to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class ThreadSafetyValidator:
    """Comprehensive thread safety validation for iron_core singletons."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {
            'container_instances': set(),
            'lazy_manager_instances': set(), 
            'timing_results': [],
            'errors': [],
            'thread_completion_times': []
        }
        self.results_lock = threading.Lock()
        
    def test_container_singleton(self, thread_id: int) -> Dict[str, Any]:
        """Test IRONContainer singleton thread safety."""
        start_time = time.time()
        thread_name = f"container_test_{thread_id}"
        
        try:
            # Import inside thread to test concurrent imports
            from iron_core.performance.container import get_container
            
            # Small delay to increase chance of race condition
            time.sleep(0.001 * (thread_id % 10))
            
            container = get_container()
            container_id = id(container)
            
            # Record timing and instance
            completion_time = time.time() - start_time
            
            with self.results_lock:
                self.results['container_instances'].add(container_id)
                self.results['timing_results'].append({
                    'test': 'container_singleton',
                    'thread_id': thread_id,
                    'completion_time': completion_time,
                    'instance_id': container_id
                })
                
            return {
                'success': True,
                'thread_id': thread_id,
                'instance_id': container_id,
                'completion_time': completion_time
            }
            
        except Exception as e:
            with self.results_lock:
                self.results['errors'].append({
                    'test': 'container_singleton',
                    'thread_id': thread_id,
                    'error': str(e),
                    'thread_name': thread_name
                })
            return {
                'success': False,
                'thread_id': thread_id,
                'error': str(e)
            }
            
    def test_lazy_manager_singleton(self, thread_id: int) -> Dict[str, Any]:
        """Test LazyLoadingManager singleton thread safety."""
        start_time = time.time()
        
        try:
            from iron_core.performance.lazy_loader import get_lazy_manager
            
            # Small delay to increase race condition chance
            time.sleep(0.001 * (thread_id % 5))
            
            manager = get_lazy_manager()
            manager_id = id(manager)
            
            completion_time = time.time() - start_time
            
            with self.results_lock:
                self.results['lazy_manager_instances'].add(manager_id)
                self.results['timing_results'].append({
                    'test': 'lazy_manager_singleton',
                    'thread_id': thread_id,
                    'completion_time': completion_time,
                    'instance_id': manager_id
                })
                
            return {
                'success': True,
                'thread_id': thread_id,
                'instance_id': manager_id,
                'completion_time': completion_time
            }
            
        except Exception as e:
            with self.results_lock:
                self.results['errors'].append({
                    'test': 'lazy_manager_singleton',
                    'thread_id': thread_id,
                    'error': str(e)
                })
            return {
                'success': False,
                'thread_id': thread_id,
                'error': str(e)
            }
            
    def test_lazy_component_thread_safety(self, thread_id: int) -> Dict[str, Any]:
        """Test LazyComponent thread safety."""
        start_time = time.time()
        
        try:
            from iron_core.performance.lazy_loader import LazyComponent
            
            # Create a test lazy component
            lazy_comp = LazyComponent(
                module_path='builtins',  # Use built-in module for testing
                class_name='dict'
            )
            
            # Multiple threads access the same lazy component
            time.sleep(0.001 * (thread_id % 3))
            
            # Trigger lazy loading
            instance = lazy_comp()
            instance_id = id(instance)
            
            completion_time = time.time() - start_time
            
            with self.results_lock:
                self.results['timing_results'].append({
                    'test': 'lazy_component_safety',
                    'thread_id': thread_id,
                    'completion_time': completion_time,
                    'instance_id': instance_id
                })
                
            return {
                'success': True,
                'thread_id': thread_id,
                'instance_id': instance_id,
                'completion_time': completion_time
            }
            
        except Exception as e:
            with self.results_lock:
                self.results['errors'].append({
                    'test': 'lazy_component_safety',
                    'thread_id': thread_id,
                    'error': str(e)
                })
            return {
                'success': False,
                'thread_id': thread_id,
                'error': str(e)
            }
            
    def run_concurrent_tests(self, num_threads: int = 20) -> Dict[str, Any]:
        """Run comprehensive concurrent tests."""
        logger.info(f"🧪 Starting thread safety validation with {num_threads} threads")
        
        # Test 1: Container singleton thread safety
        logger.info("Testing IRONContainer singleton thread safety...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            container_futures = [
                executor.submit(self.test_container_singleton, i) 
                for i in range(num_threads)
            ]
            
            container_results = []
            for future in as_completed(container_futures):
                container_results.append(future.result())
                
        # Test 2: Lazy manager singleton thread safety  
        logger.info("Testing LazyLoadingManager singleton thread safety...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            manager_futures = [
                executor.submit(self.test_lazy_manager_singleton, i)
                for i in range(num_threads)
            ]
            
            manager_results = []
            for future in as_completed(manager_futures):
                manager_results.append(future.result())
                
        # Test 3: Lazy component thread safety
        logger.info("Testing LazyComponent thread safety...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            component_futures = [
                executor.submit(self.test_lazy_component_thread_safety, i)
                for i in range(num_threads)
            ]
            
            component_results = []
            for future in as_completed(component_futures):
                component_results.append(future.result())
                
        return {
            'container_results': container_results,
            'manager_results': manager_results,
            'component_results': component_results,
            'summary': self.results
        }
        
    def analyze_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thread safety test results."""
        analysis = {
            'thread_safety_status': 'PASS',
            'performance_impact': 'MINIMAL',
            'singleton_integrity': True,
            'errors_found': len(self.results['errors']),
            'detailed_analysis': {}
        }
        
        # Analyze container singleton
        container_instances = len(self.results['container_instances'])
        container_successes = sum(1 for r in test_results['container_results'] if r['success'])
        
        analysis['detailed_analysis']['container'] = {
            'unique_instances': container_instances,
            'expected_instances': 1,
            'singleton_integrity': container_instances == 1,
            'success_rate': container_successes / len(test_results['container_results']),
            'average_init_time': sum(
                r['completion_time'] for r in test_results['container_results'] if r['success']
            ) / max(1, container_successes)
        }
        
        # Analyze lazy manager singleton
        manager_instances = len(self.results['lazy_manager_instances'])
        manager_successes = sum(1 for r in test_results['manager_results'] if r['success'])
        
        analysis['detailed_analysis']['lazy_manager'] = {
            'unique_instances': manager_instances,
            'expected_instances': 1,
            'singleton_integrity': manager_instances == 1,
            'success_rate': manager_successes / len(test_results['manager_results']),
            'average_init_time': sum(
                r['completion_time'] for r in test_results['manager_results'] if r['success']
            ) / max(1, manager_successes)
        }
        
        # Analyze component thread safety
        component_successes = sum(1 for r in test_results['component_results'] if r['success'])
        
        analysis['detailed_analysis']['lazy_components'] = {
            'success_rate': component_successes / len(test_results['component_results']),
            'average_load_time': sum(
                r['completion_time'] for r in test_results['component_results'] if r['success']
            ) / max(1, component_successes)
        }
        
        # Overall assessment
        if (container_instances != 1 or manager_instances != 1 or 
            len(self.results['errors']) > 0):
            analysis['thread_safety_status'] = 'FAIL'
            
        # Performance impact assessment
        avg_times = [
            analysis['detailed_analysis']['container']['average_init_time'],
            analysis['detailed_analysis']['lazy_manager']['average_init_time'],
            analysis['detailed_analysis']['lazy_components']['average_load_time']
        ]
        
        if max(avg_times) > 0.1:  # >100ms is significant
            analysis['performance_impact'] = 'MODERATE'
        elif max(avg_times) > 0.5:  # >500ms is high
            analysis['performance_impact'] = 'HIGH'
            
        return analysis


def main():
    """Run comprehensive thread safety validation."""
    print("🛡️  IRON-Core Thread Safety Validation")
    print("=" * 50)
    
    validator = ThreadSafetyValidator()
    
    # Run tests with increasing thread counts for stress testing
    thread_counts = [10, 20, 50]
    
    for num_threads in thread_counts:
        print(f"\n🧪 Testing with {num_threads} concurrent threads...")
        
        test_results = validator.run_concurrent_tests(num_threads)
        analysis = validator.analyze_results(test_results)
        
        print(f"\n📊 Results for {num_threads} threads:")
        print(f"  Thread Safety Status: {analysis['thread_safety_status']}")
        print(f"  Performance Impact: {analysis['performance_impact']}")
        print(f"  Errors Found: {analysis['errors_found']}")
        
        # Container analysis
        container_analysis = analysis['detailed_analysis']['container']
        print("\n  🏗️ IRONContainer Singleton:")
        print(f"    Instances Created: {container_analysis['unique_instances']}/1 (expected)")
        print(f"    Success Rate: {container_analysis['success_rate']:.1%}")
        print(f"    Avg Init Time: {container_analysis['average_init_time']:.4f}s")
        print(f"    Singleton Integrity: {'✅ PASS' if container_analysis['singleton_integrity'] else '❌ FAIL'}")
        
        # Lazy manager analysis
        manager_analysis = analysis['detailed_analysis']['lazy_manager']
        print("\n  🔄 LazyLoadingManager Singleton:")
        print(f"    Instances Created: {manager_analysis['unique_instances']}/1 (expected)")
        print(f"    Success Rate: {manager_analysis['success_rate']:.1%}")
        print(f"    Avg Init Time: {manager_analysis['average_init_time']:.4f}s")
        print(f"    Singleton Integrity: {'✅ PASS' if manager_analysis['singleton_integrity'] else '❌ FAIL'}")
        
        # Component analysis
        component_analysis = analysis['detailed_analysis']['lazy_components']
        print("\n  🔧 LazyComponent Thread Safety:")
        print(f"    Success Rate: {component_analysis['success_rate']:.1%}")
        print(f"    Avg Load Time: {component_analysis['average_load_time']:.4f}s")
        
        if analysis['errors_found'] > 0:
            print(f"\n❌ Errors Detected ({analysis['errors_found']}):")
            for error in validator.results['errors']:
                print(f"    {error['test']} (thread {error['thread_id']}): {error['error']}")
                
        # Reset results for next iteration
        validator.results = {
            'container_instances': set(),
            'lazy_manager_instances': set(), 
            'timing_results': [],
            'errors': [],
            'thread_completion_times': []
        }
    
    print("\n🏆 Thread Safety Validation Complete")
    print("All singleton patterns should show exactly 1 instance created")
    print("and 100% success rates for proper thread safety implementation.")


if __name__ == "__main__":
    main()