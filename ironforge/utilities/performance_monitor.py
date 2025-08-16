"""
Performance Monitor
Utility for monitoring IRONFORGE component performance
"""

import time
import logging
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitor performance of IRONFORGE components
    Track timing, memory usage, and component initialization
    """
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("Performance Monitor initialized")
    
    @contextmanager
    def monitor_component(self, component_name: str):
        """
        Context manager for monitoring component performance
        
        Args:
            component_name: Name of component being monitored
        """
        start_time = time.time()
        self.logger.info(f"Starting performance monitoring for {component_name}")
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if component_name not in self.metrics:
                self.metrics[component_name] = []
            
            self.metrics[component_name].append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })
            
            self.logger.info(f"Performance monitoring complete for {component_name}: {duration:.3f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance metrics"""
        
        summary = {}
        for component_name, measurements in self.metrics.items():
            if measurements:
                durations = [m['duration'] for m in measurements]
                summary[component_name] = {
                    'count': len(measurements),
                    'total_duration': sum(durations),
                    'average_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations)
                }
        
        return summary