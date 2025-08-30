"""
IRONFORGE Performance Monitoring Configuration

Defines performance contracts, thresholds, and monitoring settings for the
IRONFORGE archaeological discovery pipeline. These configurations ensure
strict adherence to the production performance requirements.

Performance Contracts (Golden Standards):
- Single Session Processing: <3 seconds
- Full Discovery (57 sessions): <180 seconds  
- Memory Footprint: <100MB total usage
- Authenticity Threshold: >87% for production patterns
- Initialization: <2 seconds with lazy loading
- Monitoring Overhead: Sub-millisecond impact
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import json
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels for performance monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class PerformanceStatus(Enum):
    """Performance status indicators."""
    OPTIMAL = "optimal"      # All metrics within target ranges
    ACCEPTABLE = "acceptable"  # Minor degradation but within limits
    DEGRADED = "degraded"    # Performance issues detected
    CRITICAL = "critical"    # Severe performance problems


@dataclass
class StageThresholds:
    """Performance thresholds for each pipeline stage."""
    
    # Core pipeline stage thresholds (seconds)
    discovery_seconds: float = 60.0      # Discovery stage target: <60s for 57 sessions
    confluence_seconds: float = 30.0     # Confluence scoring target: <30s
    validation_seconds: float = 15.0     # Validation stage target: <15s
    reporting_seconds: float = 75.0      # Reporting/dashboard target: <75s
    
    # System initialization and operations
    initialization_seconds: float = 2.0  # Container/system init: <2s
    session_processing_seconds: float = 3.0  # Single session: <3s
    full_discovery_seconds: float = 180.0    # Complete pipeline: <180s
    
    # Memory and resource thresholds
    memory_limit_mb: float = 100.0       # Memory footprint: <100MB
    monitoring_overhead_ms: float = 1.0   # Monitoring impact: <1ms
    
    # Quality and authenticity thresholds
    authenticity_threshold: float = 0.87  # Pattern authenticity: >87%
    duplication_threshold: float = 0.25   # Pattern duplication: <25%
    temporal_coherence_threshold: float = 0.70  # Temporal coherence: >70%
    
    def to_dict(self) -> Dict[str, float]:
        """Convert thresholds to dictionary format."""
        return {
            'discovery_seconds': self.discovery_seconds,
            'confluence_seconds': self.confluence_seconds,
            'validation_seconds': self.validation_seconds,
            'reporting_seconds': self.reporting_seconds,
            'initialization_seconds': self.initialization_seconds,
            'session_processing_seconds': self.session_processing_seconds,
            'full_discovery_seconds': self.full_discovery_seconds,
            'memory_limit_mb': self.memory_limit_mb,
            'monitoring_overhead_ms': self.monitoring_overhead_ms,
            'authenticity_threshold': self.authenticity_threshold,
            'duplication_threshold': self.duplication_threshold,
            'temporal_coherence_threshold': self.temporal_coherence_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'StageThresholds':
        """Create StageThresholds from dictionary."""
        return cls(**data)


@dataclass
class MonitoringSettings:
    """Settings for performance monitoring behavior."""
    
    # Monitoring intervals and timing
    monitoring_interval_seconds: float = 5.0      # Background monitoring frequency
    metrics_retention_hours: int = 24             # How long to keep detailed metrics
    alert_cooldown_seconds: float = 300.0         # Minimum time between duplicate alerts
    performance_sampling_rate: float = 1.0       # Fraction of operations to sample (1.0 = all)
    
    # Dashboard and reporting
    dashboard_update_interval_seconds: float = 2.0  # Real-time dashboard refresh
    dashboard_history_points: int = 100            # Number of historical data points
    export_performance_logs: bool = True           # Export performance data to files
    
    # Optimization and analysis
    enable_automatic_optimization: bool = True     # Enable automatic performance tuning
    bottleneck_detection_threshold: float = 1.5   # Multiplier for bottleneck detection
    regression_detection_sensitivity: float = 0.2  # Sensitivity for regression detection
    
    # Container and lazy loading
    container_preload_components: List[str] = field(default_factory=lambda: [
        "enhanced_graph_builder", 
        "tgat_discovery", 
        "pattern_graduation",
        "confluence_scoring"
    ])
    lazy_loading_timeout_seconds: float = 10.0    # Timeout for lazy component loading
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary format."""
        return {
            'monitoring_interval_seconds': self.monitoring_interval_seconds,
            'metrics_retention_hours': self.metrics_retention_hours,
            'alert_cooldown_seconds': self.alert_cooldown_seconds,
            'performance_sampling_rate': self.performance_sampling_rate,
            'dashboard_update_interval_seconds': self.dashboard_update_interval_seconds,
            'dashboard_history_points': self.dashboard_history_points,
            'export_performance_logs': self.export_performance_logs,
            'enable_automatic_optimization': self.enable_automatic_optimization,
            'bottleneck_detection_threshold': self.bottleneck_detection_threshold,
            'regression_detection_sensitivity': self.regression_detection_sensitivity,
            'container_preload_components': self.container_preload_components,
            'lazy_loading_timeout_seconds': self.lazy_loading_timeout_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringSettings':
        """Create MonitoringSettings from dictionary."""
        return cls(**data)


@dataclass
class AlertConfiguration:
    """Configuration for performance alerts and notifications."""
    
    # Alert thresholds (multipliers of base thresholds)
    warning_threshold_multiplier: float = 0.8     # Warn at 80% of threshold
    critical_threshold_multiplier: float = 1.0    # Critical at 100% of threshold
    
    # Alert categories and their settings
    enable_timing_alerts: bool = True              # Stage timing alerts
    enable_memory_alerts: bool = True              # Memory usage alerts
    enable_quality_alerts: bool = True             # Pattern quality alerts
    enable_regression_alerts: bool = True          # Performance regression alerts
    
    # Alert delivery and handling
    alert_channels: List[str] = field(default_factory=lambda: ['log', 'dashboard'])
    max_alerts_per_hour: int = 20                  # Rate limiting for alerts
    alert_aggregation_window_seconds: float = 60.0  # Group similar alerts
    
    # Specific alert configurations
    memory_alert_threshold_mb: float = 80.0        # Alert when memory exceeds 80MB
    session_timeout_alert_threshold: float = 2.5   # Alert when session >2.5s
    authenticity_alert_threshold: float = 0.90     # Alert when authenticity <90%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert configuration to dictionary format."""
        return {
            'warning_threshold_multiplier': self.warning_threshold_multiplier,
            'critical_threshold_multiplier': self.critical_threshold_multiplier,
            'enable_timing_alerts': self.enable_timing_alerts,
            'enable_memory_alerts': self.enable_memory_alerts,
            'enable_quality_alerts': self.enable_quality_alerts,
            'enable_regression_alerts': self.enable_regression_alerts,
            'alert_channels': self.alert_channels,
            'max_alerts_per_hour': self.max_alerts_per_hour,
            'alert_aggregation_window_seconds': self.alert_aggregation_window_seconds,
            'memory_alert_threshold_mb': self.memory_alert_threshold_mb,
            'session_timeout_alert_threshold': self.session_timeout_alert_threshold,
            'authenticity_alert_threshold': self.authenticity_alert_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertConfiguration':
        """Create AlertConfiguration from dictionary."""
        return cls(**data)


@dataclass
class OptimizationSettings:
    """Settings for automated performance optimization."""
    
    # Optimization triggers and thresholds
    enable_memory_optimization: bool = True        # Automatic memory optimization
    enable_lazy_loading_tuning: bool = True       # Optimize lazy loading patterns
    enable_garbage_collection_tuning: bool = True  # GC optimization
    
    # Memory optimization settings
    memory_optimization_threshold: float = 85.0   # Trigger optimization at 85MB
    garbage_collection_frequency_seconds: float = 30.0  # Periodic GC frequency
    memory_leak_detection_enabled: bool = True    # Enable memory leak detection
    
    # Performance tuning parameters
    batch_size_optimization: bool = True          # Optimize batch processing sizes
    thread_pool_optimization: bool = True         # Optimize thread pool sizes
    cache_optimization: bool = True               # Optimize caching strategies
    
    # Container optimization
    container_warmup_enabled: bool = True         # Pre-warm critical components
    component_pooling_enabled: bool = True        # Pool frequently used components
    lazy_loading_prediction: bool = True          # Predict and preload components
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert optimization settings to dictionary format."""
        return {
            'enable_memory_optimization': self.enable_memory_optimization,
            'enable_lazy_loading_tuning': self.enable_lazy_loading_tuning,
            'enable_garbage_collection_tuning': self.enable_garbage_collection_tuning,
            'memory_optimization_threshold': self.memory_optimization_threshold,
            'garbage_collection_frequency_seconds': self.garbage_collection_frequency_seconds,
            'memory_leak_detection_enabled': self.memory_leak_detection_enabled,
            'batch_size_optimization': self.batch_size_optimization,
            'thread_pool_optimization': self.thread_pool_optimization,
            'cache_optimization': self.cache_optimization,
            'container_warmup_enabled': self.container_warmup_enabled,
            'component_pooling_enabled': self.component_pooling_enabled,
            'lazy_loading_prediction': self.lazy_loading_prediction
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationSettings':
        """Create OptimizationSettings from dictionary."""
        return cls(**data)


@dataclass
class IRONFORGEPerformanceConfig:
    """
    Complete IRONFORGE Performance Monitoring Configuration
    
    This configuration defines all performance contracts, monitoring settings,
    alert configurations, and optimization parameters for the IRONFORGE
    archaeological discovery pipeline.
    """
    
    # Core performance contracts
    stage_thresholds: StageThresholds = field(default_factory=StageThresholds)
    
    # Monitoring behavior settings
    monitoring_settings: MonitoringSettings = field(default_factory=MonitoringSettings)
    
    # Alert and notification configuration
    alert_config: AlertConfiguration = field(default_factory=AlertConfiguration)
    
    # Automated optimization settings
    optimization_settings: OptimizationSettings = field(default_factory=OptimizationSettings)
    
    # Derived convenience properties for backward compatibility
    @property
    def memory_limit_mb(self) -> float:
        """Memory limit in MB."""
        return self.stage_thresholds.memory_limit_mb
    
    @property
    def authenticity_threshold(self) -> float:
        """Pattern authenticity threshold."""
        return self.stage_thresholds.authenticity_threshold
    
    @property
    def monitoring_interval_seconds(self) -> float:
        """Background monitoring interval."""
        return self.monitoring_settings.monitoring_interval_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete configuration to dictionary format."""
        return {
            'stage_thresholds': self.stage_thresholds.to_dict(),
            'monitoring_settings': self.monitoring_settings.to_dict(),
            'alert_config': self.alert_config.to_dict(),
            'optimization_settings': self.optimization_settings.to_dict(),
            'version': '1.0.0',
            'description': 'IRONFORGE Pipeline Performance Monitoring Configuration'
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRONFORGEPerformanceConfig':
        """Create IRONFORGEPerformanceConfig from dictionary."""
        return cls(
            stage_thresholds=StageThresholds.from_dict(data.get('stage_thresholds', {})),
            monitoring_settings=MonitoringSettings.from_dict(data.get('monitoring_settings', {})),
            alert_config=AlertConfiguration.from_dict(data.get('alert_config', {})),
            optimization_settings=OptimizationSettings.from_dict(data.get('optimization_settings', {}))
        )
    
    @classmethod
    def from_file(cls, file_path: str) -> 'IRONFORGEPerformanceConfig':
        """Load configuration from YAML or JSON file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {path.suffix}")
        
        return cls.from_dict(data)
    
    def save_to_file(self, file_path: str):
        """Save configuration to YAML or JSON file."""
        path = Path(file_path)
        data = self.to_dict()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(data, f, indent=2, sort_keys=False)
            elif path.suffix.lower() == '.json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration format: {path.suffix}")
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Validate thresholds are reasonable
        if self.stage_thresholds.session_processing_seconds <= 0:
            issues.append("session_processing_seconds must be positive")
        
        if self.stage_thresholds.session_processing_seconds > 10.0:
            issues.append("session_processing_seconds seems too high (>10s)")
        
        if self.stage_thresholds.memory_limit_mb <= 0:
            issues.append("memory_limit_mb must be positive")
        
        if self.stage_thresholds.authenticity_threshold < 0.5 or self.stage_thresholds.authenticity_threshold > 1.0:
            issues.append("authenticity_threshold must be between 0.5 and 1.0")
        
        # Validate monitoring intervals
        if self.monitoring_settings.monitoring_interval_seconds <= 0:
            issues.append("monitoring_interval_seconds must be positive")
        
        if self.monitoring_settings.performance_sampling_rate < 0.0 or self.monitoring_settings.performance_sampling_rate > 1.0:
            issues.append("performance_sampling_rate must be between 0.0 and 1.0")
        
        # Validate alert settings
        if self.alert_config.max_alerts_per_hour <= 0:
            issues.append("max_alerts_per_hour must be positive")
        
        # Check threshold consistency
        total_stage_time = (
            self.stage_thresholds.discovery_seconds +
            self.stage_thresholds.confluence_seconds +
            self.stage_thresholds.validation_seconds +
            self.stage_thresholds.reporting_seconds
        )
        
        if total_stage_time > self.stage_thresholds.full_discovery_seconds:
            issues.append(
                f"Sum of individual stage thresholds ({total_stage_time}s) "
                f"exceeds full discovery threshold ({self.stage_thresholds.full_discovery_seconds}s)"
            )
        
        return issues
    
    def get_performance_contracts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance contracts in a structured format for validation.
        
        Returns:
            Dictionary mapping contract names to their specifications
        """
        return {
            'session_processing': {
                'threshold': self.stage_thresholds.session_processing_seconds,
                'unit': 'seconds',
                'operator': 'less_than',
                'description': 'Single session processing time must be under 3 seconds'
            },
            'full_discovery': {
                'threshold': self.stage_thresholds.full_discovery_seconds,
                'unit': 'seconds',
                'operator': 'less_than',
                'description': 'Full discovery pipeline must complete within 180 seconds'
            },
            'memory_usage': {
                'threshold': self.stage_thresholds.memory_limit_mb,
                'unit': 'megabytes',
                'operator': 'less_than',
                'description': 'Total memory footprint must remain under 100MB'
            },
            'pattern_authenticity': {
                'threshold': self.stage_thresholds.authenticity_threshold,
                'unit': 'percentage',
                'operator': 'greater_than',
                'description': 'Pattern authenticity scores must exceed 87%'
            },
            'initialization_time': {
                'threshold': self.stage_thresholds.initialization_seconds,
                'unit': 'seconds',
                'operator': 'less_than',
                'description': 'System initialization with lazy loading must complete within 2 seconds'
            },
            'monitoring_overhead': {
                'threshold': self.stage_thresholds.monitoring_overhead_ms,
                'unit': 'milliseconds',
                'operator': 'less_than',
                'description': 'Performance monitoring overhead must be sub-millisecond'
            }
        }
    
    def get_optimization_targets(self) -> Dict[str, Any]:
        """Get optimization targets and current settings."""
        return {
            'memory_optimization': {
                'enabled': self.optimization_settings.enable_memory_optimization,
                'threshold_mb': self.optimization_settings.memory_optimization_threshold,
                'gc_frequency_seconds': self.optimization_settings.garbage_collection_frequency_seconds
            },
            'lazy_loading': {
                'enabled': self.optimization_settings.enable_lazy_loading_tuning,
                'timeout_seconds': self.monitoring_settings.lazy_loading_timeout_seconds,
                'preload_components': self.monitoring_settings.container_preload_components
            },
            'batch_processing': {
                'enabled': self.optimization_settings.batch_size_optimization,
                'thread_pool_optimization': self.optimization_settings.thread_pool_optimization
            },
            'caching': {
                'enabled': self.optimization_settings.cache_optimization,
                'component_pooling': self.optimization_settings.component_pooling_enabled
            }
        }


def create_development_config() -> IRONFORGEPerformanceConfig:
    """Create a development-friendly performance configuration."""
    config = IRONFORGEPerformanceConfig()
    
    # Relax some thresholds for development
    config.stage_thresholds.session_processing_seconds = 5.0  # Allow 5s for dev
    config.stage_thresholds.full_discovery_seconds = 300.0    # Allow 5 minutes for dev
    config.stage_thresholds.memory_limit_mb = 150.0           # Allow 150MB for dev
    
    # More frequent monitoring in development
    config.monitoring_settings.monitoring_interval_seconds = 2.0
    config.monitoring_settings.dashboard_update_interval_seconds = 1.0
    
    # More verbose alerts in development
    config.alert_config.max_alerts_per_hour = 100
    config.alert_config.alert_channels = ['log', 'dashboard', 'console']
    
    return config


def create_production_config() -> IRONFORGEPerformanceConfig:
    """Create a production-optimized performance configuration."""
    config = IRONFORGEPerformanceConfig()
    
    # Strict production thresholds (defaults are already production-ready)
    config.stage_thresholds.session_processing_seconds = 3.0
    config.stage_thresholds.full_discovery_seconds = 180.0
    config.stage_thresholds.memory_limit_mb = 100.0
    
    # Optimized monitoring for production
    config.monitoring_settings.monitoring_interval_seconds = 10.0
    config.monitoring_settings.performance_sampling_rate = 0.1  # Sample 10% in production
    
    # Conservative alerts for production
    config.alert_config.max_alerts_per_hour = 10
    config.alert_config.alert_aggregation_window_seconds = 300.0  # 5-minute aggregation
    
    # Aggressive optimization for production
    config.optimization_settings.memory_optimization_threshold = 80.0
    config.optimization_settings.garbage_collection_frequency_seconds = 60.0
    
    return config


def load_or_create_config(config_path: Optional[str] = None, 
                         environment: str = "development") -> IRONFORGEPerformanceConfig:
    """
    Load configuration from file or create default based on environment.
    
    Args:
        config_path: Optional path to configuration file
        environment: Environment type ('development' or 'production')
        
    Returns:
        IRONFORGEPerformanceConfig instance
    """
    if config_path and Path(config_path).exists():
        return IRONFORGEPerformanceConfig.from_file(config_path)
    
    if environment.lower() == "production":
        return create_production_config()
    else:
        return create_development_config()


# Example configuration files for reference
EXAMPLE_CONFIG_YAML = """
# IRONFORGE Performance Monitoring Configuration
version: "1.0.0"
description: "IRONFORGE Pipeline Performance Monitoring Configuration"

stage_thresholds:
  discovery_seconds: 60.0
  confluence_seconds: 30.0
  validation_seconds: 15.0
  reporting_seconds: 75.0
  session_processing_seconds: 3.0
  full_discovery_seconds: 180.0
  memory_limit_mb: 100.0
  authenticity_threshold: 0.87

monitoring_settings:
  monitoring_interval_seconds: 5.0
  dashboard_update_interval_seconds: 2.0
  export_performance_logs: true
  enable_automatic_optimization: true

alert_config:
  enable_timing_alerts: true
  enable_memory_alerts: true
  enable_quality_alerts: true
  max_alerts_per_hour: 20

optimization_settings:
  enable_memory_optimization: true
  enable_lazy_loading_tuning: true
  container_warmup_enabled: true
"""


if __name__ == "__main__":
    # Example usage and validation
    config = IRONFORGEPerformanceConfig()
    
    print("🎯 IRONFORGE Performance Configuration")
    print("=" * 50)
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print(f"❌ Configuration issues found:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✅ Configuration is valid")
    
    # Display performance contracts
    print("\n📋 Performance Contracts:")
    contracts = config.get_performance_contracts()
    for name, contract in contracts.items():
        print(f"  {name}: {contract['operator']} {contract['threshold']} {contract['unit']}")
    
    # Save example configuration
    example_path = Path("example_performance_config.yaml")
    config.save_to_file(str(example_path))
    print(f"\n💾 Example configuration saved to: {example_path}")