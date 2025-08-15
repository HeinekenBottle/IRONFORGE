# IRONPULSE Mathematical Hooks System

from typing import Callable, Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import numpy as np
import logging
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)

class HookType(Enum):
    """Types of mathematical model hooks"""
    PRE_COMPUTATION = "pre_computation"
    POST_COMPUTATION = "post_computation"
    PARAMETER_UPDATE = "parameter_update"
    VALIDATION_FAILURE = "validation_failure"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    SYSTEM_HEALTH = "system_health"
    DRIFT_DETECTION = "drift_detection"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RecoveryAction(Enum):
    """Automated recovery actions"""
    LOG_ONLY = "log_only"
    PARAMETER_RESET = "parameter_reset"
    MODEL_RECALIBRATION = "model_recalibration"
    FALLBACK_MODEL = "fallback_model"
    SYSTEM_SHUTDOWN = "system_shutdown"

@dataclass
class HookContext:
    """Context information for hook execution"""
    hook_type: HookType
    model_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

@dataclass
class AlertEvent:
    """Mathematical model alert event"""
    alert_id: str
    level: AlertLevel
    message: str
    model_id: str
    timestamp: datetime
    context: Dict[str, Any]
    recovery_action: RecoveryAction
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class MathematicalHook(ABC):
    """Base class for mathematical model hooks"""
    
    def __init__(self, hook_id: str, enabled: bool = True):
        self.hook_id = hook_id
        self.enabled = enabled
        self.execution_count = 0
        self.last_execution: Optional[datetime] = None
        self.execution_history: deque = deque(maxlen=100)
    
    @abstractmethod
    async def execute(self, context: HookContext) -> Dict[str, Any]:
        """Execute hook logic"""
        pass
    
    async def safe_execute(self, context: HookContext) -> Dict[str, Any]:
        """Execute hook with error handling and monitoring"""
        if not self.enabled:
            return {"status": "disabled", "hook_id": self.hook_id}
        
        start_time = datetime.now()
        
        try:
            result = await self.execute(context)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.execution_count += 1
            self.last_execution = start_time
            
            # Store execution history
            self.execution_history.append({
                "timestamp": start_time.isoformat(),
                "execution_time_seconds": execution_time,
                "success": True,
                "context_model_id": context.model_id
            })
            
            result.update({
                "hook_id": self.hook_id,
                "execution_time_seconds": execution_time,
                "execution_count": self.execution_count
            })
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.execution_history.append({
                "timestamp": start_time.isoformat(),
                "execution_time_seconds": execution_time,
                "success": False,
                "error": str(e),
                "context_model_id": context.model_id
            })
            
            logger.error(f"Hook {self.hook_id} execution failed: {e}")
            
            return {
                "hook_id": self.hook_id,
                "status": "error",
                "error": str(e),
                "execution_time_seconds": execution_time
            }

class ParameterDriftHook(MathematicalHook):
    """
    Detects parameter drift in mathematical models.
    Uses statistical tests and trend analysis to identify when parameters
    deviate significantly from their historical values.
    """
    
    def __init__(self, drift_threshold: float = 0.15, window_size: int = 20):
        super().__init__("parameter_drift_detector")
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.parameter_history: Dict[str, deque] = {}
        self.drift_alerts: List[AlertEvent] = []
    
    async def execute(self, context: HookContext) -> Dict[str, Any]:
        """Detect parameter drift and trigger alerts"""
        
        current_params = context.data.get("parameters", {})
        model_id = context.model_id
        
        # Initialize parameter history for new models
        if model_id not in self.parameter_history:
            self.parameter_history[model_id] = deque(maxlen=self.window_size)
        
        history = self.parameter_history[model_id]
        
        # Store current parameters
        history.append({
            "timestamp": context.timestamp.isoformat(),
            "parameters": current_params.copy()
        })
        
        # Need at least 5 historical points for meaningful drift detection
        if len(history) < 5:
            return {
                "drift_detected": False,
                "reason": "insufficient_history",
                "history_length": len(history)
            }
        
        # Analyze drift for each parameter
        drift_analysis = {}
        overall_drift_score = 0.0
        drifted_parameters = []
        
        for param_name, current_value in current_params.items():
            if not isinstance(current_value, (int, float)):
                continue  # Skip non-numeric parameters
            
            # Extract historical values for this parameter
            historical_values = []
            for entry in history:
                if param_name in entry["parameters"]:
                    historical_values.append(entry["parameters"][param_name])
            
            if len(historical_values) < 3:
                continue  # Need minimum history
            
            # Calculate drift metrics
            drift_score = self._calculate_parameter_drift(
                historical_values[:-1], current_value
            )
            
            drift_analysis[param_name] = {
                "current_value": current_value,
                "historical_mean": float(np.mean(historical_values[:-1])),
                "historical_std": float(np.std(historical_values[:-1])),
                "drift_score": drift_score,
                "drift_detected": drift_score > self.drift_threshold
            }
            
            if drift_score > self.drift_threshold:
                drifted_parameters.append(param_name)
                overall_drift_score = max(overall_drift_score, drift_score)
        
        # Generate alert if significant drift detected
        if drifted_parameters:
            alert = AlertEvent(
                alert_id=f"drift_{model_id}_{context.timestamp.strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING if overall_drift_score < 0.3 else AlertLevel.CRITICAL,
                message=f"Parameter drift detected in {model_id}: {', '.join(drifted_parameters)}",
                model_id=model_id,
                timestamp=context.timestamp,
                context={
                    "drift_score": overall_drift_score,
                    "drifted_parameters": drifted_parameters,
                    "drift_analysis": drift_analysis
                },
                recovery_action=RecoveryAction.MODEL_RECALIBRATION if overall_drift_score > 0.25 else RecoveryAction.LOG_ONLY
            )
            
            self.drift_alerts.append(alert)
            
            return {
                "drift_detected": True,
                "drift_score": overall_drift_score,
                "drifted_parameters": drifted_parameters,
                "alert_generated": True,
                "alert_level": alert.level.value,
                "recovery_action": alert.recovery_action.value,
                "drift_analysis": drift_analysis
            }
        
        return {
            "drift_detected": False,
            "drift_score": overall_drift_score,
            "parameters_analyzed": len(drift_analysis),
            "drift_analysis": drift_analysis
        }
    
    def _calculate_parameter_drift(self, historical_values: List[float], current_value: float) -> float:
        """
        Calculate parameter drift score using statistical methods.
        
        Uses combination of:
        1. Z-score based on historical distribution
        2. Trend analysis (is parameter consistently moving away?)
        3. Relative change magnitude
        """
        
        if len(historical_values) == 0:
            return 0.0
        
        historical_array = np.array(historical_values)
        
        # Method 1: Z-score approach
        mean = np.mean(historical_array)
        std = np.std(historical_array)
        
        if std == 0:
            z_score = 0.0 if current_value == mean else 1.0
        else:
            z_score = abs(current_value - mean) / std
        
        # Method 2: Trend analysis
        if len(historical_values) >= 3:
            # Check if there's a consistent trend
            recent_trend = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
            predicted_next = historical_values[-1] + recent_trend
            trend_deviation = abs(current_value - predicted_next) / (abs(predicted_next) + 1e-6)
        else:
            trend_deviation = 0.0
        
        # Method 3: Relative change magnitude
        relative_change = abs(current_value - historical_values[-1]) / (abs(historical_values[-1]) + 1e-6)
        
        # Combine methods (weighted average)
        drift_score = (
            0.5 * min(z_score / 3.0, 1.0) +         # Z-score contribution (capped at 3 sigma)
            0.3 * min(trend_deviation, 1.0) +       # Trend deviation contribution
            0.2 * min(relative_change, 1.0)         # Relative change contribution
        )
        
        return float(drift_score)

class PerformanceDegradationHook(MathematicalHook):
    """
    Monitors mathematical model performance degradation.
    Tracks execution time, memory usage, and accuracy metrics.
    """
    
    def __init__(self, 
                 execution_time_threshold: float = 500.0,  # ms
                 memory_threshold: float = 100.0,          # MB  
                 accuracy_threshold: float = 0.85):        # 85% minimum
        super().__init__("performance_degradation_monitor")
        self.execution_time_threshold = execution_time_threshold
        self.memory_threshold = memory_threshold
        self.accuracy_threshold = accuracy_threshold
        self.performance_history: Dict[str, deque] = {}
    
    async def execute(self, context: HookContext) -> Dict[str, Any]:
        """Monitor performance metrics and detect degradation"""
        
        model_id = context.model_id
        performance_data = context.data.get("performance_metrics", {})
        
        # Initialize history for new models
        if model_id not in self.performance_history:
            self.performance_history[model_id] = deque(maxlen=50)
        
        history = self.performance_history[model_id]
        
        # Extract performance metrics
        execution_time = performance_data.get("execution_time_ms", 0.0)
        memory_usage = performance_data.get("memory_usage_mb", 0.0)
        accuracy = performance_data.get("accuracy", 1.0)
        
        # Store current metrics
        current_metrics = {
            "timestamp": context.timestamp.isoformat(),
            "execution_time_ms": execution_time,
            "memory_usage_mb": memory_usage,
            "accuracy": accuracy
        }
        
        history.append(current_metrics)
        
        # Analyze performance degradation
        degradation_issues = []
        
        # Check execution time threshold
        if execution_time > self.execution_time_threshold:
            degradation_issues.append({
                "type": "execution_time",
                "current": execution_time,
                "threshold": self.execution_time_threshold,
                "severity": "high" if execution_time > self.execution_time_threshold * 2 else "medium"
            })
        
        # Check memory usage
        if memory_usage > self.memory_threshold:
            degradation_issues.append({
                "type": "memory_usage",
                "current": memory_usage,
                "threshold": self.memory_threshold,
                "severity": "high" if memory_usage > self.memory_threshold * 2 else "medium"
            })
        
        # Check accuracy degradation
        if accuracy < self.accuracy_threshold:
            degradation_issues.append({
                "type": "accuracy",
                "current": accuracy,
                "threshold": self.accuracy_threshold,
                "severity": "critical" if accuracy < self.accuracy_threshold * 0.8 else "high"
            })
        
        # Analyze trends if we have enough history
        trend_analysis = {}
        if len(history) >= 10:
            trend_analysis = self._analyze_performance_trends(history)
        
        # Generate alert if issues detected
        if degradation_issues:
            max_severity = max(issue["severity"] for issue in degradation_issues)
            alert_level = {
                "medium": AlertLevel.WARNING,
                "high": AlertLevel.CRITICAL,
                "critical": AlertLevel.EMERGENCY
            }.get(max_severity, AlertLevel.WARNING)
            
            return {
                "degradation_detected": True,
                "issues": degradation_issues,
                "alert_level": alert_level.value,
                "trend_analysis": trend_analysis,
                "recommended_action": self._recommend_recovery_action(degradation_issues),
                "current_metrics": current_metrics
            }
        
        return {
            "degradation_detected": False,
            "current_metrics": current_metrics,
            "trend_analysis": trend_analysis,
            "performance_score": self._calculate_performance_score(current_metrics)
        }
    
    def _analyze_performance_trends(self, history: deque) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        if len(history) < 5:
            return {"insufficient_data": True}
        
        # Extract time series data
        execution_times = [entry["execution_time_ms"] for entry in history]
        memory_usage = [entry["memory_usage_mb"] for entry in history]
        accuracies = [entry["accuracy"] for entry in history]
        
        trends = {}
        
        # Analyze execution time trend
        if len(execution_times) >= 5:
            exec_trend = np.polyfit(range(len(execution_times)), execution_times, 1)[0]
            trends["execution_time_trend"] = {
                "slope": float(exec_trend),
                "direction": "increasing" if exec_trend > 0.1 else "decreasing" if exec_trend < -0.1 else "stable",
                "concern_level": "high" if exec_trend > 10 else "medium" if exec_trend > 1 else "low"
            }
        
        # Analyze memory trend  
        if len(memory_usage) >= 5:
            mem_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
            trends["memory_usage_trend"] = {
                "slope": float(mem_trend),
                "direction": "increasing" if mem_trend > 0.01 else "decreasing" if mem_trend < -0.01 else "stable",
                "concern_level": "high" if mem_trend > 1.0 else "medium" if mem_trend > 0.1 else "low"
            }
        
        # Analyze accuracy trend
        if len(accuracies) >= 5:
            acc_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            trends["accuracy_trend"] = {
                "slope": float(acc_trend),
                "direction": "improving" if acc_trend > 0.001 else "degrading" if acc_trend < -0.001 else "stable",
                "concern_level": "high" if acc_trend < -0.01 else "medium" if acc_trend < -0.001 else "low"
            }
        
        return trends
    
    def _recommend_recovery_action(self, issues: List[Dict[str, Any]]) -> str:
        """Recommend recovery action based on performance issues"""
        
        critical_issues = [i for i in issues if i["severity"] == "critical"]
        high_issues = [i for i in issues if i["severity"] == "high"]
        
        if critical_issues:
            if any(i["type"] == "accuracy" for i in critical_issues):
                return "immediate_model_recalibration"
            else:
                return "resource_scaling_required"
        
        if high_issues:
            if any(i["type"] == "execution_time" for i in high_issues):
                return "algorithm_optimization_needed"
            elif any(i["type"] == "memory_usage" for i in high_issues):
                return "memory_optimization_required"
        
        return "performance_monitoring_continue"
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-1, higher is better)"""
        
        execution_score = max(0, 1 - metrics["execution_time_ms"] / (self.execution_time_threshold * 2))
        memory_score = max(0, 1 - metrics["memory_usage_mb"] / (self.memory_threshold * 2))
        accuracy_score = metrics["accuracy"]
        
        # Weighted average
        overall_score = (
            0.3 * execution_score +
            0.2 * memory_score +
            0.5 * accuracy_score
        )
        
        return float(overall_score)

class MathematicalInvariantValidationHook(MathematicalHook):
    """
    Validates mathematical invariants and constraints.
    Ensures mathematical models maintain their theoretical properties.
    """
    
    def __init__(self):
        super().__init__("mathematical_invariant_validator")
        self.validation_failures: List[AlertEvent] = []
    
    async def execute(self, context: HookContext) -> Dict[str, Any]:
        """Validate mathematical invariants"""
        
        model_id = context.model_id
        parameters = context.data.get("parameters", {})
        results = context.data.get("results", {})
        
        validation_results = []
        invariant_violations = []
        
        # Hawkes Process Invariant Validation
        if "hawkes" in model_id.lower():
            hawkes_validation = self._validate_hawkes_invariants(parameters, results)
            validation_results.append(hawkes_validation)
            
            if not hawkes_validation["valid"]:
                invariant_violations.extend(hawkes_validation["violations"])
        
        # HTF System Invariant Validation
        if "htf" in model_id.lower():
            htf_validation = self._validate_htf_invariants(parameters, results)
            validation_results.append(htf_validation)
            
            if not htf_validation["valid"]:
                invariant_violations.extend(htf_validation["violations"])
        
        # General Mathematical Constraints
        general_validation = self._validate_general_constraints(parameters, results)
        validation_results.append(general_validation)
        
        if not general_validation["valid"]:
            invariant_violations.extend(general_validation["violations"])
        
        # Generate alert if violations found
        if invariant_violations:
            alert = AlertEvent(
                alert_id=f"invariant_{model_id}_{context.timestamp.strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.CRITICAL,
                message=f"Mathematical invariant violations in {model_id}",
                model_id=model_id,
                timestamp=context.timestamp,
                context={
                    "violations": invariant_violations,
                    "parameters": parameters,
                    "results": results
                },
                recovery_action=RecoveryAction.PARAMETER_RESET
            )
            
            self.validation_failures.append(alert)
            
            return {
                "validation_passed": False,
                "violations": invariant_violations,
                "alert_generated": True,
                "recovery_action": RecoveryAction.PARAMETER_RESET.value,
                "validation_details": validation_results
            }
        
        return {
            "validation_passed": True,
            "violations": [],
            "validation_details": validation_results
        }
    
    def _validate_hawkes_invariants(self, parameters: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Hawkes process mathematical invariants"""
        
        violations = []
        
        mu = parameters.get("mu", 0)
        alpha = parameters.get("alpha", 0)
        beta = parameters.get("beta", 1)
        
        # Invariant 1: Positive baseline intensity
        if mu <= 0:
            violations.append("baseline_intensity_non_positive")
        
        # Invariant 2: Non-negative excitation
        if alpha < 0:
            violations.append("excitation_negative")
        
        # Invariant 3: Positive decay rate
        if beta <= 0:
            violations.append("decay_rate_non_positive")
        
        # Invariant 4: Stability condition α/β < 1
        if beta > 0 and alpha / beta >= 1:
            violations.append("stability_condition_violated")
        
        # Invariant 5: Non-negative intensity values
        if "intensities" in results:
            intensities = results["intensities"]
            if isinstance(intensities, (list, np.ndarray)):
                if np.any(np.array(intensities) < 0):
                    violations.append("negative_intensity_values")
        
        return {
            "model_type": "hawkes_process",
            "valid": len(violations) == 0,
            "violations": violations,
            "parameters_checked": ["mu", "alpha", "beta"],
            "stability_ratio": alpha / beta if beta > 0 else float('inf')
        }
    
    def _validate_htf_invariants(self, parameters: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HTF system mathematical invariants"""
        
        violations = []
        
        # HTF-specific parameter validation
        gamma_base = parameters.get("gamma_base", 0)
        beta_h = parameters.get("beta_h", 1)
        beta_s = parameters.get("beta_s", 1)
        
        # Invariant 1: Non-negative coupling
        if gamma_base < 0:
            violations.append("negative_coupling_strength")
        
        # Invariant 2: Timescale separation (HTF should have longer memory)
        if beta_h > 0 and beta_s > 0 and beta_h >= beta_s:
            violations.append("timescale_separation_violated")
        
        return {
            "model_type": "htf_coupling",
            "valid": len(violations) == 0,
            "violations": violations,
            "parameters_checked": ["gamma_base", "beta_h", "beta_s"],
            "timescale_ratio": beta_h / beta_s if beta_s > 0 else 0
        }
    
    def _validate_general_constraints(self, parameters: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate general mathematical constraints"""
        
        violations = []
        
        # Check for NaN or infinite values in parameters
        for param_name, param_value in parameters.items():
            if isinstance(param_value, (int, float)):
                if np.isnan(param_value):
                    violations.append(f"parameter_{param_name}_is_nan")
                elif np.isinf(param_value):
                    violations.append(f"parameter_{param_name}_is_infinite")
        
        # Check for NaN or infinite values in results
        if "intensities" in results:
            intensities = results["intensities"]
            if isinstance(intensities, (list, np.ndarray)):
                intensity_array = np.array(intensities)
                if np.any(np.isnan(intensity_array)):
                    violations.append("result_contains_nan")
                if np.any(np.isinf(intensity_array)):
                    violations.append("result_contains_infinite")
        
        return {
            "model_type": "general",
            "valid": len(violations) == 0,
            "violations": violations,
            "checks_performed": ["nan_check", "infinity_check"]
        }

class HookManager:
    """
    Central manager for mathematical model hooks.
    Coordinates hook execution, manages alerts, and handles recovery actions.
    """
    
    def __init__(self):
        self.hooks: Dict[HookType, List[MathematicalHook]] = {}
        self.hook_history: deque = deque(maxlen=1000)
        self.active_alerts: List[AlertEvent] = []
        self.recovery_actions_log: List[Dict[str, Any]] = []
    
    def register_hook(self, hook_type: HookType, hook: MathematicalHook) -> None:
        """Register a hook for specific events"""
        if hook_type not in self.hooks:
            self.hooks[hook_type] = []
        self.hooks[hook_type].append(hook)
        
        logger.info(f"Registered hook {hook.hook_id} for {hook_type.value}")
    
    async def trigger_hooks(self, context: HookContext) -> List[Dict[str, Any]]:
        """Execute all hooks for a given context"""
        
        if context.hook_type not in self.hooks:
            return []
        
        results = []
        
        # Execute all hooks for this type concurrently
        hook_tasks = []
        for hook in self.hooks[context.hook_type]:
            hook_tasks.append(hook.safe_execute(context))
        
        # Wait for all hooks to complete
        hook_results = await asyncio.gather(*hook_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(hook_results):
            if isinstance(result, Exception):
                logger.error(f"Hook execution failed: {result}")
                results.append({
                    "hook_id": self.hooks[context.hook_type][i].hook_id,
                    "status": "exception",
                    "error": str(result)
                })
            else:
                results.append(result)
        
        # Store execution history
        self.hook_history.append({
            "timestamp": context.timestamp.isoformat(),
            "hook_type": context.hook_type.value,
            "model_id": context.model_id,
            "hooks_executed": len(hook_tasks),
            "hooks_succeeded": len([r for r in results if r.get("status") != "error"]),
            "correlation_id": context.correlation_id
        })
        
        return results
    
    def setup_standard_hooks(self) -> None:
        """Setup standard mathematical model hooks"""
        
        # Parameter drift monitoring
        drift_hook = ParameterDriftHook(drift_threshold=0.15, window_size=20)
        self.register_hook(HookType.PARAMETER_UPDATE, drift_hook)
        
        # Performance monitoring
        perf_hook = PerformanceDegradationHook(
            execution_time_threshold=500.0,
            memory_threshold=100.0,
            accuracy_threshold=0.85
        )
        self.register_hook(HookType.POST_COMPUTATION, perf_hook)
        
        # Mathematical invariant validation
        invariant_hook = MathematicalInvariantValidationHook()
        self.register_hook(HookType.VALIDATION_FAILURE, invariant_hook)
        self.register_hook(HookType.POST_COMPUTATION, invariant_hook)
        
        logger.info("Standard mathematical hooks setup completed")
    
    def get_active_alerts(self, model_id: Optional[str] = None) -> List[AlertEvent]:
        """Get active alerts, optionally filtered by model"""
        
        alerts = [alert for alert in self.active_alerts if not alert.resolved]
        
        if model_id:
            alerts = [alert for alert in alerts if alert.model_id == model_id]
        
        return alerts
    
    def get_hook_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all hooks"""
        
        summary = {
            "total_hooks_registered": sum(len(hooks) for hooks in self.hooks.values()),
            "hook_types": list(self.hooks.keys()),
            "execution_history_length": len(self.hook_history),
            "active_alerts": len(self.get_active_alerts()),
            "hooks_by_type": {}
        }
        
        for hook_type, hooks in self.hooks.items():
            summary["hooks_by_type"][hook_type.value] = {
                "count": len(hooks),
                "hooks": [
                    {
                        "hook_id": hook.hook_id,
                        "enabled": hook.enabled,
                        "execution_count": hook.execution_count,
                        "last_execution": hook.last_execution.isoformat() if hook.last_execution else None
                    }
                    for hook in hooks
                ]
            }
        
        return summary

def create_oracle_hook_manager() -> HookManager:
    """Create HookManager with Oracle-specific configuration"""
    
    manager = HookManager()
    manager.setup_standard_hooks()
    
    # Oracle-specific parameter drift thresholds
    oracle_drift_hook = ParameterDriftHook(
        drift_threshold=0.1,  # More sensitive for Oracle
        window_size=30        # Longer history for Oracle
    )
    manager.register_hook(HookType.PARAMETER_UPDATE, oracle_drift_hook)
    
    # Oracle performance requirements
    oracle_perf_hook = PerformanceDegradationHook(
        execution_time_threshold=200.0,  # Oracle SLI: 200ms
        memory_threshold=50.0,           # Oracle memory limit: 50MB
        accuracy_threshold=0.91          # Oracle accuracy: 91%
    )
    manager.register_hook(HookType.POST_COMPUTATION, oracle_perf_hook)
    
    return manager

if __name__ == "__main__":
    print("🔗 MATHEMATICAL HOOKS SYSTEM TESTING")
    print("=" * 50)
    
    # Create hook manager
    manager = create_oracle_hook_manager()
    
    # Display hook summary
    summary = manager.get_hook_performance_summary()
    print(f"\n📊 HOOKS REGISTERED:")
    print(f"  Total Hooks: {summary['total_hooks_registered']}")
    for hook_type, info in summary["hooks_by_type"].items():
        print(f"  {hook_type}: {info['count']} hooks")
    
    # Test parameter drift detection
    print(f"\n🔍 TESTING PARAMETER DRIFT DETECTION:")
    
    async def test_drift_detection():
        # Simulate parameter drift scenario
        contexts = [
            HookContext(
                hook_type=HookType.PARAMETER_UPDATE,
                model_id="test_hawkes",
                timestamp=datetime.now(),
                data={"parameters": {"mu": 0.02, "alpha": 35.0, "beta": 0.004}}
            ),
            HookContext(
                hook_type=HookType.PARAMETER_UPDATE, 
                model_id="test_hawkes",
                timestamp=datetime.now() + timedelta(minutes=5),
                data={"parameters": {"mu": 0.02, "alpha": 38.0, "beta": 0.004}}  # Drift in alpha
            ),
            HookContext(
                hook_type=HookType.PARAMETER_UPDATE,
                model_id="test_hawkes", 
                timestamp=datetime.now() + timedelta(minutes=10),
                data={"parameters": {"mu": 0.02, "alpha": 45.0, "beta": 0.004}}  # More drift
            )
        ]
        
        for i, context in enumerate(contexts):
            results = await manager.trigger_hooks(context)
            
            drift_results = [r for r in results if "drift_detected" in r]
            if drift_results:
                drift_result = drift_results[0]
                print(f"  Step {i+1}: {'⚠️ DRIFT' if drift_result['drift_detected'] else '✅ OK'} "
                      f"(score: {drift_result.get('drift_score', 0):.3f})")
    
    # Run async test
    import asyncio
    asyncio.run(test_drift_detection())
    
    print(f"\n✅ Mathematical hooks system testing completed")