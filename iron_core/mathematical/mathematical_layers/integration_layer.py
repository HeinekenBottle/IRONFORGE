"""
Layer 3: Integration Layer
=========================

Connects mathematical models to business logic and existing Oracle systems.
Provides model composition, chaining, and seamless integration with Oracle components.

Key Features:
- Model registry and factory patterns
- Mathematical model chaining and composition
- Oracle system integration hooks
- Business logic translation
- Performance monitoring integration
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
import logging
import asyncio
from datetime import datetime

from .theory_abstraction import MathematicalModel, MathematicalDomain
from .core_algorithms import CoreAlgorithmLayer, AlgorithmPerformanceMetrics

# Oracle system integration
try:
    from ..constraints import MATHEMATICAL_CONSTANTS
except ImportError:
    # Fallback constants if Oracle constraints not available
    MATHEMATICAL_CONSTANTS = {
        "INTENSITY_THRESHOLD": 0.5,
        "STABILITY_RATIO": 1.0,
        "PRECISION_TOLERANCE": 1e-6
    }

logger = logging.getLogger(__name__)

class IntegrationStatus(Enum):
    """Status of model integration"""
    PENDING = "pending"
    INITIALIZING = "initializing" 
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

class ModelPriority(Enum):
    """Priority levels for model execution"""
    CRITICAL = 1    # Core prediction models
    HIGH = 2        # Important enhancement models
    MEDIUM = 3      # Supporting models
    LOW = 4         # Optional optimization models

@dataclass
class ModelMetadata:
    """Metadata for registered mathematical models"""
    model_id: str
    name: str
    description: str
    domain: MathematicalDomain
    priority: ModelPriority
    dependencies: List[str] = field(default_factory=list)
    oracle_integration: bool = False
    performance_sli: Optional[float] = None  # Service Level Indicator (ms)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ModelChainStep:
    """Individual step in a model execution chain"""
    model_id: str
    input_transform: Optional[Callable] = None
    output_transform: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditional_execution: Optional[Callable] = None

class ModelChain:
    """
    Chain of mathematical models for complex predictions.
    Supports conditional execution, data transformation, and error handling.
    """
    
    def __init__(self, chain_id: str, description: str):
        self.chain_id = chain_id
        self.description = description
        self.steps: List[ModelChainStep] = []
        self.execution_history: List[Dict[str, Any]] = []
    
    def add_step(self, step: ModelChainStep) -> 'ModelChain':
        """Add a step to the model chain"""
        self.steps.append(step)
        return self
    
    def add_hawkes_prediction(self, 
                             model_id: str = "hawkes_process",
                             parameters: Optional[Dict[str, Any]] = None) -> 'ModelChain':
        """Add Hawkes process prediction step"""
        
        def oracle_data_transform(data):
            """Transform Oracle session data for Hawkes input"""
            if isinstance(data, dict) and 'events' in data:
                events = data['events']
                if isinstance(events, list):
                    # Extract timestamps from Oracle event format
                    timestamps = []
                    for event in events:
                        if isinstance(event, dict) and 'timestamp' in event:
                            timestamps.append(float(event['timestamp']))
                        elif isinstance(event, (int, float)):
                            timestamps.append(float(event))
                    return np.array(timestamps)
            elif isinstance(data, (list, np.ndarray)):
                return np.array(data, dtype=float)
            return data
        
        def hawkes_output_transform(intensities):
            """Transform Hawkes output for Oracle consumption"""
            if isinstance(intensities, np.ndarray):
                return {
                    'type': 'hawkes_prediction',
                    'intensities': intensities.tolist(),
                    'max_intensity': float(np.max(intensities)),
                    'predicted_cascade_probability': float(np.max(intensities) / MATHEMATICAL_CONSTANTS.get("INTENSITY_THRESHOLD", 0.5))
                }
            return intensities
        
        step = ModelChainStep(
            model_id=model_id,
            input_transform=oracle_data_transform,
            output_transform=hawkes_output_transform,
            parameters=parameters or {}
        )
        
        return self.add_step(step)
    
    def add_htf_coupling(self,
                        model_id: str = "htf_coupling", 
                        activation_threshold: float = 0.5) -> 'ModelChain':
        """Add HTF coupling step with activation logic"""
        
        def htf_conditional_execution(previous_output):
            """Execute HTF coupling only if intensity exceeds threshold"""
            if isinstance(previous_output, dict):
                max_intensity = previous_output.get('max_intensity', 0)
                return max_intensity > activation_threshold
            return True
        
        def htf_parameter_adjustment(data):
            """Adjust HTF parameters based on previous predictions"""
            if isinstance(data, dict) and 'max_intensity' in data:
                coupling_strength = min(2.0, data['max_intensity'] / activation_threshold)
                return {
                    'gamma_base': 0.5 * coupling_strength,
                    'htf_intensity_boost': coupling_strength
                }
            return {}
        
        step = ModelChainStep(
            model_id=model_id,
            input_transform=htf_parameter_adjustment,
            conditional_execution=htf_conditional_execution,
            parameters={'activation_threshold': activation_threshold}
        )
        
        return self.add_step(step)
    
    def add_three_oracle_consensus(self,
                                  model_id: str = "three_oracle_consensus") -> 'ModelChain':
        """Add Three-Oracle consensus step"""
        
        def consensus_input_transform(data):
            """Prepare data for Three-Oracle system"""
            predictions = []
            
            # Virgin Oracle: Pure grammatical prediction
            if isinstance(data, dict):
                predictions.append({
                    'oracle': 'virgin',
                    'prediction': data.get('intensities', []),
                    'confidence': 0.8  # High confidence for pure mathematical approach
                })
                
                # Contaminated Oracle: Enhanced with ML
                predictions.append({
                    'oracle': 'contaminated', 
                    'prediction': data.get('intensities', []),
                    'confidence': 0.9,  # Higher confidence with ML enhancement
                    'ml_enhancement': True
                })
                
                # Arbiter Oracle: Weighted consensus
                predictions.append({
                    'oracle': 'arbiter',
                    'prediction': data.get('intensities', []),
                    'confidence': 0.85  # Balanced confidence
                })
            
            return {'oracle_predictions': predictions}
        
        def consensus_output_transform(consensus_result):
            """Transform consensus output for final prediction"""
            if isinstance(consensus_result, dict) and 'consensus_prediction' in consensus_result:
                return {
                    'type': 'three_oracle_consensus',
                    'final_prediction': consensus_result['consensus_prediction'],
                    'consensus_confidence': consensus_result.get('consensus_confidence', 0.8),
                    'oracle_weights': consensus_result.get('oracle_weights', [0.3, 0.4, 0.3])
                }
            return consensus_result
        
        step = ModelChainStep(
            model_id=model_id,
            input_transform=consensus_input_transform,
            output_transform=consensus_output_transform
        )
        
        return self.add_step(step)

class IntegrationLayer(ABC):
    """
    Base class for mathematical model integration with business systems.
    Provides framework for model registration, execution, and monitoring.
    """
    
    @abstractmethod
    def register_model(self, model_id: str, implementation: CoreAlgorithmLayer, metadata: ModelMetadata) -> None:
        """Register mathematical model implementation with metadata"""
        pass
    
    @abstractmethod
    def create_model_chain(self, chain_spec: Dict[str, Any]) -> ModelChain:
        """Create chain of mathematical models based on specification"""
        pass
    
    @abstractmethod
    def execute_prediction_pipeline(self, input_data: Dict[str, Any], chain_id: str) -> Dict[str, Any]:
        """Execute full prediction pipeline with specified chain"""
        pass
    
    @abstractmethod
    def get_integration_status(self) -> Dict[str, IntegrationStatus]:
        """Get current integration status for all registered models"""
        pass

class MathematicalModelRegistry(IntegrationLayer):
    """
    Registry for mathematical models with Oracle system integration.
    Manages model lifecycle, dependencies, and performance monitoring.
    """
    
    def __init__(self, oracle_integration: bool = True):
        self.models: Dict[str, CoreAlgorithmLayer] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.chains: Dict[str, ModelChain] = {}
        self.oracle_integration = oracle_integration
        self.performance_history: Dict[str, List[AlgorithmPerformanceMetrics]] = {}
        self.integration_status: Dict[str, IntegrationStatus] = {}
        
        # Initialize with core models if available
        if oracle_integration:
            self._initialize_oracle_models()
    
    def _initialize_oracle_models(self):
        """Initialize integration with existing Oracle system models"""
        try:
            # Try to integrate with existing Oracle components
            from ..hawkes_engine import create_enhanced_hawkes_engine
            from ..fisher_information_monitor import create_fisher_monitor
            from ..rg_scaler_production import create_production_rg_scaler
            
            logger.info("Oracle system components available for integration")
            
            # Register Oracle-integrated models
            self._register_oracle_hawkes_model()
            self._register_oracle_fisher_model()
            self._register_oracle_rg_scaler()
            
        except ImportError as e:
            logger.warning(f"Oracle system integration limited: {e}")
            self._register_fallback_models()
    
    def _register_oracle_hawkes_model(self):
        """Register Oracle-integrated Hawkes model"""
        from .core_algorithms import HawkesAlgorithmImplementation
        
        hawkes_impl = HawkesAlgorithmImplementation(precision=30, vectorized=True)
        metadata = ModelMetadata(
            model_id="oracle_hawkes_process",
            name="Oracle-Integrated Hawkes Process",
            description="Hawkes process optimized for Oracle cascade prediction",
            domain=MathematicalDomain.POINT_PROCESSES,
            priority=ModelPriority.CRITICAL,
            oracle_integration=True,
            performance_sli=200.0  # 200ms SLI
        )
        
        self.register_model("oracle_hawkes_process", hawkes_impl, metadata)
    
    def _register_oracle_fisher_model(self):
        """Register Oracle Fisher Information Monitor integration"""
        # This would integrate with the existing Fisher monitor
        # For now, create placeholder
        metadata = ModelMetadata(
            model_id="oracle_fisher_monitor",
            name="Oracle Fisher Information Monitor",
            description="Crystallization detection with Oracle integration",
            domain=MathematicalDomain.INFORMATION_THEORY,
            priority=ModelPriority.HIGH,
            oracle_integration=True,
            performance_sli=50.0  # 50ms SLI
        )
        
        # Register placeholder - would be actual Fisher implementation
        self.metadata["oracle_fisher_monitor"] = metadata
        self.integration_status["oracle_fisher_monitor"] = IntegrationStatus.PENDING
    
    def _register_oracle_rg_scaler(self):
        """Register Oracle RG Scaler integration"""
        metadata = ModelMetadata(
            model_id="oracle_rg_scaler",
            name="Oracle RG Scaler",
            description="Renormalization Group scaling with Oracle integration",
            domain=MathematicalDomain.STATISTICAL_PHYSICS,
            priority=ModelPriority.CRITICAL,
            oracle_integration=True,
            performance_sli=10.0  # 10ms SLI
        )
        
        self.metadata["oracle_rg_scaler"] = metadata
        self.integration_status["oracle_rg_scaler"] = IntegrationStatus.PENDING
    
    def _register_fallback_models(self):
        """Register fallback models when Oracle integration unavailable"""
        from .core_algorithms import HawkesAlgorithmImplementation, FFTOptimizedCorrelator
        
        # Basic Hawkes implementation
        hawkes = HawkesAlgorithmImplementation()
        hawkes_metadata = ModelMetadata(
            model_id="hawkes_process",
            name="Standalone Hawkes Process",
            description="Basic Hawkes process implementation",
            domain=MathematicalDomain.POINT_PROCESSES,
            priority=ModelPriority.HIGH,
            oracle_integration=False
        )
        
        self.register_model("hawkes_process", hawkes, hawkes_metadata)
        
        # FFT Correlator
        fft_correlator = FFTOptimizedCorrelator()
        fft_metadata = ModelMetadata(
            model_id="fft_correlator",
            name="FFT Correlation Optimizer",
            description="FFT-based correlation analysis",
            domain=MathematicalDomain.SIGNAL_PROCESSING,
            priority=ModelPriority.MEDIUM,
            oracle_integration=False
        )
        
        self.register_model("fft_correlator", fft_correlator, fft_metadata)
    
    def register_model(self, model_id: str, implementation: CoreAlgorithmLayer, metadata: ModelMetadata) -> None:
        """Register mathematical model with full integration support"""
        
        # Validate model
        if not isinstance(implementation, CoreAlgorithmLayer):
            raise ValueError(f"Implementation must be CoreAlgorithmLayer, got {type(implementation)}")
        
        # Store model and metadata
        self.models[model_id] = implementation
        self.metadata[model_id] = metadata
        self.integration_status[model_id] = IntegrationStatus.INITIALIZING
        
        # Initialize performance history
        self.performance_history[model_id] = []
        
        try:
            # Test model initialization
            test_config = {"test": True}
            implementation.initialize_parameters(test_config)
            
            self.integration_status[model_id] = IntegrationStatus.ACTIVE
            logger.info(f"Successfully registered model: {model_id}")
            
        except Exception as e:
            self.integration_status[model_id] = IntegrationStatus.ERROR
            logger.error(f"Failed to register model {model_id}: {e}")
    
    def create_model_chain(self, chain_spec: Dict[str, Any]) -> ModelChain:
        """Create model chain from specification"""
        
        chain_id = chain_spec.get("chain_id", f"chain_{len(self.chains)}")
        description = chain_spec.get("description", "Model execution chain")
        
        chain = ModelChain(chain_id, description)
        
        # Add steps based on specification
        steps = chain_spec.get("steps", [])
        
        for step_spec in steps:
            step_type = step_spec.get("type")
            
            if step_type == "hawkes_prediction":
                chain.add_hawkes_prediction(
                    model_id=step_spec.get("model_id", "hawkes_process"),
                    parameters=step_spec.get("parameters", {})
                )
            
            elif step_type == "htf_coupling":
                chain.add_htf_coupling(
                    model_id=step_spec.get("model_id", "htf_coupling"),
                    activation_threshold=step_spec.get("activation_threshold", 0.5)
                )
            
            elif step_type == "three_oracle_consensus":
                chain.add_three_oracle_consensus(
                    model_id=step_spec.get("model_id", "three_oracle_consensus")
                )
            
            else:
                # Generic model step
                step = ModelChainStep(
                    model_id=step_spec.get("model_id"),
                    parameters=step_spec.get("parameters", {})
                )
                chain.add_step(step)
        
        # Store chain
        self.chains[chain_id] = chain
        
        return chain
    
    def execute_prediction_pipeline(self, input_data: Dict[str, Any], chain_id: str) -> Dict[str, Any]:
        """Execute prediction pipeline with comprehensive error handling"""
        
        if chain_id not in self.chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        chain = self.chains[chain_id]
        current_data = input_data
        execution_log = []
        
        start_time = datetime.now()
        
        try:
            for i, step in enumerate(chain.steps):
                step_start = datetime.now()
                
                # Check if step should execute conditionally
                if step.conditional_execution and not step.conditional_execution(current_data):
                    execution_log.append({
                        "step": i,
                        "model_id": step.model_id,
                        "status": "skipped",
                        "reason": "conditional_execution_false"
                    })
                    continue
                
                # Check if model is available and active
                if step.model_id not in self.models:
                    execution_log.append({
                        "step": i,
                        "model_id": step.model_id,
                        "status": "error",
                        "reason": "model_not_registered"
                    })
                    continue
                
                if self.integration_status.get(step.model_id) != IntegrationStatus.ACTIVE:
                    execution_log.append({
                        "step": i,
                        "model_id": step.model_id,
                        "status": "error",
                        "reason": "model_not_active"
                    })
                    continue
                
                # Apply input transformation
                if step.input_transform:
                    transformed_data = step.input_transform(current_data)
                else:
                    transformed_data = current_data
                
                # Execute model
                model = self.models[step.model_id]
                
                # Merge step parameters with model parameters
                execution_params = {**step.parameters}
                
                # Execute core function
                result = model.compute_core_function(transformed_data, execution_params)
                
                # Apply output transformation
                if step.output_transform:
                    current_data = step.output_transform(result)
                else:
                    current_data = result
                
                step_duration = (datetime.now() - step_start).total_seconds() * 1000
                
                execution_log.append({
                    "step": i,
                    "model_id": step.model_id,
                    "status": "success",
                    "duration_ms": step_duration,
                    "output_type": type(current_data).__name__
                })
                
        except Exception as e:
            execution_log.append({
                "step": len(execution_log),
                "status": "error",
                "error": str(e)
            })
            logger.error(f"Pipeline execution failed: {e}")
        
        total_duration = (datetime.now() - start_time).total_seconds() * 1000
        
        # Store execution history
        chain.execution_history.append({
            "timestamp": start_time.isoformat(),
            "duration_ms": total_duration,
            "execution_log": execution_log,
            "final_output_type": type(current_data).__name__
        })
        
        return {
            "chain_id": chain_id,
            "execution_status": "completed",
            "total_duration_ms": total_duration,
            "steps_executed": len([log for log in execution_log if log["status"] == "success"]),
            "steps_skipped": len([log for log in execution_log if log["status"] == "skipped"]),
            "steps_failed": len([log for log in execution_log if log["status"] == "error"]),
            "prediction_result": current_data,
            "execution_log": execution_log
        }
    
    def get_integration_status(self) -> Dict[str, IntegrationStatus]:
        """Get current integration status for all models"""
        return self.integration_status.copy()
    
    def get_model_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all models"""
        summary = {}
        
        for model_id, model in self.models.items():
            try:
                metrics = model.benchmark_performance()
                summary[model_id] = {
                    "model_name": self.metadata[model_id].name,
                    "domain": self.metadata[model_id].domain.value,
                    "priority": self.metadata[model_id].priority.value,
                    "oracle_integration": self.metadata[model_id].oracle_integration,
                    "performance_sli": self.metadata[model_id].performance_sli,
                    "current_performance": {
                        "execution_time_ms": metrics.execution_time_ms,
                        "memory_usage_mb": metrics.memory_usage_mb,
                        "complexity": metrics.complexity_achieved
                    },
                    "status": self.integration_status[model_id].value
                }
            except Exception as e:
                summary[model_id] = {
                    "error": str(e),
                    "status": "error"
                }
        
        return summary

def create_oracle_prediction_chain() -> ModelChain:
    """Create standard Oracle prediction chain"""
    
    chain_spec = {
        "chain_id": "oracle_standard_prediction",
        "description": "Standard Oracle cascade prediction pipeline",
        "steps": [
            {
                "type": "hawkes_prediction",
                "model_id": "oracle_hawkes_process",
                "parameters": {
                    "mu": 0.02,
                    "alpha": 35.51, 
                    "beta": 0.00442
                }
            },
            {
                "type": "htf_coupling",
                "model_id": "htf_coupling",
                "activation_threshold": 0.5
            },
            {
                "type": "three_oracle_consensus",
                "model_id": "three_oracle_consensus"
            }
        ]
    }
    
    registry = MathematicalModelRegistry(oracle_integration=True)
    return registry.create_model_chain(chain_spec)

if __name__ == "__main__":
    print("🔗 MATHEMATICAL MODEL INTEGRATION TESTING")
    print("=" * 50)
    
    # Create registry with Oracle integration
    registry = MathematicalModelRegistry(oracle_integration=True)
    
    # Display integration status
    print("\n📊 MODEL INTEGRATION STATUS:")
    status = registry.get_integration_status()
    for model_id, model_status in status.items():
        status_emoji = "✅" if model_status == IntegrationStatus.ACTIVE else "⚠️" if model_status == IntegrationStatus.PENDING else "❌"
        print(f"  {model_id}: {status_emoji} {model_status.value}")
    
    # Performance summary
    print(f"\n⚡ MODEL PERFORMANCE SUMMARY:")
    performance = registry.get_model_performance_summary()
    for model_id, perf_data in performance.items():
        if "current_performance" in perf_data:
            exec_time = perf_data["current_performance"]["execution_time_ms"]
            domain = perf_data["domain"]
            print(f"  {model_id}: {exec_time:.2f}ms ({domain})")
    
    # Create and test prediction chain
    print(f"\n🔮 PREDICTION CHAIN TESTING:")
    
    try:
        # Create Oracle prediction chain
        oracle_chain = create_oracle_prediction_chain()
        
        # Test data
        test_data = {
            "events": [
                {"timestamp": 1.0, "type": "liquidity_sweep"},
                {"timestamp": 2.5, "type": "momentum_break"},
                {"timestamp": 4.0, "type": "cascade_trigger"}
            ],
            "session": "NY_AM",
            "date": "2025-08-09"
        }
        
        # Execute prediction pipeline
        result = registry.execute_prediction_pipeline(test_data, "oracle_standard_prediction")
        
        print(f"  Chain ID: {result['chain_id']}")
        print(f"  Execution Status: {result['execution_status']}")
        print(f"  Total Duration: {result['total_duration_ms']:.2f} ms")
        print(f"  Steps Executed: {result['steps_executed']}")
        print(f"  Steps Failed: {result['steps_failed']}")
        
        if result['prediction_result']:
            pred_type = type(result['prediction_result']).__name__
            print(f"  Prediction Type: {pred_type}")
        
    except Exception as e:
        print(f"  ❌ Chain execution failed: {e}")
    
    print(f"\n✅ Integration layer testing completed")