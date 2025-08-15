"""
Layer 5: API Interface Layer
============================

Clean external APIs for mathematical models and integration endpoints.
Provides REST API, WebSocket, and programmatic interfaces for mathematical predictions.

Key Features:
- RESTful API endpoints for mathematical operations
- WebSocket real-time prediction streaming
- Batch processing API
- Model management endpoints
- Performance monitoring APIs
- Health check and status endpoints
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import asyncio

# FastAPI imports (optional dependency)
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    
    # Mock classes for graceful degradation
    class BaseModel:
        pass
    
    class FastAPI:
        pass

from .integration_layer import MathematicalModelRegistry, ModelChain
from ..mathematical_hooks import HookManager

logger = logging.getLogger(__name__)

# Pydantic models for API request/response schemas
class PredictionRequest(BaseModel):
    """Request schema for mathematical predictions"""
    model_type: str = Field(..., description="Type of mathematical model to use")
    session_data: Dict[str, Any] = Field(..., description="Session data for prediction")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Model parameters")
    performance_target: Optional[Dict[str, float]] = Field(
        default={"max_execution_time_seconds": 5.0, "min_accuracy": 0.85},
        description="Performance requirements"
    )

class PredictionResponse(BaseModel):
    """Response schema for mathematical predictions"""
    success: bool = Field(..., description="Whether prediction was successful")
    prediction_result: Optional[Dict[str, Any]] = Field(default=None, description="Prediction results")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    model_info: Dict[str, str] = Field(..., description="Information about the model used")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

class ValidationRequest(BaseModel):
    """Request schema for model validation"""
    model_type: str = Field(..., description="Type of mathematical model to validate")
    validation_level: str = Field(default="standard", description="Validation thoroughness level")
    test_data: Optional[Dict[str, Any]] = Field(default=None, description="Test data for validation")

class ValidationResponse(BaseModel):
    """Response schema for model validation"""
    validation_status: str = Field(..., description="Overall validation status")
    success_rate: float = Field(..., description="Percentage of tests passed")
    validation_report: str = Field(..., description="Detailed validation report")
    execution_time_ms: float = Field(..., description="Validation execution time")

class OptimizationRequest(BaseModel):
    """Request schema for parameter optimization"""
    model_type: str = Field(..., description="Type of mathematical model to optimize")
    training_data: Dict[str, Any] = Field(..., description="Training data for optimization")
    optimization_target: str = Field(default="accuracy", description="Optimization objective")

class OptimizationResponse(BaseModel):
    """Response schema for parameter optimization"""
    success: bool = Field(..., description="Whether optimization was successful")
    optimized_parameters: Dict[str, Any] = Field(..., description="Optimized model parameters")
    optimization_metrics: Dict[str, Any] = Field(..., description="Optimization performance metrics")
    execution_time_ms: float = Field(..., description="Optimization execution time")

class StatusResponse(BaseModel):
    """Response schema for system status"""
    system_status: str = Field(..., description="Overall system health status")
    components: Dict[str, str] = Field(..., description="Status of system components")
    performance_metrics: Dict[str, Any] = Field(..., description="System performance metrics")
    timestamp: str = Field(..., description="Status check timestamp")

class APIInterfaceLayer(ABC):
    """
    Base class for API interface implementations.
    Provides framework for exposing mathematical models via APIs.
    """
    
    @abstractmethod
    def expose_prediction_endpoint(self, model_registry: MathematicalModelRegistry) -> None:
        """Expose prediction functionality as API endpoint"""
        pass
    
    @abstractmethod
    def health_check_endpoint(self) -> Dict[str, Any]:
        """Mathematical model health check endpoint"""
        pass
    
    @abstractmethod
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the API server"""
        pass

class MathematicalModelAPI(APIInterfaceLayer):
    """
    FastAPI-based REST API for mathematical models.
    Provides comprehensive API endpoints for all mathematical operations.
    """
    
    def __init__(self, model_registry: MathematicalModelRegistry, hook_manager: Optional[HookManager] = None):
        self.model_registry = model_registry
        self.hook_manager = hook_manager
        self.app = None
        self.websocket_connections: List[WebSocket] = []
        self.request_count = 0
        self.start_time = datetime.now()
        
        if FASTAPI_AVAILABLE:
            self._setup_fastapi_app()
        else:
            logger.warning("FastAPI not available - API functionality will be limited")
    
    def _setup_fastapi_app(self):
        """Setup FastAPI application with all endpoints"""
        
        self.app = FastAPI(
            title="Mathematical Models API",
            description="API for Oracle mathematical model predictions, validation, and optimization",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add middleware for request counting
        @self.app.middleware("http")
        async def count_requests(request, call_next):
            self.request_count += 1
            response = await call_next(request)
            return response
        
        # Setup all endpoints
        self._setup_prediction_endpoints()
        self._setup_validation_endpoints()
        self._setup_optimization_endpoints()
        self._setup_management_endpoints()
        self._setup_websocket_endpoints()
    
    def _setup_prediction_endpoints(self):
        """Setup prediction-related API endpoints"""
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            """Mathematical model prediction endpoint"""
            
            start_time = datetime.now()
            
            try:
                # Validate model type
                if request.model_type not in self.model_registry.models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model type '{request.model_type}' not available. "
                               f"Available models: {list(self.model_registry.models.keys())}"
                    )
                
                # Create prediction chain
                chain_spec = {
                    "chain_id": f"{request.model_type}_api_prediction",
                    "description": f"API prediction for {request.model_type}",
                    "steps": [
                        {
                            "type": "generic",
                            "model_id": request.model_type,
                            "parameters": request.parameters
                        }
                    ]
                }
                
                chain = self.model_registry.create_model_chain(chain_spec)
                
                # Execute prediction
                prediction_result = self.model_registry.execute_prediction_pipeline(
                    request.session_data, chain.chain_id
                )
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Log performance metrics in background
                if self.hook_manager:
                    background_tasks.add_task(
                        self._log_prediction_metrics,
                        request.model_type,
                        execution_time,
                        True
                    )
                
                # Broadcast to WebSocket clients
                if self.websocket_connections:
                    background_tasks.add_task(
                        self._broadcast_prediction_result,
                        request.model_type,
                        prediction_result
                    )
                
                return PredictionResponse(
                    success=True,
                    prediction_result=prediction_result,
                    execution_time_ms=execution_time,
                    model_info={
                        "model_type": request.model_type,
                        "chain_id": chain.chain_id,
                        "parameters": str(request.parameters)
                    }
                )
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Log error in background
                if self.hook_manager:
                    background_tasks.add_task(
                        self._log_prediction_metrics,
                        request.model_type,
                        execution_time,
                        False
                    )
                
                logger.error(f"Prediction failed: {e}")
                
                return PredictionResponse(
                    success=False,
                    prediction_result=None,
                    execution_time_ms=execution_time,
                    model_info={"model_type": request.model_type},
                    error_message=str(e)
                )
        
        @self.app.post("/batch_predict")
        async def batch_predict(requests: List[PredictionRequest]):
            """Batch prediction endpoint for multiple requests"""
            
            results = []
            
            for request in requests:
                # Process each request individually
                result = await predict(request, BackgroundTasks())
                results.append(result)
            
            return {
                "batch_size": len(requests),
                "results": results,
                "success_count": len([r for r in results if r.success]),
                "total_execution_time_ms": sum(r.execution_time_ms for r in results)
            }
    
    def _setup_validation_endpoints(self):
        """Setup validation-related API endpoints"""
        
        @self.app.post("/validate", response_model=ValidationResponse)
        async def validate_model(request: ValidationRequest):
            """Model validation endpoint"""
            
            start_time = datetime.now()
            
            try:
                # Import here to avoid circular imports
                from .validation_framework import ValidationLevel, create_validation_framework
                
                # Validate model type
                if request.model_type not in self.model_registry.models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model type '{request.model_type}' not available"
                    )
                
                # Create validation framework
                validation_level = ValidationLevel(request.validation_level)
                framework = create_validation_framework(validation_level)
                
                # Get model
                model = self.model_registry.models[request.model_type]
                
                # Run validation
                validation_results = framework.comprehensive_validation(
                    model, request.model_type, request.test_data
                )
                
                # Generate report
                validation_report = framework.generate_validation_report(validation_results)
                
                # Calculate success rate
                total_tests = sum(suite.get_summary()["total"] for suite in validation_results.values())
                total_passed = sum(suite.get_summary()["pass"] for suite in validation_results.values())
                success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
                
                # Determine overall status
                if success_rate >= 80:
                    validation_status = "PASS"
                elif success_rate >= 60:
                    validation_status = "WARNING"
                else:
                    validation_status = "FAIL"
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return ValidationResponse(
                    validation_status=validation_status,
                    success_rate=success_rate,
                    validation_report=validation_report,
                    execution_time_ms=execution_time
                )
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"Validation failed: {e}")
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Validation failed: {str(e)}"
                )
    
    def _setup_optimization_endpoints(self):
        """Setup optimization-related API endpoints"""
        
        @self.app.post("/optimize", response_model=OptimizationResponse)
        async def optimize_parameters(request: OptimizationRequest):
            """Parameter optimization endpoint"""
            
            start_time = datetime.now()
            
            try:
                # Validate model type
                if request.model_type not in self.model_registry.models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model type '{request.model_type}' not available"
                    )
                
                # Get model
                model = self.model_registry.models[request.model_type]
                
                # Prepare training data
                training_events = []
                if isinstance(request.training_data, dict) and "events" in request.training_data:
                    for event in request.training_data["events"]:
                        if isinstance(event, dict) and "timestamp" in event:
                            training_events.append(event["timestamp"])
                        elif isinstance(event, (int, float)):
                            training_events.append(event)
                elif isinstance(request.training_data, list):
                    training_events = request.training_data
                
                if not training_events:
                    raise HTTPException(
                        status_code=400,
                        detail="No valid training events found in training_data"
                    )
                
                import numpy as np
                training_array = np.array(training_events)
                
                # Run optimization
                optimization_result = model.optimize_parameters(training_array)
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return OptimizationResponse(
                    success=optimization_result.get("optimization_success", False),
                    optimized_parameters={
                        k: v for k, v in optimization_result.items()
                        if k in ["mu", "alpha", "beta"] and isinstance(v, (int, float))
                    },
                    optimization_metrics={
                        "method": optimization_result.get("method", "unknown"),
                        "final_objective": optimization_result.get("final_objective"),
                        "iterations": optimization_result.get("iterations", 0)
                    },
                    execution_time_ms=execution_time
                )
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"Optimization failed: {e}")
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Optimization failed: {str(e)}"
                )
    
    def _setup_management_endpoints(self):
        """Setup management and monitoring endpoints"""
        
        @self.app.get("/health", response_model=StatusResponse)
        async def health_check():
            """System health check endpoint"""
            
            return StatusResponse(**self.health_check_endpoint())
        
        @self.app.get("/models")
        async def list_models():
            """List available mathematical models"""
            
            models_info = {}
            
            for model_id, model in self.model_registry.models.items():
                metadata = self.model_registry.metadata.get(model_id)
                integration_status = self.model_registry.integration_status.get(model_id)
                
                models_info[model_id] = {
                    "name": metadata.name if metadata else model_id,
                    "description": metadata.description if metadata else "No description",
                    "domain": metadata.domain.value if metadata else "unknown",
                    "priority": metadata.priority.value if metadata else "unknown",
                    "oracle_integration": metadata.oracle_integration if metadata else False,
                    "status": integration_status.value if integration_status else "unknown"
                }
            
            return {
                "total_models": len(models_info),
                "models": models_info,
                "model_ids": list(models_info.keys())
            }
        
        @self.app.get("/chains")
        async def list_prediction_chains():
            """List available prediction chains"""
            
            chains_info = {}
            
            for chain_id, chain in self.model_registry.chains.items():
                chains_info[chain_id] = {
                    "description": chain.description,
                    "steps_count": len(chain.steps),
                    "execution_history_length": len(chain.execution_history)
                }
            
            return {
                "total_chains": len(chains_info),
                "chains": chains_info,
                "chain_ids": list(chains_info.keys())
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get API and system performance metrics"""
            
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # Get model performance summary
            performance_summary = self.model_registry.get_model_performance_summary()
            
            # Get hook manager metrics if available
            hook_metrics = {}
            if self.hook_manager:
                hook_metrics = self.hook_manager.get_hook_performance_summary()
            
            return {
                "api_metrics": {
                    "uptime_seconds": uptime_seconds,
                    "total_requests": self.request_count,
                    "requests_per_minute": (self.request_count / (uptime_seconds / 60)) if uptime_seconds > 0 else 0,
                    "websocket_connections": len(self.websocket_connections)
                },
                "model_performance": performance_summary,
                "hook_metrics": hook_metrics
            }
    
    def _setup_websocket_endpoints(self):
        """Setup WebSocket endpoints for real-time updates"""
        
        @self.app.websocket("/ws/predictions")
        async def websocket_predictions(websocket: WebSocket):
            """WebSocket endpoint for real-time prediction streaming"""
            
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Wait for client messages
                    data = await websocket.receive_text()
                    
                    try:
                        # Parse prediction request
                        request_data = json.loads(data)
                        
                        # Create prediction request
                        prediction_request = PredictionRequest(**request_data)
                        
                        # Process prediction (reuse HTTP endpoint logic)
                        response = await predict(prediction_request, BackgroundTasks())
                        
                        # Send response back to client
                        await websocket.send_text(json.dumps({
                            "type": "prediction_response",
                            "data": response.dict(),
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": str(e),
                            "timestamp": datetime.now().isoformat()
                        }))
                        
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
                logger.info("WebSocket client disconnected")
    
    async def _log_prediction_metrics(self, model_type: str, execution_time_ms: float, success: bool):
        """Log prediction metrics using hook system"""
        
        if not self.hook_manager:
            return
        
        from ..mathematical_hooks import HookContext, HookType
        
        hook_context = HookContext(
            hook_type=HookType.POST_COMPUTATION,
            model_id=model_type,
            timestamp=datetime.now(),
            data={
                "performance_metrics": {
                    "execution_time_ms": execution_time_ms,
                    "accuracy": 1.0 if success else 0.0,
                    "memory_usage_mb": 10.0  # Estimate
                },
                "api_context": True,
                "success": success
            }
        )
        
        try:
            await self.hook_manager.trigger_hooks(hook_context)
        except Exception as e:
            logger.warning(f"Hook logging failed: {e}")
    
    async def _broadcast_prediction_result(self, model_type: str, prediction_result: Dict[str, Any]):
        """Broadcast prediction result to WebSocket clients"""
        
        if not self.websocket_connections:
            return
        
        broadcast_message = json.dumps({
            "type": "prediction_broadcast",
            "model_type": model_type,
            "prediction_result": prediction_result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to all connected clients
        disconnected_clients = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(broadcast_message)
            except Exception:
                disconnected_clients.append(websocket)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.websocket_connections:
                self.websocket_connections.remove(client)
    
    def expose_prediction_endpoint(self, model_registry: MathematicalModelRegistry) -> None:
        """Expose prediction functionality as API endpoint"""
        # This is handled in _setup_fastapi_app()
        pass
    
    def health_check_endpoint(self) -> Dict[str, Any]:
        """Mathematical model health check endpoint"""
        
        # Check model registry status
        integration_status = self.model_registry.get_integration_status()
        active_models = len([s for s in integration_status.values() if s.name == "ACTIVE"])
        total_models = len(integration_status)
        
        # Check hook manager status
        hook_status = "healthy" if self.hook_manager else "not_initialized"
        
        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Determine overall system status
        if active_models == total_models and total_models > 0:
            system_status = "healthy"
        elif active_models > 0:
            system_status = "degraded"
        else:
            system_status = "unhealthy"
        
        return {
            "system_status": system_status,
            "components": {
                "model_registry": "healthy" if self.model_registry else "unhealthy",
                "hook_manager": hook_status,
                "api_server": "healthy",
                "websocket": f"healthy ({len(self.websocket_connections)} connections)"
            },
            "performance_metrics": {
                "uptime_seconds": uptime_seconds,
                "total_requests": self.request_count,
                "active_models": active_models,
                "total_models": total_models,
                "websocket_connections": len(self.websocket_connections)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the API server"""
        
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available - cannot start API server")
        
        if not self.app:
            raise RuntimeError("FastAPI app not initialized")
        
        logger.info(f"Starting Mathematical Models API server on {host}:{port}")
        logger.info(f"API documentation available at: http://{host}:{port}/docs")
        logger.info(f"WebSocket endpoint: ws://{host}:{port}/ws/predictions")
        
        # Start server
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

def create_mathematical_api(model_registry: MathematicalModelRegistry, hook_manager: Optional[HookManager] = None) -> MathematicalModelAPI:
    """Create mathematical model API with registry and hooks"""
    
    api = MathematicalModelAPI(model_registry, hook_manager)
    
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available - API functionality will be limited")
        logger.info("Install FastAPI with: pip install fastapi uvicorn")
    
    return api

if __name__ == "__main__":
    print("🌐 MATHEMATICAL MODELS API TESTING")
    print("=" * 50)
    
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available - cannot test API functionality")
        print("Install with: pip install fastapi uvicorn")
        exit(1)
    
    # Create model registry and API
    from .integration_layer import MathematicalModelRegistry
    from ..mathematical_hooks import create_oracle_hook_manager
    
    print("🔧 Setting up components...")
    model_registry = MathematicalModelRegistry(oracle_integration=False)  # Use fallback models
    hook_manager = create_oracle_hook_manager()
    
    api = create_mathematical_api(model_registry, hook_manager)
    
    print(f"✅ API created successfully")
    print(f"📊 Components initialized:")
    print(f"  Model Registry: {'✅' if api.model_registry else '❌'}")
    print(f"  Hook Manager: {'✅' if api.hook_manager else '❌'}")
    print(f"  FastAPI App: {'✅' if api.app else '❌'}")
    
    # Test health check
    print(f"\n🏥 HEALTH CHECK:")
    health_status = api.health_check_endpoint()
    print(f"  System Status: {health_status['system_status']}")
    print(f"  Active Models: {health_status['performance_metrics']['active_models']}")
    print(f"  Total Models: {health_status['performance_metrics']['total_models']}")
    
    print(f"\n🚀 To start the API server, use:")
    print(f"  api.start_api_server(host='0.0.0.0', port=8000)")
    print(f"\n📖 API Documentation will be available at:")
    print(f"  http://localhost:8000/docs")
    print(f"  http://localhost:8000/redoc")
    print(f"\n🔌 WebSocket endpoint:")
    print(f"  ws://localhost:8000/ws/predictions")
    
    print(f"\n✅ Mathematical models API testing completed")