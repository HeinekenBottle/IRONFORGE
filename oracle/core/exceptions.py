#!/usr/bin/env python3
"""
Oracle System Exceptions
Centralized exception hierarchy for the Oracle training system
"""

from typing import Optional, Dict, Any


class OracleError(Exception):
    """Base exception for all Oracle system errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {super().__str__()}"
        return super().__str__()


class OracleConfigError(OracleError):
    """Raised when Oracle configuration is invalid"""
    pass


class OracleDataError(OracleError):
    """Raised when data-related operations fail"""
    pass


class OracleModelError(OracleError):
    """Raised when model operations fail"""
    pass


class OracleTrainingError(OracleError):
    """Raised when training operations fail"""
    pass


class OracleValidationError(OracleError):
    """Raised when validation operations fail"""
    pass


class AuditError(OracleDataError):
    """Raised when session audit operations fail"""
    pass


class SessionMappingError(OracleDataError):
    """Raised when session mapping operations fail"""
    pass


class DataBuilderError(OracleDataError):
    """Raised when data building operations fail"""
    pass


class PairsBuilderError(OracleDataError):
    """Raised when training pairs building fails"""
    pass


class NormalizationError(OracleDataError):
    """Raised when data normalization fails"""
    pass


class EvaluationError(OracleError):
    """Raised when model evaluation fails"""
    pass


class TGATIntegrationError(OracleError):
    """Raised when TGAT integration fails"""
    pass


class GraphConstructionError(OracleDataError):
    """Raised when graph construction fails"""
    pass


class EmbeddingComputationError(OracleError):
    """Raised when embedding computation fails"""
    pass


# Exception factory functions for common error patterns

def create_audit_error(error_code: str, session_id: str, details: str) -> AuditError:
    """Create standardized audit error with context"""
    return AuditError(
        f"Audit failed for session {session_id}: {details}",
        error_code=error_code,
        context={'session_id': session_id, 'operation': 'audit'}
    )


def create_session_mapping_error(operation: str, details: str, 
                                context: Optional[Dict[str, Any]] = None) -> SessionMappingError:
    """Create standardized session mapping error"""
    return SessionMappingError(
        f"Session mapping failed during {operation}: {details}",
        error_code='SESSION_MAPPING_ERROR',
        context=context or {}
    )


def create_data_builder_error(session_id: str, stage: str, 
                             details: str) -> DataBuilderError:
    """Create standardized data builder error"""
    return DataBuilderError(
        f"Data building failed for session {session_id} at stage {stage}: {details}",
        error_code='DATA_BUILDER_ERROR',
        context={'session_id': session_id, 'stage': stage}
    )


def create_training_error(epoch: Optional[int], details: str, 
                         metrics: Optional[Dict[str, float]] = None) -> OracleTrainingError:
    """Create standardized training error"""
    message = f"Training failed: {details}"
    if epoch is not None:
        message = f"Training failed at epoch {epoch}: {details}"
    
    context = {'operation': 'training'}
    if epoch is not None:
        context['epoch'] = epoch
    if metrics:
        context['metrics'] = metrics
        
    return OracleTrainingError(
        message,
        error_code='TRAINING_ERROR',
        context=context
    )


def create_model_error(operation: str, model_path: str, 
                      details: str) -> OracleModelError:
    """Create standardized model error"""
    return OracleModelError(
        f"Model {operation} failed for {model_path}: {details}",
        error_code='MODEL_ERROR',
        context={'operation': operation, 'model_path': model_path}
    )


def create_validation_error(validation_type: str, expected: Any, 
                           actual: Any, details: str = "") -> OracleValidationError:
    """Create standardized validation error"""
    message = f"Validation failed for {validation_type}: expected {expected}, got {actual}"
    if details:
        message += f" - {details}"
        
    return OracleValidationError(
        message,
        error_code='VALIDATION_ERROR',
        context={
            'validation_type': validation_type,
            'expected': expected,
            'actual': actual
        }
    )


def create_tgat_integration_error(operation: str, details: str) -> TGATIntegrationError:
    """Create standardized TGAT integration error"""
    return TGATIntegrationError(
        f"TGAT integration failed during {operation}: {details}",
        error_code='TGAT_INTEGRATION_ERROR',
        context={'operation': operation}
    )


def create_embedding_error(session_id: str, node_count: int, 
                          details: str) -> EmbeddingComputationError:
    """Create standardized embedding computation error"""
    return EmbeddingComputationError(
        f"Embedding computation failed for session {session_id} ({node_count} nodes): {details}",
        error_code='EMBEDDING_COMPUTATION_FAILED',
        context={'session_id': session_id, 'node_count': node_count}
    )


def create_graph_construction_error(session_id: str, stage: str, 
                                   details: str) -> GraphConstructionError:
    """Create standardized graph construction error"""
    return GraphConstructionError(
        f"Graph construction failed for session {session_id} at {stage}: {details}",
        error_code='GRAPH_CONSTRUCTION_FAILED',
        context={'session_id': session_id, 'stage': stage}
    )


# Error context helpers

def add_error_context(error: OracleError, **kwargs) -> OracleError:
    """Add additional context to an existing Oracle error"""
    error.context.update(kwargs)
    return error


def get_error_summary(error: OracleError) -> Dict[str, Any]:
    """Get structured summary of Oracle error for logging/reporting"""
    return {
        'error_type': type(error).__name__,
        'error_code': error.error_code,
        'message': str(error),
        'context': error.context
    }


def format_error_for_logging(error: OracleError) -> str:
    """Format Oracle error for structured logging"""
    summary = get_error_summary(error)
    
    parts = [f"{summary['error_type']}: {summary['message']}"]
    
    if summary['error_code']:
        parts.append(f"Code: {summary['error_code']}")
        
    if summary['context']:
        context_str = ", ".join(f"{k}={v}" for k, v in summary['context'].items())
        parts.append(f"Context: {context_str}")
        
    return " | ".join(parts)


# Exception handling decorators

def handle_oracle_errors(error_type: type = OracleError, 
                        default_error_code: str = "UNKNOWN_ERROR"):
    """Decorator to catch and re-raise exceptions as Oracle errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except OracleError:
                # Re-raise Oracle errors as-is
                raise
            except Exception as e:
                # Convert other exceptions to Oracle errors
                raise error_type(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    error_code=default_error_code,
                    context={'function': func.__name__, 'original_error': type(e).__name__}
                ) from e
        return wrapper
    return decorator


def validate_oracle_operation(validation_func):
    """Decorator to validate Oracle operations before execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Run validation
            is_valid, errors = validation_func(*args, **kwargs)
            if not is_valid:
                raise OracleValidationError(
                    f"Validation failed for {func.__name__}: {'; '.join(errors)}",
                    error_code='VALIDATION_ERROR',
                    context={'function': func.__name__, 'validation_errors': errors}
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator
