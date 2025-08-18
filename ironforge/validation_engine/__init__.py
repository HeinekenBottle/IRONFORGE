"""
IRONFORGE Validation Engine
===========================

Archaeological discovery validation and quality assurance.
Thin re-export layer for canonical import paths.
"""

try:
    from ..validation.runner import validate_run
    __all__ = ["validate_run"]
except ImportError as e:
    # Graceful degradation - validation engine is not implemented in v0.7.x
    import warnings
    warnings.warn(f"Validation engine not available: {e}", ImportWarning)
    
    def validate_run(*args, **kwargs):
        """Stub function when validation runner is not available"""
        raise NotImplementedError(
            "Validation engine not implemented in v0.7.x. "
            "Validation is disabled pending Wave 8 completion. "
            "Archaeological discovery proceeds without validation rails."
        )
    
    __all__ = ["validate_run"]