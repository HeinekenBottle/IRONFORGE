"""
IRONFORGE Temporal Engine
========================

Temporal discovery engine for archaeological pattern discovery via TGAT.
Thin re-export layer for canonical import paths.
"""

try:
    from ..learning.discovery_pipeline import run_discovery
    __all__ = ["run_discovery"]
except ImportError as e:
    # Graceful degradation if discovery pipeline is not fully implemented
    import warnings
    warnings.warn(f"Temporal engine not fully available: {e}", ImportWarning)
    
    def run_discovery(*args, **kwargs):
        """Stub function when discovery pipeline is not available"""
        raise NotImplementedError(
            "Temporal discovery engine not fully implemented. "
            "Missing components in ironforge.learning.tgat_discovery. "
            "This is expected in v0.7.x until TGAT spine completion."
        )
    
    __all__ = ["run_discovery"]