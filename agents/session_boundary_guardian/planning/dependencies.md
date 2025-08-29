# Session Boundary Guardian - Dependencies Configuration

## Environment Variables Configuration

```bash
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
LLM_MODEL=gpt-4o-mini

# Session Boundary Guardian Configuration
BOUNDARY_VALIDATION_STRICT=true
CONTAMINATION_DETECTION_ENABLED=true
HTF_COMPLIANCE_THRESHOLD=0.99
ARCHAEOLOGICAL_INTEGRITY_ENABLED=true
BOUNDARY_AUDIT_TIMEOUT=30
SESSION_ISOLATION_ENFORCEMENT=strict

# Performance Configuration
MAX_CONCURRENT_VALIDATIONS=5
BOUNDARY_CACHE_SIZE=1000
VALIDATION_TIMEOUT_SECONDS=1
AUDIT_MEMORY_LIMIT_MB=50

# Integration Configuration
IRONFORGE_CONTAINER_INTEGRATION=true
PIPELINE_STAGE_MONITORING=true
REAL_TIME_VALIDATION=true
BOUNDARY_VIOLATION_ALERTS=true

# Logging Configuration
BOUNDARY_LOG_LEVEL=INFO
CONTAMINATION_LOG_LEVEL=WARNING
HTF_COMPLIANCE_LOG_LEVEL=INFO
ARCHAEOLOGICAL_LOG_LEVEL=INFO
```

## Settings Configuration

```python
from pydantic import BaseSettings, Field
from typing import Dict, Any, List

class SessionBoundaryGuardianSettings(BaseSettings):
    # Core validation settings
    boundary_validation_strict: bool = Field(default=True)
    contamination_detection_enabled: bool = Field(default=True) 
    htf_compliance_threshold: float = Field(default=0.99)
    archaeological_integrity_enabled: bool = Field(default=True)
    
    # Performance settings
    boundary_audit_timeout: int = Field(default=30)
    max_concurrent_validations: int = Field(default=5)
    boundary_cache_size: int = Field(default=1000)
    validation_timeout_seconds: int = Field(default=1)
    audit_memory_limit_mb: int = Field(default=50)
    
    # Session isolation settings
    session_isolation_enforcement: str = Field(default="strict")
    cross_session_edge_tolerance: int = Field(default=0)
    temporal_boundary_precision_ms: int = Field(default=1)
    
    # HTF compliance settings
    htf_last_closed_only: bool = Field(default=True)
    intra_candle_detection: bool = Field(default=True)
    htf_feature_validation: List[str] = Field(default=["f45", "f46", "f47", "f48", "f49", "f50"])
    
    # Archaeological integrity settings
    temporal_non_locality_validation: bool = Field(default=True)
    zone_calculation_boundary_check: bool = Field(default=True)
    theory_b_compliance: bool = Field(default=True)
    dimensional_anchor_validation: bool = Field(default=True)
    
    # Integration settings
    ironforge_container_integration: bool = Field(default=True)
    pipeline_stage_monitoring: bool = Field(default=True)
    real_time_validation: bool = Field(default=True)
    boundary_violation_alerts: bool = Field(default=True)
    
    # Logging settings
    boundary_log_level: str = Field(default="INFO")
    contamination_log_level: str = Field(default="WARNING")
    htf_compliance_log_level: str = Field(default="INFO")
    archaeological_log_level: str = Field(default="INFO")
    
    class Config:
        env_prefix = "BOUNDARY_GUARDIAN_"
```

## Dependencies Dataclass

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.synthesis.pattern_graduation import PatternGraduation
from ironforge.validation.quality_gates import QualityGates

@dataclass
class SessionBoundaryGuardianDependencies:
    # Core IRONFORGE components
    enhanced_graph_builder: EnhancedGraphBuilder
    pattern_graduation: PatternGraduation
    quality_gates: QualityGates
    
    # Agent-specific components
    settings: SessionBoundaryGuardianSettings
    boundary_validator: Any
    contamination_detector: Any
    htf_compliance_checker: Any
    archaeological_integrity_validator: Any
    
    # Performance monitoring
    performance_monitor: Any
    cache_manager: Any
    
    # Integration components
    container_integration: Any
    pipeline_monitor: Any
    alert_system: Any
    
    # Logging and reporting
    logger: Any
    report_generator: Any
    
    # Session metadata
    current_session_context: Optional[Dict[str, Any]] = None
    boundary_rules_cache: Optional[Dict[str, Any]] = None
    validation_history: Optional[List[Dict[str, Any]]] = None
```

## Python Packages

```txt
# Core dependencies
pydantic>=1.10.0
asyncio
dataclasses
typing-extensions

# IRONFORGE integration
ironforge>=1.0.0

# Data processing
pandas>=1.5.0
numpy>=1.20.0
pyarrow>=10.0.0

# Validation and analysis
networkx>=2.8.0
scipy>=1.9.0

# Performance monitoring
psutil>=5.9.0
memory-profiler>=0.60.0

# Logging and reporting
structlog>=22.0.0
rich>=12.0.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.20.0
pytest-mock>=3.8.0
```

## Container Integration Configuration

```python
# IRONFORGE container system integration
CONTAINER_COMPONENTS = {
    "enhanced_graph_builder": "ironforge.learning.enhanced_graph_builder.EnhancedGraphBuilder",
    "pattern_graduation": "ironforge.synthesis.pattern_graduation.PatternGraduation", 
    "quality_gates": "ironforge.validation.quality_gates.QualityGates",
    "performance_monitor": "ironforge.utilities.performance.PerformanceMonitor"
}

# Lazy loading configuration
LAZY_LOAD_COMPONENTS = [
    "boundary_validator",
    "contamination_detector", 
    "htf_compliance_checker",
    "archaeological_integrity_validator"
]

# Integration points
PIPELINE_INTEGRATION_POINTS = {
    "discovery": "validate_session_isolation",
    "confluence": "audit_htf_compliance", 
    "validation": "detect_contamination",
    "reporting": "audit_archaeological_integrity"
}
```

## Data Contract Configuration

```python
# Golden invariants validation
GOLDEN_INVARIANTS = {
    "events": ["Expansion", "Consolidation", "Retracement", "Reversal", "Liquidity Taken", "Redelivery"],
    "edge_intents": ["TEMPORAL_NEXT", "MOVEMENT_TRANSITION", "LIQ_LINK", "CONTEXT"],
    "node_features": 51,  # f0-f50
    "edge_features": 20,  # e0-e19
    "session_boundary_rule": "no_cross_session_edges",
    "htf_rule": "last_closed_only"
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    "session_isolation": 1.0,  # 100% compliance required
    "contamination_detection": 0.99,  # 99% accuracy required
    "htf_compliance": 0.99,  # 99% compliance required
    "archaeological_integrity": 0.98  # 98% integrity required
}
```