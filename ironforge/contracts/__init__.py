"""
IRONFORGE Contract Enforcement System
====================================

Runtime validators for Golden Invariants and system contracts.
Ensures compliance with:
- Event taxonomy: exactly 6 types
- Edge intents: exactly 4 types  
- Feature dimensions: 51D max nodes, 20D edges
- HTF compliance: last-closed only
- Session isolation: no cross-session edges
"""

from .validators import (
    EventTypeValidator,
    EdgeIntentValidator,
    FeatureDimensionValidator,
    HTFComplianceValidator,
    SessionIsolationValidator,
    SchemaValidator,
    validate_golden_invariants,
    ContractViolationError,
)

from .enforcement import (
    enforce_contracts,
    contract_guard,
    validate_session_data,
    validate_graph_topology,
)

__all__ = [
    # Validators
    "EventTypeValidator",
    "EdgeIntentValidator", 
    "FeatureDimensionValidator",
    "HTFComplianceValidator",
    "SessionIsolationValidator",
    "SchemaValidator",
    "validate_golden_invariants",
    "ContractViolationError",
    
    # Enforcement
    "enforce_contracts",
    "contract_guard",
    "validate_session_data",
    "validate_graph_topology",
]
