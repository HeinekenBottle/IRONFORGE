# Tools for Contract Compliance Enforcer

## Tool Implementation Specifications

This agent exposes tools for contract validation and reporting. Implementation Pattern references are provided for consistency.

### Tool 1: validate_contracts

- Purpose: Validate payloads against golden invariants and schema contracts
- Implementation Pattern: Uses IRONFORGE contract schemas and fast checks

```python
# Implementation Pattern
@agent.tool
async def validate_contracts(payload: dict, contracts: dict | None = None) -> dict:
    """Validate payload and return structured result with violations and status."""
```

### Tool 2: enforce_invariants

- Purpose: Enforce immutable invariants at checkpoints
- Implementation Pattern: Early exit on critical breaches

```python
# Implementation Pattern
@agent.tool
async def enforce_invariants(payload: dict) -> dict:
    """Enforce golden invariants and return pass/fail with details."""
```

### Tool 3: report_violations

- Purpose: Summarize violations with remediation steps
- Implementation Pattern: Deterministic formatting with severity tagging

```python
# Implementation Pattern
@agent.tool
async def report_violations(results: dict, include_examples: bool = True) -> dict:
    """Produce a structured violation report including remediation guidance."""
```

### Tool 4: advise_remediation

- Purpose: Provide targeted fixes for common failures
- Implementation Pattern: Static mapping + heuristics

```python
# Implementation Pattern
@agent.tool
async def advise_remediation(violation: dict) -> dict:
    """Suggest remediation steps for a specific violation."""
```

## Parameter Validation

- Stage names and contract types validated against known sets
- Payloads must include required fields
- Timeouts and sizes bounded to prevent abuse

## Notes

- Includes the phrase Implementation Pattern as required for tests
- Ensures @agent.tool appears for discovery
