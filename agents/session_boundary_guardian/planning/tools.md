# Tools for Session Boundary Guardian

## Tool Implementation Specifications

### Tool 1: validate_session_isolation

**Purpose**: Validate that graph structures maintain strict session boundaries
**Pattern**: `@agent.tool` (context-aware)
**Parameters**:
- `graph_data` (Dict[str, Any]): Graph structure to validate
- `session_metadata` (Dict[str, Any]): Session boundary information

**Implementation Pattern**:
```python
@agent.tool
async def validate_session_isolation(
    ctx: RunContext[AgentDependencies],
    graph_data: Dict[str, Any],
    session_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate that graph structure maintains strict session boundaries.
    
    Returns:
    - is_valid: Boolean indicating session isolation compliance
    - violations: List of detected boundary violations
    - edge_audit: Details of cross-session edge analysis
    - recommendations: Remediation steps for violations
    """
```

### Tool 2: detect_contamination

**Purpose**: Identify cross-session learning contamination in ML workflows
**Pattern**: `@agent.tool` (context-aware)
**Parameters**:
- `pipeline_data` (Dict[str, Any]): Pipeline data to analyze
- `contamination_patterns` (List[str]): Known contamination signatures

**Implementation Pattern**:
```python
@agent.tool
async def detect_contamination(
    ctx: RunContext[AgentDependencies],
    pipeline_data: Dict[str, Any],
    contamination_patterns: List[str] = None
) -> Dict[str, Any]:
    """
    Detect cross-session learning contamination in ML pipelines.
    
    Returns:
    - contamination_detected: Boolean indicating contamination presence
    - contamination_sources: List of contamination sources
    - severity_level: Contamination severity assessment
    - remediation_actions: Steps to resolve contamination
    """
```

### Tool 3: audit_htf_compliance

**Purpose**: Validate HTF rule compliance (last-closed data only)
**Pattern**: `@agent.tool` (context-aware)
**Parameters**:
- `htf_features` (Dict[str, Any]): HTF features to validate (f45-f50)
- `candle_metadata` (Dict[str, Any]): Candle timing information

**Implementation Pattern**:
```python
@agent.tool
async def audit_htf_compliance(
    ctx: RunContext[AgentDependencies],
    htf_features: Dict[str, Any],
    candle_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate HTF rule compliance for last-closed data only.
    
    Returns:
    - compliance_status: Boolean indicating HTF rule compliance
    - violations: List of HTF rule violations
    - feature_audit: Analysis of f45-f50 feature compliance
    - timing_analysis: Candle close time validation results
    """
```

### Tool 4: enforce_boundary_constraints

**Purpose**: Apply boundary constraints to pipeline stages
**Pattern**: `@agent.tool` (context-aware)
**Parameters**:
- `stage_name` (str): Name of pipeline stage to validate
- `stage_data` (Dict[str, Any]): Stage data to validate
- `boundary_rules` (Dict[str, Any]): Boundary constraint rules

**Implementation Pattern**:
```python
@agent.tool
async def enforce_boundary_constraints(
    ctx: RunContext[AgentDependencies],
    stage_name: str,
    stage_data: Dict[str, Any],
    boundary_rules: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Apply boundary constraints to pipeline stages.
    
    Returns:
    - constraints_applied: Boolean indicating successful constraint application
    - stage_compliance: Stage compliance assessment
    - boundary_violations: List of boundary constraint violations
    - enforcement_actions: Actions taken to enforce boundaries
    """
```

### Tool 5: audit_archaeological_integrity

**Purpose**: Ensure archaeological discovery maintains temporal non-locality
**Pattern**: `@agent.tool` (context-aware)
**Parameters**:
- `archaeological_data` (Dict[str, Any]): Archaeological discovery data
- `zone_calculations` (Dict[str, Any]): 40% zone calculation data

**Implementation Pattern**:
```python
@agent.tool
async def audit_archaeological_integrity(
    ctx: RunContext[AgentDependencies],
    archaeological_data: Dict[str, Any],
    zone_calculations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ensure archaeological discovery maintains temporal non-locality.
    
    Returns:
    - integrity_maintained: Boolean indicating archaeological integrity
    - temporal_violations: List of temporal non-locality violations
    - zone_compliance: 40% zone calculation compliance assessment
    - theory_b_validation: Theory B temporal ordering validation
    """
```

### Tool 6: generate_boundary_report

**Purpose**: Generate comprehensive session boundary validation report
**Pattern**: `@agent.tool` (context-aware)
**Parameters**:
- `validation_results` (Dict[str, Any]): Consolidated validation results
- `report_format` (str): Output format (json, markdown, html)

**Implementation Pattern**:
```python
@agent.tool
async def generate_boundary_report(
    ctx: RunContext[AgentDependencies],
    validation_results: Dict[str, Any],
    report_format: str = "json"
) -> Dict[str, Any]:
    """
    Generate comprehensive session boundary validation report.
    
    Returns:
    - report_content: Formatted boundary validation report
    - summary_metrics: Key validation metrics
    - recommendations: Remediation recommendations
    - next_actions: Suggested follow-up actions
    """
```

## Parameter Validation

All tools implement strict parameter validation:
- Required parameters must be provided
- Data structures validated against expected schemas
- Temporal data verified for proper formatting
- Session metadata validated for completeness

## Error Handling

Each tool implements comprehensive error handling:
- Invalid input parameter handling
- Boundary validation failure responses
- Contamination detection error recovery
- Archaeological integrity violation responses

## Performance Considerations

- **Caching**: Session boundary rules cached for repeated validation
- **Optimization**: Graph traversal optimized for large session datasets
- **Streaming**: Large pipeline data processed in streaming fashion
- **Parallelization**: Independent validation tasks run in parallel

## Integration Points

Tools integrate with:
- IRONFORGE Enhanced Graph Builder (boundary validation)
- TGAT Discovery pipeline (contamination detection)
- Confluence Scoring (HTF compliance validation)
- Archaeological Zone Detector (temporal integrity validation)