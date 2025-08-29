# Contract Compliance Enforcer Agent - Requirements Specification

## Executive Summary
The Contract Compliance Enforcer Agent serves as the guardian of IRONFORGE's data integrity by validating golden invariants, enforcing data contracts, and preventing corrupted data from entering the archaeological discovery pipeline. This agent is critical infrastructure that ensures the system maintains its foundational principles: exactly 6 event types, 4 edge intent types, 51D node features (f0-f50), 20D edge features (e0-e19), session boundary isolation, and HTF last-closed rules.

## Agent Classification
- **Complexity Level**: High - Complex data validation with multi-dimensional contract enforcement
- **Priority**: Critical - Core data integrity for IRONFORGE operations
- **Agent Type**: Validation Agent with contract enforcement capabilities
- **Integration Level**: Deep - Validates all data operations across the entire pipeline

## Functional Requirements

### FR1: Golden Invariant Validation
- **Requirement**: Enforce IRONFORGE's 5 core golden invariants that must never change
- **Acceptance Criteria**: 
  - Validate exactly 6 event types (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
  - Validate exactly 4 edge intent types (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)
  - Validate 51D node features (f0-f50) with proper dimensionality
  - Validate 20D edge features (e0-e19) with proper dimensionality
  - Validate session boundary isolation (no cross-session edges)
- **Priority**: Critical

### FR2: HTF Rule Compliance
- **Requirement**: Enforce Higher Timeframe (HTF) last-closed data rules
- **Acceptance Criteria**:
  - Validate HTF features (f45-f50) use only last-closed candle data
  - Prevent intra-candle HTF data contamination
  - Validate temporal consistency in HTF feature calculations
  - Block pipeline progression on HTF rule violations
- **Priority**: Critical

### FR3: Data Contract Pre-Validation
- **Requirement**: Validate data contracts before any IRONFORGE pipeline operation
- **Acceptance Criteria**:
  - Check data schema compliance before processing
  - Validate feature dimensions and data types
  - Ensure data completeness and integrity
  - Provide detailed validation reports with specific violations
- **Priority**: High

### FR4: Session Isolation Enforcement
- **Requirement**: Ensure strict session boundary isolation throughout the pipeline
- **Acceptance Criteria**:
  - Detect and prevent cross-session edge creation
  - Validate session-specific data processing
  - Ensure no information leakage between sessions
  - Maintain archaeological discovery principles
- **Priority**: Critical

### FR5: Corrupted Data Detection
- **Requirement**: Identify and block corrupted or malformed data from entering the pipeline
- **Acceptance Criteria**:
  - Detect data corruption, missing values, and malformed structures
  - Validate data ranges and statistical consistency
  - Implement data quality scoring and thresholds
  - Provide corruption remediation recommendations
- **Priority**: High

## Technical Requirements

### TR1: Model Integration
- **Primary Model**: OpenAI gpt-4o-mini for contract violation analysis and remediation guidance
- **Reasoning**: Cost-effective model with sufficient reasoning for data validation decisions
- **Fallback**: Claude-3-haiku for high-throughput validation tasks

### TR2: IRONFORGE Data Integration
- **Data Types**: Must validate all IRONFORGE data formats (JSON, Parquet, Graph structures)
- **Schema Validation**: Support for dynamic schema validation and evolution
- **Performance**: Validate data contracts in <2 seconds per session
- **Coverage**: 100% validation coverage for all pipeline data operations

### TR3: Contract Definition Management
- **Storage**: YAML-based contract definitions with version control
- **Evolution**: Support for contract evolution while maintaining backward compatibility
- **Documentation**: Self-documenting contract violations with remediation guidance

### TR4: Integration Points
- **Pipeline Integration**: Hook into all IRONFORGE pipeline stages for pre-processing validation
- **Agent Communication**: Coordinate with pipeline-orchestrator for validation workflows
- **Reporting**: Integrate with reporting systems for validation metrics and compliance dashboards

## Dependencies and Environment

### External Dependencies
- **IRONFORGE Core**: Complete pipeline infrastructure for data access
- **Schema Validation**: Pydantic/JSONSchema for contract definition and validation
- **Data Processing**: Pandas/Polars for data analysis and validation
- **Statistical Analysis**: Scipy/NumPy for data quality analysis

### Agent Dependencies
- **pipeline-orchestrator**: For validation workflow coordination
- **session-boundary-guardian**: For cross-session contamination detection
- **archaeological-zone-detector**: For temporal non-locality validation
- **minidash-enhancer**: For validation reporting and dashboard integration

### Infrastructure
- **Memory**: <20MB footprint for validation operations
- **Processing**: Sub-second validation for most data contracts
- **Storage**: Contract definitions and violation history tracking
- **Alerting**: Real-time alerts for critical contract violations

## Success Criteria

### Primary Success Metrics
1. **Contract Compliance Rate**: 100% enforcement of golden invariants
2. **Validation Speed**: <2 seconds validation time per session
3. **Corruption Detection Rate**: >99% accuracy in identifying corrupted data
4. **False Positive Rate**: <1% false positive rate for validation failures

### Secondary Success Metrics
1. **Remediation Success**: >90% successful remediation guidance
2. **Pipeline Integration**: Seamless integration without performance degradation
3. **Documentation Coverage**: 100% documented contract violations with guidance
4. **Monitoring Coverage**: Complete visibility into validation operations and trends

## Security and Compliance

### Data Security
- **Access Control**: Restrict contract modification to authorized personnel only
- **Audit Trail**: Complete logging of all validation operations and decisions
- **Data Privacy**: Ensure no sensitive data exposure during validation operations

### IRONFORGE Compliance
- **Archaeological Principles**: Validate 40% zone awareness and temporal non-locality
- **Golden Invariants**: Absolute enforcement of the 5 core invariants
- **Performance Requirements**: Maintain <180-second total pipeline processing
- **Quality Standards**: Support 87% authenticity threshold validation

## Testing Requirements

### Unit Testing
- **Contract Validation**: Test validation logic for each contract type
- **Golden Invariant Testing**: Comprehensive testing of all 5 invariants
- **Data Corruption Detection**: Test corruption detection with synthetic corrupt data
- **Performance Testing**: Validate sub-second performance requirements

### Integration Testing  
- **Pipeline Integration**: End-to-end validation integration with all pipeline stages
- **Agent Coordination**: Test coordination with other IRONFORGE agents
- **Contract Evolution**: Test backward compatibility during contract updates
- **Error Scenarios**: Test graceful handling of validation failures

### Compliance Testing
- **Golden Invariant Compliance**: Verify 100% enforcement under all conditions
- **HTF Rule Compliance**: Test HTF validation under various data scenarios
- **Session Isolation**: Test session boundary enforcement under edge conditions
- **Data Quality**: Test corruption detection across various data quality scenarios

## Assumptions Made

### Technical Assumptions
- **Data Format Stability**: Core IRONFORGE data formats remain stable
- **Schema Evolution**: Contract schemas can evolve without breaking existing validations
- **Performance Characteristics**: Validation operations remain sub-second at current data volumes
- **Infrastructure Availability**: IRONFORGE infrastructure components available for validation

### Business Assumptions
- **Contract Immutability**: Golden invariants represent immutable business rules
- **Quality Requirements**: 100% contract compliance remains non-negotiable
- **Performance Requirements**: Sub-second validation remains critical for user experience
- **Documentation Requirements**: All contract violations require detailed remediation guidance

### Integration Assumptions
- **Agent Cooperation**: Other IRONFORGE agents respect contract enforcement decisions
- **Pipeline Integration**: Pipeline stages can be modified to include pre-validation hooks
- **Reporting Integration**: Validation metrics can be integrated into existing dashboards
- **Alert Integration**: Critical violations can trigger immediate alerting mechanisms

## Risk Assessment

### High Risks
1. **False Positives**: Incorrect validation failures blocking legitimate data
2. **Performance Degradation**: Validation adding significant pipeline latency
3. **Contract Drift**: Gradual erosion of contract enforcement over time
4. **Integration Complexity**: Complex integration with existing pipeline components

### Mitigation Strategies
1. **Extensive Testing**: Comprehensive test suites with edge case coverage
2. **Performance Monitoring**: Real-time performance tracking with optimization
3. **Regular Audits**: Periodic audits of contract enforcement effectiveness
4. **Gradual Integration**: Phased integration approach with rollback capabilities

## Contract Definition Examples

### Golden Invariant Contracts
```yaml
golden_invariants:
  event_types:
    required_count: 6
    allowed_types:
      - "Expansion"
      - "Consolidation" 
      - "Retracement"
      - "Reversal"
      - "Liquidity Taken"
      - "Redelivery"
    
  edge_intent_types:
    required_count: 4
    allowed_types:
      - "TEMPORAL_NEXT"
      - "MOVEMENT_TRANSITION"
      - "LIQ_LINK"
      - "CONTEXT"
    
  feature_dimensions:
    node_features:
      required_count: 51
      range: "f0-f50"
    edge_features:
      required_count: 20  
      range: "e0-e19"
```

### HTF Rule Contracts
```yaml
htf_rules:
  htf_features:
    feature_range: "f45-f50"
    data_source: "last_closed_only"
    prohibited_sources:
      - "intra_candle"
      - "current_candle"
      - "future_data"
    
  temporal_consistency:
    validation: "required"
    lookback_limit: "previous_closed_candles_only"
```

This contract compliance enforcer agent serves as the foundational guardian of IRONFORGE's data integrity, ensuring that all archaeological discovery operations maintain the system's core principles and quality standards.