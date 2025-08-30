# Session Boundary Guardian Agent - Requirements Specification

## Executive Summary
The Session Boundary Guardian Agent enforces IRONFORGE's critical session isolation principles by validating that no cross-session contamination occurs in the archaeological discovery pipeline. This agent is essential for maintaining the integrity of IRONFORGE's temporal non-locality analysis and ensuring that patterns discovered within sessions remain temporally isolated, preventing future information from leaking into historical analysis.

## Agent Classification
- **Complexity Level**: High - Complex validation with multi-stage boundary checking
- **Priority**: Critical - Core data integrity for archaeological discovery
- **Agent Type**: Validation Agent with boundary enforcement capabilities
- **Integration Level**: Deep - Integrates with all pipeline stages

## Functional Requirements

### FR1: Session Isolation Validation
- **Requirement**: Validate that graph construction maintains strict session boundaries
- **Acceptance Criteria**: 
  - No cross-session edges in any graph structure
  - All temporal connections respect session boundaries
  - HTF features only use last-closed data from previous sessions
  - No intra-session future information leakage
- **Priority**: Critical

### FR2: Cross-Session Contamination Detection
- **Requirement**: Detect and prevent cross-session learning contamination in ML pipelines
- **Acceptance Criteria**:
  - Identify any model inputs that contain future session information
  - Validate that training data respects temporal boundaries
  - Ensure TGAT attention mechanisms don't span sessions
  - Detect any contamination in pattern discovery workflows
- **Priority**: Critical

### FR3: HTF Rule Compliance
- **Requirement**: Enforce the HTF rule that only last-closed candle data can be used
- **Acceptance Criteria**:
  - Validate that f45-f50 HTF features use only closed candle data
  - Prevent any intra-candle HTF feature contamination
  - Ensure temporal ordering respects market close times
  - Audit all HTF feature calculations for compliance
- **Priority**: High

### FR4: Pipeline Stage Boundary Validation
- **Requirement**: Validate boundary constraints across all IRONFORGE pipeline stages
- **Acceptance Criteria**:
  - Discovery stage maintains session isolation
  - Confluence scoring doesn't leak cross-session information
  - Validation stage respects temporal boundaries
  - Reporting maintains session context integrity
- **Priority**: High

### FR5: Archaeological Integrity Maintenance
- **Requirement**: Ensure archaeological discovery principles maintain temporal non-locality
- **Acceptance Criteria**:
  - 40% zone calculations respect session boundaries
  - Dimensional anchor points don't use future information
  - Theory B validation maintains proper temporal ordering
  - Archaeological intelligence remains session-isolated
- **Priority**: High

## Technical Requirements
- **Model Integration**: OpenAI gpt-4o-mini for contamination analysis and reporting
- **Performance**: <1 second per session validation, <30 seconds full pipeline audit
- **Quality**: >99% boundary violation detection accuracy
- **Integration**: Real-time validation during all pipeline stages

## Success Criteria
- **Primary Metric**: Zero cross-session contamination incidents in production
- **Performance**: Sub-second session boundary validation
- **Coverage**: 100% of pipeline stages validated for boundary compliance
- **Quality**: >99% accuracy in contamination detection

## Risk Assessment
- **High Risk**: Cross-session contamination could invalidate all archaeological discoveries
- **Medium Risk**: Performance impact from validation overhead
- **Low Risk**: False positive contamination detection

## Assumptions
- Session boundaries are clearly defined in input data
- All pipeline stages provide sufficient metadata for boundary validation
- Archaeological discovery maintains session-based temporal isolation
- HTF features respect market timing and closure rules