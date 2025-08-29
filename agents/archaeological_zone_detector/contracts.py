"""
Archaeological Zone Detection Data Contracts
============================================

Data contract validation for archaeological intelligence within IRONFORGE pipeline.
Enforces golden invariants and archaeological compliance requirements.

Contract Categories:
- Golden Invariants: 6 events, 4 edge intents, 51D/20D features (immutable)
- Archaeological Constants: 40% anchoring, 7.55 precision, 87% authenticity
- Session Isolation: Absolute boundary respect, HTF last-closed compliance
- Performance Contracts: <3s processing, >95% accuracy, <100MB memory
- Integration Contracts: TGAT compatibility, Enhanced Graph Builder compliance

Validation Modes:
- strict: Fail on any contract violation (production)
- warn: Log warnings but continue (development)
- ignore: Skip validation (testing only - not recommended)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch

from ironforge.constants import (
    EVENT_TYPES,
    EDGE_INTENTS,
    NODE_FEATURE_DIM_STANDARD,
    NODE_FEATURE_DIM_HTF,
    EDGE_FEATURE_DIM,
)

from .ironforge_config import ArchaeologicalConfig

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Contract validation enforcement modes"""
    STRICT = "strict"        # Fail on violations (production)
    WARN = "warn"           # Log warnings only (development)
    IGNORE = "ignore"       # Skip validation (testing)


class ContractViolationType(Enum):
    """Types of contract violations"""
    GOLDEN_INVARIANT = "golden_invariant"
    ARCHAEOLOGICAL_CONSTANT = "archaeological_constant"
    SESSION_ISOLATION = "session_isolation"
    PERFORMANCE_REQUIREMENT = "performance_requirement"
    INTEGRATION_COMPLIANCE = "integration_compliance"
    DATA_INTEGRITY = "data_integrity"
    HTF_COMPLIANCE = "htf_compliance"


@dataclass
class ContractViolation:
    """Contract violation details"""
    violation_type: ContractViolationType
    contract_name: str
    description: str
    expected_value: Any
    actual_value: Any
    severity: str  # "critical", "major", "minor", "warning"
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.contract_name}: {self.description} "
            f"(expected: {self.expected_value}, actual: {self.actual_value})"
        )


@dataclass
class ValidationResult:
    """Complete validation result for archaeological analysis"""
    session_id: str
    validation_passed: bool
    violations: List[ContractViolation]
    validation_timestamp: float
    validation_duration: float
    performance_metrics: Dict[str, float]
    
    @property
    def critical_violations(self) -> List[ContractViolation]:
        return [v for v in self.violations if v.severity == "critical"]
    
    @property
    def has_critical_violations(self) -> bool:
        return len(self.critical_violations) > 0
    
    def get_violations_by_type(self, violation_type: ContractViolationType) -> List[ContractViolation]:
        return [v for v in self.violations if v.violation_type == violation_type]


class ArchaeologicalContractValidator:
    """
    Archaeological Zone Detection Contract Validator
    
    Validates archaeological analysis against IRONFORGE golden invariants and
    archaeological intelligence requirements. Ensures production-grade compliance
    with all contract obligations.
    
    Golden Invariants (Never Change):
    - Event Types: Exactly 6 canonical types
    - Edge Intents: Exactly 4 canonical types  
    - Feature Dimensions: 51D nodes (45D standard + 6D HTF), 20D edges
    - Session Boundaries: Absolute isolation enforcement
    - HTF Compliance: Last-closed only, no intra-candle data
    
    Archaeological Constants:
    - Dimensional Anchoring: 40% of previous range
    - Precision Target: 7.55-point accuracy
    - Authenticity Threshold: 87% minimum for production
    """
    
    def __init__(
        self,
        config: Optional[ArchaeologicalConfig] = None,
        validation_mode: ValidationMode = ValidationMode.STRICT
    ):
        self.config = config if config is not None else ArchaeologicalConfig()
        self.validation_mode = validation_mode
        self.validation_config = self.config.validation
        
        # Contract enforcement flags
        self.enforce_golden_invariants = self.validation_config.golden_invariant_enforcement
        self.enforce_archaeological_constants = True  # Always enforce
        self.enforce_session_isolation = self.validation_config.session_isolation_validation
        self.enforce_performance_requirements = True  # Always enforce
        
        logger.debug(f"Archaeological Contract Validator initialized (mode: {validation_mode.value})")
    
    def validate_archaeological_output(
        self,
        archaeological_zones: List[Any],
        session_id: str,
        session_data: Optional[pd.DataFrame] = None,
        enhanced_graph: Optional[nx.Graph] = None
    ) -> Dict[str, bool]:
        """
        Validate archaeological analysis output against all contracts
        
        Args:
            archaeological_zones: Generated archaeological zones
            session_id: Session identifier for tracking
            session_data: Optional session data for validation
            enhanced_graph: Optional enhanced graph for validation
            
        Returns:
            Contract validation results dictionary
        """
        validation_start = time.time()
        violations = []
        
        try:
            # Golden Invariant Validation
            if self.enforce_golden_invariants:
                golden_violations = self._validate_golden_invariants(
                    archaeological_zones, session_data, enhanced_graph, session_id
                )
                violations.extend(golden_violations)
            
            # Archaeological Constant Validation
            if self.enforce_archaeological_constants:
                constant_violations = self._validate_archaeological_constants(
                    archaeological_zones, session_id
                )
                violations.extend(constant_violations)
            
            # Session Isolation Validation
            if self.enforce_session_isolation:
                isolation_violations = self._validate_session_isolation(
                    archaeological_zones, session_data, session_id
                )
                violations.extend(isolation_violations)
            
            # Performance Requirement Validation
            if self.enforce_performance_requirements:
                performance_violations = self._validate_performance_requirements(
                    archaeological_zones, validation_start, session_id
                )
                violations.extend(performance_violations)
            
            # Integration Compliance Validation
            integration_violations = self._validate_integration_compliance(
                archaeological_zones, enhanced_graph, session_id
            )
            violations.extend(integration_violations)
            
            # Compile validation results
            validation_duration = time.time() - validation_start
            validation_passed = not any(v.severity == "critical" for v in violations)
            
            # Handle violations based on validation mode
            self._handle_violations(violations, session_id)
            
            # Return contract results dictionary
            contract_results = {
                "golden_invariants": not any(
                    v.violation_type == ContractViolationType.GOLDEN_INVARIANT and v.severity == "critical"
                    for v in violations
                ),
                "archaeological_constants": not any(
                    v.violation_type == ContractViolationType.ARCHAEOLOGICAL_CONSTANT and v.severity == "critical"
                    for v in violations
                ),
                "session_isolation": not any(
                    v.violation_type == ContractViolationType.SESSION_ISOLATION and v.severity == "critical"
                    for v in violations
                ),
                "performance_requirements": not any(
                    v.violation_type == ContractViolationType.PERFORMANCE_REQUIREMENT and v.severity == "critical"
                    for v in violations
                ),
                "integration_compliance": not any(
                    v.violation_type == ContractViolationType.INTEGRATION_COMPLIANCE and v.severity == "critical"
                    for v in violations
                ),
                "overall_passed": validation_passed,
                "total_violations": len(violations),
                "critical_violations": len([v for v in violations if v.severity == "critical"]),
                "validation_duration": validation_duration
            }
            
            if not validation_passed:
                logger.warning(
                    f"Archaeological contract validation failed for {session_id}: "
                    f"{len([v for v in violations if v.severity == 'critical'])} critical violations"
                )
            
            return contract_results
            
        except Exception as e:
            logger.error(f"Contract validation failed for {session_id}: {e}")
            return {
                "golden_invariants": False,
                "archaeological_constants": False,
                "session_isolation": False,
                "performance_requirements": False,
                "integration_compliance": False,
                "overall_passed": False,
                "validation_error": str(e)
            }
    
    def validate_event_taxonomy(self, event_types: List[str]) -> None:
        """
        Validate event type taxonomy against golden invariants
        
        Args:
            event_types: List of event types to validate
            
        Raises:
            ArchaeologicalContractViolationError: If event taxonomy is invalid
        """
        if not self.enforce_golden_invariants:
            return
        
        expected_types = set(EVENT_TYPES)
        actual_types = set(event_types)
        
        # Check exact match requirement
        if expected_types != actual_types:
            missing_types = expected_types - actual_types
            extra_types = actual_types - expected_types
            
            violation = ContractViolation(
                violation_type=ContractViolationType.GOLDEN_INVARIANT,
                contract_name="event_taxonomy",
                description=f"Event taxonomy mismatch: missing {missing_types}, extra {extra_types}",
                expected_value=list(expected_types),
                actual_value=list(actual_types),
                severity="critical"
            )
            
            self._handle_single_violation(violation)
    
    def validate_feature_dimensions(
        self, 
        actual_node_dims: int, 
        actual_edge_dims: Optional[int] = None
    ) -> None:
        """
        Validate feature dimensions against golden invariants
        
        Args:
            actual_node_dims: Actual node feature dimensions
            actual_edge_dims: Actual edge feature dimensions (optional)
            
        Raises:
            ArchaeologicalContractViolationError: If dimensions are invalid
        """
        if not self.enforce_golden_invariants:
            return
        
        # Validate node dimensions
        valid_node_dims = [NODE_FEATURE_DIM_STANDARD, NODE_FEATURE_DIM_HTF]
        if actual_node_dims not in valid_node_dims:
            violation = ContractViolation(
                violation_type=ContractViolationType.GOLDEN_INVARIANT,
                contract_name="node_feature_dimensions",
                description=f"Invalid node feature dimensions",
                expected_value=f"{NODE_FEATURE_DIM_STANDARD} or {NODE_FEATURE_DIM_HTF}",
                actual_value=actual_node_dims,
                severity="critical"
            )
            self._handle_single_violation(violation)
        
        # Validate edge dimensions if provided
        if actual_edge_dims is not None and actual_edge_dims != EDGE_FEATURE_DIM:
            violation = ContractViolation(
                violation_type=ContractViolationType.GOLDEN_INVARIANT,
                contract_name="edge_feature_dimensions",
                description=f"Invalid edge feature dimensions",
                expected_value=EDGE_FEATURE_DIM,
                actual_value=actual_edge_dims,
                severity="critical"
            )
            self._handle_single_violation(violation)
    
    def validate_session_isolation(
        self, 
        session_data: pd.DataFrame, 
        session_id: str
    ) -> None:
        """
        Validate absolute session isolation enforcement
        
        Args:
            session_data: Session data to validate
            session_id: Session identifier
            
        Raises:
            ArchaeologicalContractViolationError: If session isolation is violated
        """
        if not self.enforce_session_isolation:
            return
        
        # Check for cross-session contamination indicators
        contamination_indicators = [
            "prev_session", "next_session", "cross_session",
            "session_bridge", "inter_session"
        ]
        
        for column in session_data.columns:
            for indicator in contamination_indicators:
                if indicator.lower() in column.lower():
                    violation = ContractViolation(
                        violation_type=ContractViolationType.SESSION_ISOLATION,
                        contract_name="session_contamination",
                        description=f"Potential cross-session contamination in column '{column}'",
                        expected_value="No cross-session references",
                        actual_value=f"Column '{column}' contains cross-session indicator",
                        severity="critical",
                        session_id=session_id
                    )
                    self._handle_single_violation(violation)
        
        # Validate session boundary markers
        if 'session_id' in session_data.columns:
            unique_sessions = session_data['session_id'].nunique()
            if unique_sessions > 1:
                violation = ContractViolation(
                    violation_type=ContractViolationType.SESSION_ISOLATION,
                    contract_name="session_boundary",
                    description=f"Multiple sessions detected in single session data",
                    expected_value=1,
                    actual_value=unique_sessions,
                    severity="critical",
                    session_id=session_id
                )
                self._handle_single_violation(violation)
    
    def _validate_golden_invariants(
        self,
        archaeological_zones: List[Any],
        session_data: Optional[pd.DataFrame],
        enhanced_graph: Optional[nx.Graph],
        session_id: str
    ) -> List[ContractViolation]:
        """Validate all golden invariant contracts"""
        violations = []
        
        try:
            # Event type validation (if session data provided)
            if session_data is not None and 'event_type' in session_data.columns:
                try:
                    event_types = session_data['event_type'].unique().tolist()
                    self.validate_event_taxonomy(event_types)
                except Exception as e:
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.GOLDEN_INVARIANT,
                        contract_name="event_taxonomy_validation",
                        description=f"Event taxonomy validation failed: {str(e)}",
                        expected_value="Valid event taxonomy",
                        actual_value="Validation failure",
                        severity="critical",
                        session_id=session_id
                    ))
            
            # Feature dimension validation (if enhanced graph provided)
            if enhanced_graph is not None:
                node_features = None
                edge_features = None
                
                # Extract feature dimensions from graph
                for node_id, node_data in enhanced_graph.nodes(data=True):
                    if 'features' in node_data:
                        if isinstance(node_data['features'], torch.Tensor):
                            node_features = node_data['features'].shape[-1]
                        elif hasattr(node_data['features'], '__len__'):
                            node_features = len(node_data['features'])
                        break
                
                for u, v, edge_data in enhanced_graph.edges(data=True):
                    if 'features' in edge_data:
                        if isinstance(edge_data['features'], torch.Tensor):
                            edge_features = edge_data['features'].shape[-1]
                        elif hasattr(edge_data['features'], '__len__'):
                            edge_features = len(edge_data['features'])
                        break
                
                # Validate extracted dimensions
                if node_features is not None:
                    try:
                        self.validate_feature_dimensions(node_features, edge_features)
                    except Exception as e:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.GOLDEN_INVARIANT,
                            contract_name="feature_dimensions",
                            description=f"Feature dimension validation failed: {str(e)}",
                            expected_value="Valid feature dimensions",
                            actual_value=f"Node: {node_features}, Edge: {edge_features}",
                            severity="critical",
                            session_id=session_id
                        ))
            
            # Edge intent validation (if enhanced graph provided)
            if enhanced_graph is not None:
                edge_intents = set()
                for u, v, edge_data in enhanced_graph.edges(data=True):
                    if 'intent' in edge_data:
                        edge_intents.add(edge_data['intent'])
                
                if edge_intents:
                    expected_intents = set(EDGE_INTENTS)
                    if not edge_intents.issubset(expected_intents):
                        invalid_intents = edge_intents - expected_intents
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.GOLDEN_INVARIANT,
                            contract_name="edge_intent_taxonomy",
                            description=f"Invalid edge intents detected: {invalid_intents}",
                            expected_value=list(expected_intents),
                            actual_value=list(edge_intents),
                            severity="critical",
                            session_id=session_id
                        ))
            
        except Exception as e:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.GOLDEN_INVARIANT,
                contract_name="golden_invariant_validation",
                description=f"Golden invariant validation error: {str(e)}",
                expected_value="Successful validation",
                actual_value="Validation error",
                severity="major",
                session_id=session_id
            ))
        
        return violations
    
    def _validate_archaeological_constants(
        self,
        archaeological_zones: List[Any],
        session_id: str
    ) -> List[ContractViolation]:
        """Validate archaeological constants compliance"""
        violations = []
        
        try:
            # Validate 40% dimensional anchoring
            for i, zone in enumerate(archaeological_zones):
                if hasattr(zone, 'zone_range') or isinstance(zone, dict):
                    # Extract zone properties
                    zone_dict = zone.__dict__ if hasattr(zone, '__dict__') else zone
                    
                    # Check anchor percentage consistency
                    previous_range = zone_dict.get('previous_range', 0.0)
                    zone_width = zone_dict.get('zone_width', 0.0)
                    
                    if previous_range > 0 and zone_width > 0:
                        actual_percentage = zone_width / previous_range
                        expected_percentage = self.config.dimensional_anchor.anchor_percentage
                        
                        # Allow some tolerance for archaeological constant
                        tolerance = 0.05  # 5% tolerance
                        if abs(actual_percentage - expected_percentage) > tolerance:
                            violations.append(ContractViolation(
                                violation_type=ContractViolationType.ARCHAEOLOGICAL_CONSTANT,
                                contract_name="dimensional_anchor_percentage",
                                description=f"Zone {i} anchor percentage deviation",
                                expected_value=expected_percentage,
                                actual_value=actual_percentage,
                                severity="major",
                                session_id=session_id
                            ))
                    
                    # Check authenticity threshold
                    authenticity_score = zone_dict.get('authenticity_score', 0.0)
                    if authenticity_score > 0:  # Only check if score is available
                        min_authenticity = self.config.authenticity.authenticity_threshold
                        if authenticity_score < min_authenticity:
                            violations.append(ContractViolation(
                                violation_type=ContractViolationType.ARCHAEOLOGICAL_CONSTANT,
                                contract_name="authenticity_threshold",
                                description=f"Zone {i} below authenticity threshold",
                                expected_value=f">= {min_authenticity}",
                                actual_value=authenticity_score,
                                severity="minor",  # Minor since this filters zones
                                session_id=session_id
                            ))
                    
                    # Check precision target approach
                    precision_score = zone_dict.get('precision_score', 0.0)
                    if precision_score > 0:
                        precision_target = self.config.dimensional_anchor.precision_target
                        precision_deviation = abs(precision_score - precision_target) / precision_target
                        
                        # Warn if precision is far from target
                        if precision_deviation > 0.5:  # 50% deviation
                            violations.append(ContractViolation(
                                violation_type=ContractViolationType.ARCHAEOLOGICAL_CONSTANT,
                                contract_name="precision_target",
                                description=f"Zone {i} precision far from target",
                                expected_value=f"~{precision_target}",
                                actual_value=precision_score,
                                severity="warning",
                                session_id=session_id
                            ))
            
        except Exception as e:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.ARCHAEOLOGICAL_CONSTANT,
                contract_name="archaeological_constant_validation",
                description=f"Archaeological constant validation error: {str(e)}",
                expected_value="Successful validation",
                actual_value="Validation error",
                severity="major",
                session_id=session_id
            ))
        
        return violations
    
    def _validate_session_isolation(
        self,
        archaeological_zones: List[Any],
        session_data: Optional[pd.DataFrame],
        session_id: str
    ) -> List[ContractViolation]:
        """Validate session isolation compliance"""
        violations = []
        
        try:
            # Validate session data isolation
            if session_data is not None:
                try:
                    self.validate_session_isolation(session_data, session_id)
                except Exception as e:
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.SESSION_ISOLATION,
                        contract_name="session_data_isolation",
                        description=f"Session isolation validation failed: {str(e)}",
                        expected_value="Isolated session data",
                        actual_value="Contaminated session data",
                        severity="critical",
                        session_id=session_id
                    ))
            
            # Validate zones belong to single session
            zone_sessions = set()
            for zone in archaeological_zones:
                zone_dict = zone.__dict__ if hasattr(zone, '__dict__') else zone
                zone_session_id = zone_dict.get('session_id', session_id)
                zone_sessions.add(zone_session_id)
            
            if len(zone_sessions) > 1:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.SESSION_ISOLATION,
                    contract_name="zone_session_consistency",
                    description=f"Zones span multiple sessions",
                    expected_value="Single session",
                    actual_value=f"{len(zone_sessions)} sessions: {zone_sessions}",
                    severity="critical",
                    session_id=session_id
                ))
            
        except Exception as e:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.SESSION_ISOLATION,
                contract_name="session_isolation_validation",
                description=f"Session isolation validation error: {str(e)}",
                expected_value="Successful validation",
                actual_value="Validation error",
                severity="major",
                session_id=session_id
            ))
        
        return violations
    
    def _validate_performance_requirements(
        self,
        archaeological_zones: List[Any],
        validation_start: float,
        session_id: str
    ) -> List[ContractViolation]:
        """Validate performance requirements"""
        violations = []
        
        try:
            # Check processing time
            current_time = time.time()
            processing_time = current_time - validation_start
            max_processing_time = self.config.performance.max_session_processing_time
            
            if processing_time > max_processing_time:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.PERFORMANCE_REQUIREMENT,
                    contract_name="processing_time_limit",
                    description=f"Processing time exceeded limit",
                    expected_value=f"<= {max_processing_time}s",
                    actual_value=f"{processing_time:.3f}s",
                    severity="major",
                    session_id=session_id
                ))
            
            # Check zone count reasonableness
            zone_count = len(archaeological_zones)
            min_zones = self.config.authenticity.minimum_zones_for_analysis
            max_zones = self.config.authenticity.maximum_zones_per_session
            
            if zone_count < min_zones:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.PERFORMANCE_REQUIREMENT,
                    contract_name="minimum_zone_count",
                    description=f"Insufficient zones for analysis",
                    expected_value=f">= {min_zones}",
                    actual_value=zone_count,
                    severity="minor",
                    session_id=session_id
                ))
            elif zone_count > max_zones:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.PERFORMANCE_REQUIREMENT,
                    contract_name="maximum_zone_count",
                    description=f"Too many zones may impact performance",
                    expected_value=f"<= {max_zones}",
                    actual_value=zone_count,
                    severity="warning",
                    session_id=session_id
                ))
            
        except Exception as e:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.PERFORMANCE_REQUIREMENT,
                contract_name="performance_validation",
                description=f"Performance validation error: {str(e)}",
                expected_value="Successful validation",
                actual_value="Validation error",
                severity="major",
                session_id=session_id
            ))
        
        return violations
    
    def _validate_integration_compliance(
        self,
        archaeological_zones: List[Any],
        enhanced_graph: Optional[nx.Graph],
        session_id: str
    ) -> List[ContractViolation]:
        """Validate IRONFORGE integration compliance"""
        violations = []
        
        try:
            # Validate archaeological zone structure
            for i, zone in enumerate(archaeological_zones):
                zone_dict = zone.__dict__ if hasattr(zone, '__dict__') else zone
                
                # Required archaeological zone properties
                required_properties = [
                    'anchor_point', 'zone_range', 'confidence', 
                    'session_id', 'authenticity_score'
                ]
                
                for prop in required_properties:
                    if prop not in zone_dict:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.INTEGRATION_COMPLIANCE,
                            contract_name="zone_structure",
                            description=f"Zone {i} missing required property '{prop}'",
                            expected_value=f"Property '{prop}' present",
                            actual_value=f"Property '{prop}' missing",
                            severity="major",
                            session_id=session_id
                        ))
                
                # Validate zone range structure
                zone_range = zone_dict.get('zone_range')
                if zone_range is not None:
                    if not isinstance(zone_range, (tuple, list)) or len(zone_range) != 2:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.INTEGRATION_COMPLIANCE,
                            contract_name="zone_range_structure",
                            description=f"Zone {i} has invalid zone_range structure",
                            expected_value="Tuple/list of length 2",
                            actual_value=f"{type(zone_range)} of length {len(zone_range) if hasattr(zone_range, '__len__') else 'unknown'}",
                            severity="major",
                            session_id=session_id
                        ))
                    elif zone_range[0] >= zone_range[1]:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.INTEGRATION_COMPLIANCE,
                            contract_name="zone_range_order",
                            description=f"Zone {i} has invalid zone_range order",
                            expected_value="zone_range[0] < zone_range[1]",
                            actual_value=f"zone_range = {zone_range}",
                            severity="major",
                            session_id=session_id
                        ))
            
            # Validate enhanced graph compatibility (if provided)
            if enhanced_graph is not None:
                # Check graph structure
                if not isinstance(enhanced_graph, nx.Graph):
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.INTEGRATION_COMPLIANCE,
                        contract_name="graph_type",
                        description=f"Enhanced graph is not NetworkX Graph",
                        expected_value="networkx.Graph",
                        actual_value=str(type(enhanced_graph)),
                        severity="critical",
                        session_id=session_id
                    ))
                
                # Check for required graph properties
                if len(enhanced_graph.nodes) == 0:
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.INTEGRATION_COMPLIANCE,
                        contract_name="graph_nodes",
                        description=f"Enhanced graph has no nodes",
                        expected_value="> 0 nodes",
                        actual_value="0 nodes",
                        severity="major",
                        session_id=session_id
                    ))
            
        except Exception as e:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.INTEGRATION_COMPLIANCE,
                contract_name="integration_compliance_validation",
                description=f"Integration compliance validation error: {str(e)}",
                expected_value="Successful validation",
                actual_value="Validation error",
                severity="major",
                session_id=session_id
            ))
        
        return violations
    
    def _handle_violations(self, violations: List[ContractViolation], session_id: str) -> None:
        """Handle contract violations based on validation mode"""
        if not violations:
            return
        
        critical_violations = [v for v in violations if v.severity == "critical"]
        major_violations = [v for v in violations if v.severity == "major"]
        minor_violations = [v for v in violations if v.severity == "minor"]
        warnings = [v for v in violations if v.severity == "warning"]
        
        # Log all violations
        for violation in violations:
            if violation.severity == "critical":
                logger.error(f"CRITICAL CONTRACT VIOLATION: {violation}")
            elif violation.severity == "major":
                logger.error(f"MAJOR CONTRACT VIOLATION: {violation}")
            elif violation.severity == "minor":
                logger.warning(f"MINOR CONTRACT VIOLATION: {violation}")
            else:
                logger.info(f"CONTRACT WARNING: {violation}")
        
        # Handle based on validation mode
        if self.validation_mode == ValidationMode.STRICT:
            if critical_violations:
                raise ArchaeologicalContractViolationError(
                    f"Critical contract violations in session {session_id}: "
                    f"{[str(v) for v in critical_violations]}"
                )
            elif major_violations and self.validation_config.validation_failure_mode == "strict":
                raise ArchaeologicalContractViolationError(
                    f"Major contract violations in session {session_id}: "
                    f"{[str(v) for v in major_violations]}"
                )
        
        elif self.validation_mode == ValidationMode.WARN:
            if critical_violations or major_violations:
                logger.warning(
                    f"Contract violations detected in session {session_id} "
                    f"({len(critical_violations)} critical, {len(major_violations)} major) - "
                    f"continuing due to WARN mode"
                )
        
        # IGNORE mode: no action taken
    
    def _handle_single_violation(self, violation: ContractViolation) -> None:
        """Handle a single contract violation immediately"""
        if self.validation_mode == ValidationMode.STRICT and violation.severity == "critical":
            raise ArchaeologicalContractViolationError(str(violation))
        elif self.validation_mode == ValidationMode.WARN:
            if violation.severity == "critical":
                logger.error(f"CONTRACT VIOLATION: {violation}")
            elif violation.severity == "major":
                logger.warning(f"CONTRACT VIOLATION: {violation}")
            else:
                logger.info(f"CONTRACT NOTE: {violation}")
        # IGNORE mode: no action


class ArchaeologicalContractViolationError(Exception):
    """Raised when archaeological contract validation fails in strict mode"""
    pass


class HTFComplianceValidator:
    """
    High Timeframe (HTF) Compliance Validator
    
    Validates HTF feature compliance with last-closed data requirements.
    Ensures no intra-candle HTF data contamination in archaeological analysis.
    """
    
    def __init__(self, config: ArchaeologicalConfig):
        self.config = config
        self.session_config = config.session_isolation
        logger.debug("HTF Compliance Validator initialized")
    
    def validate_htf_compliance(
        self,
        session_data: pd.DataFrame,
        session_id: str
    ) -> List[ContractViolation]:
        """
        Validate HTF compliance for session data
        
        Args:
            session_data: Session data to validate
            session_id: Session identifier
            
        Returns:
            List of HTF compliance violations
        """
        violations = []
        
        if not self.session_config.htf_last_closed_only:
            return violations  # HTF compliance not enforced
        
        try:
            # Find HTF feature columns (f45-f50)
            htf_columns = [
                col for col in session_data.columns 
                if col.startswith('f') and col[1:].isdigit() 
                and int(col[1:]) in range(45, 51)
            ]
            
            if not htf_columns:
                return violations  # No HTF features present
            
            # Validate HTF features don't change intra-session (last-closed requirement)
            for col in htf_columns:
                htf_values = session_data[col].dropna()
                if len(htf_values) > 1:
                    # Check if HTF values change within session
                    if htf_values.nunique() > 1:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.HTF_COMPLIANCE,
                            contract_name="htf_last_closed_only",
                            description=f"HTF feature {col} changes within session",
                            expected_value="Constant HTF values (last-closed)",
                            actual_value=f"{htf_values.nunique()} unique values",
                            severity="critical",
                            session_id=session_id
                        ))
            
            # Validate no intra-candle HTF indicators
            intra_candle_indicators = [
                'intra_candle', 'real_time', 'live', 'current_candle'
            ]
            
            for col in session_data.columns:
                for indicator in intra_candle_indicators:
                    if indicator.lower() in col.lower():
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.HTF_COMPLIANCE,
                            contract_name="intra_candle_rejection",
                            description=f"Potential intra-candle HTF data in column '{col}'",
                            expected_value="No intra-candle HTF data",
                            actual_value=f"Column '{col}' suggests intra-candle data",
                            severity="major",
                            session_id=session_id
                        ))
            
        except Exception as e:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.HTF_COMPLIANCE,
                contract_name="htf_compliance_validation",
                description=f"HTF compliance validation error: {str(e)}",
                expected_value="Successful HTF validation",
                actual_value="Validation error",
                severity="major",
                session_id=session_id
            ))
        
        return violations


class PerformanceContractValidator:
    """
    Performance Contract Validator
    
    Validates performance requirements for archaeological zone detection:
    - <3s session processing time
    - >95% anchor accuracy
    - <100MB memory usage
    - <1s zone detection time
    """
    
    def __init__(self, config: ArchaeologicalConfig):
        self.config = config
        self.perf_config = config.performance
        logger.debug("Performance Contract Validator initialized")
    
    def validate_performance_contracts(
        self,
        processing_metrics: Dict[str, float],
        session_id: str
    ) -> List[ContractViolation]:
        """
        Validate performance contracts
        
        Args:
            processing_metrics: Performance metrics dictionary
            session_id: Session identifier
            
        Returns:
            List of performance contract violations
        """
        violations = []
        
        try:
            # Validate session processing time
            processing_time = processing_metrics.get('processing_time', 0.0)
            if processing_time > self.perf_config.max_session_processing_time:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.PERFORMANCE_REQUIREMENT,
                    contract_name="session_processing_time",
                    description=f"Session processing time exceeded limit",
                    expected_value=f"<= {self.perf_config.max_session_processing_time}s",
                    actual_value=f"{processing_time:.3f}s",
                    severity="major",
                    session_id=session_id
                ))
            
            # Validate zone detection time
            detection_time = processing_metrics.get('detection_time', 0.0)
            if detection_time > self.perf_config.max_detection_time:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.PERFORMANCE_REQUIREMENT,
                    contract_name="zone_detection_time",
                    description=f"Zone detection time exceeded limit",
                    expected_value=f"<= {self.perf_config.max_detection_time}s",
                    actual_value=f"{detection_time:.3f}s",
                    severity="major",
                    session_id=session_id
                ))
            
            # Validate anchor accuracy
            anchor_accuracy = processing_metrics.get('accuracy', 0.0)
            if anchor_accuracy < self.perf_config.min_anchor_accuracy:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.PERFORMANCE_REQUIREMENT,
                    contract_name="anchor_accuracy",
                    description=f"Anchor accuracy below minimum requirement",
                    expected_value=f">= {self.perf_config.min_anchor_accuracy}%",
                    actual_value=f"{anchor_accuracy:.1f}%",
                    severity="major",
                    session_id=session_id
                ))
            
            # Validate memory usage
            memory_usage = processing_metrics.get('memory_usage_mb', 0.0)
            if memory_usage > self.perf_config.max_memory_usage_mb:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.PERFORMANCE_REQUIREMENT,
                    contract_name="memory_usage",
                    description=f"Memory usage exceeded limit",
                    expected_value=f"<= {self.perf_config.max_memory_usage_mb}MB",
                    actual_value=f"{memory_usage:.1f}MB",
                    severity="minor",  # Minor as it may not affect correctness
                    session_id=session_id
                ))
            
        except Exception as e:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.PERFORMANCE_REQUIREMENT,
                contract_name="performance_validation",
                description=f"Performance validation error: {str(e)}",
                expected_value="Successful performance validation",
                actual_value="Validation error",
                severity="major",
                session_id=session_id
            ))
        
        return violations


# Factory functions for contract validation
def create_strict_validator(config: Optional[ArchaeologicalConfig] = None) -> ArchaeologicalContractValidator:
    """Create strict contract validator for production use"""
    return ArchaeologicalContractValidator(config, ValidationMode.STRICT)


def create_development_validator(config: Optional[ArchaeologicalConfig] = None) -> ArchaeologicalContractValidator:
    """Create warning-mode validator for development use"""
    return ArchaeologicalContractValidator(config, ValidationMode.WARN)


def create_testing_validator(config: Optional[ArchaeologicalConfig] = None) -> ArchaeologicalContractValidator:
    """Create permissive validator for testing use"""
    return ArchaeologicalContractValidator(config, ValidationMode.IGNORE)


# Export all contract components
__all__ = [
    "ArchaeologicalContractValidator",
    "HTFComplianceValidator",
    "PerformanceContractValidator",
    "ContractViolation",
    "ValidationResult",
    "ValidationMode",
    "ContractViolationType",
    "ArchaeologicalContractViolationError",
    "create_strict_validator",
    "create_development_validator", 
    "create_testing_validator"
]