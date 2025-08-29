"""
IRONFORGE Contract Validators
============================

Runtime validators for Golden Invariants enforcement.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd
import networkx as nx

from ironforge.constants import (
    EVENT_TYPES,
    EDGE_INTENTS,
    NODE_FEATURE_DIM_STANDARD,
    NODE_FEATURE_DIM_HTF,
    EDGE_FEATURE_DIM,
)

logger = logging.getLogger(__name__)


class ContractViolationError(Exception):
    """Raised when a Golden Invariant contract is violated."""
    pass


class EventTypeValidator:
    """Validates event type taxonomy compliance."""
    
    @staticmethod
    def validate_event_types(event_types: List[str]) -> bool:
        """Validate that event types match exactly the 6 canonical types."""
        if len(event_types) != 6:
            raise ContractViolationError(
                f"Event taxonomy violation: Expected exactly 6 event types, got {len(event_types)}"
            )
        
        expected_set = set(EVENT_TYPES)
        actual_set = set(event_types)
        
        if expected_set != actual_set:
            missing = expected_set - actual_set
            extra = actual_set - expected_set
            raise ContractViolationError(
                f"Event taxonomy violation: Missing {missing}, Extra {extra}"
            )
        
        return True
    
    @staticmethod
    def validate_event_data(events: List[Dict[str, Any]]) -> bool:
        """Validate event data contains only canonical event types."""
        for event in events:
            event_type = event.get("type") or event.get("kind") or event.get("event_type")
            if event_type and event_type not in EVENT_TYPES:
                raise ContractViolationError(
                    f"Invalid event type '{event_type}' not in canonical taxonomy: {EVENT_TYPES}"
                )
        return True


class EdgeIntentValidator:
    """Validates edge intent taxonomy compliance."""
    
    @staticmethod
    def validate_edge_intents(edge_intents: List[str]) -> bool:
        """Validate that edge intents match exactly the 4 canonical types."""
        if len(edge_intents) != 4:
            raise ContractViolationError(
                f"Edge intent violation: Expected exactly 4 edge intents, got {len(edge_intents)}"
            )
        
        expected_set = set(EDGE_INTENTS)
        actual_set = set(edge_intents)
        
        if expected_set != actual_set:
            missing = expected_set - actual_set
            extra = actual_set - expected_set
            raise ContractViolationError(
                f"Edge intent violation: Missing {missing}, Extra {extra}"
            )
        
        return True
    
    @staticmethod
    def validate_edge_data(edges: List[Dict[str, Any]]) -> bool:
        """Validate edge data contains only canonical edge intents."""
        for edge in edges:
            intent = edge.get("intent") or edge.get("etype") or edge.get("edge_intent")
            if intent and intent not in EDGE_INTENTS:
                raise ContractViolationError(
                    f"Invalid edge intent '{intent}' not in canonical taxonomy: {EDGE_INTENTS}"
                )
        return True


class FeatureDimensionValidator:
    """Validates feature dimension compliance."""
    
    @staticmethod
    def validate_node_features(features: Union[List[float], pd.DataFrame], htf_enabled: bool = False) -> bool:
        """Validate node feature dimensions."""
        expected_dim = NODE_FEATURE_DIM_HTF if htf_enabled else NODE_FEATURE_DIM_STANDARD
        
        if isinstance(features, pd.DataFrame):
            # Check feature columns
            feature_cols = [col for col in features.columns if col.startswith('f') and col[1:].isdigit()]
            actual_dim = len(feature_cols)
        elif isinstance(features, list):
            actual_dim = len(features)
        else:
            raise ContractViolationError(f"Invalid feature format: {type(features)}")
        
        if actual_dim > NODE_FEATURE_DIM_HTF:
            raise ContractViolationError(
                f"Node feature dimension violation: {actual_dim}D exceeds maximum {NODE_FEATURE_DIM_HTF}D"
            )
        
        if htf_enabled and actual_dim != NODE_FEATURE_DIM_HTF:
            raise ContractViolationError(
                f"HTF mode requires exactly {NODE_FEATURE_DIM_HTF}D features, got {actual_dim}D"
            )
        elif not htf_enabled and actual_dim != NODE_FEATURE_DIM_STANDARD:
            raise ContractViolationError(
                f"Standard mode requires exactly {NODE_FEATURE_DIM_STANDARD}D features, got {actual_dim}D"
            )
        
        return True
    
    @staticmethod
    def validate_edge_features(features: Union[List[float], pd.DataFrame]) -> bool:
        """Validate edge feature dimensions."""
        if isinstance(features, pd.DataFrame):
            # Check feature columns
            feature_cols = [col for col in features.columns if col.startswith('e') and col[1:].isdigit()]
            actual_dim = len(feature_cols)
        elif isinstance(features, list):
            actual_dim = len(features)
        else:
            raise ContractViolationError(f"Invalid feature format: {type(features)}")
        
        if actual_dim != EDGE_FEATURE_DIM:
            raise ContractViolationError(
                f"Edge feature dimension violation: Expected {EDGE_FEATURE_DIM}D, got {actual_dim}D"
            )
        
        return True


class HTFComplianceValidator:
    """Validates HTF (High Timeframe) compliance - last-closed only."""
    
    @staticmethod
    def validate_htf_usage(data: Dict[str, Any]) -> bool:
        """Validate HTF data uses only last-closed candle data."""
        # Check for intra-candle HTF usage patterns
        forbidden_patterns = [
            "intra_candle", "current_candle", "live_candle", 
            "real_time", "streaming", "tick_data"
        ]
        
        def check_dict_recursively(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check key names for forbidden patterns
                    key_lower = key.lower()
                    for pattern in forbidden_patterns:
                        if pattern in key_lower:
                            raise ContractViolationError(
                                f"HTF compliance violation: Forbidden pattern '{pattern}' in key '{current_path}'"
                            )
                    
                    # Check string values for forbidden patterns
                    if isinstance(value, str):
                        value_lower = value.lower()
                        for pattern in forbidden_patterns:
                            if pattern in value_lower:
                                raise ContractViolationError(
                                    f"HTF compliance violation: Forbidden pattern '{pattern}' in value at '{current_path}'"
                                )
                    
                    check_dict_recursively(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_dict_recursively(item, f"{path}[{i}]")
        
        check_dict_recursively(data)
        return True


class SessionIsolationValidator:
    """Validates session isolation - no cross-session edges."""
    
    @staticmethod
    def validate_session_isolation(graph: nx.Graph, session_id: str) -> bool:
        """Validate that graph contains no cross-session edges."""
        for u, v, data in graph.edges(data=True):
            # Check node session IDs
            u_session = graph.nodes[u].get('session_id', session_id)
            v_session = graph.nodes[v].get('session_id', session_id)
            
            if u_session != v_session:
                raise ContractViolationError(
                    f"Session isolation violation: Edge between sessions '{u_session}' and '{v_session}'"
                )
            
            if u_session != session_id or v_session != session_id:
                raise ContractViolationError(
                    f"Session isolation violation: Edge outside expected session '{session_id}'"
                )
        
        return True
    
    @staticmethod
    def validate_edge_list(edges: List[Dict[str, Any]], session_id: str) -> bool:
        """Validate edge list for session isolation."""
        for edge in edges:
            src_session = edge.get('src_session') or edge.get('session_id')
            dst_session = edge.get('dst_session') or edge.get('session_id')
            
            if src_session and src_session != session_id:
                raise ContractViolationError(
                    f"Session isolation violation: Source session '{src_session}' != '{session_id}'"
                )
            
            if dst_session and dst_session != session_id:
                raise ContractViolationError(
                    f"Session isolation violation: Target session '{dst_session}' != '{session_id}'"
                )
        
        return True


class SchemaValidator:
    """Validates parquet schema compliance."""
    
    @staticmethod
    def validate_nodes_schema(df: pd.DataFrame, htf_enabled: bool = False) -> bool:
        """Validate nodes.parquet schema."""
        required_cols = ['node_id', 't', 'kind']
        expected_dim = NODE_FEATURE_DIM_HTF if htf_enabled else NODE_FEATURE_DIM_STANDARD
        expected_feature_cols = [f'f{i}' for i in range(expected_dim)]
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ContractViolationError(f"Missing required node columns: {missing_cols}")
        
        # Check feature columns
        actual_feature_cols = [col for col in df.columns if col.startswith('f') and col[1:].isdigit()]
        if set(actual_feature_cols) != set(expected_feature_cols):
            raise ContractViolationError(
                f"Node feature columns mismatch: Expected {expected_feature_cols}, got {actual_feature_cols}"
            )
        
        return True
    
    @staticmethod
    def validate_edges_schema(df: pd.DataFrame) -> bool:
        """Validate edges.parquet schema."""
        required_cols = ['src', 'dst', 'etype', 'dt']
        expected_feature_cols = [f'e{i}' for i in range(EDGE_FEATURE_DIM)]
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ContractViolationError(f"Missing required edge columns: {missing_cols}")
        
        # Check feature columns
        actual_feature_cols = [col for col in df.columns if col.startswith('e') and col[1:].isdigit()]
        if set(actual_feature_cols) != set(expected_feature_cols):
            raise ContractViolationError(
                f"Edge feature columns mismatch: Expected {expected_feature_cols}, got {actual_feature_cols}"
            )
        
        return True


def validate_golden_invariants(
    event_types: Optional[List[str]] = None,
    edge_intents: Optional[List[str]] = None,
    node_features: Optional[Union[List[float], pd.DataFrame]] = None,
    edge_features: Optional[Union[List[float], pd.DataFrame]] = None,
    htf_data: Optional[Dict[str, Any]] = None,
    graph: Optional[nx.Graph] = None,
    session_id: Optional[str] = None,
    htf_enabled: bool = False,
) -> bool:
    """
    Comprehensive validation of all Golden Invariants.
    
    Args:
        event_types: List of event types to validate
        edge_intents: List of edge intents to validate
        node_features: Node features to validate
        edge_features: Edge features to validate
        htf_data: HTF data to validate for compliance
        graph: Graph to validate for session isolation
        session_id: Session ID for isolation validation
        htf_enabled: Whether HTF mode is enabled
    
    Returns:
        True if all validations pass
        
    Raises:
        ContractViolationError: If any Golden Invariant is violated
    """
    logger.info("Validating Golden Invariants...")
    
    if event_types is not None:
        EventTypeValidator.validate_event_types(event_types)
        logger.debug("✅ Event type taxonomy validated")
    
    if edge_intents is not None:
        EdgeIntentValidator.validate_edge_intents(edge_intents)
        logger.debug("✅ Edge intent taxonomy validated")
    
    if node_features is not None:
        FeatureDimensionValidator.validate_node_features(node_features, htf_enabled)
        logger.debug("✅ Node feature dimensions validated")
    
    if edge_features is not None:
        FeatureDimensionValidator.validate_edge_features(edge_features)
        logger.debug("✅ Edge feature dimensions validated")
    
    if htf_data is not None:
        HTFComplianceValidator.validate_htf_usage(htf_data)
        logger.debug("✅ HTF compliance validated")
    
    if graph is not None and session_id is not None:
        SessionIsolationValidator.validate_session_isolation(graph, session_id)
        logger.debug("✅ Session isolation validated")
    
    logger.info("All Golden Invariants validated successfully")
    return True
