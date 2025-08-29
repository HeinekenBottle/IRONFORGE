"""
IRONFORGE Contract Enforcement
==============================

Decorators and utilities for enforcing contracts at runtime.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union
import pandas as pd
import networkx as nx

from .validators import (
    validate_golden_invariants,
    SchemaValidator,
    ContractViolationError,
)

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def contract_guard(
    validate_events: bool = False,
    validate_edges: bool = False,
    validate_features: bool = False,
    validate_htf: bool = False,
    validate_isolation: bool = False,
    htf_enabled: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to enforce contracts on function calls.
    
    Args:
        validate_events: Validate event type taxonomy
        validate_edges: Validate edge intent taxonomy
        validate_features: Validate feature dimensions
        validate_htf: Validate HTF compliance
        validate_isolation: Validate session isolation
        htf_enabled: Whether HTF mode is enabled
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract validation data from arguments
            validation_data = _extract_validation_data(args, kwargs)
            
            try:
                # Perform requested validations
                if validate_events and 'event_types' in validation_data:
                    validate_golden_invariants(event_types=validation_data['event_types'])
                
                if validate_edges and 'edge_intents' in validation_data:
                    validate_golden_invariants(edge_intents=validation_data['edge_intents'])
                
                if validate_features:
                    if 'node_features' in validation_data:
                        validate_golden_invariants(
                            node_features=validation_data['node_features'],
                            htf_enabled=htf_enabled
                        )
                    if 'edge_features' in validation_data:
                        validate_golden_invariants(edge_features=validation_data['edge_features'])
                
                if validate_htf and 'htf_data' in validation_data:
                    validate_golden_invariants(htf_data=validation_data['htf_data'])
                
                if validate_isolation and 'graph' in validation_data and 'session_id' in validation_data:
                    validate_golden_invariants(
                        graph=validation_data['graph'],
                        session_id=validation_data['session_id']
                    )
                
                # Call original function
                return func(*args, **kwargs)
                
            except ContractViolationError as e:
                logger.error(f"Contract violation in {func.__name__}: {e}")
                raise
            
        return wrapper
    return decorator


def enforce_contracts(data: Dict[str, Any], htf_enabled: bool = False) -> bool:
    """
    Enforce all applicable contracts on a data dictionary.
    
    Args:
        data: Data dictionary to validate
        htf_enabled: Whether HTF mode is enabled
        
    Returns:
        True if all contracts pass
        
    Raises:
        ContractViolationError: If any contract is violated
    """
    logger.info("Enforcing contracts on data...")
    
    # Extract validation components
    event_types = data.get('event_types')
    edge_intents = data.get('edge_intents')
    node_features = data.get('node_features')
    edge_features = data.get('edge_features')
    htf_data = data.get('htf_data')
    graph = data.get('graph')
    session_id = data.get('session_id')
    
    # Run comprehensive validation
    validate_golden_invariants(
        event_types=event_types,
        edge_intents=edge_intents,
        node_features=node_features,
        edge_features=edge_features,
        htf_data=htf_data,
        graph=graph,
        session_id=session_id,
        htf_enabled=htf_enabled,
    )
    
    logger.info("All contracts enforced successfully")
    return True


def validate_session_data(session_data: Dict[str, Any], session_id: str, htf_enabled: bool = False) -> bool:
    """
    Validate session data for contract compliance.
    
    Args:
        session_data: Session data dictionary
        session_id: Session identifier
        htf_enabled: Whether HTF mode is enabled
        
    Returns:
        True if validation passes
        
    Raises:
        ContractViolationError: If validation fails
    """
    logger.info(f"Validating session data for {session_id}")
    
    # Validate events if present
    events = session_data.get('events', [])
    if events:
        from .validators import EventTypeValidator
        EventTypeValidator.validate_event_data(events)
    
    # Validate edges if present
    edges = session_data.get('edges', [])
    if edges:
        from .validators import EdgeIntentValidator, SessionIsolationValidator
        EdgeIntentValidator.validate_edge_data(edges)
        SessionIsolationValidator.validate_edge_list(edges, session_id)
    
    # Validate HTF data if present
    htf_data = {k: v for k, v in session_data.items() if 'htf' in k.lower()}
    if htf_data:
        from .validators import HTFComplianceValidator
        HTFComplianceValidator.validate_htf_usage(htf_data)
    
    logger.info(f"Session data validation passed for {session_id}")
    return True


def validate_graph_topology(graph: nx.Graph, session_id: str) -> bool:
    """
    Validate graph topology for contract compliance.
    
    Args:
        graph: NetworkX graph to validate
        session_id: Session identifier
        
    Returns:
        True if validation passes
        
    Raises:
        ContractViolationError: If validation fails
    """
    logger.info(f"Validating graph topology for {session_id}")
    
    from .validators import SessionIsolationValidator
    
    # Validate session isolation
    SessionIsolationValidator.validate_session_isolation(graph, session_id)
    
    # Validate node features if present
    for node_id, attrs in graph.nodes(data=True):
        if 'feature' in attrs:
            feature = attrs['feature']
            if hasattr(feature, 'shape'):
                # Tensor-like object
                feature_dim = feature.shape[0] if len(feature.shape) > 0 else 0
            elif isinstance(feature, (list, tuple)):
                feature_dim = len(feature)
            else:
                continue
            
            from .validators import FeatureDimensionValidator
            FeatureDimensionValidator.validate_node_features([0] * feature_dim)
    
    # Validate edge features if present
    for u, v, attrs in graph.edges(data=True):
        if 'feature' in attrs:
            feature = attrs['feature']
            if hasattr(feature, 'shape'):
                # Tensor-like object
                feature_dim = feature.shape[0] if len(feature.shape) > 0 else 0
            elif isinstance(feature, (list, tuple)):
                feature_dim = len(feature)
            else:
                continue
            
            from .validators import FeatureDimensionValidator
            FeatureDimensionValidator.validate_edge_features([0] * feature_dim)
    
    logger.info(f"Graph topology validation passed for {session_id}")
    return True


def validate_parquet_schemas(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, htf_enabled: bool = False) -> bool:
    """
    Validate parquet file schemas for contract compliance.
    
    Args:
        nodes_df: Nodes dataframe
        edges_df: Edges dataframe
        htf_enabled: Whether HTF mode is enabled
        
    Returns:
        True if validation passes
        
    Raises:
        ContractViolationError: If validation fails
    """
    logger.info("Validating parquet schemas")
    
    SchemaValidator.validate_nodes_schema(nodes_df, htf_enabled)
    SchemaValidator.validate_edges_schema(edges_df)
    
    logger.info("Parquet schema validation passed")
    return True


def _extract_validation_data(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract validation data from function arguments."""
    validation_data = {}
    
    # Look for common argument patterns
    for arg in args:
        if isinstance(arg, dict):
            # Check for session data patterns
            if 'events' in arg:
                events = arg['events']
                if events and isinstance(events[0], dict):
                    event_types = [e.get('type') or e.get('kind') for e in events if e.get('type') or e.get('kind')]
                    if event_types:
                        validation_data['event_types'] = list(set(event_types))
            
            if 'edges' in arg:
                edges = arg['edges']
                if edges and isinstance(edges[0], dict):
                    edge_intents = [e.get('intent') or e.get('etype') for e in edges if e.get('intent') or e.get('etype')]
                    if edge_intents:
                        validation_data['edge_intents'] = list(set(edge_intents))
            
            # Check for HTF data
            htf_keys = [k for k in arg.keys() if 'htf' in k.lower()]
            if htf_keys:
                validation_data['htf_data'] = {k: arg[k] for k in htf_keys}
        
        elif isinstance(arg, nx.Graph):
            validation_data['graph'] = arg
        
        elif isinstance(arg, pd.DataFrame):
            # Check if it's nodes or edges dataframe
            if 'node_id' in arg.columns:
                validation_data['node_features'] = arg
            elif 'src' in arg.columns and 'dst' in arg.columns:
                validation_data['edge_features'] = arg
    
    # Check kwargs
    for key, value in kwargs.items():
        if key == 'session_id':
            validation_data['session_id'] = value
        elif key == 'graph' and isinstance(value, nx.Graph):
            validation_data['graph'] = value
        elif key in ['htf_data', 'htf_context'] and isinstance(value, dict):
            validation_data['htf_data'] = value
    
    return validation_data
