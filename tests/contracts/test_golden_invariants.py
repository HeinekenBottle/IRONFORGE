"""
Test Golden Invariants Contract Enforcement
===========================================

Comprehensive tests for Golden Invariants:
- Event taxonomy: exactly 6 types
- Edge intents: exactly 4 types
- Feature dimensions: 51D max nodes, 20D edges
- HTF compliance: last-closed only
- Session isolation: no cross-session edges
"""

import pytest
import pandas as pd
import networkx as nx
import torch

from ironforge.contracts import (
    EventTypeValidator,
    EdgeIntentValidator,
    FeatureDimensionValidator,
    HTFComplianceValidator,
    SessionIsolationValidator,
    SchemaValidator,
    validate_golden_invariants,
    ContractViolationError,
)
from ironforge.constants import EVENT_TYPES, EDGE_INTENTS


class TestEventTypeValidator:
    """Test event type taxonomy validation."""
    
    def test_valid_event_types(self):
        """Test validation passes for correct event types."""
        assert EventTypeValidator.validate_event_types(EVENT_TYPES)
    
    def test_invalid_event_count(self):
        """Test validation fails for wrong number of event types."""
        with pytest.raises(ContractViolationError, match="Expected exactly 6 event types"):
            EventTypeValidator.validate_event_types(EVENT_TYPES[:5])
    
    def test_invalid_event_types(self):
        """Test validation fails for wrong event types."""
        invalid_types = EVENT_TYPES[:-1] + ["InvalidType"]
        with pytest.raises(ContractViolationError, match="Event taxonomy violation"):
            EventTypeValidator.validate_event_types(invalid_types)
    
    def test_event_data_validation(self):
        """Test event data validation."""
        valid_events = [
            {"type": "Expansion", "price": 100.0},
            {"type": "Consolidation", "price": 101.0},
        ]
        assert EventTypeValidator.validate_event_data(valid_events)
        
        invalid_events = [
            {"type": "InvalidType", "price": 100.0},
        ]
        with pytest.raises(ContractViolationError, match="Invalid event type"):
            EventTypeValidator.validate_event_data(invalid_events)


class TestEdgeIntentValidator:
    """Test edge intent taxonomy validation."""
    
    def test_valid_edge_intents(self):
        """Test validation passes for correct edge intents."""
        assert EdgeIntentValidator.validate_edge_intents(EDGE_INTENTS)
    
    def test_invalid_intent_count(self):
        """Test validation fails for wrong number of edge intents."""
        with pytest.raises(ContractViolationError, match="Expected exactly 4 edge intents"):
            EdgeIntentValidator.validate_edge_intents(EDGE_INTENTS[:3])
    
    def test_invalid_edge_intents(self):
        """Test validation fails for wrong edge intents."""
        invalid_intents = EDGE_INTENTS[:-1] + ["INVALID_INTENT"]
        with pytest.raises(ContractViolationError, match="Edge intent violation"):
            EdgeIntentValidator.validate_edge_intents(invalid_intents)
    
    def test_edge_data_validation(self):
        """Test edge data validation."""
        valid_edges = [
            {"intent": "TEMPORAL_NEXT", "src": 1, "dst": 2},
            {"intent": "MOVEMENT_TRANSITION", "src": 2, "dst": 3},
        ]
        assert EdgeIntentValidator.validate_edge_data(valid_edges)
        
        invalid_edges = [
            {"intent": "INVALID_INTENT", "src": 1, "dst": 2},
        ]
        with pytest.raises(ContractViolationError, match="Invalid edge intent"):
            EdgeIntentValidator.validate_edge_data(invalid_edges)


class TestFeatureDimensionValidator:
    """Test feature dimension validation."""
    
    def test_valid_node_features_standard(self):
        """Test validation passes for 45D node features."""
        features = [0.0] * 45
        assert FeatureDimensionValidator.validate_node_features(features, htf_enabled=False)
    
    def test_valid_node_features_htf(self):
        """Test validation passes for 51D node features."""
        features = [0.0] * 51
        assert FeatureDimensionValidator.validate_node_features(features, htf_enabled=True)
    
    def test_invalid_node_features_too_many(self):
        """Test validation fails for too many node features."""
        features = [0.0] * 52
        with pytest.raises(ContractViolationError, match="exceeds maximum"):
            FeatureDimensionValidator.validate_node_features(features)
    
    def test_invalid_node_features_wrong_count(self):
        """Test validation fails for wrong feature count."""
        features = [0.0] * 40
        with pytest.raises(ContractViolationError, match="Standard mode requires exactly"):
            FeatureDimensionValidator.validate_node_features(features, htf_enabled=False)
    
    def test_valid_edge_features(self):
        """Test validation passes for 20D edge features."""
        features = [0.0] * 20
        assert FeatureDimensionValidator.validate_edge_features(features)
    
    def test_invalid_edge_features(self):
        """Test validation fails for wrong edge feature count."""
        features = [0.0] * 19
        with pytest.raises(ContractViolationError, match="Expected 20D"):
            FeatureDimensionValidator.validate_edge_features(features)
    
    def test_dataframe_validation(self):
        """Test validation with pandas DataFrames."""
        # Valid nodes dataframe
        nodes_df = pd.DataFrame({
            'node_id': [1, 2, 3],
            **{f'f{i}': [0.0, 0.1, 0.2] for i in range(45)}
        })
        assert FeatureDimensionValidator.validate_node_features(nodes_df, htf_enabled=False)
        
        # Valid edges dataframe
        edges_df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3],
            **{f'e{i}': [0.0, 0.1] for i in range(20)}
        })
        assert FeatureDimensionValidator.validate_edge_features(edges_df)


class TestHTFComplianceValidator:
    """Test HTF compliance validation."""
    
    def test_valid_htf_data(self):
        """Test validation passes for compliant HTF data."""
        valid_data = {
            "htf_15m_close": 100.0,
            "htf_1h_close": 101.0,
            "last_closed_candle": True,
        }
        assert HTFComplianceValidator.validate_htf_usage(valid_data)
    
    def test_invalid_htf_data_intra_candle(self):
        """Test validation fails for intra-candle HTF usage."""
        invalid_data = {
            "htf_current_candle": 100.0,
            "intra_candle_data": True,
        }
        with pytest.raises(ContractViolationError, match="HTF compliance violation"):
            HTFComplianceValidator.validate_htf_usage(invalid_data)
    
    def test_invalid_htf_data_real_time(self):
        """Test validation fails for real-time HTF usage."""
        invalid_data = {
            "real_time_htf": 100.0,
            "streaming_data": "live",
        }
        with pytest.raises(ContractViolationError, match="HTF compliance violation"):
            HTFComplianceValidator.validate_htf_usage(invalid_data)


class TestSessionIsolationValidator:
    """Test session isolation validation."""
    
    def test_valid_session_isolation(self):
        """Test validation passes for isolated session."""
        graph = nx.Graph()
        graph.add_node(1, session_id="session_1")
        graph.add_node(2, session_id="session_1")
        graph.add_edge(1, 2)
        
        assert SessionIsolationValidator.validate_session_isolation(graph, "session_1")
    
    def test_invalid_session_isolation_cross_session(self):
        """Test validation fails for cross-session edges."""
        graph = nx.Graph()
        graph.add_node(1, session_id="session_1")
        graph.add_node(2, session_id="session_2")
        graph.add_edge(1, 2)
        
        with pytest.raises(ContractViolationError, match="Edge between sessions"):
            SessionIsolationValidator.validate_session_isolation(graph, "session_1")
    
    def test_edge_list_validation(self):
        """Test edge list validation for session isolation."""
        valid_edges = [
            {"src": 1, "dst": 2, "session_id": "session_1"},
            {"src": 2, "dst": 3, "session_id": "session_1"},
        ]
        assert SessionIsolationValidator.validate_edge_list(valid_edges, "session_1")
        
        invalid_edges = [
            {"src": 1, "dst": 2, "src_session": "session_1", "dst_session": "session_2"},
        ]
        with pytest.raises(ContractViolationError, match="Session isolation violation"):
            SessionIsolationValidator.validate_edge_list(invalid_edges, "session_1")


class TestSchemaValidator:
    """Test parquet schema validation."""
    
    def test_valid_nodes_schema(self):
        """Test validation passes for valid nodes schema."""
        nodes_df = pd.DataFrame({
            'node_id': [1, 2, 3],
            't': [1000, 2000, 3000],
            'kind': [0, 1, 2],
            **{f'f{i}': [0.0, 0.1, 0.2] for i in range(45)}
        })
        assert SchemaValidator.validate_nodes_schema(nodes_df, htf_enabled=False)
    
    def test_invalid_nodes_schema_missing_columns(self):
        """Test validation fails for missing required columns."""
        nodes_df = pd.DataFrame({
            'node_id': [1, 2, 3],
            # Missing 't' and 'kind'
            **{f'f{i}': [0.0, 0.1, 0.2] for i in range(45)}
        })
        with pytest.raises(ContractViolationError, match="Missing required node columns"):
            SchemaValidator.validate_nodes_schema(nodes_df)
    
    def test_valid_edges_schema(self):
        """Test validation passes for valid edges schema."""
        edges_df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3],
            'etype': [0, 1],
            'dt': [1000, 2000],
            **{f'e{i}': [0.0, 0.1] for i in range(20)}
        })
        assert SchemaValidator.validate_edges_schema(edges_df)
    
    def test_invalid_edges_schema_missing_columns(self):
        """Test validation fails for missing required columns."""
        edges_df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3],
            # Missing 'etype' and 'dt'
            **{f'e{i}': [0.0, 0.1] for i in range(20)}
        })
        with pytest.raises(ContractViolationError, match="Missing required edge columns"):
            SchemaValidator.validate_edges_schema(edges_df)


class TestGoldenInvariantsIntegration:
    """Test comprehensive Golden Invariants validation."""
    
    def test_comprehensive_validation_success(self):
        """Test all Golden Invariants validation passes."""
        # Create valid test data
        graph = nx.Graph()
        graph.add_node(1, session_id="session_1", feature=torch.zeros(45))
        graph.add_node(2, session_id="session_1", feature=torch.zeros(45))
        graph.add_edge(1, 2, feature=torch.zeros(20))
        
        assert validate_golden_invariants(
            event_types=EVENT_TYPES,
            edge_intents=EDGE_INTENTS,
            node_features=[0.0] * 45,
            edge_features=[0.0] * 20,
            htf_data={"htf_15m_close": 100.0},
            graph=graph,
            session_id="session_1",
            htf_enabled=False,
        )
    
    def test_comprehensive_validation_failure(self):
        """Test Golden Invariants validation fails appropriately."""
        with pytest.raises(ContractViolationError):
            validate_golden_invariants(
                event_types=EVENT_TYPES[:5],  # Wrong count
                edge_intents=EDGE_INTENTS,
                node_features=[0.0] * 45,
                edge_features=[0.0] * 20,
                htf_enabled=False,
            )
