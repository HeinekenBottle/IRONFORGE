"""
Taxonomy Spine Smoke Tests
==========================

Smoke tests for Wave 7.x taxonomy standardization and discovery spine.
"""


def test_engine_imports():
    """Test that all three engine entrypoints are importable and callable"""
    from ironforge.semantic_engine import score_confluence
    from ironforge.temporal_engine import run_discovery
    from ironforge.validation_engine import validate_run
    
    assert callable(run_discovery)
    assert callable(score_confluence)
    assert callable(validate_run)

def test_taxonomy_contracts():
    """Test that taxonomy contracts are properly defined"""
    from ironforge.contracts.taxonomy_v1 import TAXONOMY_V1, EdgeType, EventType
    
    # Check event types
    assert len(EventType) == 6
    assert EventType.EXPANSION == 0
    assert EventType.REDELIVERY == 5
    
    # Check edge types
    assert len(EdgeType) == 4  
    assert EdgeType.TEMPORAL_NEXT == 0
    assert EdgeType.CONTEXT == 3
    
    # Check metadata
    assert TAXONOMY_V1.taxonomy_version == "v1.0"
    assert TAXONOMY_V1.total_event_types == 6
    assert TAXONOMY_V1.total_edge_types == 4

def test_cli_commands_exist():
    """Test that CLI commands are properly registered"""
    from ironforge.sdk.cli import main
    
    # Import should not raise
    assert callable(main)
    
def test_converter_taxonomy_mapping():
    """Test that converter maps taxonomy correctly"""  
    from ironforge.converters.htf_context_processor import create_default_htf_config
    from ironforge.converters.json_to_parquet import SessionConverter
    
    config = create_default_htf_config()
    converter = SessionConverter(config)
    
    # Test taxonomy mapping
    test_events = [
        {"event_type": "expansion", "source_type": "price_movement"},
        {"event_type": "consolidation", "source_type": "price_movement"},
        {"event_type": "liquidity_taken", "source_type": "liquidity_event"},
        {"event_type": "fvg_redelivery", "source_type": "gap_event"}
    ]
    
    expected_kinds = [0, 1, 4, 5]
    
    for event, expected in zip(test_events, expected_kinds, strict=False):
        kind = converter._get_node_kind(event)
        assert kind == expected, f"Event {event} should map to kind {expected}, got {kind}"

if __name__ == "__main__":
    test_engine_imports()
    test_taxonomy_contracts() 
    test_cli_commands_exist()
    test_converter_taxonomy_mapping()
    print("✅ All taxonomy spine smoke tests passed")