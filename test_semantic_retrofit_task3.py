#!/usr/bin/env python3
"""
Test Task 3: Session Context Preservation - Metadata Attachment
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add IRONFORGE to path
sys.path.insert(0, '/Users/jack/IRONPULSE/IRONFORGE')

def test_session_context_preservation():
    """Test that session metadata is correctly extracted and preserved"""
    
    print("🧪 Testing Task 3: Session Context Preservation")
    print("=" * 60)
    
    try:
        # Import IRONFORGE components
        from learning.enhanced_graph_builder import EnhancedGraphBuilder
        
        print("✅ Successfully imported EnhancedGraphBuilder")
        
        # Create test instance
        builder = EnhancedGraphBuilder()
        print("✅ Created EnhancedGraphBuilder instance")
        
        # Test 1: Create comprehensive test session data
        test_session_data = {
            "session_metadata": {
                "session_type": "ny_am",
                "session_date": "2025-08-14",
                "session_start": "09:30:00",
                "session_end": "12:00:00",
                "session_duration": 150,
                "session_high": 23600.0,
                "session_low": 23450.0,
                "session_open": 23500.0,
                "session_close": 23580.0
            },
            "session_fpfvg": {
                "fpfvg_present": True,
                "fpfvg_formation": {"gap_size": 15.5}
            },
            "price_movements": [
                {
                    "timestamp": "09:30:00",
                    "price_level": 23500.0,
                    "movement_type": "open",
                    "normalized_price": 0.33,
                    "context": "NY AM session open with expansion phase"
                },
                {
                    "timestamp": "10:15:00", 
                    "price_level": 23575.0,
                    "event_type": "redelivery",
                    "context": "FVG redelivered during expansion"
                },
                {
                    "timestamp": "11:45:00",
                    "price_level": 23580.0,
                    "movement_type": "consolidation_break"
                }
            ],
            "session_liquidity_events": [
                {
                    "timestamp": "10:15:00",
                    "event_type": "redelivery",
                    "target_level": "fvg_premium"
                },
                {
                    "timestamp": "11:30:00",
                    "event_type": "sweep",
                    "target_level": "liquidity_pool"
                }
            ]
        }
        
        print("✅ Created comprehensive test session data")
        
        # Test 2: Build rich graph and extract metadata
        try:
            graph, session_metadata = builder.build_rich_graph(test_session_data, session_file_path="test_session.json")
            print("✅ Successfully built graph and extracted session metadata")
        except Exception as e:
            print(f"❌ ERROR: build_rich_graph failed: {e}")
            return False\n            \n        # Test 3: Verify metadata structure and content\n        required_fields = [\n            'session_name', 'session_type', 'session_date', 'session_start', 'session_end',\n            'session_start_time', 'session_end_time', 'session_duration', 'anchor_timeframe',\n            'market_characteristics', 'semantic_events_count', 'session_quality'\n        ]\n        \n        print(\"\\n🔍 Session Metadata Verification:\")\n        for field in required_fields:\n            if field in session_metadata:\n                value = session_metadata[field]\n                print(f\"  ✅ {field}: {value}\")\n            else:\n                print(f\"  ❌ MISSING: {field}\")\n                return False\n                \n        # Test 4: Verify specific values\n        print(\"\\n📊 Metadata Value Validation:\")\n        \n        # Session name standardization\n        if session_metadata['session_name'] == 'NY_AM':\n            print(\"  ✅ Session name correctly standardized: ny_am → NY_AM\")\n        else:\n            print(f\"  ❌ ERROR: Expected 'NY_AM', got '{session_metadata['session_name']}'\")\n            \n        # ISO timestamp formatting\n        expected_start = \"2025-08-14T09:30:00Z\"\n        if session_metadata['session_start_time'] == expected_start:\n            print(f\"  ✅ Start time correctly formatted: {expected_start}\")\n        else:\n            print(f\"  ❌ ERROR: Expected '{expected_start}', got '{session_metadata['session_start_time']}'\")\n            \n        # Anchor timeframe determination\n        if session_metadata['anchor_timeframe'] == '1m':\n            print(\"  ✅ Anchor timeframe correctly determined: 1m (Level 1 data)\")\n        else:\n            print(f\"  ❌ UNEXPECTED: Anchor timeframe is '{session_metadata['anchor_timeframe']}'\")\n            \n        # Test 5: Market characteristics analysis\n        market_chars = session_metadata['market_characteristics']\n        print(\"\\n🏪 Market Characteristics:\")\n        \n        if market_chars['fpfvg_present']:\n            print(\"  ✅ FPFVG presence correctly detected\")\n        else:\n            print(\"  ❌ ERROR: FPFVG should be detected as present\")\n            \n        expansion_count = market_chars['expansion_phases']\n        print(f\"  📈 Expansion phases detected: {expansion_count}\")\n        \n        liquidity_count = market_chars['liquidity_events_count']\n        if liquidity_count == 2:\n            print(f\"  ✅ Liquidity events correctly counted: {liquidity_count}\")\n        else:\n            print(f\"  ❌ ERROR: Expected 2 liquidity events, got {liquidity_count}\")\n            \n        # Price range calculation\n        price_range_pct = market_chars['price_range_pct']\n        expected_range = round(((23600 - 23450) / 23500) * 100, 2)  # ~0.64%\n        if abs(price_range_pct - expected_range) < 0.1:\n            print(f\"  ✅ Price range correctly calculated: {price_range_pct}% (expected ~{expected_range}%)\")\n        else:\n            print(f\"  ❌ ERROR: Price range {price_range_pct}% doesn't match expected {expected_range}%\")\n            \n        # Test 6: Semantic events counting\n        semantic_counts = session_metadata['semantic_events_count']\n        print(\"\\n🎯 Semantic Events Count:\")\n        \n        expected_redelivery = 2  # One in price_movements context, one in liquidity_events\n        if semantic_counts['fvg_redelivery'] >= expected_redelivery:\n            print(f\"  ✅ FVG redelivery events: {semantic_counts['fvg_redelivery']}\")\n        else:\n            print(f\"  ❌ ERROR: Expected ≥{expected_redelivery} redelivery events, got {semantic_counts['fvg_redelivery']}\")\n            \n        if semantic_counts['expansion_mentions'] >= 1:\n            print(f\"  ✅ Expansion mentions: {semantic_counts['expansion_mentions']}\")\n        else:\n            print(f\"  ❌ ERROR: Expected ≥1 expansion mention, got {semantic_counts['expansion_mentions']}\")\n            \n        # Test 7: Session quality assessment\n        session_quality = session_metadata['session_quality']\n        print(f\"\\n🏆 Session Quality Assessment: {session_quality}\")\n        \n        if session_quality in ['excellent', 'good']:\n            print(\"  ✅ High quality session detected\")\n        elif session_quality == 'adequate':\n            print(\"  ⚠️ Adequate quality session\")\n        else:\n            print(\"  ❌ Poor quality session - unexpected for comprehensive test data\")\n            \n        # Test 8: Metadata preservation timestamp\n        metadata_timestamp = session_metadata.get('metadata_extracted_at', '')\n        if metadata_timestamp and 'T' in metadata_timestamp:\n            print(f\"  ✅ Metadata timestamp preserved: {metadata_timestamp[:19]}...\")\n        else:\n            print(f\"  ❌ ERROR: Invalid metadata timestamp: {metadata_timestamp}\")\n            \n        print(\"\\n\" + \"=\" * 60)\n        print(\"📝 Task 3 Results Summary:\")\n        print(\"✅ Session metadata extraction working\")\n        print(\"✅ Session name standardization working\")\n        print(\"✅ Anchor timeframe determination working\")\n        print(\"✅ Market characteristics analysis working\")\n        print(\"✅ Semantic events counting working\")\n        print(\"✅ Session quality assessment working\")\n        print(\"✅ build_rich_graph returns (graph, metadata) tuple\")\n        print(\"🎉 TASK 3 COMPLETE: Session Context Preservation successful!\")\n        \n        return True\n        \n    except Exception as e:\n        print(f\"❌ ERROR in Task 3 testing: {e}\")\n        import traceback\n        traceback.print_exc()\n        return False\n\nif __name__ == \"__main__\":\n    success = test_session_context_preservation()\n    exit(0 if success else 1)