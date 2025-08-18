#!/usr/bin/env python3
"""
Phase 5: Simple TGAT Test
========================
Basic test to isolate TGAT discovery issues and validate pattern generation.
"""

import json
import sys
from pathlib import Path

import torch

# Add IRONFORGE to path
ironforge_root = Path(__file__).parent
sys.path.insert(0, str(ironforge_root))

def test_pattern_extraction():
    """Simple test of TGAT pattern extraction capability"""
    print("🧪 PHASE 5: SIMPLE TGAT PATTERN EXTRACTION TEST")
    print("=" * 50)
    
    try:
        # Import TGAT discovery
        from learning.tgat_discovery import IRONFORGEDiscovery
        print("✅ IRONFORGEDiscovery imported successfully")
        
        # Initialize TGAT discovery engine
        tgat_discovery = IRONFORGEDiscovery()
        print("✅ TGAT discovery engine initialized")
        
        # Create simple test tensors (simulating graph data)
        # 10 nodes, 37 features each (full TGAT expected dimension)
        X = torch.randn(10, 37)
        
        # Simple edge index (5 edges connecting nodes)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5]
        ], dtype=torch.long)
        
        print(f"✅ Test tensors created: X.shape={X.shape}, edge_index.shape={edge_index.shape}")
        
        # Test TGAT pattern discovery (correct method)
        # Create minimal edge times and metadata
        edge_times = torch.randn(edge_index.shape[1])  # Time stamps for edges
        metadata = {'session_name': 'test_session'}
        
        # Use correct learn_session method
        discovery_result = tgat_discovery.learn_session(X, edge_index, edge_times, metadata)
        embeddings = torch.tensor(discovery_result['embeddings'])
        print(f"✅ TGAT discovery successful: embeddings.shape={embeddings.shape}")
        
        # Create mock session data for pattern extraction
        mock_session_data = {
            'session_metadata': {
                'session_type': 'test_session',
                'session_date': '2025-08-14'
            },
            'energy_state': {
                'energy_density': 0.85,
                'total_accumulated': 95.5
            },
            'contamination_analysis': {
                'htf_contamination': {
                    'htf_carryover_strength': 0.75
                }
            },
            'session_liquidity_events': [
                {
                    'timestamp': '10:00:00',
                    'event_type': 'test_event',
                    'magnitude': 'medium'
                }
            ]
        }
        
        print("✅ Mock session data created for pattern testing")
        
        # Test individual pattern extraction methods
        patterns = []
        
        try:
            # Test 1: Temporal Structural Patterns
            temporal_patterns = tgat_discovery._extract_temporal_structural_patterns(
                X, embeddings, mock_session_data
            )
            patterns.extend(temporal_patterns or [])
            print(f"✅ Temporal structural patterns: {len(temporal_patterns or [])} found")
            
        except Exception as e:
            print(f"⚠️ Temporal patterns failed: {e}")
            
        try:
            # Test 2: HTF Confluence Patterns (with edge_attr)
            edge_attr = torch.randn(edge_index.shape[1], 17)  # Mock edge attributes with proper dimensions
            htf_patterns = tgat_discovery._extract_htf_confluence_patterns(
                X, embeddings, mock_session_data, edge_attr
            )
            patterns.extend(htf_patterns or [])
            print(f"✅ HTF confluence patterns: {len(htf_patterns or [])} found")
            
        except Exception as e:
            print(f"⚠️ HTF patterns failed: {e}")
            
        try:
            # Test 3: Scale Alignment Patterns (with correct parameters)
            scale_patterns = tgat_discovery._extract_scale_alignment_patterns(
                embeddings, edge_index, mock_session_data, edge_attr
            )
            patterns.extend(scale_patterns or [])
            print(f"✅ Scale alignment patterns: {len(scale_patterns or [])} found")
            
        except Exception as e:
            print(f"⚠️ Scale patterns failed: {e}")
            
        print()
        print(f"📊 TOTAL PATTERNS EXTRACTED: {len(patterns)}")
        
        if patterns:
            print("\n🔍 PATTERN ANALYSIS:")
            print("-" * 20)
            
            # Analyze pattern quality
            descriptions = [p.get('description', 'unknown') for p in patterns]
            unique_descriptions = len(set(descriptions))
            duplication_rate = ((len(patterns) - unique_descriptions) / len(patterns)) * 100.0
            
            print(f"Unique Descriptions: {unique_descriptions}/{len(patterns)}")
            print(f"Duplication Rate: {duplication_rate:.1f}%")
            
            # Show sample patterns
            print("\n📋 SAMPLE PATTERNS:")
            for i, pattern in enumerate(patterns[:3]):  # Show first 3
                print(f"{i+1}. Type: {pattern.get('type', 'unknown')}")
                print(f"   Description: {pattern.get('description', 'N/A')}")
                print(f"   Time Span: {pattern.get('time_span_hours', 'N/A')} hours")
                print()
                
            # Assessment
            if duplication_rate < 50:
                print("✅ SUCCESS: Pattern generation working with reasonable diversity")
            elif duplication_rate < 90:
                print("🔶 PARTIAL: Some pattern diversity, but needs improvement")
            else:
                print("❌ ISSUE: High duplication suggests template artifacts")
                
        else:
            print("❌ NO PATTERNS EXTRACTED - Pattern extraction failing")
            
        return {
            'tgat_working': True,
            'patterns_found': len(patterns),
            'unique_patterns': len(set(p.get('description', 'unknown') for p in patterns)),
            'duplication_rate': ((len(patterns) - len(set(p.get('description', 'unknown') for p in patterns))) / max(1, len(patterns))) * 100.0,
            'sample_patterns': patterns[:5]  # First 5 for inspection
        }
        
    except Exception as e:
        print(f"❌ TGAT test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'tgat_working': False,
            'error': str(e),
            'patterns_found': 0
        }

def test_enhanced_session_processing():
    """Test processing of actual enhanced session"""
    print("\n🏛️ ENHANCED SESSION PROCESSING TEST")
    print("=" * 40)
    
    try:
        # Load one enhanced session
        enhanced_session_path = Path(__file__).parent / 'enhanced_sessions' / 'enhanced_NY_PM_Lvl-1_2025_07_29.json'
        
        with open(enhanced_session_path, 'r') as f:
            session_data = json.load(f)
            
        print(f"✅ Loaded enhanced session: {enhanced_session_path.name}")
        
        # Validate key enhanced features
        energy_density = session_data.get('energy_state', {}).get('energy_density', 0.5)
        htf_carryover = session_data.get('contamination_analysis', {}).get('htf_contamination', {}).get('htf_carryover_strength', 0.3)
        liquidity_events = session_data.get('session_liquidity_events', [])
        
        print("Enhanced Features:")
        print(f"   Energy Density: {energy_density} ({'✅ Authentic' if energy_density != 0.5 else '❌ Default'})")
        print(f"   HTF Carryover: {htf_carryover} ({'✅ Authentic' if htf_carryover != 0.3 else '❌ Default'})")
        print(f"   Liquidity Events: {len(liquidity_events)} ({'✅ Rich' if len(liquidity_events) > 0 else '❌ Empty'})")
        
        # Check if we can extract price movements
        price_movements = session_data.get('price_movements', [])
        print(f"   Price Movements: {len(price_movements)} ({'✅ Available' if len(price_movements) > 0 else '❌ Missing'})")
        
        if len(price_movements) > 0:
            print(f"   Sample movement: {price_movements[0]}")
            
        return {
            'session_loaded': True,
            'features_enhanced': energy_density != 0.5 and htf_carryover != 0.3 and len(liquidity_events) > 0,
            'price_movements_available': len(price_movements) > 0,
            'energy_density': energy_density,
            'htf_carryover': htf_carryover,
            'liquidity_events_count': len(liquidity_events)
        }
        
    except Exception as e:
        print(f"❌ Session processing failed: {e}")
        return {
            'session_loaded': False,
            'error': str(e)
        }

def main():
    """Run Phase 5 simple tests"""
    print("🚀 PHASE 5: SIMPLE TGAT VALIDATION TESTS")
    print("=" * 45)
    print("Testing TGAT discovery capability in isolation")
    print()
    
    # Test 1: Basic TGAT pattern extraction
    tgat_test = test_pattern_extraction()
    
    # Test 2: Enhanced session processing
    session_test = test_enhanced_session_processing()
    
    # Summary
    print("\n🎯 TEST SUMMARY:")
    print("=" * 15)
    
    if tgat_test.get('tgat_working'):
        print(f"✅ TGAT Discovery: Working ({tgat_test['patterns_found']} patterns)")
        print(f"   Duplication Rate: {tgat_test['duplication_rate']:.1f}%")
        
        if tgat_test['duplication_rate'] < 50:
            print("   Assessment: ✅ Good pattern diversity")
        elif tgat_test['duplication_rate'] < 90:
            print("   Assessment: 🔶 Moderate pattern diversity")
        else:
            print("   Assessment: ❌ High duplication (template artifacts)")
    else:
        print(f"❌ TGAT Discovery: Failed ({tgat_test.get('error', 'Unknown error')})")
        
    if session_test.get('session_loaded'):
        print("✅ Enhanced Session: Loaded successfully")
        if session_test.get('features_enhanced'):
            print("   Features: ✅ Properly enhanced (authentic values)")
        else:
            print("   Features: ❌ Still contaminated (default values)")
    else:
        print("❌ Enhanced Session: Failed to load")
        
    # Overall assessment for Phase 5
    print("\n🏛️ PHASE 5 ASSESSMENT:")
    print("-" * 22)
    
    if tgat_test.get('tgat_working') and session_test.get('features_enhanced'):
        if tgat_test['duplication_rate'] < 50:
            print("✅ SUCCESS: TGAT + Enhanced features working well")
            print("   Recommendation: Proceed with full enhanced session validation")
        else:
            print("🔶 PARTIAL: TGAT working but high duplication")
            print("   Recommendation: Investigate pattern extraction logic")
    elif tgat_test.get('tgat_working'):
        print("🔶 PARTIAL: TGAT working but session enhancement may need work")  
        print("   Recommendation: Verify Phase 2 enhancement quality")
    else:
        print("❌ FAILED: Core TGAT discovery not working")
        print("   Recommendation: Fix TGAT discovery engine before proceeding")
        
    return {
        'tgat_test': tgat_test,
        'session_test': session_test
    }

if __name__ == "__main__":
    main()