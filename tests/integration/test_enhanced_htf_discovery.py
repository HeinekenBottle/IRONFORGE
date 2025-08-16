#!/usr/bin/env python3
"""
Test Enhanced HTF Cross-Timeframe Discovery
Validates that the scale-edge-aware TGAT system finds cross-TF patterns
"""
import os
import glob
import json
from orchestrator import IRONFORGE

def test_enhanced_htf_discovery():
    print("🔍 Testing Enhanced HTF Cross-Timeframe Discovery")
    print("=" * 60)
    
    # Initialize IRONFORGE with enhanced mode
    forge = IRONFORGE(use_enhanced=True)
    
    # Get a small subset of HTF sessions for testing
    htf_data_path = "/Users/jack/IRONPULSE/data/sessions/htf_enhanced/"
    session_files = glob.glob(os.path.join(htf_data_path, "*_htf.json"))
    
    # Use just 3 sessions for initial testing
    test_sessions = session_files[:3]
    
    print(f"📁 Testing with {len(test_sessions)} HTF sessions:")
    for i, session_file in enumerate(test_sessions):
        session_name = os.path.basename(session_file)
        print(f"  {i+1}. {session_name}")
    print()
    
    try:
        print("🧠 Starting enhanced cross-TF discovery...")
        results = forge.process_sessions(test_sessions)
        
        print(f"\n📊 ENHANCED DISCOVERY RESULTS:")
        print(f"  Sessions processed: {results['sessions_processed']}")
        print(f"  Total patterns discovered: {len(results['patterns_discovered'])}")
        
        # Analyze pattern types
        pattern_types = {}
        for pattern in results['patterns_discovered']:
            ptype = pattern.get('type', 'unknown')
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
        
        print(f"\n🎯 PATTERN TYPE BREAKDOWN:")
        for ptype, count in sorted(pattern_types.items()):
            print(f"    {ptype}: {count}")
        
        # Look for new cross-TF pattern types
        cross_tf_patterns = [
            'scale_alignment', 'cross_tf_confluence', 'htf_cascade', 
            'scale_enhanced_cascade', 'multi_scale_liquidity'
        ]
        
        new_pattern_count = sum(pattern_types.get(ptype, 0) for ptype in cross_tf_patterns)
        old_pattern_count = len(results['patterns_discovered']) - new_pattern_count
        
        print(f"\n✨ CROSS-TIMEFRAME PATTERN ANALYSIS:")
        print(f"  Traditional patterns: {old_pattern_count}")
        print(f"  NEW Cross-TF patterns: {new_pattern_count}")
        print(f"  Cross-TF discovery rate: {new_pattern_count/len(results['patterns_discovered'])*100:.1f}%")
        
        # Show sample cross-TF patterns
        print(f"\n🔍 SAMPLE CROSS-TIMEFRAME PATTERNS:")
        for pattern in results['patterns_discovered'][:10]:
            if pattern['type'] in cross_tf_patterns:
                print(f"  • {pattern['type']}: {pattern}")
                break
        
        # Test success criteria
        success = True
        if new_pattern_count == 0:
            print(f"\n❌ FAILURE: No cross-timeframe patterns discovered!")
            success = False
        elif new_pattern_count < 5:
            print(f"\n⚠️ WARNING: Only {new_pattern_count} cross-TF patterns (expected >5)")
        else:
            print(f"\n✅ SUCCESS: {new_pattern_count} cross-timeframe patterns discovered!")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during enhanced discovery: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_htf_discovery()
    if success:
        print(f"\n🎉 Enhanced HTF discovery test PASSED")
        print(f"🚀 Ready for full 61-session test")
    else:
        print(f"\n💥 Enhanced HTF discovery test FAILED")
        print(f"🔧 Check implementation for issues")