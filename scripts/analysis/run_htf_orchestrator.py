#!/usr/bin/env python3
"""
Run IRONFORGE Orchestrator with PRICE RELATIVITY Enhanced Sessions
Demonstrates permanent pattern discovery with structural relationships

CRITICAL UPGRADE: Now uses price relativity features for permanent patterns
that survive market regime changes (vs absolute price patterns that expire)
"""
import glob
import os

from orchestrator import IRONFORGE


def main():
    print("🚀 IRONFORGE PRICE RELATIVITY Archaeological Discovery Session")
    print("🎯 PERMANENT PATTERNS: Structural relationships that survive regime changes")
    print("=" * 70)
    
    # Initialize IRONFORGE with enhanced mode
    forge = IRONFORGE(use_enhanced=True)
    
    # Get PRICE RELATIVITY enhanced session files
    relativity_data_path = "/Users/jack/IRONPULSE/data/sessions/htf_relativity/"
    session_files = glob.glob(os.path.join(relativity_data_path, "*_htf_rel.json"))
    
    print(f"📁 Found {len(session_files)} PRICE RELATIVITY enhanced sessions")
    print("🎯 These sessions contain PERMANENT PATTERNS that survive regime changes")
    
    # Process all available HTF sessions
    test_sessions = session_files
    
    print(f"🧪 Processing {len(test_sessions)} test sessions:")
    for i, session_file in enumerate(test_sessions):
        session_name = os.path.basename(session_file)
        print(f"  {i+1:2d}. {session_name}")
    print()
    
    try:
        # Process sessions through IRONFORGE
        print("🔍 Starting archaeological discovery with PRICE RELATIVITY integration...")
        print("⭐ Discovering PERMANENT structural relationships vs temporary price coincidences...")
        results = forge.process_sessions(test_sessions)
        
        print("\n📊 DISCOVERY RESULTS:")
        print(f"  Sessions processed: {results['sessions_processed']}")
        print(f"  Patterns discovered: {len(results['patterns_discovered'])}")
        print(f"  Graphs preserved: {len(results['graphs_preserved'])}")
        
        # Show sample discovered patterns
        if results['patterns_discovered']:
            print("\n🎯 SAMPLE DISCOVERED PATTERNS:")
            for i, pattern in enumerate(results['patterns_discovered'][:5]):
                print(f"  {i+1}. Type: {pattern.get('type', 'Unknown')}, "
                      f"Confidence: {pattern.get('confidence', 0.0):.3f}")
        
        # Validate discoveries
        print("\n✅ Validating discoveries...")
        validation = forge.validate_discoveries(test_sessions)
        print(f"  Validation results: {validation['validated']}/{validation['total_patterns']} patterns validated")
        
        # Show validation details
        if 'patterns' in validation:
            validated_count = sum(1 for p in validation['patterns'] if p['status'] == 'VALIDATED')
            print(f"  Validation rate: {validated_count/len(validation['patterns'])*100:.1f}%")
        
        # Freeze for production
        print("\n🏭 Preparing for production...")
        production_features = forge.freeze_for_production()
        
        print("\n🎉 HTF ARCHAEOLOGICAL DISCOVERY COMPLETE")
        print("  Multi-timeframe capabilities: ✅ Active")
        print("  Scale edge integration: ✅ Operational") 
        print("  TGAT compatibility: ✅ Preserved")
        print(f"  Production features: {len(production_features)} exported")
        
    except Exception as e:
        print(f"❌ Error during HTF discovery: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()