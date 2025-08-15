"""
Manual IRONFORGE runner with Unicode fix
"""
import sys
sys.path.append('/Users/jack/IRONPULSE')

from IRONFORGE.unicode_fix import load_clean_sessions
from IRONFORGE.orchestrator import IRONFORGE
import json

def run_full_discovery():
    print("Loading 66 sanitized sessions...")
    sessions = load_clean_sessions()
    
    print("\nInitializing IRONFORGE...")
    forge = IRONFORGE()
    
    # Process all sessions
    print("\nRunning archaeological discovery on 66 sessions...")
    session_files = [filepath for filepath, _ in sessions]
    
    results = forge.process_sessions(session_files)
    
    print(f"\n🎯 Discovery Results:")
    print(f"Sessions: {results['sessions_processed']}")
    print(f"Patterns: {len(results['patterns_discovered'])}")
    
    # Validate discoveries
    print("\nValidating patterns...")
    validation = forge.validate_discoveries(session_files)
    
    print(f"\n✅ Validation Results:")
    print(f"Total patterns: {validation['total_patterns']}")
    print(f"Validated: {validation['validated']}")
    
    # Save results
    with open('discovery_results_66.json', 'w') as f:
        json.dump({
            'discovery': results,
            'validation': validation
        }, f, indent=2)
    
    print("\nResults saved to discovery_results_66.json")
    
    return results, validation

if __name__ == "__main__":
    run_full_discovery()
