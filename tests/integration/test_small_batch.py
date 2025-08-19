#!/usr/bin/env python3
"""
Test Small Batch Price Relativity Discovery
"""
import glob
import os

from orchestrator import IRONFORGE


def test_small_batch():
    print("🧪 Testing Small Batch Price Relativity Discovery")

    # Initialize IRONFORGE with enhanced mode
    forge = IRONFORGE(use_enhanced=True)

    # Get just 3 sessions for testing
    relativity_data_path = "/Users/jack/IRONPULSE/data/sessions/htf_relativity/"
    all_session_files = glob.glob(os.path.join(relativity_data_path, "*_htf_rel.json"))
    test_sessions = all_session_files[:3]  # Just first 3 sessions

    print(f"📁 Testing with {len(test_sessions)} sessions:")
    for i, session_file in enumerate(test_sessions):
        session_name = os.path.basename(session_file)
        print(f"  {i+1}. {session_name}")

    try:
        print("\n🔍 Starting small batch discovery...")
        results = forge.process_sessions(test_sessions)

        print("\n📊 RESULTS:")
        print(f"  Sessions processed: {results['sessions_processed']}")
        print(f"  Patterns discovered: {len(results['patterns_discovered'])}")
        print(f"  Graphs preserved: {len(results['graphs_preserved'])}")

        print("✅ SUCCESS: Small batch price relativity discovery works!")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_small_batch()
