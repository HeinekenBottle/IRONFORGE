#!/usr/bin/env python3
"""
IRONFORGE SDK Test & Usage Examples
===================================
Practical tests and usage examples for the IRONFORGE Discovery SDK.
Demonstrates real workflows and validates functionality.

This script tests:
1. Basic SDK initialization and pattern discovery
2. Cross-session analysis capabilities  
3. Pattern intelligence workflows
4. Daily discovery workflows
5. Performance and reliability

Author: IRONFORGE Archaeological Discovery System
Date: August 14, 2025
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))


def test_sdk_initialization():
    """Test basic SDK initialization"""
    print("🔧 Testing SDK Initialization...")

    try:
        from ironforge_discovery_sdk import IRONFORGEDiscoverySDK

        start_time = time.time()
        sdk = IRONFORGEDiscoverySDK()
        init_time = time.time() - start_time

        print(f"✅ SDK initialized in {init_time:.2f}s")
        print(f"   Enhanced sessions path: {sdk.enhanced_sessions_path}")
        print(f"   Discovery cache path: {sdk.discovery_cache_path}")

        # Test session discovery
        session_files = list(sdk.enhanced_sessions_path.glob("enhanced_rel_*.json"))
        print(f"   Found {len(session_files)} enhanced sessions")

        if session_files:
            print(f"   Example session: {session_files[0].name}")
            return sdk, True
        else:
            print("❌ No enhanced sessions found")
            return sdk, False

    except Exception as e:
        print(f"❌ SDK initialization failed: {e}")
        return None, False


def test_pattern_discovery(sdk):
    """Test pattern discovery on sample session"""
    print("\n🔍 Testing Pattern Discovery...")

    try:
        # Find a test session
        session_files = list(sdk.enhanced_sessions_path.glob("enhanced_rel_NY_PM_*.json"))

        if not session_files:
            print("❌ No NY_PM sessions found for testing")
            return False

        test_session = session_files[0]
        print(f"   Testing session: {test_session.name}")

        start_time = time.time()
        patterns = sdk.discover_session_patterns(test_session)
        discovery_time = time.time() - start_time

        print(f"✅ Pattern discovery completed in {discovery_time:.2f}s")
        print(f"   Patterns found: {len(patterns)}")

        if patterns:
            print("   Sample patterns:")
            for i, pattern in enumerate(patterns[:3], 1):
                print(f"     {i}. {pattern.pattern_type}: {pattern.description}")
                print(
                    f"        Confidence: {pattern.confidence:.2f}, Position: {pattern.structural_position:.2f}"
                )

        return len(patterns) > 0

    except Exception as e:
        print(f"❌ Pattern discovery failed: {e}")
        return False


def test_cross_session_analysis(sdk):
    """Test cross-session pattern analysis"""
    print("\n🔗 Testing Cross-Session Analysis...")

    try:
        # Ensure we have some patterns in the database
        if not sdk.pattern_database:
            print("   Loading pattern database...")
            session_files = list(sdk.enhanced_sessions_path.glob("enhanced_rel_*.json"))[
                :5
            ]  # Test with 5 sessions

            for session_file in session_files:
                sdk.discover_session_patterns(session_file)

        print(f"   Pattern database size: {len(sdk.pattern_database)}")

        if len(sdk.pattern_database) < 2:
            print("❌ Insufficient patterns for cross-session analysis")
            return False

        # Test cross-session link finding
        start_time = time.time()
        links = sdk.find_cross_session_links(min_similarity=0.5)
        analysis_time = time.time() - start_time

        print(f"✅ Cross-session analysis completed in {analysis_time:.2f}s")
        print(f"   Cross-session links found: {len(links)}")

        if links:
            print("   Sample links:")
            for i, link in enumerate(links[:3], 1):
                print(f"     {i}. {link.link_type}: {link.description}")
                print(
                    f"        Strength: {link.link_strength:.2f}, Distance: {link.temporal_distance_days:.1f} days"
                )

        return True

    except Exception as e:
        print(f"❌ Cross-session analysis failed: {e}")
        return False


def test_pattern_intelligence():
    """Test pattern intelligence engine"""
    print("\n🧠 Testing Pattern Intelligence...")

    try:
        from ironforge_discovery_sdk import IRONFORGEDiscoverySDK
        from pattern_intelligence import PatternIntelligenceEngine

        sdk = IRONFORGEDiscoverySDK()
        intel_engine = PatternIntelligenceEngine(sdk)

        # Load some patterns for testing
        session_files = list(sdk.enhanced_sessions_path.glob("enhanced_rel_*.json"))[:3]
        for session_file in session_files:
            sdk.discover_session_patterns(session_file)

        if not sdk.pattern_database:
            print("❌ No patterns available for intelligence testing")
            return False

        print(f"   Analyzing {len(sdk.pattern_database)} patterns")

        # Test trend analysis
        start_time = time.time()
        trends = intel_engine.analyze_pattern_trends(days_lookback=14)
        trend_time = time.time() - start_time

        print(f"✅ Trend analysis completed in {trend_time:.2f}s")
        print(f"   Pattern trends identified: {len(trends)}")

        # Test regime identification
        start_time = time.time()
        regimes = intel_engine.identify_market_regimes(min_sessions=2)
        regime_time = time.time() - start_time

        print(f"✅ Regime identification completed in {regime_time:.2f}s")
        print(f"   Market regimes identified: {len(regimes)}")

        # Test intelligence summary
        summary = intel_engine.get_intelligence_summary()
        print(f"   Intelligence summary generated: {len(summary)} metrics")

        return True

    except Exception as e:
        print(f"❌ Pattern intelligence testing failed: {e}")
        return False


def test_daily_workflows():
    """Test daily discovery workflows"""
    print("\n📋 Testing Daily Workflows...")

    try:
        from daily_discovery_workflows import DailyDiscoveryWorkflows

        workflows = DailyDiscoveryWorkflows()

        # Test morning analysis (with limited lookback for speed)
        print("   Testing morning market analysis...")
        start_time = time.time()
        morning_analysis = workflows.morning_market_analysis(days_lookback=3)
        morning_time = time.time() - start_time

        print(f"✅ Morning analysis completed in {morning_time:.2f}s")
        print(f"   Confidence level: {morning_analysis.confidence_level}")
        print(f"   Regime status: {morning_analysis.regime_status}")
        print(f"   Trading insights: {len(morning_analysis.trading_insights)}")

        # Test session hunting
        print("   Testing session pattern hunting...")
        start_time = time.time()
        session_result = workflows.hunt_session_patterns("NY_PM")
        hunt_time = time.time() - start_time

        print(f"✅ Session hunting completed in {hunt_time:.2f}s")
        print(f"   Patterns found: {session_result.strength_indicators.get('pattern_count', 0)}")
        print(f"   Immediate insights: {len(session_result.immediate_insights)}")

        return True

    except Exception as e:
        print(f"❌ Daily workflows testing failed: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions for quick access"""
    print("\n⚡ Testing Convenience Functions...")

    try:
        # Test quick discovery
        from ironforge_discovery_sdk import analyze_session_patterns

        print("   Testing quick session analysis...")
        start_time = time.time()
        patterns = analyze_session_patterns("NY_PM")
        quick_time = time.time() - start_time

        print(f"✅ Quick session analysis completed in {quick_time:.2f}s")
        print(f"   Patterns found: {len(patterns)}")

        # Test morning prep convenience function
        from daily_discovery_workflows import morning_prep

        print("   Testing morning prep convenience function...")
        start_time = time.time()
        prep_result = morning_prep(days_back=2)  # Limited for speed
        time.time() - start_time

        print(f"✅ Morning prep completed in {quick_time:.2f}s")
        print(f"   Confidence: {prep_result.confidence_level}")

        return True

    except Exception as e:
        print(f"❌ Convenience functions testing failed: {e}")
        return False


def demonstrate_practical_usage():
    """Demonstrate practical daily usage scenarios"""
    print("\n💡 Practical Usage Demonstration")
    print("=" * 50)

    print("\n🌅 Scenario 1: Morning Market Preparation")
    print("-" * 40)

    try:
        from daily_discovery_workflows import morning_prep

        print("Running: morning_prep(days_back=5)")
        analysis = morning_prep(days_back=5)

        print(
            f"Result: {analysis.confidence_level} confidence, {len(analysis.trading_insights)} insights"
        )

    except Exception as e:
        print(f"❌ Morning prep demo failed: {e}")

    print("\n🎯 Scenario 2: Session Pattern Hunting")
    print("-" * 40)

    try:
        from daily_discovery_workflows import hunt_patterns

        print("Running: hunt_patterns('NY_PM')")
        result = hunt_patterns("NY_PM")

        pattern_count = result.strength_indicators.get("pattern_count", 0)
        avg_confidence = result.strength_indicators.get("avg_confidence", 0)
        print(f"Result: {pattern_count} patterns found, {avg_confidence:.2f} avg confidence")

    except Exception as e:
        print(f"❌ Pattern hunting demo failed: {e}")

    print("\n🔍 Scenario 3: Find Similar Patterns")
    print("-" * 40)

    try:
        from pattern_intelligence import find_similar_patterns

        print("Running: find_similar_patterns('NY_PM')")
        matches = find_similar_patterns("NY_PM")

        print(f"Result: {len(matches)} similar patterns found")

    except Exception as e:
        print(f"❌ Similar patterns demo failed: {e}")


def run_performance_test():
    """Run performance benchmarks"""
    print("\n⚡ Performance Benchmarks")
    print("=" * 30)

    try:
        from ironforge_discovery_sdk import IRONFORGEDiscoverySDK

        sdk = IRONFORGEDiscoverySDK()

        # Test initialization time
        start_time = time.time()
        sdk = IRONFORGEDiscoverySDK()
        init_time = time.time() - start_time

        # Test single session discovery time
        session_files = list(sdk.enhanced_sessions_path.glob("enhanced_rel_*.json"))
        if session_files:
            start_time = time.time()
            patterns = sdk.discover_session_patterns(session_files[0])
            single_discovery_time = time.time() - start_time

            print("📊 Performance Results:")
            print(f"   SDK Initialization: {init_time:.2f}s")
            print(f"   Single Session Discovery: {single_discovery_time:.2f}s")
            print(f"   Patterns per Second: {len(patterns)/single_discovery_time:.1f}")

            # Estimate full discovery time
            estimated_full_time = single_discovery_time * len(session_files)
            print(
                f"   Estimated Full Discovery ({len(session_files)} sessions): {estimated_full_time:.1f}s"
            )

            if estimated_full_time < 300:  # Under 5 minutes
                print("✅ Performance: Excellent for daily use")
            elif estimated_full_time < 600:  # Under 10 minutes
                print("✅ Performance: Good for daily use")
            else:
                print("⚠️ Performance: May be slow for daily use")

    except Exception as e:
        print(f"❌ Performance test failed: {e}")


def main():
    """Run complete SDK test suite"""
    print("🏛️ IRONFORGE SDK Test Suite")
    print("=" * 50)
    print(f"Test started: {datetime.now().isoformat()}")

    test_results = []

    # Core functionality tests
    sdk, init_success = test_sdk_initialization()
    test_results.append(("SDK Initialization", init_success))

    if init_success and sdk:
        discovery_success = test_pattern_discovery(sdk)
        test_results.append(("Pattern Discovery", discovery_success))

        cross_session_success = test_cross_session_analysis(sdk)
        test_results.append(("Cross-Session Analysis", cross_session_success))

    # Intelligence and workflow tests
    intel_success = test_pattern_intelligence()
    test_results.append(("Pattern Intelligence", intel_success))

    workflow_success = test_daily_workflows()
    test_results.append(("Daily Workflows", workflow_success))

    convenience_success = test_convenience_functions()
    test_results.append(("Convenience Functions", convenience_success))

    # Performance test
    run_performance_test()

    # Practical demonstration
    demonstrate_practical_usage()

    # Final results
    print("\n" + "=" * 50)
    print("🏆 TEST RESULTS SUMMARY")
    print("=" * 50)

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed_tests += 1

    success_rate = (passed_tests / total_tests) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")

    if success_rate >= 80:
        print("🎉 SDK is ready for production use!")
    elif success_rate >= 60:
        print("⚠️ SDK has minor issues but is usable")
    else:
        print("❌ SDK requires fixes before use")

    print(f"\nTest completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
