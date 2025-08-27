#!/usr/bin/env python3
"""
🔧 Quick Test: Refined Detection Performance
==========================================

Test the refined sweep detector with a limited dataset to validate improvements.
"""

import logging
import sys
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

import json

from config import get_config

from ironforge.analysis.refined_sweep_detector import RefinedSweepDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def quick_test_refined_detection():
    """Quick test of refined detection with limited data"""

    print("🔧 QUICK TEST: Refined Sweep Detection")
    print("=" * 45)
    print("Testing detection improvements with 5 sessions...")
    print()

    try:
        # Initialize detector
        detector = RefinedSweepDetector()

        # Load limited dataset (5 sessions)
        config = get_config()
        enhanced_sessions_path = Path(config.get_enhanced_data_path())
        session_files = list(enhanced_sessions_path.glob("enhanced_rel_*.json"))[:5]

        enhanced_sessions = []
        for session_file in session_files:
            try:
                with open(session_file) as f:
                    session_data = json.load(f)
                    enhanced_sessions.append(session_data)
            except Exception as e:
                logger.warning(f"Failed to load {session_file}: {e}")

        sessions_data = {"enhanced_sessions": enhanced_sessions}

        print(f"📊 Loaded {len(enhanced_sessions)} sessions for testing")
        print()

        # Test refined detection
        logger.info("Testing refined sweep detection...")
        refined_sweeps = detector.detect_refined_sweeps(sessions_data)

        # Display results
        weekly_sweeps = refined_sweeps["weekly_sweeps"]
        daily_sweeps = refined_sweeps["daily_sweeps"]
        pm_executions = refined_sweeps["pm_executions"]

        print("✅ DETECTION RESULTS:")
        print(f"  Weekly Sweeps: {len(weekly_sweeps)}")
        print(f"  Daily Sweeps: {len(daily_sweeps)}")
        print(f"  PM Executions: {len(pm_executions)}")
        print(f"  Bridge Nodes Loaded: {len(detector.bridge_nodes)}")
        print()

        # Show sample detections
        if weekly_sweeps:
            print("🗓️  SAMPLE WEEKLY SWEEPS:")
            for i, sweep in enumerate(weekly_sweeps[:3], 1):
                print(f"  {i}. {sweep.session_id} @ {sweep.timestamp}")
                print(
                    f"     Price: {sweep.price_level:.1f}, Confidence: {sweep.detection_confidence:.3f}"
                )
                print(f"     ATR Penetration: {sweep.atr_penetration_pct:.1f}%")
                print(f"     Bridge Aligned: {sweep.bridge_node_aligned}")
                print()

        if daily_sweeps:
            print("📈 SAMPLE DAILY SWEEPS:")
            for i, sweep in enumerate(daily_sweeps[:3], 1):
                print(f"  {i}. {sweep.session_id} @ {sweep.timestamp}")
                print(
                    f"     Price: {sweep.price_level:.1f}, Confidence: {sweep.detection_confidence:.3f}"
                )
                print(f"     Type: {sweep.sweep_type}")
                print()

        if pm_executions:
            print("⏰ SAMPLE PM EXECUTIONS:")
            for i, execution in enumerate(pm_executions[:3], 1):
                print(f"  {i}. {execution.session_id} @ {execution.timestamp}")
                print(
                    f"     Price: {execution.price_level:.1f}, Category: {execution.pm_belt_category}"
                )
                print(f"     Confidence: {execution.detection_confidence:.3f}")
                print()

        # Assessment
        total_detections = len(weekly_sweeps) + len(daily_sweeps) + len(pm_executions)

        print("📊 DETECTION ASSESSMENT:")
        print(f"  Total Detections: {total_detections}")

        if len(weekly_sweeps) > 0:
            print("  ✅ Weekly Detection: WORKING")
        else:
            print("  ❌ Weekly Detection: NEEDS MORE REFINEMENT")

        if len(pm_executions) > 0:
            print("  ✅ PM Belt Detection: WORKING")
        else:
            print("  ❌ PM Belt Detection: NEEDS MORE REFINEMENT")

        if total_detections > 10:
            print("  🎯 OVERALL: GOOD DETECTION VOLUME")
        elif total_detections > 5:
            print("  🔧 OVERALL: MODERATE - CONTINUE REFINEMENTS")
        else:
            print("  ⚠️  OVERALL: LOW VOLUME - NEED THRESHOLD ADJUSTMENTS")

        print()
        print("🔧 Quick test complete! Refinements ready for full analysis.")

        return {
            "weekly_count": len(weekly_sweeps),
            "daily_count": len(daily_sweeps),
            "pm_count": len(pm_executions),
            "total_count": total_detections,
        }

    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    results = quick_test_refined_detection()
    if results:
        exit_code = 0 if results["total_count"] > 0 else 1
        sys.exit(exit_code)
    else:
        sys.exit(1)
