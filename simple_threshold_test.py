#!/usr/bin/env python3
"""
🔧 Simple Threshold Test
=======================

Direct test to see if lowering thresholds gets detections flowing.
Focus on the core issue: getting ANY sweeps detected.
"""

import json
import sys
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

from config import get_config


def simple_detection_test():
    """Simple test with very low thresholds to get detections"""
    
    print("🔧 SIMPLE THRESHOLD TEST")
    print("=" * 25)
    print("Goal: Get ANY sweeps detected with very low thresholds")
    print()
    
    try:
        # Load one session for testing
        config = get_config()
        enhanced_sessions_path = Path(config.get_enhanced_data_path())
        session_files = list(enhanced_sessions_path.glob("enhanced_rel_*.json"))
        
        if not session_files:
            print("❌ No enhanced session files found")
            return False
        
        # Load first session
        with open(session_files[0], 'r') as f:
            session = json.load(f)
        
        session_name = session.get('session_name', 'unknown')
        print(f"📊 Testing session: {session_name}")
        print()
        
        # Test Weekly detection - very permissive
        weekly_candidates = []
        event_sources = ['semantic_events', 'session_liquidity_events', 'structural_events', 'price_movements']
        
        for source in event_sources:
            events = session.get(source, [])
            print(f"🔍 {source}: {len(events)} events")
            
            for event in events:
                event_text = str(event).lower()
                price = event.get('price_level', event.get('price', 0))
                
                # Very permissive weekly detection
                weekly_indicators = ['week', 'session', 'structural', 'high', 'low', 'break']
                if any(indicator in event_text for indicator in weekly_indicators):
                    if price and float(price) > 0:
                        weekly_candidates.append({
                            'source': source,
                            'event': event,
                            'price': float(price),
                            'text_sample': str(event)[:100]
                        })
        
        print(f"🗓️  Weekly candidates found: {len(weekly_candidates)}")
        if weekly_candidates:
            for i, candidate in enumerate(weekly_candidates[:3], 1):
                print(f"  {i}. Price: {candidate['price']:.1f} from {candidate['source']}")
                print(f"     Text: {candidate['text_sample']}")
                print()
        
        # Test PM belt detection - very permissive
        pm_candidates = []
        
        for source in event_sources:
            events = session.get(source, [])
            for event in events:
                timestamp = event.get('timestamp', '')
                price = event.get('price_level', event.get('price', 0))
                
                # Very permissive PM detection - any time with PM or 14:xx
                if ('14:' in timestamp or 'pm' in str(event).lower() or 
                    '13:' in timestamp or '15:' in timestamp):
                    if price and float(price) > 0:
                        pm_candidates.append({
                            'source': source,
                            'event': event,
                            'price': float(price),
                            'timestamp': timestamp
                        })
        
        print(f"⏰ PM candidates found: {len(pm_candidates)}")
        if pm_candidates:
            for i, candidate in enumerate(pm_candidates[:3], 1):
                print(f"  {i}. {candidate['timestamp']} - Price: {candidate['price']:.1f}")
                print(f"     Source: {candidate['source']}")
                print()
        
        # Test Daily detection - very permissive
        daily_candidates = []
        sweep_keywords = ['sweep', 'grab', 'hunt', 'raid', 'pierce', 'break', 'test', 'reject']
        
        for source in event_sources:
            events = session.get(source, [])
            for event in events:
                event_text = str(event).lower()
                price = event.get('price_level', event.get('price', 0))
                
                if any(keyword in event_text for keyword in sweep_keywords):
                    if price and float(price) > 0:
                        daily_candidates.append({
                            'source': source,
                            'event': event,
                            'price': float(price),
                            'keyword_matched': [kw for kw in sweep_keywords if kw in event_text]
                        })
        
        print(f"📈 Daily sweep candidates: {len(daily_candidates)}")
        if daily_candidates:
            for i, candidate in enumerate(daily_candidates[:3], 1):
                print(f"  {i}. Price: {candidate['price']:.1f}")
                print(f"     Keywords: {candidate['keyword_matched']}")
                print(f"     Source: {candidate['source']}")
                print()
        
        # Summary assessment
        total_candidates = len(weekly_candidates) + len(pm_candidates) + len(daily_candidates)
        
        print("📊 SIMPLE DETECTION SUMMARY:")
        print(f"  Weekly candidates: {len(weekly_candidates)}")
        print(f"  PM candidates: {len(pm_candidates)}")
        print(f"  Daily candidates: {len(daily_candidates)}")
        print(f"  Total candidates: {total_candidates}")
        print()
        
        if total_candidates == 0:
            print("❌ ISSUE: No candidates found even with very low thresholds")
            print("   → Check event data structure and content")
            return False
        elif total_candidates < 5:
            print("⚠️  LOW: Few candidates found")
            print("   → Data sparse or needs different detection approach")
            return True
        else:
            print("✅ SUCCESS: Candidates found!")
            print("   → Ready to implement proper cascade analysis")
            return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_event_structure():
    """Show the actual structure of events to understand data format"""
    
    print("\n🔍 EVENT STRUCTURE ANALYSIS")
    print("=" * 30)
    
    try:
        config = get_config()
        enhanced_sessions_path = Path(config.get_enhanced_data_path())
        session_files = list(enhanced_sessions_path.glob("enhanced_rel_*.json"))
        
        if not session_files:
            print("❌ No session files found")
            return
        
        with open(session_files[0], 'r') as f:
            session = json.load(f)
        
        print(f"📊 Session keys: {list(session.keys())}")
        print()
        
        # Show structure of each event source
        event_sources = ['semantic_events', 'session_liquidity_events', 'structural_events', 'price_movements']
        
        for source in event_sources:
            events = session.get(source, [])
            print(f"🔍 {source}: {len(events)} events")
            
            if events and len(events) > 0:
                # Show first event structure
                first_event = events[0]
                print(f"   Sample event keys: {list(first_event.keys()) if isinstance(first_event, dict) else 'Not a dict'}")
                print(f"   Sample event: {str(first_event)[:200]}...")
                print()
        
    except Exception as e:
        print(f"❌ Structure analysis failed: {e}")

if __name__ == "__main__":
    print("Starting simple threshold test...")
    
    # First show data structure
    show_event_structure()
    
    # Then test detection
    success = simple_detection_test()
    
    sys.exit(0 if success else 1)