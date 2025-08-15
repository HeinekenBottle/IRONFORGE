#!/usr/bin/env python3
"""
Concrete Analysis: What Do These Sub-Patterns Actually Mean?
===========================================================
Move beyond vague descriptions to specific market mechanics
"""

import json
import pickle
import numpy as np
import glob
from pathlib import Path
from collections import defaultdict, Counter

def analyze_actual_events_by_subpattern():
    """Analyze what actual semantic events occur in each sub-pattern"""
    
    print("🔍 CONCRETE EVENT ANALYSIS BY SUB-PATTERN")
    print("=" * 60)
    
    # Load preserved graphs to get actual semantic events
    graph_files = glob.glob("/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/full_graph_store/*2025_08*.pkl")[:10]
    
    subpattern_events = {0: [], 1: [], 2: []}
    subpattern_timing = {0: [], 1: [], 2: []}
    subpattern_sessions = {0: [], 1: [], 2: []}
    
    print(f"📊 Analyzing {len(graph_files)} recent sessions...")
    
    for graph_file in graph_files:
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            session_name = Path(graph_file).stem.replace('_graph_', '_').split('_202')[0]
            rich_features = graph_data.get('rich_node_features', [])
            
            for feature in rich_features:
                # Extract actual semantic events
                events_found = []
                
                if hasattr(feature, 'expansion_phase_flag') and feature.expansion_phase_flag > 0.0:
                    events_found.append('expansion_phase')
                if hasattr(feature, 'consolidation_flag') and feature.consolidation_flag > 0.0:
                    events_found.append('consolidation')
                if hasattr(feature, 'liq_sweep_flag') and feature.liq_sweep_flag > 0.0:
                    events_found.append('liq_sweep')
                if hasattr(feature, 'fvg_redelivery_flag') and feature.fvg_redelivery_flag > 0.0:
                    events_found.append('fvg_redelivery')
                if hasattr(feature, 'reversal_flag') and feature.reversal_flag > 0.0:
                    events_found.append('reversal')
                if hasattr(feature, 'retracement_flag') and feature.retracement_flag > 0.0:
                    events_found.append('retracement')
                
                if events_found:
                    # Determine which sub-pattern this belongs to based on features
                    phase_open = getattr(feature, 'phase_open', 0.0)
                    normalized_time = getattr(feature, 'normalized_time', 0.5)
                    session_position = getattr(feature, 'session_position', 0.5)
                    time_minutes = getattr(feature, 'time_minutes', 0.0)
                    weekend_proximity = getattr(feature, 'weekend_proximity', 0.0)
                    
                    # Classify into sub-patterns based on discovered characteristics
                    if phase_open < 0.3 and (normalized_time > 0.6 or session_position > 0.6):
                        # Sub-pattern 0: Late session (anti-opening, high position)
                        subpattern = 0
                    elif phase_open > 0.7 and (normalized_time < 0.4 and session_position < 0.4):
                        # Sub-pattern 1: Early session (opening phase, low position) 
                        subpattern = 1
                    elif weekend_proximity < 0.3:
                        # Sub-pattern 2: Mid-week patterns
                        subpattern = 2
                    else:
                        continue  # Don't classify unclear cases
                    
                    subpattern_events[subpattern].extend(events_found)
                    subpattern_timing[subpattern].append({
                        'time_minutes': time_minutes,
                        'normalized_time': normalized_time,
                        'session_position': session_position,
                        'phase_open': phase_open
                    })
                    subpattern_sessions[subpattern].append(session_name)
        
        except Exception as e:
            print(f"  ⚠️ Error processing {Path(graph_file).stem}: {e}")
    
    # Analyze results
    for pattern_id in [0, 1, 2]:
        events = subpattern_events[pattern_id]
        timing = subpattern_timing[pattern_id]
        sessions = subpattern_sessions[pattern_id]
        
        if not events:
            continue
            
        print(f"\n🎯 SUB-PATTERN {pattern_id} CONCRETE ANALYSIS:")
        print("-" * 50)
        
        # Event type distribution
        event_counts = Counter(events)
        total_events = len(events)
        
        print(f"📊 Actual Events ({total_events} total):")
        for event_type, count in event_counts.most_common():
            pct = (count / total_events) * 100
            print(f"   {event_type}: {count} events ({pct:.1f}%)")
        
        # Timing analysis
        if timing:
            avg_time = np.mean([t['time_minutes'] for t in timing if t['time_minutes'] > 0])
            avg_normalized = np.mean([t['normalized_time'] for t in timing])
            avg_position = np.mean([t['session_position'] for t in timing])
            avg_phase_open = np.mean([t['phase_open'] for t in timing])
            
            print(f"⏰ Actual Timing:")
            print(f"   Average time: {avg_time:.1f} minutes into session")
            print(f"   Normalized time: {avg_normalized:.2f} (0=start, 1=end)")
            print(f"   Session position: {avg_position:.2f}")
            print(f"   Opening phase: {avg_phase_open:.2f} (1=opening 20%)")
        
        # Session analysis
        session_counts = Counter(sessions)
        print(f"📍 Top Sessions:")
        for session, count in session_counts.most_common(3):
            print(f"   {session}: {count} events")

def analyze_market_mechanics():
    """Analyze what these patterns mean in terms of actual market mechanics"""
    
    print(f"\n🔧 CONCRETE MARKET MECHANICS ANALYSIS")
    print("=" * 60)
    
    # Load actual discovered patterns to see their characteristics
    patterns_file = "/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/discovered_patterns.json"
    
    try:
        with open(patterns_file, 'r') as f:
            patterns = json.load(f)
        
        print(f"📊 Analyzing {len(patterns)} TGAT patterns for concrete mechanics...")
        
        # Group patterns by type and analyze features
        pattern_mechanics = defaultdict(list)
        
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            features = pattern.get('features', {})
            
            # Extract concrete characteristics
            pattern_mechanics[pattern_type].append({
                'confidence': pattern.get('confidence_score', 0.0),
                'permanence': pattern.get('permanence_score', 0.0),
                'session': pattern.get('session', 'unknown'),
                'description': pattern.get('description', 'no_description'),
                'time_span': pattern.get('time_span_hours', 0),
                'features': features
            })
        
        # Analyze each pattern type
        for ptype, pattern_list in pattern_mechanics.items():
            if not pattern_list:
                continue
                
            print(f"\n🏗️ {ptype.upper()} MECHANICS:")
            print("-" * 40)
            
            # Confidence and permanence analysis
            confidences = [p['confidence'] for p in pattern_list if p['confidence'] > 0]
            permanences = [p['permanence'] for p in pattern_list if p['permanence'] > 0]
            time_spans = [p['time_span'] for p in pattern_list if p['time_span'] > 0]
            
            if confidences:
                print(f"📈 Quality Metrics:")
                print(f"   Avg confidence: {np.mean(confidences):.3f}")
                print(f"   Avg permanence: {np.mean(permanences):.3f}")
                if time_spans:
                    print(f"   Avg duration: {np.mean(time_spans):.1f} hours")
            
            # Session distribution
            session_dist = Counter(p['session'] for p in pattern_list)
            print(f"📍 Session Distribution:")
            for session, count in session_dist.most_common(3):
                pct = (count / len(pattern_list)) * 100
                print(f"   {session}: {count} patterns ({pct:.1f}%)")
            
            # Description analysis
            descriptions = [p['description'] for p in pattern_list if p['description'] != 'no_description']
            if descriptions:
                desc_counts = Counter(descriptions)
                print(f"🔍 Common Descriptions:")
                for desc, count in desc_counts.most_common(3):
                    print(f"   '{desc}': {count} patterns")
    
    except Exception as e:
        print(f"❌ Error analyzing pattern mechanics: {e}")

def analyze_specific_examples():
    """Look at specific examples of each sub-pattern"""
    
    print(f"\n📋 SPECIFIC PATTERN EXAMPLES")
    print("=" * 60)
    
    # Load patterns and find specific examples
    patterns_file = "/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/discovered_patterns.json"
    
    try:
        with open(patterns_file, 'r') as f:
            patterns = json.load(f)
        
        # Group by pattern type and show concrete examples
        pattern_types = {}
        for pattern in patterns:
            ptype = pattern.get('type', 'unknown')
            if ptype not in pattern_types:
                pattern_types[ptype] = []
            pattern_types[ptype].append(pattern)
        
        for ptype, pattern_list in pattern_types.items():
            print(f"\n🔍 {ptype.upper()} EXAMPLES:")
            print("-" * 40)
            
            # Show top 3 highest confidence examples
            sorted_patterns = sorted(pattern_list, key=lambda x: x.get('confidence_score', 0), reverse=True)
            
            for i, pattern in enumerate(sorted_patterns[:3]):
                confidence = pattern.get('confidence_score', 0)
                session = pattern.get('session', 'unknown')
                description = pattern.get('description', 'no_description')
                time_span = pattern.get('time_span_hours', 0)
                
                print(f"   Example {i+1}:")
                print(f"     Session: {session}")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Duration: {time_span:.1f}h")
                print(f"     Description: {description}")
                print()
    
    except Exception as e:
        print(f"❌ Error loading patterns: {e}")

def decode_price_levels():
    """Analyze what specific price levels and movements these patterns represent"""
    
    print(f"\n💰 PRICE LEVEL ANALYSIS")
    print("=" * 60)
    
    # This would require access to the actual price data in the patterns
    # For now, let's analyze what we can infer from the preserved graphs
    
    graph_files = glob.glob("/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/full_graph_store/*2025_08*.pkl")[:5]
    
    price_analysis = []
    
    for graph_file in graph_files:
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            rich_features = graph_data.get('rich_node_features', [])
            session_name = Path(graph_file).stem.replace('_graph_', '_').split('_202')[0]
            
            for feature in rich_features:
                if hasattr(feature, 'normalized_price') and hasattr(feature, 'pct_from_open'):
                    price_info = {
                        'session': session_name,
                        'normalized_price': getattr(feature, 'normalized_price', 0),
                        'pct_from_open': getattr(feature, 'pct_from_open', 0),
                        'pct_from_high': getattr(feature, 'pct_from_high', 0),
                        'pct_from_low': getattr(feature, 'pct_from_low', 0),
                        'has_expansion': getattr(feature, 'expansion_phase_flag', 0) > 0,
                        'has_consolidation': getattr(feature, 'consolidation_flag', 0) > 0,
                        'has_liq_sweep': getattr(feature, 'liq_sweep_flag', 0) > 0
                    }
                    price_analysis.append(price_info)
        
        except Exception as e:
            continue
    
    if price_analysis:
        print(f"📊 Analyzed {len(price_analysis)} price points across events")
        
        # Expansion phase price characteristics
        expansion_prices = [p for p in price_analysis if p['has_expansion']]
        if expansion_prices:
            avg_norm_price = np.mean([p['normalized_price'] for p in expansion_prices])
            avg_pct_open = np.mean([p['pct_from_open'] for p in expansion_prices])
            
            print(f"\n🔥 EXPANSION PHASE PRICE CHARACTERISTICS:")
            print(f"   Events: {len(expansion_prices)}")
            print(f"   Avg normalized price: {avg_norm_price:.2f} (0=low, 1=high)")
            print(f"   Avg % from open: {avg_pct_open:.1f}%")
        
        # Consolidation phase price characteristics  
        consolidation_prices = [p for p in price_analysis if p['has_consolidation']]
        if consolidation_prices:
            avg_norm_price = np.mean([p['normalized_price'] for p in consolidation_prices])
            avg_pct_open = np.mean([p['pct_from_open'] for p in consolidation_prices])
            
            print(f"\n📊 CONSOLIDATION PHASE PRICE CHARACTERISTICS:")
            print(f"   Events: {len(consolidation_prices)}")
            print(f"   Avg normalized price: {avg_norm_price:.2f} (0=low, 1=high)")
            print(f"   Avg % from open: {avg_pct_open:.1f}%")
        
        # Liquidity sweep characteristics
        liq_sweep_prices = [p for p in price_analysis if p['has_liq_sweep']]
        if liq_sweep_prices:
            avg_norm_price = np.mean([p['normalized_price'] for p in liq_sweep_prices])
            avg_pct_high = np.mean([p['pct_from_high'] for p in liq_sweep_prices])
            avg_pct_low = np.mean([p['pct_from_low'] for p in liq_sweep_prices])
            
            print(f"\n⚡ LIQUIDITY SWEEP PRICE CHARACTERISTICS:")
            print(f"   Events: {len(liq_sweep_prices)}")
            print(f"   Avg normalized price: {avg_norm_price:.2f}")
            print(f"   Avg % from high: {avg_pct_high:.1f}%")
            print(f"   Avg % from low: {avg_pct_low:.1f}%")

def main():
    """Main concrete analysis"""
    
    print("🔍 CONCRETE SUB-PATTERN ANALYSIS")
    print("=" * 80)
    print("Moving beyond vague descriptions to specific market mechanics")
    print("=" * 80)
    
    # Analyze actual events by sub-pattern
    analyze_actual_events_by_subpattern()
    
    # Analyze market mechanics
    analyze_market_mechanics()
    
    # Show specific examples
    analyze_specific_examples()
    
    # Analyze price levels
    decode_price_levels()
    
    print(f"\n✅ CONCRETE ANALYSIS COMPLETE")
    print("Now we know exactly what these patterns represent in market terms!")

if __name__ == "__main__":
    main()